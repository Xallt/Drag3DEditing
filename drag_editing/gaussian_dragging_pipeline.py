import sys

sys.path.insert(0, "deps/DragDiffusion")
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from drag_pipeline import DragPipeline
from pytorch_lightning import seed_everything
from types import SimpleNamespace
from argparse import ArgumentParser
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.gaussian_renderer import render
import numpy as np
from drag_editing.lora_utils import train_lora
from drag_editing.drag_utils import drag_diffusion_update
from drag_editing.utils import to_homogeneous, simple_camera_to_c2w_k
import torch.nn.functional as F
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from copy import deepcopy
from tqdm import tqdm


class GaussianDraggingPipeline:
    def __init__(
        self,
        gaussians,
        cameras,
        model_path="runwayml/stable-diffusion-v1-5",
        lora_path="lora_tmp",
        vae_path="default",
        lora_step=80,
        lora_lr=0.0005,
        lora_batch_size=4,
        lora_rank=16,
        start_step=0,
        start_layer=10,
        latent_lr=0.01,
        inversion_strength=0.7,
        lam=0.1,
        n_pix_step=80,
    ):
        self.gaussians = gaussians
        self.cameras = cameras

        self.model_path = model_path
        self.lora_path = lora_path
        self.vae_path = vae_path

        self.lora_step = lora_step
        self.lora_lr = lora_lr
        self.lora_batch_size = lora_batch_size
        self.lora_rank = lora_rank

        self.start_step = start_step
        self.start_layer = start_layer
        self.latent_lr = latent_lr
        self.inversion_strength = inversion_strength
        self.lam = lam
        self.n_pix_step = n_pix_step

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.background_tensor = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    def preprocess_image(self, image, device, dtype=torch.float32):
        image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
        image = image.moveaxis(-1, 0)[None]
        image = image.to(device, dtype)
        return image

    def initialize(self, prompt):
        self.prompt = prompt
        self.renders = []
        for cam_idx in tqdm(range(len(self.cameras))):
            camera = self.cameras[cam_idx]
            # Obtain source image from gaussians
            with torch.no_grad():
                render_pkg = render(camera, self.gaussians, self.pipe, self.background_tensor)

            self.renders.append(render_pkg)

    def initialize_dragging(self):
        assert getattr(self, "prompt", None) is not None, "Please set prompt first"

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not hasattr(self, "model"):
            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            self.model = DragPipeline.from_pretrained(
                self.model_path, scheduler=scheduler, torch_dtype=torch.float16
            ).to(self.device)
            # call this function to override unet forward function,
            # so that intermediate features are returned after forward
            self.model.modify_unet_forward()

            # set vae
            if self.vae_path != "default":
                self.model.vae = AutoencoderKL.from_pretrained(self.vae_path).to(
                    self.model.vae.device, self.model.vae.dtype
                )

            # set lora
            if self.lora_path == "":
                print("applying default parameters")
                self.model.unet.set_default_attn_processor()
            else:
                print("applying lora: " + self.lora_path)
                self.model.unet.load_attn_procs(self.lora_path)
            # obtain text embeddings

            # off load model to cpu, which save some memory.
            self.model.enable_model_cpu_offload()

        text_embeddings = self.model.get_text_embeddings(self.prompt)

        self.args = SimpleNamespace()
        self.args.n_inference_step = 50
        self.args.n_actual_inference_step = round(
            self.inversion_strength * self.args.n_inference_step
        )
        self.args.guidance_scale = 1.0

        self.args.unet_feature_idx = [3]

        self.args.r_m = 1
        self.args.r_p = 3
        self.args.lam = self.lam

        self.args.lr = self.latent_lr
        self.args.n_pix_step = self.n_pix_step

        if not hasattr(self, "inverted_codes"):
            self.inverted_codes = []

            seed_everything(42)

            for cam_idx in tqdm(range(len(self.cameras))):
                # initialize parameters

                # Obtain source image from gaussians
                source_image = self.renders[cam_idx]["render"][None] * 2 - 1
                source_image = source_image.to(torch.float16)

                # invert the source image
                # the latent code resolution is too small, only 64*64
                invert_code = self.model.invert(
                    source_image,
                    self.prompt,
                    encoder_hidden_states=text_embeddings,
                    guidance_scale=self.args.guidance_scale,
                    num_inference_steps=self.args.n_inference_step,
                    num_actual_inference_steps=self.args.n_actual_inference_step,
                )
                self.inverted_codes.append(invert_code)

                # empty cache to save memory
                torch.cuda.empty_cache()

        self.model.scheduler.set_timesteps(self.args.n_inference_step)
        self.t = self.model.scheduler.timesteps[
            self.args.n_inference_step - self.args.n_actual_inference_step
        ]

        # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
        # convert dtype to float for optimization
        self.text_embeddings = text_embeddings.float()
        self.model.unet = self.model.unet.float()

    def train_lora(self):
        assert getattr(self, "prompt", None) is not None, "Please set prompt first"

        imgs = [
            (self.renders[i]["render"].moveaxis(0, -1).cpu().numpy().copy() * 255).astype(np.uint8)
            for i in range(len(self.renders))
        ]
        train_lora(
            imgs,
            self.prompt,
            self.model_path,
            self.vae_path,
            self.lora_path,
            self.lora_step,
            self.lora_lr,
            self.lora_batch_size,
            self.lora_rank,
        )

    def drag(self, handle_points_3d, target_points_3d):
        for cam_idx in range(len(self.cameras)):
            camera = self.cameras[cam_idx]
            invert_code = self.inverted_codes[cam_idx]
            init_code = invert_code
            init_code_orig = deepcopy(init_code)

            # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
            # convert dtype to float for optimization
            init_code = init_code.float()
            text_embeddings = self.text_embeddings.float()
            self.model.unet = self.model.unet.float()

            # preparing editing meta data (handle, target, mask)
            mask = np.ones((camera.image_height, camera.image_width))
            mask = torch.from_numpy(mask).float() / 255.0
            mask[mask > 0.0] = 1.0
            mask = mask[None, None].cuda()
            sup_res_h = int(0.5 * camera.image_height)
            sup_res_w = int(0.5 * camera.image_width)
            mask = F.interpolate(mask, (sup_res_h, sup_res_w), mode="nearest")

            def proj(c2w, K, p):
                p = to_homogeneous(p)
                p = (p @ c2w.inverse().T)[..., :3]
                p = p @ K.T
                p = p[..., :2] / p[..., 2]
                return p

            c2w, K = simple_camera_to_c2w_k(camera)
            handle_points, target_points = (
                proj(c2w, K, handle_points_3d),
                proj(c2w, K, target_points_3d),
            )
            updated_init_code = self.drag_diffusion_update(
                self.model,
                init_code,
                self.text_embeddings,
                self.t,
                handle_points,
                target_points,
                mask,
                self.args,
                sup_res_h,
                sup_res_w
            )

            updated_init_code = updated_init_code.half()
            text_embeddings = text_embeddings.half()
            self.model.unet = self.model.unet.half()

            # hijack the attention module
            # inject the reference branch to guide the generation
            editor = MutualSelfAttentionControl(
                start_step=self.start_step,
                start_layer=self.start_layer,
                total_steps=self.n_inference_step,
                guidance_scale=self.guidance_scale,
            )
            if self.lora_path == "":
                register_attention_editor_diffusers(self.model, editor, attn_processor="attn_proc")
            else:
                register_attention_editor_diffusers(
                    self.model, editor, attn_processor="lora_attn_proc"
                )

            # inference the synthesized image
            gen_image = self.model(
                prompt=self.prompt,
                encoder_hidden_states=torch.cat([text_embeddings] * 2, dim=0),
                batch_size=2,
                latents=torch.cat([init_code_orig, updated_init_code], dim=0),
                guidance_scale=self.args.guidance_scale,
                num_inference_steps=self.args.n_inference_step,
                num_actual_inference_steps=self.args.n_actual_inference_step,
            )[1].unsqueeze(dim=0)

            # resize gen_image into the size of source_image
            # we do this because shape of gen_image will be rounded to multipliers of 8
            gen_image = F.interpolate(gen_image, (camera.image_height, camera.image_width), mode="bilinear")

            out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
            out_image = (out_image * 255).astype(np.uint8)
            return out_image

    def point_tracking(self, F0, F1, handle_points, handle_points_init, args):
        with torch.no_grad():
            _, _, max_r, max_c = F0.shape
            for i in range(len(handle_points)):
                pi0, pi = handle_points_init[i], handle_points[i]
                f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

                r1, r2 = max(0, int(pi[0]) - args.r_p), min(max_r, int(pi[0]) + args.r_p + 1)
                c1, c2 = max(0, int(pi[1]) - args.r_p), min(max_c, int(pi[1]) + args.r_p + 1)
                F1_neighbor = F1[:, :, r1:r2, c1:c2]
                all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
                all_dist = all_dist.squeeze(dim=0)
                row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
                # handle_points[i][0] = pi[0] - args.r_p + row
                # handle_points[i][1] = pi[1] - args.r_p + col
                handle_points[i][0] = r1 + row
                handle_points[i][1] = c1 + col
            return handle_points

    def check_handle_reach_target(self, handle_points, target_points):
        # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
        all_dist = list(map(lambda p, q: (p - q).norm(), handle_points, target_points))
        return (torch.tensor(all_dist) < 2.0).all()

    # obtain the bilinear interpolated feature patch centered around (x, y) with radius r
    def interpolate_feature_patch(self, feat, y1, y2, x1, x2):
        x1_floor = torch.floor(x1).long()
        x1_cell = x1_floor + 1
        dx = torch.floor(x2).long() - torch.floor(x1).long()

        y1_floor = torch.floor(y1).long()
        y1_cell = y1_floor + 1
        dy = torch.floor(y2).long() - torch.floor(y1).long()

        wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
        wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
        wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
        wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

        Ia = feat[:, :, y1_floor : y1_floor + dy, x1_floor : x1_floor + dx]
        Ib = feat[:, :, y1_cell : y1_cell + dy, x1_floor : x1_floor + dx]
        Ic = feat[:, :, y1_floor : y1_floor + dy, x1_cell : x1_cell + dx]
        Id = feat[:, :, y1_cell : y1_cell + dy, x1_cell : x1_cell + dx]

        return Ia * wa + Ib * wb + Ic * wc + Id * wd

    def drag_diffusion_update(
        self, model, init_code, text_embeddings, t, handle_points, target_points, mask, args, sup_res_h, sup_res_w
    ):
        assert len(handle_points) == len(
            target_points
        ), "number of handle point must equals target points"
        if text_embeddings is None:
            text_embeddings = model.get_text_embeddings(args.prompt)

        # the init output feature of unet
        with torch.no_grad():
            unet_output, F0 = model.forward_unet_features(
                init_code,
                t,
                encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx,
                interp_res_h=sup_res_h,
                interp_res_w=sup_res_w,
            )
            x_prev_0, _ = model.step(unet_output, t, init_code)
            # init_code_orig = copy.deepcopy(init_code)

        # prepare optimizable init_code and optimizer
        init_code.requires_grad_(True)
        optimizer = torch.optim.Adam([init_code], lr=args.lr)

        # prepare for point tracking and background regularization
        handle_points_init = deepcopy(handle_points)
        interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode="nearest")
        using_mask = interp_mask.sum() != 0.0

        # prepare amp scaler for mixed-precision training
        scaler = torch.cuda.amp.GradScaler()
        for step_idx in range(args.n_pix_step):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                unet_output, F1 = model.forward_unet_features(
                    init_code,
                    t,
                    encoder_hidden_states=text_embeddings,
                    layer_idx=args.unet_feature_idx,
                    interp_res_h=sup_res_h,
                    interp_res_w=sup_res_w,
                )
                x_prev_updated, _ = model.step(unet_output, t, init_code)

                # do point tracking to update handle points before computing motion supervision loss
                if step_idx != 0:
                    handle_points = self.point_tracking(
                        F0, F1, handle_points, handle_points_init, args
                    )
                    print("new handle points", handle_points)

                # break if all handle points have reached the targets
                if self.check_handle_reach_target(handle_points, target_points):
                    break

                loss = 0.0
                _, _, max_r, max_c = F0.shape
                for i in range(len(handle_points)):
                    pi, ti = handle_points[i], target_points[i]
                    # skip if the distance between target and source is less than 1
                    if (ti - pi).norm() < 2.0:
                        continue

                    di = (ti - pi) / (ti - pi).norm()

                    # motion supervision
                    # with boundary protection
                    r1, r2 = max(0, int(pi[0]) - args.r_m), min(max_r, int(pi[0]) + args.r_m + 1)
                    c1, c2 = max(0, int(pi[1]) - args.r_m), min(max_c, int(pi[1]) + args.r_m + 1)
                    f0_patch = F1[:, :, r1:r2, c1:c2].detach()
                    f1_patch = self.interpolate_feature_patch(
                        F1, r1 + di[0], r2 + di[0], c1 + di[1], c2 + di[1]
                    )

                    # original code, without boundary protection
                    # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                    # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                    loss += ((2 * args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

                # masked region must stay unchanged
                if using_mask:
                    loss += (
                        args.lam * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
                    )
                # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
                print("loss total=%f" % (loss.item()))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return init_code
