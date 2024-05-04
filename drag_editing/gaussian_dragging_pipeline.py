import sys

sys.path.insert(0, "deps/DragDiffusion")
import torch
from diffusers import DDIMScheduler, AutoencoderKL
from types import SimpleNamespace
from argparse import ArgumentParser
from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.arguments import OptimizationParams
from gaussiansplatting.gaussian_renderer import render
import numpy as np
from drag_editing.lora_utils import train_lora
from drag_editing.utils import to_homogeneous, simple_camera_to_c2w_k
from drag_editing.drag_pipeline import DragPipeline
import torch.nn.functional as F
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
        n_inference_step=50,
        model_cpu_offload=False,
        attention_slicing=False,
        half_precision=False,
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
        self.n_inference_step = n_inference_step

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.background_tensor = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.model_cpu_offload = model_cpu_offload
        self.attention_slicing = attention_slicing
        self.half_precision = half_precision

    def preprocess_image(self, image, device, dtype=torch.float32):
        image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
        image = image.moveaxis(-1, 0)[None]
        image = image.to(device, dtype)
        return image

    def initialize(self, prompt):
        self.prompt = prompt
        self.dtype = torch.float16 if self.half_precision else torch.float32

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
                self.model_path, scheduler=scheduler, torch_dtype=self.dtype
            )
            # call this function to override unet forward function,
            # so that intermediate features are returned after forward
            self.model.modify_unet_forward()

            # set vae
            if self.vae_path != "default":
                self.model.vae = AutoencoderKL.from_pretrained(self.vae_path).to(
                    self.model.vae.device, self.model.dtype
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
            # self.model.enable_model_cpu_offload()

            self.model.to(self.device)
            self.model.text_encoder.requires_grad_(False)
            self.model.unet.requires_grad_(False)
            self.model.vae.requires_grad_(False)

        self.text_embeddings = self.model.get_text_embeddings(self.prompt)

        self.args = SimpleNamespace()
        self.args.n_inference_step = self.n_inference_step
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

        self.model.scheduler.set_timesteps(self.args.n_inference_step)
        self.t = self.model.scheduler.timesteps[
            self.args.n_inference_step - self.args.n_actual_inference_step
        ]


    def get_image(self, cam_idx):
        render_pkg = render(self.cameras[cam_idx], self.gaussians, self.pipe, self.background_tensor)
        return render_pkg["render"] # [3, H, W]

    def get_init_code(self, cam_idx, rgb=None, progress_bar=False):
        if rgb is None:
            rgb = self.get_image(cam_idx)
        source_image = rgb[None] * 2 - 1
        source_image = source_image.to(self.dtype)

        # invert the source image
        # the latent code resolution is too small, only 64*64
        invert_code = self.model.invert(
            source_image,
            self.prompt,
            encoder_hidden_states=self.text_embeddings,
            guidance_scale=self.args.guidance_scale,
            num_actual_inference_steps=self.args.n_actual_inference_step,
            progress_bar=progress_bar
        )
        return invert_code

    def train_lora(self):
        assert getattr(self, "prompt", None) is not None, "Please set prompt first"

        renders = []
        for cam_idx in tqdm(range(len(self.cameras))):
            camera = self.cameras[cam_idx]
            # Obtain source image from gaussians
            with torch.no_grad():
                render_pkg = render(camera, self.gaussians, self.pipe, self.background_tensor)

            renders.append(render_pkg)

        imgs = [
            (renders[i]["render"].moveaxis(0, -1).cpu().numpy().copy() * 255).astype(np.uint8)
            for i in range(len(renders))
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
    def proj(self, c2w, K, p):
        p = to_homogeneous(p)
        p = (p @ c2w.inverse().T)[..., :3]
        p = p @ K.T
        p = p[..., :2] / p[..., 2]
        return p

    def point_tracking(self, F0_at_handles, F1, handle_points, args, cam_idx):
        with torch.no_grad():
            _, _, max_r, max_c = F1.shape
            new_handle_points = torch.zeros_like(handle_points)
            for i in range(len(handle_points)):
                if not self.valid_track_mask[cam_idx][i]:
                    new_handle_points[i] = handle_points[i]
                    continue
                pi = handle_points[i]
                f0 = F0_at_handles[i]

                r1, r2 = max(0, int(pi[0]) - args.r_p), min(max_r, int(pi[0]) + args.r_p + 1)
                c1, c2 = max(0, int(pi[1]) - args.r_p), min(max_c, int(pi[1]) + args.r_p + 1)
                F1_neighbor = F1[:, :, r1:r2, c1:c2]
                all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
                all_dist = all_dist.squeeze(dim=0)
                row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
                # handle_points[i][0] = pi[0] - args.r_p + row
                # handle_points[i][1] = pi[1] - args.r_p + col
                new_handle_points[i][0] = r1 + row
                new_handle_points[i][1] = c1 + col
            return new_handle_points

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

    def get_optimizer(self):
        parser = ArgumentParser(description="Training script parameters")
        opt_config = OptimizationParams(parser, max_steps=1800, lr_scaler=100)
        self.gaussians.training_setup(opt_config)
        return self.gaussians.optimizer

    def enable_optimizations(self):
        if self.model_cpu_offload:
            self.model.enable_model_cpu_offload()
        if self.attention_slicing:
            self.model.enable_attention_slicing()


    def drag(
        self, handle_points_3d, target_points_3d, debug=False
    ):
        assert len(handle_points_3d) == len(
            target_points_3d
        ), "number of handle point must equals target points"

        
        self.unet_outputs = []
        self.F0_at_handles = []
        # self.x_prev_0s = []
        self.handle_points = []
        self.target_points = []
        self.valid_track_mask = []
        # self.interp_masks = []
        # using_mask = False
        for cam_idx in tqdm(range(len(self.cameras)), desc="Rendering references"):
            camera = self.cameras[cam_idx]
            sup_res_h = 64
            sup_res_w = 64
            # the init output feature of unet
            with torch.no_grad():
                rgb = self.get_image(cam_idx)
                # mask = torch.ones(rgb.shape[:2]).to(rgb) # TODO: mask
                init_code = self.get_init_code(cam_idx, rgb=rgb)
                unet_output, F0 = self.model.forward_unet_features(
                    init_code,
                    self.t,
                    encoder_hidden_states=self.text_embeddings,
                    layer_idx=self.args.unet_feature_idx,
                    interp_res_h=sup_res_h,
                    interp_res_w=sup_res_w,
                )
                # x_prev_0, _ = self.model.step(unet_output, self.t, init_code)
                c2w, K = simple_camera_to_c2w_k(self.cameras[cam_idx])
                handle_points = self.proj(c2w, K, handle_points_3d)
                target_points = self.proj(c2w, K, target_points_3d)

                if debug:
                    rgb_debug = (rgb.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0).copy()
                    sel_pix = torch.stack([handle_points, target_points], dim=1).view(-1, 2).cpu().numpy().astype(np.int32)
                    # print(sel_pix, rgb_debug.shape)
                    rgb_debug = get_points(rgb_debug, sel_pix)
                    Image.fromarray(rgb_debug).save(f"tmp/debug_{cam_idx}.png")

                handle_points[:, 0] *= sup_res_w / camera.image_width
                handle_points[:, 1] *= sup_res_h / camera.image_height
                target_points[:, 0] *= sup_res_w / camera.image_width
                target_points[:, 1] *= sup_res_h / camera.image_height

                valid_track_mask = [True for _ in range(len(handle_points))]
                for ps in [handle_points, target_points]:
                    for i, p in enumerate(ps):
                        if p[0] < 0 or p[0] >= camera.image_width or p[1] < 0 or p[1] >= camera.image_height:
                            valid_track_mask[i] = False
                self.valid_track_mask.append(valid_track_mask)


                F0_at_handles = []
                for i, handle_point in enumerate(handle_points):
                    if not valid_track_mask[i]:
                        F0_at_handles.append(None)
                        continue
                    F0_at_handle = F0[:, :, int(handle_point[1]), int(handle_point[0])]
                    F0_at_handles.append(F0_at_handle)
                self.F0_at_handles.append(F0_at_handles)

                self.handle_points.append(handle_points)
                self.target_points.append(target_points)
                self.unet_outputs.append(unet_output)
                # self.x_prev_0s.append(x_prev_0)
                # interp_mask = F.interpolate(mask[None, None], (init_code.shape[2], init_code.shape[3]), mode="nearest")[0, 0]
                # self.interp_masks.append(interp_mask)
                # using_mask = using_mask or interp_mask.sum() != 0.0

        optimizer = self.get_optimizer()

        self.enable_optimizations()

        for step_idx in tqdm(range(self.args.n_pix_step)):
            cam_idx = torch.randint(low=0, high=len(self.cameras), size=())
            camera = self.cameras[cam_idx]
            sup_res_h = 64
            sup_res_w = 64

            init_code = self.get_init_code(cam_idx)
            _, F1 = self.model.forward_unet_features(
                init_code,
                self.t,
                encoder_hidden_states=self.text_embeddings,
                layer_idx=self.args.unet_feature_idx,
                interp_res_h=sup_res_h,
                interp_res_w=sup_res_w,
            )
            # x_prev_updated, _ = self.model.step(unet_output, self.t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            handle_points, target_points = self.handle_points[cam_idx], self.target_points[cam_idx]
            if step_idx != 0:
                self.handle_points[cam_idx] = self.point_tracking(
                    self.F0_at_handles[cam_idx], F1, self.handle_points[cam_idx], self.args, cam_idx
                )
                handle_points = self.handle_points[cam_idx]
                print("new handle points", handle_points)
                print("target points", target_points)

            # break if all handle points have reached the targets
            # if self.check_handle_reach_target(handle_points, target_points):
            #     break

            loss = 0.0
            _, _, max_r, max_c = F1.shape
            for i in range(len(handle_points)):
                if not self.valid_track_mask[cam_idx][i]:
                    continue
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.0:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                # with boundary protection
                r1, r2 = max(0, int(pi[0]) - self.args.r_m), min(max_r, int(pi[0]) + self.args.r_m + 1)
                c1, c2 = max(0, int(pi[1]) - self.args.r_m), min(max_c, int(pi[1]) + self.args.r_m + 1)
                f0_patch = F1[:, :, r1:r2, c1:c2].detach()
                f1_patch = self.interpolate_feature_patch(
                    F1, r1 + di[0], r2 + di[0], c1 + di[1], c2 + di[1]
                )

                # original code, without boundary protection
                # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                loss += ((2 * self.args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)
            if type(loss) is float:
                #no valid tracking point
                continue

            # masked region must stay unchanged
            # if using_mask:
            #     loss += (
            #         self.args.lam * ((x_prev_updated - x_prev_0) * (1.0 - self.interp_masks[cam_idx])).abs().sum()
            #     )
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print("loss total=%f" % (loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("done")