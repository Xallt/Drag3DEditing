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
        if not hasattr(self, 'model'):
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
        self.inverted_codes = []

        self.args = SimpleNamespace()
        self.args.n_inference_step = 50
        self.args.n_actual_inference_step = round(self.inversion_strength * self.args.n_inference_step)
        self.args.guidance_scale = 1.0

        self.args.unet_feature_idx = [3]

        self.args.r_m = 1
        self.args.r_p = 3
        self.args.lam = self.lam

        self.args.lr = self.latent_lr
        self.args.n_pix_step = self.n_pix_step
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
            mask = np.ones(camera.height, camera.width)
            mask = torch.from_numpy(mask).float() / 255.0
            mask[mask > 0.0] = 1.0
            mask = mask[None, None].cuda()
            sup_res_h = int(0.5 * camera.height)
            sup_res_w = int(0.5 * camera.width)
            mask = F.interpolate(mask, (sup_res_h, sup_res_w), mode="nearest")
            def proj(c2w, K, p):
                p = to_homogeneous(p)
                p = (p @ np.linalg.inv(c2w).T)[:3]
                p = p @ K.T
                p = p[..., :2] / p[..., 2]
                return p
            c2w, K = simple_camera_to_c2w_k(camera)
            handle_points, target_points = proj(c2w, K, handle_points_3d), proj(c2w, K, target_points_3d)
            updated_init_code = drag_diffusion_update(
                self.model, init_code, self.text_embeddings, self.t, handle_points, target_points, mask, self.args
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
                register_attention_editor_diffusers(self.model, editor, attn_processor="lora_attn_proc")

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
            gen_image = F.interpolate(gen_image, (camera.height, camera.width), mode="bilinear")

            out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
            out_image = (out_image * 255).astype(np.uint8)
            return out_image
