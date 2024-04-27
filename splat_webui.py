import sys
sys.path.insert(0, "deps/GaussianEditor")
import time
import numpy as np
import torch
from typing import Dict, Any

import viser
import viser.transforms as tf
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage
from PIL import Image

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser

from gaussiansplatting.scene.camera_scene import CamScene
import math

import os
import random

import datetime

import cv2
from drag_editing.utils import unproject, to_homogeneous, get_points, c2w_k_to_simple_camera
from drag_editing.lora_utils import train_lora
from drag_editing.drag_utils import run_drag
from drag_editing.gaussian_dragging_pipeline import GaussianDraggingPipeline
from tqdm import tqdm
from pathlib import Path



class WebUI:
    def __init__(self, cfg) -> None:
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.port = 8084
        # training cfg

        # from original system
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        self.training = False
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.origin_frames = {}

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.tmp_path = Path("./tmp")
        self.tmp_path.mkdir(exist_ok=True)


        self.server = viser.ViserServer(port=self.port)
        self.add_theme()
        self.draw_flag = True
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=512
            )

            # self.FoV_slider = self.server.add_gui_slider(
            #     "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            # )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                ],
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        @self.save_button.on_click
        def _(_):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
            self.gaussian.save_ply(os.path.join("ui_result", "{}.ply".format(formatted_time)))

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value


        self.reset_up_button = self.server.add_gui_button(
            "Reset up direction",
            icon=viser.Icon.ARROW_AUTOFIT_UP,
            hint="Reset the orbit up direction.",
        )

        @self.reset_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            event.client.camera.up_direction = tf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

        with self.server.add_gui_folder("Dragging controls"):
            self.add_drag_handle = self.server.add_gui_button("Add drag handle")
            self.clear_button_handle = self.server.add_gui_button("Clear handles")

        self.drag_handles = []

        @self.add_drag_handle.on_click
        def _(_):
            self.add_drag_handle.disabled = True

            @self.server.on_scene_click
            def scene_click_cb(message: viser.ScenePointerEvent) -> None:
                pass
                click_pos = self.scene_pointer_event_to_click_pos(message)
                c2w, K = self.camera_params(message.client.camera)
                depth = self.render_cache["depth"]
                c2w, K = torch.from_numpy(c2w).to(depth), torch.from_numpy(K).to(depth)
                unprojected_points3d = unproject(c2w, K, torch.tensor(list(click_pos))[None].to(depth), depth[0, :, :, 0])

                self.add_drag_handle.disabled = False
                self.server.remove_scene_click_callback(scene_click_cb)

                handle = self.server.add_transform_controls(
                    f"/control_{len(self.drag_handles)}",
                    scale=1,
                    disable_sliders=True,
                    disable_rotations=True,
                )
                pos = unprojected_points3d.view(-1).cpu().numpy()
                handle.position = pos.copy()
                self.drag_handles.append((pos, handle))

        @self.clear_button_handle.on_click
        def _(_):
            for _, handle in self.drag_handles:
                handle.remove()
            self.drag_handles.clear()

        with self.server.add_gui_folder("Editing"):
            self.prompt_handle = self.server.add_gui_text("SD Prompt", "")
            self.train_lora_handle = self.server.add_gui_button("Train LoRA")
            self.dragging_handle = self.server.add_gui_button("Start Dragging")

        @self.train_lora_handle.on_click
        def _(_):
            model_path = "runwayml/stable-diffusion-v1-5"
            vae_path = "default"
            lora_path = "./lora_tmp"
            lora_step = 80
            lora_lr = 0.0005
            lora_batch_size = 4
            lora_rank = 16
            imgs = []
            for cam in self.colmap_cameras:
                with torch.no_grad():
                    render_pkg = self.render(cam)
                imgs.append((render_pkg["comp_rgb"][0].cpu().numpy().copy() * 255).astype(np.uint8))
            train_lora(
                imgs,
                self.prompt_handle.value,
                model_path,
                vae_path,
                lora_path,
                lora_step,
                lora_lr,
                lora_batch_size,
                lora_rank,
            )

        @self.dragging_handle.on_click
        def _(_):
            dragging_pipeline = GaussianDraggingPipeline(self.gaussian, self.colmap_cameras, n_inference_step = 2, inversion_strength=0.5, n_pix_step=int(1e6))
            dragging_pipeline.initialize(self.prompt_handle.value)
            handle_points, target_points = [], []
            for fixed_pos, handle in self.drag_handles:
                handle_points.append(fixed_pos.tolist())
                target_points.append(handle.position.tolist())
            handle_points = torch.tensor(handle_points, dtype=torch.float32, device='cuda')
            target_points = torch.tensor(target_points, dtype=torch.float32, device='cuda')
            dragging_pipeline.drag(handle_points, target_points)


        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

    def camera_params(self, camera, wxyz=None, position=None, resolution=None):
        if wxyz is None:
            wxyz = camera.wxyz
        if position is None:
            position = camera.position
        if resolution is None:
            resolution = self.resolution_slider.value
        R = tf.SO3(wxyz).as_matrix()
        T = position

        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = T

        width = int(resolution)
        height = int(width / self.aspect)

        K = np.array([
            [camera.fov * width / camera.aspect / 2 , 0, width / 2],
            [0, camera.fov * height / 2, height / 2],
            [0, 0, 1]
        ])

        return c2w, K


    def scene_pointer_event_to_click_pos(self, message: viser.ScenePointerEvent):
        ray_d = np.array(message.ray_direction)
        camera = message.client.camera
        c2w, K = self.camera_params(camera)

        ray_d = c2w[:3, :3].T @ ray_d
        ray_d /= ray_d[2]

        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)

        x, y = (K @ ray_d)[:2]

        # x, y = ray_d[0] * camera.fov / camera.aspect, ray_d[1] * camera.fov
        # x, y = int(x * width / 2 + width / 2), int(y * height / 2 + height / 2)
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        return x, y

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser = ArgumentParser(description="Training script parameters"),
            max_steps= self.edit_train_steps.value,
            lr_scaler = self.gs_lr_scaler.value,
            lr_final_scaler = self.gs_lr_end_scaler.value,
            color_lr_scaler = self.color_lr_scaler.value,
            opacity_lr_scaler = self.opacity_lr_scaler.value,
            scaling_lr_scaler = self.scaling_lr_scaler.value,
            rotation_lr_scaler = self.rotation_lr_scaler.value,

        )
        opt = OmegaConf.create(vars(opt))
        # opt.update(self.training_args)
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)

    def render(
        self,
        cam,
        local=False,
        train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        self.gaussian.localize = False  # reverse

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        self.render_cache = render_pkg

        return {
            **render_pkg,
        }

    @property
    def viser_cam(self):
        return list(self.server.get_clients().values())[0].camera

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        return c2w_k_to_simple_camera(*self.camera_params(self.viser_cam))

    def click_cb(self, pointer):
        pass

    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key][0]  # H W C
        if out_key == "masks":
            out_img = output[out_key][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        elif out_key == "depth":
            out_img = output[out_key][0].to(torch.float32).repeat(1, 1, 3)
            out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        if len(self.drag_handles) > 0:
            out_img_np = out_img.cpu().moveaxis(0, -1).numpy().copy()
            c2w, K = self.camera_params(self.viser_cam)
            c2w = c2w
            K = K
            def proj(c2w, K, p):
                p = to_homogeneous(p)
                p = (np.linalg.inv(c2w) @ p)[:3]
                p = K @ p
                p = p[:2] / p[2]
                return p
            for fixed_pos, handle in self.drag_handles:
                p0 = proj(c2w, K, fixed_pos)
                cv2.circle(out_img_np, (int(p0[0]), int(p0[1])), 5, (255, 255, 0), -1)
                p1 = proj(c2w, K, handle.position)
                cv2.circle(out_img_np, (int(p1[0]), int(p1[1])), 5, (255, 0, 0), -1)
                cv2.arrowedLine(out_img_np, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 255, 255), 2, tipLength=0.2)
            out_img = torch.from_numpy(out_img_np).cuda().moveaxis(-1, 0)

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            # if self.viewer_need_update:
            self.update_viewer()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        output = self.render(gs_camera)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")

    def densify_and_prune(self, step):
        if step <= self.densify_until_step.value:
            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval.value == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent.value,
                    min_opacity=self.min_opacity.value,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )

    def add_theme(self):
        buttons = (
            TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md",
            ),
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/buaacyw/GaussianEditor",
            ),
            TitlebarButton(
                text="Yiwen Chen",
                icon=None,
                href="https://buaacyw.github.io/",
            ),
            TitlebarButton(
                text="Zilong Chen",
                icon=None,
                href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://github.com/buaacyw/gaussian-editor/raw/master/static/images/logo.png",
            image_alt="GaussianEditor Logo",
            href="https://buaacyw.github.io/gaussian-editor/",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()
