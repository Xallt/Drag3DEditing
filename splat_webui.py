import sys
sys.path.insert(0, "deps/GaussianEditor")
import time
import numpy as np
import torch
from typing import Dict, Any
import trimesh

import viser
import viser.transforms as tf
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.utils.general_utils import build_rotation

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser

from gaussiansplatting.scene.camera_scene import CamScene
import dgl

import os
import random

import datetime

import cv2
from drag_editing.utils import unproject, to_homogeneous, get_points, c2w_k_to_simple_camera
# from drag_editing.lora_utils import train_lora
# from drag_editing.drag_utils import run_drag
# from drag_editing.gaussian_dragging_pipeline import GaussianDraggingPipeline
from tqdm import tqdm
from pathlib import Path
from threading import Lock

from arap_webui import DraggingControlPointer, RayHit, FixedControlPointer, to_numpy

sys.path.insert(0, "deps/as-rigid-as-possible")
from arap import Deformer

import sdguidance

class WebUI:
    def __init__(self, cfg) -> None:
        self.viewer_lock = Lock()
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

        self.mesh = trimesh.load(cfg.mesh)
        self.mesh_handle = self.server.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
            # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
            visible=False,
        )
        bounding_sphere = self.mesh.bounding_sphere
        self.mesh_diameter = 2 * bounding_sphere.primitive.radius

        proximity_query = trimesh.proximity.ProximityQuery(self.mesh)
        _, self.vertex_matches = proximity_query.vertex(self.gaussian._xyz.detach().cpu().numpy())

        self.deformer = Deformer()

        edges = np.concatenate((self.mesh.faces[:, [0, 1]], self.mesh.faces[:, [1, 2]], self.mesh.faces[:, [0, 2]]), axis=0)
        edges = np.concatenate((edges, np.flip(edges, axis=1)), axis=0)
        edges = np.unique(edges, axis=0)
        self.graph = dgl.graph(edges.tolist(), device='cuda')


        self.hit_pos_handles = []
        self.hit_pos_controls = []

        self.draw_flag = True
        self.view_mode = "gaussian"
        with self.server.add_gui_folder("View Mode") as self.view_folder:
            self.switch_view_button = self.server.add_gui_button("Switch View to Mesh")
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=512
            )

            # self.FoV_slider = self.server.add_gui_slider(
            #     "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            # )


            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                    "depth",
                    "opacity"
                ],
            )

            self.load_button = self.server.add_gui_button("Load Gaussian")
            self.load_path = self.server.add_gui_text("Load Path", self.gs_source)
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        @self.switch_view_button.on_click
        def on_switch_view_button_click(_):
            if self.view_mode == "gaussian":
                self.mesh_handle.visible = True
                self.view_mode = "mesh"
                self.switch_view_button.remove()
                with self.view_folder:
                    self.switch_view_button = self.server.add_gui_button("Switch View to Gaussian")
                    self.switch_view_button.on_click(on_switch_view_button_click)
            elif self.view_mode == "mesh":
                self.mesh_handle.visible = False
                self.view_mode = "gaussian"
                self.switch_view_button._impl.label = "Switch View to Mesh"
                self.switch_view_button.remove()
                with self.view_folder:
                    self.switch_view_button = self.server.add_gui_button("Switch View to Mesh")
                    self.switch_view_button.on_click(on_switch_view_button_click)
            else:
                raise ValueError(f"Unknown view mode: {self.view_mode}")

        @self.load_button.on_click
        def _(_):
            if not Path(self.load_path.value).exists():
                print(f"ERROR: Specified file {self.load_path.value} doesn't exist.")
                return
            with self.viewer_lock:
                self.gaussian.load_ply(self.load_path.value)

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
            self.neighborhood_radius_slider = self.server.add_gui_slider(
                "Neighborhood radius", min=2, max=50, step=1, initial_value=5
            )

        self.drag_handles = []

        @self.add_drag_handle.on_click
        def _(_):
            self.add_drag_handle.disabled = True

            @self.server.on_scene_pointer("click")
            def scene_click_cb(message: viser.ScenePointerEvent) -> None:
                ray_hit = self.ray_intersection(message.ray_origin, message.ray_direction)
                if ray_hit is None:
                    return
                hit_pos, tri_index, normal = ray_hit.hit_pos, ray_hit.tri_index, ray_hit.normal

                # Successful click => remove callback.
                self.server.remove_scene_pointer_callback()

                self.add_hit_handle(hit_pos, f"/hit_pos_{len(self.hit_pos_handles)}")
                handle = self.server.add_transform_controls(
                    f"/control_{len(self.hit_pos_handles)}",
                    scale=0.05 * self.mesh_diameter,
                    disable_sliders=True,
                    disable_rotations=True,
                )
                handle.position = hit_pos + normal * 0.03
                self.hit_pos_controls.append(DraggingControlPointer(hit_pos, tri_index, handle))
                self.add_hit_handle(np.zeros(3), f"/control_{len(self.hit_pos_handles)}/sphere")
                

                # Visualize affected area
                d = 10
                self.set_deformable_area(tri_index, d)

        @self.clear_button_handle.on_click
        def _(_):
            for handle in self.hit_pos_handles:
                handle.remove()
            for handle in self.hit_pos_controls:
                handle.control.remove()
            self.hit_pos_handles.clear()
            self.hit_pos_controls.clear()

            self.add_drag_handle.disabled = False

        with self.server.add_gui_folder("Preprocessing") as self.preprocessing_folder:
            self.sphere_cut_button_handle = self.server.add_gui_button("Add sphere for cutting")
            self.sphere_cut_data = None

        @self.neighborhood_radius_slider.on_update
        def _(_):
            self.set_deformable_area(self.hit_pos_controls[0].tri_index, self.neighborhood_radius_slider.value)

        @self.sphere_cut_button_handle.on_click
        def _(_):
            opacity_backup = self.gaussian._opacity.data.clone()
            self.sphere_cut_button_handle.disabled = True
            self.sphere_cut_data = {}
            # Create a handle in the center of the scene with position controls
            self.sphere_cut_data["handle"] = self.server.add_transform_controls(
                "/sphere_cut",
                scale=1,
                disable_sliders=True,
                disable_rotations=True,
            )

            with self.preprocessing_folder:
                self.sphere_cut_data["radius"] = self.server.add_gui_slider(
                    "Sphere radius", min=0.1, max=10, step=0.1, initial_value=1
                )
                cut_button = self.server.add_gui_button("Cut")

            handle = self.sphere_cut_data["handle"]
            def on_sphere_cut_update(_):
                pos = handle.position
                with self.viewer_lock:
                    self.gaussian._opacity.data[:] = opacity_backup
                    self.gaussian._opacity.data[
                        torch.norm(self.gaussian.get_xyz - torch.tensor(pos).to("cuda"), dim=-1) >= self.sphere_cut_data["radius"].value
                    ] = -100.0
            handle.on_update(on_sphere_cut_update)
            self.sphere_cut_data["radius"].on_update(on_sphere_cut_update)
        
            @cut_button.on_click
            def _(_):
                pos = handle.position
                with self.viewer_lock:
                    cut_mask = torch.norm(self.gaussian.get_xyz - torch.tensor(pos).to("cuda"), dim=-1) < self.sphere_cut_data["radius"].value
                    self.gaussian._opacity.data[:] = opacity_backup

                    self.gaussian._xyz.data = self.gaussian._xyz.data[cut_mask]
                    self.gaussian._opacity.data = self.gaussian._opacity.data[cut_mask]
                    self.gaussian._features_dc.data = self.gaussian._features_dc.data[cut_mask]
                    self.gaussian._features_rest.data = self.gaussian._features_rest.data[cut_mask]
                    self.gaussian._scaling.data = self.gaussian._scaling.data[cut_mask]
                    self.gaussian._rotation.data = self.gaussian._rotation.data[cut_mask]

                self.sphere_cut_button_handle.disabled = False
                self.sphere_cut_data["handle"].remove()
                self.sphere_cut_data["radius"].remove()
                cut_button.remove()
                self.sphere_cut_data = None


        with self.server.add_gui_folder("Editing"):
            self.prompt_handle = self.server.add_gui_text("SD Prompt", "")
            # self.train_lora_handle = self.server.add_gui_button("Train LoRA")
            self.dragging_handle = self.server.add_gui_button("Start Dragging")
            self.sds_training_handle = self.server.add_gui_button("Start SDS Training")

        # @self.train_lora_handle.on_click
        # def _(_):
        #     model_path = "runwayml/stable-diffusion-v1-5"
        #     vae_path = "default"
        #     lora_path = "./lora_tmp"
        #     lora_step = 80
        #     lora_lr = 0.0005
        #     lora_batch_size = 4
        #     lora_rank = 16
        #     imgs = []
        #     for cam in self.colmap_cameras:
        #         with torch.no_grad():
        #             render_pkg = self.render(cam)
        #         imgs.append((render_pkg["comp_rgb"][0].cpu().numpy().copy() * 255).astype(np.uint8))
        #     train_lora(
        #         imgs,
        #         self.prompt_handle.value,
        #         model_path,
        #         vae_path,
        #         lora_path,
        #         lora_step,
        #         lora_lr,
        #         lora_batch_size,
        #         lora_rank,
        #     )


        @self.dragging_handle.on_click
        def _(_):
            vertices, faces = self.get_mesh_to_edit()
            with self.viewer_lock:
                self.deformer.set_mesh(vertices, faces)

            deformation = np.eye(4)
            deformation[:3, 3] = self.hit_pos_controls[0].control.position - self.hit_pos_controls[0].hit_pos
            self.deformer.set_deformation(deformation)

            selection_list = []
            for control in self.hit_pos_controls:
                selection_list += self.mesh.faces[control.tri_index].tolist()

            old_to_new_vertex_mapping = np.zeros(len(self.mesh.vertices), dtype=int)
            old_to_new_vertex_mapping[self.affected_nodes_mask] = np.arange(len(self.affected_nodes))
            border_nodes_new_indices = old_to_new_vertex_mapping[self.border_nodes]
            selection_nodes_new_indices = old_to_new_vertex_mapping[self.selected_nodes]

            selection = {
                "selection": selection_nodes_new_indices, # in the bfs list, the first 3 elements are the queried face
                "fixed": border_nodes_new_indices,
            }
            self.deformer.set_selection(selection["selection"], selection["fixed"])

            self.deformer.apply_deformation(10)
            del self.deformer.graph

        @self.sds_training_handle.on_click
        def _(_):
            optimizer = self.get_optimizer()
            sd = sdguidance.StableDiffusionGuidance({})
            prompt_utils = sdguidance.StableDiffusionPromptProcessor({})()
            num_epochs = 100
            for i in tqdm(range(num_epochs)):
                for cam_idx in range(len(self.colmap_cameras)):
                    render_pkg = render(self.colmap_cameras[cam_idx], self.gaussian, self.pipe, self.background_tensor)
                    guidance_out = sd(
                        render_pkg['render'].permute(1, 2, 0)[None],
                        prompt_utils, 
                        rgb_as_latents=False, 
                        elevation=torch.tensor([0]).to('cuda').float(), 
                        azimuth=torch.tensor([0]).to('cuda').float(), 
                        camera_distances=torch.tensor([1]).to('cuda').float(), 
                    )

                    optimizer.zero_grad()
                    guidance_out['loss_sds'].backward()
                    optimizer.step()
                    
                    


        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

    def get_optimizer(self):
        parser = ArgumentParser(description="Training script parameters")
        opt_config = OptimizationParams(parser, max_steps=1800, lr_scaler=100)
        self.gaussian.training_setup(opt_config)
        l = [
            {
                "params": [self.gaussian._features_dc],
                "lr": opt_config.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self.gaussian._features_rest],
                "lr": opt_config.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self.gaussian._scaling],
                "lr": opt_config.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self.gaussian._rotation],
                "lr": opt_config.rotation_lr,
                "name": "rotation",
            },
            {
                "params": [self.gaussian._opacity],
                "lr": opt_config.opacity_lr,
                "name": "opacity",
            },
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizer
    def set_deformable_area(self, tri_index, d):
        with self.viewer_lock:
            self.affected_nodes = []
            self.selected_nodes = []
            self.border_nodes = []
            for i, node_list in enumerate(dgl.traversal.bfs_nodes_generator(self.graph, self.mesh.faces[tri_index])):
                if i == 0:
                    self.selected_nodes = node_list
                if i == d:
                    break
                self.affected_nodes += node_list
                if i == d - 1:
                    self.border_nodes = node_list
            self.affected_nodes_mask = np.zeros(len(self.mesh.vertices), dtype=bool)
            self.affected_nodes_mask[self.affected_nodes] = True
        vertex_color = np.ones((len(self.mesh.vertices), 3))
        vertex_color[self.affected_nodes] = [1, 0, 0]
        visuals = trimesh.visual.ColorVisuals(vertex_colors=vertex_color)
        self.mesh.visual = visuals
        self.mesh_handle = self.server.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
            # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
            visible=self.view_mode == "mesh",
        )

    def get_mesh_to_edit(self):
        vertices = self.mesh.vertices[self.affected_nodes_mask]

        old_to_new_vertex_mapping = np.zeros(len(self.mesh.vertices), dtype=int)
        old_to_new_vertex_mapping[self.affected_nodes_mask] = np.arange(len(self.affected_nodes))

        faces = self.mesh.faces
        faces_affected_mask = self.affected_nodes_mask[faces]
        faces_affected_mask = faces_affected_mask.all(axis=1)
        faces = faces[faces_affected_mask]
        faces = old_to_new_vertex_mapping[faces] # faces remapped to the new vertex indices
        return vertices, faces
    def add_hit_handle(self, hit_pos, name, color=(1.0, 0.0, 0.0)):
        # Create a sphere at the hit location.
        hit_pos_mesh = trimesh.creation.icosphere(radius=0.005 * self.mesh_diameter)
        hit_pos_mesh.visual.vertex_colors = (*color, 1.0)  # type: ignore
        hit_pos_handle = self.server.add_mesh_trimesh(
            name=name,
            mesh=hit_pos_mesh,
            position=hit_pos,
        )
        self.hit_pos_handles.append(hit_pos_handle)
        return hit_pos_handle
    
    def ray_intersection(self, ray_origin, ray_direction):
        R_world_mesh = tf.SO3(self.mesh_handle.wxyz)
        R_mesh_world = R_world_mesh.inverse()
        origin = (R_mesh_world @ np.array(ray_origin)).reshape(1, 3)
        direction = (R_mesh_world @ np.array(ray_direction)).reshape(1, 3)
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        hit_pos, _, tri_index = intersector.intersects_location(origin, direction)

        if len(hit_pos) == 0:
            return None

        # Get the first hit position (based on distance from the ray origin).
        hit_pos_idx = np.argmin(np.linalg.norm(hit_pos - origin, axis=1))
        hit_pos = hit_pos[hit_pos_idx]
        tri_index = tri_index[hit_pos_idx]
        normal = self.mesh.face_normals[tri_index]

        # Transform into world space.
        hit_pos = R_world_mesh @ hit_pos
        normal = R_world_mesh @ normal

        return RayHit(hit_pos, tri_index, normal)
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

    def has_active_client(self):
        return len(list(self.server.get_clients().values())) > 0

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
            self.viser_cam.up_direction = tf.SO3(self.viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
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

        if len(self.hit_pos_controls) > 0:
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
            for hit_control in self.hit_pos_controls:
                p0 = proj(c2w, K, hit_control.hit_pos)
                p1 = proj(c2w, K, hit_control.control.position)
                cv2.arrowedLine(out_img_np, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (255, 255, 255), 2, tipLength=0.2)
            out_img = torch.from_numpy(out_img_np).cuda().moveaxis(-1, 0)

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def set_vertices(self, vertices):
        self.mesh.vertices = vertices
        self.mesh_handle = self.server.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
            # wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
            visible=self.view_mode == "mesh",
        )
    def render_loop(self):
        while True:
            if self.has_active_client():
                self.update_viewer()
            if hasattr(self.deformer, "graph"):
                if not np.allclose(to_numpy(self.deformer.verts_prime), self.mesh.vertices[self.affected_nodes_mask], atol=1e-5):
                    verts_updated = self.mesh.vertices.copy()
                    verts_updated[self.affected_nodes_mask] = to_numpy(self.deformer.verts_prime)
                    cell_rotations = np.eye(3)[None].repeat(len(self.mesh.vertices), axis=0)
                    cell_rotations[self.affected_nodes_mask] = to_numpy(self.deformer.cell_rotations)
                    self.gaussian._xyz.data += torch.from_numpy((verts_updated - self.mesh.vertices)[self.vertex_matches]).to(self.gaussian._xyz.data)
                    self.gaussian._rotation.data = torch.from_numpy(
                        tf.SO3.from_matrix(
                            cell_rotations[self.vertex_matches] @ \
                            build_rotation(self.gaussian._rotation).detach().cpu().numpy()
                        ).as_quaternion_xyzw()[:, [3, 0, 1, 2]]
                    ).to(self.gaussian._rotation)
                    self.set_vertices(verts_updated)
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        with self.viewer_lock:
            with torch.no_grad():
                output = self.render(gs_camera)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()
