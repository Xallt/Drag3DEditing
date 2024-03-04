"""ARAP (As-Rigid-As-Possible) deformation using a web UI.
"""

import time
from typing import List

import numpy as np
import trimesh.creation
import trimesh.ray
import viser
import viser.transforms as tf
from dataclasses import dataclass
import torch

import tyro

import sys

sys.path.insert(0, "deps/as-rigid-as-possible")
from arap import Deformer


def to_numpy(m):
    if isinstance(m, np.ndarray):
        return m
    if isinstance(m, torch.Tensor):
        return m.cpu().detach().numpy()
    raise ValueError(f"Unsupported type: {type(m)}")


@dataclass
class DraggingControlPointer:
    hit_pos: np.ndarray
    tri_index: int
    control: viser.TransformControlsHandle


@dataclass
class FixedControlPointer:
    hit_pos: np.ndarray
    tri_index: int


@dataclass
class RayHit:
    hit_pos: np.ndarray
    tri_index: int
    normal: np.ndarray


class DraggingViserUI:
    def __init__(self, mesh_path):
        self.server = viser.ViserServer()

        self.mesh_path = mesh_path
        self.mesh = trimesh.load_mesh(mesh_path)
        assert isinstance(self.mesh, trimesh.Trimesh)
        print(
            f"Loaded mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces."
        )

        # Normalize the mesh to fit in a unit sphere.
        bounding_sphere = self.mesh.bounding_sphere
        mesh_diameter = 2 * bounding_sphere.primitive.radius
        self.mesh.apply_scale(1.0 / mesh_diameter)

        self.mesh_handle = self.server.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
            wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
        )

        self.deformer = Deformer()
        self.deformer.set_mesh(self.mesh.vertices, self.mesh.faces)

        with self.server.add_gui_folder("Handles"):
            self.add_drag_handle = self.server.add_gui_button("Add drag handle")
            self.add_fixed_handle = self.server.add_gui_button("Add fixed point")
            self.clear_button_handle = self.server.add_gui_button("Clear handles")

        self.add_drag_handle.on_click(
            lambda *args, **kwargs: self.on_add_drag_click(*args, **kwargs)
        )
        self.clear_button_handle.on_click(
            lambda *args, **kwargs: self.on_clear_button_handle_click(*args, **kwargs)
        )
        self.add_fixed_handle.on_click(
            lambda *args, **kwargs: self.on_add_fixed_click(*args, **kwargs)
        )

        self.hit_pos_handles: List[viser.GlbHandle] = []
        self.hit_pos_controls: List[DraggingControlPointer] = []
        self.fixed_handles: List[FixedControlPointer] = []

        self.drag_button = self.server.add_gui_button("Drag")

        self.drag_button.on_click(
            lambda *args, **kwargs: self.on_drag_button_click(*args, **kwargs)
        )

        with self.server.add_gui_folder("Dragging"):
            self.stop_deformation_handle = self.server.add_gui_button("Stop deformation")
            self.reset_dragging_handle = self.server.add_gui_button("Reset")

        @self.stop_deformation_handle.on_click
        def _(_):
            self.deformer.stop()

        @self.reset_dragging_handle.on_click
        def _(_):
            self.deformer.reset()

    def on_add_fixed_click(self, _):
        self.add_fixed_handle.disabled = True

        @self.server.on_scene_click
        def scene_click_cb(message: viser.ScenePointerEvent) -> None:
            ray_hit = self.ray_intersection(message.ray_origin, message.ray_direction)
            if ray_hit is None:
                return
            hit_pos, tri_index = ray_hit.hit_pos, ray_hit.tri_index

            # Successful click => remove callback.
            self.add_fixed_handle.disabled = False
            self.server.remove_scene_click_callback(scene_click_cb)

            self.add_hit_handle(
                hit_pos, f"/fixed_{len(self.hit_pos_handles)}", color=(1.0, 1.0, 0.0)
            )
            self.fixed_handles.append(FixedControlPointer(hit_pos, tri_index))

    def on_drag_button_click(self, _):
        self.deformer.reset()

        deformation = np.eye(4)
        deformation[:3, 3] = tf.SO3.from_x_radians(np.pi / 2).inverse().as_matrix() @ (
            self.hit_pos_controls[0].control.position - self.hit_pos_controls[0].hit_pos
        )
        self.deformer.set_deformation(deformation)

        fixed_tri_indices = [control.tri_index for control in self.fixed_handles]

        selection = {
            "selection": list(self.mesh.faces[self.hit_pos_controls[0].tri_index]),
            "fixed": list(set(self.mesh.faces[fixed_tri_indices].reshape(-1))),
        }
        self.deformer.set_selection(selection["selection"], selection["fixed"])

        self.deformer.apply_deformation(100)

        # vertices will be updated in the update loop

    def on_clear_button_handle_click(self, _):
        for handle in self.hit_pos_handles:
            handle.remove()
        for handle in self.hit_pos_controls:
            handle.control.remove()
        self.hit_pos_handles.clear()
        self.hit_pos_controls.clear()

    def add_hit_handle(self, hit_pos, name, color=(1.0, 0.0, 0.0)):
        # Create a sphere at the hit location.
        hit_pos_mesh = trimesh.creation.icosphere(radius=0.005)
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

    def on_add_drag_click(self, _):
        self.add_drag_handle.disabled = True

        @self.server.on_scene_click
        def scene_click_cb(message: viser.ScenePointerEvent) -> None:
            ray_hit = self.ray_intersection(message.ray_origin, message.ray_direction)
            if ray_hit is None:
                return
            hit_pos, tri_index, normal = ray_hit.hit_pos, ray_hit.tri_index, ray_hit.normal

            # Successful click => remove callback.
            self.add_drag_handle.disabled = False
            self.server.remove_scene_click_callback(scene_click_cb)

            self.add_hit_handle(hit_pos, f"/hit_pos_{len(self.hit_pos_handles)}")
            handle = self.server.add_transform_controls(
                f"/control_{len(self.hit_pos_handles)}",
                scale=0.05,
                disable_sliders=True,
                disable_rotations=True,
            )
            handle.position = hit_pos + normal * 0.03
            self.hit_pos_controls.append(DraggingControlPointer(hit_pos, tri_index, handle))
            self.add_hit_handle(np.zeros(3), f"/control_{len(self.hit_pos_handles)}/sphere")

    def set_vertices(self, vertices):
        self.mesh.vertices = vertices
        self.mesh_handle = self.server.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
            wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
        )

    def update(self):
        if hasattr(self, "deformer") and hasattr(self.deformer, "verts_prime"):
            if not np.allclose(to_numpy(self.deformer.verts_prime), self.mesh.vertices, atol=1e-5):
                self.set_vertices(to_numpy(self.deformer.verts_prime))

    @staticmethod
    def main(mesh_path: str):
        dragging_ui = DraggingViserUI(mesh_path)

        while True:
            time.sleep(0.05)
            dragging_ui.update()


if __name__ == "__main__":
    tyro.cli(DraggingViserUI.main, description=__doc__)
