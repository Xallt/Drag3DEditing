"""ARAP (As-Rigid-As-Possible) deformation using a web UI.
"""

import time
from typing import List

import numpy as np
import trimesh.creation
import trimesh.ray
import viser
import viser.transforms as tf

import tyro

import sys

sys.path.insert(0, "deps/as-rigid-as-possible")
from arap import Deformer


class DraggingViserUI:
    def __init__(self, mesh_path):
        self.server = viser.ViserServer()

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

        with self.server.add_gui_folder("Handles"):
            self.add_button_handle = self.server.add_gui_button("Add handle")
            self.clear_button_handle = self.server.add_gui_button("Clear handles")

        self.add_button_handle.on_click(
            lambda *args, **kwargs: self.on_add_button_click(*args, **kwargs)
        )
        self.clear_button_handle.on_click(
            lambda *args, **kwargs: self.on_clear_button_handle_click(*args, **kwargs)
        )

        self.hit_pos_handles: List[viser.GlbHandle] = []
        self.hit_pos_controls: List[viser.TransformControls] = []

        self.drag_button = self.server.add_gui_button("Drag")

        self.drag_button.on_click(
            lambda *args, **kwargs: self.on_drag_button_click(*args, **kwargs)
        )

    def on_drag_button_click(self, _):
        deformer = Deformer()

        deformer.set_mesh(self.mesh.vertices, self.mesh.faces)

        deformation = np.eye(4)
        deformation[:3, 3] = self.hit_pos_controls[0].position - self.hit_pos_handles[0].position
        deformer.set_deformation(deformation)

        selection = {
            "selection": [0],
            "fixed": [],
        }
        deformer.set_selection(selection["selection"], selection["fixed"])

        deformer.apply_deformation(100)

        self.mesh.vertices = deformer.verts_prime
        self.mesh_handle = self.server.add_mesh_trimesh(
            name="/mesh",
            mesh=self.mesh,
            wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
            position=(0.0, 0.0, 0.0),
        )

    def on_clear_button_handle_click(self, _):
        for handle in self.hit_pos_handles:
            handle.remove()
        self.hit_pos_handles.clear()

    def add_hit_handle(self, hit_pos, name):
        # Create a sphere at the hit location.
        hit_pos_mesh = trimesh.creation.icosphere(radius=0.005)
        hit_pos_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
        hit_pos_handle = self.server.add_mesh_trimesh(
            name=name,
            mesh=hit_pos_mesh,
            position=hit_pos,
        )
        self.hit_pos_handles.append(hit_pos_handle)
        return hit_pos_handle

    def on_add_button_click(self, _):
        self.add_button_handle.disabled = True

        @self.server.on_scene_click
        def scene_click_cb(message: viser.ScenePointerEvent) -> None:
            # Check for intersection with the mesh, using trimesh's ray-mesh intersection.
            # Note that mesh is in the mesh frame, so we need to transform the ray.
            R_world_mesh = tf.SO3(self.mesh_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            origin = (R_mesh_world @ np.array(message.ray_origin)).reshape(1, 3)
            direction = (R_mesh_world @ np.array(message.ray_direction)).reshape(1, 3)
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
            hit_pos, _, tri_index = intersector.intersects_location(origin, direction)

            if len(hit_pos) == 0:
                return

            # Successful click => remove callback.
            self.add_button_handle.disabled = False
            self.server.remove_scene_click_callback(scene_click_cb)

            # Get the first hit position (based on distance from the ray origin).
            hit_pos_idx = np.argmin(np.linalg.norm(hit_pos - origin, axis=1))
            hit_pos = hit_pos[hit_pos_idx]
            tri_index = tri_index[hit_pos_idx]

            normal = self.mesh.face_normals[tri_index]

            hit_pos = R_world_mesh @ hit_pos
            normal = R_world_mesh @ normal

            self.add_hit_handle(hit_pos, f"/hit_pos_{len(self.hit_pos_handles)}")
            handle = self.server.add_transform_controls(
                f"/control_{len(self.hit_pos_handles)}",
                scale=0.05,
                disable_sliders=True,
                disable_rotations=True,
            )
            handle.position = hit_pos + normal * 0.03
            self.hit_pos_controls.append(handle)
            self.add_hit_handle(np.zeros(3), f"/control_{len(self.hit_pos_handles)}/sphere")


def main(mesh_path: str):
    dragging_ui = DraggingViserUI(mesh_path)

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main, description=__doc__)
