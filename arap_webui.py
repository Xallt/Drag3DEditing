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


def main(mesh_path: str):
    server = viser.ViserServer()

    mesh = trimesh.load_mesh(mesh_path)
    assert isinstance(mesh, trimesh.Trimesh)
    print(
        f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces."
    )

    # Normalize the mesh to fit in a unit sphere.
    bounding_sphere = mesh.bounding_sphere
    mesh_diameter = 2 * bounding_sphere.primitive.radius
    mesh.apply_scale(1.0 / mesh_diameter)

    mesh_handle = server.add_mesh_trimesh(
        name="/mesh",
        mesh=mesh,
        wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
        position=(0.0, 0.0, 0.0),
    )

    hit_pos_handles: List[viser.GlbHandle] = []

    with server.add_gui_folder("Handles"):
        add_button_handle = server.add_gui_button("Add handle")
        clear_button_handle = server.add_gui_button("Clear handles")

    @add_button_handle.on_click
    def _(_):
        add_button_handle.disabled = True

        def add_hit_handle(hit_pos):
            # Create a sphere at the hit location.
            hit_pos_mesh = trimesh.creation.icosphere(radius=0.005)
            hit_pos_mesh.visual.vertex_colors = (1.0, 0.0, 0.0, 1.0)  # type: ignore
            hit_pos_handle = server.add_mesh_trimesh(
                name=f"/hit_pos_{len(hit_pos_handles)}",
                mesh=hit_pos_mesh,
                position=hit_pos,
            )
            hit_pos_handles.append(hit_pos_handle)
            return hit_pos_handle

        @server.on_scene_click
        def scene_click_cb(message: viser.ScenePointerEvent) -> None:
            # Check for intersection with the mesh, using trimesh's ray-mesh intersection.
            # Note that mesh is in the mesh frame, so we need to transform the ray.
            R_world_mesh = tf.SO3(mesh_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            origin = (R_mesh_world @ np.array(message.ray_origin)).reshape(1, 3)
            direction = (R_mesh_world @ np.array(message.ray_direction)).reshape(1, 3)
            intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
            hit_pos, _, tri_index = intersector.intersects_location(origin, direction)

            if len(hit_pos) == 0:
                return

            # Successful click => remove callback.
            add_button_handle.disabled = False
            server.remove_scene_click_callback(scene_click_cb)

            # Get the first hit position (based on distance from the ray origin).
            hit_pos_idx = np.argmin(np.linalg.norm(hit_pos - origin, axis=1))
            hit_pos = hit_pos[hit_pos_idx]
            tri_index = tri_index[hit_pos_idx]

            normal = mesh.face_normals[tri_index]

            hit_pos = R_world_mesh @ hit_pos
            normal = R_world_mesh @ normal

            add_hit_handle(hit_pos)
            offset_sphere_handle = add_hit_handle(hit_pos + normal * 0.03)
            handle = server.add_transform_controls(
                "/control",
                scale=0.05,
                disable_sliders=True,
                disable_rotations=True,
            )
            handle.position = hit_pos

            @handle.on_update
            def _(_):
                # Update the hit position.
                offset_sphere_handle.position = handle.position

    @clear_button_handle.on_click
    def _(_):
        for handle in hit_pos_handles:
            handle.remove()
        hit_pos_handles.clear()

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main, description=__doc__)
