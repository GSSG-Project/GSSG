import os

import numpy as np
import torch
from PIL import Image

from gssg.utils.camera_utils import loadCam


def render_detections(
    mapper,
    train_cameras,
    dataset_params,
    save_path,
    max_items,
    render_semantics=False,
    scene_graph=None,
):
    save_dir = os.path.join(save_path, "renders")
    os.makedirs(save_dir, exist_ok=True)

    objects_dir = os.path.join(save_dir, "objects")
    if render_semantics:
        os.makedirs(objects_dir, exist_ok=True)
        semantics = mapper.global_params["semantics"]
        if torch.is_tensor(semantics):
            unique_semantic_ids = torch.unique(semantics).tolist()
        else:
            unique_semantic_ids = np.unique(np.array(semantics)).tolist()

    for frame_id, frame_info in enumerate(train_cameras):
        if frame_id % 10 == 0 or frame_id == max_items - 1:
            curr_frame = loadCam(
                dataset_params,
                frame_id,
                frame_info,
                dataset_params.resolution_scales[0],
            )

            # ---------- Normal render ----------
            render_output = mapper.renderer.render(
                curr_frame,
                mapper.global_params,
            )

            img_np = render_output["render"].permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray((img_np * 255).astype(np.uint8))
            img.save(os.path.join(save_dir, f"{frame_id}.png"))

    # ---------- Semantic renders ----------
    if render_semantics:
        for semantic_id in unique_semantic_ids:
            semantic_id = int(semantic_id)
            obj = scene_graph.get_object_by_id(semantic_id)
            if not obj:
                continue
            cam = loadCam(
                dataset_params,
                obj.frame_id,
                train_cameras[obj.frame_id],
                dataset_params.resolution_scales[0],
            )
            sem_output = mapper.renderer.render(
                cam,
                mapper.global_params,
                semantic_id=semantic_id,
            )

            sem_img_np = sem_output["render"].permute(1, 2, 0).cpu().numpy()

            sem_img = Image.fromarray((sem_img_np * 255).astype(np.uint8))

            sem_img.save(
                os.path.join(
                    objects_dir,
                    f"{semantic_id}.png",
                )
            )
    print("Saving Done")
