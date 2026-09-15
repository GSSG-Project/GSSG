import logging
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gssg.utils.general_utils import (
    build_covariance_from_scaling_rotation,
    build_rotation,
    devF,
    devI,
    inverse_sigmoid,
)
from gssg.utils.sh_utils import RGB2SH, SEMANTIC_COLOR_MAP, SH2RGB
from gssg.utils.utils import bbox_filter, compute_rot, l2_norm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Background seeds (pixels outside every instance mask, semantic id 0) may grow to
# BG_SCALE_MULT x max_radius: walls get covered by fewer, larger splats, and the
# radius-based seed dedup then thins future wall seeds to match. Object seeds keep
# the tight max_radius cap, so object geometry (and the S score it carries) is
# untouched. Sampling weights alone cannot shrink the map: density is radius-limited,
# not sampling-limited.
# Configured from `bg_scale_mult` (forced to 1.0 on sampling_version 1 or gamma=0, so
# the uniform baseline disables the whole objectness feature) — read it via the module
# attribute, a from-import freezes the default.
BG_SCALE_MULT = 1.5


def configure_bg_scale_mult(mult):
    global BG_SCALE_MULT
    BG_SCALE_MULT = float(mult)
    print(f"[sampling] bg_scale_mult={BG_SCALE_MULT}")


def gaussian_ply_row_array(
    xyz, features_dc, features_rest, opacity, scaling, rotation, confidence=None, anchor=None
):
    """Format gaussian tensors into the [N, D] float32 array layout that
    save_model_ply / construct_list_of_attributes expect
    (x,y,z, nx,ny,nz, f_dc..., f_rest..., opacity, scale..., rot..., [confidence])."""
    xyz_np = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz_np)  # zero normals, matching save_model_ply
    f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    cols = [
        xyz_np,
        normals,
        f_dc,
        f_rest,
        opacity.detach().cpu().numpy(),
        scaling.detach().cpu().numpy(),
        rotation.detach().cpu().numpy(),
    ]
    if confidence is not None:
        cols.append(confidence.detach().cpu().numpy())
    if anchor is not None:
        cols.append(anchor.detach().cpu().numpy())
    return np.concatenate(cols, axis=1).astype(np.float32)


def write_ply_streaming(path, attr_names, batch_iter, total_count):
    """Write a binary_little_endian PLY of `total_count` float32 vertices with columns
    `attr_names`, consuming `batch_iter` (each item a [n_i, len(attr_names)] float32
    array) one at a time so peak memory is a single batch. Returns vertices written."""
    rec_dtype = np.dtype([(a, "<f4") for a in attr_names])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {int(total_count)}\n"
        + "".join(f"property float {a}\n" for a in attr_names)
        + "end_header\n"
    )
    written = 0
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for batch in batch_iter:
            if batch is None or batch.shape[0] == 0:
                continue
            if batch.shape[1] != len(attr_names):
                raise ValueError(
                    f"PLY stream: batch has {batch.shape[1]} cols, expected {len(attr_names)}"
                )
            rec = np.empty(batch.shape[0], dtype=rec_dtype)
            for j, a in enumerate(attr_names):
                rec[a] = batch[:, j]
            f.write(rec.tobytes())
            written += int(batch.shape[0])
    if written != int(total_count):
        logging.warning(
            "[PLY] streamed %d vertices but header declared %d (PLY may be unreadable)",
            written,
            int(total_count),
        )
    return written


class GaussianPointCloud:
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args, scene_graph, name) -> None:
        self.name = name
        self.rotation_activation = None
        self.inverse_opacity_activation = None
        self.opacity_activation = None
        self.covariance_activation = None
        self.scaling_inverse_activation = None
        self.scaling_activation = None
        # gaussian optimize parameters
        self._xyz = devF(torch.empty(0, 3))  # [0,3] so cell_of([:, axis]) is safe when empty
        self._features_dc = devF(torch.empty(0))
        self._features_rest = devF(torch.empty(0))
        self._scaling = devF(torch.empty(0))
        self._rotation = devF(torch.empty(0))
        self._opacity = devF(torch.empty(0))
        self._semantic = devF(torch.empty(0))

        self.scene_graph = scene_graph

        # map management parameters
        self._normal = devF(torch.empty(0))
        self._confidence = devF(torch.empty(0))
        self._add_tick = devI(torch.empty(0))
        # Immutable creation-frame stamp: set once at birth and never rewritten (unlike
        # _add_tick, which age-pruning / stable->active downgrade resets), so a gaussian
        # can always be traced to the frame that placed it.
        self._anchor_frame = devI(torch.empty(0))

        # error counter
        self._depth_error_counter = devI(torch.empty(0))
        self._color_error_counter = devI(torch.empty(0))

        # Mutation counter, bumped by every op that changes the tensors a render-param
        # dict is derived from, so callers can cache param dicts and rebuild only on change.
        self._version = 0

        self.init_opacity = args.init_opacity
        self.scale_factor = args.scale_factor
        self.min_radius = args.min_radius
        self.max_radius = args.max_radius
        self.max_sh_degree = args.max_sh_degree
        self.active_sh_degree = args.active_sh_degree
        assert self.active_sh_degree <= self.max_sh_degree
        self.xyz_factor = devF(torch.tensor(args.xyz_factor))
        self.setup_functions()

    def bump_version(self):
        self._version += 1

    def load(self, ply_path):
        plydata = PlyData.read(ply_path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        P = xyz.shape[0]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        if "confidence" in plydata.elements[0]:
            confidences = np.asarray(plydata.elements[0]["confidence"])[..., np.newaxis]
        else:
            confidences = np.zeros((P, 1))

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )
        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        )

        self._features_rest = (
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
        )
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        self._normal = self.get_normal
        self._confidence = torch.tensor(confidences, dtype=torch.float, device="cuda")

        self._add_tick = torch.zeros([P, 1], dtype=torch.int32, device="cuda")
        self._anchor_frame = torch.zeros([P, 1], dtype=torch.int32, device="cuda")
        self._depth_error_counter = torch.zeros([P, 1], dtype=torch.int32, device="cuda")
        self._color_error_counter = torch.zeros([P, 1], dtype=torch.int32, device="cuda")
        self.bump_version()

    # Lossless serialization: the full attribute schema, unlike PLY (which drops
    # _semantic, zeroes counters, hard-codes cuda). _normal is derived but stored for
    # restore convenience.
    _BLOB_ATTRS = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "_semantic",
        "_normal",
        "_confidence",
        "_add_tick",
        "_anchor_frame",
        "_depth_error_counter",
        "_color_error_counter",
    )

    def to_blob(self):
        """Detached CPU dict of the full attribute schema (ready for torch.save)."""
        return {a: getattr(self, a).detach().cpu() for a in self._BLOB_ATTRS}

    def save_blob(self, path):
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.to_blob(), path)

    def from_blob(self, blob, device="cuda"):
        """Restore the full schema from a to_blob() dict onto `device` (in place)."""
        for a in self._BLOB_ATTRS:
            setattr(self, a, blob[a].to(device))
        self.bump_version()
        return self

    def load_blob(self, path, device="cuda"):
        return self.from_blob(torch.load(path, map_location="cpu"), device)

    def delete(self, delete_mask):
        # A mask squeezed from a 1-point cloud arrives 0-dim; 0-dim bool indexing would
        # insert a phantom leading dim and corrupt every attribute shape.
        delete_mask = delete_mask.reshape(-1)
        self._xyz = self._xyz[~delete_mask]
        self._features_dc = self._features_dc[~delete_mask]
        self._features_rest = self._features_rest[~delete_mask]
        self._scaling = self._scaling[~delete_mask]
        self._rotation = self._rotation[~delete_mask]
        self._opacity = self._opacity[~delete_mask]
        self._semantic = self._semantic[~delete_mask]
        self._normal = self._normal[~delete_mask]
        self._confidence = self._confidence[~delete_mask]
        self._add_tick = self._add_tick[~delete_mask]
        self._anchor_frame = self._anchor_frame[~delete_mask]
        self._depth_error_counter = self._depth_error_counter[~delete_mask]
        self._color_error_counter = self._color_error_counter[~delete_mask]
        self.bump_version()

    def remove(self, remove_mask):
        remove_mask = remove_mask.reshape(-1)  # see delete(): 0-dim masks corrupt shapes
        xyz = self._xyz[remove_mask]
        features_dc = self._features_dc[remove_mask]
        features_rest = self._features_rest[remove_mask]
        scaling = self._scaling[remove_mask]
        rotation = self._rotation[remove_mask]
        opacity = self._opacity[remove_mask]
        semantic = self._semantic[remove_mask]
        normal = self._normal[remove_mask]
        confidence = self._confidence[remove_mask]
        add_tick = self._add_tick[remove_mask]
        anchor_frame = self._anchor_frame[remove_mask]
        depth_error_counter = self._depth_error_counter[remove_mask]
        color_error_counter = self._color_error_counter[remove_mask]

        gaussian_params = {
            "xyz": xyz,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "semantic": semantic,
            "normal": normal,
            "confidence": confidence,
            "add_tick": add_tick,
            "anchor_frame": anchor_frame,
            "depth_error_counter": depth_error_counter,
            "color_error_counter": color_error_counter,
        }
        self.delete(remove_mask)
        return gaussian_params

    def detach(self):
        self._xyz = self._xyz.detach()
        self._features_dc = self._features_dc.detach()
        self._features_rest = self._features_rest.detach()
        self._scaling = self._scaling.detach()
        self._rotation = self._rotation.detach()
        self._opacity = self._opacity.detach()
        self._semantic = self._semantic.detach()
        # _normal is cached from get_normal (a non-leaf grad tensor when set while
        # parametrized); detach it too or copy.deepcopy of the cloud rejects it.
        self._normal = self._normal.detach()
        self.bump_version()

    def to_(self, device):
        """Move every tensor attribute to `device` in place; returns self."""
        for name, val in list(vars(self).items()):
            if torch.is_tensor(val):
                setattr(self, name, val.detach().to(device))
        return self

    def clone_to_cpu(self):
        """Detached CPU snapshot: tensor attributes are deep-copied to host, everything
        else is shared by reference, so a worker thread can serialize the cloud while the
        live one keeps mutating on the GPU."""
        snap = self.__class__.__new__(self.__class__)
        for name, val in vars(self).items():
            snap.__dict__[name] = val.detach().to("cpu", copy=True) if torch.is_tensor(val) else val
        return snap

    def parametrize(self, update_args):
        self.bump_version()
        self._xyz = nn.Parameter(self._xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.requires_grad_(True))
        self._semantic = nn.Parameter(self._semantic.requires_grad_(True))
        param_groups = [
            {
                "params": [self._xyz],
                "lr": update_args.position_lr,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": update_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": update_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": update_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": update_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": update_args.rotation_lr,
                "name": "rotation",
            },
        ]
        return param_groups

    def cat(self, paramters):
        if not torch.any(paramters["xyz"]):
            return
        self._xyz = torch.cat([self._xyz, paramters["xyz"]])
        self._features_dc = torch.cat([self._features_dc, paramters["features_dc"]])
        self._features_rest = torch.cat([self._features_rest, paramters["features_rest"]])
        self._scaling = torch.cat([self._scaling, paramters["scaling"]])
        self._rotation = torch.cat([self._rotation, paramters["rotation"]], dim=0)
        self._opacity = torch.cat([self._opacity, paramters["opacity"]])
        self._semantic = torch.cat([self._semantic, paramters["semantic"]])
        self._confidence = torch.cat([self._confidence, paramters["confidence"]])
        # Carry the incoming rows' normals (O(new)) instead of rebuilding the whole
        # cloud's normals. The stored _normal is not a render input (every consumer reads
        # the get_normal property), so this only keeps it shape-consistent for detach/to_.
        self._normal = torch.cat([self._normal, paramters["normal"]])
        self._add_tick = torch.cat([self._add_tick, paramters["add_tick"]])
        self._anchor_frame = torch.cat([self._anchor_frame, paramters["anchor_frame"]])
        self._depth_error_counter = torch.cat(
            [self._depth_error_counter, paramters["depth_error_counter"]]
        )
        self._color_error_counter = torch.cat(
            [self._color_error_counter, paramters["color_error_counter"]]
        )
        self.bump_version()

    def add_empty_points(self, xyz, normal, color, time, semantic=None):
        """
        :param xyz: [N, 3]
        :param normal: [N, 3]
        :param color: [N, 3]
        """
        assert xyz.shape[0] == color.shape[0] and color.shape[0] == normal.shape[0]
        if semantic is not None:
            assert xyz.shape[0] == semantic.shape[0]
        if xyz.shape[0] < 1:
            return
        mag = l2_norm(normal)
        normal = normal / (mag + 1e-8)
        valid_normal_mask = normal.sum(dim=-1) != 0
        xyz = xyz[valid_normal_mask]
        normal = normal[valid_normal_mask]
        color = color[valid_normal_mask]
        points_num = xyz.shape[0]
        if semantic is not None:
            semantic = semantic[valid_normal_mask]
        features = devF(torch.zeros((points_num, 3, (self.max_sh_degree + 1) ** 2)))
        sh_color = RGB2SH(color)
        features[:, :3, 0] = sh_color
        features[:, 3:, 1:] = 0.0
        raw_scales = devF(torch.ones(points_num, 3)) * 1e-6
        scales = torch.log(raw_scales)
        if self.xyz_factor[0] == 1 and self.xyz_factor[1] == 1 and self.xyz_factor[2] == 1:
            rots = devF(torch.zeros((points_num, 4)))
            rots[:, 0] = 1
        else:
            z_axis = devF(torch.tensor([0, 0, 1]).repeat(points_num, 1))
            rots = compute_rot(z_axis, normal)
        opacities = inverse_sigmoid(self.init_opacity * devF(torch.ones((points_num, 1))))
        confidence = devF(torch.zeros([points_num, 1]))
        add_tick = time * devI(torch.ones([points_num, 1]))
        anchor_frame = time * devI(torch.ones([points_num, 1]))

        depth_error_counter = devI(torch.zeros([points_num, 1]))
        color_error_counter = devI(torch.zeros([points_num, 1]))
        if xyz is None or not torch.any(xyz):
            return

        add_params = {
            "xyz": xyz,
            "features_dc": features[..., 0:1].transpose(1, 2).contiguous(),
            "features_rest": features[..., 1:].transpose(1, 2).contiguous(),
            "scaling": scales,
            "rotation": rots,
            "opacity": opacities,
            "semantic": semantic,
            "normal": normal,
            "confidence": confidence,
            "add_tick": add_tick,
            "anchor_frame": anchor_frame,
            "depth_error_counter": depth_error_counter,
            "color_error_counter": color_error_counter,
        }
        self.cat(add_params)

    def update_geometry(self, extra_xyz, extra_radius):
        xyz = self.get_xyz
        points_num = self.get_points_num
        if points_num == 0:
            return
        radius = self.get_radius

        if torch.numel(extra_xyz) > 0:
            inbbox_mask = bbox_filter(xyz, extra_xyz)
            extra_xyz = extra_xyz[inbbox_mask]
            extra_radius = extra_radius[inbbox_mask]

        total_xyz = torch.cat([xyz, extra_xyz])
        total_radius = torch.cat([radius, extra_radius])

        knn_indices = distCUDA2(total_xyz.float().cuda())
        knn_indices = knn_indices[:points_num].long()

        if knn_indices.dim() == 1:
            knn_indices = knn_indices[:, None]

        k = knn_indices.shape[1]

        dist_list = []
        for i in range(k):
            # Clamp the neighbor radius so enlarged background splats (BG_SCALE_MULT)
            # do not widen the invalid-seed deletion band around them — object seeds
            # next to a big wall splat must survive scale-init.
            dist_i = (
                torch.norm(xyz - total_xyz[knn_indices[:, i]], p=2, dim=1)
                - 3 * total_radius[knn_indices[:, i]].clamp(max=self.max_radius)
            )
            dist_list.append(dist_i)

        while len(dist_list) < 3:
            dist_list.append(torch.zeros_like(dist_list[0]))

        dist_0, dist_1, dist_2 = dist_list[:3]

        invalid_dist_0 = dist_0 < 0
        invalid_dist_1 = dist_1 < 0
        invalid_dist_2 = dist_2 < 0

        invalid_scale_mask = invalid_dist_0 | invalid_dist_1 | invalid_dist_2

        dist2 = (dist_0**2 + dist_1**2 + dist_2**2) / 3
        scales = torch.sqrt(dist2)
        max_r = torch.full_like(scales, self.max_radius)
        if self._semantic is not None and self._semantic.shape[0] == scales.shape[0]:
            max_r[self._semantic.view(-1) <= 0] = BG_SCALE_MULT * self.max_radius
        scales = torch.minimum(scales.clamp(min=self.min_radius), max_r)

        if (~invalid_scale_mask).sum() == 0:
            self.delete(invalid_scale_mask)
        else:
            scales = scales[..., None].repeat(1, 3)
            factor_scales = self.scale_factor * torch.mul(scales, self.xyz_factor)
            log_scales = torch.log(factor_scales)
            self._scaling = log_scales
            self.bump_version()
            self.delete(invalid_scale_mask)

    def construct_list_of_attributes(self, include_confidence=True, include_anchor=False):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        for i in range(self._scaling.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            attrs.append(f"rot_{i}")
        if include_confidence:
            attrs.append("confidence")
        if include_anchor:
            attrs.append("anchor_frame")
        return attrs

    def save_model_ply(
        self,
        path,
        include_confidence=True,
        visualize_semantics=False,
        semantics_only=False,
        scene_graph=None,
        include_anchor=False,
    ):
        if self.get_points_num == 0:
            return
        if semantics_only:
            logging.info("Filtering to view only splats with semantic info...")
            mask = (self._semantic > 0).squeeze()
            logging.info(f"Number of points: {mask.count_nonzero()}")
            if mask.sum() == 0:
                logging.info("No semantic points found to save.")
                return
            xyz_tensor = self._xyz[mask]
            features_dc_tensor = self._features_dc[mask]
            features_rest_tensor = self._features_rest[mask]
            opacity_tensor = self._opacity[mask]
            semantic_tensor = self._semantic[mask]
            scaling_tensor = self._scaling[mask]
            rotation_tensor = self._rotation[mask]
            confidence_tensor = self._confidence[mask]
            anchor_tensor = self._anchor_frame[mask]
        else:
            xyz_tensor = self._xyz
            features_dc_tensor = self._features_dc
            features_rest_tensor = self._features_rest
            opacity_tensor = self._opacity
            semantic_tensor = self._semantic
            scaling_tensor = self._scaling
            rotation_tensor = self._rotation
            confidence_tensor = self._confidence
            anchor_tensor = self._anchor_frame

        xyz = xyz_tensor.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            features_dc_tensor.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            features_rest_tensor.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = opacity_tensor.detach().cpu().numpy()
        semantics = semantic_tensor.detach().cpu().numpy()
        scale = scaling_tensor.detach().cpu().numpy()
        rotation = rotation_tensor.detach().cpu().numpy()
        confidence = confidence_tensor.detach().cpu().numpy()
        anchor = anchor_tensor.detach().cpu().numpy()

        unique_ids, counts = np.unique(semantics, return_counts=True)
        dict(zip(unique_ids, counts, strict=False))

        if visualize_semantics:
            logging.info("Visualizing semantics with color blending...")
            original_colors_rgb = SH2RGB(features_dc_tensor.detach())
            blended_colors_rgb = original_colors_rgb.clone().cpu().numpy()

            for i in range(len(blended_colors_rgb)):
                semantic_id = int(semantics[i].item())
                if semantic_id == 0:
                    continue
                color_id = scene_graph.get_color_id(semantic_id) if scene_graph else semantic_id
                semantic_color = np.array(SEMANTIC_COLOR_MAP.get(color_id, [1.0, 1.0, 1.0]))
                blended_colors_rgb[i] = semantic_color

            new_f_dc_tensor = RGB2SH(torch.from_numpy(blended_colors_rgb).float().cuda())
            f_dc = (
                new_f_dc_tensor.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
        dtype_full = [
            (attribute, "f4")
            for attribute in self.construct_list_of_attributes(include_confidence, include_anchor)
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        cols = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if include_confidence:
            cols.append(confidence)
        if include_anchor:
            cols.append(anchor)
        attributes = np.concatenate(cols, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
        logging.info(f"Model saved to {path}")

    def save_color_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        elements = np.empty(
            xyz.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        color = SH2RGB(f_dc.reshape(-1, 3)) * 255
        attributes = np.concatenate((xyz, color), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        file_name = os.path.basename(path)
        file_base = os.path.dirname(path)
        color_name = file_name.split(".")
        color_name = color_name[0] + "_color" + "." + color_name[1]
        color_path = os.path.join(file_base, color_name)
        PlyData([el]).write(color_path)

    def _group_by_semantic_id(self, xyz_tensor, semantic_tensor):
        """Groups XYZ coordinates by their semantic ID."""
        grouped_data = {}
        unique_sids = torch.unique(semantic_tensor)

        for sid_tensor in unique_sids:
            sid = int(sid_tensor.item())
            if sid <= 0:  # skip unlabeled / background
                continue

            mask = semantic_tensor.view(-1) == sid
            grouped_data[sid] = xyz_tensor[mask]

        return grouped_data

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_points_num(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_radius(self):
        scales = self.get_scaling
        min_length, _ = torch.min(scales, dim=1)
        radius = (torch.sum(scales, dim=1) - min_length) / 2
        return radius

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_R(self):
        return build_rotation(self.rotation_activation(self._rotation))

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    @property
    def get_normal(self):
        scales = self.get_scaling
        R = self.get_R
        min_indices = torch.argmin(scales, dim=1)
        normal = torch.gather(
            R.transpose(1, 2),
            1,
            min_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )
        normal = normal[:, 0, :]
        mag = l2_norm(normal)
        return normal / (mag + 1e-8)

    @property
    def get_plane(self):
        scales = self.get_scaling
        R = self.get_R
        plane_indices = scales.argsort(dim=1)[:, 1:]
        plane0 = torch.gather(
            R.transpose(1, 2),
            1,
            plane_indices[:, 0].unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )[:, 0, :]
        plane1 = torch.gather(
            R.transpose(1, 2),
            1,
            plane_indices[:, 1].unsqueeze(1).unsqueeze(2).expand(-1, -1, 3),
        )[:, 0, :]
        plane0 = plane0 / (l2_norm(plane0) + 1e-8)
        plane1 = plane1 / (l2_norm(plane1) + 1e-8)
        axis0 = torch.gather(scales, 1, plane_indices[:, 0:1])
        axis1 = torch.gather(scales, 1, plane_indices[:, 1:])
        return plane0, plane1, axis0, axis1

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        if features_dc.shape[1] == 3 and features_dc.shape[2] == 1:
            features_dc = features_dc.transpose(1, 2)

        if features_rest.shape[1] == 3 and features_rest.shape[2] == 15:
            features_rest = features_rest.transpose(1, 2)

        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_semantic(self):
        return self._semantic

    @property
    def get_color(self):
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
        color = SH2RGB(f_dc.reshape(-1, 3))
        return color

    @property
    def get_confidence(self):
        return self._confidence

    @property
    def get_add_tick(self):
        return self._add_tick

    @property
    def get_anchor_frame(self):
        return self._anchor_frame
