import gc
import logging
import os
import time as _time

import cv2
import numpy as np
import open_clip
import rerun as rr
import torch
import torch.nn.functional as F


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


import torchvision.transforms as T
from mobileclip.modules.common.mobileone import reparameterize_model
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from ultralytics import FastSAM

from gssg.utils.encoder_utils import make_square_bboxes
from gssg.utils.object_names import OBJECT_NAMES
from gssg.utils.paths import checkpoint, repo_path
from gssg.utils.utils import to_numpy

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

FAST_SAM_MODEL_PATH = checkpoint("FastSAM-x.pt")
# YOLOE prompt-free open-vocab segmentation. Falls back to the bare name (ultralytics
# auto-downloads) when not present in GSSG_CHECKPOINTS.
YOLOE_MODEL_PATH = checkpoint("yoloe-11l-seg-pf.pt")
SAM3_BPE_PATH = repo_path("thirdparty", "bpe_simple_vocab_16e6.txt.gz")
MOBILECLIP_MODEL_NAME = "MobileCLIP2-S0"
MOBILECLIP_PRETRAINED_PATH = checkpoint("mobileclip2_s0.pt")


class SemanticEncoder:
    def __init__(self, args=None, clip_only=False):
        self.fastsam = None
        self.sam3 = None
        self.bestsam = None
        self.yoloe = None
        self.mobileclip_model = None
        self.mobileclip_preprocess = None
        self.mobile_clip_tokenizer = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.semantic_id = 1  # 0 means N/A (background)
        self.features_by_object_names = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.objectness_axis = None
        self.clip_only = clip_only

        self.seg_model = "fastsam"
        self.seg_min_area = 500.0
        self.seg_max_area = float("inf")

        self.debug = True
        if args and hasattr(args, "clip_model"):
            self.clip_model_name = args.clip_model
        else:
            self.clip_model_name = "ViT-H-14"

        print(f"Using CLIP model: {self.clip_model_name}")

        # Backend selection:
        #   MobileCLIP*  -> thirdparty/mobileclip + local checkpoint
        #   ViT-H-14     -> OpenCLIP LAION
        #   ViT-L-14 etc -> OpenAI CLIP
        self.use_mobileclip = self.clip_model_name.startswith("MobileCLIP")

        if self.use_mobileclip:
            # Local checkpoint path (overridable via config). open_clip loads it
            # with pretrained=<path>.
            _ckpt = (
                getattr(args, "clip_checkpoint_path", None) if args else None
            ) or MOBILECLIP_PRETRAINED_PATH
            self.pretrained_path = os.path.expanduser(_ckpt)
        elif self.clip_model_name == "ViT-H-14":
            self.pretrained_path = "laion2b_s32b_b79k"
        else:
            self.pretrained_path = "openai"

        if args and not self.clip_only:
            self.seg_model = args.seg_model
            self.seg_min_area = args.seg_min_area
            self.seg_max_area = args.seg_max_area

        # CLIP crop pipeline knobs
        self.clip_mask_blur = getattr(args, "clip_mask_blur", False) if args else False

        # Per-frame SAM/CLIP timing accumulators (printed every fps_log_every frames).
        self._seg_ms = 0.0
        self._clip_ms = 0.0  # encode_image (model only)
        self._clip_total_ms = 0.0  # full feature extraction incl. pre-encode
        self._post_ms = 0.0  # mask postproc inside get_map (cv2.fillPoly loop)
        self._tim_frames = 0
        self._tim_log_every = getattr(args, "fps_log_every", 20) if args else 20

        # BestSAM (optional) — only used when seg_model == "bestsam"
        self.bestsam_engine_path = (
            getattr(
                args,
                "bestsam_engine_path",
                "thirdparty/bestsam/engines/bestsam_fp16_r560.plan",
            )
            if args
            else "thirdparty/bestsam/engines/bestsam_fp16_r560.plan"
        )
        self.bestsam_prompt = (
            getattr(args, "bestsam_prompt", "visible object") if args else "visible object"
        )
        self.bestsam_score_threshold = (
            getattr(args, "bestsam_score_threshold", 0.25) if args else 0.25
        )
        self.bestsam_mask_threshold = getattr(args, "bestsam_mask_threshold", 0.5) if args else 0.5
        # YOLOE (optional) — only used when seg_model == "yoloe".
        self.yoloe_conf = getattr(args, "yoloe_conf", 0.25) if args else 0.25
        self.yoloe_imgsz = getattr(args, "yoloe_imgsz", 1024) if args else 1024
        # MobileCLIP S0/S1/S2 (and L-14 variant) expect raw 0-1 input — they bake the
        # channel mean/std into the model. OpenAI/LAION CLIP needs standard normalization.
        mobileclip_no_norm = self.use_mobileclip and not (
            self.clip_model_name.endswith("S3")
            or self.clip_model_name.endswith("S4")
            or self.clip_model_name.endswith("L-14")
        )
        if mobileclip_no_norm:
            self.clip_normalizer = torch.nn.Identity()
        else:
            self.clip_normalizer = T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            )

        self.setup()

    def setup(self):
        if not self.clip_only:
            # Heal device-specific backend choices (a config's BestSAM engine may not be
            # built on this machine) so the same config runs on desktop and Jetson.
            from gssg.utils.platform import resolve_seg_backend

            self.seg_model, self.bestsam_engine_path = resolve_seg_backend(
                self.seg_model, self.bestsam_engine_path
            )
            if self.seg_model == "sam3":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                sam3_model = build_sam3_image_model(bpe_path=SAM3_BPE_PATH)
                sam3_model.to(self.device)
                self.sam3 = Sam3Processor(sam3_model, confidence_threshold=0.25)
                logging.info("SAM3 Model loaded successfully.")
            elif self.seg_model == "bestsam":
                # Deferred import — keeps SAM3-only / FastSAM-only paths working on
                # machines without tensorrt.
                import sys

                bestsam_dir = repo_path("thirdparty", "bestsam")
                if bestsam_dir not in sys.path:
                    sys.path.insert(0, bestsam_dir)
                from bestsam import BestSAM  # noqa: E402

                engine_path = os.path.expanduser(self.bestsam_engine_path)
                self.bestsam = BestSAM(
                    engine_path,
                    prompt=self.bestsam_prompt,
                    score_threshold=self.bestsam_score_threshold,
                    mask_threshold=self.bestsam_mask_threshold,
                )
                logging.info(
                    f"BestSAM loaded (engine={engine_path}, prompt='{self.bestsam_prompt}')."
                )
            elif self.seg_model == "yoloe":
                # Open-vocab segmentation (Ultralytics YOLOE prompt-free), lazily imported
                # like bestsam. The -pf weight fuses the vocab head (no runtime set_vocab);
                # CLIP re-embeds every mask crop, so YOLOE's class labels are discarded and
                # only masks/boxes/conf are consumed.
                from ultralytics import YOLOE

                yoloe_path = (
                    YOLOE_MODEL_PATH if os.path.exists(YOLOE_MODEL_PATH) else "yoloe-11l-seg-pf.pt"
                )
                self.yoloe = YOLOE(yoloe_path)
                logging.info(f"YOLOE prompt-free seg loaded ({yoloe_path}).")
            else:
                self.fastsam = FastSAM(FAST_SAM_MODEL_PATH)
        # CLIP / MobileCLIP — same open_clip API, different model registration.
        if self.use_mobileclip:
            # MobileCLIP S0/S1/S2 (and L-14) bake mean/std into the model — pass identity
            # stats so open_clip's preprocess doesn't double-normalize.
            model_kwargs = {}
            if not (
                self.clip_model_name.endswith("S3")
                or self.clip_model_name.endswith("S4")
                or self.clip_model_name.endswith("L-14")
            ):
                model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.pretrained_path,
                **model_kwargs,
            )
            # Fold training-time BN/MobileOne structures into a single conv per block
            # (pure-inference speedup).
            self.clip_model = reparameterize_model(self.clip_model)
        else:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.pretrained_path,
            )
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)

        # Input HxW the encoder was trained at. open_clip exposes this on the preprocess
        # transform; fall back to 224 if not found.
        try:
            size = self.clip_preprocess.transforms[0].size
            self.clip_input_size = int(size[0] if isinstance(size, (tuple, list)) else size)
        except Exception:
            self.clip_input_size = 256 if self.use_mobileclip else 224

    def get_map(self, frame, visualize, sigma=3, radius=5):
        """Run one segmentation pass plus one batched CLIP forward over (per-bbox crops +
        whole frame), then build the semantic / objectness maps from the polygons.

        Returns: (semantic_map[H,W,1], semantic_results{id:(feat_cpu,conf)},
                  objectness_map[H,W,1], frame_embedding[1,D]).
        """
        H, W = frame.shape[1], frame.shape[2]

        masks, bboxes, confidences, _ = self.segment_image(frame)
        K_in = len(masks)

        # Whole-image input (always — the scene graph needs the embedding).
        whole_in = frame if frame.ndim == 4 else frame.unsqueeze(0)
        whole_in = F.interpolate(
            whole_in,
            size=(self.clip_input_size, self.clip_input_size),
            mode="bilinear",
            align_corners=False,
        )  # (1, 3, S, S)

        # Single batched CLIP encode over (K crops + 1 whole).
        _cuda_sync()
        _outer_t0 = _time.perf_counter()

        if K_in > 0:
            crops_batch = make_square_bboxes(
                frame,
                bboxes,
                masks,
                target_size=self.clip_input_size,
                apply_mask=self.clip_mask_blur,
            )  # (K, 3, S, S)
        else:
            crops_batch = torch.empty(
                (0, 3, self.clip_input_size, self.clip_input_size),
                device=whole_in.device,
                dtype=whole_in.dtype,
            )

        if crops_batch.shape[0] > 0:
            batch = torch.cat([crops_batch, whole_in], dim=0)  # (K+1, 3, S, S)
        else:
            batch = whole_in  # (1, 3, S, S)
        batch = self.clip_normalizer(batch)

        _cuda_sync()
        t0 = _time.perf_counter()
        with torch.no_grad():
            feats = self.clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        _cuda_sync()
        self._clip_ms += (_time.perf_counter() - t0) * 1000.0

        K = crops_batch.shape[0]
        if K > 0:
            # Single CPU transfer for all K per-bbox features.
            per_bbox_feats_cpu = feats[:K].cpu()
            frame_embedding = feats[K : K + 1]
        else:
            per_bbox_feats_cpu = None
            frame_embedding = feats[0:1]

        self._clip_total_ms += (_time.perf_counter() - _outer_t0) * 1000.0

        # Build semantic_map / objectness_map / semantic_results.
        _post_t0 = _time.perf_counter()
        semantic_map = torch.zeros((H, W), dtype=torch.int32)
        objectness_map = torch.zeros((H, W), dtype=torch.float32)
        semantic_results = {}
        annotation_entries = [] if visualize else None

        # make_square_bboxes can skip degenerate bboxes; iterate min length.
        n_kept = min(K, K_in)
        for i in range(n_kept):
            mask_xy = masks[i]
            conf = confidences[i]
            mask_img = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask_img, [mask_xy.astype(np.int32)], 1)
            mask_tensor = torch.from_numpy(mask_img.astype(bool))
            conf_tensor = torch.tensor(conf, dtype=torch.float32)
            semantic_map[mask_tensor] = self.semantic_id
            objectness_map[mask_tensor] = conf_tensor
            semantic_results[self.semantic_id] = (per_bbox_feats_cpu[i].unsqueeze(0), conf)
            if visualize:
                color = tuple(np.random.randint(0, 255, 3).tolist())
                annotation_entries.append((str(self.semantic_id), f"obj_{self.semantic_id}", color))
            self.semantic_id += 1

        if visualize:
            rr.log("scene/annotations", rr.AnnotationContext(annotation_entries), static=True)
            semantic_np = semantic_map.cpu().numpy().astype(np.uint8)
            rr.log("scene/semantic_map", rr.SegmentationImage(semantic_np))
            rr.log("scene/objectness_map", rr.Image(objectness_map.cpu().numpy()))

        self._post_ms += (_time.perf_counter() - _post_t0) * 1000.0

        # Accumulators are flushed by get_and_reset_perf_stats() on the FPS log cadence.
        self._tim_frames += 1

        return (
            semantic_map.unsqueeze(-1),
            semantic_results,
            objectness_map.unsqueeze(-1),
            frame_embedding,
        )

    def get_semantic_features(self, frame_tensor, bboxes, masks=None):
        """Encode per-bbox CLIP features for a frame. get_map() batches these with the
        whole-image embedding; this is kept for external callers."""
        _cuda_sync()
        _outer_t0 = _time.perf_counter()
        if frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.unsqueeze(0)

        crops_batch = make_square_bboxes(
            frame_tensor,
            bboxes,
            masks,
            target_size=self.clip_input_size,
            apply_mask=self.clip_mask_blur,
        )
        if crops_batch.shape[0] == 0:
            _cuda_sync()
            self._clip_total_ms += (_time.perf_counter() - _outer_t0) * 1000.0
            return []

        image_batch = self.clip_normalizer(crops_batch)

        _cuda_sync()
        t0 = _time.perf_counter()
        with torch.no_grad():
            batch_features = self.clip_model.encode_image(image_batch)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
        _cuda_sync()
        self._clip_ms += (_time.perf_counter() - t0) * 1000.0

        object_features = [f.unsqueeze(0) for f in batch_features]
        self._clip_total_ms += (_time.perf_counter() - _outer_t0) * 1000.0
        return object_features

    def get_and_reset_perf_stats(self, n: int):
        """Return per-frame timing averages over the last `n` frames and reset the
        accumulators."""
        n = max(1, int(n))
        stats = {
            "sam_ms": self._seg_ms / n,
            "clip_ms": self._clip_ms / n,
            "clip_prep_ms": (self._clip_total_ms - self._clip_ms) / n,
            "mask_post_ms": self._post_ms / n,
        }
        self._seg_ms = 0.0
        self._clip_ms = 0.0
        self._clip_total_ms = 0.0
        self._post_ms = 0.0
        return stats

    def release(self):
        """Free all GPU models (seg backend + CLIP). The encoder is unusable afterwards."""
        bestsam = getattr(self, "bestsam", None)
        if bestsam is not None:
            try:
                bestsam.close()
            except Exception:
                pass
        for name in ("sam3", "bestsam", "yoloe", "fastsam", "clip_model", "clip_preprocess"):
            if hasattr(self, name):
                setattr(self, name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_whole_image_feature(self, frame_tensor):
        """Encode the whole-image CLIP embedding. get_map() also returns this as its 4th
        value; this wrapper is kept for external callers."""
        input_img = frame_tensor if frame_tensor.ndim == 4 else frame_tensor.unsqueeze(0)
        resized = F.interpolate(
            input_img,
            size=(self.clip_input_size, self.clip_input_size),
            mode="bilinear",
            align_corners=False,
        )
        normalized = self.clip_normalizer(resized)
        with torch.no_grad():
            emb = self.clip_model.encode_image(normalized)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def segment_image(self, frame):
        _cuda_sync()
        t0 = _time.perf_counter()
        if self.seg_model == "sam3":
            out = self.get_segmentation_sam3(frame)
        elif self.seg_model == "bestsam":
            out = self.get_segmentation_bestsam(frame)
        elif self.seg_model == "yoloe":
            out = self.get_segmentation_yoloe(frame)
        else:
            out = self.get_segmentation_fastsam(frame)
        _cuda_sync()
        self._seg_ms += (_time.perf_counter() - t0) * 1000.0
        return out

    def get_segmentation_bestsam(self, frame):
        # `frame` is a CHW float tensor in [0, 1]; BestSAM.infer takes HxWx3 uint8 RGB.
        if frame.ndim == 4:
            frame = frame.squeeze(0)
        img_np = (frame.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)

        result = self.bestsam.infer(img_np)
        masks = result["masks"]  # (K, H, W) bool
        bboxes = result["boxes"]  # (K, 4) xyxy float32
        confidences = result["scores"]  # (K,) float32

        if masks.shape[0] == 0:
            return [], np.array([]), np.array([]), np.array([])

        binary_masks = masks.astype(np.uint8)
        masks, bboxes, confs = self._resolve_overlapping_masks_pixelwise(
            binary_masks, bboxes, confidences
        )
        return self._masks_to_polygons(masks, bboxes, confs)

    def get_segmentation_sam3(self, frame):
        inference_state = self.sam3.set_image(frame.squeeze(0))
        self.sam3.reset_all_prompts(inference_state)
        results = self.sam3.set_text_prompt(state=inference_state, prompt="visible object")

        masks = to_numpy(results["masks"])
        confidences = to_numpy(results["scores"])
        bboxes = to_numpy(results["boxes"])

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        binary_masks = (masks > 0).astype(np.uint8)

        masks, bboxes, confs = self._resolve_overlapping_masks_pixelwise(
            binary_masks, bboxes, confidences
        )
        return self._masks_to_polygons(masks, bboxes, confs)

    def get_segmentation_fastsam(self, frame):
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)

        original_H, original_W = frame.shape[2:]
        target_H = (original_H // 32) * 32
        target_W = (original_W // 32) * 32
        resized_frame = F.interpolate(
            frame, size=(target_H, target_W), mode="bilinear", align_corners=False
        )

        results = self.fastsam.predict(
            resized_frame,
            imgsz=1024,
            conf=0.30,
            iou=0.8,
            device="cuda",
            half=True,
            retina_masks=True,
            verbose=False,
        )

        if results[0].masks is None:
            return [], [], [], []

        raw_masks = results[0].masks.data
        if raw_masks.shape[1:] != (original_H, original_W):
            raw_masks = F.interpolate(
                raw_masks.unsqueeze(1),
                size=(original_H, original_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        binary_masks = (raw_masks > 0.5).cpu().numpy().astype(np.uint8)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        masks, bboxes, confs = self._resolve_overlapping_masks_pixelwise(
            binary_masks, bboxes, confidences
        )
        return self._masks_to_polygons(masks, bboxes, confs)

    def get_segmentation_yoloe(self, frame):
        # YOLOE prompt-free open-vocab segmentation. Shares the ultralytics Model API with
        # FastSAM; only masks/boxes/conf are used (labels discarded, CLIP re-embeds crops).
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)

        original_H, original_W = frame.shape[2:]
        target_H = (original_H // 32) * 32
        target_W = (original_W // 32) * 32
        resized_frame = F.interpolate(
            frame, size=(target_H, target_W), mode="bilinear", align_corners=False
        )

        results = self.yoloe.predict(
            resized_frame,
            imgsz=self.yoloe_imgsz,
            conf=self.yoloe_conf,
            iou=0.7,
            device="cuda",
            half=True,
            retina_masks=True,
            verbose=False,
        )

        if results[0].masks is None:
            return [], [], [], []

        raw_masks = results[0].masks.data
        if raw_masks.shape[1:] != (original_H, original_W):
            raw_masks = F.interpolate(
                raw_masks.unsqueeze(1),
                size=(original_H, original_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        binary_masks = (raw_masks > 0.5).cpu().numpy().astype(np.uint8)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        masks, bboxes, confs = self._resolve_overlapping_masks_pixelwise(
            binary_masks, bboxes, confidences
        )
        return self._masks_to_polygons(masks, bboxes, confs)

    def _resolve_overlapping_masks_pixelwise(self, binary_masks, bboxes, confidences):
        if len(confidences) == 0:
            return binary_masks, bboxes, confidences

        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.cpu().numpy()
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()

        confidences = confidences.flatten()
        if binary_masks.ndim == 4:
            binary_masks = binary_masks.squeeze(1)
        H, W = binary_masks.shape[1:]

        # Painter's algorithm: process low confidence first so high confidence overwrites.
        sorted_indices = np.argsort(confidences)

        ownership_map = np.full((H, W), -1, dtype=np.int32)
        for idx in sorted_indices:
            mask_bool = binary_masks[idx] > 0
            ownership_map[mask_bool] = idx

        # Extract trimmed masks and filter by area.
        final_masks = []
        final_bboxes = []
        final_confs = []

        min_area = getattr(self, "seg_min_area", 500.0)
        max_area = getattr(self, "seg_max_area", float("inf"))

        for idx in range(len(confidences)):
            trimmed_mask = ownership_map == idx
            area = np.count_nonzero(trimmed_mask)

            if min_area <= area <= max_area:
                final_masks.append(trimmed_mask.astype(np.uint8))
                final_bboxes.append(bboxes[idx])
                final_confs.append(confidences[idx])

        if not final_masks:
            return np.array([]), np.array([]), np.array([])

        return np.array(final_masks), np.array(final_bboxes), np.array(final_confs)

    def _masks_to_polygons(self, binary_masks, bboxes, confidences):
        """Convert binary masks to polygon contours, dropping ones below the area threshold."""
        if len(binary_masks) == 0:
            return [], np.array([]), np.array([]), np.array([])

        filtered_masks = []
        filtered_bboxes = []
        filtered_confidences = []
        filtered_area_sizes = []

        # Re-check min/max area in case splitting created degenerate shapes (e.g. lines).
        min_area = getattr(self, "seg_min_area", 500.0)
        max_area = getattr(self, "seg_max_area", float("inf"))

        for i, mask_img in enumerate(binary_masks):
            if mask_img.max() <= 1:
                mask_img = mask_img * 255

            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if min_area <= area <= max_area:
                polygon = (
                    largest_contour.squeeze(1) if largest_contour.ndim == 3 else largest_contour
                )
                filtered_masks.append(polygon)
                filtered_bboxes.append(bboxes[i])
                filtered_confidences.append(float(confidences[i]))
                filtered_area_sizes.append(float(area))

        return (
            filtered_masks,
            np.array(filtered_bboxes),
            np.array(filtered_confidences),
            np.array(filtered_area_sizes),
        )

    def encode_text(self, text: str) -> torch.Tensor:
        tokenized_text = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokenized_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.squeeze(0)

    def assign_word_features(self):
        logging.info("Assigning object features")
        for object_name in OBJECT_NAMES:
            feature_vector = self.encode_text(object_name)
            self.features_by_object_names[object_name] = feature_vector
        logging.info("Assigning object features completed")
