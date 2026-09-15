# THIRD_PARTY_LICENSES / NOTICE

This file lists the third-party software and model components used by GSSG and their license terms. The top-level [LICENSE](../LICENSE) is research /
non-commercial because several upstream components forbid commercial use (the
Inria/MPII Gaussian-Splatting code inherited via RTG-SLAM, and the Apple
MobileCLIP model weights), and the optional FastSAM backend is AGPL-3.0. Where
licenses conflict, the **most restrictive** terms apply to the affected component.

Components are either vendored under `thirdparty/` (most as git submodules) or
pulled in as runtime Python dependencies / model checkpoints.

---

## Components that constrain the overall license (non-commercial / copyleft)

### 3D Gaussian Splatting — rasterizer (diff-gaussian-rasterization-w-pose)
- **Source:** Inria + Max Planck Institut fuer Informatik (MPII), via RTG-SLAM
  (https://github.com/MisEty/RTG-SLAM) and the original Gaussian-Splatting
  (https://github.com/graphdeco-inria/gaussian-splatting).
- **License:** Inria Gaussian-Splatting License — **NON-COMMERCIAL**, research and
  evaluation use only. Commercial use prohibited without prior written consent
  from Inria (stip-sophia.transfert@inria.fr).
- **Where:** `thirdparty/gaussian-rasterizer/LICENSE.md` (vendored).

### simple-knn
- **Source:** fork of https://gitlab.inria.fr/bkerbl/simple-knn (Inria).
- **License:** Inria Gaussian-Splatting License (same lineage) — **NON-COMMERCIAL**,
  research-only. No separate LICENSE file ships in the working tree.
- **Where:** `thirdparty/simple-knn/` (vendored).

### cuda-utils
- **Source:** RTG-SLAM / Gaussian-Splatting lineage.
- **License:** Inria Gaussian-Splatting License — **NON-COMMERCIAL**, research-only.
  No separate LICENSE file present.
- **Where:** `thirdparty/cuda-utils/` (vendored; not a submodule).

### RTG-SLAM (architectural base)
- **Source:** https://github.com/MisEty/RTG-SLAM
- **License:** academic/research use; derived from and bound by the Inria
  Gaussian-Splatting License — **NON-COMMERCIAL**.
- **Where:** the core SLAM pipeline (`gssg/map/`, `gssg/run.py`) builds on RTG-SLAM.
  The Inria/GRAPHDECO header is retained in first-party files copied from this
  lineage: `gssg/utils/arguments.py`, `general_utils.py`, `graphics_utils.py`,
  `camera_utils.py`, `loss_utils.py`, and `gssg/dataset_reader/{__init__,cameras}.py`.

### FastSAM (via Ultralytics) — optional segmentation backend
- **Source:** Ultralytics; FastSAM-x checkpoint from CASIA-LMC-Lab.
- **License:** **AGPL-3.0** — strong network copyleft. Any conveyed or
  network-served combined work that uses FastSAM must be released under AGPL-3.0.
  Commercial / closed use requires an Ultralytics Enterprise License.
- **Where:** runtime dependency in `gssg/scene_graph/semantic_encoder.py`
  (`from ultralytics import YOLO, FastSAM`); selectable via `seg_model: "fastsam"`.

### MobileCLIP / MobileCLIP2-S0 — Apple
- **Source:** Apple Inc. (https://github.com/apple/ml-mobileclip).
- **License:** wrapper **code: MIT**. **Model weights: Apple Machine Learning
  Research Model License** — for "Research Purposes" (non-commercial) ONLY;
  excludes commercial product/service use (`LICENSE_MODELS`). **Training data:
  CC-BY-NC-ND 4.0** (`LICENSE_DATA`). The default `mobileclip2_s0.pt` checkpoint
  is therefore **NON-COMMERCIAL**.
- **Where:** `thirdparty/mobileclip/{LICENSE, LICENSE_MODELS, LICENSE_DATA}`
  (git submodule). Imported in `gssg/scene_graph/semantic_encoder.py`.

---

## Conditionally-permissive components (commercial allowed, with conditions)

### SAM 3 (Segment Anything Model 3) — Meta — default segmentation backend
- **Source:** facebook/sam3 (Meta).
- **License:** Meta "SAM License" (Nov 19, 2025). Royalty-free limited license to
  use/reproduce/distribute/modify, **including commercial use**, but subject to a
  SAM Acceptable Use Policy and Trade Controls (no military/weapons/ITAR/sanctioned
  uses); must redistribute under the same Agreement and acknowledge SAM in
  publications. Not an OSI-permissive license; weights are gated (HuggingFace token).
- **Where:** `thirdparty/sam3/LICENSE` (git submodule); default `seg_model` backend.

### BestSAM (TensorRT runtime for SAM 3) — optional backend
- **Source:** BestSAM (to be released as a companion repository).
- **License:** **code is MIT**. However, BestSAM is a
  TensorRT runtime over HuggingFace `transformers.Sam3Model` using the gated
  `facebook/sam3` weights, so the **Meta SAM License** governs the model/weights.
- **Where:** `thirdparty/bestsam/` (not included in this release; the `bestsam`
  backend falls back to SAM 3 when absent); selectable via `seg_model: "bestsam"`.

---

## Permissive components (attribution only; commercial-ok)

### PyTorch3D — Meta
- **License:** **BSD-3-Clause.** Commercial use permitted; retain copyright/disclaimer.
- **Where:** `thirdparty/pytorch3d/LICENSE` (git submodule). Used for `knn_points`
  in `gssg/map/mapper.py`.

### open_clip
- **License:** **MIT.** Commercial use permitted; requires attribution. (Individual
  pretrained weights, e.g. LAION ViT-H-14, carry their own terms.)
- **Where:** runtime dependency (`import open_clip` in `semantic_encoder.py`).

### FAISS — Meta
- **License:** **MIT.** Commercial use permitted; requires attribution.
- **Where:** runtime dependency (`import faiss` in `gssg/scene_graph/vector_db.py`) —
  the HNSW semantic index.

### PlenOctree (spherical-harmonics utilities)
- **License:** **BSD-style** (Copyright 2021 The PlenOctree Authors). Commercial use
  permitted; retain the notice.
- **Where:** header embedded in `gssg/utils/sh_utils.py`.

---

## Summary

The Inria/MPII Gaussian-Splatting code (rasterizer + simple-knn + cuda-utils,
inherited through RTG-SLAM) and the Apple MobileCLIP model weights are licensed
for non-commercial research/evaluation use only, and the optional FastSAM backend
is AGPL-3.0. Therefore the GSSG repository **as a whole is licensed for
non-commercial, academic, and research use only.** The remaining components — SAM 3
(Meta SAM License, commercial-with-conditions), BestSAM (MIT code over SAM weights),
MobileCLIP wrapper code (MIT), PyTorch3D (BSD-3-Clause), open_clip (MIT), and
PlenOctree (BSD) — are compatible and require only attribution and notice retention,
which this file provides.

For commercial licensing of the upstream Gaussian-Splatting code, contact Inria
(stip-sophia.transfert@inria.fr). For commercial use of FastSAM, obtain an
Ultralytics Enterprise License. For SAM 3, comply with Meta's SAM License,
Acceptable Use Policy, and Trade Controls. The Apple MobileCLIP weights are not
available for commercial use under the shipped license.
