# GSSG: Semantic Gaussian Splatting SLAM with a Queryable Scene Graph

GSSG reconstructs a 3D Gaussian Splatting map from an RGB-D stream and, online, builds a
hierarchical scene graph (objects → rooms → floors) with open-vocabulary CLIP embeddings.
The map and graph can be browsed in a web viewer and queried in natural language via an
LLM agent. Out-of-core submapping keeps GPU memory bounded on long traverses.

Code release for an anonymous submission. Research / non-commercial use only — see
[LICENSE](LICENSE) and [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md).

## Install

```bash
git clone --recursive <this repo> && cd GSSG
conda env create -f environment.yaml        # environment-cuda13.yaml on CUDA 13,
conda activate gssg                         # environment-jetson.yaml / environment-thor.yaml on Jetson
export CUDA_HOME=$CONDA_PREFIX LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install -e thirdparty/simple-knn/ --no-build-isolation
pip install -e thirdparty/gaussian-rasterizer/ --no-build-isolation --config-settings editable_mode=compat
pip install -e thirdparty/cuda-utils/ --no-build-isolation --config-settings editable_mode=compat
pip install -e thirdparty/pytorch3d/ --no-build-isolation
pip install -e thirdparty/mobileclip/ thirdparty/sam3/
pip install -e . && pip install numpy==1.26.4
```

`glm/glm.hpp not found` → `conda install -c conda-forge ninja glm`.

**Checkpoints** go in `checkpoints/` (or `$GSSG_CHECKPOINTS`): `mobileclip2_s0.pt`
([MobileCLIP2-S0](https://huggingface.co/apple/MobileCLIP2-S0)), optionally `FastSAM-x.pt`.
SAM 3 weights download from HuggingFace (`facebook/sam3`, gated — set `HF_TOKEN`).

> The TensorRT SAM 3 backend (`seg_model: "bestsam"`, used by the `low`/`mid` presets) is
> released separately and is not required: when it is absent GSSG falls back to SAM 3
> in PyTorch automatically.

## Run

```bash
# Replica: https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -> data/replica/<scene>
python gssg/run.py --config configs/_scenes/replica_room0.yaml --quality mid
python gssg/run.py --config configs/_scenes/replica_office0_submap.yaml   # out-of-core submapping

# Live ZED 2i over ROS 2 (multi-process, real-time budget)
python gssg/run_mp.py --config configs/datasets/ros2_zed.yaml

./bin/gssg-run            # or: interactive wizard (env, ROS and GPU backend resolved for you)
```

Configs are layered (`base` ← `quality/{low,mid,high,live,replay}` ← `datasets/*` ←
`_scenes/*`); see [configs/README.md](configs/README.md). Each run writes the map (PLY),
the scene graph (JSON + FAISS index), a resolved `config.yaml`, `runtime.json`
(per-stage timings) and an OpenLex3D-format export under `save_path/`.

Offline room segmentation of a saved map:
`python -m gssg.scene_graph.room_segmentation --input <stable.ply> --method ours_v2`.

## Viewer and LLM agent

```bash
export ANTHROPIC_API_KEY=...   # or GOOGLE_API_KEY / OPENAI_API_KEY
python visualization/server.py --scene <run-name-under-output/> --port 8001   # http://localhost:8001
```

See [visualization/README.md](visualization/README.md).

## Citation

```bibtex
@inproceedings{gssg2026,
  title     = {GSSG: Semantic Gaussian Splatting SLAM with a Queryable Scene Graph},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```

## Acknowledgements

Built on [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(Inria / MPII). Uses [SAM 3](https://github.com/facebookresearch/sam3),
[MobileCLIP2](https://github.com/apple/ml-mobileclip), [OpenCLIP](https://github.com/mlfoundations/open_clip),
[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), [PyTorch3D](https://github.com/facebookresearch/pytorch3d),
[FAISS](https://github.com/facebookresearch/faiss) and [PlayCanvas](https://playcanvas.com).
Room-segmentation baselines reimplement [Hydra](https://github.com/MIT-SPARK/Hydra) and
[HOV-SG](https://github.com/hovsg/HOV-SG). Evaluation uses
[OpenLex3D](https://github.com/OpenLex3D/openlex3d), [Replica](https://github.com/facebookresearch/Replica-Dataset)
and [HM3D](https://aihabitat.org/datasets/hm3d/).
