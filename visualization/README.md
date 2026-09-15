# Visualization — Scene Graph Dashboard

A single-page web UI (PlayCanvas frontend + FastAPI backend) that loads a saved
run from `output/<scene>/` (PLY + scene graph + FAISS vector DB) and lets you
inspect objects/rooms, run CLIP text queries, and drive an LLM agent over the
scene graph.

## Layout

```
visualization/
├── server.py            FastAPI backend (scene discovery, CLIP query, LLM agent, SSE)
├── blacklist.json       per-scene object-id blacklist
├── web/                 frontend (ES modules + importmap, no build step)
├── llm/
│   ├── agent.py         provider-agnostic tool-use loop
│   ├── interface.py     SceneGraphInterface / VectorDBInterface / SpatialIndexInterface
│   ├── providers.py     Anthropic / Google / OpenAI backend auto-detect
│   ├── prompts.py       system prompt
│   └── ros_controller.py  ROS 2 navigation control
├── nav.py               PRM path planning on the map + robot nav-service bridge
└── interactive_render.py   Flask real-time Gaussian PLY renderer (camera orbit)
```

## Running

```bash
# from repo root (after `pip install -e .`, PYTHONPATH no longer needed)
python visualization/server.py --scene <scene> --clip-model ViT-H-14 --port 8001
# or, with autoreload:
python -m uvicorn visualization.server:app --reload --port 8001
```

Open <http://localhost:8001>. `--scene` defaults to the newest dir under `output/`;
`--clip-model` must match what the scene was indexed with (the FAISS dim is
auto-detected at load).

Set one API key for the LLM agent (auto-detect order: Claude → Gemini → GPT):

```bash
export ANTHROPIC_API_KEY=...   # or GOOGLE_API_KEY / OPENAI_API_KEY
# force a provider/model: ATLAS_LLM_PROVIDER=anthropic|google|openai, ATLAS_LLM_MODEL=<id>
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | health + active scene + LLM provider |
| GET | `/api/scenes` | list scenes under `output/` |
| POST | `/api/scenes/select` | switch active scene |
| GET | `/api/scene` / `/api/scene/scene_graph` / `/api/scene/ply` | scene metadata / graph JSON / latest stable PLY |
| POST | `/api/clip_query` | direct CLIP top-k |
| POST | `/api/llm_query` | full agent run (transcript) |
| GET | `/api/llm_query_stream?q=` | SSE stream of agent events |

## Standalone debug viewers

```bash
python visualization/interactive_render.py   # real-time Gaussian renderer
```
