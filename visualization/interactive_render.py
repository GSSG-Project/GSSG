import json
import math
import os
import time

import cv2
import numpy as np
import torch
from diff_gaussian_rasterization_depth import (
    GaussianRasterizationSettings as GaussianRasterizationSettings_depth,
)
from diff_gaussian_rasterization_depth import GaussianRasterizer as GaussianRasterizer_depth
from flask import Flask, Response, jsonify, render_template_string, request
from torch import tensor

from gssg.utils.export_frame import import_frame_data
from gssg.utils.utils import devI

app = Flask(__name__)

CONFIG = {
    "width": 1200,
    "height": 680,
    "bg_w": 0.1,
    "fov_y": 1.03,
    "znear": 0.01,
    "zfar": 100.0,
    "orbit_radius": 5.0,
}


def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (tanHalfFovX * 2 * znear)
    P[1, 1] = 2.0 * znear / (tanHalfFovY * 2 * znear)
    P[0, 2] = 0.0
    P[1, 2] = 0.0
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class SceneState:
    def __init__(self):
        self.means3D = None
        self.opacity = None
        self.scales = None
        self.rotations = None
        self.shs = None
        self.normal = None
        self.loaded = False

        self.pos = np.array(
            [2.410268783569336, 1.0370515584945679, -0.059667740017175674], dtype=np.float32
        )
        # Pitch (X), Yaw (Y), Roll (Z)
        self.rot = np.array(
            [-0.34888285398483276, 1.8058810234069824, -1.4999990463256836], dtype=np.float32
        )

        self.move_speed = 0.15
        self.look_speed = 0.03
        self.orbit_speed = 0.01
        self.pan_speed = 0.01

    def load(self, path):
        try:
            if path.endswith("data.pt"):
                data_path = path
                base = os.path.dirname(path)
            else:
                base = path
                data_path = os.path.join(path, "data.pt")

            print(f"Loading {data_path}...")
            gd = torch.load(data_path, map_location="cuda:0")
            self.means3D = gd["xyz"]
            self.opacity = gd["opacity"]
            self.scales = gd["scales"]
            self.rotations = gd["rotations"]
            self.shs = gd["shs"]
            self.normal = gd["normal"]
            self.loaded = True

            # If a sibling frame.pt exists, its pose overrides the defaults above.
            fp = os.path.join(base, "frame.pt")
            if os.path.exists(fp):
                try:
                    print("Found frame.pt, updating pose...")
                    w_view, _, cam_center = import_frame_data(path=fp)
                    self.pos = cam_center.cpu().numpy()
                    W2C = w_view.cpu().numpy()
                    self.rot[1] = math.atan2(W2C[0, 2], W2C[0, 0])
                    self.rot[0] = math.atan2(W2C[2, 1], W2C[1, 1])
                    self.rot[2] = 0.0
                except Exception:
                    print("Error loading frame.pt, keeping defaults.")
            else:
                print("No frame.pt found, using defaults.")

            return "Loaded"
        except Exception as e:
            return str(e)

    def get_vectors(self):
        rx, ry, rz = self.rot
        # Pitch (X), Yaw (Y), Roll (Z)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Rx @ Ry
        R_inv = R.T
        return R, R_inv[:, 0], R_inv[:, 1], R_inv[:, 2]  # Right, Up, Forward

    def update(self, inputs):
        if not self.loaded:
            return

        keys = inputs.get("keys", {})
        mouse = inputs.get("mouse", {})

        _, right, up, forward = self.get_vectors()

        flat_fwd = np.array([forward[0], 0, forward[2]])
        if np.linalg.norm(flat_fwd) > 0:
            flat_fwd /= np.linalg.norm(flat_fwd)

        flat_right = np.array([right[0], 0, right[2]])
        if np.linalg.norm(flat_right) > 0:
            flat_right /= np.linalg.norm(flat_right)

        if keys.get("ArrowUp"):
            self.pos += flat_fwd * self.move_speed
        if keys.get("ArrowDown"):
            self.pos -= flat_fwd * self.move_speed
        if keys.get("ArrowLeft"):
            self.pos -= flat_right * self.move_speed
        if keys.get("ArrowRight"):
            self.pos += flat_right * self.move_speed

        if keys.get("Space"):
            self.pos += np.array([0, 1, 0]) * self.move_speed

        if keys.get("KeyW"):
            self.rot[0] -= self.look_speed  # Pitch Up
        if keys.get("KeyS"):
            self.rot[0] += self.look_speed  # Pitch Down
        if keys.get("KeyA"):
            self.rot[1] += self.look_speed  # Yaw Left
        if keys.get("KeyD"):
            self.rot[1] -= self.look_speed  # Yaw Right

        if keys.get("KeyQ"):
            self.rot[2] -= self.look_speed  # Roll CCW
        if keys.get("KeyE"):
            self.rot[2] += self.look_speed  # Roll CW

        pan_x = mouse.get("pan_x", 0)
        pan_y = mouse.get("pan_y", 0)
        if pan_x or pan_y:
            self.pos -= right * pan_x * self.pan_speed
            self.pos += up * pan_y * self.pan_speed

        zoom = mouse.get("zoom", 0)
        if zoom:
            self.pos += forward * zoom * self.move_speed * 0.5

        orb_x = mouse.get("orbit_x", 0)
        orb_y = mouse.get("orbit_y", 0)

        if keys.get("KeyI"):
            orb_y = -5
        if keys.get("KeyK"):
            orb_y = 5
        if keys.get("KeyJ"):
            orb_x = -5
        if keys.get("KeyL"):
            orb_x = 5

        if orb_x or orb_y:
            pivot = self.pos + (forward * CONFIG["orbit_radius"])
            self.pos -= right * orb_x * self.orbit_speed
            self.pos += up * orb_y * self.orbit_speed
            new_fwd = pivot - self.pos
            new_fwd /= np.linalg.norm(new_fwd)
            self.rot[0] = math.asin(-new_fwd[1])
            self.rot[1] = math.atan2(new_fwd[0], new_fwd[2])
            self.rot[2] = 0.0

    def render(self):
        if not self.loaded:
            return np.zeros((CONFIG["height"], CONFIG["width"], 3), dtype=np.uint8)

        R, _, _, _ = self.get_vectors()
        t = -R @ self.pos
        view_mat = np.eye(4, dtype=np.float32)
        view_mat[:3, :3] = R
        view_mat[:3, 3] = t
        w2c = torch.tensor(view_mat).cuda().transpose(0, 1)

        width, height = CONFIG["width"], CONFIG["height"]
        fov_x = 2 * math.atan(math.tan(CONFIG["fov_y"] / 2) * (width / height))
        proj = (
            get_projection_matrix(CONFIG["znear"], CONFIG["zfar"], fov_x, CONFIG["fov_y"])
            .cuda()
            .transpose(0, 1)
        )
        full_proj = (w2c.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        cam_center = w2c.inverse()[3, :3]

        bg = tensor([CONFIG["bg_w"]] * 3, device="cuda:0")
        tanfovx = math.tan(fov_x * 0.5)
        tanfovy = math.tan(CONFIG["fov_y"] * 0.5)

        settings = GaussianRasterizationSettings_depth(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=3,
            campos=cam_center,
            opaque_threshold=0.6,
            depth_threshold=1.0,
            normal_threshold=0.5,
            color_sigma=3.0,
            prefiltered=False,
            debug=False,
            cx=(width / 2) - 0.5,
            cy=(height / 2) - 0.5,
            T_threshold=0.0001,
        )

        res = GaussianRasterizer_depth(raster_settings=settings)(
            means3D=self.means3D,
            opacities=self.opacity,
            shs=self.shs,
            colors_precomp=None,
            scales=self.scales,
            rotations=self.rotations,
            cov3D_precomp=None,
            normal_w=self.normal,
            tile_mask=devI(torch.ones((height + 15) // 16, (width + 15) // 16, dtype=torch.int32)),
        )

        img = res[0].detach().cpu().numpy().transpose(1, 2, 0)
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


scene = SceneState()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/load", methods=["POST"])
def load():
    return jsonify({"msg": scene.load(request.json.get("path", ""))})


@app.route("/control", methods=["POST"])
def control():
    scene.update(request.json)
    return jsonify({"status": "ok"})


def gen_frames():
    while True:
        frame = scene.render()
        _, buffer = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/save_json", methods=["POST"])
def save_j():
    path = f"render_{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump({"position": scene.pos.tolist(), "rotation": scene.rot.tolist()}, f)
    return jsonify({"msg": path})


@app.route("/save_image", methods=["POST"])
def save_i():
    path = f"render_{int(time.time())}.png"
    cv2.imwrite(path, cv2.cvtColor(scene.render(), cv2.COLOR_RGB2BGR))
    return jsonify({"msg": path})


@app.route("/load_json", methods=["POST"])
def load_j():
    try:
        with open(request.json.get("path", "")) as f:
            d = json.load(f)
        scene.pos = np.array(d["position"])
        scene.rot = np.array(d["rotation"])
        return jsonify({"msg": "Loaded"})
    except Exception as e:
        return jsonify({"msg": str(e)})


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Gaussian Smooth Viewer</title>
    <style>
        body { background: #111; color: #ccc; margin: 0; font-family: sans-serif; overflow: hidden; }
        #controls { position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; z-index: 100; pointer-events: auto; }
        #viewport { width: 100vw; height: 100vh; display: flex; align-items: center; justify-content: center; background: #000; }
        img { max-width: 100%; max-height: 100%; user-select: none; pointer-events: none; }
        input { background: #222; color: white; border: 1px solid #444; padding: 5px; border-radius: 4px; }
        button { cursor: pointer; background: #444; color: white; border: none; padding: 6px 12px; margin-top: 5px; border-radius: 4px; }
        button:hover { background: #555; }
        .row { margin-bottom: 5px; }
    </style>
</head>
<body>

<div id="controls">
    <div class="row">
        <input type="text" id="path" value="output/room0_v2_a1/gaussian_map/" style="width:250px">
        <button onclick="load()">Load</button>
    </div>
    <div class="row">
        <button onclick="saveJ()">Save JSON</button>
        <button onclick="saveI()">Save Img</button>
        <button onclick="loadJ()">Load JSON</button>
    </div>
    <div id="status" style="margin-top:10px; font-size: 0.9em; color: #aaa;">Ready</div>
    <div style="margin-top:10px; font-size: 0.8em; color: #888;">
        WASD: Look | Arrows: Move<br>
        Left Drag: Orbit | Right Drag: Pan<br>
        Shift+Scroll: Pan | Ctrl+Scroll: Zoom
    </div>
</div>

<div id="viewport" oncontextmenu="return false;">
    <img src="/video_feed" id="stream">
</div>

<script>
    const keys = {};
    const mouse = { orbit_x:0, orbit_y:0, pan_x:0, pan_y:0, zoom:0 };
    let leftDown = false;
    let rightDown = false;
    let lastX = 0, lastY = 0;

    function updateStatus(m) { document.getElementById('status').innerText = m; }
    function load() { fetch('/load', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path:document.getElementById('path').value})}).then(r=>r.json()).then(d=>updateStatus(d.msg)); }
    function saveJ() { fetch('/save_json', {method:'POST'}).then(r=>r.json()).then(d=>updateStatus("Saved "+d.msg)); }
    function saveI() { fetch('/save_image', {method:'POST'}).then(r=>r.json()).then(d=>updateStatus("Saved "+d.msg)); }
    function loadJ() { fetch('/load_json', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({path:prompt("Full Path to JSON:")})}).then(r=>r.json()).then(d=>updateStatus(d.msg)); }

    window.addEventListener('keydown', e => { if(e.target.tagName !== 'INPUT') keys[e.code] = true; });
    window.addEventListener('keyup', e => { keys[e.code] = false; });

    const vp = document.getElementById('viewport');

    vp.addEventListener('mousedown', e => {
        if(e.button===0) leftDown = true;
        if(e.button===2) rightDown = true;
        lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { leftDown = false; rightDown = false; });

    vp.addEventListener('mousemove', e => {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;

        if (leftDown) { 
            mouse.orbit_x += dx;
            mouse.orbit_y += dy;
        }
        if (rightDown || (e.shiftKey && leftDown)) { 
            mouse.pan_x += dx;
            mouse.pan_y += dy;
        }
    });

    vp.addEventListener('wheel', e => {
        e.preventDefault();
        if (e.ctrlKey) {
            mouse.zoom -= e.deltaY * 0.1; 
        } else if (e.shiftKey) {
            mouse.pan_x += e.deltaX; 
            mouse.pan_y += e.deltaY;
        } else {
            mouse.orbit_x += e.deltaX * 0.5; 
            mouse.orbit_y += e.deltaY * 0.5;
        }
    }, {passive:false});

    setInterval(() => {
        const hasInput = Object.values(keys).some(k=>k) || 
                         mouse.orbit_x!==0 || mouse.orbit_y!==0 || 
                         mouse.pan_x!==0 || mouse.pan_y!==0 || mouse.zoom!==0;

        if (hasInput) {
            fetch('/control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({keys: keys, mouse: mouse})
            });
            mouse.orbit_x = 0; mouse.orbit_y = 0;
            mouse.pan_x = 0; mouse.pan_y = 0;
            mouse.zoom = 0;
        }
    }, 33);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
