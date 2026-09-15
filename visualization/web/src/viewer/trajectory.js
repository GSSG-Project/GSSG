import * as pc from 'playcanvas';

// Renders the recorded trajectory as a PRM: the sequential polyline, the
// proximity (shortcut) edges for the current threshold, the last planned path,
// and live robot / target markers. Immediate-mode (app.drawLines) every frame,
// same pattern as BoxLayer; line arrays are prebuilt on data changes so the
// per-frame cost is just the draw submissions.

const SEQ_COLOR = new pc.Color(0.30, 0.75, 1.00, 0.85);   // recorded path
const PROX_COLOR = new pc.Color(0.55, 0.60, 0.75, 0.28);  // shortcut edges
const PATH_COLOR = new pc.Color(1.00, 0.62, 0.20, 1.0);   // planned route
const PATH_GHOST = new pc.Color(1.00, 0.62, 0.20, 0.35);  // planned route through walls
const ROBOT_COLOR = new pc.Color(0.35, 1.00, 0.55, 1.0);
const TARGET_COLOR = new pc.Color(1.00, 0.35, 0.35, 1.0);

function toVec3s(nodesPC) {
  return nodesPC.map((p) => new pc.Vec3(p[0], p[1], p[2]));
}

function polylinePairs(vecs) {
  const out = [];
  for (let i = 0; i + 1 < vecs.length; i++) out.push(vecs[i], vecs[i + 1]);
  return out;
}

function fillColors(n, color) {
  return new Array(n).fill(color);
}

export class TrajectoryLayer {
  constructor(app) {
    this.app = app;
    this.visible = true;
    this.nodes = [];        // pc.Vec3 per trajectory node (PC frame)
    this._seqPairs = [];
    this._seqColors = [];
    this._proxPairs = [];
    this._proxColors = [];
    this._pathPairs = [];
    this._pathColors = [];
    this._pathGhostColors = [];
    this.robotPos = null;   // pc.Vec3 | null
    this.targetPos = null;  // pc.Vec3 | null
    this._min = new pc.Vec3();
    this._max = new pc.Vec3();
    this._tickHandler = this.tick.bind(this);
    app.on('update', this._tickHandler);
  }

  setVisible(v) { this.visible = v; }

  setNodes(nodesPC) {
    this.nodes = toVec3s(nodesPC);
    this._seqPairs = polylinePairs(this.nodes);
    this._seqColors = fillColors(this._seqPairs.length, SEQ_COLOR);
    this.setProxEdges([]);
    this.setPath(null);
  }

  // pairs: [[i, j], ...] node-index pairs.
  setProxEdges(pairs) {
    const out = [];
    for (const [i, j] of pairs) {
      if (this.nodes[i] && this.nodes[j]) out.push(this.nodes[i], this.nodes[j]);
    }
    this._proxPairs = out;
    this._proxColors = fillColors(out.length, PROX_COLOR);
  }

  // nodesPC: planned path positions (PC frame), or null to clear.
  setPath(nodesPC) {
    if (!nodesPC || nodesPC.length < 2) {
      this._pathPairs = [];
      this._pathColors = [];
      this._pathGhostColors = [];
      return;
    }
    this._pathPairs = polylinePairs(toVec3s(nodesPC));
    this._pathColors = fillColors(this._pathPairs.length, PATH_COLOR);
    this._pathGhostColors = fillColors(this._pathPairs.length, PATH_GHOST);
  }

  setRobot(posPC) {
    this.robotPos = posPC ? new pc.Vec3(posPC[0], posPC[1], posPC[2]) : null;
  }

  setTarget(posPC) {
    this.targetPos = posPC ? new pc.Vec3(posPC[0], posPC[1], posPC[2]) : null;
  }

  _drawMarker(pos, color, half) {
    this._min.set(pos.x - half, pos.y - half, pos.z - half);
    this._max.set(pos.x + half, pos.y + half, pos.z + half);
    this.app.drawWireAlignedBox(this._min, this._max, color, false);
    // Vertical flag pole so the marker reads from top-down views too.
    this.app.drawLine(pos, new pc.Vec3(pos.x, pos.y + 0.6, pos.z), color, false);
  }

  tick() {
    if (!this.visible) return;
    if (this._seqPairs.length) this.app.drawLines(this._seqPairs, this._seqColors, true);
    if (this._proxPairs.length) this.app.drawLines(this._proxPairs, this._proxColors, true);
    if (this._pathPairs.length) {
      this.app.drawLines(this._pathPairs, this._pathColors, true);
      this.app.drawLines(this._pathPairs, this._pathGhostColors, false); // visible through walls
    }
    if (this.robotPos) this._drawMarker(this.robotPos, ROBOT_COLOR, 0.14);
    if (this.targetPos) this._drawMarker(this.targetPos, TARGET_COLOR, 0.10);
  }

  dispose() {
    this.app.off('update', this._tickHandler);
  }
}
