// Data → PlayCanvas (Y-up) coordinate transform, driven by the scene's
// vertical axis from the run manifest:
//   up_axis 2  → Z-up (Replica/ROS). Bring to PC Y-up via a -90° X rotation:
//                (x, y, z) → (x, z, -y).
//   up_axis 1  → Y-up (HM3D). Already in PC's frame → identity.
//
// The splat/cell entities use the matching local Euler-X (see pcEulerX), and
// object AABBs are transformed at load so picking, framing, and labels all
// operate in one (PC Y-up) coordinate system. Call setUpAxis() once per scene
// load, before transforming any geometry.

let UP_AXIS = 2; // default Z-up preserves the historical Replica behavior

export function setUpAxis(axis) {
  UP_AXIS = axis === 1 ? 1 : 2;
}

export function getUpAxis() {
  return UP_AXIS;
}

// Local Euler-X (degrees) applied to splat/cell entities for the current axis.
export function pcEulerX() {
  return UP_AXIS === 2 ? -90 : 0;
}

// Back-compat constant (the previous hardcoded Replica value). Prefer pcEulerX().
export const REPLICA_TO_PC_EULER_X = -90;

export function pointReplicaToPC(p) {
  return UP_AXIS === 2 ? [p[0], p[2], -p[1]] : [p[0], p[1], p[2]];
}

// Inverse of pointReplicaToPC: PC Y-up world → the run's data frame. Used to hand
// viewer-space picks (object centers) back to the backend planner, which works in
// data-frame ground-plane coordinates.
export function pointPCToReplica(p) {
  return UP_AXIS === 2 ? [p[0], -p[2], p[1]] : [p[0], p[1], p[2]];
}

// (x, y, z) → (x, z, -y) preserves axis-alignment but swaps which input axis
// defines each output axis (Z-up case). Identity for the Y-up case.
export function aabbReplicaToPC(minIn, maxIn) {
  if (UP_AXIS === 2) {
    return {
      min: [minIn[0], minIn[2], -maxIn[1]],
      max: [maxIn[0], maxIn[2], -minIn[1]],
    };
  }
  return {
    min: [minIn[0], minIn[1], minIn[2]],
    max: [maxIn[0], maxIn[1], maxIn[2]],
  };
}

export function sizeReplicaToPC(sizeIn) {
  return UP_AXIS === 2 ? [sizeIn[0], sizeIn[2], sizeIn[1]] : [sizeIn[0], sizeIn[1], sizeIn[2]];
}
