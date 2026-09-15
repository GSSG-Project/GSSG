// Streaming PLY parser tailored to the 3DGS layout produced by this capture.
// We parse once, keep the parsed splats in memory, and rebuild a "clean" PLY
// blob each time the density slider changes — keeping only the properties
// PlayCanvas's gsplat loader expects (it does not document tolerance for
// extra fields like the `confidence` float at the end of each vertex).

const REQUIRED = [
  'x', 'y', 'z',
  'f_dc_0', 'f_dc_1', 'f_dc_2',
  'opacity',
  'scale_0', 'scale_1', 'scale_2',
  'rot_0', 'rot_1', 'rot_2', 'rot_3',
];
// f_rest_0..44 are degree-1..3 SH coefficients — keep them for fidelity.
const SH_REST = Array.from({ length: 45 }, (_, i) => `f_rest_${i}`);
const KEEP_PROPS = [...REQUIRED, ...SH_REST];

function parseHeader(buffer) {
  const view = new Uint8Array(buffer);
  // PLY header is ASCII; scan for "end_header\n".
  const needle = new TextEncoder().encode('end_header\n');
  let end = -1;
  outer: for (let i = 0; i < view.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) {
      if (view[i + j] !== needle[j]) continue outer;
    }
    end = i + needle.length;
    break;
  }
  if (end < 0) throw new Error('PLY: end_header not found');

  const text = new TextDecoder().decode(view.subarray(0, end));
  const lines = text.split('\n');
  const props = [];
  let count = 0;
  let binary = false;
  let littleEndian = true;

  for (const line of lines) {
    if (line.startsWith('format ')) {
      binary = line.includes('binary');
      littleEndian = line.includes('little_endian');
    } else if (line.startsWith('element vertex ')) {
      count = parseInt(line.split(' ')[2], 10);
    } else if (line.startsWith('property ')) {
      const [, type, name] = line.split(' ');
      props.push({ name, type });
    }
  }

  if (!binary) throw new Error('PLY: only binary little-endian PLY supported');
  if (!littleEndian) throw new Error('PLY: big-endian not supported');

  // All properties in this file are float32 — verify.
  const stride = props.reduce((s, p) => s + (p.type === 'float' ? 4 : p.type === 'uchar' ? 1 : p.type === 'double' ? 8 : 4), 0);
  return { props, count, dataOffset: end, stride };
}

export async function fetchPly(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`PLY fetch failed: ${res.status}`);
  const total = Number(res.headers.get('Content-Length') || 0);
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress && total) onProgress(received / total);
  }
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    buffer.set(c, offset);
    offset += c.length;
  }
  return buffer.buffer;
}

// Parse vertex data into a struct-of-arrays representation we can repack.
export function parsePly(buffer) {
  const { props, count, dataOffset, stride } = parseHeader(buffer);

  // Verify every prop is float32 — true for this file but worth asserting.
  for (const p of props) {
    if (p.type !== 'float') throw new Error(`PLY: unexpected non-float property ${p.name}:${p.type}`);
  }

  const dv = new DataView(buffer, dataOffset);
  const propIndex = new Map(props.map((p, i) => [p.name, i]));

  // Allocate per-property Float32Array.
  const data = {};
  for (const p of props) data[p.name] = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const base = i * stride;
    for (let j = 0; j < props.length; j++) {
      data[props[j].name][i] = dv.getFloat32(base + j * 4, true);
    }
  }

  // Build a permutation sorted by confidence DESC (fallback: opacity DESC).
  const conf = data.confidence || data.opacity;
  const indices = new Uint32Array(count);
  for (let i = 0; i < count; i++) indices[i] = i;
  // In-place sort — typed-array sort is in-place and stable enough.
  const tmp = Array.from(indices).sort((a, b) => conf[b] - conf[a]);
  for (let i = 0; i < count; i++) indices[i] = tmp[i];

  return { props, count, data, sortedIndices: indices };
}

// Rebuild a clean PLY blob containing only the first `numSplats` indices
// (in sortedIndices order) and only the KEEP_PROPS properties.
//
// opts.sizeScale / opts.opacityScale bake appearance into the stored values,
// since the stock PlayCanvas gsplat shader has no size/opacity uniform:
//   scale_* is stored as log(scale)  → multiply size  = add log(sizeScale)
//   opacity is stored as logit(o)     → scale rendered o then re-logit.
export function buildCleanPly(parsed, numSplats, opts = {}) {
  const n = Math.max(1, Math.min(numSplats | 0, parsed.count));
  const keepProps = KEEP_PROPS.filter((name) => parsed.data[name]);
  const stride = keepProps.length * 4;

  const sizeScale = opts.sizeScale ?? 1;
  const opacityScale = opts.opacityScale ?? 1;
  const logSize = sizeScale > 0 && sizeScale !== 1 ? Math.log(sizeScale) : 0;
  const isScale = keepProps.map((p) => logSize !== 0 && p.startsWith('scale_'));
  const isOpacity = keepProps.map((p) => opacityScale !== 1 && p === 'opacity');

  const header =
    'ply\n' +
    'format binary_little_endian 1.0\n' +
    `element vertex ${n}\n` +
    keepProps.map((p) => `property float ${p}`).join('\n') + '\n' +
    'end_header\n';
  const headerBytes = new TextEncoder().encode(header);

  const body = new ArrayBuffer(n * stride);
  const dv = new DataView(body);
  const idx = parsed.sortedIndices;

  for (let i = 0; i < n; i++) {
    const src = idx[i];
    const rowOff = i * stride;
    for (let j = 0; j < keepProps.length; j++) {
      let v = parsed.data[keepProps[j]][src];
      if (isScale[j]) {
        v += logSize;
      } else if (isOpacity[j]) {
        const o = 1 / (1 + Math.exp(-v)); // sigmoid → rendered opacity
        const o2 = Math.min(1 - 1e-4, Math.max(1e-4, o * opacityScale));
        v = Math.log(o2 / (1 - o2)); // back to logit
      }
      dv.setFloat32(rowOff + j * 4, v, true);
    }
  }

  const blob = new Blob([headerBytes, body], { type: 'application/octet-stream' });
  return blob;
}

// Compute the axis-aligned bounding box of all kept splats, for camera framing.
export function plyBoundingBox(parsed) {
  const { data, count } = parsed;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = data.x[i], y = data.y[i], z = data.z[i];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}
