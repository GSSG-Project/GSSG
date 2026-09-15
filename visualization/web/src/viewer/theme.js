import * as pc from 'playcanvas';

// Two palettes. CSS vars on :root for the DOM; clearColor on the camera for
// the viewport. We also update --accent and friends.

export const PALETTES = {
  night: {
    bg: '#0a0c0f',
    'bg-elev': '#11141a',
    surface: '#161a21',
    'surface-2': '#1c2128',
    'surface-3': '#232932',
    border: 'rgba(255,255,255,0.06)',
    'border-strong': 'rgba(255,255,255,0.10)',
    'border-bright': 'rgba(255,255,255,0.16)',
    text: '#e6e4dd',
    'text-2': '#b6b8b8',
    'text-3': '#7a7e85',
    'text-4': '#4a4f57',
    accent: '#c4f542',
    info: '#6ec3ff',
    sel: '#ffb84d',
    warn: '#ff8a3d',
    danger: '#ff5a5a',
    ok: '#5ae0a0',
    magenta: '#ff7ad5',
    viewport: [0.038, 0.046, 0.058],
  },
  day: {
    bg: '#f3f1ea',
    'bg-elev': '#ebe8df',
    surface: '#ffffff',
    'surface-2': '#f7f5ed',
    'surface-3': '#e6e2d6',
    border: 'rgba(20,20,20,0.08)',
    'border-strong': 'rgba(20,20,20,0.14)',
    'border-bright': 'rgba(20,20,20,0.24)',
    text: '#1a1d20',
    'text-2': '#3d4147',
    'text-3': '#6b7079',
    'text-4': '#9aa0a8',
    accent: '#5d8a06',
    info: '#1f76c9',
    sel: '#d97706',
    warn: '#c2410c',
    danger: '#c81d3a',
    ok: '#0f8a55',
    magenta: '#b03d8a',
    viewport: [0.965, 0.953, 0.918],
  },
};

export function applyTheme(name, camera) {
  const p = PALETTES[name] || PALETTES.night;
  const root = document.documentElement;
  for (const [k, v] of Object.entries(p)) {
    if (k === 'viewport') continue;
    root.style.setProperty(`--${k}`, v);
  }
  root.setAttribute('data-theme', name);
  if (camera) {
    camera.camera.clearColor = new pc.Color(p.viewport[0], p.viewport[1], p.viewport[2]);
  }
  try { localStorage.setItem('atlas.theme', name); } catch (_) {}
}

export function getInitialTheme() {
  try {
    const t = localStorage.getItem('atlas.theme');
    if (t === 'night' || t === 'day') return t;
  } catch (_) {}
  return 'night';
}
