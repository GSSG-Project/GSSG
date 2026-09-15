// Corner HUD stats. Lightweight — updates from main loop callbacks.

export function createHud(els) {
  els.hudTL.innerHTML = `
    <div class="stat"><span class="lbl">scene</span><b id="hudScene">room-2 · frame 0066</b></div>
  `;
  els.hudTR.innerHTML = `
    <div class="stat"><span class="lbl">fps</span><b id="hudFps">—</b></div>
    <div class="stat"><span class="lbl">splats</span><b id="hudSplats">—</b></div>
  `;
  els.hudBR.innerHTML = `
    <div class="stat"><span class="lbl">cam</span><b id="hudCam">—</b></div>
  `;
  els.hudBL.innerHTML = `
    <div class="ticker">
      <span class="stat" style="border-color:var(--accent);"><span class="lbl">ready</span><b style="color:var(--accent)">LIVE</b></span>
    </div>
  `;

  const fpsEl = els.hudTR.querySelector('#hudFps');
  const splatsEl = els.hudTR.querySelector('#hudSplats');
  const camEl = els.hudBR.querySelector('#hudCam');

  return {
    setFps(n) { fpsEl.textContent = n.toFixed(0); },
    setSplats(n) { splatsEl.textContent = formatK(n); },
    setCam(p, target) {
      camEl.textContent = `(${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}) → (${target.x.toFixed(1)}, ${target.y.toFixed(1)}, ${target.z.toFixed(1)})`;
    },
  };
}

export function createStatusBar(container) {
  container.innerHTML = `
    <div class="sb-section sb-left">
      <span class="sb-pill"><span class="led"></span>idle</span>
      <span class="sb-text" id="sbSelected">no selection</span>
    </div>
    <div class="sb-section sb-center">
      <span class="sb-text" id="sbCount">— splats</span>
    </div>
    <div class="sb-section sb-right">
      <span class="sb-text">click = pick · drag = orbit · arrows = move · WASD/IJKL = look · space/shift = up·down</span>
    </div>
  `;
  return {
    setSelected(o) {
      const el = container.querySelector('#sbSelected');
      if (!o) el.textContent = 'no selection';
      else el.textContent = `object #${o.id} · ${o.roomId == null ? 'unassigned' : 'room ' + o.roomId}`;
    },
    setCount(n) {
      container.querySelector('#sbCount').textContent = `${formatK(n)} splats`;
    },
  };
}

function formatK(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}
