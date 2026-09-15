// Render-side controls: density, splat size, opacity, splat rotation X,
// occlusion mode + alphaClip threshold, ghost outline toggle.

export function createSettings(panel, initial, hooks) {
  const state = { ...initial };

  function row(key, label, min, max, step, unit) {
    return `
      <div class="tweak-row" data-key="${key}">
        <div class="tweak-h"><span>${label}</span><span class="tweak-v" data-val="${key}"></span></div>
        <input type="range" min="${min}" max="${max}" step="${step}" value="${state[key]}" data-input="${key}" />
        <input type="hidden" data-unit="${unit || ''}" />
      </div>
    `;
  }

  function toggleRow(key, label) {
    return `
      <div class="tweak-toggle" data-toggle="${key}">
        <span class="tt-name">${label}</span>
        <span class="tt-sw ${state[key] ? 'on' : ''}"></span>
      </div>
    `;
  }

  function segmentedRow(key, label, options) {
    return `
      <div class="tweak-seg" data-seg="${key}">
        <div class="tweak-h"><span>${label}</span></div>
        <div class="seg-btns">
          ${options.map((o) => `
            <button class="seg-btn ${state[key] === o.value ? 'active' : ''}" data-value="${o.value}" title="${o.hint || ''}">${o.label}</button>
          `).join('')}
        </div>
      </div>
    `;
  }

  function setValDisplay(key, fmt) {
    const el = panel.querySelector(`[data-val="${key}"]`);
    if (el) el.textContent = fmt(state[key]);
  }

  function render() {
    panel.innerHTML = `
      <div class="tweaks">
        ${row('density', 'Splat density', 0.05, 1.0, 0.05)}
        ${row('size', 'Splat size', 0.4, 2.5, 0.05)}
        ${row('opacity', 'Splat opacity', 0.2, 1.5, 0.05)}
        ${row('rotX', 'Splat tilt X', -180, 180, 90)}
        ${segmentedRow('depthMode', 'Box occlusion', [
          { value: 'blend',  label: 'Off',     hint: 'Beautiful splats, lines always on top.' },
          { value: 'soft',   label: 'Soft',    hint: 'Smooth splats, lines clipped behind splats.' },
          { value: 'dither', label: 'Strict',  hint: 'Hardest occlusion via dithered alpha. Grainy.' },
        ])}
        ${segmentedRow('boxMode', 'Show boxes', [
          { value: 'all',      label: 'All',      hint: 'Every object (far ones fade out).' },
          { value: 'nearby',   label: 'Nearby',   hint: 'Only objects around the camera target.' },
          { value: 'selected', label: 'Selected', hint: 'Only the selected/hovered object.' },
        ])}
        ${toggleRow('ghost', 'See-through selected box')}
      </div>
    `;
    setValDisplay('density', (v) => `${Math.round(v * 100)} %`);
    setValDisplay('size', (v) => `${v.toFixed(2)} ×`);
    setValDisplay('opacity', (v) => `${v.toFixed(2)} ×`);
    setValDisplay('rotX', (v) => `${v}°`);

    panel.querySelectorAll('input[type="range"]').forEach((inp) => {
      const key = inp.dataset.input;
      inp.addEventListener('input', () => {
        state[key] = Number(inp.value);
        if (key === 'density') setValDisplay(key, (v) => `${Math.round(v * 100)} %`);
        else if (key === 'rotX') setValDisplay(key, (v) => `${v}°`);
        else if (key === 'alphaClip') setValDisplay(key, (v) => v.toFixed(2));
        else setValDisplay(key, (v) => `${v.toFixed(2)} ×`);
      });
      // Debounce density (rebuilds the splat asset); fire others immediately.
      if (key === 'density') {
        let t = null;
        inp.addEventListener('input', () => {
          clearTimeout(t);
          t = setTimeout(() => hooks.onDensity(state.density), 250);
        });
      } else if (key === 'size') {
        inp.addEventListener('input', () => hooks.onSize(state.size));
      } else if (key === 'opacity') {
        inp.addEventListener('input', () => hooks.onOpacity(state.opacity));
      } else if (key === 'rotX') {
        inp.addEventListener('change', () => hooks.onRotX(state.rotX));
      }
    });

    panel.querySelectorAll('[data-toggle]').forEach((el) => {
      const key = el.dataset.toggle;
      el.addEventListener('click', () => {
        state[key] = !state[key];
        el.querySelector('.tt-sw').classList.toggle('on', state[key]);
        if (key === 'ghost') hooks.onGhost(state[key]);
      });
    });

    panel.querySelectorAll('[data-seg]').forEach((seg) => {
      const key = seg.dataset.seg;
      seg.querySelectorAll('.seg-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
          state[key] = btn.dataset.value;
          seg.querySelectorAll('.seg-btn').forEach((b) => b.classList.toggle('active', b === btn));
          if (key === 'depthMode') hooks.onDepthMode(state[key]);
          else if (key === 'boxMode') hooks.onBoxMode?.(state[key]);
        });
      });
    });
  }

  render();
  return { state };
}
