// Inspector: shows details of the selected object.

import { robot, goRobot } from '../data/robot.js?v=4';

export function createInspector(panel, subtitle) {
  let selected = null;

  function render() {
    if (!selected) {
      subtitle.textContent = '· no selection';
      panel.innerHTML = `<div class="insp insp-empty">
        <div class="insp-empty-msg">
          Click a bounding box in the viewport or a node in the scene graph to inspect.
        </div>
      </div>`;
      return;
    }

    const o = selected;
    subtitle.textContent = `· object #${o.id}`;
    const dx = o.size[0].toFixed(2);
    const dy = o.size[1].toFixed(2);
    const dz = o.size[2].toFixed(2);
    const cx = o.center[0].toFixed(3);
    const cy = o.center[1].toFixed(3);
    const cz = o.center[2].toFixed(3);
    const roomLabel = o.roomId == null ? 'unassigned' : `room ${o.roomId}`;
    const roomColor = o.roomId === 1 ? 'var(--info)' : o.roomId === 2 ? 'var(--magenta)' : 'var(--text-3)';

    panel.innerHTML = `
      <div class="insp">
        <div class="insp-head">
          <div class="id-row">
            <span class="tag"><span class="swatch" style="background:${roomColor}"></span>${roomLabel}</span>
            <span class="tag">vec #${o.vectorId}</span>
          </div>
          <h2>object #${o.id}</h2>
          <div class="sub">${dx} × ${dy} × ${dz} m · centered at (${cx}, ${cy}, ${cz})</div>
        </div>

        <div class="group">
          <div class="gh">Geometry</div>
          <div class="dims">
            <div class="d"><span class="k">Δx</span><span class="v">${dx}<small>m</small></span></div>
            <div class="d"><span class="k">Δy</span><span class="v">${dy}<small>m</small></span></div>
            <div class="d"><span class="k">Δz</span><span class="v">${dz}<small>m</small></span></div>
          </div>
        </div>

        <div class="group">
          <div class="gh">Position</div>
          <div class="kv">
            <span class="k">center.x</span><span class="v">${cx}<span class="unit">m</span></span>
            <span class="k">center.y</span><span class="v">${cy}<span class="unit">m</span></span>
            <span class="k">center.z</span><span class="v">${cz}<span class="unit">m</span></span>
            <span class="k">min</span><span class="v">${vec3(o.min)}</span>
            <span class="k">max</span><span class="v">${vec3(o.max)}</span>
          </div>
        </div>

        <div class="group">
          <div class="gh">Knowledge link</div>
          <div class="kv">
            <span class="k">vector_id</span><span class="v">${o.vectorId}</span>
            <span class="k">room_id</span><span class="v">${o.roomId ?? '—'}</span>
          </div>
          <div class="banner"><span class="b-dot"></span>vector_db query · available after indexing</div>
          <div class="action-row">
            <button data-action="frame">Frame</button>
            <button class="primary" data-action="similar" disabled>Find similar</button>
          </div>
          <div class="action-row">
            <button class="insp-go" data-action="go" ${robot.connected ? '' : 'disabled'}
              title="${robot.connected
    ? 'Submit a live trajectory to this object'
    : 'Connect the robot (top bar) to drive'}">GO — drive robot here</button>
          </div>
        </div>
      </div>
    `;

    panel.querySelector('[data-action="frame"]')?.addEventListener('click', () => {
      if (selected && handlers.onFrame) handlers.onFrame(selected);
    });
    panel.querySelector('[data-action="go"]')?.addEventListener('click', () => {
      if (selected) goRobot(false);
    });
  }

  const handlers = {};

  function vec3(v) {
    return `(${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)})`;
  }

  render();

  return {
    setSelected(o) { selected = o; render(); },
    onFrame(cb) { handlers.onFrame = cb; },
  };
}
