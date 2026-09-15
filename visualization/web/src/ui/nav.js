// Nav panel: PRM display controls + the plan/GO section.
//
// The robot connection itself lives in the TOP BAR (ui/robot-bar.js); this
// panel is about the roadmap and the current route. Selecting an object only
// previews a route — the robot moves exclusively via the GO button here (or
// the GO buttons in the inspector / query results, which share the same store).

import {
  robot, onRobot, goRobot, cancelRobot, clearPlan, setThresh, previewPlan,
} from '../data/robot.js?v=4';

export function createNav(panel, hooks) {
  const local = { show: true, hasTrajectory: false };

  panel.innerHTML = `
    <div class="tweaks nav-panel">
      <div class="tweak-toggle" data-nav-toggle="show">
        <span class="tt-name">Show trajectory / PRM</span>
        <span class="tt-sw on"></span>
      </div>
      <div class="tweak-row">
        <div class="tweak-h"><span>Link threshold</span><span class="tweak-v" id="navThreshVal"></span></div>
        <input type="range" min="0.05" max="1.5" step="0.05" value="${robot.thresh}" id="navThresh" />
      </div>
      <div class="nav-info" id="navGraphInfo">no trajectory in this scene</div>

      <div class="nav-sec">Route</div>
      <div class="nav-info" id="navGoal">no goal — select an object</div>
      <div class="nav-info" id="navPlanInfo"></div>
      <button class="nav-go-btn" id="navGo" disabled>GO</button>
      <div class="nav-btns">
        <button class="btn" id="navDry" disabled title="Full robot pipeline, commands recorded — nothing moves">Dry run</button>
        <button class="btn" id="navCancel" title="Graceful abort of the current run">Cancel run</button>
        <button class="btn" id="navClear" title="Clear the previewed route">Clear</button>
      </div>
      <div class="nav-info nav-hint">Robot connection → top bar. GO submits a live
      trajectory; Dry run exercises the robot pipeline without motion.</div>
    </div>
  `;

  const el = (id) => panel.querySelector(`#${id}`);
  const threshVal = el('navThreshVal');
  const graphInfo = el('navGraphInfo');
  const goalInfo = el('navGoal');
  const planInfo = el('navPlanInfo');
  const goBtn = el('navGo');
  const dryBtn = el('navDry');

  let lastShortcuts = null;

  function fmtThresh() { threshVal.textContent = `${robot.thresh.toFixed(2)} m`; }

  function setGraphInfo(info) {
    local.hasTrajectory = !!info;
    lastShortcuts = info ? info.shortcuts : null;
    graphInfo.textContent = info
      ? `${info.nodes} nodes · ${info.nodes - 1} sequential · ${info.shortcuts} shortcuts @ ${robot.thresh.toFixed(2)} m`
      : 'no trajectory in this scene';
  }

  // ── wiring ──────────────────────────────────────────────────────────────────

  panel.querySelector('[data-nav-toggle="show"]').addEventListener('click', (e) => {
    local.show = !local.show;
    e.currentTarget.querySelector('.tt-sw').classList.toggle('on', local.show);
    hooks.onShow?.(local.show);
  });

  let threshTimer = null;
  el('navThresh').addEventListener('input', (e) => {
    setThresh(Number(e.target.value));
    fmtThresh();
    clearTimeout(threshTimer);
    threshTimer = setTimeout(() => {
      hooks.onThresh?.(robot.thresh);   // refresh drawn edges
      if (robot.goal && robot.connected) previewPlan();  // re-route on new graph
    }, 250);
  });

  goBtn.addEventListener('click', () => goRobot(false));
  dryBtn.addEventListener('click', () => goRobot(true));
  el('navCancel').addEventListener('click', () => cancelRobot());
  el('navClear').addEventListener('click', () => clearPlan());

  // ── store → panel ───────────────────────────────────────────────────────────

  onRobot((r) => {
    goalInfo.textContent = r.goal ? `goal: ${r.goal.label}` : 'no goal — select an object';

    const ready = r.connected && r.goal && local.hasTrajectory && !r.planning;
    goBtn.disabled = !ready;
    dryBtn.disabled = !ready;
    goBtn.textContent = r.planning ? '…' : 'GO';
    goBtn.title = ready
      ? `Drive the robot to ${r.goal.label} along the previewed route`
      : !r.connected ? 'Connect the robot in the top bar first'
        : !r.goal ? 'Select an object first'
          : 'No trajectory in this scene';

    const lines = [];
    if (r.plan) {
      const t = r.plan.request.target;
      const modeLabel = { preview: 'preview', dry: 'sent (dry-run)', live: 'sent LIVE' }[r.plan.mode];
      lines.push(
        `route: ${r.plan.path_indices.length} nodes → (${t.x.toFixed(2)}, ${t.y.toFixed(2)}, `
        + `${t.yaw_deg.toFixed(0)}°) · ${modeLabel}`,
      );
      if (r.plan.localized === false) lines.push(`⚠ robot not localized (${r.plan.spatial})`);
    }
    if (r.message) lines.push(r.message);
    if (r.lastError) lines.push(`✗ ${r.lastError}`);
    planInfo.innerHTML = lines.map((l) => `<div>${l}</div>`).join('');
    planInfo.classList.toggle('bad', !!r.lastError);
  });

  fmtThresh();

  return {
    setGraphInfo,
    get show() { return local.show; },
  };
}
