// Builds the static DOM skeleton: top bar, left rail, viewport, right rail,
// status bar. The right rail now hosts a tabbed area with two panels:
//   • Inspector (selected object details, identical to gssg-viewer)
//   • Query     (CLIP top-k + LLM agent, new)

export function buildShell(root) {
  root.innerHTML = `
    <div class="app">
      <div class="topbar">
        <span class="logo">
          <span class="dot" id="logoDot"></span>
          ATLAS
        </span>
        <span class="crumbs">
          <span>GSSG</span><span class="sep">/</span>
          <span class="here" id="crumbScene">— · loading</span>
        </span>
        <span class="spacer"></span>
        <span class="robot-bar" id="robotBar"></span>
        <select id="sceneSelect" class="scene-select" title="Active scene"></select>
        <span class="pill"><span class="led"></span><span id="providerPill">LLM —</span></span>
        <span class="pill" id="splatCountPill"><span class="led" style="background:var(--info);box-shadow:0 0 6px var(--info);"></span><span id="splatCountText">— splats</span></span>
        <button class="btn" id="themeToggle" title="Toggle day/night">☾</button>
        <button class="btn primary" id="frameAllBtn">Frame scene</button>
      </div>

      <div class="workspace">
        <div class="rail">
          <div class="panel">
            <div class="panel-h">Layers</div>
            <div class="panel-body fixed" id="layersPanel"></div>
          </div>

          <div class="panel" style="flex:1">
            <div class="panel-h">Scene Graph <span style="color:var(--text-4); margin-left:4px;">· objects</span></div>
            <div class="tree-search"><input id="sgSearch" placeholder="Search by id, room…" /></div>
            <div class="panel-body" id="sceneGraphPanel"></div>
          </div>

          <div class="panel">
            <div class="panel-h">Rooms <span style="color:var(--text-4); margin-left:4px;" id="roomFilterCount">· all</span></div>
            <div class="panel-body fixed" id="roomFilterPanel"></div>
          </div>
        </div>

        <div class="viewport-wrap">
          <div class="viewport" id="viewport">
            <div class="grid-bg"></div>
            <canvas id="viewport-canvas"></canvas>
            <div class="hud" id="hud">
              <div class="labels" id="labelLayer"></div>
              <div class="corner tl" id="hudTL"></div>
              <div class="corner tr" id="hudTR"></div>
              <div class="corner bl" id="hudBL"></div>
              <div class="corner br" id="hudBR"></div>
              <div class="floor-strip" id="floorStrip"></div>
              <div class="vp-tools" id="vpTools"></div>
            </div>
            <div class="loader" id="loader">
              <div class="loader-inner">
                <div class="loader-title">LOADING SCENE</div>
                <div class="loader-bar"><div class="loader-fill" id="loaderFill"></div></div>
                <div class="loader-status" id="loaderStatus">connecting…</div>
              </div>
            </div>
          </div>
          <div class="statusbar" id="statusbar"></div>
        </div>

        <div class="rail right">
          <div class="rtabs">
            <span class="rtab active" data-rtab="inspector">Inspector</span>
            <span class="rtab" data-rtab="query">Query</span>
            <span class="rtab" data-rtab="nav">Nav</span>
            <span class="rtab" data-rtab="render">Render</span>
          </div>

          <div class="rtab-panel active" data-rtab-panel="inspector">
            <div class="panel" style="flex:1; border-bottom:0;">
              <div class="panel-h">Inspector <span style="color:var(--text-4); margin-left:4px;" id="inspectorSubtitle">· no selection</span></div>
              <div class="panel-body" id="inspectorPanel"></div>
            </div>
          </div>

          <div class="rtab-panel" data-rtab-panel="query">
            <div class="panel" style="flex:1; border-bottom:0;">
              <div class="panel-h">Query <span style="color:var(--text-4); margin-left:4px;" id="queryStatus">· idle</span></div>
              <div class="panel-body" id="queryPanel"></div>
            </div>
          </div>

          <div class="rtab-panel" data-rtab-panel="nav">
            <div class="panel" style="flex:1; border-bottom:0;">
              <div class="panel-h">Nav <span style="color:var(--text-4); margin-left:4px;" id="navSubtitle">· PRM + robot</span></div>
              <div class="panel-body" id="navPanel"></div>
            </div>
          </div>

          <div class="rtab-panel" data-rtab-panel="render">
            <div class="panel" style="flex:1; border-bottom:0;">
              <div class="panel-h">Render</div>
              <div class="panel-body fixed" id="settingsPanel"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  // Tab wiring.
  const tabs = root.querySelectorAll('.rtab');
  const panels = root.querySelectorAll('.rtab-panel');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const key = tab.dataset.rtab;
      tabs.forEach((t) => t.classList.toggle('active', t === tab));
      panels.forEach((p) => p.classList.toggle('active', p.dataset.rtabPanel === key));
    });
  });

  return {
    viewport: root.querySelector('#viewport'),
    canvas: root.querySelector('#viewport-canvas'),
    layersPanel: root.querySelector('#layersPanel'),
    sceneGraphPanel: root.querySelector('#sceneGraphPanel'),
    sgSearch: root.querySelector('#sgSearch'),
    roomFilterPanel: root.querySelector('#roomFilterPanel'),
    roomFilterCount: root.querySelector('#roomFilterCount'),
    inspectorPanel: root.querySelector('#inspectorPanel'),
    inspectorSubtitle: root.querySelector('#inspectorSubtitle'),
    queryPanel: root.querySelector('#queryPanel'),
    queryStatus: root.querySelector('#queryStatus'),
    navPanel: root.querySelector('#navPanel'),
    navSubtitle: root.querySelector('#navSubtitle'),
    robotBar: root.querySelector('#robotBar'),
    settingsPanel: root.querySelector('#settingsPanel'),
    floorStrip: root.querySelector('#floorStrip'),
    vpTools: root.querySelector('#vpTools'),
    hud: root.querySelector('#hud'),
    hudTL: root.querySelector('#hudTL'),
    hudTR: root.querySelector('#hudTR'),
    hudBL: root.querySelector('#hudBL'),
    hudBR: root.querySelector('#hudBR'),
    labelLayer: root.querySelector('#labelLayer'),
    loader: root.querySelector('#loader'),
    loaderFill: root.querySelector('#loaderFill'),
    loaderStatus: root.querySelector('#loaderStatus'),
    statusbar: root.querySelector('#statusbar'),
    themeToggle: root.querySelector('#themeToggle'),
    frameAllBtn: root.querySelector('#frameAllBtn'),
    splatCountText: root.querySelector('#splatCountText'),
    crumbScene: root.querySelector('#crumbScene'),
    providerPill: root.querySelector('#providerPill'),
    sceneSelect: root.querySelector('#sceneSelect'),
    activateTab(name) {
      tabs.forEach((t) => t.classList.toggle('active', t.dataset.rtab === name));
      panels.forEach((p) => p.classList.toggle('active', p.dataset.rtabPanel === name));
    },
  };
}