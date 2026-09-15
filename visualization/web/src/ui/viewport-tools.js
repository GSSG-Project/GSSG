// Floating viewport toolbar: frame-all, frame-selected, reset rotation.

export function createViewportTools(container, hooks) {
  container.innerHTML = `
    <div class="tbtn" data-tool="frameAll" title="Frame all">⊞</div>
    <div class="tbtn" data-tool="frameSel" title="Frame selected">◎</div>
    <div class="tdiv"></div>
    <div class="tbtn" data-tool="top" title="Top-down view">⤓</div>
    <div class="tbtn" data-tool="side" title="Side view">→</div>
  `;
  container.querySelectorAll('[data-tool]').forEach((el) => {
    el.addEventListener('click', () => hooks.onTool(el.dataset.tool));
  });
}
