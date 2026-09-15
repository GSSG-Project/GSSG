// Vertical floor selector. Designed for multi-floor; we only have floor 0
// today but the affordance is in place.

export function createFloorStrip(container, floors, activeFloorId, hooks) {
  let active = activeFloorId;

  function render() {
    container.innerHTML = floors.map((f) => `
      <div class="fbtn ${f === active ? 'active' : ''}" data-floor="${f}">
        <span class="fnum">${f}</span>
        <span>floor ${f === 0 ? 'G' : f}</span>
        <span class="floor-bar"></span>
      </div>
    `).join('') + `
      <div class="fbtn dim" title="Add floor (future)">
        <span class="fnum">+</span>
        <span style="opacity:.5">add floor</span>
        <span class="floor-bar"></span>
      </div>
    `;
    container.querySelectorAll('.fbtn[data-floor]').forEach((el) => {
      el.addEventListener('click', () => {
        active = Number(el.dataset.floor);
        render();
        hooks.onChange(active);
      });
    });
  }
  render();
}
