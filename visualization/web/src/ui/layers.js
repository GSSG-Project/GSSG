// Layer toggle cards: splats / boxes / labels / grid.

const LAYER_DEFS = [
  { key: 'splats', name: 'Gaussian splats', tag: 'METRIC', color: '#c4f542', initial: true },
  { key: 'boxes',  name: 'Bounding boxes',  tag: 'OBJECT',  color: '#6ec3ff', initial: true },
  { key: 'labels', name: 'Object labels',   tag: 'TEXT',    color: '#ff7ad5', initial: true },
  { key: 'grid',   name: 'Reference grid',  tag: 'GRID',    color: '#7a7e85', initial: true },
];

export function createLayers(container, onChange) {
  const state = {};
  for (const def of LAYER_DEFS) state[def.key] = def.initial;

  function render() {
    container.innerHTML = `<div class="layer-stack">${LAYER_DEFS.map((def) => `
      <div class="layer-card ${state[def.key] ? 'active' : ''}" data-key="${def.key}">
        <div class="row">
          <span class="swatch" style="background:${def.color}">${def.tag[0]}</span>
          <span class="name">${def.name}<small>${def.tag}</small></span>
          <span class="meta" data-meta="${def.key}"></span>
          <span class="vis ${state[def.key] ? '' : 'off'}">${state[def.key] ? '●' : '○'}</span>
        </div>
      </div>
    `).join('')}</div>`;

    for (const def of LAYER_DEFS) {
      const card = container.querySelector(`[data-key="${def.key}"]`);
      card.addEventListener('click', () => {
        state[def.key] = !state[def.key];
        render();
        onChange(def.key, state[def.key], { ...state });
      });
    }
  }

  render();

  return {
    state,
    setMeta(key, text) {
      const meta = container.querySelector(`[data-meta="${key}"]`);
      if (meta) meta.textContent = text;
    },
  };
}
