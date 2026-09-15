// Scrollable object list grouped by room. Click → select.

export function createSceneGraph(panel, searchInput, objects, hooks) {
  let query = '';
  let selectedId = null;

  function matches(o) {
    if (!query) return true;
    const q = query.toLowerCase();
    if (String(o.id).includes(q)) return true;
    if (o.roomId != null && (`room ${o.roomId}`.includes(q) || `r${o.roomId}`.includes(q))) return true;
    if (String(o.vectorId).includes(q)) return true;
    return false;
  }

  function render() {
    const groups = new Map();
    for (const o of objects) {
      if (!matches(o)) continue;
      const k = o.roomId == null ? 'none' : o.roomId;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(o);
    }
    const sortedKeys = [...groups.keys()].sort((a, b) => {
      if (a === 'none') return 1;
      if (b === 'none') return -1;
      return a - b;
    });

    const parts = [];
    for (const k of sortedKeys) {
      const list = groups.get(k);
      const roomLabel = k === 'none' ? 'unassigned' : `room ${k}`;
      parts.push(`<div class="tree-group">
        <div class="tree-group-h">
          <span class="caret">▾</span>
          <span class="icon">▢</span>
          <span class="label">${roomLabel}</span>
          <span class="id">${list.length}</span>
        </div>
        <div class="children">
          ${list.map((o) => `
            <div class="node ${selectedId === o.id ? 'selected' : ''}" data-id="${o.id}">
              <span class="caret"></span>
              <span class="icon" style="color:${roomColor(o.roomId)}">◧</span>
              <span class="label">object #${o.id}</span>
              <span class="id">v${o.vectorId}</span>
            </div>
          `).join('')}
        </div>
      </div>`);
    }
    panel.innerHTML = parts.join('') || `<div class="tree-empty">no matches</div>`;

    panel.querySelectorAll('.node').forEach((node) => {
      const id = Number(node.dataset.id);
      node.addEventListener('click', () => hooks.onSelect(id));
      node.addEventListener('mouseenter', () => hooks.onHover(id));
      node.addEventListener('mouseleave', () => hooks.onHover(null));
    });
  }

  function roomColor(r) {
    if (r === 1) return 'var(--info)';
    if (r === 2) return 'var(--magenta)';
    return 'var(--text-4)';
  }

  searchInput.addEventListener('input', (e) => {
    query = e.target.value.trim();
    render();
  });

  render();

  return {
    setSelected(id) {
      selectedId = id;
      render();
    },
  };
}
