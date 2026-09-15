// Room chips — toggleable. When all are off (or none selected), means "all".

export function createRoomFilter(panel, countLabel, objects, hooks) {
  const counts = new Map();
  for (const o of objects) {
    const k = o.roomId == null ? 'none' : o.roomId;
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const keys = [...counts.keys()].sort((a, b) => {
    if (a === 'none') return 1;
    if (b === 'none') return -1;
    return a - b;
  });

  const active = new Set();

  function chipColor(k) {
    if (k === 1) return 'var(--info)';
    if (k === 2) return 'var(--magenta)';
    return 'var(--text-3)';
  }

  function render() {
    panel.innerHTML = `<div class="chips">${keys.map((k) => `
      <span class="chip ${active.has(k) ? 'active' : ''}" data-key="${k}">
        <span class="ch-sw" style="background:${chipColor(k)}"></span>
        ${k === 'none' ? 'unassigned' : `room ${k}`}
        <span class="ch-n">${counts.get(k)}</span>
      </span>
    `).join('')}</div>`;

    panel.querySelectorAll('.chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const key = chip.dataset.key === 'none' ? 'none' : Number(chip.dataset.key);
        if (active.has(key)) active.delete(key);
        else active.add(key);
        render();
        const filter = active.size === 0 ? null : new Set(active);
        countLabel.textContent = active.size === 0 ? '· all' : `· ${active.size}`;
        hooks.onChange(filter);
      });
    });
  }

  render();
}
