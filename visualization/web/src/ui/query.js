// Query panel: CLIP top-k (simple) + LLM agent (streaming).
//
// Mode switching is local. Selecting a result row delegates to the host
// (`hooks.onSelect(id)`) to drive viewport selection; only the → button frames.

import { clipQuery, llmQueryStream } from '../data/api.js?v=3';
import { robot, goRobot } from '../data/robot.js?v=4';

export function createQuery(panel, statusEl, hooks) {
  let mode = 'agent'; // 'simple' | 'agent'
  let currentStream = null;

  function render() {
    panel.innerHTML = `
      <div class="qry">
        <div class="qry-mode">
          <button class="qmode-btn ${mode === 'agent' ? 'active' : ''}" data-mode="agent" title="LangGraph-style tool-use agent with the chosen LLM backend.">Agent</button>
          <button class="qmode-btn ${mode === 'simple' ? 'active' : ''}" data-mode="simple" title="Single CLIP text → FAISS top-k.">CLIP top-k</button>
        </div>

        <form class="qry-input" id="qryForm">
          <input id="qryText" placeholder="${placeholderFor(mode)}" autocomplete="off" spellcheck="false" />
          <button class="qry-go" type="submit">↵</button>
        </form>

        <div class="qry-hint" id="qryHint">${hintFor(mode)}</div>

        <div class="qry-trail" id="qryTrail"></div>
        <div class="qry-results" id="qryResults"></div>
      </div>
    `;

    panel.querySelectorAll('.qmode-btn').forEach((b) => {
      b.addEventListener('click', () => {
        if (mode === b.dataset.mode) return;
        mode = b.dataset.mode;
        abortStream();
        clearResults();
        clearTrail();
        render();
        focusInput();
      });
    });

    const form = panel.querySelector('#qryForm');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = panel.querySelector('#qryText').value.trim();
      if (!text) return;
      submit(text);
    });
  }

  function focusInput() {
    const inp = panel.querySelector('#qryText');
    if (inp) inp.focus();
  }

  function setStatus(s) {
    statusEl.textContent = s ? `· ${s}` : '· idle';
  }

  function clearResults() {
    const el = panel.querySelector('#qryResults');
    if (el) el.innerHTML = '';
  }
  function clearTrail() {
    const el = panel.querySelector('#qryTrail');
    if (el) el.innerHTML = '';
  }

  function abortStream() {
    if (currentStream) {
      currentStream.abort();
      currentStream = null;
    }
  }

  function submit(text) {
    abortStream();
    clearResults();
    clearTrail();
    if (mode === 'simple') runSimple(text);
    else runAgent(text);
  }

  async function runSimple(text) {
    setStatus('searching…');
    try {
      const data = await clipQuery(text, 10);
      const matches = dedupByCanonical(data.matches || []);
      renderResults(matches, { kind: 'simple', query: text });
      setStatus(`${matches.length} matches`);
      // Auto-highlight + frame the top-1 so the user sees the answer
      // immediately without an extra click.
      if (matches.length > 0 && hooks?.onSelect) {
        hooks.onSelect(matches[0].id, { frame: false });
      }
    } catch (e) {
      renderError(e.message);
      setStatus('error');
    }
  }

  // Collapse query results that point at the same visible (canonical) id,
  // keeping the highest-scored entry. Required because the backend searches
  // the full object set while the viewer dedups visually.
  function dedupByCanonical(matches) {
    const resolve = hooks?.resolveCanonical;
    if (!resolve) return matches;
    const best = new Map(); // canonical id → match
    for (const m of matches) {
      const c = resolve(m.id);
      const prev = best.get(c);
      if (!prev || (m.score ?? 0) > (prev.score ?? 0)) {
        best.set(c, { ...m, id: c });
      }
    }
    return [...best.values()].sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  function runAgent(text) {
    setStatus('streaming…');
    const trail = panel.querySelector('#qryTrail');
    appendTrail(trail, agentLine('user', text));
    let lastTextLine = null;
    let terminalEvt = null;
    let provider = '—';

    currentStream = llmQueryStream(text, (evt) => {
      switch (evt.type) {
        case 'meta':
          provider = `${evt.provider}/${evt.model || ''}`;
          appendTrail(trail, agentLine('meta', `agent · ${provider}`));
          break;
        case 'assistant_text':
          lastTextLine = appendTrail(trail, agentLine('assistant', evt.text));
          break;
        case 'tool_call':
          appendTrail(trail, agentLine('tool_call',
            `${evt.name}(${shortArgs(evt.args)})`, evt.id));
          break;
        case 'tool_result':
          appendTrail(trail, agentLine(
            evt.is_error ? 'tool_error' : 'tool_result',
            shortResult(evt.content),
            evt.id,
          ));
          break;
        case 'terminal':
          // Resolve hidden ids to their canonical visible representative,
          // and rewrite id mentions in the reason text. Downstream display
          // (trail line + result row + selection) all uses the normalized
          // event so the user only ever sees the visible id.
          terminalEvt = normalizeTerminal(evt);
          appendTrail(trail, agentLine('terminal', renderTerminal(terminalEvt)));
          break;
        case 'done':
          setStatus(`done (${evt.iterations} turns)`);
          if (terminalEvt) {
            renderTerminalResult(terminalEvt);
          }
          abortStream();
          break;
        case 'error':
          appendTrail(trail, agentLine('error', evt.message || JSON.stringify(evt)));
          setStatus('error');
          abortStream();
          break;
      }
      trail.scrollTop = trail.scrollHeight;
    });
  }

  // Map an absorbed-by-dedup id back to a visible canonical id and rewrite
  // any id mentions in the reason text so the trail / result row never show
  // a hidden id. `origId` is preserved on the event for tooltip context.
  function normalizeTerminal(evt) {
    const resolve = hooks?.resolveCanonical;
    if (!resolve) return evt;
    const origId = evt.object_id;
    const canonical = origId == null ? null : resolve(origId);
    const reason = rewriteIdsInText(evt.reason, resolve);
    return { ...evt, object_id: canonical, reason, _origId: origId };
  }

  function renderTerminal(evt) {
    if (evt.kind === 'finding') {
      const obj = evt.object_id == null ? 'no match' : `object #${evt.object_id}`;
      const note = evt._origId != null && evt._origId !== evt.object_id
        ? ` <small style="color:var(--text-4)">(redirected from #${evt._origId})</small>`
        : '';
      return `<b>finding:</b> ${obj}${note} — ${escapeHtml(evt.reason || '')}`;
    }
    if (evt.kind === 'navigate') {
      return `<b>navigate:</b> object #${evt.object_id}`;
    }
    return `<b>${evt.kind}</b>`;
  }

  function renderTerminalResult(evt) {
    const oid = evt.object_id;
    if (oid == null) return;
    renderResults([{ id: oid, reason: evt.reason, kind: evt.kind }], {
      kind: 'agent', query: '',
    });
    if (hooks?.onSelect) hooks.onSelect(oid, { frame: false });
  }

  function renderResults(matches, ctx) {
    const el = panel.querySelector('#qryResults');
    if (!el) return;
    if (!matches.length) {
      el.innerHTML = `<div class="qry-empty">No matches.</div>`;
      return;
    }
    const canDrive = robot.connected;
    const row = (m, cls = '') => `
      <div class="qry-row ${cls}" data-id="${m.id}">
        <div class="qr-id">#${m.id}</div>
        <div class="qr-mid">
          ${m.score != null ? `<div class="qr-score">score ${Number(m.score).toFixed(3)}</div>` : ''}
          ${m.reason ? `<div class="qr-reason">${escapeHtml(m.reason)}</div>` : ''}
          ${m.room_id != null ? `<div class="qr-room">room ${m.room_id}</div>` : ''}
        </div>
        <button class="qr-go" data-id="${m.id}" title="Frame in viewport">→</button>
        <button class="qr-drive" data-id="${m.id}" ${canDrive ? '' : 'disabled'}
          title="${canDrive ? 'Drive the robot to this object' : 'Connect the robot (top bar) to drive'}">GO</button>
      </div>`;
    // CLIP top-k: lead with the best match, fold the rest away.
    const [best, ...rest] = matches;
    const fold = ctx.kind === 'simple' && rest.length > 0;
    el.innerHTML = fold
      ? `${row(best, 'qry-best')}
         <details class="qry-alts">
           <summary>${rest.length} alternative${rest.length === 1 ? '' : 's'}</summary>
           ${rest.map((m) => row(m)).join('')}
         </details>`
      : matches.map((m) => row(m)).join('');

    el.querySelectorAll('.qry-row').forEach((row) => {
      row.addEventListener('click', () => {
        const id = Number(row.dataset.id);
        if (hooks?.onSelect) hooks.onSelect(id, { frame: false });
      });
    });
    el.querySelectorAll('.qr-drive').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = Number(btn.dataset.id);
        // Select first (sets the shared goal), then submit the live trajectory.
        if (hooks?.onSelect) hooks.onSelect(id, { frame: false });
        goRobot(false);
      });
    });
    el.querySelectorAll('.qr-go').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = Number(btn.dataset.id);
        if (hooks?.onSelect) hooks.onSelect(id, { frame: true });
      });
    });
  }

  function renderError(msg) {
    const el = panel.querySelector('#qryResults');
    if (el) el.innerHTML = `<div class="qry-err">${escapeHtml(msg)}</div>`;
  }

  render();
  setStatus('');

  return {
    setMode(m) { mode = m; render(); },
    abort: abortStream,
  };
}

// Rewrite "#NNN" / "object #NNN" / "object NNN" mentions in free text to
// the canonical visible id. Bare numbers (scores, distances) are left
// alone — only forms with the `#` or `object` prefix get touched.
function rewriteIdsInText(text, resolve) {
  if (!text || !resolve) return text;
  return text.replace(/(\bobject\s+#?|#)(\d+)\b/gi, (m, prefix, num) => {
    const id = Number(num);
    const c = resolve(id);
    return c === id ? m : `${prefix}${c}`;
  });
}

function placeholderFor(mode) {
  if (mode === 'simple') return 'photo of a sofa…';
  return 'Where is the kitchen sink?';
}
function hintFor(mode) {
  if (mode === 'simple') return 'Direct CLIP top-k. Fast, no LLM. Score is cosine similarity.';
  return 'LLM agent uses tools (search, spatial, distances). Trail shows every step.';
}

function agentLine(kind, html, id) {
  const div = document.createElement('div');
  div.className = `q-line q-${kind}`;
  if (id) div.dataset.callId = id;
  const labels = {
    user: 'YOU', meta: 'META', assistant: 'AGENT', tool_call: 'CALL',
    tool_result: 'RESULT', tool_error: 'ERROR', terminal: 'FINISH', error: 'FAIL',
  };
  const lab = labels[kind] || kind.toUpperCase();
  div.innerHTML = `<span class="q-tag q-tag-${kind}">${lab}</span><span class="q-body">${html}</span>`;
  return div;
}

function appendTrail(trail, node) {
  trail.appendChild(node);
  return node;
}

function shortArgs(args) {
  if (!args || typeof args !== 'object') return '';
  try {
    const parts = [];
    for (const [k, v] of Object.entries(args)) {
      let s = JSON.stringify(v);
      if (s && s.length > 60) s = s.slice(0, 57) + '…';
      parts.push(`${k}=${s}`);
    }
    return escapeHtml(parts.join(', '));
  } catch (_) { return ''; }
}

function shortResult(content) {
  let s = content;
  if (typeof s !== 'string') s = JSON.stringify(s);
  // Try to summarize JSON arrays of {id, score}
  try {
    const parsed = JSON.parse(s);
    if (Array.isArray(parsed)) {
      if (parsed.length === 0) return escapeHtml('[] (no result)');
      if (parsed.length <= 4) return escapeHtml(JSON.stringify(parsed));
      return escapeHtml(JSON.stringify(parsed.slice(0, 3)) + ` … (${parsed.length} total)`);
    }
    if (parsed && typeof parsed === 'object' && parsed.error) {
      return `<i>error:</i> ${escapeHtml(String(parsed.error))}`;
    }
  } catch (_) { /* not JSON */ }
  if (s.length > 240) s = s.slice(0, 237) + '…';
  return escapeHtml(s);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}