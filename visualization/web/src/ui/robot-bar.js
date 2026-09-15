// Top-bar robot widget: address + Connect/Disconnect, live state pill, and the
// EMERGENCY STOP button (visible whenever a robot is connected). This is the one
// place the robot connection lives — every panel reads the same store.

import {
  robot, onRobot, connectRobot, disconnectRobot, setAddress, estopRobot,
} from '../data/robot.js?v=4';

export function createRobotBar(container) {
  container.innerHTML = `
    <input id="rbAddr" class="rb-addr" placeholder="robot ip[:8100]"
           value="${robot.address}" spellcheck="false" title="robot nav service address" />
    <button class="btn" id="rbConnect">Connect robot</button>
    <span class="pill rb-pill" id="rbPill" style="display:none">
      <span class="led" id="rbLed"></span><span id="rbText"></span>
    </span>
    <button class="rb-estop" id="rbEstop" style="display:none"
            title="POST /api/estop — immediate brake + abort">EMERGENCY STOP</button>
  `;

  const addr = container.querySelector('#rbAddr');
  const connectBtn = container.querySelector('#rbConnect');
  const pill = container.querySelector('#rbPill');
  const led = container.querySelector('#rbLed');
  const text = container.querySelector('#rbText');
  const estopBtn = container.querySelector('#rbEstop');

  addr.addEventListener('change', () => setAddress(addr.value));
  addr.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { setAddress(addr.value); connectBtn.click(); }
  });

  connectBtn.addEventListener('click', async () => {
    if (robot.connected) {
      await disconnectRobot();
    } else {
      setAddress(addr.value);
      try {
        await connectRobot();
      } catch (e) {
        text.textContent = e.message;  // address parse errors
        pill.style.display = '';
        led.style.background = 'var(--danger)';
      }
    }
  });

  estopBtn.addEventListener('click', () => estopRobot());

  onRobot((r) => {
    addr.disabled = r.connected || r.connecting;
    connectBtn.textContent = r.connecting
      ? 'connecting…'
      : r.connected ? 'Disconnect' : 'Connect robot';
    connectBtn.classList.toggle('rb-on', r.connected);
    estopBtn.style.display = r.connected ? '' : 'none';

    if (!r.connected) {
      pill.style.display = 'none';
      return;
    }
    pill.style.display = '';
    if (!r.reachable) {
      led.style.background = 'var(--danger)';
      led.style.boxShadow = '0 0 6px var(--danger)';
      text.textContent = 'unreachable';
      return;
    }
    const st = r.status || {};
    const spatial = st.tracking?.spatial ?? '—';
    const drivable = spatial === 'KNOWN_MAP' || spatial === 'LOOP_CLOSED';
    const robotOk = st.robot?.connected === true;
    const color = robotOk && drivable ? 'var(--ok)' : 'var(--warn)';
    led.style.background = color;
    led.style.boxShadow = `0 0 6px ${color}`;
    text.textContent = `${st.state ?? '—'} · ${spatial}${robotOk ? '' : ' · no robot'}`;
  });
}
