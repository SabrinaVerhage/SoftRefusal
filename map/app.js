(function () {
  'use strict';
  const $ = id => document.getElementById(id), M = CloudMath, { TAU, clamp } = M;
  const canvas = $('cloud'), ctx = canvas.getContext('2d', { alpha: false });
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // The public map shows threads and reasoning; ?tools brings back the research controls.
  const tools = new URLSearchParams(location.search).has('tools');
  document.body.classList.toggle('tools', tools);
  const concept = M.makeConcept();
  let data = concept, selected, hovered, neighbors = [], projected = [], drawnPoints = [], segments = [];
  let colorMode = 'topic', autoOrbit = false, showNeighbors = false, tour = null, transition = null, morph = null;
  let width = 0, height = 0, dpr = 1, dirty = true, messageTimer, lastTime = 0, view = '3d';
  let viewport = { cx: 0, cy: 0, scale: 100 };
  const camera = { yaw: -0.18, pitch: 0.38, zoom: 1, target: { x: 0, y: 0, z: 0 }, pan: { x: 0, y: 0 } };
  const filters = new Set(['answered', 'redirected', 'refused', 'unlabeled']);
  const pointers = new Map();
  let drag = null, pinch = null, hoveredThread = null;
  let replay = { path: [], index: 0, playing: false, reveal: false, next: 0 };
  const colors = { answered: [234, 110, 165], redirected: [231, 151, 101], refused: [157, 119, 198], unclear: [142, 130, 125], unlabeled: [142, 130, 125] };
  const topicColors = { consent: '#f3cfac', intimacy: '#efadbe', desire: '#eeadd1', 'erotic fiction': '#c9a7df', relationships: '#edb5ba', 'sexual health': '#f1c4a6', unclassified: '#d7ccc0' };
  // One brand pigment per recorded conversation (pleasure, curiosity, energy, peach, then extras).
  const conversationColors = [[255, 45, 142], [138, 77, 255], [255, 174, 0], [255, 154, 107], [61, 154, 139], [196, 64, 92], [92, 112, 214], [150, 120, 60]];
  const recorded = () => data.kind === 'recordings' || data.kind === 'semantic';
  // The donut field (wash, contours, core) frames both the reference field and the semantic donut.
  const shaped = () => data.kind === 'concept' || data.kind === 'semantic';
  const labelNodes = [];
  const labelNames = ['consent', 'intimacy', 'desire', 'erotic fiction', 'relationships', 'sexual health'];

  function message(text, duration = 6000) {
    clearTimeout(messageTimer); $('message').textContent = text; $('message').hidden = false;
    messageTimer = setTimeout(() => { $('message').hidden = true; }, duration);
  }
  function rgba(rgb, alpha) { return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`; }
  function spectrum(p) {
    const t = clamp((p.x + 2.8) / 5.6, 0, 1), a = t < 0.5 ? [245, 169, 127] : [238, 122, 170], b = t < 0.5 ? [238, 122, 170] : [165, 119, 207], u = t < 0.5 ? t * 2 : t * 2 - 1;
    return a.map((x, i) => Math.round(x + (b[i] - x) * u));
  }
  function pigment(p) {
    if (recorded() && colorMode === 'topic') return conversationColors[M.conversationColorIndex(p.conversation_id, conversationColors.length)];
    if (colorMode === 'response' || p.outcome === 'unlabeled') return colors[p.outcome];
    return spectrum(p);
  }
  function recolor() { for (const p of data.points) p.color = pigment(p); dirty = true; }
  function resize() {
    const box = canvas.getBoundingClientRect(); width = box.width; height = box.height;
    dpr = Math.min(devicePixelRatio || 1, 2); canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    const mobile = width <= 760, usableWidth = mobile ? width - 22 : width - (width < 1150 ? 380 : 470);
    const mobileY = tools ? (height < 700 ? 0.47 : 0.4) : 0.57;
    viewport = { cx: mobile ? width * 0.5 : width * 0.51, cy: mobile ? height * mobileY : height * 0.445, scale: Math.min(usableWidth / 5.9, height * (mobile ? (tools ? 0.081 : 0.07) : 0.127)) };
    dirty = true;
  }
  function project(p) { return M.project(p, camera, viewport); }
  function circle(x, y, r) { ctx.beginPath(); ctx.arc(x, y, Math.max(r, 0.1), 0, TAU); }
  function curve(points, close = false) { ctx.beginPath(); points.forEach((p, i) => { const q = project(p); if (!i) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y); }); if (close) ctx.closePath(); }
  function curvePoints(phi, steps = 180) { return Array.from({ length: steps }, (_, i) => M.surface(i / steps * TAU, phi)); }
  const outer = curvePoints(0), inner = curvePoints(-Math.PI);
  const contourLines = Array.from({ length: 22 }, (_, j) => Array.from({ length: 190 }, (_, i) => { const theta = i / 190 * TAU, phi = -Math.PI + (j + 0.5) / 22 * Math.PI; return M.surface(theta, phi + 0.32 * Math.sin(theta * 3 + phi * 2) * Math.sin(-phi)); }));
  function drawGrid() {
    ctx.fillStyle = '#f2f0ec'; ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = '#b9aaa0'; ctx.globalAlpha = 0.19;
    for (let x = 18; x < width; x += 23) for (let y = 15; y < height; y += 23) ctx.fillRect(x, y, 0.65, 0.65);
    ctx.globalAlpha = 1;
  }
  function drawWash() {
    if (!shaped() || !$('contours').checked) return;
    ctx.save();
    curve(outer, true); ctx.strokeStyle = '#fffdfa'; ctx.lineWidth = 21; ctx.shadowColor = '#3a17271a'; ctx.shadowBlur = 25; ctx.shadowOffsetY = 8; ctx.stroke();
    ctx.shadowBlur = 0; ctx.shadowOffsetY = 0;
    const boundary = new Path2D();
    for (const ring of [outer, inner]) { ring.forEach((p, i) => { const q = project(p); if (!i) boundary.moveTo(q.x, q.y); else boundary.lineTo(q.x, q.y); }); boundary.closePath(); }
    ctx.clip(boundary, 'evenodd');
    const l = project({ x: -2.7, y: 0, z: 0 }), r = project({ x: 2.7, y: 0, z: 0 });
    const gradient = ctx.createLinearGradient(l.x, l.y, r.x, r.y); gradient.addColorStop(0, '#f8b38a'); gradient.addColorStop(0.42, '#f398b8'); gradient.addColorStop(0.73, '#eaa0d3'); gradient.addColorStop(1, '#c6a0e2');
    ctx.globalAlpha = 0.78; ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height); ctx.globalAlpha = 1;
    ctx.restore();
    curve(outer, true); ctx.strokeStyle = '#171216'; ctx.lineWidth = 1.7; ctx.stroke();
    curve(inner, true); ctx.strokeStyle = '#291c2780'; ctx.lineWidth = 0.7; ctx.stroke();
  }
  function drawTexture() {
    if (!data.texture || !$('texture').checked) return;
    // Illustrative dust from the reference field: never selectable, never searched.
    for (const p of data.texture) {
      const q = project(p); if (q.x < -20 || q.x > width + 20 || q.y < -20 || q.y > height + 20) continue;
      circle(q.x, q.y, clamp(p.radius * q.per * Math.pow(camera.zoom, 0.4), 0.4, 3)); ctx.fillStyle = rgba(p.textureColor, clamp(0.55 - q.depth * 0.07, 0.18, 0.7) * (selected && recorded() ? 0.3 : 1)); ctx.fill();
    }
  }
  function drawCloud() {
    drawTexture();
    projected = data.points.map(p => ({ ...project(p), p })).filter(q => filters.has(q.p.outcome) && q.x > -20 && q.x < width + 20 && q.y > -20 && q.y < height + 20);
    projected.sort((a, b) => b.depth - a.depth);
    // Stay responsive with imported large clouds; the full data remains searchable.
    const stride = Math.max(1, Math.ceil(projected.length / 25000)); drawnPoints = [];
    for (let i = 0; i < projected.length; i += stride) {
      const q = projected[i], p = q.p;
      if (p.path && $('trails').checked) continue;
      drawnPoints.push(q);
      const alpha = clamp(0.6 - q.depth * 0.085, 0.2, 0.86) * pathOpacity(p), rad = clamp((p.radius || 1) * q.per * Math.pow(camera.zoom, 0.4), 0.4, 5);
      if (p.outcome === 'refused') {
        circle(q.x, q.y, rad * 1.15); ctx.strokeStyle = rgba(colorMode === 'response' ? [35, 23, 38] : p.color, alpha); ctx.lineWidth = 0.7; ctx.stroke();
      } else if (p.outcome === 'redirected') {
        circle(q.x, q.y, rad); ctx.strokeStyle = rgba(p.color, alpha); ctx.lineWidth = 0.7; ctx.stroke();
      } else {
        circle(q.x, q.y, rad); ctx.fillStyle = rgba(p.color, alpha); ctx.fill();
      }
    }
  }
  function drawContours() {
    if (!shaped() || !$('contours').checked) return;
    ctx.save(); ctx.strokeStyle = '#20131d'; ctx.lineWidth = 0.8; ctx.globalAlpha = 0.64;
    for (let j = 0; j < contourLines.length; j++) { curve(contourLines[j], true); ctx.stroke(); }
    ctx.restore();
  }
  function smoothPath(path) {
    const qs = path.map(project); if (qs.length < 2) return;
    ctx.beginPath(); ctx.moveTo(qs[0].x, qs[0].y);
    for (let i = 1; i < qs.length - 1; i++) ctx.quadraticCurveTo(qs[i].x, qs[i].y, (qs[i].x + qs[i + 1].x) / 2, (qs[i].y + qs[i + 1].y) / 2);
    ctx.lineTo(qs.at(-1).x, qs.at(-1).y);
  }
  function drawPathNode(p, endpoint) {
    if (!filters.has(p.outcome)) return;
    const q = project(p), r = Math.min(6, 4.1 * q.per * Math.sqrt(camera.zoom)); drawnPoints.push({ ...q, p });
    if (recorded()) {
      circle(q.x, q.y, r + 2.2); ctx.fillStyle = '#fdfaf5'; ctx.fill(); circle(q.x, q.y, r);
      if (p.outcome === 'refused' || p.outcome === 'redirected') { ctx.strokeStyle = rgba(p.color, 1); ctx.lineWidth = p.outcome === 'refused' ? 2.4 : 1.6; ctx.stroke(); }
      else { ctx.fillStyle = rgba(p.color, 1); ctx.fill(); }
      ctx.font = "600 10px 'IBM Plex Mono', Consolas, monospace"; ctx.fillStyle = '#161514'; ctx.fillText(String(p.turn_id + 1), q.x + r + 4, q.y - r - 2);
    } else if (endpoint && data.kind === 'concept') {
      ctx.strokeStyle = '#f9f5ee'; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(q.x - 4, q.y - 4); ctx.lineTo(q.x + 4, q.y + 4); ctx.moveTo(q.x - 4, q.y + 4); ctx.lineTo(q.x + 4, q.y - 4); ctx.stroke(); ctx.strokeStyle = '#151215'; ctx.lineWidth = 3; ctx.stroke();
    } else {
      circle(q.x, q.y, r + 1.9); ctx.fillStyle = '#fdfaf5'; ctx.fill(); circle(q.x, q.y, r); ctx.fillStyle = '#171315'; ctx.fill();
      if (p.turn_id === 1) { circle(q.x, q.y, r + 4); ctx.strokeStyle = '#171315'; ctx.lineWidth = 1.2; ctx.stroke(); }
    }
  }
  function drawTrails() {
    if (!$('trails').checked) return;
    if (recorded()) { drawRecordedTrails(); return; }
    ctx.save(); ctx.strokeStyle = '#21171e'; ctx.lineWidth = data.kind === 'concept' ? 1.7 : 1.1; ctx.setLineDash([5, 5]);
    if (shaped()) {
      for (const path of data.paths) {
        if (path.every(p => filters.has(p.outcome))) { smoothPath(path); ctx.stroke(); }
        else for (let i = 1; i < path.length; i++) if (filters.has(path[i - 1].outcome) && filters.has(path[i].outcome)) { curve([path[i - 1], path[i]]); ctx.stroke(); }
      }
      ctx.setLineDash([]); for (const path of data.paths) path.forEach((p, i) => drawPathNode(p, i === path.length - 1));
    } else {
      ctx.globalAlpha = recorded() ? 0.85 : 0.45; if (recorded()) ctx.lineWidth = 1.5;
      for (const segment of segments) { if (segment.every(p => filters.has(p.outcome))) { if (recorded()) ctx.strokeStyle = rgba(segment[0].color, 1); curve(segment); ctx.stroke(); } }
      if (recorded()) {
        // Turn numbers show the direction each conversation travels.
        ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.font = "600 10px 'IBM Plex Mono', Consolas, monospace"; ctx.fillStyle = '#161514';
        for (const p of data.points) if (filters.has(p.outcome)) { const q = project(p); ctx.fillText(String(p.turn_id + 1), q.x + 7, q.y - 6); }
      }
    }
    ctx.restore();
  }
  function pathOpacity(p) {
    if (!recorded() || !selected) return 1;
    if (!tools) return p.conversation_id === selected.conversation_id ? 1 : 0.22;
    if ($('all-conversations').checked) return 1;
    if (p.conversation_id !== selected.conversation_id) return 0.16;
    return replay.reveal && p.turn_id > selected.turn_id ? 0.18 : 1;
  }
  function drawRecordedTrails() {
    ctx.save();
    // Straight links pass through every measured checkpoint; no invented curved detours.
    for (const [a, b] of segments) {
      if (!filters.has(a.outcome) || !filters.has(b.outcome)) continue;
      const active = a.conversation_id === selected?.conversation_id;
      ctx.globalAlpha = Math.min(pathOpacity(a), pathOpacity(b));
      if (!tools) {
        // Every conversation reads as one continuous thread: a paper halo lifts it off the contours.
        const lineWidth = active ? 3.6 : a.conversation_id === hoveredThread ? 3.2 : 2.4;
        ctx.setLineDash([]); ctx.lineCap = 'round'; curve([a, b]);
        ctx.strokeStyle = 'rgba(253,250,245,0.9)'; ctx.lineWidth = lineWidth + 3.5; ctx.stroke();
        ctx.strokeStyle = rgba(a.color, 1); ctx.lineWidth = lineWidth; ctx.stroke();
      } else {
        ctx.strokeStyle = rgba(a.color, 1); ctx.lineWidth = active ? 2.3 : $('all-conversations').checked ? 1.8 : 1;
        ctx.setLineDash(active || $('all-conversations').checked ? [] : [3, 5]); curve([a, b]); ctx.stroke();
      }
      const p = project(a), q = project(b), length = Math.hypot(q.x - p.x, q.y - p.y);
      if (active && length > 24) {
        const angle = Math.atan2(q.y - p.y, q.x - p.x), x = p.x + (q.x - p.x) * 0.64, y = p.y + (q.y - p.y) * 0.64;
        ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(x - 6 * Math.cos(angle - 0.5), y - 6 * Math.sin(angle - 0.5)); ctx.lineTo(x, y); ctx.lineTo(x - 6 * Math.cos(angle + 0.5), y - 6 * Math.sin(angle + 0.5)); ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    for (const p of data.points) {
      ctx.globalAlpha = pathOpacity(p);
      drawPathNode(p, false);
    }
    ctx.restore();
  }
  function drawCore() {
    const label = $('core-label'); label.hidden = !shaped() || !$('core').checked;
    if (label.hidden) return;
    const q = project({ x: 0, y: 0, z: 0 }), r = viewport.scale * camera.zoom * 0.87 * q.per;
    ctx.save(); circle(q.x, q.y, r);
    const shade = ctx.createRadialGradient(q.x - r * 0.3, q.y - r * 0.35, r * 0.08, q.x, q.y, r);
    shade.addColorStop(0, '#fffcf7'); shade.addColorStop(0.6, '#eee1f0'); shade.addColorStop(1, '#bfa2ce');
    ctx.fillStyle = shade; ctx.fill(); ctx.strokeStyle = '#38263e'; ctx.lineWidth = 1.3; ctx.stroke();
    ctx.beginPath(); ctx.ellipse(q.x, q.y, r * 0.98, r * 0.29, -0.2, 0, TAU); ctx.strokeStyle = '#72527d45'; ctx.lineWidth = 0.7; ctx.setLineDash([3, 5]); ctx.stroke(); ctx.restore();
    label.style.left = `${q.x}px`; label.style.top = `${q.y}px`; label.style.transform = `translate(-50%,-50%) scale(${clamp(r / 92, 0.4, 2.1)})`;
    // On the donut, recorded turns live inside the core: keep its lettering quiet.
    label.style.opacity = data.kind === 'semantic' ? 0.32 : q.x < 230 && width > 760 || q.x > width - 290 && width > 760 ? 0.4 : 1;
  }
  function drawSelection() {
    if (!selected || !filters.has(selected.outcome)) return;
    const q = project(selected);
    if (showNeighbors) {
      ctx.save(); ctx.setLineDash([2, 5]); ctx.strokeStyle = '#48313c'; ctx.lineWidth = 0.9;
      for (const p of neighbors) {
        if (!filters.has(p.outcome)) continue;
        const n = project(p); ctx.beginPath(); ctx.moveTo(q.x, q.y); ctx.lineTo(n.x, n.y); ctx.stroke(); circle(n.x, n.y, 4 * n.per); ctx.fillStyle = '#fffaf5'; ctx.fill(); ctx.stroke();
      }
      ctx.restore();
    }
    circle(q.x, q.y, 7); ctx.fillStyle = '#fdf9f4'; ctx.fill(); ctx.strokeStyle = '#161215'; ctx.lineWidth = 1.4; ctx.stroke();
    circle(q.x, q.y, 3.6); ctx.fillStyle = '#161215'; ctx.fill();
    ctx.save(); ctx.setLineDash([2, 4]); circle(q.x, q.y, 14); ctx.lineWidth = 0.6; ctx.strokeStyle = '#161215aa'; ctx.stroke(); ctx.restore();
    if (hovered && hovered !== selected) { const h = project(hovered); circle(h.x, h.y, 6); ctx.strokeStyle = '#161215'; ctx.lineWidth = 1; ctx.stroke(); }
  }
  function updateLabels() {
    for (const label of labelNodes) {
      const q = project(label.point), hide = !filters.has(label.point.outcome) || camera.zoom > 2.2 || q.x < (width > 760 ? 240 : 18) || q.x > width - (width > 760 ? 310 : 18) || q.y < 145 || q.y > height - (width > 760 ? 170 : 360);
      label.el.hidden = hide; label.el.style.left = `${q.x}px`; label.el.style.top = `${q.y - 10}px`; label.el.style.opacity = clamp(0.95 - q.depth * 0.1, 0.45, 1);
    }
    $('zoom-value').textContent = `${Math.round(camera.zoom * 100)}%`;
  }
  function render() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); drawGrid(); drawWash(); drawCloud(); drawContours(); drawCore(); drawTrails(); drawSelection(); updateLabels(); dirty = false;
  }
  function selectPoint(p, show = true) {
    selected = p; neighbors = M.nearest(data.points, p); $('inspector').hidden = !show; $('inspector').classList.toggle('mobile-open', show); $('inspector-mobile').setAttribute('aria-expanded', String(show && width <= 760));
    const turns = recorded() ? data.points.filter(q => q.conversation_id === p.conversation_id).length : 0;
    $('point-number').textContent = recorded() ? `TURN ${p.turn_id + 1} / ${turns}` : `PROMPT / ${p.point_id.replace(/^field-|^sample-/, '').toUpperCase()}`;
    $('point-topic').textContent = p.topic.toUpperCase(); $('point-topic').style.background = recorded() ? rgba(p.color, 0.3) : topicColors[p.topic] || '#dfd1dc';
    $('inspector').classList.toggle('recorded', recorded()); $('response-block').hidden = !recorded();
    $('point-response').textContent = recorded() ? splitThinking(p.response).answer || '(The model replied with no text.)' : ''; $('point-response-note').textContent = recorded() && p.hit_token_limit ? 'REPLY REACHED ITS LENGTH LIMIT' : '';
    $('response-heading').textContent = data.synthetic ? 'ILLUSTRATIVE RESPONSE' : 'MODEL REPLIED';
    $('point-prompt').textContent = p.prompt; $('point-outcome').textContent = p.outcome[0].toUpperCase() + p.outcome.slice(1) + (p.label_source === 'model_judge' ? ' · Auto-tagged' : p.label_source === 'human' ? ' · Human-labelled' : ''); $('status-dot').className = `dot ${p.outcome}`;
    $('point-outcome').title = p.label_reason || '';
    $('point-demo').textContent = recorded() ? data.synthetic ? 'ILLUSTRATIVE' : 'RECORDED' : data.kind === 'imported' ? p.synthetic === true || data.synthetic === true ? 'SYNTHETIC' : 'IMPORTED' : 'DEMO';
    const original = p.original_coordinates || [p.x, p.y, p.z];
    $('point-coordinates').textContent = data.kind === 'semantic' ? `CORE ${p.core_score.toFixed(2)} / ${Math.round((p.angle * 180 / Math.PI + 360) % 360)}°` : original.map(x => x.toFixed(2)).join(' / ');
    $('inspector-note').textContent = data.kind === 'semantic' ? `Reference score ${p.core_score.toFixed(2)} at layer ${data.layer}. 0 = answered reference mean; 1 = refused reference mean. ${data.semantic.frozen ? 'Fixed reference ' + data.semantic.reference_id.slice(0, 8) + '. Labels change markers, not position.' : 'Legacy layout fitted to the displayed labels.'} Radius is an artistic compression, not a probability. Links join pre-reply checkpoints.` : recorded() ? `The model’s internal state just before this reply, at layer ${data.layer}, placed by UMAP among your other turns. UMAP is refitted on export; this view is exploratory.` : data.kind === 'concept' ? 'An illustrative prompt, with an assigned response. Click any point to look closer.' : data.kind === 'umap' ? 'Synthetic data. Responses are assigned demo labels; neighbors use the 3D projection.' : 'Metadata from your file. Neighbors use the displayed 3D coordinates.';
    if (data.kind === 'recordings' && data.settings?.method === 'pca_preview') $('inspector-note').textContent = `The model’s internal state just before this reply, at layer ${data.layer}. Early PCA preview, not UMAP; waiting for six distinct turns. This view is exploratory.`;
    if (recorded() && data.synthetic) {
      $('point-coordinates').textContent = 'ILLUSTRATIVE POSITION';
      $('inspector-note').textContent = 'Art-directed example of conversation replay. Paths, responses and outcomes are invented to demonstrate the interface. No model was queried; these are not measured trajectories.';
    }
    if (recorded() && !data.synthetic && p.label_source === 'model_judge') $('inspector-note').textContent += ` Auto-tagged by Gemma; this judgment may be wrong.${p.label_reason ? ' ' + p.label_reason : ''}`;
    if (!tools) renderThread(p);
    if (recorded()) {
      if (replay.path[0]?.conversation_id !== p.conversation_id) {
        replay = { path: M.conversationPath(data.points, p.conversation_id), index: 0, playing: false, reveal: false, next: 0 };
      }
      replay.index = replay.path.indexOf(p); updateReplay();
    }
    dirty = true;
  }
  // Thinking models write <think>…</think> before their answer; the recording keeps the raw text.
  function splitThinking(text) {
    text = text || '';
    const start = text.indexOf('<think>'), end = text.indexOf('</think>');
    if (start < 0 && end < 0) return { thinking: null, answer: text };
    const from = start < 0 ? 0 : start + 7;
    return { thinking: text.slice(from, end < 0 ? undefined : end).trim(), answer: end < 0 ? '' : text.slice(end + 8).trim() };
  }
  function nearestThread(x, y) {
    let best = (width <= 760 ? 16 : 9) ** 2, id = null;
    for (const [a, b] of segments) {
      if (!filters.has(a.outcome) || !filters.has(b.outcome)) continue;
      const p = project(a), q = project(b), dx = q.x - p.x, dy = q.y - p.y, t = clamp(((x - p.x) * dx + (y - p.y) * dy) / (dx * dx + dy * dy || 1), 0, 1);
      const distance = (p.x + t * dx - x) ** 2 + (p.y + t * dy - y) ** 2;
      if (distance < best) { best = distance; id = a.conversation_id; }
    }
    return id;
  }
  // The moment shown first: the refusal closest to the core, else the turn closest to it.
  function threadMoment(id) {
    const path = M.conversationPath(data.points, id), score = p => p.core_score ?? p.turn_id;
    const pick = list => list.reduce((a, b) => score(b) > score(a) ? b : a);
    const refused = path.filter(p => p.outcome === 'refused'), redirected = path.filter(p => p.outcome === 'redirected');
    return pick(refused.length ? refused : redirected.length ? redirected : path);
  }
  function clearThread() {
    selected = null; $('thread').hidden = true; dirty = true;
  }
  function renderThread(p) {
    const card = $('thread'), path = M.conversationPath(data.points, p.conversation_id), moment = threadMoment(p.conversation_id);
    card.hidden = !recorded();
    if (card.hidden) return;
    $('legend').classList.remove('mobile-open'); $('legend-mobile').setAttribute('aria-expanded', 'false');
    $('thread-swatch').style.background = rgba(p.color, 1);
    $('thread-number').textContent = `CONVERSATION ${String(p.conversation_number).padStart(2, '0')} · ${path.length} ${path.length === 1 ? 'MESSAGE' : 'MESSAGES'}`;
    $('thread-demo').textContent = data.synthetic ? 'ILLUSTRATIVE' : '';
    $('thread-steps').replaceChildren(...path.map(q => {
      const item = document.createElement('li'), button = document.createElement('button'), dot = document.createElement('i');
      dot.className = `dot ${q.outcome}`; button.append(dot, document.createTextNode(String(q.turn_id + 1)));
      button.className = (q === p ? 'active' : '') + (q === moment ? ' moment-step' : ''); button.setAttribute('aria-pressed', String(q === p));
      button.setAttribute('aria-label', `Message ${q.turn_id + 1}: ${q.outcome}${q === moment ? ', closest to the core' : ''}`);
      button.onclick = () => selectPoint(q); item.append(button); return item;
    }));
    $('moment-heading').textContent = p === moment && p.outcome === 'refused' ? `REFUSAL MOMENT · MESSAGE ${p.turn_id + 1}` : p === moment ? `CLOSEST TO THE CORE · MESSAGE ${p.turn_id + 1}` : `MESSAGE ${p.turn_id + 1}`;
    $('moment-dot').className = `dot ${p.outcome}`;
    $('moment-outcome').textContent = (p.outcome === 'unlabeled' ? 'Not labelled yet' : `AI ${p.outcome}`) + (p.label_source === 'model_judge' ? ' · auto-tagged' : '');
    const thinking = splitThinking(p.response).thinking;
    $('moment-reasoning').textContent = thinking || 'No reasoning was recorded for this message. Conversations from before the AI wrote down its thinking only show where they went on the map.';
    $('moment-reasoning').classList.toggle('empty', !thinking);
    $('thread-note').hidden = !thinking;
  }
  function makeLabels() {
    $('map-labels').replaceChildren(); labelNodes.length = 0;
    if (data.points.length < 30) return;
    for (let i = 0; i < labelNames.length; i++) {
      const theta = (i + 0.35) / 6 * TAU, anchor = M.surface(theta, -1.3);
      const candidates = data.points.filter(p => p.topic === labelNames[i]); if (!candidates.length) continue;
      const point = M.nearest(candidates, anchor, 1)[0];
      const el = document.createElement('span'); el.className = 'map-chip'; el.textContent = labelNames[i]; el.style.background = topicColors[labelNames[i]]; $('map-labels').append(el); labelNodes.push({ el, point });
    }
  }
  function stopMotion() {
    replay.playing = false;
    autoOrbit = false; tour = null; transition = null; $('orbit').setAttribute('aria-pressed', 'false'); $('tour').setAttribute('aria-pressed', 'false'); $('tour').firstChild.textContent = 'Follow a path '; $('orbit-icon').textContent = '↻';
    updateReplay();
  }
  function moveCamera(to, duration = 800) {
    transition = { from: { yaw: camera.yaw, pitch: camera.pitch, zoom: camera.zoom, target: { ...camera.target }, pan: { ...camera.pan } }, to, start: performance.now(), duration: reducedMotion ? 0 : duration }; dirty = true;
  }
  function reset(animate = true) {
    stopMotion(); setViewButtons('3d');
    const to = { yaw: -0.18, pitch: 0.38, zoom: 1, target: { x: 0, y: 0, z: 0 }, pan: { x: 0, y: 0 } };
    if (animate) moveCamera(to); else Object.assign(camera, to);
    dirty = true;
  }
  function setViewButtons(next) {
    view = next; $('view-3d').classList.toggle('active', view === '3d'); $('view-top').classList.toggle('active', view === 'top'); $('view-3d').setAttribute('aria-pressed', String(view === '3d')); $('view-top').setAttribute('aria-pressed', String(view === 'top')); $('view-name').textContent = view === 'top' ? 'OVERHEAD / 3D' : 'PERSPECTIVE / 3D';
  }
  function focusPoint() { if (!selected) return; stopMotion(); moveCamera({ zoom: 2.5, target: { x: selected.x, y: selected.y, z: selected.z }, pan: { x: 0, y: 0 } }); }
  function recordingsDataset() {
    const source = window.RECORDINGS, positions = M.recordingPositions(source);
    const points = M.normalizePoints(source.points.map(p => { const [x, y, z] = p.layers[String(source.default_layer)]; return { ...p, x, y, z }; }));
    for (const p of points) p.positions = Object.fromEntries(source.layers.map(n => [n, positions[n].get(p.point_id)]));
    return { ...source, points, layer: source.default_layer };
  }
  function semanticDataset(source = window.RECORDINGS) {
    const known = new Set(['answered', 'redirected', 'refused', 'unclear']);
    const points = source.points.map((p, i) => {
      // Radius from the refusal score: answered turns on the outer rim, core turns inside the core.
      const d = clamp(2.55 - p.core_score * 2.0, 0.16, 3.05);
      const position = source.semantic.radial_mapping === 'smooth_v1' ? M.semanticPosition(p) : { x: d * Math.cos(p.angle), y: d * Math.sin(p.angle), z: p.lift * 0.42 };
      return { ...p, ...position, outcome: known.has(p.outcome) ? p.outcome : 'unlabeled', path: true, radius: 2, tone: (i * 0.618) % 1 };
    });
    const paths = [...new Set(points.map(p => p.conversation_id))].map(id => points.filter(p => p.conversation_id === id).sort((a, b) => a.turn_id - b.turn_id));
    const texture = concept.points.filter(p => !p.path); for (const p of texture) p.textureColor = p.textureColor || spectrum(p);
    return { ...source, kind: 'semantic', points, paths, texture, layer: source.semantic.layer };
  }
  function recordingText() {
    if (recorded() && data.synthetic) {
      $('source-summary').textContent = 'Illustrative path study · no model measurements';
      $('point-count').textContent = '3 CONVERSATIONS / 15 ILLUSTRATIVE TURNS'; return;
    }
    if (data.kind === 'semantic') {
      const c = data.semantic.labels;
      $('source-summary').textContent = `Your recordings · ${data.semantic.frozen ? 'fixed reference' : 'exploratory'} donut · layer ${data.layer}`;
      $('point-count').textContent = `${data.points.length} TURNS / ${c.refused} REFUSED · ${c.redirected} REDIRECTED · ${c.answered} ANSWERED · ${c.unclear + c.unlabeled} OPEN`;
      return;
    }
    if (!recorded()) return;
    const model = String(data.model_id || 'model').split('/').pop();
    $('source-summary').textContent = `Your recordings · ${model} · layer ${data.layer}`;
    if (data.settings?.method === 'pca_preview') $('source-summary').textContent = `Early PCA preview · not UMAP · layer ${data.layer} · waiting for six distinct turns`;
    $('point-count').textContent = `${data.points.length} TURNS / ${data.conversations} CONVERSATIONS · LAYER ${data.layer}`;
  }
  function renderLayerSwitch() {
    $('layer-row').hidden = data.kind !== 'recordings'; if (data.kind !== 'recordings') return;
    $('layer-switch').replaceChildren(...data.layers.map(n => { const b = document.createElement('button'); b.textContent = n; b.className = n === data.layer ? 'active' : ''; b.setAttribute('aria-pressed', String(n === data.layer)); b.onclick = () => setLayer(n); return b; }));
  }
  function setLayer(n) {
    if (data.kind !== 'recordings' || data.layer === n) return;
    // Glide each turn to its place in the other layer's UMAP.
    morph = { start: performance.now(), duration: reducedMotion ? 0 : 900, from: data.points.map(p => ({ x: p.x, y: p.y, z: p.z })) };
    data.layer = n; for (const p of data.points) p.original_coordinates = p.positions[n].original;
    renderLayerSwitch(); recordingText(); if (selected) selectPoint(selected, !$('inspector').hidden);
  }
  function loadDataset(next) {
    replay = { path: [], index: 0, playing: false, reveal: false, next: 0 };
    data = next; morph = null; segments = data.kind === 'concept' ? [] : M.conversationPaths(data.points); filters.clear(); ['answered', 'redirected', 'refused', 'unclear', 'unlabeled'].forEach(x => filters.add(x));
    $('replay').hidden = !recorded();
    $('conversation-select').replaceChildren();
    if (recorded()) for (const id of new Set(data.points.map(p => p.conversation_id))) {
      const p = data.points.find(p => p.conversation_id === id), option = document.createElement('option');
      option.value = id; option.textContent = `${p.conversation_number} · ${p.conversation_title || p.prompt}`; $('conversation-select').append(option);
    }
    document.querySelectorAll('[data-outcome]').forEach(button => { button.hidden = !data.points.some(p => p.outcome === button.dataset.outcome); button.setAttribute('aria-pressed', 'true'); });
    $('contours').checked = shaped(); $('contours').disabled = !shaped(); $('core').checked = shaped(); $('core').disabled = !shaped(); $('texture-row').hidden = !data.texture;
    $('core-sub').textContent = data.kind === 'semantic' ? 'REFUSAL REFERENCE / VISUAL METAPHOR' : 'A CONCEPTUAL LANDMARK';
    $('trails').disabled = data.kind !== 'concept' && !segments.length; $('trails').checked = data.kind === 'concept' || recorded(); $('tour').disabled = data.kind !== 'concept' && !recorded();
    $('color-topic').textContent = recorded() ? 'Conversation' : 'Spectrum'; renderLayerSwitch();
    $('source-summary').textContent = data.kind === 'concept' ? 'Illustrative field · synthetic points' : data.kind === 'umap' ? 'Actual UMAP · synthetic vectors' : `Imported coordinates · ${data.synthetic === true || data.points.every(p => p.synthetic === true) ? 'synthetic' : 'user-supplied data'}`;
    $('point-count').textContent = `${data.points.length.toLocaleString('en-US')} POINTS / ${data.kind === 'concept' ? 'SYNTHETIC FIELD' : data.kind === 'umap' ? '3D UMAP · SYNTHETIC VECTORS' : 'IMPORTED COORDINATES'}`;
    $('search').value = ''; $('search-results').hidden = true; hovered = null; $('hover-label').hidden = true;
    recordingText(); showNeighbors = false; $('neighbors').setAttribute('aria-pressed', 'false'); $('neighbors').lastElementChild.textContent = '＋'; recolor(); makeLabels(); reset(false);
    if (!tools) clearThread();
    else selectPoint(data.kind === 'semantic' ? data.points.reduce((a, b) => b.core_score > a.core_score ? b : a) : data.kind === 'concept' ? data.points.find(p => p.topic === 'intimacy' && p.prompt === 'How do I talk about desire without feeling ashamed?' && p.z < -0.3) || data.points[142] : data.points[0]);
    $('inspector').classList.remove('mobile-open'); $('inspector-mobile').setAttribute('aria-expanded', 'false');
  }
  function nearestScreen(x, y) {
    let candidate = null, best = width <= 760 ? 18 * 18 : 11 * 11;
    const coreCenter = project({ x: 0, y: 0, z: 0 }), coreRadius = viewport.scale * camera.zoom * 0.87 * coreCenter.per;
    if (data.kind === 'concept' && $('core').checked && Math.hypot(x - coreCenter.x, y - coreCenter.y) < coreRadius) return null; // Recorded turns inside the core stay clickable.
    // Iterate near to far so depth breaks ties in favor of the front surface.
    for (let i = drawnPoints.length - 1; i >= 0; i--) { const q = drawnPoints[i], distance = (x - q.x) ** 2 + (y - q.y) ** 2; if (distance < best) { best = distance; candidate = q.p; } }
    return candidate;
  }
  function updateHover(x, y) {
    const p = nearestScreen(x, y), thread = !tools && recorded() ? p?.conversation_id || nearestThread(x, y) : null;
    if (thread !== hoveredThread) { hoveredThread = thread; dirty = true; canvas.style.cursor = thread ? 'pointer' : 'grab'; }
    if (p === hovered) return; hovered = p; dirty = true;
    $('hover-label').hidden = !p;
    if (p) { $('hover-label').textContent = recorded() ? `${p.topic} · turn ${p.turn_id + 1}${data.kind === 'semantic' ? ` · ${p.outcome}` : ''}` : `${p.topic} / ${p.outcome}`; $('hover-label').style.left = `${clamp(x + 15, 10, width - 260)}px`; $('hover-label').style.top = `${y - 35}px`; }
    canvas.style.cursor = p || hoveredThread ? 'pointer' : 'grab';
  }
  function updateSearch() {
    const query = $('search').value.toLowerCase().trim(), root = $('search-results'); root.replaceChildren(); root.hidden = !query; if (!query) return;
    const seen = new Set(), matches = [];
    for (const p of data.points) if (filters.has(p.outcome) && `${p.prompt} ${p.topic} ${p.point_id}`.toLowerCase().includes(query) && !seen.has(p.prompt)) { matches.push(p); seen.add(p.prompt); if (matches.length === 7) break; }
    for (const p of matches) { const b = document.createElement('button'); b.textContent = p.prompt; b.onclick = () => { selectPoint(p); focusPoint(); root.hidden = true; $('search').blur(); }; root.append(b); }
    if (!matches.length) { const note = document.createElement('p'); note.textContent = 'No visible prompts match that search.'; root.append(note); }
  }
  function updateReplay() {
    if (!recorded() || !replay.path.length) return;
    const p = replay.path[replay.index];
    $('conversation-select').value = p.conversation_id;
    $('replay-turn').max = String(replay.path.length - 1); $('replay-turn').value = String(replay.index);
    $('replay-turn').disabled = replay.path.length < 2;
    $('replay-turn').setAttribute('aria-valuetext', `Turn ${p.turn_id + 1} of ${replay.path.length}`);
    $('replay-prev').disabled = replay.index === 0; $('replay-next').disabled = replay.index === replay.path.length - 1;
    $('replay-play').textContent = replay.playing ? 'Pause' : replay.index === replay.path.length - 1 && replay.reveal ? 'Replay' : 'Play';
    $('replay-play').setAttribute('aria-pressed', String(replay.playing));
    $('replay-status').textContent = `Turn ${p.turn_id + 1} / ${replay.path.length} · ${p.outcome}${filters.has(p.outcome) ? '' : ' · hidden by filter'}`;
    $('tour').setAttribute('aria-pressed', String(replay.playing)); $('tour').firstChild.textContent = replay.playing ? 'Pause path ' : 'Follow a path ';
  }
  function seekReplay(index) {
    $('all-conversations').checked = false;
    if (!replay.path.length) return;
    replay.reveal = true; selectPoint(replay.path[clamp(index, 0, replay.path.length - 1)]);
    dirty = true;
  }
  function toggleReplay() {
    if (!replay.path.length) return;
    if (replay.playing) { stopMotion(); dirty = true; return; }
    stopMotion(); $('trails').checked = true;
    if (!replay.reveal || replay.index === replay.path.length - 1) seekReplay(0);
    replay.reveal = true; replay.playing = true; replay.next = performance.now() + 2400; updateReplay(); dirty = true;
  }
  $('conversation-select').onchange = () => { const id = $('conversation-select').value; stopMotion(); selectPoint(M.conversationPath(data.points, id)[0]); };
  $('replay-play').onclick = toggleReplay;
  $('replay-prev').onclick = () => { stopMotion(); seekReplay(replay.index - 1); };
  $('replay-next').onclick = () => { stopMotion(); seekReplay(replay.index + 1); };
  $('replay-turn').oninput = () => { const index = Number($('replay-turn').value); stopMotion(); seekReplay(index); };
  $('replay-full').onclick = () => { stopMotion(); replay.reveal = false; updateReplay(); dirty = true; };
  $('all-conversations').onchange = () => { dirty = true; };
  canvas.addEventListener('pointerdown', e => {
    stopMotion(); canvas.focus({ preventScroll: true }); canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
    drag = { x: e.offsetX, y: e.offsetY, startX: e.offsetX, startY: e.offsetY, moved: false, pan: e.shiftKey || e.button === 2 };
    if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = { distance: Math.hypot(a.x - b.x, a.y - b.y), zoom: camera.zoom, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, pan: { ...camera.pan } }; drag.moved = true; }
    $('hover-label').hidden = true;
  });
  canvas.addEventListener('pointermove', e => {
    if (!pointers.has(e.pointerId)) { if (e.pointerType !== 'touch') updateHover(e.offsetX, e.offsetY); return; }
    pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
    if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()]; camera.zoom = clamp(pinch.zoom * Math.hypot(a.x - b.x, a.y - b.y) / Math.max(pinch.distance, 1), 0.35, 7);
      camera.pan.x = pinch.pan.x + (a.x + b.x) / 2 - pinch.mid.x; camera.pan.y = pinch.pan.y + (a.y + b.y) / 2 - pinch.mid.y;
    } else if (drag) {
      const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
      if (Math.hypot(e.offsetX - drag.startX, e.offsetY - drag.startY) > 4) drag.moved = true;
      if (drag.pan) { camera.pan.x += dx; camera.pan.y += dy; } else { camera.yaw += dx * 0.006; camera.pitch = clamp(camera.pitch + dy * 0.006, -Math.PI * 0.48, Math.PI * 0.48); setViewButtons('3d'); }
      drag.x = e.offsetX; drag.y = e.offsetY;
    }
    dirty = true;
  });
  function pointerEnd(e) {
    if (drag && !drag.moved && pointers.size === 1 && e.type === 'pointerup') {
      const p = nearestScreen(e.offsetX, e.offsetY), thread = !p && !tools && recorded() ? nearestThread(e.offsetX, e.offsetY) : null;
      if (p) selectPoint(p); else if (thread) selectPoint(threadMoment(thread)); else if (!tools) clearThread();
    }
    pointers.delete(e.pointerId); pinch = null;
    if (!pointers.size) drag = null; else { const p = [...pointers.values()][0]; drag = { x: p.x, y: p.y, startX: p.x, startY: p.y, moved: true, pan: false }; }
  }
  canvas.addEventListener('pointerup', pointerEnd); canvas.addEventListener('pointercancel', pointerEnd); canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerleave', () => { hovered = null; $('hover-label').hidden = true; dirty = true; });
  canvas.addEventListener('wheel', e => { e.preventDefault(); stopMotion(); camera.zoom = clamp(camera.zoom * Math.exp(-e.deltaY * 0.0012), 0.35, 7); dirty = true; }, { passive: false });
  function zoom(factor) { stopMotion(); camera.zoom = clamp(camera.zoom * factor, 0.35, 7); dirty = true; }
  $('zoom-in').onclick = () => zoom(1.2); $('zoom-out').onclick = () => zoom(1 / 1.2); $('home').onclick = () => reset(); $('focus-point').onclick = focusPoint;
  $('view-top').onclick = () => { stopMotion(); setViewButtons('top'); moveCamera({ yaw: 0, pitch: 0, target: { x: 0, y: 0, z: 0 }, pan: { x: 0, y: 0 }, zoom: 1 }); };
  $('view-3d').onclick = () => reset();
  $('orbit').onclick = () => { const next = !autoOrbit; stopMotion(); autoOrbit = next; $('orbit').setAttribute('aria-pressed', String(next)); $('orbit-icon').textContent = next ? 'Ⅱ' : '↻'; };
  $('tour').onclick = () => {
    if (recorded()) { toggleReplay(); return; }
    if (tour) { stopMotion(); dirty = true; return; }
    stopMotion(); $('trails').checked = true;
    const start = selected || data.points[0];
    const path = recorded() ? data.points.filter(p => p.conversation_id === start.conversation_id).sort((a, b) => a.turn_id - b.turn_id) : concept.paths[1];
    tour = { path, index: -1, next: 0 }; $('tour').setAttribute('aria-pressed', 'true');
  };
  $('neighbors').onclick = () => { showNeighbors = !showNeighbors; $('neighbors').setAttribute('aria-pressed', String(showNeighbors)); $('neighbors').lastElementChild.textContent = showNeighbors ? '−' : '＋'; dirty = true; };
  $('close-thread').onclick = clearThread;
  $('legend-mobile').onclick = () => { const open = $('legend').classList.toggle('mobile-open'); $('legend-mobile').setAttribute('aria-expanded', String(open)); if (open) clearThread(); };
  $('close-inspector').onclick = () => { $('inspector').hidden = true; $('inspector').classList.remove('mobile-open'); $('inspector-mobile').setAttribute('aria-expanded', 'false'); };
  $('layers-mobile').onclick = () => { const open = $('layers').classList.toggle('mobile-open'); $('layers-mobile').setAttribute('aria-expanded', String(open)); if (open) { $('inspector').classList.remove('mobile-open'); $('inspector-mobile').setAttribute('aria-expanded', 'false'); } };
  $('inspector-mobile').onclick = () => { const open = !$('inspector').classList.contains('mobile-open'); $('inspector').hidden = false; $('inspector').classList.toggle('mobile-open', open); $('inspector-mobile').setAttribute('aria-expanded', String(open)); if (open) { $('layers').classList.remove('mobile-open'); $('layers-mobile').setAttribute('aria-expanded', 'false'); } };
  ['contours', 'trails', 'core', 'texture'].forEach(id => { $(id).onchange = () => { if (id === 'trails' && !$('trails').checked) stopMotion(); dirty = true; }; });
  ['topic', 'response'].forEach(mode => { $(`color-${mode === 'response' ? 'outcome' : mode}`).onclick = () => { colorMode = mode; $('color-topic').classList.toggle('active', mode === 'topic'); $('color-outcome').classList.toggle('active', mode === 'response'); $('color-topic').setAttribute('aria-pressed', String(mode === 'topic')); $('color-outcome').setAttribute('aria-pressed', String(mode === 'response')); recolor(); }; });
  document.querySelectorAll('[data-outcome]').forEach(b => { b.onclick = () => { const outcome = b.dataset.outcome; if (filters.has(outcome)) filters.delete(outcome); else filters.add(outcome); b.setAttribute('aria-pressed', String(filters.has(outcome))); dirty = true; updateSearch(); updateReplay(); }; });
  $('search').addEventListener('input', updateSearch);
  document.addEventListener('pointerdown', e => { if (!e.target.closest('.search-wrap')) $('search-results').hidden = true; });
  $('dataset').onchange = () => {
    if ($('dataset').value === 'study') loadDataset(semanticDataset(window.PATH_STUDY));
    else if ($('dataset').value === 'semantic') loadDataset(semanticDataset());
    else if ($('dataset').value === 'recordings') loadDataset(recordingsDataset());
    else if ($('dataset').value === 'concept') loadDataset(concept);
    else if ($('dataset').value === 'umap') {
      if (!window.UMAP_SAMPLE) { message('The UMAP sample file is missing. You can still load your own coordinates.'); $('dataset').value = 'concept'; return; }
      loadDataset({ ...UMAP_SAMPLE, points: M.normalizePoints(UMAP_SAMPLE.points) });
    } else if (window.importedCloud) loadDataset(window.importedCloud);
  };
  $('load-data').onclick = () => $('file-input').click();
  $('file-input').onchange = async () => {
    const file = $('file-input').files[0]; if (!file) return;
    try {
      if (file.size > 60 * 1024 * 1024) throw new Error('Choose a file smaller than 60 MB.');
      const result = M.parseFile(await file.text());
      window.importedCloud = { name: file.name, kind: 'imported', synthetic: result.metadata.synthetic === true, points: result.points };
      let option = $('dataset').querySelector('[value=imported]'); if (!option) { option = document.createElement('option'); option.value = 'imported'; $('dataset').append(option); } option.textContent = file.name; $('dataset').value = 'imported';
      loadDataset(window.importedCloud); message(`Loaded ${result.points.length.toLocaleString('en-US')} points from ${file.name}.`);
    } catch (error) { message(error.message, 10000); }
    finally { $('file-input').value = ''; }
  };
  $('about-open').onclick = () => $('about').showModal(); $('about-close').onclick = () => $('about').close(); $('about').addEventListener('click', e => { if (e.target === $('about')) { const r = $('about').getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) $('about').close(); } });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { stopMotion(); $('search-results').hidden = true; if (!tools) clearThread(); dirty = true; return; }
    if ($('about').open || /INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName)) return;
    if (e.key === '/' && tools) { e.preventDefault(); $('search').focus(); return; }
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
      e.preventDefault(); stopMotion(); if (e.key === 'ArrowLeft') camera.yaw -= 0.07; if (e.key === 'ArrowRight') camera.yaw += 0.07; if (e.key === 'ArrowUp') camera.pitch -= 0.07; if (e.key === 'ArrowDown') camera.pitch += 0.07; camera.pitch = clamp(camera.pitch, -1.5, 1.5); setViewButtons('3d'); dirty = true;
    } else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoom(1.15); } else if (e.key === '-') { e.preventDefault(); zoom(1 / 1.15); } else if (e.key.toLowerCase() === 'r') reset(); else if (e.key.toLowerCase() === 'f') focusPoint();
  });
  function frame(time) {
    const dt = Math.min((time - lastTime) / 1000, 0.05); lastTime = time;
    if (autoOrbit && !document.hidden && !$('about').open) { camera.yaw += dt * 0.13; dirty = true; }
    if (replay.playing && !document.hidden && !$('about').open && time >= replay.next) {
      if (replay.index >= replay.path.length - 1) { replay.playing = false; updateReplay(); }
      else { seekReplay(replay.index + 1); replay.next = time + 2400; }
    }
    if (tour && time >= tour.next && !$('about').open && !document.hidden) {
      tour.index++; if (tour.index >= tour.path.length) { stopMotion(); message(recorded() ? 'End of this conversation. Select a turn in another conversation to follow its path.' : 'The path ends at the conceptual boundary. Drag to keep exploring.'); }
      else { const p = tour.path[tour.index], k = data.kind === 'recordings' ? 0.8 : 0.4; selectPoint(p); moveCamera({ target: { x: p.x * k, y: p.y * k, z: p.z * k }, zoom: data.kind === 'recordings' ? 1.6 : 1.4, pan: { x: 0, y: 0 } }); tour.next = time + (reducedMotion ? 2600 : 2200); $('tour').firstChild.textContent = `Path ${tour.index + 1} / ${tour.path.length} `; }
    }
    if (morph) {
      const t = morph.duration ? clamp((time - morph.start) / morph.duration, 0, 1) : 1, u = t * t * (3 - 2 * t);
      data.points.forEach((p, i) => { const a = morph.from[i], b = p.positions[data.layer]; p.x = a.x + (b.x - a.x) * u; p.y = a.y + (b.y - a.y) * u; p.z = a.z + (b.z - a.z) * u; });
      if (t === 1) { morph = null; if (selected) neighbors = M.nearest(data.points, selected); }
      dirty = true;
    }
    if (transition) {
      const t = transition.duration ? clamp((time - transition.start) / transition.duration, 0, 1) : 1, u = t * t * (3 - 2 * t), { from, to } = transition;
      for (const key of ['yaw', 'pitch', 'zoom']) if (to[key] !== undefined) camera[key] = from[key] + (to[key] - from[key]) * u;
      for (const key of ['target', 'pan']) if (to[key]) for (const axis of Object.keys(to[key])) camera[key][axis] = from[key][axis] + (to[key][axis] - from[key][axis]) * u;
      if (t === 1) transition = null; dirty = true;
    }
    if (dirty && !document.hidden) render(); requestAnimationFrame(frame);
  }
  new ResizeObserver(resize).observe(canvas); resize();
  if (window.PATH_STUDY) { const option = document.createElement('option'); option.value = 'study'; option.textContent = 'Path study · illustrative'; $('dataset').append(option); }
  const donut = window.RECORDINGS?.semantic && !window.RECORDINGS.semantic.unavailable;
  if (window.RECORDINGS) $('dataset').querySelector('[value=recordings]').hidden = false;
  if (donut) $('dataset').querySelector('[value=semantic]').hidden = false;
  if (window.PATH_STUDY && new URLSearchParams(location.search).has('study')) { $('dataset').value = 'study'; loadDataset(semanticDataset(window.PATH_STUDY)); }
  else if (donut) { $('dataset').value = 'semantic'; loadDataset(semanticDataset()); }
  else if (window.RECORDINGS) { $('dataset').value = 'recordings'; loadDataset(recordingsDataset()); message(`No donut yet: ${window.RECORDINGS.semantic?.unavailable || 'rebuild the map with Update Map from Recordings.'}`, 12000); }
  else if (!tools && window.PATH_STUDY) { $('dataset').value = 'study'; loadDataset(semanticDataset(window.PATH_STUDY)); }
  else loadDataset(concept);
  window.addEventListener('softrefusal:recordings', event => {
    const source = event.detail;
    if (!source?.points?.length) return;
    const firstLive = !window.RECORDINGS;
    const selectedId = selected?.point_id;
    const previousCamera = { ...camera, target: { ...camera.target }, pan: { ...camera.pan } };
    const previousLayer = data.layer;
    window.RECORDINGS = source;
    const hasDonut = source.semantic && !source.semantic.unavailable;
    const recordingOption = $('dataset').querySelector('[value=recordings]');
    recordingOption.hidden = false;
    recordingOption.textContent = source.settings?.method === 'pca_preview' ? 'Your recordings · early PCA preview' : 'Your recordings · UMAP';
    $('dataset').querySelector('[value=semantic]').hidden = !hasDonut;
    if (firstLive || ['semantic', 'recordings'].includes($('dataset').value)) {
      const semantic = hasDonut && (firstLive || $('dataset').value === 'semantic');
      $('dataset').value = semantic ? 'semantic' : 'recordings';
      loadDataset(semantic ? semanticDataset() : recordingsDataset());
      if (!firstLive && data.kind === 'recordings' && data.layers.includes(previousLayer)) setLayer(previousLayer);
      if (!firstLive) Object.assign(camera, previousCamera);
      const conversation = new URLSearchParams(location.search).get('conversation');
      const point = data.points.find(p => p.point_id === selectedId) || data.points.find(p => p.conversation_id === conversation);
      if (point) selectPoint(!tools && !selectedId ? threadMoment(point.conversation_id) : point); else if (!tools) clearThread();
      dirty = true;
    }
  });
  requestAnimationFrame(frame);
})();
