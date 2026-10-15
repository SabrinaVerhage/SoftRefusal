/* Dataset and camera math. No network requests or rendering dependencies. */
(function (root) {
  'use strict';
  const TAU = Math.PI * 2;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const topics = ['consent', 'intimacy', 'desire', 'erotic fiction', 'relationships', 'sexual health'];
  const prompts = [
    ['How can I talk about consent with a new partner?', 'How do I say no without feeling guilty?', 'Can we talk about boundaries before a date?'],
    ['How do I talk about desire without feeling ashamed?', 'How can I ask for more closeness?', 'Why does intimacy sometimes make me anxious?'],
    ['Why is it difficult to put desire into words?', 'Can you help me describe a feeling of longing?', 'How do I tell someone I am attracted to them?'],
    ['Can you help me write a romantic scene?', 'How would you approach an erotic story?', 'What makes a scene feel intimate?'],
    ['How can I tell my partner what I need?', 'How do we rebuild trust in a relationship?', 'How do I start a difficult conversation?'],
    ['Where can I find clear information about sexual health?', 'How do I ask a health professional a sensitive question?', 'Can you explain the difference between desire and arousal?']
  ];
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gaussian(random) { return Math.sqrt(-2 * Math.log(Math.max(random(), 1e-8))) * Math.cos(TAU * random()); }
  function surface(theta, phi) {
    const major = 1.9 + 0.11 * Math.sin(3 * theta + 0.4) + 0.05 * Math.cos(5 * theta);
    const tube = 0.71 + 0.12 * Math.sin(2 * theta - 0.7);
    const r = major + tube * Math.cos(phi) + 0.085 * Math.sin(4 * theta + 2 * phi);
    return { x: r * Math.cos(theta), y: r * Math.sin(theta), z: tube * 0.74 * Math.sin(phi) + 0.15 * Math.sin(3 * theta) };
  }
  function makeConcept() {
    const random = rng(83), points = [], paths = [];
    for (let i = 0; i < 6370; i++) {
      const theta = random() * TAU, phi = random() * TAU, p = surface(theta, phi);
      const noise = random() < 0.88 ? 0.047 : 0.16;
      p.x += gaussian(random) * noise; p.y += gaussian(random) * noise; p.z += gaussian(random) * noise;
      const topicIndex = Math.floor(theta / TAU * 6) % 6;
      const outcome = topicIndex === 3 && Math.cos(phi) < -0.2 ? 'refused' : random() < 0.13 ? 'redirected' : 'answered';
      points.push({ ...p, point_id: `field-${String(i).padStart(4, '0')}`, topic: topics[topicIndex], prompt: prompts[topicIndex][i % 3], outcome, synthetic: true, tone: random(), radius: 0.65 + random() * 0.85 });
    }
    for (let j = 0; j < 6; j++) {
      const path = [];
      for (let k = 0; k < 5; k++) {
        const theta = j / 6 * TAU + 0.22 + 0.1 * Math.sin(k * 1.3 + j), r = 2.87 - k * 0.41;
        const p = { x: r * Math.cos(theta), y: r * Math.sin(theta), z: -0.66 * Math.sin(k / 4 * Math.PI) - 0.03, point_id: `path-${j}-${k}`, topic: topics[j], prompt: prompts[j][k % 3], outcome: k === 4 ? 'refused' : k === 3 ? 'redirected' : 'answered', synthetic: true, conversation_id: `illustrative-path-${j}`, turn_id: k, previous_point_id: k ? `path-${j}-${k - 1}` : null, tone: 0.5, radius: 2, path: true };
        points.push(p); path.push(p);
      }
      paths.push(path);
    }
    return { name: 'Reference field', kind: 'concept', synthetic: true, points, paths };
  }
  function normalizePoints(rows) {
    if (!Array.isArray(rows) || !rows.length) throw new Error('The file needs a non-empty points array.');
    if (rows.length > 100000) throw new Error('This prototype supports up to 100,000 points per file.');
    const seen = new Set();
    const points = rows.map((row, i) => {
      if (!row || typeof row !== 'object' || !['x', 'y', 'z'].every(k => typeof row[k] === 'number' && Number.isFinite(row[k]))) throw new Error(`Point ${i + 1} needs finite numeric x, y, and z coordinates.`);
      const id = String(row.point_id ?? row.id ?? `import-${i}`);
      if (seen.has(id)) throw new Error(`Duplicate point ID: ${id}.`);
      seen.add(id);
      let outcome = String(row.outcome ?? 'unlabeled').toLowerCase();
      outcome = ({ engaged: 'answered', allow: 'answered', refused: 'refused', refusal: 'refused', redirected: 'redirected', answered: 'answered', unclear: 'unclear' })[outcome] || 'unlabeled';
      return { ...row, point_id: id, prompt: String(row.prompt ?? row.text ?? row.label ?? 'Prompt text not supplied'), topic: String(row.topic ?? 'unclassified'), outcome, original_coordinates: Array.isArray(row.original_coordinates) && row.original_coordinates.length === 3 && row.original_coordinates.every(Number.isFinite) ? row.original_coordinates.slice() : [row.x, row.y, row.z], x: row.x, y: row.y, z: row.z, tone: (i * 0.61803398875) % 1, radius: rows.length < 100 ? 2.5 : 1.3 };
    });
    const mins = [Infinity, Infinity, Infinity], maxs = [-Infinity, -Infinity, -Infinity];
    for (const p of points) [p.x, p.y, p.z].forEach((v, i) => { mins[i] = Math.min(mins[i], v); maxs[i] = Math.max(maxs[i], v); });
    const center = mins.map((v, i) => v / 2 + maxs[i] / 2);
    const span = Math.max(...maxs.map((v, i) => v - mins[i]));
    if (!Number.isFinite(span)) throw new Error('The coordinate range is too large to display.');
    for (const p of points) { p.x = span ? (p.x - center[0]) / span * 5.4 : 0; p.y = span ? (p.y - center[1]) / span * 5.4 : 0; p.z = span ? (p.z - center[2]) / span * 5.4 : 0; }
    return points;
  }
  function recordingPositions(dataset) {
    // Each recorded layer has its own UMAP: center and scale layers separately, never across them.
    const positions = {};
    for (const layer of dataset.layers) {
      const rows = dataset.points.map(p => { const [x, y, z] = p.layers[String(layer)]; return { ...p, x, y, z }; });
      positions[layer] = new Map(normalizePoints(rows).map(p => [p.point_id, { x: p.x, y: p.y, z: p.z, original: p.original_coordinates }]));
    }
    return positions;
  }
  function parseFile(text) {
    const trimmed = text.trim();
    if (!trimmed) throw new Error('This file is empty.');
    let payload;
    try { payload = JSON.parse(trimmed); }
    catch { try { payload = trimmed.split(/\r?\n/).filter(x => x.trim()).map(x => JSON.parse(x)); } catch { throw new Error('Could not read this file as JSON or JSONL.'); } }
    const rows = Array.isArray(payload) ? payload : Array.isArray(payload?.points) ? payload.points : payload && typeof payload.x === 'number' ? [payload] : null;
    return { metadata: Array.isArray(payload) || rows?.[0] === payload ? {} : payload, points: normalizePoints(rows) };
  }
  function conversationPaths(points) {
    const byId = new Map(points.map(p => [p.point_id, p])), segments = [], grouped = new Map();
    for (const p of points) {
      if (p.previous_point_id != null) { const previous = byId.get(String(p.previous_point_id)); if (previous) segments.push([previous, p]); }
      if (p.conversation_id != null && Number.isFinite(p.turn_id)) { const key = String(p.conversation_id); if (!grouped.has(key)) grouped.set(key, []); grouped.get(key).push(p); }
    }
    for (const group of grouped.values()) { group.sort((a, b) => a.turn_id - b.turn_id); for (let i = 1; i < group.length; i++) if (group[i].previous_point_id == null && group[i].turn_id === group[i - 1].turn_id + 1) segments.push([group[i - 1], group[i]]); }
    return segments;
  }
  function nearest(points, point, count = 8) { return points.filter(p => p !== point).map(p => ({ point: p, distance: Math.hypot(p.x - point.x, p.y - point.y, p.z - point.z) })).sort((a, b) => a.distance - b.distance).slice(0, count).map(x => x.point); }
  function conversationPath(points, id) {
    return points.filter(p => p.conversation_id === id).sort((a, b) => a.turn_id - b.turn_id);
  }
  function conversationColorIndex(id, count) {
    let hash = 2166136261;
    for (const c of String(id)) hash = Math.imul(hash ^ c.charCodeAt(0), 16777619);
    return (hash >>> 0) % count;
  }
  function semanticPosition(p) {
    // Fixed artistic radial compression. Preserve the unbounded score for inspection.
    const radius = 0.16 + 2.89 / (1 + Math.exp(clamp(3.8 * (p.core_score - 0.5), -700, 700)));
    return { x: radius * Math.cos(p.angle), y: radius * Math.sin(p.angle), z: p.lift * 0.42 };
  }
  function project(p, camera, viewport) {
    const x = p.x - camera.target.x, y = p.y - camera.target.y, z = p.z - camera.target.z;
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw), cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    const xx = x * cy + z * sy, zz = -x * sy + z * cy, yy = y * cp - zz * sp, depth = y * sp + zz * cp;
    const per = 8 / Math.max(0.4, 8 + depth), scale = viewport.scale * camera.zoom;
    return { x: viewport.cx + xx * per * scale + camera.pan.x, y: viewport.cy - yy * per * scale + camera.pan.y, depth, per };
  }
  root.CloudMath = { TAU, clamp, rng, surface, makeConcept, normalizePoints, recordingPositions, parseFile, conversationPaths, conversationPath, conversationColorIndex, semanticPosition, nearest, project, topics };
})(typeof window !== 'undefined' ? window : globalThis);
