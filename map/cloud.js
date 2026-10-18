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
    const angle = p.display_angle ?? p.topic_angle ?? p.angle, lift = p.topic_lift ?? p.lift ?? 0;
    return { x: radius * Math.cos(angle), y: radius * Math.sin(angle), z: lift * 0.42 };
  }
  function pointTime(p) {
    const time = Date.parse(p.completed_utc || '');
    return Number.isFinite(time) ? time : 0;
  }
  function isRecentPoint(p, now = Date.now()) {
    const time = pointTime(p);
    return time > 0 && now >= time && now - time < 5 * 60 * 1000;
  }
  function recentConversations(points, count = 10) {
    const groups = new Map();
    for (const p of points) {
      if (p.conversation_id == null) continue;
      const old = groups.get(p.conversation_id) || { time: 0, number: 0 };
      groups.set(p.conversation_id, { time: Math.max(old.time, pointTime(p)), number: Math.max(old.number, p.conversation_number || 0) });
    }
    return new Set([...groups].sort((a, b) => b[1].time - a[1].time || b[1].number - a[1].number || String(a[0]).localeCompare(String(b[0]))).slice(0, count).map(([id]) => id));
  }
  function featuredTurns(points, count = 5) {
    return [...recentConversations(points, count)].flatMap(id => conversationPath(points, id));
  }
  function conversationLanes(points) {
    const openings = new Map();
    for (const p of points) {
      const old = openings.get(p.conversation_id);
      if (!old || p.turn_id < old.turn_id) openings.set(p.conversation_id, p);
    }
    return points.map(p => {
      const opening = openings.get(p.conversation_id);
      if (!Number.isFinite(opening?.topic_angle)) return {...p};
      const angle = Number.isFinite(p.topic_angle) ? p.topic_angle : opening.topic_angle;
      const delta = Math.atan2(Math.sin(angle - opening.topic_angle), Math.cos(angle - opening.topic_angle));
      return {...p, display_angle: opening.topic_angle + clamp(delta * 0.12, -0.20, 0.20),
        opening_topic_id: opening.topic_id, opening_topic: opening.topic};
    });
  }
  // Local prompt-word hints, not a claim that model activations have named semantic axes.
  function themeRegions(points, exportedThemes) {
    if (points.some(p => p.opening_topic_id)) {
      const groups = new Map();
      for (const p of points) {
        if (!p.opening_topic_id) continue;
        if (!groups.has(p.opening_topic_id)) groups.set(p.opening_topic_id, []);
        groups.get(p.opening_topic_id).push(p);
      }
      return [...groups.values()].map(members => {
        const openings = new Map();
        for (const p of members) if (!openings.has(p.conversation_id) || p.turn_id < openings.get(p.conversation_id).turn_id) openings.set(p.conversation_id,p);
        const starts = [...openings.values()];
        const point = Object.fromEntries(['x','y','z'].map(key => [key, starts.reduce((sum,p) => sum + p[key],0) / starts.length]));
        return {name: members[0].opening_topic, point, members, count: members.length};
      });
    }
    if (exportedThemes?.regions) {
      const byId = new Map(points.map(p => [p.point_id, p]));
      return exportedThemes.regions.flatMap(region => {
        const members = region.point_ids.map(id => byId.get(id)).filter(Boolean);
        if (!members.length) return [];
        const point = Object.fromEntries(['x', 'y', 'z'].map(key => [key, members.reduce((sum, p) => sum + p[key], 0) / members.length]));
        return [{name: region.name, point, members, count: members.length}];
      });
    }
    const themes = [
      ['consent & boundaries', /\b(consent|boundar\w*|permission|comfortable|respect|agree)\b/i],
      ['romance & intimacy', /\b(romanti\w*|kiss\w*|intima\w*|cuddl\w*|tender\w*|love|flirt\w*)\b/i],
      ['desire & fantasy', /\b(desire|fantas\w*|erotic\w*|arous\w*|pleasure|spicy|sensual\w*)\b/i],
      ['stories & roleplay', /\b(stor\w*|fiction|roleplay|role.play|character\w*|scene\w*|poem\w*)\b/i],
      ['relationships', /\b(partner\w*|relationship\w*|dating|boyfriend|girlfriend|marriage|couple\w*)\b/i],
      ['bodies & health', /\b(bod\w*|health\w*|anatom\w*|contracept\w*|pregnan\w*|orgasm\w*|sex\w*)\b/i]
    ];
    const regions = [];
    for (const [name, pattern] of themes) {
      const matching = points.filter(p => pattern.test(p.prompt || ''));
      // Bound annotation work on large imports; the full cloud remains intact.
      const stride = Math.max(1, Math.ceil(matching.length / 600));
      const candidates = matching.filter((_, i) => i % stride === 0);
      if (candidates.length < 2) continue;
      let cluster = [];
      for (const anchor of candidates) {
        const nearby = candidates.filter(p => Math.hypot(p.x - anchor.x, p.y - anchor.y, p.z - anchor.z) <= 1.25);
        if (nearby.length > cluster.length) cluster = nearby;
      }
      if (cluster.length < 2) continue;
      const center = Object.fromEntries(['x', 'y', 'z'].map(k => [k, cluster.reduce((sum, p) => sum + p[k], 0) / cluster.length]));
      if (regions.some(r => Math.hypot(r.point.x - center.x, r.point.y - center.y, r.point.z - center.z) < 0.65)) continue;
      regions.push({ name, point: center, members: cluster, count: cluster.length });
    }
    return regions;
  }
  function makeLandscape(points) {
    // One annular surface supplies BOTH pigment and contour vertices. Its soft
    // lobes follow actual turns. A positive spacing field lets the contours part
    // around each real marker without crossing or moving its topic/refusal lane.
    const bins = 96, profile = new Float64Array(bins), stride = Math.max(1, Math.ceil(points.length / 2000));
    for (let k = 0; k < points.length; k += stride) {
      const angle = Math.atan2(points[k].y, points[k].x);
      for (let j = 0; j < bins; j++) {
        const delta = Math.atan2(Math.sin(j / bins * TAU - angle), Math.cos(j / bins * TAU - angle));
        profile[j] += Math.exp(-0.5 * (delta / 0.36) ** 2) * stride;
      }
    }
    const peak = Math.max(1e-12, ...profile);
    const density = angle => {
      const index = ((angle / TAU % 1 + 1) % 1) * bins, a = Math.floor(index), t = index - a;
      return (profile[a % bins] * (1 - t) + profile[(a + 1) % bins] * t) / peak;
    };
    const bounds = angle => ({inner: 0.98 + 0.07 * Math.sin(3 * angle) + 0.03 * Math.cos(5 * angle),
      outer: 2.86 + 0.24 * density(angle) + 0.18 * Math.sin(3 * angle + 0.4) + 0.09 * Math.cos(5 * angle)});
    const radiusAt = (angle,u) => {
      const {inner,outer}=bounds(angle);
      return inner+(outer-inner)*(u+0.07*density(angle)*Math.sin(Math.PI*u));
    };
    const columns=320, samples=129, radialWidth=.17, streamWidth=.30;
    const pebbles=[];
    for(let k=0;k<points.length;k+=stride){
      const p=points[k],radius=Math.hypot(p.x,p.y);
      if(radius>.87&&radius<3.7)pebbles.push({radius,angle:Math.atan2(p.y,p.x)});
    }
    const flow=Array.from({length:columns},(_,i)=>{
      const angle=i/columns*TAU;
      const radii=Float64Array.from({length:samples},(_,j)=>radiusAt(angle,j/(samples-1)));
      const pressure=new Float64Array(samples),cdf=new Float64Array(samples);
      for(const pebble of pebbles){
        const delta=Math.atan2(Math.sin(angle-pebble.angle),Math.cos(angle-pebble.angle));
        const across=delta*Math.max(pebble.radius,.98);
        if(Math.abs(across)>3*streamWidth)continue;
        const lateral=Math.exp(-.5*(across/streamWidth)**2);
        for(let j=0;j<samples;j++){
          const along=radii[j]-pebble.radius;
          if(Math.abs(along)>3*radialWidth)continue;
          pressure[j]=Math.max(pressure[j],lateral*Math.exp(-.5*(along/radialWidth)**2));
        }
      }
      // Lower contour density beside a marker creates breathing room. The floor
      // keeps the inverse cumulative field strictly ordered, even in dense groups.
      let previous=1;
      for(let j=0;j<samples;j++){
        const weight=.16+.84*(1-pressure[j])**2;
        if(j)cdf[j]=cdf[j-1]+(previous+weight)/2;
        previous=weight;
      }
      for(let j=1;j<samples;j++)cdf[j]/=cdf[samples-1];
      return {radii,pressure,cdf};
    });
    function interpolate(xs,ys,value){
      if(value<=xs[0])return ys[0];
      if(value>=xs[xs.length-1])return ys[ys.length-1];
      let lo=0,hi=xs.length-1;
      while(hi-lo>1){const mid=(lo+hi)>>1;if(xs[mid]<value)lo=mid;else hi=mid;}
      const t=(value-xs[lo])/(xs[hi]-xs[lo]);
      return ys[lo]+(ys[hi]-ys[lo])*t;
    }
    const pressureAt=(angle,radius)=>{
      const index=((angle/TAU%1+1)%1)*columns,a=Math.floor(index),t=index-a;
      const first=flow[a%columns],next=flow[(a+1)%columns];
      return interpolate(first.radii,first.pressure,radius)*(1-t)+interpolate(next.radii,next.pressure,radius)*t;
    };
    const heightAt = (x,y) => {
      const angle = Math.atan2(y,x), {inner,outer} = bounds(angle), u = clamp((Math.hypot(x,y) - inner) / (outer - inner),0,1);
      return Math.sin(Math.PI * u) * (0.4 + 0.09 * density(angle) + 0.06 * Math.sin(2 * angle)+.075*pressureAt(angle,Math.hypot(x,y)));
    };
    const rings = Array.from({length:25},(_,j) => Array.from({length:columns},(_,i) => {
      const angle = i / columns * TAU,u=j/24;
      const radius=interpolate(flow[i].cdf,flow[i].radii,u);
      const x = radius * Math.cos(angle), y = radius * Math.sin(angle);
      return {x,y,z:heightAt(x,y)};
    }));
    return {rings, inner:rings[0], outer:rings.at(-1), heightAt};
  }
  function surfaceTrail(a,b,landscape) {
    // A visual connector through real endpoints, following the surface rather
    // than a chord that slices sideways across the core. No extra data markers.
    const from = Math.atan2(a.y,a.x), to = Math.atan2(b.y,b.x);
    const delta = Math.atan2(Math.sin(to-from),Math.cos(to-from)), ra = Math.hypot(a.x,a.y), rb = Math.hypot(b.x,b.y);
    const steps = Math.max(12,Math.ceil(Math.abs(ra-rb)*10));
    return Array.from({length:steps+1},(_,i) => {
      if (!i) return a;
      if (i === steps) return b;
      const t = i / steps, angle = from + delta * (t*t*(3-2*t)), radius = ra + (rb-ra)*t;
      const x = radius*Math.cos(angle),y = radius*Math.sin(angle);
      return {x,y,z:landscape.heightAt(x,y)};
    });
  }
  function project(p, camera, viewport) {
    const x = p.x - camera.target.x, y = p.y - camera.target.y, z = p.z - camera.target.z;
    const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw), cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
    const xx = x * cy + z * sy, zz = -x * sy + z * cy, yy = y * cp - zz * sp, depth = y * sp + zz * cp;
    const per = 8 / Math.max(0.4, 8 + depth), scale = viewport.scale * camera.zoom;
    return { x: viewport.cx + xx * per * scale + camera.pan.x, y: viewport.cy - yy * per * scale + camera.pan.y, depth, per };
  }
  root.CloudMath = { TAU, clamp, rng, surface, makeConcept, normalizePoints, recordingPositions, parseFile, conversationPaths, conversationPath, conversationColorIndex, semanticPosition, nearest, project, topics, pointTime, isRecentPoint, recentConversations, featuredTurns, themeRegions, conversationLanes, makeLandscape, surfaceTrail };
})(typeof window !== 'undefined' ? window : globalThis);
