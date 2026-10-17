/* Run with node verify.cjs. These checks protect imported scientific coordinates. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
require('./cloud.js');
const M = global.CloudMath;
const activityNow = Date.parse('2026-10-17T12:00:00Z');
assert(M.isRecentPoint({completed_utc: '2026-10-17T11:56:00Z'}, activityNow));
assert(!M.isRecentPoint({completed_utc: '2026-10-17T11:55:00Z'}, activityNow));
assert(!M.isRecentPoint({}, activityNow));
assert(!M.isRecentPoint({completed_utc: '2026-10-17T12:01:00Z'}, activityNow));
const history = Array.from({length: 12}, (_, i) => ({conversation_id: `c${i}`, conversation_number: i + 1}));
assert.deepEqual([...M.recentConversations(history)], ['c11','c10','c9','c8','c7','c6','c5','c4','c3','c2']);
history.push({conversation_id: 'c0', conversation_number: 1, completed_utc: '2026-10-17T11:59:00Z'});
assert(M.recentConversations(history).has('c0'));
assert(!M.recentConversations(history).has('c2'));
const featuredFixture = Array.from({length:7}, (_,i)=>[2,0,1].map(turn_id=>({conversation_id:`thread${i}`,conversation_number:i+1,turn_id}))).flat();
const featured = M.featuredTurns(featuredFixture);
assert.deepEqual([...new Set(featured.map(p=>p.conversation_id))], ['thread6','thread5','thread4','thread3','thread2']);
assert.deepEqual(featured.slice(0,3).map(p=>p.turn_id),[0,1,2]);
assert.deepEqual(M.featuredTurns([]),[]);
const themed = [{x:0,y:0,z:0,prompt:'A romantic kiss'}, {x:.2,y:.1,z:0,prompt:'A tender kiss'}, {x:5,y:5,z:5,prompt:'Distant romance'}];
const regions = M.themeRegions(themed);
assert.equal(regions[0].name, 'romance & intimacy');
assert.equal(regions[0].count, 2);
assert.equal(regions[0].point.x, .1);
assert.deepEqual(M.themeRegions([{x:0,y:0,z:0,prompt:'hello'}]), []);
const concept = M.makeConcept();
assert.equal(concept.points.length, 6400);
assert.equal(new Set(concept.points.map(p => p.point_id)).size, 6400);
assert(concept.points.every(p => ['x', 'y', 'z'].every(k => Number.isFinite(p[k]))));
const source = [{ point_id: 'a', x: 5, y: 10, z: -3 }, { point_id: 'b', x: 7, y: 13, z: 1 }, { point_id: 'c', x: 8, y: 9, z: 2 }];
const normalized = M.normalizePoints(source);
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
assert(Math.abs(distance(normalized[0], normalized[1]) / distance(normalized[0], normalized[2]) - distance(source[0], source[1]) / distance(source[0], source[2])) < 1e-12);
assert.deepEqual(normalized[0].original_coordinates, [5, 10, -3]);
assert.equal(normalized[0].outcome, 'unlabeled');
assert.equal(source[0].x, 5);
assert.equal(M.parseFile(source.map(p => JSON.stringify(p)).join('\r\n')).points.length, 3);
assert.equal(M.parseFile(JSON.stringify({ points: source })).points.length, 3);
assert.equal(M.parseFile(JSON.stringify(source[0])).points.length, 1);
assert.throws(() => M.parseFile(''), /empty/);
assert.throws(() => M.parseFile('nope'), /JSON/);
assert.throws(() => M.normalizePoints([{ x: null, y: 1, z: 2 }]), /numeric/);
assert.throws(() => M.normalizePoints([{ x: Infinity, y: 1, z: 2 }]), /numeric/);
assert.throws(() => M.normalizePoints([source[0], source[0]]), /Duplicate/);
assert.equal(M.nearest(normalized, normalized[0], 1)[0].point_id, 'b');
const paths = M.conversationPaths([{ point_id: 'a', conversation_id: 'x', turn_id: 0 }, { point_id: 'b', conversation_id: 'x', turn_id: 1, previous_point_id: 'a' }, { point_id: 'c', conversation_id: 'x', turn_id: 2, previous_point_id: 'b' }]);
assert.equal(paths.length, 2);
assert.equal(M.conversationPaths([{ point_id: 'a', conversation_id: 'x', turn_id: 0 }, { point_id: 'b', conversation_id: 'x', turn_id: 1, previous_point_id: 'a' }, { point_id: 'c', conversation_id: 'x', turn_id: 2 }]).length, 2);
assert.equal(M.normalizePoints([{ x: 0, y: 0, z: 0 }, { x: 1e-15, y: 0, z: 0 }])[1].x, 2.7);
const sample = JSON.parse(fs.readFileSync(path.join(__dirname, 'data/sample-umap.json'), 'utf8'));
assert.equal(sample.kind, 'umap'); assert.equal(sample.synthetic, true); assert.equal(sample.points.length, 3900);
assert.equal(M.normalizePoints(sample.points).length, 3900);
// Self-contained export-shaped fixture; fresh checkouts have no private runs/ directory.
const recording = Array.from({ length: 12 }, (_, i) => JSON.stringify({
  point_id: `fixture-${i}`, conversation_id: `fixture-${Math.floor(i / 2)}`, turn_id: i % 2,
  x: i * 0.2, y: i % 3, z: i % 2, synthetic: true, outcome: 'unlabeled'
})).join('\n');
const recorded = M.parseFile(recording).points;
assert.equal(recorded.length, 12); assert(recorded.every(p => p.synthetic === true && p.outcome === 'unlabeled')); assert.equal(M.conversationPaths(recorded).length, 6);
const layered = { layers: [12, 24], points: source.map((p, i) => ({ point_id: p.point_id, layers: { 12: [p.x, p.y, p.z], 24: [p.x * 100 + i, -p.y, p.z * 3] } })) };
const byLayer = M.recordingPositions(layered);
const chronological = M.conversationPath([{ conversation_id: 'b', turn_id: 0 }, { conversation_id: 'a', turn_id: 2 }, { conversation_id: 'a', turn_id: 0 }], 'a');
assert.deepEqual(chronological.map(p => p.turn_id), [0, 2]);
assert.equal(M.conversationColorIndex('stable-id', 8), M.conversationColorIndex('stable-id', 8));
const scorePoints = [-2, 0, 0.5, 1, 3].map(core_score => M.semanticPosition({ core_score, angle: 0.3, lift: 0.2 }));
assert(scorePoints.every(p => Object.values(p).every(Number.isFinite)));
assert(scorePoints.slice(1).every((p, i) => Math.hypot(p.x, p.y) < Math.hypot(scorePoints[i].x, scorePoints[i].y)));
assert.deepEqual(M.semanticPosition({ core_score: 0.7, angle: 0.3, lift: 0.2 }), M.semanticPosition({ core_score: 0.7, angle: 0.3, lift: 0.2, outcome: 'refused' }));
assert.equal(M.normalizePoints([{ x: 0, y: 0, z: 0, outcome: 'unclear' }])[0].outcome, 'unclear');
assert.deepEqual(byLayer[12].get('a').original, [5, 10, -3]);
assert(Math.abs(distance(byLayer[12].get('a'), byLayer[12].get('b')) / distance(byLayer[12].get('a'), byLayer[12].get('c')) - distance(source[0], source[1]) / distance(source[0], source[2])) < 1e-12);
assert(Math.max(...[...byLayer[24].values()].map(p => Math.abs(p.x))) <= 2.7 + 1e-12); // Each layer scaled on its own.
const exported = path.join(__dirname, 'data/recordings.json');
if (fs.existsSync(exported)) {
  const r = JSON.parse(fs.readFileSync(exported, 'utf8'));
  assert.equal(r.kind, 'recordings'); assert.equal(r.synthetic, false); assert(r.layers.includes(r.default_layer));
  assert(r.points.every(p => r.layers.every(n => p.layers[String(n)].length === 3) && typeof p.prompt === 'string' && typeof p.response === 'string'));
  assert.equal(M.conversationPaths(r.points).length, r.points.length - r.conversations);
}
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8'), app = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(x => x[1]));
for (const match of app.matchAll(/\$\('([^']+)'\)/g)) assert(ids.has(match[1]), `Missing DOM ID: ${match[1]}`);
console.log('Passed: 6,400-point field; genuine UMAP sample; JSON/JSONL; distance-preserving display; original coordinates; invalid data; explicit paths; Recording Room compatibility; DOM wiring.');
