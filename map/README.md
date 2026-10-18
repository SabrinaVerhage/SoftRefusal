# The shape of refusal

A standalone 3D point-cloud prototype for **Mapping the Sexual Avoidance of Artificial Intelligence**. Its visual direction comes from the refusal-core reference and the sticker/keychain iterations in `../spatial-exploration-prototype`: cream paper, black contour lines, peach–pink–lilac pigment, dashed conversation paths, punched cards, and hollow refusal markers.

## Open

Double-click **Open UMAP Prototype.cmd**. It starts a local server in the background, waits until the prototype is ready, and opens your browser. Opening it again reuses the existing server. The launcher uses the existing Recording Room Python environment when available, otherwise installed Python. Everything needed for the viewer is included; no build, extra packages, internet connection, or model download is required.

The usual address is <http://127.0.0.1:8777>. If that port belongs to another application, the launcher chooses a free port up to 8786 and opens the correct address. If something fails, the launcher keeps its error message visible. Server details are saved in `runtime/server.log`.

Double-click **Stop UMAP Prototype.cmd** to stop the background server. Closing the browser does not stop it.

For a local browser preview, run `python serve.py` from this folder, then open <http://127.0.0.1:8777>. The server is bound to this computer and serves only this folder. Stop it with Ctrl+C.

## Explore

- Drag to orbit; scroll to zoom; Shift-drag or right-drag to pan. Touch supports dragging, pinching, and two-finger panning.
- Click a point to inspect its prompt and response. **Go to this point** moves into its neighborhood; **Reset** returns to the whole map.
- Search for a prompt, topic, or point ID. Search results include only visible response categories.
- **Spectrum** applies the reference’s spatial color gradient. **Response** colors by the supplied or assigned response label. Filled dots indicate answered, hollow warm rings redirected, and hollow violet rings refused.
- Toggle the contours, conversation paths, and conceptual core. Filter answered, redirected, refused, or unlabeled points.
- **Show 8 nearest points** draws neighbors measured in the displayed 3D coordinates.
- **Orbit** starts/stops a slow camera orbit. **Follow a path** tours a five-step illustrative trajectory in the reference field.
- Arrow keys orbit, +/− zoom, R resets, F focuses the selected point, / opens search, and Escape stops motion. Motion is paused while the browser is hidden; reduced-motion preferences remove animated camera transitions.

## Your recordings

Double-click **Update Map from Recordings.cmd**. It collects every completed conversation from the cloud GPU recording room (`../recorder/runs/gpu-chat`), writes `data/recordings.js`, and opens the map.

### Your recordings · donut (needs labels)

In cloud mode Gemma automatically tags each saved reply in a separate evaluation request (answered / redirected / refused / unclear). Correct a tag with the buttons under the reply; human corrections always take precedence. **Auto-tag saved replies** fills missing tags or retries failures in the selected conversation. Then update the map. You need at least one *refused* and one *answered* reply, plus six distinct recorded turns for the accompanying UMAP export. These are execution minima, not evidence of a reliable refusal direction. Automatic tags are fallible model judgments, not built-in refusal signals.

The viewer marks automatic and human labels separately and shows the automatic explanation. The first sphere reference can use automatic labels; its anchor provenance is saved. Corrections change markers on the next export. Existing reference geometry stays fixed until an explicit `--rebuild-reference`; review anchors before treating a reference as meaningful.

- **A fixed refusal reference.** The first eligible export saves `data/recordings.reference.json`. Later exports reuse its axis, PCA centre/basis and vertical scale. Refusal scores and the original internal-state angle/lift stay stable when turns are added or labels corrected. The public viewer now uses separate prompt-theme angle/lift fields, so text-theme reassessment can move a point around the ring without changing its refusal score. Keep the reference with your recordings; losing it causes a new reference to be fitted.
- **Distance from the centre follows a reference score.** The direction is the mean unit-normalized refused state minus the mean unit-normalized answered state, at layer 24 by default. Score 0 is the reference answered mean; 1 is the reference refused mean. A fixed smooth radial compression places these toward the rim and inside the central sphere. The score is not a probability. The sphere is a visual metaphor, not a discovered physical boundary.
- **Topic regions.** Each export reassesses all visitor prompts using seeded TF-IDF text similarity and earlier prompts in each conversation. The public display anchors each conversation to its opening topic, limiting later angular drift to 0.20 radians in either direction. This makes inward/outward movement legible instead of drawing chords between distant topics. Labels name populated opening regions. This is a thematic arrangement, not an activation embedding. `angle`/`lift` preserve residual-state PCA values; `topic_angle`/`topic_lift` preserve per-turn text estimates. Display-only `display_angle` supplies the lane. UMAP and the research view remain independent.
- **Lines and markers.** Black dotted tails connect actual checkpoints along the shared surface; tangent arrowheads show message order. No intermediate markers are invented. Larger heads mark the latest turn. Filled black dots with paper outlines are answered; open rings redirected; filled dots with an extra outer ring refused. Refusal score still sets radius, so increasing scores approach the core and decreasing scores can retreat. Selecting a conversation fades the surrounding paths. Research controls retain straight links and conversation colours.
- **One contour surface.** Pigment is filled between the very same rings used for the contour strokes, so they remain aligned while orbiting. The illustrative relief's soft lobes respond to the angular density of real turns; it is not a separate density-map overlay. For more than 2,000 turns the profile uses a bounded, weighted sample. Points and visual connectors use the same surface height. Raw refusal scores and stored coordinates remain unchanged.
- **Flow around dots.** A bounded local pressure field opens contour spacing around real markers, like water around pebbles, with a small shared rise in the surface. A strictly positive spacing field and inverse cumulative distribution keep rings nested; inner/outer edges keep their existing shape. Nearby dots blend without multiplying the deformation. The viewer rebuilds this relief only when loading updated data, rather than during each animation frame. It needs no additional UMAP fit or model request. Conversation angles, radii, marker identities and inward path order are unchanged.
- **Replay.** Choose a conversation above the point card, then play/pause, step backward/forward or scrub the turn slider. Future turns stay faint until reached. **Full path** restores full visibility of the selected path. The camera remains under your control. Hidden outcome categories remain hidden during replay, with a status note when the selected turn is filtered out.
- The reference groups' mean scores are fixed by construction. Label edits update markers immediately on export but do not retrain the geometry. Redirected turns never substitute for refused anchors. If the two group means coincide, the exporter reports that no distinct direction exists and still exports UMAP.
- The public viewer shows only recorded dots. The pink donut surface and core remain illustrative; no decorative point cloud is added. Until recordings load, the viewer shows an empty donut.
- **Featured conversations.** The five newest conversations loop in order, showing each turn's prompt and reply for ten seconds. Generated reasoning can be expanded. Pause/resume with **Pause features / Play features**; interacting with the map pauses the loop. Refreshes retain its place while that conversation remains in the newest five. The loop pauses while the tab is hidden or About is open. **Orbit** starts on by default, except with reduced motion enabled.
- **Mobile.** The explanation starts collapsed under **About the map**. A larger default view and compact controls leave more space for the field. The header uses an inline SVG star to avoid emoji substitution.

To deliberately fit a new reference, run the map command with `--rebuild-reference`. It archives the old reference as `recordings.reference-<id>.json` before replacing it. Use a different output path to preserve a complete alternate map. Measurement changes (model/revision, precision, template, implementation version, layer or token site) require a different output path or an explicit rebuild. The separate UMAP views still refit on each export.

**Path study · illustrative** in the dataset menu demonstrates returning to a topic, moving toward an alternative, and staying near a boundary. All its positions, responses and outcomes are authored examples; no model was queried. Open `http://127.0.0.1:8777/?study` to go directly to it.

### Your recordings · UMAP

- Each point is one turn: Gemma 3 12B's internal state at the final token before it replied. Dashed lines follow each conversation; numbers give the turn order. Colors mark conversations.
- **Model layer** switches between the recorded layers (12, 24, 31, 41). Each layer is its own UMAP, so turns glide to new places; distances are only comparable within one layer.
- The point card shows your message and Gemma's reply. **Follow a path** walks through the selected turn's conversation.
- 4B laptop recordings and 12B cloud recordings are never mixed in one map. The data file contains your conversation text: it is ignored by git and stays on this computer.

With only a few turns, UMAP's layout is a sketch, not a finding.

## Two included datasets

**Reference field:** 6,400 deterministic, art-directed synthetic points in a 3D toroidal landscape. The contours, paths, and central “refusal core” are conceptual design elements. Prompts and outcomes are assigned for demonstration. This field is not a UMAP computation or an inferred refusal boundary.

**3D UMAP sample:** a genuine three-dimensional UMAP reduction of 3,900 synthetic 48-dimensional vectors. UMAP settings: 36 neighbors, minimum distance 0.3, Euclidean metric, seed 42, three output dimensions. Synthetic vectors come from a toroidal manifold lifted linearly into 48 dimensions with small Gaussian noise. Display coordinates receive a rigid PCA rotation, centering, and uniform scaling only. No nonlinear shaping is applied after UMAP. Prompt/response labels are illustrative and do not come from a model. Conceptual contours and core are disabled in this mode.

The sample is included as `data/sample-umap.json` and as an equivalent script for opening from the filesystem. To regenerate it with the existing Recording Room environment, run this from the project root:

```powershell
.\recorder\.venv\Scripts\python.exe map/generate_sample.py
```

UMAP axes have no intrinsic semantic meaning. Projected proximity or a trajectory is not a refusal measurement or evidence of causation.

## Load your coordinates

**Load points** reads a local JSON or JSONL file entirely in the browser. It accepts a JSON array, an object with a `points` array, or one JSON point per line. It can read the existing `recorder/runs/synthetic-smoke-3d/points.jsonl` directly.

```json
{
  "point_id": "conversation-a-turn-2",
  "x": 1.2,
  "y": -0.4,
  "z": 2.1,
  "prompt": "How do I talk about boundaries?",
  "topic": "consent",
  "outcome": "answered",
  "conversation_id": "conversation-a",
  "turn_id": 2,
  "previous_point_id": "conversation-a-turn-1"
}
```

Only finite numeric `x`, `y`, and `z` values are required. IDs must be unique when supplied. Missing responses stay **unlabeled**, and missing prompts are shown as **Prompt text not supplied**. Original coordinates appear in the inspector. Coordinates are centered and uniformly scaled to fit the view; they are never warped into the reference shape. Paths use explicit previous-point links or adjacent turn numbers within a conversation. A file’s text is rendered as text, never HTML.

The viewer accepts up to 100,000 points and files up to 60 MB. For large clouds it samples at most 25,000 visible points for drawing, while retaining all points for search and neighbor calculations. The full count is reported in the footer. Imported data is kept only for the current browser session.

## Files

`index.html`, `style.css`, `app.js`, and `cloud.js` form the viewer. `generate_sample.py` rebuilds the genuine UMAP sample. `launch.py` starts/reuses or stops the local server; `serve.py` serves the viewer. `verify.cjs` checks data parsing, coordinate preservation, neighbor calculations, and compatibility with Recording Room exports.

`node verify.cjs` is self-contained and does not require private recordings. `check-replay.cjs` exercises replay in a local browser (Node 20+, Playwright and installed Edge required); it opens the local HTML directly, or uses `MAP_TEST_URL` when supplied. Screenshots go to ignored `runtime/`. The recorder's `tests/test_reference.py` uses only NumPy and unittest to check reference persistence, relabelling, append stability, incompatible measurements and degenerate directions. Its export tests stub UMAP and make no claim about live-model behaviour.
