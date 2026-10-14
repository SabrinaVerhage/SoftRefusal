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

Label replies in the recording room with the buttons under each Gemma reply (answered / redirected / refused / unclear), then update the map. You need at least one *refused* (or *redirected*) and one *answered* reply.

- **Distance from the centre = refusal score.** From Gemma's internal state at layer 24 before each reply: the direction from your *answered* turns' average state to your *refused* turns' average state is the refusal axis. Each turn's position along it is its score: 0 sits on the outer rim with the answered turns, 1 inside the core with the refused turns.
- **Angle = semantic approach.** What remains of the state once the refusal axis is removed (its first two principal components).
- **Lines** follow each conversation in order: diving inward means pressing towards the core, circling the rim means being deflected. Filled dots are answered, rings redirected, thick rings refused; colours mark conversations.
- The axis is defined by the labelled turns it shows, so they land where their labels say by construction. Unlabelled and *redirected* turns are not used to build it, so where they land is the interesting part. It sharpens as you label more.
- The faint dot texture is the illustrative reference field (toggle **Field texture**); only the larger dots on the paths are your data.

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
