// Home: hero, real counts from the repo (labels.txt, models/, config.yaml), cards to the pages.
import { api, app, esc, isStatic, route } from './core.js';

const mb = b => (b / 1e6).toFixed(1) + ' MB';

route('home', async () => {
  const info = await api('/info');
  const t = info.config.models.tracker || {}, d = info.config.models.detector || {};
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Computer vision · detection · segmentation · tracking</p>
    <h1 class="glow">Object Detection and Tracking System</h1>
    <p class="lead">A PyQt5 desktop application that detects objects in images and videos with YOLOv8 (PyTorch or
      Core ML), outlines them with Segment Anything or an OpenCV fallback, follows them from frame to frame with an
      IoU tracker, and can ask a local or cloud LLM to describe each object.</p>
    <a class="btn" href="#/detect">Detect objects</a><a class="btn alt" href="#/tracker">Tracker demo</a><a class="btn alt" href="#/architecture">Architecture</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.n_classes}</b><span>COCO classes in data/models/labels.txt</span></div>
    <div class="tile"><b>${info.n_detectors} · ${info.n_segmentors} · ${info.n_trackers}</b><span>detector · segmentor · tracker back ends</span></div>
    <div class="tile"><b>${info.n_modules}</b><span>Python modules, ${info.n_lines.toLocaleString()} lines</span></div>
    <div class="tile"><b>${esc(d.type || '')} @ ${d.confidence ?? ''}</b><span>configured detector and confidence threshold</span></div>
    <div class="tile"><b>${t.match_thresh ?? ''} / ${t.track_buffer ?? ''}</b><span>tracker match threshold / buffer (frames)</span></div>
  </div>
  ${isStatic() ? `<div class="callout">This is the static showcase. The models are large and carry their own licences,
    so nothing is detected here: run the local app (About) to upload an image or video and see the overlays. What does run
    in your browser is the post-processing: the repo's IoU tracker and the GUI's overlay drawing, ported to JavaScript and
    tested against the Python code. The demo scenes are synthetic.</div>` : ''}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/detect"><b>Detect</b><span>Upload an image or a video, set the confidence and back end, and view boxes, outlines and tracks${isStatic() ? ' (local app only)' : ''}.</span></a>
    <a class="card" href="#/tracker"><b>Tracker</b><span>The simple IoU tracker on a synthetic scene: change the IoU threshold and buffer and watch identities hold or break.</span></a>
    <a class="card" href="#/segment"><b>Segment</b><span>The OpenCV fallback segmentation (used without a SAM checkpoint) on a synthetic scene, computed by the repo's code.</span></a>
    <a class="card" href="#/architecture"><b>Architecture</b><span>How main.py, the GUI, the models and the utilities connect, the processing pipeline, and every module.</span></a>
    <a class="card" href="#/models"><b>Models</b><span>Detector, segmentor, tracker and LLM back ends, the config.yaml settings and the ${info.n_classes} classes.</span></a>
    <a class="card" href="#/about"><b>About</b><span>Running the desktop app and the web app, credits and notes.</span></a>
  </div>
  <h2>Model files in the repository</h2>
  <p>${info.weights_in_repo.map(w => `<span class="badge">${esc(w.file)} · ${mb(w.bytes)}</span>`).join(' ')}
    <span class="badge">SAM checkpoint: downloaded separately (README)</span></p>`;
});
