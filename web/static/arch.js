// Architecture: data flow (INTERFACE.md), the per-image and per-video pipeline (gui/main_window.ProcessingThread),
// and every module with its classes and functions (read from the source with ast).
import { api, app, esc, route } from './core.js';

const box = (x, y, w, h, cls, title, sub) => `<rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}"/>
  <text x="${x + 10}" y="${y + 20}">${esc(title)}</text>${sub ? `<text class="sub" x="${x + 10}" y="${y + 37}">${esc(sub)}</text>` : ''}`;
const arrow = d => `<path d="${d}"/>`;

function diagram() {
  return `<svg class="diagram" viewBox="0 0 880 345" role="img" aria-label="Data flow: main.py loads config.yaml, authenticates with auth, then opens the GUI, which runs the detector, segmentor, tracker and LLM and renders the results">
  <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#94a3c6"/></marker></defs>
  ${box(20, 20, 170, 46, 'a', 'main.py', 'arguments, system info')}
  ${box(20, 100, 170, 46, '', 'config.yaml', 'models, tracker, GUI')}
  ${box(20, 180, 170, 46, '', 'auth/', 'login, salted hashes')}
  ${box(20, 260, 170, 46, '', 'utils/', 'system info, formats')}
  ${box(250, 120, 190, 66, 'b', 'gui/main_window.py', 'ProcessingThread (QThread)')}
  ${box(510, 10, 180, 46, 'a', 'models/detector.py', 'YOLOv8 · Core ML · TF')}
  ${box(510, 72, 180, 46, 'a', 'models/segmentor.py', 'SAM · OpenCV fallback')}
  ${box(510, 134, 180, 46, 'a', 'models/tracker.py', 'ByteTrack · SORT · IoU')}
  ${box(510, 196, 180, 46, '', 'models/llm_integration', 'Ollama · cloud · CLI')}
  ${box(510, 264, 180, 46, '', 'plugins/', 'BasePlugin, manager')}
  ${box(725, 72, 148, 46, 'c', 'postprocess/', 'IoU, colours, CoreML')}
  ${box(725, 160, 148, 46, 'b', 'gui/visualization', 'overlays, zoom')}
  ${arrow('M105 66 V100')}${arrow('M105 146 V180')}${arrow('M190 43 C230 43 230 140 250 140')}
  ${arrow('M190 203 C220 203 225 170 250 170')}${arrow('M190 283 C230 283 240 186 300 186')}
  ${arrow('M440 140 C480 140 480 33 510 33')}${arrow('M440 148 C480 148 480 95 510 95')}${arrow('M440 156 H510')}
  ${arrow('M440 166 C480 166 480 219 510 219')}${arrow('M440 176 C470 176 470 287 510 287')}
  ${arrow('M690 33 C710 33 705 85 725 85')}${arrow('M690 157 C710 157 705 105 725 105')}
  ${arrow('M380 186 V335 H800 V206')}
  </svg>`;
}

route('architecture', async () => {
  const groups = await api('/modules');
  app().innerHTML = `
  <h1>Architecture</h1>
  <p class="lead">main.py detects the hardware (CUDA, MPS or CPU), loads config.yaml, shows the login window and opens
    the main window. Processing runs on a QThread that calls the models and emits results to the visualisation widget.</p>
  <div class="panel">${diagram()}</div>
  <div class="two">
    <div class="panel"><h3>An image</h3><ol class="steps">
      <li>cv2.imread the file.</li><li><code>detector.detect(image)</code>: boxes, scores, classes, names, centres.</li>
      <li><code>segmentor.segment(image, boxes)</code> when there are detections: masks and contours.</li>
      <li>Optional <code>llm_analyzer.analyze_objects</code>: one description per detection crop.</li>
      <li>Emit to the visualisation; save the annotated image and a JSON of the detections to output/.</li></ol></div>
    <div class="panel"><h3>A video, frame by frame</h3><ol class="steps">
      <li>Read the next frame (cv2.VideoCapture).</li><li><code>detector.detect(frame)</code>.</li>
      <li><code>tracker.update(detections)</code>: tracks and the active track IDs.</li>
      <li><code>segmentor.segment(frame, boxes)</code>.</li><li>LLM descriptions every 30th frame, if enabled.</li>
      <li>Emit, write the annotated frame to an XVID video, update the progress bar.</li></ol></div>
  </div>
  <h2>Modules</h2>
  <p class="mute">${groups.reduce((s, g) => s + g.modules.length, 0)} modules, read from the source code.</p>
  ${groups.map(g => `<h3>${esc(g.title)} <small class="mute">${esc(g.package)}</small></h3>
    ${g.modules.map(m => `<div class="mod"><code>${esc(m.file)}</code> <small>${m.lines} lines</small>
      ${m.doc ? `<div class="mute">${esc(m.doc)}</div>` : ''}
      <div>${m.classes.map(c => `<span class="tag a" title="${esc(c.methods.join(', '))}">${esc(c.name)}${c.bases.length ? `(${esc(c.bases.join(', '))})` : ''}</span>`).join(' ')}
        ${m.functions.map(f => `<span class="tag">${esc(f)}()</span>`).join(' ')}</div></div>`).join('')}`).join('')}`;
});
