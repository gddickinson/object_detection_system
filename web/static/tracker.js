// Tracker: the repo's simple IoU tracker (track_engine.js, a port of postprocess/iou_tracker.py) on a synthetic scene.
import { $, animate, api, app, esc, route, slider } from './core.js';
import { drawDetections, drawTracks, trackColor } from './draw.js';
import { backdrop, makeSequence } from './scene.js';
import { runTracker } from './track_engine.js';

route('tracker', async () => {
  const demo = await api('/demo/tracker');
  const cfg = demo.config;
  const st = { thresh: cfg.simple_iou_thresh, buffer: cfg.track_buffer, scene: 'py', frame: 0, playing: true,
    labels: true, trails: true, js: { n_objects: 8, speed: 5, jitter: 2, miss_rate: 0.1, seed: 1 } };
  app().innerHTML = `
  <h1>Tracker</h1>
  <p class="lead">Each frame's detections go to the same greedy IoU tracker the desktop app runs: a live track takes the
    unmatched same-class detection whose IoU with its last box is highest and <b>above the match threshold</b>; the rest
    start new tracks; a track stays live for <b>buffer</b> frames. Boxes are coloured by track ID. The scene is
    synthetic: ground truth is known, so identity changes can be counted.</p>
  <div class="callout">config.yaml selects <code>type: ${esc(cfg.type)}</code> (ultralytics' ByteTrack, local app). This
    simple tracker runs for <code>type: simple</code> and whenever ByteTrack or SORT cannot be created; its minimum IoU is
    <code>simple_iou_thresh: ${cfg.simple_iou_thresh}</code>, buffer <code>track_buffer: ${cfg.track_buffer}</code>. A high
    threshold such as 0.8 breaks moving objects' identities almost every frame.</div>
  <div class="vis">
    <div>
      <div class="stagebox"><canvas id="cv" width="640" height="360" role="img" aria-label="Synthetic scene with tracked detection boxes"></canvas></div>
      <div class="playbar"><button id="play" class="alt">Pause</button>
        <label for="fr" class="mute">Frame <b id="frl">1</b></label><input type="range" id="fr" min="0" value="0"></div>
      <div class="stats" id="stats"></div>
    </div>
    <div class="panel form">
      <fieldset><legend>Tracker</legend><div id="tctl"></div>
        <div class="toggles"><label><input type="checkbox" id="lab" checked> labels</label>
          <label><input type="checkbox" id="trl" checked> trajectories</label></div></fieldset>
      <fieldset><legend>Scene</legend>
        <div class="seg" role="group" aria-label="Scene"><button data-s="py" class="on">Python reference</button><button data-s="js">Random (JS)</button></div>
        <p class="hint" id="shint"></p><div id="sctl"></div></fieldset>
    </div>
  </div>
  <h2>Python reference</h2>
  <p>The Python code (postprocess/iou_tracker.py) tracked the reference sequence with these settings when the data were
    exported; your browser re-runs the JS port on the same detections.</p>
  <div class="scroll"><table class="tbl" id="ref"></table></div>`;

  const cv = $('#cv'), g = cv.getContext('2d'), fr = $('#fr');
  let seq, run, hist;
  const compute = () => {
    seq = st.scene === 'py' ? demo.sequence : makeSequence({ ...st.js, n_frames: 90 });
    run = runTracker(seq.frames, st.buffer, st.thresh);
    hist = new Map();  // track id -> [[frame, cx, cy]]
    run.frames.forEach((f, k) => f.det_track.forEach((id, i) => {
      const b = seq.frames[k].boxes[i];
      if (!hist.has(id)) hist.set(id, []);
      hist.get(id).push([k, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2]);
    }));
    fr.max = seq.frames.length - 1; st.frame = Math.min(st.frame, seq.frames.length - 1);
    const nd = seq.frames.reduce((s, f) => s + f.boxes.length, 0);
    $('#stats').innerHTML = [[seq.params.n_objects, 'true objects'], [run.n_tracks, 'tracks started'],
      [run.id_changes, 'identity changes'], [nd, 'detections'], [seq.frames.length, 'frames']]
      .map(([v, l]) => `<div><b>${v}</b><span>${l}</span></div>`).join('');
    $('#shint').textContent = st.scene === 'py'
      ? `web/synthetic.py, seed ${seq.params.seed}: ${seq.params.n_objects} objects, speed ${seq.params.speed} px/frame, jitter ${seq.params.jitter} px, miss rate ${seq.params.miss_rate}.`
      : 'Generated in your browser with the same motion model (different random numbers).';
    $('#sctl').classList.toggle('hide', st.scene === 'py');
    draw();
  };
  const draw = () => {
    const k = st.frame, f = seq.frames[k], ids = run.frames[k].det_track;
    backdrop(g, cv.width, cv.height);
    const dets = f.boxes.map((b, i) => ({ box: b, score: f.scores[i], class_name: seq.names[f.classes[i]] || String(f.classes[i]),
      label: `${seq.names[f.classes[i]]}: ${f.scores[i].toFixed(2)}`, id: ids[i] }));
    const tracks = dets.map(d => ({ id: d.id, box: d.box,
      trail: st.trails ? hist.get(d.id).filter(p => p[0] <= k).slice(-20).map(p => [p[1], p[2]]) : null }));
    drawTracks(g, tracks, { labels: st.labels });
    drawDetections(g, dets, { labels: st.labels, centers: true, color: d => trackColor(d.id) });
    $('#frl').textContent = `${k + 1} / ${seq.frames.length}`; fr.value = k;
  };

  slider($('#tctl'), { id: 'th', label: 'Match threshold (min IoU)', min: 0.05, max: 0.95, step: 0.05, value: st.thresh, fmt: v => v.toFixed(2) },
    v => { st.thresh = v; compute(); });
  slider($('#tctl'), { id: 'bu', label: 'Track buffer (frames)', min: 0, max: 60, value: st.buffer }, v => { st.buffer = v; compute(); });
  const sc = $('#sctl');
  slider(sc, { id: 'no', label: 'Objects', min: 1, max: 16, value: st.js.n_objects }, v => { st.js.n_objects = v; compute(); });
  slider(sc, { id: 'sp', label: 'Speed', min: 1, max: 14, step: 0.5, value: st.js.speed, unit: ' px/frame' }, v => { st.js.speed = v; compute(); });
  slider(sc, { id: 'ji', label: 'Box jitter', min: 0, max: 8, step: 0.5, value: st.js.jitter, unit: ' px' }, v => { st.js.jitter = v; compute(); });
  slider(sc, { id: 'mr', label: 'Miss rate', min: 0, max: 0.6, step: 0.05, value: st.js.miss_rate, fmt: v => v.toFixed(2) }, v => { st.js.miss_rate = v; compute(); });
  const nb = document.createElement('button'); nb.textContent = 'New random scene'; nb.className = 'alt';
  nb.onclick = () => { st.js.seed++; compute(); }; sc.appendChild(nb);
  document.querySelectorAll('.seg button').forEach(b => b.onclick = () => {
    document.querySelectorAll('.seg button').forEach(x => x.classList.toggle('on', x === b)); st.scene = b.dataset.s; compute();
  });
  $('#lab').onchange = e => { st.labels = e.target.checked; draw(); };
  $('#trl').onchange = e => { st.trails = e.target.checked; draw(); };
  fr.oninput = () => { st.frame = +fr.value; draw(); };
  $('#play').onclick = () => { st.playing = !st.playing; $('#play').textContent = st.playing ? 'Pause' : 'Play'; };
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { st.playing = false; $('#play').textContent = 'Play'; }

  $('#ref').innerHTML = '<tr><th>match threshold</th><th>buffer</th><th>tracks (Python)</th><th>identity changes (Python)</th><th>JS port</th></tr>'
    + demo.runs.map(py => {
      const js = runTracker(demo.sequence.frames, py.track_buffer, py.match_thresh);
      const same = js.n_tracks === py.n_tracks && js.id_changes === py.id_changes
        && py.frames.every((f, i) => f.det_track.join() === js.frames[i].det_track.join());
      return `<tr><td>${py.match_thresh}${py.match_thresh === cfg.simple_iou_thresh ? ' (config)' : ''}</td><td>${py.track_buffer}</td>
        <td>${py.n_tracks}</td><td>${py.id_changes}</td><td><span class="badge ${same ? 'ok' : 'no'}">${same ? 'identical, every frame' : 'differs'}</span></td></tr>`;
    }).join('');

  compute();
  let last = 0;
  animate(t => { if (st.playing && t - last > 110) { last = t; st.frame = (st.frame + 1) % seq.frames.length; draw(); } });
});
