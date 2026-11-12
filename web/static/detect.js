// Detect: upload an image or video to the local app; the repo's models run in Python and the overlays are drawn here
// (draw.js, following gui/visualization.py). On the static site this page explains what the local app does.
import { $, animate, api, app, download, esc, isStatic, needsServer, route, slider } from './core.js';
import { drawDetections, drawSegments, drawTracks, scaleOf } from './draw.js';
import { classColor } from './track_engine.js';

const IMG = /\.(jpe?g|png|bmp|tiff?|webp)$/i, VID = /\.(mp4|avi|mov|mkv|webm|m4v)$/i;

route('detect', async () => {
  if (isStatic()) {
    app().innerHTML = `<h1>Detect</h1>${needsServer('Detection, segmentation and video tracking')}
      <p class="lead">In the local app you choose an image or a video, a confidence threshold and a back end (Core ML or
        YOLOv8 PyTorch, or "auto" as the desktop app picks), and the repo's own model classes run on your computer. The
        page then draws the GUI's overlays: class-coloured boxes with "name: score", centre points, segmentation outlines
        (SAM, or the OpenCV fallback) and, for videos, track IDs with trajectories; the results JSON downloads in the
        desktop app's format.</p>
      <p>Try the parts that run without the models: the <a href="#/tracker">tracker demo</a> and the
        <a href="#/segment">segmentation demo</a>.</p>`;
    return;
  }
  const status = await api('/status', { fresh: true });
  const st = { conf: 0.5, backend: 'auto', segment: true, labels: true, centers: true, maxFrames: 300, stride: 1, file: null, res: null };
  const pk = Object.entries(status.packages).map(([k, v]) => `<span class="badge ${v ? 'ok' : 'no'}">${esc(k)}</span>`).join(' ');
  const fl = Object.entries(status.files).map(([k, v]) => `<span class="badge ${v ? 'ok' : 'no'}">${esc(k)}</span>`).join(' ');
  app().innerHTML = `
  <h1>Detect</h1>
  <div class="vis">
    <div id="out"><p class="mute">Choose an image or a video, or run the example image.</p></div>
    <div class="panel form">
      <fieldset><legend>Input</legend>
        <label for="file">Image or video</label><input type="file" id="file" accept="image/*,video/*,.avi,.mkv">
        <button id="ex" class="alt">Use data/examples/cats.jpeg</button></fieldset>
      <fieldset><legend>Model</legend>
        <label for="be">Detector back end</label><select id="be">${status.backends.map(b => `<option>${b}</option>`).join('')}</select>
        <p class="hint">auto = create_detector (Core ML on Apple Silicon with coremltools, else YOLOv8).</p>
        <div id="cctl"></div>
        <label><input type="checkbox" id="sg" checked> segment (images)</label>
        <div id="vctl"></div></fieldset>
      <button id="run" disabled>Run</button>
      <p class="hint" id="msg"></p>
      <fieldset><legend>Display</legend><div class="toggles">
        <label><input type="checkbox" id="lb" checked> labels</label><label><input type="checkbox" id="ce" checked> centres</label></div></fieldset>
      <fieldset><legend>This computer</legend><p>${pk}</p><p>${fl}</p>
        ${status.ready ? '' : '<p class="err">Inference needs torch, ultralytics and opencv-python (pip install -r requirements.txt).</p>'}</fieldset>
    </div>
  </div>`;
  slider($('#cctl'), { id: 'cf', label: 'Confidence threshold', min: 0.05, max: 0.95, step: 0.05, value: st.conf, fmt: v => v.toFixed(2) }, v => { st.conf = v; });
  slider($('#vctl'), { id: 'mf', label: 'Video: max frames', min: 10, max: 1500, step: 10, value: st.maxFrames }, v => { st.maxFrames = v; });
  slider($('#vctl'), { id: 'sd', label: 'Video: every nth frame', min: 1, max: 10, value: st.stride }, v => { st.stride = v; });
  $('#be').onchange = e => { st.backend = e.target.value; };
  $('#sg').onchange = e => { st.segment = e.target.checked; };
  $('#lb').onchange = e => { st.labels = e.target.checked; redraw(); };
  $('#ce').onchange = e => { st.centers = e.target.checked; redraw(); };
  $('#file').onchange = e => { st.file = e.target.files[0] || null; $('#run').disabled = !st.file; $('#msg').textContent = st.file ? st.file.name : ''; };
  $('#ex').onclick = () => { st.file = null; run(true); };
  $('#run').onclick = () => run(false);
  let redraw = () => {};

  async function run(example) {
    const isVid = !example && VID.test(st.file.name);
    if (!example && !isVid && !IMG.test(st.file.name)) { $('#msg').textContent = 'Unsupported file type.'; return; }
    const fd = new FormData();
    if (example) fd.append('example', 'cats'); else fd.append('file', st.file);
    fd.append('confidence', st.conf); fd.append('backend', st.backend); fd.append('segment', st.segment ? '1' : '0');
    if (isVid) { fd.append('max_frames', st.maxFrames); fd.append('stride', st.stride); }
    $('#run').disabled = true; $('#msg').textContent = 'Running (the first run loads the model)...';
    try {
      const res = await api(isVid ? '/detect_video' : '/detect', { body: fd });
      const src = example ? 'api/example.jpg' : URL.createObjectURL(st.file);
      $('#msg').textContent = '';
      isVid ? showVideo(res, src) : showImage(res, src, example ? 'cats.jpeg' : st.file.name);
    } catch (e) { $('#msg').innerHTML = `<span class="err">${esc(e.message)}</span>`; }
    $('#run').disabled = !st.file;
  }

  function table(dets) {
    return `<div class="scroll"><table class="tbl"><tr><th>#</th><th>class</th><th>score</th><th>box (x1, y1, x2, y2)</th></tr>
      ${dets.map((d, i) => `<tr><td>${i + 1}</td><td><i class="sw" style="background:${classColor(d.class_name)}"></i>${esc(d.class_name)}</td>
        <td>${d.score.toFixed(3)}</td><td>${d.box.map(v => v.toFixed(0)).join(', ')}</td></tr>`).join('')}</table></div>`;
  }

  function showImage(r, src, name) {
    $('#out').innerHTML = `<div class="viewer"><img id="im" src="${src}" alt="Input image ${esc(name)}" width="${r.width}" height="${r.height}">
      <canvas class="over" id="cv" width="${r.width}" height="${r.height}" aria-hidden="true"></canvas></div>
      <p>${r.detections.length} detections · ${esc(r.backend)} · segmentation: ${esc(r.segmentor)} · ${r.ms.detect} ms detect, ${r.ms.segment} ms segment
        <button class="alt" id="dl">Download results JSON</button></p>${table(r.detections)}`;
    const g = $('#cv').getContext('2d'), k = scaleOf(r.width);
    redraw = () => { g.clearRect(0, 0, r.width, r.height); drawSegments(g, r.contours, k);
      drawDetections(g, r.detections, { labels: st.labels, centers: st.centers, k }); };
    redraw();
    $('#dl').onclick = () => download(name.replace(/\.[^.]+$/, '') + '.json', JSON.stringify(r.results_json, null, 2), 'application/json');
  }

  function showVideo(r, src) {
    $('#out').innerHTML = `<div class="viewer"><video id="vd" src="${src}" controls muted playsinline></video>
      <canvas class="over" id="cv" width="${r.width}" height="${r.height}" aria-hidden="true"></canvas></div>
      <p>${r.processed} of ${r.total_frames} frames processed (every ${r.stride}) in ${r.seconds} s · ${esc(r.backend)} ·
        tracker: ${esc(r.tracker)} · ${r.n_tracks} tracks <button class="alt" id="dl">Download frames JSON</button></p>
      <p class="mute">Overlays follow the video's current frame; the browser must be able to play this format.</p><div id="ft"></div>`;
    const vd = $('#vd'), g = $('#cv').getContext('2d'), k = scaleOf(r.width);
    const byIndex = new Map(r.frames.map(f => [f.index, f]));
    let shown = -1;
    redraw = () => {
      let i = Math.floor(vd.currentTime * r.fps + 1e-6);
      i -= i % r.stride;
      const f = byIndex.get(i);
      g.clearRect(0, 0, r.width, r.height);
      if (!f) return;
      drawTracks(g, f.tracks, { labels: st.labels, k });
      drawDetections(g, f.detections, { labels: st.labels, centers: st.centers, k });
      if (i !== shown) { shown = i; $('#ft').innerHTML = `<h3>Frame ${i + 1}</h3>${table(f.detections)}`; }
    };
    animate(() => redraw());
    $('#dl').onclick = () => download('tracks.json', JSON.stringify(r, null, 1), 'application/json');
  }
});
