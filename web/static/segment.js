// Segment: the OpenCV fallback of models/segmentor.py on a synthetic scene (precomputed by scripts/make_segment_demo.py).
import { $, api, app, esc, route } from './core.js';
import { drawDetections, drawSegments } from './draw.js';

route('segment', async () => {
  const d = await api('/demo/segment');
  const show = { seg: true, boxes: true, labels: true };
  app().innerHTML = `
  <h1>Segment</h1>
  <p class="lead">Without a SAM checkpoint, <code>ObjectSegmentor</code> falls back to OpenCV inside each detection box:
    grayscale, 5 x 5 Gaussian blur, adaptive Gaussian threshold (block 11, C 2, inverted) and the largest external
    contour. Below is that code's output on a synthetic scene of five filled shapes, with boxes 10 px larger than each
    shape standing in for detections.</p>
  <div class="vis">
    <div><div class="viewer"><img id="im" src="${esc(d.image)}" alt="Synthetic scene: an ellipse, a rotated rectangle, a disc, a triangle and a blob on a textured background" width="${d.width}" height="${d.height}">
      <canvas class="over" id="cv" width="${d.width}" height="${d.height}" aria-hidden="true"></canvas></div>
      <div class="toggles"><label><input type="checkbox" id="t-seg" checked> masks and outlines</label>
        <label><input type="checkbox" id="t-boxes" checked> boxes</label><label><input type="checkbox" id="t-labels" checked> labels</label></div></div>
    <div class="panel"><h3>Result</h3><p class="mute">${esc(d.method)}, computed by the Python code; image
      web/static/demo/segment_scene.png (${d.width} x ${d.height}).</p>
      <div class="scroll"><table class="tbl"><tr><th>object</th><th>box px²</th><th>mask px²</th><th>mask / box</th><th>outline pts</th></tr>
      ${d.objects.map(o => `<tr><td>${esc(o.name)}</td><td>${o.box_area}</td><td>${o.mask_area}</td>
        <td>${(o.mask_area / o.box_area).toFixed(2)}</td><td>${o.contour.length}</td></tr>`).join('')}</table></div>
      <p class="mute">On these flat, high-contrast shapes the outline follows each shape. SAM, when its checkpoint is
        installed, is prompted with the same boxes.</p></div>
  </div>`;
  const g = $('#cv').getContext('2d');
  const draw = () => {
    g.clearRect(0, 0, d.width, d.height);
    if (show.seg) drawSegments(g, d.objects.map(o => o.contour));
    if (show.boxes) drawDetections(g, d.objects.map(o => ({ box: o.box, class_name: o.name, label: o.name, score: 1 })),
      { labels: show.labels, centers: show.labels });
  };
  for (const k of Object.keys(show)) $('#t-' + k).onchange = e => { show[k] = e.target.checked; draw(); };
  draw();
});
