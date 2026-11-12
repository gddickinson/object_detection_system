// Models: back ends as implemented in models/, config.yaml settings, the class list with its box colours.
import { api, app, esc, route } from './core.js';

const size = b => b == null ? '<span class="badge no">not in repo</span>' : `<span class="badge ok">${(b / 1e6).toFixed(1)} MB in repo</span>`;
const card = x => `<div class="card"><b>${esc(x.name)}</b><small>${esc(x.cls || '')} · ${esc(x.source)}${x.library ? ' · ' + esc(x.library) : ''}</small>
  <span>${esc(x.how || '')}</span>${x.chosen ? `<small>${esc(x.chosen)}</small>` : ''}
  ${x.weights ? `<small>${esc(x.weights)} ${size(x.weights_bytes)}</small>` : ''}</div>`;

function kv(obj, prefix = '') {
  return Object.entries(obj || {}).map(([k, v]) => v && typeof v === 'object' && !Array.isArray(v)
    ? kv(v, prefix + k + '.') : `<dt>${esc(prefix + k)}</dt><dd>${esc(JSON.stringify(v))}</dd>`).join('');
}

route('models', async () => {
  const [m, labels] = await Promise.all([api('/models'), api('/labels')]);
  app().innerHTML = `
  <h1>Models</h1>
  <p class="lead">Every model sits behind an abstract base class in models/base.py (<code>BaseDetector.detect</code>,
    <code>BaseSegmentor.segment</code>, <code>BaseTracker.update</code>), so the GUI does not depend on the back end.
    The models, weights and libraries are third-party and have their own licences; model weights are not part of this web
    site.</p>
  <h2>Detectors</h2><div class="grid wide">${m.detectors.map(card).join('')}</div>
  <h2>Segmentors</h2><div class="grid wide">${m.segmentors.map(card).join('')}</div>
  <h2>Trackers</h2><div class="grid wide">${m.trackers.map(card).join('')}</div>
  <h2>LLM descriptions</h2>
  <p>Optional: each detection crop is sent to an LLM for a short description (<code>LLMAnalyzer</code>), shown in place of
    the class name. Not used by the web app.</p>
  <div class="grid">${m.llms.map(card).join('')}</div>
  <h2>config.yaml: models</h2><div class="panel"><dl class="kv">${kv(m.config)}</dl></div>
  <h2>${labels.length} classes</h2>
  <p class="mute">data/models/labels.txt (COCO), with the box colour the GUI gives each class (hue from crc32 of the name).</p>
  <div class="classes">${labels.map(l => `<span title="class ${l.id}"><i class="sw" style="background:${l.color}"></i>${l.id} ${esc(l.name)}</span>`).join('')}</div>`;
});
