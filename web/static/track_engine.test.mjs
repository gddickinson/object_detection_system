// node web/static/track_engine.test.mjs: the JS port must reproduce tests/fixtures/tracker_cases.json
// (written by scripts/make_tracker_fixture.py from the repo's Python code).
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { boxIou, classColorBgr, runTracker } from './track_engine.js';

const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/tracker_cases.json', import.meta.url)));
let n = 0;
for (const c of fx.iou) { assert.ok(Math.abs(boxIou(c.a, c.b) - c.iou) < 1e-12, JSON.stringify(c)); n++; }
for (const c of fx.colors) { assert.deepEqual(classColorBgr(c.name), c.bgr, c.name); n++; }
for (const c of fx.tracker) {
  for (const py of c.runs) {
    const js = runTracker(c.sequence.frames, py.track_buffer, py.match_thresh);
    assert.equal(js.n_tracks, py.n_tracks);
    assert.equal(js.id_changes, py.id_changes);
    py.frames.forEach((f, i) => {
      assert.deepEqual(js.frames[i].active, f.active, `frame ${i}`);
      assert.deepEqual(js.frames[i].det_track, f.det_track, `frame ${i}`);
    });
    n++;
  }
}
console.log(`track_engine: ${n} cases match the Python output`);
