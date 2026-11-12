// Pure JS port of the repo's post-processing (no DOM): box IoU and the simple IoU tracker
// (postprocess/iou_tracker.py), the per-class colour (postprocess/colors.py) and the run/identity bookkeeping of
// web/synthetic.run_tracker. Tested against Python output by track_engine.test.mjs.

export function boxIou(a, b) {
  const xl = Math.max(a[0], b[0]), yt = Math.max(a[1], b[1]);
  const xr = Math.min(a[2], b[2]), yb = Math.min(a[3], b[3]);
  if (xr < xl || yb < yt) return 0;
  const inter = (xr - xl) * (yb - yt);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

// tracks: Map id -> {boxes, cls, score, first, last}. Mutated in place; returns {active, nextId, detTrack}.
export function associateSimple(tracks, boxes, scores, classes, frame, nextId, buffer, thresh) {
  const detTrack = new Array(boxes.length).fill(0);
  if (!boxes.length) return { active: [], nextId, detTrack };
  const active = [], matched = new Set();
  for (const [id, t] of tracks) {
    if (frame - t.last > buffer || !t.boxes.length) continue;
    const last = t.boxes[t.boxes.length - 1];
    let best = thresh, bi = -1;
    for (let i = 0; i < boxes.length; i++) {
      if (matched.has(i) || t.cls !== classes[i]) continue;
      const v = boxIou(last, boxes[i]);
      if (v > best) { best = v; bi = i; }
    }
    if (bi >= 0) { matched.add(bi); t.boxes.push(boxes[bi]); t.last = frame; active.push(id); detTrack[bi] = id; }
  }
  for (let i = 0; i < boxes.length; i++) {
    if (matched.has(i)) continue;
    tracks.set(nextId, { boxes: [boxes[i]], cls: classes[i], score: scores[i], first: frame, last: frame });
    active.push(nextId); detTrack[i] = nextId; nextId++;
  }
  return { active, nextId, detTrack };
}

// Run the tracker over frames [{boxes, scores, classes, gt?}] (frame numbers start at 1, as ObjectTracker counts).
export function runTracker(frames, buffer = 30, thresh = 0.8) {
  const tracks = new Map(), lastId = new Map(), out = [];
  let nextId = 1, changes = 0;
  frames.forEach((f, k) => {
    const r = associateSimple(tracks, f.boxes, f.scores, f.classes, k + 1, nextId, buffer, thresh);
    nextId = r.nextId;
    (f.gt || []).forEach((g, i) => {
      if (lastId.has(g) && lastId.get(g) !== r.detTrack[i]) changes++;
      lastId.set(g, r.detTrack[i]);
    });
    out.push({ active: r.active, det_track: r.detTrack });
  });
  return { track_buffer: buffer, match_thresh: thresh, frames: out, n_tracks: nextId - 1, id_changes: changes, tracks };
}

// --- class colours: zlib.crc32(name) % 255 -> HSV hue, s = v = 0.8 (colorsys.hsv_to_rgb), as BGR like OpenCV ---
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
export function crc32(str) {
  let c = 0xFFFFFFFF;
  for (const b of new TextEncoder().encode(str)) c = CRC[(c ^ b) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
function hsvToRgb(h, s, v) {  // Python colorsys.hsv_to_rgb
  if (s === 0) return [v, v, v];
  let i = Math.floor(h * 6); const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
  i %= 6;
  return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i];
}
export function classColorBgr(name) {
  const [r, g, b] = hsvToRgb((crc32(String(name)) % 255) / 255, 0.8, 0.8);
  return [Math.trunc(b * 255), Math.trunc(g * 255), Math.trunc(r * 255)];
}
export function classColor(name) {
  const [b, g, r] = classColorBgr(name);
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}
