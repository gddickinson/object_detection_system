// Synthetic detection sequences in the browser: the same model as web/synthetic.make_sequence (objects of four COCO
// classes at constant velocity bouncing off the edges, Gaussian corner jitter, random scores, missed detections), but
// with JS random numbers, so a given seed differs from the Python one. Also draws a simple scene behind the boxes.
import { rng } from './core.js';

export const KINDS = [[0, 'person', 34, 80], [2, 'car', 90, 46], [16, 'dog', 52, 36], [1, 'bicycle', 60, 44]];

export function makeSequence({ n_objects = 6, n_frames = 60, width = 640, height = 360, speed = 4, jitter = 2,
  miss_rate = 0.1, seed = 1 } = {}) {
  const r = rng(seed * 7919 + 13);
  const gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const obj = Array.from({ length: n_objects }, (_, i) => {
    const k = KINDS[i % KINDS.length], a = r() * 2 * Math.PI, s = speed * (0.5 + r());
    return { cls: k[0], w: k[2], h: k[3], x: r() * (width - k[2]), y: r() * (height - k[3]), vx: Math.cos(a) * s, vy: Math.sin(a) * s };
  });
  const frames = [];
  for (let f = 0; f < n_frames; f++) {
    const fr = { boxes: [], scores: [], classes: [], gt: [] };
    obj.forEach((o, i) => {
      o.x += o.vx; o.y += o.vy;
      if (o.x < 0 || o.x + o.w > width) o.vx *= -1;
      if (o.y < 0 || o.y + o.h > height) o.vy *= -1;
      o.x = Math.min(Math.max(o.x, 0), width - o.w); o.y = Math.min(Math.max(o.y, 0), height - o.h);
      const seen = r() >= miss_rate, n = [gauss(), gauss(), gauss(), gauss()].map(v => v * jitter), sc = 0.5 + 0.45 * r();
      if (!seen) return;
      fr.boxes.push([o.x + n[0], o.y + n[1], o.x + o.w + n[2], o.y + o.h + n[3]].map(v => Math.round(v * 100) / 100));
      fr.scores.push(Math.round(sc * 1000) / 1000); fr.classes.push(o.cls); fr.gt.push(i);
    });
    frames.push(fr);
  }
  return { params: { n_objects, n_frames, width, height, speed, jitter, miss_rate, seed },
    names: Object.fromEntries(KINDS.map(k => [String(k[0]), k[1]])), frames };
}

// A plain backdrop (sky, ground) so the boxes read as a scene; objects themselves are not drawn: only detections.
export function backdrop(g, w, h) {
  const sky = g.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#0d1630'); sky.addColorStop(0.55, '#16234a'); sky.addColorStop(0.551, '#13261d'); sky.addColorStop(1, '#0a160f');
  g.fillStyle = sky; g.fillRect(0, 0, w, h);
  g.strokeStyle = 'rgba(255,255,255,.06)'; g.lineWidth = 1;
  for (let x = 0; x <= w; x += 40) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
  for (let y = 0; y <= h; y += 40) { g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
}
