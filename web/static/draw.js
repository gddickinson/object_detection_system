// Overlay drawing, following the desktop GUI (gui/visualization.py): boxes in the class colour with a filled
// "name: score" label above, a red centre dot with its (x, y), segmentation fills (alpha 0.3) with outlines, and for
// tracks "ID: n" above the box with the last 20 centres as a trajectory. Sizes scale with the image width.
import { classColor } from './track_engine.js';

export const trackColor = id => `hsl(${(id * 137.508) % 360} 85% 60%)`;
const segColor = i => `hsl(${(i * 97 + 30) % 360} 80% 55%)`;

export function scaleOf(width) { return Math.max(1, width / 640); }

export function drawSegments(g, contours, k = 1) {
  contours.forEach((c, i) => {
    if (!c || c.length < 3) return;
    g.beginPath(); g.moveTo(c[0][0], c[0][1]);
    for (let j = 1; j < c.length; j++) g.lineTo(c[j][0], c[j][1]);
    g.closePath();
    g.fillStyle = segColor(i); g.globalAlpha = 0.3; g.fill(); g.globalAlpha = 1;
    g.strokeStyle = segColor(i); g.lineWidth = 2 * k; g.stroke();
  });
}

function label(g, text, x, y, bg, k) {
  g.font = `${Math.round(13 * k)}px system-ui, sans-serif`;
  const w = g.measureText(text).width + 6 * k, h = 18 * k;
  const top = y - h < 0 ? y : y - h;
  g.fillStyle = bg; g.fillRect(x, top, w, h);
  g.fillStyle = '#fff'; g.textBaseline = 'middle'; g.fillText(text, x + 3 * k, top + h / 2);
}

// dets: [{box, score, class_name, center?, label?}]
export function drawDetections(g, dets, { labels = true, centers = true, k = 1, color = null } = {}) {
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.box;
    const c = color ? color(d) : classColor(d.class_name);
    g.strokeStyle = c; g.lineWidth = 2 * k; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (labels) label(g, d.label || `${d.class_name}: ${d.score.toFixed(2)}`, x1, y1, c, k);
    if (centers) {
      const [cx, cy] = d.center || [(x1 + x2) / 2, (y1 + y2) / 2];
      g.fillStyle = '#ff2020'; g.beginPath(); g.arc(cx, cy, 4 * k, 0, 2 * Math.PI); g.fill();
      if (labels) {
        g.font = `${Math.round(12 * k)}px ui-monospace, monospace`; g.textBaseline = 'middle';
        g.fillText(`(${Math.trunc(cx)}, ${Math.trunc(cy)})`, cx + 8 * k, cy);
      }
    }
  }
}

// tracks: [{id, box, trail: [[x, y], ...]}]
export function drawTracks(g, tracks, { labels = true, k = 1 } = {}) {
  for (const t of tracks) {
    const c = trackColor(t.id);
    const [x1, y1, x2, y2] = t.box;
    g.strokeStyle = c; g.lineWidth = 2 * k; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (t.trail && t.trail.length > 1) {
      g.beginPath(); g.moveTo(t.trail[0][0], t.trail[0][1]);
      for (const p of t.trail.slice(1)) g.lineTo(p[0], p[1]);
      g.stroke();
    }
    if (labels) {
      g.font = `bold ${Math.round(13 * k)}px ui-monospace, monospace`; g.fillStyle = c; g.textBaseline = 'bottom';
      g.fillText(`ID: ${t.id}`, x1, Math.max(14 * k, y1 - 20 * k));
    }
  }
}
