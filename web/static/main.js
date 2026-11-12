// Entry point: lazy page modules, the mode label, a subtle "detection boxes" background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = {
  'home.js': ['home'], 'detect.js': ['detect'], 'tracker.js': ['tracker'], 'segment.js': ['segment'],
  'arch.js': ['architecture'], 'models.js': ['models'], 'about.js': ['about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: no models run here; the tracker runs in your browser (JS port tested against the Python code)'
  : 'local app: detection, segmentation and tracking run in Python on this computer';

// Background: a few faint boxes drifting slowly, like tracked detections. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const cols = ['#39ff9f', '#ff5ca8', '#ffd23f'];
  const bs = Array.from({ length: 9 }, (_, i) => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight,
    w: 40 + Math.random() * 90, h: 30 + Math.random() * 80, vx: (Math.random() - 0.5) * 0.6,
    vy: (Math.random() - 0.5) * 0.6, c: cols[i % 3] }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 50) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    g.globalAlpha = 0.16; g.lineWidth = 1.5;
    for (const b of bs) {
      b.x = (b.x + b.vx + canvas.width) % canvas.width; b.y = (b.y + b.vy + canvas.height) % canvas.height;
      g.strokeStyle = b.c; g.strokeRect(b.x, b.y, b.w, b.h);
      g.fillStyle = b.c; g.fillRect(b.x + b.w / 2 - 2, b.y + b.h / 2 - 2, 4, 4);
    }
    g.globalAlpha = 1;
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
