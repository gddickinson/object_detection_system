// Static-site answers for api(path, body): GETs come from the JSON exported at build time (scripts/build_static.py);
// model inference needs the local Python app.
export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\//, '');
  if (body) throw new Error('Running the models needs the local app (python run_web.py); see About.');
  if (clean === 'status') return { static: true, ready: false, packages: {}, files: {} };
  const r = await fetch(`api/${clean}.json`);
  if (!r.ok) throw new Error(`${r.status}: api/${clean}.json is not part of the static site`);
  return r.json();
}
