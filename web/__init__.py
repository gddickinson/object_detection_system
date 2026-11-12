"""Web interface (FastAPI + plain ES-module front end) for the Object Detection and Tracking System.

Contents:
- app.py        routes under /api and STATIC_ENDPOINTS (exported for the static site)
- catalog.py    what the repo contains: back ends, config.yaml, labels, modules (ast), counts
- synthetic.py  synthetic detection sequences + run_tracker (the repo's simple IoU tracker)
- demos.py      tracker demo (live) and segmentation demo (precomputed JSON in demo/)
- inference.py  local-only detection/segmentation/tracking with the repo's model classes (lazy torch import)
- static/       front end (see INTERFACE.md)
"""
