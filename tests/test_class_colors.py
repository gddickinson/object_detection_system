"""Class colours must be the same in every run (regression: get_color_for_class used Python's hash(), which is salted
per process, so 'person' got a different colour each time the app started)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from postprocess.colors import class_color_bgr

ROOT = Path(__file__).resolve().parent.parent

PROBE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('vis', sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print([m.VisualizationWidget.get_color_for_class(None, n) for n in ('person', 'cat', 'potted plant')])
"""


def test_known_values():
    # zlib.crc32 of the UTF-8 name, mod 255, as the HSV hue (s = v = 0.8), returned in BGR order
    assert class_color_bgr("person") == (56, 204, 40)
    assert len({class_color_bgr(n) for n in ("person", "cat", "dog", "car")}) == 4


@pytest.mark.parametrize("module", ["gui/visualization.py", "utils/visualization.py"])
def test_gui_colours_stable_across_runs(module):
    pytest.importorskip("PyQt5.QtWidgets")
    pytest.importorskip("cv2")
    outs = set()
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
        r = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / module)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1
    assert outs.pop() == str([class_color_bgr(n) for n in ("person", "cat", "potted plant")])
