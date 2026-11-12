"""Per-class box colours that stay the same from run to run (and match the web front end's draw.js)."""
import colorsys
import zlib


def class_color_bgr(class_name):
    """BGR colour for a class name: hue from crc32(name) % 255, saturation and value 0.8."""
    hash_val = zlib.crc32(str(class_name).encode("utf-8")) % 255
    r, g, b = colorsys.hsv_to_rgb(hash_val / 255.0, 0.8, 0.8)
    return (int(b * 255), int(g * 255), int(r * 255))
