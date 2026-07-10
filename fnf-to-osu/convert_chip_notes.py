#!/usr/bin/env python3
"""Convert Psych Engine chip notes atlas into osu!mania-style assets.

What this script does:
1. Reads Sparrow XML subtextures from NOTE_assets-chip.xml.
2. Rebuilds trimmed frames using frameX/frameY/frameWidth/frameHeight when present.
3. Applies Psych Engine runtime RGBPalette recoloring per direction.
4. Fits and centers sprites to match current Circle Fullsize dimensions.
5. Generates circle-shaped osu!mania hit lighting.
6. Exports finished assets to /convert.

Psych Engine references used:
- backend/ClientPrefs.hx: default arrowRGB palette.
- shaders/RGBPalette.hx: recolor formula.
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, cast

from PIL import Image, ImageChops, ImageDraw, ImageFilter


# Lane palettes (AARRGGBB): (main, highlight, shadow)
ARROW_RGB: Dict[str, Tuple[int, int, int]] = {
    "left": (0xFFC24B99, 0xFFFFFFFF, 0xFF3C1F56),  # purple
    "down": (0xFF00FFFF, 0xFFFFFFFF, 0xFF1542B7),  # blue
    "up": (0xFF12FA05, 0xFFFFFFFF, 0xFF0A4447),  # green
    "right": (0xFFF9393F, 0xFFFFFFFF, 0xFF651038),  # red
    "center": (0xFFCDCDCD, 0xFFFFFFFF, 0xFF8A8A8A),  # light gray
    "left2": (0xFFFFFF00, 0xFFFFFFFF, 0xFF6A5C10),  # yellow
    "down2": (0xFFB84DFF, 0xFFFFFFFF, 0xFF44205F),  # purple
    "up2": (0xFFFF9A1F, 0xFFFFFFFF, 0xFF6B340A),  # orange
    "right2": (0xFF2349A6, 0xFFFFFFFF, 0xFF101F4A),  # dark blue
}


def argb_to_rgbf(c: int) -> Tuple[float, float, float]:
    r = (c >> 16) & 0xFF
    g = (c >> 8) & 0xFF
    b = c & 0xFF
    return (r / 255.0, g / 255.0, b / 255.0)


def recolor_rgbpalette(src: Image.Image, palette_triplet: Tuple[int, int, int]) -> Image.Image:
    """Apply Psych RGBPalette shader math:

    out.rgb = min(src.r * R + src.g * G + src.b * B, 1.0)
    out.a = src.a

    where R/G/B are target palette vectors.
    """
    src = src.convert("RGBA")
    r_vec = argb_to_rgbf(palette_triplet[0])
    g_vec = argb_to_rgbf(palette_triplet[1])
    b_vec = argb_to_rgbf(palette_triplet[2])

    out = Image.new("RGBA", src.size)
    width, height = src.size

    for y in range(height):
        for x in range(width):
            pr, pg, pb, pa = cast(tuple[int, int, int, int], src.getpixel((x, y)))
            rf = pr / 255.0
            gf = pg / 255.0
            bf = pb / 255.0

            nr = min(rf * r_vec[0] + gf * g_vec[0] + bf * b_vec[0], 1.0)
            ng = min(rf * r_vec[1] + gf * g_vec[1] + bf * b_vec[1], 1.0)
            nb = min(rf * r_vec[2] + gf * g_vec[2] + bf * b_vec[2], 1.0)

            out.putpixel((x, y), (int(round(nr * 255)), int(round(ng * 255)), int(round(nb * 255)), pa))

    return out


def fit_center(
    img: Image.Image,
    target_size: Tuple[int, int],
    mode: str = "contain",
    scale_mult: float = 1.0,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    tw, th = target_size
    if mode == "stretch":
        return img.resize((tw, th), resample)

    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    scale = min(tw / iw, th / ih) * max(0.01, scale_mult)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))

    resized = img.resize((nw, nh), resample)
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    x = (tw - nw) // 2
    y = (th - nh) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def fit_center_top_at_mid(
    img: Image.Image,
    target_size: Tuple[int, int],
    scale_mult: float = 1.0,
    y_offset: int = 0,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Fit sprite inside target, centered on X, with top edge at Y midpoint."""
    tw, th = target_size
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return Image.new("RGBA", (tw, th), (0, 0, 0, 0))

    scale = min(tw / iw, th / ih) * max(0.01, scale_mult)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))

    resized = img.resize((nw, nh), resample)
    canvas = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    x = (tw - nw) // 2
    y = (th // 2) + y_offset
    canvas.paste(resized, (x, y), resized)
    return canvas


def apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.split()[3]
    clamped = max(0.0, min(1.0, opacity))
    lut = [int(round(i * clamped)) for i in range(256)]
    alpha = alpha.point(lut)
    rgba.putalpha(alpha)
    return rgba


def scale_luminance(mask: Image.Image, strength: float) -> Image.Image:
    """Scale a single-channel mask while preserving its luminance values."""
    clamped = max(0.0, min(1.0, strength))
    lut = [int(round(i * clamped)) for i in range(256)]
    return mask.convert("L").point(lut)


def generate_circle_lighting(*, hold: bool, pulse: float = 0.5) -> Image.Image:
    """Generate additive tap flashes and hold rings for the circle receptors.

    White is intentional: legacy hit lighting is a single shared texture rather
    than a per-lane asset, so baking one lane colour into it would be incorrect.
    """
    scale = 4
    width = 165 if hold else 285
    height = 245 if hold else 333
    clamped_pulse = max(0.0, min(1.0, pulse))
    diameter = 52 if hold else 60
    centre_x = width / 2
    centre_y = (height / 2) - 24

    mask = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(mask)
    radius = diameter * scale / 2
    cx = centre_x * scale
    cy = centre_y * scale
    bounds = (
        int(round(cx - radius)),
        int(round(cy - radius)),
        int(round(cx + radius)),
        int(round(cy + radius)),
    )
    draw.ellipse(bounds, fill=255)

    if hold:
        ring = Image.new("L", mask.size, 0)
        ring_draw = ImageDraw.Draw(ring)
        ring_draw.ellipse(bounds, outline=255, width=4 * scale)

        halo = scale_luminance(
            mask.filter(
                ImageFilter.GaussianBlur(radius=(6 + (8 * clamped_pulse)) * scale)
            ),
            0.06 + (0.20 * clamped_pulse),
        )
        soft_ring = scale_luminance(
            ring.filter(
                ImageFilter.GaussianBlur(radius=(2 + (3 * clamped_pulse)) * scale)
            ),
            0.10 + (0.08 * clamped_pulse),
        )
        hard_ring = scale_luminance(
            ring.filter(ImageFilter.GaussianBlur(radius=1.25 * scale)),
            0.10 + (0.02 * clamped_pulse),
        )
        alpha = ImageChops.add(ImageChops.add(halo, soft_ring), hard_ring)
    else:
        peak_alpha = 61
        falloff_radius = 28.0
        alpha = Image.new("L", (width, height), 0)
        alpha.putdata(
            [
                int(
                    round(
                        peak_alpha
                        * math.exp(
                            -math.hypot(x - centre_x, y - centre_y)
                            / falloff_radius
                        )
                    )
                )
                for y in range(height)
                for x in range(width)
            ]
        )

    alpha = alpha.resize((width, height), Image.Resampling.LANCZOS)

    glow = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    glow.putalpha(alpha)
    return glow


def hold_top_row_strip(hold_img: Image.Image) -> Image.Image:
    """Return a 1px-high strip from the top row for osu hold-stretch behavior."""
    rgba = hold_img.convert("RGBA")
    w, h = rgba.size
    if w <= 0 or h <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    return rgba.crop((0, 0, w, 1))


def trim_alpha(img: Image.Image) -> Image.Image:
    """Trim fully transparent outer pixels to avoid undersized fitting."""
    rgba = img.convert("RGBA")
    alpha = rgba.split()[3]
    bbox = alpha.getbbox()
    if bbox is None:
        return rgba
    return rgba.crop(bbox)


class Atlas:
    def __init__(self, xml_path: Path):
        root = ET.parse(xml_path).getroot()
        image_rel = root.attrib["imagePath"]
        self.atlas_image = Image.open(xml_path.parent / image_rel).convert("RGBA")
        self.sub: Dict[str, Dict[str, int]] = {}

        for node in root.findall("SubTexture"):
            data = {k: int(v) for k, v in node.attrib.items() if k != "name"}
            self.sub[node.attrib["name"]] = data

    def extract(self, name: str) -> Image.Image:
        if name not in self.sub:
            raise KeyError(f"SubTexture '{name}' not found in XML")

        s = self.sub[name]
        x, y = s["x"], s["y"]
        w, h = s["width"], s["height"]

        crop = self.atlas_image.crop((x, y, x + w, y + h))

        if "frameWidth" in s and "frameHeight" in s:
            fw, fh = s["frameWidth"], s["frameHeight"]
            fx, fy = s.get("frameX", 0), s.get("frameY", 0)

            rebuilt = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            paste_x = -fx
            paste_y = -fy
            rebuilt.paste(crop, (paste_x, paste_y), crop)
            return rebuilt

        return crop


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_png(img: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    img.save(path, "PNG")


def ordered_dither_alpha(img: Image.Image, strength: float = 1.5) -> Image.Image:
    """Dither low-alpha gradients with a stable 8x8 Bayer pattern."""
    bayer_8x8 = (
        (0, 48, 12, 60, 3, 51, 15, 63),
        (32, 16, 44, 28, 35, 19, 47, 31),
        (8, 56, 4, 52, 11, 59, 7, 55),
        (40, 24, 36, 20, 43, 27, 39, 23),
        (2, 50, 14, 62, 1, 49, 13, 61),
        (34, 18, 46, 30, 33, 17, 45, 29),
        (10, 58, 6, 54, 9, 57, 5, 53),
        (42, 26, 38, 22, 41, 25, 37, 21),
    )

    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    width, height = alpha.size
    dithered = Image.new("L", alpha.size, 0)

    for y in range(height):
        for x in range(width):
            value = int(alpha.getpixel((x, y)))
            if value == 0:
                continue

            threshold = ((bayer_8x8[y % 8][x % 8] + 0.5) / 64.0) - 0.5
            adjusted = round(value + (threshold * strength))
            dithered.putpixel((x, y), max(0, min(255, adjusted)))

    rgba.putalpha(dithered)
    return rgba


def save_png_with_2x(
    img: Image.Image,
    path: Path,
    *,
    dither_alpha: bool = False,
) -> None:
    sd = ordered_dither_alpha(img) if dither_alpha else img
    save_png(sd, path)
    w, h = img.size
    img_2x = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    if dither_alpha:
        img_2x = ordered_dither_alpha(img_2x)
    save_png(img_2x, path.with_name(f"{path.stem}@2x{path.suffix}"))


def clean_output(out_root: Path) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Psych chip note atlas to osu assets")
    parser.add_argument("--xml", default="fnf-to-osu/NOTE_assets-chip.xml", help="Path to NOTE_assets-chip.xml")
    parser.add_argument("--out", default="fnf-to-osu/convert", help="Output folder")
    parser.add_argument("--root", default=".", help="Skin root (contains mania/) for reference sizes")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    xml_path = (root / args.xml).resolve()
    out_root = (root / args.out).resolve()

    clean_output(out_root)

    atlas = Atlas(xml_path)

    # Fixed export sizes (do not derive from existing files at runtime).
    note_size = (125, 125)
    hold_size = (125, 1)
    tail_size = (125, 125)
    key_size = (125, 175)

    note_mini_size = (96, 96)
    hold_mini_size = (96, 1)
    tail_mini_size = (96, 96)
    key_mini_size = (96, 128)

    # output_name, source_direction_name, source_color_name, source_xml_dir, palette
    lane_specs = [
        ("left", "left", "purple", "LEFT", ARROW_RGB["left"]),
        ("down", "down", "blue", "DOWN", ARROW_RGB["down"]),
        ("up", "up", "green", "UP", ARROW_RGB["up"]),
        ("right", "right", "red", "RIGHT", ARROW_RGB["right"]),
        ("center", "down", "blue", "DOWN", ARROW_RGB["center"]),
        ("left2", "left", "purple", "LEFT", ARROW_RGB["left2"]),
        ("down2", "down", "blue", "DOWN", ARROW_RGB["down2"]),
        ("up2", "up", "green", "UP", ARROW_RGB["up2"]),
        ("right2", "right", "red", "RIGHT", ARROW_RGB["right2"]),
    ]

    def emit_set(
        note_dir: Path,
        key_dir: Path,
        set_note_size: Tuple[int, int],
        set_hold_size: Tuple[int, int],
        set_tail_size: Tuple[int, int],
        set_key_size: Tuple[int, int],
    ) -> None:
        for out_name, src_dir_name, src_color_name, _src_xml_dir, pal in lane_specs:
            head = recolor_rgbpalette(atlas.extract(f"{src_color_name}0000"), pal)
            hold = recolor_rgbpalette(atlas.extract(f"{src_color_name} hold piece0000"), pal)
            tail = recolor_rgbpalette(atlas.extract(f"{src_color_name} hold end0000"), pal)

            # Requested mapping: <dir>_confirm_0 -> <dir>D
            d_key = recolor_rgbpalette(atlas.extract(f"{src_dir_name} confirm0000"), pal)
            d_key = trim_alpha(d_key)

            save_png_with_2x(fit_center(head, set_note_size, "contain"), note_dir / f"{out_name}.png")
            hold_strip = hold_top_row_strip(hold)
            hold_strip = fit_center(hold_strip, set_hold_size, "stretch")
            hold_strip = apply_opacity(hold_strip, 0.5)
            save_png_with_2x(hold_strip, note_dir / f"{out_name}L.png")

            tail_img = fit_center_top_at_mid(tail, set_tail_size, y_offset=-1)
            tail_img = apply_opacity(tail_img, 0.5)
            save_png_with_2x(tail_img, note_dir / f"{out_name}T.png")

            d_canvas = fit_center(
                d_key,
                set_key_size,
                "contain",
                scale_mult=1.12,
                resample=Image.Resampling.BICUBIC,
            )
            d_canvas = d_canvas.filter(ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=2))
            save_png_with_2x(d_canvas, key_dir / f"{out_name}D.png")

        # Keep base key gray
        key_base_src = atlas.extract("arrowDOWN0000")
        key_base_src = trim_alpha(key_base_src)
        key_base = fit_center(key_base_src, set_key_size, "contain")
        save_png_with_2x(key_base, key_dir / "key.png")

    emit_set(out_root / "notes", out_root / "keys", note_size, hold_size, tail_size, key_size)
    emit_set(
        out_root / "notesmini",
        out_root / "keysmini",
        note_mini_size,
        hold_mini_size,
        tail_mini_size,
        key_mini_size,
    )

    save_png_with_2x(
        generate_circle_lighting(hold=False),
        out_root / "lightingN.png",
        dither_alpha=True,
    )
    save_png_with_2x(
        generate_circle_lighting(hold=True),
        out_root / "lightingL.png",
        dither_alpha=True,
    )

    hold_light_frame_count = 3
    hold_light_pulse = (0.0, 0.75, 0.15)
    for frame_index in range(hold_light_frame_count):
        pulse = hold_light_pulse[frame_index]
        frame = generate_circle_lighting(hold=True, pulse=pulse)
        save_png_with_2x(
            frame,
            out_root / f"lightingL-{frame_index}.png",
            dither_alpha=True,
        )

    print(f"Done. Exported converted assets to: {out_root}")


if __name__ == "__main__":
    main()
