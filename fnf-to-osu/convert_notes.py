#!/usr/bin/env python3
"""Build osu!mania notes and lighting from the supplied FNF atlases.

osu!mania has one shared hit animation and one shared hold animation for all
lanes. Its receptor images cannot animate. Those engine limits prevent exact
per-lane splash colours and the V-Slice hold start/end transitions.

The hold cover atlas comes from FunkinCrew/funkin.assets, shared/images/
holdCoverBlue. V-Slice plays its four middle frames in a loop at 24 FPS.
"""

from __future__ import annotations

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, cast

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


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
    canvas.paste(resized, (x, y))
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
    canvas.paste(resized, (x, y))
    return canvas


def apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    rgba = img.convert("RGBA")
    alpha = rgba.split()[3]
    clamped = max(0.0, min(1.0, opacity))
    lut = [int(round(i * clamped)) for i in range(256)]
    alpha = alpha.point(lut)
    rgba.putalpha(alpha)
    return rgba


def add_release_dot(tail: Image.Image, hold_strip: Image.Image,
                    main_color: int) -> Image.Image:
    """Match the cap's flat edge to the hold body and mark the release point."""
    width, height = tail.size
    join_y = height // 2 - 1
    result = tail.copy().convert("RGBA")
    # At osu!'s fractional-pixel join, a transparent edge leaves a dark line
    # and a full-strength body strip creates a bright line. Half-strength body
    # pixels blend the two layers without changing the cap's dimensions.
    alpha = result.getchannel("A")
    bounds = alpha.getbbox()
    if bounds:
        join = hold_strip.resize((width, 2), Image.Resampling.NEAREST)
        join.putalpha(join.getchannel("A").point(lambda value: round(value * 0.5)))
        result.paste(join, (0, bounds[1]))

    scale = 4
    dot_size = (width * scale, height * scale)
    x, y = width * scale // 2, join_y * scale
    radius = max(4, round(width * 0.055 * scale))
    color = tuple(round(channel * 0.28 + 255 * 0.72)
                  for channel in ((main_color >> 16) & 255,
                                  (main_color >> 8) & 255, main_color & 255))

    halo = Image.new("RGBA", dot_size, (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.ellipse((x - radius * 1.5, y - radius * 1.5,
                       x + radius * 1.5, y + radius * 1.5),
                      fill=(*color, 85))
    halo = halo.filter(ImageFilter.GaussianBlur(radius=2.5 * scale))
    dot = Image.new("RGBA", dot_size, (0, 0, 0, 0))
    dot_draw = ImageDraw.Draw(dot)
    dot_draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=(*color, 240))
    core = max(2, round(radius * 0.32))
    dot_draw.ellipse((x - core, y - core, x + core, y + core),
                     fill=(255, 255, 255, 250))
    halo.alpha_composite(dot)
    result.alpha_composite(halo.resize(tail.size, Image.Resampling.LANCZOS))
    return result


def neutral_lighting(img: Image.Image) -> Image.Image:
    """Keep the atlas's bright detail without tinting every lane one colour."""
    rgba = img.convert("RGBA")
    alpha = ImageChops.multiply(rgba.getchannel("A"), ImageOps.grayscale(rgba))
    return Image.merge("RGBA", (Image.new("L", rgba.size, 255),) * 3 + (alpha,))


def animation_frames(atlas: "Atlas", prefix: str) -> list[Image.Image]:
    names = [n for n in atlas.sub if n.startswith(prefix)]
    names.sort(key=lambda n: int(re.search(r"(\d+)$", n).group(1)))
    if not names:
        raise KeyError(f"No frames with prefix {prefix!r} in atlas")
    return [atlas.extract(name) for name in names]


def save_animation(frames: list[Image.Image], path: Path, size: Tuple[int, int],
                   *, source_fps: int = 24, output_fps: int = 60,
                   align_first_frame: bool = False, art_scale: float = 1.0,
                   x_offset: int = 0, y_offset: int = 0) -> None:
    """Repeat source frames to preserve FNF timing in osu!'s 60 FPS lighting."""
    count = round(len(frames) * output_fps / source_fps)

    def render(source: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        art_size = tuple(max(1, round(d * art_scale)) for d in target_size)
        image = fit_center(neutral_lighting(source), art_size)
        x = (target_size[0] - art_size[0]) // 2
        y = (target_size[1] - art_size[1]) // 2
        if align_first_frame:
            first = fit_center(neutral_lighting(frames[0]), art_size)
            bounds = first.getchannel("A").getbbox()
            if bounds:
                x += round(art_size[0] / 2 - (bounds[0] + bounds[2]) / 2)
                y += round(art_size[1] / 2 - (bounds[1] + bounds[3]) / 2)
        x += round(x_offset * target_size[0] / size[0])
        y += round(y_offset * target_size[0] / size[0])
        canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
        canvas.paste(image, (x, y))
        return canvas

    rendered: dict[tuple[int, int], Image.Image] = {}
    for index in range(count):
        source_index = min(len(frames) - 1, int(index * source_fps / output_fps))
        for multiplier in (1, 2):
            key = (source_index, multiplier)
            if key not in rendered:
                target_size = (size[0] * multiplier, size[1] * multiplier)
                rendered[key] = render(frames[source_index], target_size)
            filename = path.with_name(f"{path.stem}-{index}{'@2x' if multiplier == 2 else ''}{path.suffix}")
            save_png(rendered[key], filename)
        if index == 0:
            save_png(rendered[(source_index, 1)], path)
            save_png(rendered[(source_index, 2)],
                     path.with_name(f"{path.stem}@2x{path.suffix}"))


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
    def __init__(self, xml_path: Path, image_path: Path | None = None):
        root = ET.parse(xml_path).getroot()
        default_image = xml_path.parent / root.attrib["imagePath"]
        # Both supplied XML files still contain their pre-rename imagePath.
        image_file = image_path or (default_image if default_image.exists() else xml_path.with_suffix(".png"))
        self.atlas_image = Image.open(image_file).convert("RGBA")
        self.sub: Dict[str, Dict[str, int | bool]] = {}

        for node in root.findall("SubTexture"):
            data = {k: (v == "true" if k == "rotated" else int(v))
                    for k, v in node.attrib.items() if k != "name"}
            self.sub[node.attrib["name"]] = data

    def extract(self, name: str) -> Image.Image:
        if name not in self.sub:
            raise KeyError(f"SubTexture '{name}' not found in XML")

        s = self.sub[name]
        x, y = s["x"], s["y"]
        w, h = s["width"], s["height"]

        crop = self.atlas_image.crop((x, y, x + w, y + h))
        if s.get("rotated", False):
            crop = crop.transpose(Image.Transpose.ROTATE_90)

        if "frameWidth" in s and "frameHeight" in s:
            fw, fh = s["frameWidth"], s["frameHeight"]
            fx, fy = s.get("frameX", 0), s.get("frameY", 0)

            rebuilt = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            paste_x = -fx
            paste_y = -fy
            rebuilt.paste(crop, (paste_x, paste_y))
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
    parser = argparse.ArgumentParser(description="Convert FNF note atlases to osu!mania assets")
    parser.add_argument(
        "--xml",
        default="Notes.xml",
        help="Path to Notes.xml",
    )
    parser.add_argument(
        "--splashes-xml", default="Splashes.xml", help="Path to Splashes.xml",
    )
    parser.add_argument(
        "--splashes-png", default="Spashes.png", help="Path to the supplied splash atlas PNG",
    )
    parser.add_argument(
        "--out",
        default="convert",
        help="Output folder",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Atlas directory; reference skin is its parent",
    )
    parser.add_argument("--install", action="store_true",
                        help="Copy the generated PNGs into the skin after conversion")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    xml_path = (root / args.xml).resolve()
    out_root = (root / args.out).resolve()

    atlas = Atlas(xml_path)
    splashes = Atlas((root / args.splashes_xml).resolve(), (root / args.splashes_png).resolve())
    cover = Atlas(root / "holdCoverBlue.xml")
    skin_root = root.parent

    if out_root != root / "convert":
        raise ValueError("Output must be the atlas directory's convert folder")
    clean_output(out_root)

    note_size = Image.open(skin_root / "mania/notes/left.png").size
    key_size = Image.open(skin_root / "mania/keys/key.png").size

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
            tail_img = add_release_dot(tail_img, hold_strip, pal[0])
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

    emit_set(out_root / "mania/notes", out_root / "mania/keys", note_size,
             (note_size[0], 1), note_size, key_size)

    # Lighting assets are shared by every lane. These canvases match the
    # existing skin, while Skin.ini controls their rendered width.
    splash_size = tuple(round(n * note_size[0] / 125) for n in (285, 333))
    cover_size = tuple(round(n * note_size[0] / 125) for n in (165, 245))
    save_animation(animation_frames(splashes, "PurpC instance 1"),
                   out_root / "lightingN.png", splash_size,
                   art_scale=0.5, x_offset=0, y_offset=-47)
    # Calibrated against the receptor centers in-game. Keep the artwork centered
    # on X and move its visible center into the circle, not the hold body.
    save_animation(animation_frames(cover, "holdCoverBlue"),
                   out_root / "lightingL.png", cover_size,
                   align_first_frame=True, art_scale=0.66,
                   x_offset=1, y_offset=-round(key_size[1] * 0.12))

    if args.install:
        for png in out_root.rglob("*.png"):
            destination = skin_root / png.relative_to(out_root)
            ensure_dir(destination.parent)
            shutil.copy2(png, destination)

    print(f"Done. Exported converted assets to: {out_root}")
    if args.install:
        print(f"Installed PNG assets into: {skin_root}")


if __name__ == "__main__":
    main()
