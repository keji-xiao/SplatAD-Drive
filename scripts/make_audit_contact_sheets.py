"""Build one review contact sheet per camera from exported audit overlays."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build(audit: Path):
    overlay = audit/"lidar_camera_overlay"
    manifest = json.loads((overlay/"manifest.json").read_text(encoding="utf-8"))
    frames = {(r["sensor_name"], r["camera_index"]): r for r in manifest["frames"]}
    grouped = defaultdict(list)
    for path in sorted(overlay.glob("*.png")):
        camera, index = path.stem.rsplit("_", 1)
        grouped[camera].append((int(index), path))
    output = audit/"contact_sheets"
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14) if Path("C:/Windows/Fonts/arial.ttf").exists() else ImageFont.load_default()
    result = []
    for camera, files in grouped.items():
        columns, tile_width, tile_height, label_height, header = 4, 480, 270, 35, 35
        rows = (len(files)+columns-1)//columns
        sheet = Image.new("RGB", (columns*tile_width, header+rows*(tile_height+label_height)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"PandaSet 028 | {camera} | {len(files)} frames | nominal-pose overlay (not row-time compensated)", fill="black", font=font)
        for cell, (index, path) in enumerate(files):
            with Image.open(path) as original:
                image = original.convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x, y = cell%columns*tile_width, header+cell//columns*(tile_height+label_height)
            sheet.paste(image, (x+(tile_width-image.width)//2, y+(tile_height-image.height)//2))
            frame = frames[(camera, index)]
            draw.text((x+5, y+tile_height+3), f"{path.name}  t={frame['camera_time']:.3f}s  camera-lidar={-frame['time_difference_s']*1000:+.1f}ms", fill="black", font=font)
            draw.text((x+5, y+tile_height+18), f"visible projected points: {frame['visible_point_count']}", fill="black", font=font)
        target = output/f"{camera}_20frames.jpg"
        sheet.save(target, quality=94)
        result.append({"camera": camera, "frames": len(files), "contact_sheet": str(target),
                       "first_overlay": str(files[0][1]), "last_overlay": str(files[-1][1])})
    (output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("outputs/audit_corrected"))
    args = parser.parse_args()
    build(args.audit)
