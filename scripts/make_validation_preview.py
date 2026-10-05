"""Create contact sheets and a timed GT/reconstruction preview from real samples."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    directories = sorted(args.validation.glob("sample_*"))
    if not directories:
        raise ValueError("No completed validation samples")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    rows = (len(directories)+3)//4
    sheets = {kind: Image.new("RGB", (1920, rows*305+35), "white") for kind in ("rgb", "overlay")}
    for kind, sheet in sheets.items():
        ImageDraw.Draw(sheet).text((8, 6), f"Real baseline checkpoint | held-out front camera | {len(directories)} samples | {kind}", fill="black", font=font)
    previews, timestamps = [], []
    for index, directory in enumerate(directories):
        sensor = json.loads((directory/"sensors.json").read_text())
        timestamps.append(sensor["camera_request_timestamp"])
        for kind, sheet in sheets.items():
            with Image.open(directory/"baseline"/(kind+".png")) as raw:
                thumb = raw.convert("RGB").resize((480, 270), Image.Resampling.LANCZOS)
            x, y = index%4*480, 35+index//4*305
            sheet.paste(thumb, (x, y))
            ImageDraw.Draw(sheet).text((x+6, y+273), f"sample {index:02d} | t={timestamps[-1]:.3f}s", fill="black", font=font)
        preview = Image.new("RGB", (1280, 390), "white")
        for column, filename in enumerate((directory/"gt_rgb_resized.png", directory/"baseline/rgb.png")):
            with Image.open(filename) as raw:
                preview.paste(raw.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS), (column*640, 30))
        draw = ImageDraw.Draw(preview)
        draw.text((10, 5), "Ground truth", fill="black", font=font)
        draw.text((650, 5), f"SplatAD-Drive | t={timestamps[-1]:.3f}s", fill="black", font=font)
        previews.append(preview)
    for kind, sheet in sheets.items():
        sheet.save(args.validation/(kind+"_contact.jpg"), quality=94)
    durations = [max(10, round((b-a)*1000)) for a, b in zip(timestamps, timestamps[1:])]
    durations.append(durations[-1] if durations else 1000)
    previews[0].save(args.validation/"reconstruction.gif", save_all=True, append_images=previews[1:],
                     duration=durations, loop=0, disposal=2)
    (args.validation/"preview_manifest.json").write_text(json.dumps({
        "samples": len(directories), "timestamps": timestamps, "duration_ms": durations,
        "scope": "Real held-out front-camera sequence, display resized with GIF palette quantization; original PNG/NPZ retained",
        "overlay_scope": "LiDAR per-ray linear translation only; no camera row, angular, or dynamic cross-sensor-time compensation",
    }, indent=2), encoding="utf-8")
    print((args.validation/"reconstruction.gif").resolve())


if __name__ == "__main__":
    main()
