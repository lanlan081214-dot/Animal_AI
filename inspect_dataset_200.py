import os
import csv
import math
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
VAL_DIR = BASE_DIR / "data_200" / "val"
ACCURACY_CSV = BASE_DIR / "reports" / "v6_class_accuracy.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "dataset_inspection_200"

NUM_WORST = 20
PER_CLASS = 18
COLS = 6
W, H = 180, 180
PAD = 8
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scientific_name(folder):
    parts = folder.split("_")
    return f"{parts[-2]} {parts[-1]}" if len(parts) >= 2 else folder


def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def font(size, bold=False):
    paths = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]
        if bold else
        ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    for x in paths:
        if os.path.exists(x):
            try:
                return ImageFont.truetype(x, size)
            except Exception:
                pass
    return ImageFont.load_default()


def list_images(folder):
    return sorted(
        [x for x in folder.iterdir()
         if x.is_file() and x.suffix.lower() in EXTS],
        key=lambda x: x.name.lower()
    )


def read_accuracy():
    if not ACCURACY_CSV.exists():
        return []
    rows = []
    with open(ACCURACY_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = {str(k).lower().strip(): v for k, v in r.items()}
            name = next((d[k] for k in
                         ("class", "class_name", "scientific_name", "folder", "folder_name")
                         if k in d), None)
            acc = next((d[k] for k in
                        ("accuracy", "top1", "top1_accuracy", "top-1", "top_1")
                        if k in d), None)
            if name is not None and acc is not None:
                try:
                    rows.append((str(name), float(acc)))
                except ValueError:
                    pass
    return sorted(rows, key=lambda x: x[1])


def find_folder(folders, name):
    q = name.lower().replace("_", " ").strip()
    for f in folders:
        if scientific_name(f.name).lower() == q:
            return f
    for f in folders:
        if q in f.name.lower():
            return f
    return None


def inspect_images(images):
    valid, broken, sizes = [], [], []
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes.append(im.size)
            valid.append(p)
        except Exception as e:
            broken.append((p, str(e)))
    return valid, broken, sizes


def contact_sheet(images, title, subtitle, out):
    images = images[:PER_CLASS]
    rows = max(1, math.ceil(len(images) / COLS))
    canvas = Image.new(
        "RGB",
        (COLS * (W + PAD) + PAD,
         70 + rows * (H + 38 + PAD) + PAD),
        "white"
    )
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 8), title, fill="black", font=font(22, True))
    d.text((PAD, 38), subtitle, fill="black", font=font(14))

    for i, path in enumerate(images):
        x = PAD + (i % COLS) * (W + PAD)
        y = 70 + (i // COLS) * (H + 38 + PAD)
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((W, H))
                cell = Image.new("RGB", (W, H), "#eeeeee")
                cell.paste(im, ((W - im.width)//2, (H - im.height)//2))
                canvas.paste(cell, (x, y))
        except Exception:
            d.rectangle((x, y, x+W, y+H), outline="red", width=3)
            d.text((x+10, y+10), "LOI ANH", fill="red", font=font(14))
        d.text((x, y+H+3), f"{i+1}. {path.name[:24]}",
               fill="black", font=font(12))
    canvas.save(out, quality=95)


def main():
    print("=" * 70)
    print("ANIMAL AI - INSPECT DATASET 200")
    print("=" * 70)

    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay: {VAL_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = sorted([x for x in VAL_DIR.iterdir() if x.is_dir()],
                     key=lambda x: x.name.lower())
    print(f"Validation classes: {len(folders)}")

    accuracy = read_accuracy()
    selected = []

    if accuracy:
        print(f"Doc accuracy: {ACCURACY_CSV}")
        for name, acc in accuracy[:NUM_WORST]:
            folder = find_folder(folders, name)
            if folder:
                selected.append((folder, acc))
    else:
        print("Khong co v6_class_accuracy.csv -> dung 20 class dau tien.")
        for folder in folders[:NUM_WORST]:
            selected.append((folder, None))

    report = []
    report.append("ANIMAL AI - DATASET INSPECTION 200")
    report.append("=" * 70)
    report.append(f"Validation: {VAL_DIR}")
    report.append(f"Classes inspected: {len(selected)}")
    report.append("")

    for rank, (folder, acc) in enumerate(selected, 1):
        name = scientific_name(folder.name)
        images = list_images(folder)
        valid, broken, sizes = inspect_images(images)

        out_folder = OUTPUT_DIR / f"{rank:02d}_{safe_name(name)}"
        out_folder.mkdir(parents=True, exist_ok=True)

        # Copy anh goc de ban xem rieng neu can
        img_out = out_folder / "images"
        img_out.mkdir(exist_ok=True)
        for src in valid:
            try:
                shutil.copy2(src, img_out / src.name)
            except Exception:
                pass

        subtitle = (
            f"V6 Top-1: {acc:.2f}% | "
            f"Images: {len(images)} | Valid: {len(valid)} | Broken: {len(broken)}"
            if acc is not None else
            f"Images: {len(images)} | Valid: {len(valid)} | Broken: {len(broken)}"
        )

        contact_sheet(
            valid,
            f"{rank}. {name}",
            subtitle,
            out_folder / "contact_sheet.jpg"
        )

        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1

        lines = [
            f"Species: {name}",
            f"Folder: {folder.name}",
            f"V6 Top-1: {acc:.2f}%" if acc is not None else "V6 Top-1: unknown",
            f"Total images: {len(images)}",
            f"Valid images: {len(valid)}",
            f"Broken images: {len(broken)}",
            "",
            "Image sizes:"
        ]
        for s, n in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {s[0]}x{s[1]} : {n}")
        if broken:
            lines += ["", "Broken files:"]
            for path, err in broken:
                lines.append(f"  {path.name}: {err}")

        (out_folder / "report.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        report += lines + ["", "=" * 70, ""]

        print(f"[{rank:02d}/{len(selected)}] {name}")
        if acc is not None:
            print(f"  Top-1: {acc:.2f}%")
        print(f"  Anh: {len(images)} | Loi: {len(broken)}")

    (OUTPUT_DIR / "REPORT.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("HOAN TAT")
    print("=" * 70)
    print(f"Ket qua: {OUTPUT_DIR}")
    print("\nMoi class co:")
    print("  contact_sheet.jpg  <- QUAN TRONG NHAT")
    print("  images/             <- anh goc")
    print("  report.txt")
    print("\nHay mo 20 file contact_sheet.jpg de xem dataset.")
    print("Neu anh trong dataset bi sai/qua kho/khong ro/khac loai,")
    print("gui ket qua cho minh truoc khi train tiep.")


if __name__ == "__main__":
    main()
