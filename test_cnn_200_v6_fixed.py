import os
import json
import csv
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = r"D:\MindX\CSA\Animal_AI"
TRAIN_DIR = os.path.join(ROOT, "data_200_v6", "train")
TEST_DIR = os.path.join(ROOT, "data_200_v6", "test")
MODEL_PATH = os.path.join(ROOT, "models", "animal_ai_efficientnet_b0_200_v6_best.pth")
CLASS_MAP_PATH = os.path.join(ROOT, "models", "class_names_200_v6_train.json")
REPORT_DIR = os.path.join(ROOT, "reports")
DEFAULT_IMAGE_SIZE = 192
DEFAULT_BATCH_SIZE = 32
NUM_WORKERS = 0
CPU_THREADS = 8


def read_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        if all(str(k).isdigit() for k in data):
            return [str(data[str(i)]) for i in range(len(data))]
        names = [None] * len(data)
        for name, index in data.items():
            names[int(index)] = str(name)
        return names
    raise ValueError("Unsupported class map format.")


def folder_to_species(folder):
    parts = folder.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit():
        return parts[2].replace("_", " ")
    return folder.replace("_", " ")


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("Cannot find model state_dict in checkpoint.")


def checkpoint_class_order(checkpoint):
    """Read class folder names in output-index order from the checkpoint."""
    if not isinstance(checkpoint, dict):
        return None
    mapping = checkpoint.get("class_to_idx")
    if not isinstance(mapping, dict) or not mapping:
        return None
    try:
        pairs = [(str(folder), int(index)) for folder, index in mapping.items()]
    except (TypeError, ValueError):
        return None
    indices = sorted(index for _, index in pairs)
    if indices != list(range(len(pairs))):
        raise RuntimeError("class_to_idx trong checkpoint có index không liên tục.")
    pairs.sort(key=lambda item: item[1])
    return [folder for folder, _ in pairs]


def main():
    parser = argparse.ArgumentParser(description="Test Animal AI V6.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    torch.set_num_threads(CPU_THREADS)
    device = torch.device("cpu")

    print("=" * 78)
    print("ANIMAL AI - TEST CNN V6")
    print("EFFICIENTNET-B0 | 200 SPECIES")
    print("120 MAMMALS + 50 REPTILES + 30 BIRDS")
    print("=" * 78)
    print(f"Device      : {device}")
    print(f"CPU threads : {torch.get_num_threads()}")

    for path, label in (
        (TRAIN_DIR, "train folder"),
        (TEST_DIR, "test folder"),
        (MODEL_PATH, "model checkpoint"),
        (CLASS_MAP_PATH, "class map"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}:\n{path}")

    print("\n[1/5] Loading checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    image_size = DEFAULT_IMAGE_SIZE
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("image_size"), int):
        image_size = checkpoint["image_size"]
    state_dict = get_state_dict(checkpoint)
    print(f"Model      : {MODEL_PATH}")
    print(f"Image size : {image_size}x{image_size}")

    print("\n[2/5] Checking class map...")
    json_class_names = read_class_names(CLASS_MAP_PATH)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ImageFolder sorts folder names alphabetically. Use the same order as
    # training, and prefer the class_to_idx mapping stored in the checkpoint.
    train_ds = datasets.ImageFolder(TRAIN_DIR)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
    if train_ds.classes != test_ds.classes:
        train_only = sorted(set(train_ds.classes) - set(test_ds.classes))
        test_only = sorted(set(test_ds.classes) - set(train_ds.classes))
        raise RuntimeError(
            "Train/test class folders do not match.\n"
            f"Only in train: {train_only[:10]}\n"
            f"Only in test: {test_only[:10]}"
        )

    saved_folder_order = checkpoint_class_order(checkpoint)
    if saved_folder_order is not None:
        if saved_folder_order != train_ds.classes:
            raise RuntimeError(
                "Checkpoint class_to_idx does not match the training folder order. "
                "Stopping to avoid incorrect evaluation."
            )
        ordered_folders = saved_folder_order
        print("Class order : from checkpoint class_to_idx")
    else:
        ordered_folders = train_ds.classes
        print("Class order : from ImageFolder training order")

    class_names = [folder_to_species(name) for name in ordered_folders]
    folder_names = [folder_to_species(name) for name in test_ds.classes]
    json_names = [folder_to_species(name) for name in json_class_names]

    if len(class_names) != len(test_ds.classes):
        raise RuntimeError(
            f"Class count mismatch: model={len(class_names)}, "
            f"test folders={len(test_ds.classes)}"
        )

    # JSON order may differ from ImageFolder/model order. Compare the set only;
    # do not use JSON order to label model outputs.
    if len(json_names) != len(class_names) or set(json_names) != set(class_names):
        missing = sorted(set(class_names) - set(json_names))
        extra = sorted(set(json_names) - set(class_names))
        raise RuntimeError(
            "Class names in JSON do not match dataset folders.\n"
            f"Missing in JSON: {missing[:10]}\n"
            f"Extra in JSON: {extra[:10]}"
        )

    if folder_names != class_names:
        raise RuntimeError("Test class order does not match model output order.")

    print(f"Classes     : {len(class_names)}")
    print(f"Test images : {len(test_ds)}")
    print("Class map   : OK (model output order)")

    print("\n[3/5] Building EfficientNet-B0...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch.\nMissing: {missing[:10]}\n"
            f"Unexpected: {unexpected[:10]}"
        )
    model.to(device)
    model.eval()

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    print("\n[4/5] Evaluating test set...")
    criterion = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    per_class = defaultdict(lambda: {
        "total": 0,
        "top1": 0,
        "top5": 0,
        "wrong": defaultdict(int),
    })

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch = labels.size(0)
            loss_sum += loss.item() * batch

            k = min(5, outputs.shape[1])
            top_indices = outputs.topk(k, dim=1).indices
            matches = top_indices.eq(labels.view(-1, 1))
            top1_correct += matches[:, 0].sum().item()
            top5_correct += matches.any(dim=1).sum().item()
            total += batch

            for true_idx, preds in zip(labels.tolist(), top_indices.tolist()):
                true_name = class_names[true_idx]
                rec = per_class[true_name]
                rec["total"] += 1
                if preds[0] == true_idx:
                    rec["top1"] += 1
                else:
                    rec["wrong"][class_names[preds[0]]] += 1
                if true_idx in preds:
                    rec["top5"] += 1

            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
                print(
                    f"  Batch {batch_idx}/{len(loader)} | "
                    f"Images {total}/{len(test_ds)} | "
                    f"Top-1 {top1_correct / total * 100:.2f}% | "
                    f"Top-5 {top5_correct / total * 100:.2f}%"
                )

    top1 = top1_correct / total * 100
    top5 = top5_correct / total * 100
    avg_loss = loss_sum / total
    macro_top1 = sum(
        rec["top1"] / rec["total"] * 100
        for rec in per_class.values() if rec["total"]
    ) / len(per_class)

    os.makedirs(REPORT_DIR, exist_ok=True)
    csv_path = os.path.join(REPORT_DIR, "test_v6_per_class.csv")
    summary_path = os.path.join(REPORT_DIR, "test_v6_summary.txt")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "species", "images", "top1_correct", "top1_accuracy_percent",
            "top5_correct", "top5_accuracy_percent",
            "most_common_wrong_prediction", "wrong_prediction_count",
        ])
        for species in sorted(per_class):
            rec = per_class[species]
            wrong = rec["wrong"]
            common_wrong, wrong_count = (
    max(wrong.items(), key=lambda item: item[1])
    if wrong else ("", 0)
)
            writer.writerow([
                species,
                rec["total"],
                rec["top1"],
                round(rec["top1"] / rec["total"] * 100, 2),
                rec["top5"],
                round(rec["top5"] / rec["total"] * 100, 2),
                common_wrong,
                wrong_count,
            ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ANIMAL AI - TEST CNN V6\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test images: {total}\n")
        f.write(f"Classes: {len(class_names)}\n")
        f.write(f"Image size: {image_size}x{image_size}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Test loss: {avg_loss:.4f}\n")
        f.write(f"Test Top-1: {top1:.2f}%\n")
        f.write(f"Test Top-5: {top5:.2f}%\n")
        f.write(f"Macro Top-1: {macro_top1:.2f}%\n")

    print("\n" + "=" * 78)
    print("TEST FINISHED")
    print("=" * 78)
    print(f"Test images      : {total:,}")
    print(f"Classes          : {len(class_names)}")
    print(f"Test Loss        : {avg_loss:.4f}")
    print(f"Test Top-1       : {top1:.2f}%")
    print(f"Test Top-5       : {top5:.2f}%")
    print(f"Macro Top-1      : {macro_top1:.2f}%")
    print(f"Per-class report : {csv_path}")
    print(f"Summary          : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
