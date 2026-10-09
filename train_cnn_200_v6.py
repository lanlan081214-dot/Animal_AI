from pathlib import Path
import json
import math
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAIN_DIR = ROOT / "data_200_v6" / "train"
VAL_DIR = ROOT / "data_200_v6" / "val"

CLASS_MAP_PATH = ROOT / "models" / "class_names_200_v6_fixed.json"
OUT_MODEL = ROOT / "models" / "animal_ai_efficientnet_b0_200_v6_best.pth"
OUT_CLASS_MAP = ROOT / "models" / "class_names_200_v6_train.json"

IMAGE_SIZE = 192
BATCH_SIZE = 32
EPOCHS_STAGE1 = 2
EPOCHS_STAGE2 = 6

NUM_WORKERS = 0
CPU_THREADS = 8

# Stage 1: classifier only
LR_CLASSIFIER_STAGE1 = 1e-3

# Stage 2: partial backbone fine-tuning
LR_BACKBONE_STAGE2 = 1e-5
LR_CLASSIFIER_STAGE2 = 5e-5

WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.04

PATIENCE = 2
GRAD_CLIP = 1.0

SEED = 42


# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def setup_device():
    torch.set_num_threads(CPU_THREADS)

    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"CPU threads : {torch.get_num_threads()}")
    print(f"Device      : {device}")

    return device


# ============================================================
# DATA
# ============================================================

def build_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=8,
            fill=0
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.12,
            hue=0.02
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        transforms.RandomErasing(
            p=0.08,
            scale=(0.02, 0.08),
            ratio=(0.3, 3.3),
            value=0
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )

    return train_dataset, val_dataset, train_loader, val_loader


# ============================================================
# MODEL
# ============================================================

def build_model(num_classes, device):
    print("\nLoading EfficientNet-B0 ImageNet pretrained...")

    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        print("WARNING: Khong tai duoc pretrained weights.")
        print("Dung EfficientNet-B0 khong pretrained.")
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    model = model.to(device)

    return model


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(model, epoch, best_top1, class_to_idx):
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_top1": best_top1,
            "model_state_dict": model.state_dict(),
            "num_classes": len(class_to_idx),
            "image_size": IMAGE_SIZE,
            "class_to_idx": class_to_idx,
        },
        OUT_MODEL
    )

    # Save class names in index order
    idx_to_class = {
        idx: name
        for name, idx in class_to_idx.items()
    }

    class_names = [
        idx_to_class[i]
        for i in range(len(idx_to_class))
    ]

    with open(
        OUT_CLASS_MAP,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            class_names,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    start = time.time()

    for batch_idx, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        if GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

        optimizer.step()

        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        running_correct += (
            predictions == targets
        ).sum().item()

        total += batch_size

        if batch_idx == 1 or batch_idx % 100 == 0:
            current_loss = running_loss / total
            current_acc = running_correct / total * 100

            print(
                f"  Epoch {epoch}/{total_epochs} | "
                f"Batch {batch_idx}/{len(loader)} | "
                f"Loss {current_loss:.4f} | "
                f"Acc {current_acc:.2f}%",
                flush=True
            )

    elapsed = time.time() - start

    return (
        running_loss / total,
        running_correct / total * 100,
        elapsed
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        loss = criterion(outputs, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size

        total += batch_size

        top1 = outputs.argmax(dim=1)

        top1_correct += (
            top1 == targets
        ).sum().item()

        top5 = outputs.topk(
            k=min(5, outputs.shape[1]),
            dim=1
        ).indices

        top5_correct += (
            top5 == targets.unsqueeze(1)
        ).any(dim=1).sum().item()

    avg_loss = total_loss / total
    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100

    return avg_loss, top1_acc, top5_acc


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(SEED)

    print("=" * 78)
    print("ANIMAL AI - TRAIN CNN V6")
    print("EFFICIENTNET-B0 | 200 SPECIES")
    print("120 MAMMALS + 50 REPTILES + 30 BIRDS")
    print("=" * 78)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay train dataset: {TRAIN_DIR}"
        )

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Khong tim thay val dataset: {VAL_DIR}"
        )

    device = setup_device()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\n[1/5] Loading dataset...")

    train_dataset, val_dataset, train_loader, val_loader = (
        build_dataloaders()
    )

    num_classes = len(train_dataset.classes)

    print(f"Train images : {len(train_dataset):,}")
    print(f"Val images   : {len(val_dataset):,}")
    print(f"Classes      : {num_classes}")
    print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {NUM_WORKERS}")

    if num_classes != 200:
        raise ValueError(
            f"Dataset phai co 200 classes, nhung tim thay {num_classes}"
        )

    # Verify train / val class mapping
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Class mapping train va val khong giong nhau!"
        )

    # --------------------------------------------------------
    # Save/verify class map
    # --------------------------------------------------------

    print("\n[2/5] Kiem tra class map...")

    if CLASS_MAP_PATH.exists():
        try:
            with open(
                CLASS_MAP_PATH,
                "r",
                encoding="utf-8"
            ) as f:
                fixed_map = json.load(f)

            print(
                f"Fixed class map: {len(fixed_map)} classes"
            )

            if len(fixed_map) != 200:
                print(
                    "WARNING: class_names_200_v6_fixed.json "
                    "khong co 200 classes."
                )

        except Exception as e:
            print(
                f"WARNING: Khong doc duoc fixed class map: {e}"
            )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n[3/5] Tao model...")

    model = build_model(
        num_classes,
        device
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    best_top1 = 0.0
    best_epoch = 0

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("STAGE 1 - TRAIN CLASSIFIER")
    print("=" * 78)

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LR_CLASSIFIER_STAGE1,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS_STAGE1,
        eta_min=1e-5
    )

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        print(f"\nEpoch {epoch}/{EPOCHS_STAGE1}")

        train_loss, train_acc, elapsed = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            EPOCHS_STAGE1
        )

        val_loss, val_top1, val_top5 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step()

        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Top-1  : {val_top1:.2f}%")
        print(f"Val Top-5  : {val_top5:.2f}%")
        print(f"Time       : {elapsed / 60:.1f} phut")

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_epoch = epoch

            save_checkpoint(
                model,
                epoch,
                best_top1,
                train_dataset.class_to_idx
            )

            print(
                f"NEW BEST -> {best_top1:.2f}%"
            )

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("STAGE 2 - FINE-TUNE FEATURES[4:] + CLASSIFIER")
    print("=" * 78)

    for param in model.features.parameters():
        param.requires_grad = False

    # EfficientNet-B0 features:
    # 0..8
    # Fine-tune from block 4 onward.
    for idx in range(4, len(model.features)):
        for param in model.features[idx].parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("classifier"):
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": LR_BACKBONE_STAGE2,
            },
            {
                "params": classifier_params,
                "lr": LR_CLASSIFIER_STAGE2,
            },
        ],
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        threshold=0.10,
        min_lr=[
            2e-6,
            5e-6,
        ]
    )

    no_improve = 0

    for epoch in range(
        1,
        EPOCHS_STAGE2 + 1
    ):
        global_epoch = EPOCHS_STAGE1 + epoch

        print(
            f"\nEpoch {epoch}/{EPOCHS_STAGE2} "
            f"(global {global_epoch}/{EPOCHS_STAGE1 + EPOCHS_STAGE2})"
        )

        train_loss, train_acc, elapsed = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            EPOCHS_STAGE2
        )

        val_loss, val_top1, val_top5 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_top1)

        current_lrs = [
            group["lr"]
            for group in optimizer.param_groups
        ]

        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Top-1  : {val_top1:.2f}%")
        print(f"Val Top-5  : {val_top5:.2f}%")
        print(f"LR backbone: {current_lrs[0]:.8f}")
        print(f"LR FC      : {current_lrs[1]:.8f}")
        print(f"Time       : {elapsed / 60:.1f} phut")

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_epoch = global_epoch
            no_improve = 0

            save_checkpoint(
                model,
                global_epoch,
                best_top1,
                train_dataset.class_to_idx
            )

            print(
                f"NEW BEST -> {best_top1:.2f}%"
            )

        else:
            no_improve += 1
            print(
                f"No improvement: "
                f"{no_improve}/{PATIENCE}"
            )

            if no_improve >= PATIENCE:
                print("\nEarly stopping.")
                break

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("TRAIN V6 HOAN TAT")
    print("=" * 78)

    print(f"Best Val Top-1 : {best_top1:.2f}%")
    print(f"Best Epoch     : {best_epoch}")
    print(f"Model          : {OUT_MODEL}")
    print(f"Class map      : {OUT_CLASS_MAP}")
    print("=" * 78)


if __name__ == "__main__":
    main()
