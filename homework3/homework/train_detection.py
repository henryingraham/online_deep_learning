import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
from .models import load_model, save_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train(
    exp_dir: str = "logs",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    depth_weight: float = 5.0,
    seed: int = 2024,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("No GPU found, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = PROJECT_ROOT / exp_dir / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    model = load_model("detector").to(device)
    num_workers = 2 if device.type == "cuda" else 0

    train_data = load_data(
        str(PROJECT_ROOT / "drive_data" / "train"),
        transform_pipeline="aug",
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_data = load_data(
        str(PROJECT_ROOT / "drive_data" / "val"),
        transform_pipeline="default",
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # emphasize rare lane-boundary classes
    class_weights = torch.tensor([0.2, 2.0, 2.0], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    metric = DetectionMetric()
    best_iou = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        for batch in train_data:
            image = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            optimizer.zero_grad()
            logits, pred_depth = model(image)

            seg_loss = F.cross_entropy(logits, track, weight=class_weights)
            depth_loss = F.l1_loss(pred_depth, depth)
            loss = seg_loss + depth_weight * depth_loss

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in val_data:
                image = batch["image"].to(device)
                track = batch["track"]
                depth = batch["depth"]

                pred, pred_depth = model.predict(image)
                metric.add(pred.cpu(), track, pred_depth.cpu(), depth)

        stats = metric.compute()
        print(
            f"Epoch {epoch + 1:2d}/{num_epoch}: "
            f"loss={np.mean(train_losses):.4f} "
            f"iou={stats['iou']:.4f} acc={stats['accuracy']:.4f} "
            f"depth={stats['abs_depth_error']:.4f} tp_depth={stats['tp_depth_error']:.4f}"
        )

        if stats["iou"] > best_iou:
            best_iou = stats["iou"]
            save_model(model)
            torch.save(model.state_dict(), log_dir / "detector.th")
            print(f"  saved best model (iou={best_iou:.4f})")

    print(f"Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--depth_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
