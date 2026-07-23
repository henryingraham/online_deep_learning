import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric
from .models import load_model, save_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train(
    exp_dir: str = "logs",
    num_epoch: int = 15,
    lr: float = 1e-3,
    batch_size: int = 128,
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

    log_dir = PROJECT_ROOT / exp_dir / f"classifier_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    model = load_model("classifier").to(device)
    num_workers = 2 if device.type == "cuda" else 0

    train_data = load_data(
        str(PROJECT_ROOT / "classification_data" / "train"),
        transform_pipeline="aug",
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_data = load_data(
        str(PROJECT_ROOT / "classification_data" / "val"),
        transform_pipeline="default",
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    metric = AccuracyMetric()
    best_acc = 0.0

    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        metric.reset()
        with torch.inference_mode():
            for img, label in val_data:
                img = img.to(device)
                pred = model.predict(img)
                metric.add(pred.cpu(), label)

        val_acc = metric.compute()["accuracy"]
        print(
            f"Epoch {epoch + 1:2d}/{num_epoch}: "
            f"loss={np.mean(train_losses):.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model)
            torch.save(model.state_dict(), log_dir / "classifier.th")
            print(f"  saved best model (acc={best_acc:.4f})")

    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
