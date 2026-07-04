import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import config
from dataset import bulid_dataloader
from model import BertClassifierModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT classifier on CLUE TNEWS.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_valid_batches", type=int, default=None)
    return parser.parse_args()


def build_loaders():
    train_loader = bulid_dataloader(
        split="train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    val_loader = bulid_dataloader(
        split="validation",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def set_model_train_mode(model):
    model.train()
    if hasattr(model, "bert_model"):
        # BertClassifierModel freezes BERT with torch.no_grad(), so keep encoder dropout off.
        model.bert_model.eval()


def move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, max_batches=None):
    set_model_train_mode(model)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, batch["labels"])
        total_examples += batch_size

        avg_loss = total_loss / total_examples
        avg_acc = total_correct / total_examples
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        if max_batches is not None and step >= max_batches:
            break

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc="valid", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = criterion(logits, batch["labels"])

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, batch["labels"])
        total_examples += batch_size

        avg_loss = total_loss / total_examples
        avg_acc = total_correct / total_examples
        progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        if max_batches is not None and step >= max_batches:
            break

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, train_metrics, val_metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": vars(config),
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader = build_loaders()
    model = BertClassifierModel(num_labels=config.classname_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    checkpoint_dir = Path(args.checkpoint_dir)
    best_val_acc = 0.0

    print(f"device: {device}")
    print(f"model: {config.model_name}")
    print(
        f"epochs: {config.num_epochs}, "
        f"batch_size: {config.batch_size}, "
        f"lr: {config.learning_rate}"
    )

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_valid_batches,
        )

        is_best = val_metrics["acc"] >= best_val_acc
        if is_best:
            best_val_acc = val_metrics["acc"]

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
        )
        if is_best:
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )

        print(
            f"epoch {epoch}/{config.num_epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
