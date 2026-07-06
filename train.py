import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
from config import Root_DIR, config
from dataset import bulid_dataloader
from model import BertClassifierModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT classifier on CLUE TNEWS.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_valid_batches", type=int, default=None)
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(Root_DIR) / path


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


def move_batch_to_device(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return (preds == labels).sum().item()


def get_total_training_steps(loader, max_batches=None):
    steps_per_epoch = len(loader)
    if max_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_batches)
    return steps_per_epoch * config.num_epochs


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, device, epoch, max_batches=None
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    model.train()

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
        if config.max_grad_norm is not None and config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, batch["labels"])
        total_examples += batch_size

        avg_loss = total_loss / total_examples
        avg_acc = total_correct / total_examples
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if max_batches is not None and step >= max_batches:
            break

    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    start_time = time.perf_counter()
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
        elapsed_time = time.perf_counter() - start_time
    return {
        "loss": total_loss / max(total_examples, 1),
        "acc": total_correct / max(total_examples, 1),
        "time": elapsed_time,
    }


def save_checkpoint(
    path,
    model,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.state_dict(),
        path,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader = build_loaders()
    model = BertClassifierModel(num_labels=config.classname_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_training_steps = get_total_training_steps(
        train_loader, args.max_train_batches
    )
    warmup_steps = int(total_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    checkpoint_dir = resolve_project_path(args.checkpoint_dir)
    best_val_acc = 0.0

    print(f"device: {device}")
    print(f"model: {config.model_name}")
    print(
        f"epochs: {config.num_epochs}, "
        f"batch_size: {config.batch_size}, "
        f"lr: {config.learning_rate}, "
        f"dropout: {config.dropout}, "
        f"warmup_steps: {warmup_steps}, "
        f"total_steps: {total_training_steps}, "
        f"max_grad_norm: {config.max_grad_norm}"
    )

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
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
        )
        if is_best:
            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
            )

        print(
            f"epoch {epoch}/{config.num_epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"best_val_acc={best_val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
