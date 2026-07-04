from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import config
import torch
import csv
from pathlib import Path

tokenizer = BertTokenizer.from_pretrained(config.model_name)

# 保存词表
# vocab = tokenizer.get_vocab()
# save_path = "data/vocab.csv"

# with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["token", "id"])

#     for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
#         writer.writerow([token, token_id])


def collate_fn(batch):

    sentences = [item["sentence"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    # idx = torch.tensor([item["idx"] for item in batch], dtype=torch.long)

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "token_type_ids": encoded["token_type_ids"],
        "labels": labels,
    }


def bulid_dataloader(
    split="train", batch_size=2, shuffle=True, num_workers=0, drop_last=False
):
    dataset = load_dataset(config.dataset_path, config.dataset_name, split=split)
    # print(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader


if __name__ == "__main__":
    dataloader = bulid_dataloader()
    print(next(iter(dataloader)))
