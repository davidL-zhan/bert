from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Root_DIR, config
import torch
import csv
from pathlib import Path
from typing import Any

# 全局 tokenizer 复用同一个预训练模型配置，避免每个 batch 重复加载。
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(config.model_name)

# 保存词表
# vocab = tokenizer.get_vocab()
# save_path = Path(Root_DIR) / "data" / "vocab.csv"

# with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["token", "id"])

#     for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
#         writer.writerow([token, token_id])


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    # Hugging Face Dataset 的原始样本在这里被转换成模型可直接使用的张量 batch。
    # {'text': ['对现在的化妆品真是没有抵抗能力，店家宣传、别人一说好就晕了头跟风去买。其实最简单的东西也是最好的。看了这本书，更注意从内而外调养。', '比较不错的酒店，设施比较新，交通还是比较方便，但是离市中心有一点距离，去机场方便。'], 'label': tensor([0, 1])}
    sentences: list[str] = [item[config.text_column] for item in batch]
    labels: torch.Tensor = torch.tensor(
        [item[config.label_column] for item in batch], dtype=torch.long
    )
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


def build_dataloader(
    split: str = "train",
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    # load_dataset 返回指定 split 的 Dataset，DataLoader 负责批处理和 shuffle。
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
    dataloader: DataLoader = build_dataloader()
    print(next(iter(dataloader)))
