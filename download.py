import json
from pathlib import Path

from datasets import load_dataset

# dataset = load_dataset("STARRY3056/ChnSentiCorp")
dataset = load_dataset("clue/clue", "tnews")
print(dataset)
print(dataset["train"][0])
print(dataset["train"].features)
print(dataset["train"].column_names)

TNEWS_CODE_TO_NAME = {
    "100": "民生/故事",
    "101": "文化",
    "102": "娱乐",
    "103": "体育",
    "104": "财经",
    "106": "房产",
    "107": "汽车",
    "108": "教育",
    "109": "科技",
    "110": "军事",
    "112": "旅游",
    "113": "国际",
    "114": "证券/股票",
    "115": "农业/三农",
    "116": "电竞/游戏",
}

label_codes = dataset["train"].features["label"].names
label_info = [
    {
        "label_id": label_id,
        "label_code": label_code,
        "label_name": TNEWS_CODE_TO_NAME[label_code],
    }
    for label_id, label_code in enumerate(label_codes)
]

output_path = Path("tnews_label_names.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(label_info, f, ensure_ascii=False, indent=2)

print(f"类别名已保存到: {output_path.resolve()}")

