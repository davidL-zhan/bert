import torch
from pathlib import Path
from transformers import BertTokenizer
from config import Root_DIR, config
from model import BertClassifierModel

# 推理默认优先使用 CUDA，和训练脚本的设备选择保持一致。
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 推理阶段复用同一个 tokenizer 和分类模型实例。
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(config.model_name)
model: BertClassifierModel = BertClassifierModel(config.classname_len).to(device)
checkpoint_path: Path = Path(Root_DIR) / "checkpoints" / "best.pt"
state_dict: dict[str, torch.Tensor] = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()


def predict(text: dict[str, str]) -> int:
    """接收文本，返回预测结果

    Args:
        text (dict): {"text": "文本"}
    """
    # tokenizer 输出 BatchEncoding，这里只取模型 forward 需要的两个张量。
    encoded = tokenizer(
        text["text"],
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = encoded["input_ids"].to(device)
    attention_mask: torch.Tensor = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits: torch.Tensor = model(input_ids, attention_mask)
    return int(logits.argmax().item())


if __name__ == "__main__":
    print(predict({"text": "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"}))
