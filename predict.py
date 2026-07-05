import torch
from pathlib import Path
from transformers import BertTokenizer
from config import Root_DIR, config
from model import BertClassifierModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(config.model_name)
model = BertClassifierModel(config.classname_len).to(device)
checkpoint_path = Path(Root_DIR) / "checkpoints" / "best.pt"
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()


def predict(text: dict):
    """接收文本，返回预测结果

    Args:
        text (dict): {"text": "文本"}
    """
    text = tokenizer(
        text["text"],
        padding="max_length",
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids = torch.tensor(text["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(text["attention_mask"]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    return logits.argmax().item()


if __name__ == "__main__":
    print(predict({"text": "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"}))
