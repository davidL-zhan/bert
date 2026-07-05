import torch
from transformers import BertTokenizer
from config import config
from model import BertClassifierModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(config.model_name)
model = BertClassifierModel(config.classname_len).to(device)


def predict(text: dict):
    """接收文本，返回预测结果

    Args:
        text (dict): {"text": "文本"}
    """
    text = tokenizer(
        text, padding="max_length", truncation=True, max_length=config.max_length
    )
    input_ids = torch.tensor(text["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(text["attention_mask"]).unsqueeze(0)
    logits = model(input_ids, attention_mask)
