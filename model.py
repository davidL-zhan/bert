from transformers import BertModel
import torch.nn as nn
from config import config
from transformers import BertConfig
from transformers import BertTokenizer
import torch


class BertClassifierModel(nn.Module):
    def __init__(self, num_labels=15):
        # 1- 初始化父类
        super().__init__()

        # 2- 搭建网络结构
        # 2.1- 先定义Bert模型
        self.bert_model = BertModel.from_pretrained(config.model_name)

        # 2.2- 再定义我们自己的网络结构
        in_features = BertConfig.from_pretrained(config.model_name).hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(in_features=in_features, out_features=num_labels)

    def forward(self, input_ids, attention_mask):
        # torch.no_grad()冻结bert的反向传播。如果放开，训练耗时大量增加
        # self.bert_model.eval()
        # with torch.no_grad():
        bert_output = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # 调用我们自己的网络层
        """
            last_hidden_state[:,0]和pooler_output，实际是类似的东西，都表示[CLS]的隐藏状态。
            区别：需要对last_hidden_state[:,0]经过nn.Linear和激活函数处理后，才能得到pooler_output
            对应源代码位置：BertModel文件的697行
            【推荐】：使用last_hidden_state[:,0]
        """
        # 下面两行代码的效果类似
        # return self.linear(bert_output.pooler_output)
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        return self.linear(self.dropout(cls_hidden_state))


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    model = BertClassifierModel().to(device)
    model.eval()

    sentences = ["这部电影很好看", "这个餐厅的服务太差了"]
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)

    print("device:", device)
    print("input_ids shape:", tuple(input_ids.shape))
    print("attention_mask shape:", tuple(attention_mask.shape))
    print("logits shape:", tuple(logits.shape))
    print("loss:", float(loss))
