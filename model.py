from transformers import BertModel
import torch.nn as nn
from config import config
from transformers import BertConfig
from transformers import BertTokenizer
import torch


class BertClassifierModel(nn.Module):

    # 基于预训练 BERT/MacBERT 的文本分类模型。
    def __init__(self, num_labels: int = config.classname_len) -> None:
        # 1- 初始化父类
        super().__init__()

        # 2- 搭建网络结构
        # 2.1- 先定义Bert模型
        self.bert_model: BertModel = BertModel.from_pretrained(config.model_name)

        # 2.2- 再定义我们自己的网络结构
        in_features: int = BertConfig.from_pretrained(config.model_name).hidden_size
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.linear: nn.Linear = nn.Linear(
            in_features=in_features, out_features=num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # torch.no_grad()冻结bert的反向传播。如果放开，训练耗时大量增加
        # self.bert_model.eval()
        # with torch.no_grad():
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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
        cls_hidden_state: torch.Tensor = bert_output.last_hidden_state[:, 0]
        return self.linear(self.dropout(cls_hidden_state))


if __name__ == "__main__":
    from torchinfo import summary

    torch.manual_seed(42)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(config.model_name)
    model: BertClassifierModel = BertClassifierModel(config.classname_len).to(device)
    model.eval()

    sentences: list[str] = ["这部电影很好看", "这个餐厅的服务太差了"]
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )
    input_ids: torch.Tensor = encoded["input_ids"].to(device)
    attention_mask: torch.Tensor = encoded["attention_mask"].to(device)
    labels: torch.Tensor = torch.tensor([0, 1], dtype=torch.long, device=device)

    summary(
        model,
        input_data=(input_ids, attention_mask),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=8,
    )

    with torch.inference_mode():
        logits: torch.Tensor = model(input_ids=input_ids, attention_mask=attention_mask)
        loss: torch.Tensor = nn.CrossEntropyLoss()(logits, labels)

    print("device:", device)
    print("input_ids shape:", input_ids.shape)
    print("input_ids:", input_ids)
    print("attention_mask shape:", attention_mask.shape)
    print("attention_mask :", attention_mask)
    print("logits shape:", logits.shape)
    print("logits :", logits)
    print("loss:", float(loss))
