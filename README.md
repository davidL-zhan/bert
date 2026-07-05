# BERT 中文情感分类项目

本项目是一个中文文本二分类项目，主要用于判断用户评价是正面还是负面。项目包含三条相关链路：

1. 使用 Hugging Face 的 `hfl/chinese-macbert-base` 微调/训练情感分类模型。
2. 使用训练好的 `checkpoints/best.pt` 做本地 BERT 推理。
3. 使用 DashScope OpenAI 兼容接口调用大模型做同类二分类，并通过 FastAPI + Streamlit 提供简单 Web 调用界面。

项目里还包含一个手写版 BERT 实现 `bert.py`，用于学习 BERT 结构、张量形状、注意力机制和 Hugging Face 命名兼容关系。

## 项目结构

```text
bert/
├── app.py                       # FastAPI 后端接口，提供 /predict
├── bert.py                      # 纯 PyTorch 手写 BERT 与序列分类模型
├── config.py                    # 全局配置：数据集、预训练模型、训练超参数
├── dataset.py                   # Hugging Face Dataset 加载、Tokenizer、DataLoader 构建
├── fronted.py                   # Streamlit 前端页面
├── llm_predict.py               # DashScope/OpenAI 兼容接口的大模型分类器
├── model.py                     # 基于 transformers.BertModel 的分类模型
├── predict.py                   # 加载本地 checkpoint 的 BERT 推理脚本
├── train.py                     # 训练入口
├── requirements.txt             # 主要 Python 依赖
├── data/
│   └── ChnSentiCorp_test.csv    # 本地示例测试数据
└── demo/
    ├── ChnSentiCorp.ipynb       # 数据集相关 Notebook
    ├── tokenizer.ipynb          # Tokenizer 相关 Notebook
    └── llm.ipynb                # LLM 调用实验 Notebook
```

## 环境说明

本项目在当前机器约定使用 NLP 虚拟环境：

```powershell
E:\miniconda\envs\NLP\python.exe
```

安装依赖：

```powershell
cd E:\Projects\bert
& "E:\miniconda\envs\NLP\python.exe" -m pip install -r requirements.txt
```

`requirements.txt` 当前包含训练 BERT 所需的核心依赖：

```text
torch==2.8.0+cu128
datasets==5.0.0
sentencepiece
tqdm
torchinfo
transformers
```

如果要运行 Web 接口、大模型推理或 Streamlit 页面，还需要额外确认以下包已安装：

```powershell
& "E:\miniconda\envs\NLP\python.exe" -m pip install fastapi uvicorn openai streamlit requests
```

## 数据集

配置文件 [config.py](config.py) 中默认使用：

```python
dataset_path = "STARRY3056/ChnSentiCorp"
dataset_name = None
text_column = "text"
label_column = "label"
classname_len = 2
```

也就是说，项目默认是中文情感二分类：

| 标签 | 含义 |
|---:|---|
| `0` | 负面评价 |
| `1` | 正面评价 |

`dataset.py` 通过 `datasets.load_dataset(...)` 读取 Hugging Face 数据集，并在 `collate_fn` 中完成分词、padding、truncation 和标签张量构建。每个 batch 返回：

```python
{
    "input_ids": Tensor[batch, seq_len],
    "attention_mask": Tensor[batch, seq_len],
    "token_type_ids": Tensor[batch, seq_len],
    "labels": Tensor[batch],
}
```

训练代码实际传给模型的是 `input_ids`、`attention_mask` 和 `labels`，没有把 `token_type_ids` 传入 `model.py` 的 `forward()`。

## 模型结构

主训练模型在 [model.py](model.py)：

```python
class BertClassifierModel(nn.Module):
    def __init__(self, num_labels=config.classname_len):
        self.bert_model = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_hidden_state = bert_output.last_hidden_state[:, 0]
        return self.linear(self.dropout(cls_hidden_state))
```

当前默认预训练模型是：

```python
model_name = "hfl/chinese-macbert-base"
```

模型逻辑是：

1. 输入文本先经过 `BertTokenizer` 转为 `input_ids` 和 `attention_mask`。
2. `BertModel` 输出每个 token 的上下文表示。
3. 取第 0 个位置，即 `[CLS]` 的 `last_hidden_state[:, 0]`。
4. 经过 dropout 和线性层输出二分类 logits。
5. 训练时使用 `CrossEntropyLoss`。

注意：当前 `model.py` 中没有用 `torch.no_grad()` 包住 BERT 主干，因此默认会更新 BERT backbone 和分类头参数，是标准微调路径。

## 训练

训练入口是 [train.py](train.py)。

常规训练命令：

```powershell
cd E:\Projects\bert
& "E:\miniconda\envs\NLP\python.exe" .\train.py
```

默认训练配置来自 [config.py](config.py)：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `batch_size` | `64` | 训练/验证 batch size |
| `num_workers` | `4` | DataLoader worker 数 |
| `learning_rate` | `2e-5` | AdamW 学习率 |
| `num_epochs` | `15` | 训练轮数 |
| `dropout` | `0.1` | 分类头 dropout |
| `max_length` | `128` | 文本最大 token 长度 |
| `warmup_ratio` | `0.1` | warmup 比例 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值 |

训练脚本会：

1. 构建 `train` 和 `validation` DataLoader。
2. 初始化 `BertClassifierModel`。
3. 使用 `AdamW` 优化器。
4. 使用 `get_linear_schedule_with_warmup` 做学习率调度。
5. 每个 epoch 输出训练集和验证集 loss/acc。
6. 保存 `checkpoints/last.pt`。
7. 当验证准确率刷新时保存 `checkpoints/best.pt`。

快速 smoke test 可以限制 batch 数：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\train.py --max_train_batches 2 --max_valid_batches 1
```

可选参数：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\train.py --checkpoint_dir checkpoints --device cuda
```

如果机器没有可用 GPU，可以显式使用 CPU：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\train.py --device cpu
```

## 本地 BERT 推理

本地推理入口是 [predict.py](predict.py)。它会加载：

```text
checkpoints/best.pt
```

并返回预测类别 id。

运行示例：

```powershell
cd E:\Projects\bert
& "E:\miniconda\envs\NLP\python.exe" .\predict.py
```

在代码中调用：

```python
from predict import predict

result = predict({"text": "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"})
print(result)  # 0 或 1
```

注意：`predict.py` 依赖已经训练好的 `checkpoints/best.pt`。如果文件不存在，需要先运行训练脚本生成 checkpoint。

## 大模型分类器

[llm_predict.py](llm_predict.py) 使用 OpenAI SDK 调用 DashScope 的 OpenAI 兼容接口：

```python
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

当前使用模型：

```python
model = "deepseek-v4-flash"
```

使用前需要配置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="你的百炼 API Key"
```

运行测试：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\llm_predict.py
```

`predict_llm(...)` 的输入输出形式：

```python
from llm_predict import predict_llm

res = predict_llm({"text": "我非常喜欢这个电影"})
print(res)  # "0" 或 "1"
```

提示词要求大模型严格只输出 `0` 或 `1`：

| 输出 | 含义 |
|---:|---|
| `0` | 差评 |
| `1` | 好评 |

## FastAPI 后端

后端入口是 [app.py](app.py)，接口为：

```text
POST /predict
```

请求体：

```json
{
  "text": "我非常喜欢这个电影"
}
```

返回体：

```json
{
  "label": "这是一个正面评价"
}
```

启动服务：

```powershell
cd E:\Projects\bert
& "E:\miniconda\envs\NLP\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

当前 [app.py](app.py) 中实际使用的是大模型分类路径：

```python
result = int(predict_llm(text))
```

本地 BERT 推理路径保留在代码中，但目前被注释：

```python
# result = predict(text)
```

因此启动 API 前需要保证 `DASHSCOPE_API_KEY` 已配置。如果要改回本地 BERT 推理，需要把 `predict_llm` 那行替换为 `predict(text)`，并确保 `checkpoints/best.pt` 存在。

## Streamlit 前端

前端入口是 [fronted.py](fronted.py)。它会向本地 FastAPI 服务发送请求：

```python
url = "http://127.0.0.1:8000/predict"
```

启动顺序：

1. 先启动 FastAPI：

```powershell
& "E:\miniconda\envs\NLP\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
```

2. 再启动 Streamlit：

```powershell
& "E:\miniconda\envs\NLP\python.exe" -m streamlit run .\fronted.py
```

页面会显示文本输入框，点击提交后返回：

```text
耗时：x.xxxs
预测结果：这是一个正面评价 / 这是一个负面评价
```

## 手写 BERT 实现

[bert.py](bert.py) 是一个教学用的纯 PyTorch BERT 实现，包含：

| 类 | 作用 |
|---|---|
| `BertConfig` | BERT 超参数配置 |
| `BertEmbeddings` | token、position、token type embedding |
| `BertSelfAttention` | 多头自注意力 |
| `BertSelfOutput` | attention 后的线性层、dropout、残差和 LayerNorm |
| `BertAttention` | attention 子模块封装 |
| `BertIntermediate` | FFN 第一段，`hidden_size -> intermediate_size` |
| `BertOutput` | FFN 第二段，`intermediate_size -> hidden_size` |
| `BertLayer` | 一个完整 Transformer Encoder block |
| `BertEncoder` | 多层 `BertLayer` 堆叠 |
| `BertPooler` | 取 `[CLS]` 并做 `Linear + Tanh` |
| `BertModel` | Embedding + Encoder + Pooler |
| `BertForSequenceClassification` | BERT + 分类头 |

运行 smoke test：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\bert.py
```

该脚本会随机生成 `input_ids` 和 `labels`，验证手写模型可以完成前向传播并输出：

```text
loss: ...
logits shape: (2, 2)
```

这个文件适合用于理解：

1. `input_ids` 如何变成 embedding。
2. attention mask 如何扩展成 `[batch, 1, 1, seq_len]`。
3. Q/K/V 如何拆成多头。
4. attention scores 的形状为什么是 `[batch, heads, seq_len, seq_len]`。
5. `[CLS]` 如何进入句子分类头。
6. Hugging Face BERT 常见模块命名方式。

## 常见问题

### 1. 运行 `predict.py` 报找不到 `checkpoints/best.pt`

原因：还没有训练并保存最优 checkpoint。

处理方式：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\train.py
```

或先用少量 batch 生成测试 checkpoint：

```powershell
& "E:\miniconda\envs\NLP\python.exe" .\train.py --max_train_batches 2 --max_valid_batches 1
```

### 2. API 直接启动失败或返回错误

当前 API 使用的是 `predict_llm(text)`，需要：

1. 已安装 `openai`。
2. 已配置 `DASHSCOPE_API_KEY`。
3. DashScope OpenAI 兼容 endpoint 可访问。
4. `deepseek-v4-flash` 模型名在当前账号/服务中可用。

### 3. FastAPI 和 Streamlit 的启动顺序

必须先启动 FastAPI，再启动 Streamlit。因为 `fronted.py` 会请求：

```text
http://127.0.0.1:8000/predict
```

如果后端没启动，前端会进入异常分支并显示网络错误提示。

### 4. 训练时显存不足

优先降低 [config.py](config.py) 中的：

```python
batch_size = 64
max_length = 128
```

例如把 `batch_size` 改为 `16` 或 `8`。如果仍然不够，再考虑减少 `max_length`。

### 5. `token_type_ids` 为什么没有传给模型

`dataset.py` 的 batch 中包含 `token_type_ids`，但 `model.py` 的 `forward()` 只接收：

```python
def forward(self, input_ids, attention_mask):
```

所以训练脚本在 `move_batch_to_device(...)` 中只保留了 `input_ids`、`attention_mask` 和 `labels`。这是当前训练链路的实际接口约定。

## 推荐工作流

首次运行建议按下面顺序：

```powershell
cd E:\Projects\bert

# 1. 安装依赖
& "E:\miniconda\envs\NLP\python.exe" -m pip install -r requirements.txt

# 2. 快速检查训练链路
& "E:\miniconda\envs\NLP\python.exe" .\train.py --max_train_batches 2 --max_valid_batches 1

# 3. 本地 BERT 推理
& "E:\miniconda\envs\NLP\python.exe" .\predict.py

# 4. 如果使用 LLM/API 路径，先配置 DashScope Key
$env:DASHSCOPE_API_KEY="你的百炼 API Key"

# 5. 启动后端
& "E:\miniconda\envs\NLP\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000

# 6. 新开一个 PowerShell，启动前端
& "E:\miniconda\envs\NLP\python.exe" -m streamlit run .\fronted.py
```

