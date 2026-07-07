from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# BERT 的所有超参数集中放在配置类里，后面的 Embedding、Attention、Encoder、
# 分类头都会从这个对象读取维度、层数、dropout 等设置。
@dataclass
class BertConfig:
    # 词表大小。中文 BERT 常见词表大小是 21128，每个 token id 都会映射到一个向量。
    vocab_size: int = 21128
    # 隐藏层维度，也就是每个 token 在模型内部的向量长度。
    hidden_size: int = 768
    # Transformer Encoder 层数，BERT-base 通常是 12 层。
    num_hidden_layers: int = 12
    # 多头注意力的头数。hidden_size 必须能被 num_attention_heads 整除。
    num_attention_heads: int = 12
    # 前馈网络中间层维度，BERT-base 通常是 4 * hidden_size。
    intermediate_size: int = 3072
    # 普通隐藏层 dropout，用在 embedding、attention output、feed-forward output 等位置。
    hidden_dropout_prob: float = 0.1
    # 注意力概率上的 dropout，用在 softmax 后的 attention_probs。
    attention_probs_dropout_prob: float = 0.1
    # 最大位置编码长度，输入序列长度不能超过这个值。
    max_position_embeddings: int = 512
    # segment/token type 的种类数。句子对任务通常用 0 和 1 两类。
    type_vocab_size: int = 2
    # LayerNorm 的数值稳定项。
    layer_norm_eps: float = 1e-12
    # padding token 的 id，用来生成 attention_mask，并让 padding embedding 初始化为 0。
    pad_token_id: int = 0
    # 分类任务的类别数。做 TNEWS 时应改成 15。
    num_labels: int = 2

    def __post_init__(self) -> None:
        # 多头注意力会把 hidden_size 均分到每个 head，所以这里先做配置合法性检查。
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")


# 负责把离散 token id 变成连续向量，并加上位置编码和句段编码。
class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # token embedding: [batch, seq_len] -> [batch, seq_len, hidden_size]。
        # padding_idx 指定后，padding token 对应的 embedding 在初始化时会被置零。
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        # position embedding 表示 token 在序列中的位置，例如第 0、1、2 个 token。
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
        )
        # token type embedding 用来区分句子 A / 句子 B；单句分类任务通常全是 0。
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size,
        )
        # BERT 原论文使用 embedding 相加后再做 LayerNorm 和 dropout。
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 输入变量：
        # input_ids: token id 矩阵，整数张量，形状 [batch_size, seq_len]。
        #   每个数字表示一个词/字/subword 在词表中的编号。
        # token_type_ids: 句段 id 矩阵，整数张量，形状 [batch_size, seq_len]。
        #   句子对任务中通常 0 表示句子 A，1 表示句子 B；单句分类可不传。
        # 输出形状：[batch_size, seq_len, hidden_size]。
        # input_ids 形状是 [batch_size, seq_len]。
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 如果调用方没有传 token_type_ids，默认所有 token 都属于第 0 段。
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 生成每个位置的位置 id: [0, 1, ..., seq_len - 1]，
        # 再扩展到 batch 维度，形状变成 [batch_size, seq_len]。
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)

        # 三种 embedding 逐元素相加，得到最终送入 Encoder 的 token 表示。
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


# 实现单个 Transformer 层里的多头自注意力部分。
class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        # 每个 head 分到的维度。例如 hidden_size=768、heads=12 时，每个 head 是 64 维。
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Q/K/V 三个线性投影都从 hidden_size 投影到 all_head_size。
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [batch, seq_len, all_head_size]。
        # 先拆成 [batch, seq_len, num_heads, head_dim]。
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        # 再换成 [batch, num_heads, seq_len, head_dim]，方便每个 head 独立算注意力。
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入变量：
        # hidden_states: 当前层输入的 token 表示，形状 [batch_size, seq_len, hidden_size]。
        #   对第一层来说，它来自 BertEmbeddings；对后续层来说，它来自上一层 BertLayer。
        # attention_mask: 可选的 padding mask，通常形状 [batch_size, 1, 1, seq_len]。
        #   它会广播到 [batch_size, num_heads, seq_len, seq_len]。
        #   有效 token 位置为 0，padding 位置为一个很大的负数，用来压低 softmax 概率。
        # 输出：
        # context_layer: 注意力加权后的上下文表示，形状 [batch_size, seq_len, hidden_size]。
        # attention_probs: 每个 head 的注意力概率，形状 [batch_size, num_heads, seq_len, seq_len]。
        # hidden_states: [batch, seq_len, hidden_size]。
        # 线性投影后拆成多头格式。
        # query_layer: [batch, num_heads, seq_len, head_dim]。
        query_layer = self._transpose_for_scores(self.query(hidden_states))
        key_layer = self._transpose_for_scores(self.key(hidden_states))
        value_layer = self._transpose_for_scores(self.value(hidden_states))

        # Q 和 K 做点积，得到每个 token 对其他 token 的注意力分数。
        # attention_scores: [batch, heads, seq_len, seq_len]。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 按 head 维度开方缩放，避免点积值过大导致 softmax 梯度过小。
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # attention_mask 已经被扩展成可广播到 attention_scores 的形状。
        # padding 位置会加上极小值，使 softmax 后概率接近 0。
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对最后一维做 softmax，表示每个 query token 应该关注各 key token 的概率。
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 用注意力概率加权 V，得到每个 head 的上下文表示。
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把多头结果从 [batch, heads, seq_len, head_dim]
        # 合并回 [batch, seq_len, hidden_size]。
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # [batch, seq_len, heads, head_dim]
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        # 把多头结果合并回 [batch, seq_len, hidden_size]。
        return context_layer, attention_probs


# 自注意力之后的输出子层：线性变换、dropout、残差连接、LayerNorm。
class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # 输入变量：
        # hidden_states: self-attention 的输出，形状 [batch_size, seq_len, hidden_size]。
        # input_tensor: 进入 self-attention 前的原始输入，形状 [batch_size, seq_len, hidden_size]。
        #   它用于残差连接 hidden_states + input_tensor。
        # 输出形状：[batch_size, seq_len, hidden_size]。
        # input_tensor 是进入 self-attention 前的原始输入，用于残差连接。
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


# 把 self-attention 和 attention output 组合成完整的注意力子模块。
class BertAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入变量：
        # hidden_states: 当前 Transformer 层的输入，形状 [batch_size, seq_len, hidden_size]。
        # attention_mask: 可选 padding mask，形状通常是 [batch_size, 1, 1, seq_len]。
        # 输出：
        # attention_output: attention 子层经过残差和 LayerNorm 后的输出，
        #   形状 [batch_size, seq_len, hidden_size]。
        # attention_probs: 注意力概率，形状 [batch_size, num_heads, seq_len, seq_len]。
        self_output, attention_probs = self.self(hidden_states, attention_mask)
        # attention_output 是经过残差和 LayerNorm 后的结果，供后面的前馈网络使用。
        attention_output = self.output(self_output, hidden_states)
        return attention_output, attention_probs


# Transformer 层中的前馈网络第一段：hidden_size -> intermediate_size -> GELU。
class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入变量：
        # hidden_states: attention 子层输出，形状 [batch_size, seq_len, hidden_size]。
        # 输出形状：[batch_size, seq_len, intermediate_size]。
        # 这一层把每个 token 的向量从 hidden_size 升维到 intermediate_size，再经过 GELU。
        return self.activation(self.dense(hidden_states))


# Transformer 层中的前馈网络第二段：intermediate_size -> hidden_size，
# 然后再做 dropout、残差连接和 LayerNorm。
class BertOutput(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # 输入变量：
        # hidden_states: 前馈网络第一段 BertIntermediate 的输出，
        #   形状 [batch_size, seq_len, intermediate_size]。
        # input_tensor: 进入前馈网络前的 attention_output，
        #   形状 [batch_size, seq_len, hidden_size]，用于残差连接。
        # 输出形状：[batch_size, seq_len, hidden_size]。
        # input_tensor 是进入前馈网络前的 attention_output，用于第二个残差连接。
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


# 一个完整的 Transformer Encoder block：
# Multi-Head Self-Attention -> Add & Norm -> Feed Forward -> Add & Norm。
class BertLayer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入变量：
        # hidden_states: 当前 Encoder block 的输入，形状 [batch_size, seq_len, hidden_size]。
        # attention_mask: 可选 padding mask，形状通常是 [batch_size, 1, 1, seq_len]。
        # 输出：
        # layer_output: 这个 Transformer block 的输出，形状 [batch_size, seq_len, hidden_size]。
        # attention_probs: 该层 self-attention 的注意力概率，
        #   形状 [batch_size, num_heads, seq_len, seq_len]。
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        # 前馈网络先升维到 intermediate_size，再投影回 hidden_size。
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


# 堆叠多个 BertLayer，形成 BERT 的主体 Encoder。
class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # ModuleList 会正确注册每一层参数，使其参与训练和保存。
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, ...] | None, tuple[torch.Tensor, ...] | None
    ]:
        # 输入变量：
        # hidden_states: Encoder 的初始输入，形状 [batch_size, seq_len, hidden_size]。
        #   一般来自 BertEmbeddings。
        # attention_mask: 可选 padding mask，形状通常是 [batch_size, 1, 1, seq_len]。
        # output_attentions: 是否保存并返回每一层的 attention_probs。
        # output_hidden_states: 是否保存并返回 embedding 输出和每一层输出。
        # 输出：
        # hidden_states: 最后一层 Encoder 输出，形状 [batch_size, seq_len, hidden_size]。
        # all_hidden_states: 如果启用，tuple 长度为 num_hidden_layers + 1，
        #   每个元素形状 [batch_size, seq_len, hidden_size]。
        # all_attentions: 如果启用，tuple 长度为 num_hidden_layers，
        #   每个元素形状 [batch_size, num_heads, seq_len, seq_len]。
        # 只有调用方需要时才收集中间层输出，避免无意义地占用显存。
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)

            if output_attentions:
                # attention_probs 可用于可视化每一层每个 head 的注意力分布。
                all_attentions.append(attention_probs)

        if output_hidden_states:
            # 最后一层输出也加入 hidden_states 列表。
            all_hidden_states.append(hidden_states)

        return (
            hidden_states,
            tuple(all_hidden_states) if all_hidden_states is not None else None,
            tuple(all_attentions) if all_attentions is not None else None,
        )


# Pooler 取 [CLS] 位置的向量，并通过一层 Linear + Tanh 得到句级表示。
class BertPooler(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入变量：
        # hidden_states: Encoder 最后一层输出，形状 [batch_size, seq_len, hidden_size]。
        #   第 0 个 token 通常是 [CLS]，用于代表整句。
        # 输出形状：[batch_size, hidden_size]。
        # hidden_states[:, 0] 对应每个样本的第一个 token，通常是 [CLS]。
        cls_token = hidden_states[:, 0]
        # 输出形状：[batch_size, hidden_size]。
        return self.activation(self.dense(cls_token))


# 基础 BERT 模型：Embedding + Encoder + Pooler，不包含具体任务头。
class BertModel(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # 初始化所有子模块参数，保持和 BERT 常见初始化方式一致。
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # Linear 和 Embedding 使用均值 0、标准差 0.02 的正态初始化。
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    # padding token 不携带语义，初始化后显式置零。
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm 通常初始化为恒等变换：weight=1，bias=0。
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _extend_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # 原始 attention_mask: [batch, seq_len]，1 表示有效 token，0 表示 padding。
        # 扩展后形状为 [batch, 1, 1, seq_len]，可广播到每层每个 head 的注意力分数。
        extended_mask = attention_mask[:, None, None, :].to(
            dtype=self.embeddings.word_embeddings.weight.dtype
        )
        # 有效位置变成 0，不影响 attention_scores；
        # padding 位置变成 dtype 能表示的最小值，softmax 后几乎为 0。
        return (1.0 - extended_mask) * -10000.0
        # return (1.0 - extended_mask) * torch.finfo(extended_mask.dtype).min

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None]:
        # 输入变量：
        # input_ids: token id 矩阵，整数张量，形状 [batch_size, seq_len]。
        # attention_mask: 可选 padding mask，形状 [batch_size, seq_len]。
        #   1 表示有效 token，0 表示 padding；如果不传，会根据 pad_token_id 自动生成。
        # token_type_ids: 可选句段 id，形状 [batch_size, seq_len]。
        #   单句任务通常全 0；句子对任务中 0/1 用来区分两段文本。
        # output_attentions: 是否在返回字典中包含每一层 attention_probs。
        # output_hidden_states: 是否在返回字典中包含 embedding 输出和每一层 hidden states。
        # 返回字典：
        # last_hidden_state: [batch_size, seq_len, hidden_size]。
        # pooler_output: [batch_size, hidden_size]。
        # hidden_states: 启用时为 tuple，否则为 None。
        # attentions: 启用时为 tuple，否则为 None。
        # 如果没有显式传 attention_mask，则根据 pad_token_id 自动生成。
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # 第一步：token id -> embedding 表示。
        # embedding_output 形状为 [batch, seq_len, hidden_size]。
        embedding_output = self.embeddings(input_ids, token_type_ids)
        # 第二步：把 padding mask 转换成 attention scores 可以直接相加的形式。
        extended_attention_mask = self._extend_attention_mask(attention_mask)
        # 第三步：经过多层 Transformer Encoder，得到每个 token 的上下文表示。
        sequence_output, hidden_states, attentions = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )  # 【batch, seq_len, hidden_size】
        # 第四步：取 [CLS] 位置得到句级 pooled output。
        pooled_output = self.pooler(sequence_output)

        return {
            # 每个 token 的最后一层隐藏状态，形状 [batch, seq_len, hidden_size]。
            "last_hidden_state": sequence_output,
            # [CLS] 的池化表示，常用于分类任务，形状 [batch, hidden_size]。
            "pooler_output": pooled_output,
            # 可选：每一层的 hidden states。
            "hidden_states": hidden_states,
            # 可选：每一层的 attention probabilities。
            "attentions": attentions,
        }


# 在基础 BERT 上加一个线性分类头，用于句子/文本分类任务。
class BertForSequenceClassification(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器输入是 pooled_output，输出维度是类别数 num_labels。
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 只初始化分类头参数；BERT 主干已经在 BertModel 内部初始化。
        self.classifier.apply(self.bert._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,  # 形状 [batch, seq_len]。
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        # 输入变量：
        # input_ids: token id 矩阵，整数张量，形状 [batch_size, seq_len]。
        # attention_mask: 可选 padding mask，形状 [batch_size, seq_len]。
        #   1 表示有效 token，0 表示 padding。
        # token_type_ids: 可选句段 id，形状 [batch_size, seq_len]。
        # labels: 可选分类标签，形状 [batch_size]。
        #   每个元素是类别 id，取值范围为 0 到 num_labels - 1。
        # 返回字典：
        # loss: 如果传入 labels，则是标量损失；否则为 None。
        # logits: 分类器原始输出，形状 [batch_size, num_labels]。
        # 先通过 BERT 主干拿到 pooled_output。
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # 分类任务通常对 pooled_output 再做一次 dropout。
        pooled_output = self.dropout(outputs["pooler_output"])  # 【batch, hidden_size】
        # logits 形状是 [batch_size, num_labels]，还没有经过 softmax。
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # CrossEntropyLoss 内部会做 log_softmax，所以这里传原始 logits。
            # labels 形状通常是 [batch_size]，取值范围是 0 到 num_labels - 1。
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    # 简单 smoke test：构造随机 token id 和标签，验证模型前向传播可以跑通。
    config = BertConfig(
        num_labels=2,
    )
    model = BertForSequenceClassification(config)
    # 假设 batch_size=2、seq_len=16。
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([0, 1])

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print("loss:", float(outputs["loss"].detach()))
    print("logits shape:", tuple(outputs["logits"].shape))
