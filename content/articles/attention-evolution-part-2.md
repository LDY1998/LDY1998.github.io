---
title: "注意力演进史（二）：Transformer 如何分化出 GPT 与 BERT"
category: "人工智能"
date: "2026.08.23"
featured: false
---

上一篇从基于 RNN 的 Seq2Seq 讲起。在这套架构中，Encoder 负责读取输入，Decoder 负责生成输出；Attention 的出现，让 Decoder 在生成每个词时，可以动态读取 Encoder 保留下来的信息。后来，Transformer 移除了循环结构，并用 Attention 重建了序列内部以及 Encoder 与 Decoder 之间的信息流。

不过，“使用 Attention”并没有完整说明一个模型是怎样工作的。Attention 只定义了一种动态读取信息的方法，却没有规定信息应该从哪里读取：一个位置可以读取同一序列中的其他位置，也可以读取另一段序列；可以看到完整上下文，也可以只能看到已经出现的内容；还可以通过多个 Attention Head，同时从不同表示空间中建立联系。

这些变化并不是为了制造更多术语，而是在解决不同的信息流需求。要表示一段完整的输入，模型通常需要同时利用左右上下文；要从左到右生成文本，就必须遮住尚未出现的内容，避免提前看到答案；要根据一段输入生成另一段输出，Decoder 又需要跨越两条序列，从 Encoder 中读取信息。Self-Attention、Cross-Attention、双向注意力、因果注意力和 Multi-Head Attention，正是从这些需求中产生的。

原始 Transformer 将其中几种形式组合在了一起：Encoder 完整读取输入，Decoder 只能读取已经生成的内容，同时又通过 Cross-Attention 读取 Encoder 的输出。因此，Transformer 从一开始就不是“只有一种 Attention”的固定结构，而是一套组织信息流的方法。

当研究者开始用大规模文本预训练通用语言模型时，这套结构又被进一步拆分和重组。一条路线把语言学习变成“完形填空”，让模型结合左右上下文恢复被遮住的内容，这条路线以 BERT 为代表；另一条路线把语言学习变成“不断续写”，让模型根据已有内容持续预测下一个 token，这条路线以 GPT 为代表；T5、BART 等模型则保留了完整的 Encoder–Decoder，用于从一段序列生成另一段序列。

本文将沿着这条线索展开：先讨论不同 Attention 分别解决什么信息读取问题，再分析它们如何组成原始 Transformer，最后看看 BERT、GPT 和 T5/BART 如何通过改变信息的可见范围、输入输出的组织方式与训练目标，走向三条不同的架构路线。

## Attention 的不同形态：谁在向谁读取信息？

上一篇介绍 Attention 时，我们把它概括成一次动态的信息读取：当前位置产生 Query，与其他位置的 Key 比较，再根据匹配结果汇总相应的 Value。但要真正把 Attention 放进模型，还需要回答几个更具体的问题：Query、Key 和 Value 来自同一段序列，还是不同序列？每个位置能够看到全部内容，还是只能看到一部分？同一次读取，又是否需要在多个表示空间中并行进行？

这些问题对应了几组经常一起出现的概念。Self-Attention 与 Cross-Attention 区分信息来自哪里；双向注意力与因果注意力规定哪些位置可以被看到；Multi-Head Attention 则让模型同时进行多组不同的信息读取。它们并不是互斥的选项，而是可以相互组合的设计维度。

![Attention 的三个设计维度](/images/attention-evolution-part-2/attention-design-dimensions.png)

*图：Attention 的三个设计维度。信息来源、可见范围和读取视角可以相互组合。*

### Self-Attention：从同一段序列中读取信息

在 Self-Attention 中，Query、Key 和 Value 都由同一组隐藏表示 $X$ 投影得到：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

这里的“Self”并不是说每个 token 只关注自己，而是说提出问题和提供信息的位置来自同一段序列。仍以上一篇的句子为例：

```text
The cat sat on the mat because it was tired.
```

当模型更新 `it` 的表示时，可以用 `it` 产生的 Query 与整句话中各个位置的 Key 比较。如果 `cat` 与当前问题更相关，`cat` 对应的 Value 就会以更大的权重进入 `it` 的新表示。通过这种方式，原本彼此独立的 token 表示开始交换信息，并逐渐融入上下文。

不过，Self-Attention 只说明信息来自同一段序列，并没有规定 `it` 究竟能看到哪些词。它既可以查看左右两侧的完整上下文，也可以被限制为只能查看自己之前的内容。这个范围由 Attention Mask 决定。

### 双向注意力：同时利用左右上下文

如果不加入因果限制，序列中的每个位置都可以读取其他位置的信息。处理下面这句话时：

```text
The [MASK] sat on the mat.
```

模型可以同时利用 `[MASK]` 左侧的 `The` 和右侧的 `sat on the mat`，判断这里最可能缺少 `cat` 一类的名词。这里所谓的“双向”，并不是像双向 RNN 那样分别从左向右和从右向左计算两遍，而是每个位置都能直接与左右两侧的位置建立联系。

完整上下文很适合构建文本表示。无论要判断一句话的情感、抽取其中的人名，还是为一段文本生成用于检索的向量，模型通常都希望综合目标位置前后的信息。BERT 后来采用的正是这种可见方式。

但它不适合直接训练从左到右的生成模型。如果模型在预测下一个词时已经通过 Attention 看到了后面的答案，预测任务就失去了意义。要让模型学会真正的续写，信息流必须受到进一步限制。
### 因果注意力：只能读取已经出现的内容

因果注意力（Causal Attention）为 Self-Attention 加入一个遮罩，使第 $i$ 个位置只能读取自己以及它之前的位置。其计算可以写成：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M
\right)V,
$$

其中，遮罩矩阵 $M$ 可以表示为：

$$
M_{ij}=
\begin{cases}
0, & j\leq i,\\
-\infty, & j>i.
\end{cases}
$$

Softmax 之后，未来位置得到的权重会变成零。于是，同一句话中不同位置的可见范围呈现出逐步扩大的形状：

```text
The        → The
cat        → The cat
sat        → The cat sat
on         → The cat sat on
```

训练时，模型会用当前位置形成的表示预测下一个 token。例如，读取 `The cat sat on the` 之后预测 `mat`。把每个位置上的预测组合起来，就得到自回归语言模型的分解形式：

$$
p(x_1,\ldots,x_n)
=\prod_{t=1}^{n}p(x_t\mid x_{<t}).
$$

虽然信息只能从左向右流动，训练时各个位置的 Attention 仍然可以通过矩阵运算并行计算，只需要用遮罩阻断通向未来的连接；真正生成文本时，由于下一个 token 依赖已经生成的结果，模型才必须逐步向后生成。GPT 采用的就是这种信息可见方式。

### Cross-Attention：从另一段序列中读取信息

Self-Attention 让一段序列内部的位置相互交流，但机器翻译还需要解决另一个问题：正在生成译文的 Decoder，怎样读取原文的信息？

Cross-Attention 的 Query 与 Key、Value 来自不同的信息源。假设 $X$ 是 Encoder 产生的输入表示，$Y$ 是 Decoder 当前的隐藏表示，那么可以写成：

$$
Q=YW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

以英译中为例，Encoder 先处理完整的英文句子；Decoder 在生成每个中文词时，用自己当前的状态产生 Query，再与英文表示中的 Key 比较，读取相关的 Value。生成一个人名时，它可能更多地读取原文中的对应名字；生成一个动词时，则可能转向原文中的动作及其上下文。

因此，Self-Attention 解决的是“一段序列内部如何交流”，Cross-Attention 解决的是“一段序列如何读取另一段序列”。这种机制不仅可以连接原文与译文，也可以让文本读取图像、音频或其他形式的输入；它的关键始终是 Query 与 Key、Value 来自不同的信息源。

### Multi-Head Attention：同时建立多组关系

到目前为止，我们都像是在讨论一次 Attention。但在实际的 Transformer 中，模型通常会把隐藏表示投影到多个子空间，分别进行 Attention，再把结果合并：

$$
\operatorname{head}_i
=\operatorname{Attention}
\left(QW_i^Q,KW_i^K,VW_i^V\right),
$$

$$
\operatorname{MultiHead}(Q,K,V)
=\operatorname{Concat}
\left(\operatorname{head}_1,\ldots,\operatorname{head}_h\right)W_O.
$$

Multi-Head Attention 并不是把同一次 Attention 原样重复多遍。每个 Head 都有自己独立的 $W_i^Q$、$W_i^K$ 和 $W_i^V$，因此会从隐藏表示中提取不同的特征，并用不同的方式判断两个位置是否相关。同一个词在一个 Head 中可能主要表现出“它指代什么”，在另一个 Head 中则可能表现出“它在句子中承担什么语法角色”。

来看一个假想的例子。假设一层没有因果遮罩的 Self-Attention 正在处理：

```text
The dog did not cross the street because it was too tired.
```

当模型更新 `it` 的表示时，不同 Head 可能形成下面这样的注意力分布。表中的数字只是为了说明机制，并不是某个真实模型的测量结果：

| Head | 权重较高的位置 | 可能捕捉的关系 |
|---|---|---|
| Head 1 | `dog` 0.60、`it` 0.16、`street` 0.08 | `it` 更可能指向 `dog`，而不是 `street` |
| Head 2 | `was` 0.46、`tired` 0.25、`it` 0.14 | `it` 与后面的谓语部分构成局部语法关系 |
| Head 3 | `because` 0.41、`cross` 0.22、`tired` 0.19 | 疲倦是没有过马路的原因，句子中存在因果联系 |

对第 $r$ 个 Head 来说，这些权重仍然来自 Query 与 Key 的匹配：

$$
\alpha_{\text{it},j}^{(r)}
=\operatorname{softmax}_j\!\left(
\frac{q_{\text{it}}^{(r)}\left(k_j^{(r)}\right)^{\mathsf T}}{\sqrt{d_h}}
\right).
$$

接下来，每个 Head 分别用自己的权重汇总 Value：

$$
z_{\text{it}}^{(r)}
=\sum_j \alpha_{\text{it},j}^{(r)}v_j^{(r)}.
$$

例如，Head 1 的输出会包含较多来自 `dog` 的信息，Head 2 会汇入更多来自 `was` 和 `tired` 的信息，Head 3 则更多地综合 `because`、`cross` 和 `tired`。需要注意，Head 输出的是一组加权后的向量，而不是“`dog`”或者“因果关系”这样的文字标签。

得到各个 Head 的结果后，模型不会对它们简单取平均，也不会让它们投票决定唯一答案，而是先把这些向量拼接起来，再通过一个可训练的输出矩阵 $W_O$：

$$
z_{\text{it}}
=\operatorname{Concat}\left(
z_{\text{it}}^{(1)},z_{\text{it}}^{(2)},\ldots,z_{\text{it}}^{(h)}
\right),
$$

$$
o_{\text{it}}=z_{\text{it}}W_O.
$$

拼接负责保留各个 Head 读取到的信息，$W_O$ 则学习如何在这些信息之间进行组合，并把结果重新映射回模型的隐藏表示空间。经过这一步，`it` 的新表示便可能同时包含“它更可能指向 `dog`”“它与 `was tired` 构成谓语关系”以及“疲倦解释了前面的行为”等信息。这里的“包含”仍然是一种便于理解的概括；真实信息分布在向量的许多维度中，并不会以几条清晰的句子储存。

这也说明了多头机制相对单头机制的意义。单个 Head 当然也可以同时关注多个位置，但它只有一套 Query、Key、Value 投影和一组归一化后的注意力权重。多个 Head 可以使用多套不同的匹配标准，分别读取互补的信息，再由模型学习如何整合。

不过，这种分工不是设计者提前指定的。训练过程中，有些 Head 可能更关注邻近位置，有些可能表现出长距离、指代或句法相关的模式，但不能认定每个 Head 都必然学到一种固定、清晰且可以用语言命名的功能；不同 Head 之间也可能存在重叠和冗余。上面的三个 Head 是对多头机制的直观演示，而不是对真实模型内部结构的严格解释。

Multi-Head 与前面几组概念仍然不是互斥关系：双向 Self-Attention 可以使用多个 Head，因果 Self-Attention 可以使用多个 Head，Cross-Attention 同样可以使用多个 Head。把这些维度放在一起看，它们分别回答了三个问题：

| 设计维度 | 要回答的问题 | 主要形式 |
|---|---|---|
| 信息来源 | Query 在读取谁？ | Self-Attention / Cross-Attention |
| 可见范围 | 每个位置可以看到哪里？ | 双向注意力 / 因果注意力 |
| 读取方式 | 是否并行学习多组匹配关系？ | Single-Head / Multi-Head Attention |

原始 Transformer 正是这些形式的一种组合：Encoder 使用双向 Multi-Head Self-Attention，Decoder 使用因果 Multi-Head Self-Attention，并通过 Multi-Head Cross-Attention 读取 Encoder 的输出。下一节，我们就回到这套最初的结构，看看三种信息流如何共同完成从输入到输出的转换。

## 原始 Transformer：Encoder 与 Decoder 如何协作？

原始 Transformer 延续了 Seq2Seq 的基本任务形式：给定一段输入序列，生成另一段输出序列。与早期基于 RNN 的 Seq2Seq 相比，它没有放弃 Encoder–Decoder 的分工，而是替换了两者处理序列的方式。RNN 依靠隐藏状态逐步向后传递信息，Transformer 则在每一层中使用 Attention，让不同位置直接读取彼此的表示。

以机器翻译为例，Encoder 与 Decoder 中的完整信息流如下图所示：

![原始 Transformer 的 Encoder–Decoder 数据流](/images/attention-evolution-part-2/original-transformer-dataflow.png)

*图：原始 Transformer 的 Encoder–Decoder 结构。Encoder 输出作为 Cross-Attention 的 Key 和 Value，Decoder 当前表示提供 Query。*

这里最重要的变化是，Encoder 不再把整个输入压缩成一个固定向量。它会为输入中的每个位置保留一个上下文化表示，Decoder 则在生成不同词语时，通过 Cross-Attention 动态读取这些表示。

### Encoder：为完整输入建立上下文表示

输入 Encoder 之前，每个 token 会先被转换成词向量，并加入位置信息。位置信息很重要，因为 Self-Attention 本身只根据内容建立联系，并不会天然知道 `cat` 出现在 `The` 之后、`sleeping` 出现在句子末尾。

随后，输入会经过多层结构相同的 Encoder Layer。每一层主要包含两个子层：

1. Multi-Head Self-Attention；
2. Feed-Forward Network。

Self-Attention 负责位置之间的信息交换。由于 Encoder 面对的是一段已经完整给出的输入，它通常不需要因果遮罩：`cat` 可以读取 `sleeping`，`sleeping` 也可以回头读取 `cat`。经过一层层双向 Self-Attention，每个位置都不再只表示当前单词，而是逐渐融入整句话的上下文。

Attention 之后的 Feed-Forward Network 则对每个位置分别进行非线性变换。它不会像 Attention 那样在不同 token 之间传递信息，而是对 Attention 汇总回来的结果做进一步加工。简化地说，Attention 负责“从其他位置读回什么”，Feed-Forward Network 负责“怎样处理已经读回的信息”。

每个子层外还会使用 Residual Connection 和 Layer Normalization。以原始 Transformer 的写法为例，可以概括为：

$$
\operatorname{LayerNorm}\left(x+\operatorname{Sublayer}(x)\right).
$$

Residual Connection 让原来的表示可以绕过子层直接向后传播，Layer Normalization 则帮助控制不同层中表示的尺度，使深层网络更容易训练。后来的 Transformer 在归一化位置上出现了 Pre-Norm 等变化，但不影响这里关注的基本信息流。

经过最后一层 Encoder 后，我们得到的不是一个句子向量，而是一组与输入长度对应的表示：

$$
H=(h_1,h_2,\ldots,h_n).
$$

这些表示共同构成 Decoder 可以查询的信息来源。在翻译 `The cat is sleeping.` 时，$h_{\text{cat}}$ 已经不再只是孤立的 `cat`，其中还融入了它与 `The`、`is` 和 `sleeping` 之间的关系。

### Decoder：一边读取生成历史，一边查询输入

Decoder 的目标不是一次性给出整段译文，而是按照顺序逐个预测 token。它的每一层比 Encoder 多一个子层，主要包含：

1. Causal Multi-Head Self-Attention；
2. Multi-Head Cross-Attention；
3. Feed-Forward Network。

第一个子层处理已经生成的目标序列。假设 Decoder 已经生成了：

```text
猫 正在
```

那么它可以通过 Causal Self-Attention 读取 `猫` 和 `正在`，但不能提前看到后面的 `睡觉`。这个遮罩保证模型只能根据已有译文预测接下来的内容。

第二个子层负责读取 Encoder。Decoder 经过 Causal Self-Attention 后产生 Query，Encoder 的最终输出 $H$ 则提供 Key 和 Value：

$$
Q=YW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

当 Decoder 准备生成 `睡觉` 时，它可以通过 Query 在英文输入中寻找相关信息，并为 `sleeping` 对应的表示分配较高权重。生成 `猫` 时，它可能更多地读取 `cat`；决定语序或时态表达时，又可以综合 `is sleeping` 以及整句上下文。Cross-Attention 因而不是把 Encoder 的结果一次性塞给 Decoder，而是允许 Decoder 在每个位置、每一层根据当前需要重新查询输入。

最后，Feed-Forward Network 对融合了目标端历史和输入信息的表示继续加工。经过多层 Decoder 后，当前位置的隐藏表示会被映射到词表上的分数，再通过 Softmax 得到下一个 token 的概率。

### 生成时，信息怎样一步步流动？

把上面的结构放回完整的翻译过程，可以看到 Decoder 在每一步都进行了两次不同的信息读取：

```text
开始符号
  │
  ├─ Causal Self-Attention：目前还没有中文内容
  ├─ Cross-Attention：读取 The cat is sleeping.
  └─ 预测：猫

猫
  │
  ├─ Causal Self-Attention：读取已经生成的“猫”
  ├─ Cross-Attention：再次查询英文表示
  └─ 预测：正在

猫 正在
  │
  ├─ Causal Self-Attention：读取“猫 正在”
  ├─ Cross-Attention：重点读取 is sleeping
  └─ 预测：睡觉
```

生成过程必须逐步进行，因为后一步依赖前一步实际生成的结果。训练时则不同：正确译文已经全部给出，模型可以把目标序列右移一位作为输入，并借助因果遮罩同时计算所有位置。也就是说，因果关系限制了每个位置能使用的信息，却不妨碍训练时对各个位置进行并行计算。

至此，原始 Transformer 中的三条主要信息流就清楚了：

1. Encoder 内部通过双向 Self-Attention 交流，建立完整输入的上下文表示；
2. Decoder 内部通过 Causal Self-Attention 读取已经生成的内容；
3. Decoder 通过 Cross-Attention 查询 Encoder 的输出，将输入信息带入生成过程。

人们有时会把这种分工简化成“Encoder 负责理解，Decoder 负责生成”。这句话可以帮助形成直觉，却不够准确。Encoder 真正做的是在完整可见的输入上构建上下文表示；Decoder 真正做的是在目标历史和 Encoder 输出的共同条件下预测下一个 token。所谓“理解”与“生成”，是这些信息流和训练任务最终表现出的能力，而不是两个模块预先拥有的功能标签。

原始 Transformer 把三条信息流放进了同一套架构。但当语言模型开始转向大规模预训练时，研究者不再总是需要明确的“输入序列到输出序列”转换：如果只留下能够完整读取上下文的 Encoder，会发生什么？如果去掉 Encoder 和 Cross-Attention，只保留从左到右预测的部分，又会发生什么？这些选择最终把 Transformer 推向了三条不同的路线。

## Transformer 的三条主要路线

原始 Transformer 是为机器翻译设计的，输入和输出有明确的边界：Encoder 读取原文，Decoder 生成译文。但语言模型面对的不只有翻译。文本分类需要从整段输入中提取表示，开放式写作需要根据已有内容持续向后生成，摘要和改写则仍然需要把一段输入转换成另一段输出。

当任务发生变化，原始 Transformer 中的所有部分就不一定都要保留。研究者围绕三种信息流做出了不同取舍，由此形成了三类常见架构：

| 架构 | 信息可见范围 | 典型预训练目标 | 代表模型 | 常见使用方式 |
|---|---|---|---|---|
| Encoder-Only | 每个位置读取完整输入 | Masked Language Modeling | BERT | 分类、抽取、检索与文本表示 |
| Decoder-Only | 每个位置只读取当前及之前的内容 | Next-Token Prediction | GPT | 续写、问答、对话与开放式生成 |
| Encoder–Decoder | Encoder 完整读取输入；Decoder 读取生成历史和 Encoder 输出 | Sequence-to-Sequence、去噪或 Span Corruption | T5、BART | 翻译、摘要、改写与条件生成 |

这张表描述的是三种典型设计，而不是不可跨越的边界。它们使用的基础组件高度相似，真正不同的是 Attention Mask、Cross-Attention 是否存在、输入输出怎样组织，以及模型被要求预测什么。

### Encoder-Only：BERT 路线

Encoder-Only 保留原始 Transformer 的 Encoder，去掉 Decoder 和 Cross-Attention。输入经过多层双向 Self-Attention 后，每个位置都会形成结合完整上下文的表示。

但仅有一套 Encoder 还不够，模型仍然需要一个可以从大规模无标注文本中学习的目标。BERT 的核心做法是 Masked Language Modeling：从句子中选出一部分 token 进行遮盖或替换，再让模型恢复原来的内容。直观上，它很像完形填空：

```text
输入：The [MASK] sat on the mat.
目标：cat
```

为了预测 `[MASK]`，模型可以同时利用左边的 `The` 和右边的 `sat on the mat`。训练信号迫使它学习词语在完整上下文中的含义，以及不同位置之间的关系。原始 BERT 还同时使用了 Next Sentence Prediction，用来判断两段文本在原始语料中是否相邻；不过，真正决定其双向信息流特征的，是 Masked Language Modeling 与 Encoder 的完整可见范围。

预训练完成后，BERT 通常不会像生成模型那样不断输出 token，而是把得到的上下文化表示交给下游任务使用。例如：

```text
文本 → Encoder → 上下文化表示 → 分类器 → 情感类别
文本 → Encoder → 每个位置的表示 → 标注层 → 人名、地点、机构
查询或文档 → Encoder → 向量表示 → 相似度计算 → 检索结果
```

因此，Encoder-Only 的优势并不是抽象意义上的“更懂语言”，而是它能在建立表示时同时使用左右上下文，并为整段文本或每个位置提供可供其他模块使用的表示。这使它很适合分类、匹配、抽取、重排序和检索等任务。

它的限制也来自同一设计。一个位置在训练时可以看到后面的内容，所以模型没有被训练成只依赖历史逐步续写。虽然可以通过特殊方法让 Encoder 参与生成任务，但标准 BERT 并不是天然的自回归生成模型。

### Decoder-Only：GPT 路线

Decoder-Only 这个名字容易产生一个误解：它并不是把原始 Transformer 的 Decoder 原封不动地单独拿出来。原始 Decoder 还包含用于读取 Encoder 的 Cross-Attention；当 Encoder 被去掉后，这部分也随之消失。Decoder-Only 模型主要由 Causal Self-Attention 和 Feed-Forward Network 反复堆叠而成。

它的训练目标比完形填空更直接：给定已经出现的 token，预测下一个 token。

```text
输入：The cat sat on the
目标：mat
```

在一段完整训练文本中，这个目标会出现在每个位置：

```text
The                 → cat
The cat             → sat
The cat sat         → on
The cat sat on      → the
The cat sat on the  → mat
```

模型由此学习下面的概率分解：

$$
p(x_1,\ldots,x_n)
=\prod_{t=1}^{n}p(x_t\mid x_{<t}).
$$

训练和生成因而采用同一种基本形式：训练时学习“接下来最可能出现什么”，生成时则把预测出的 token 放回上下文，继续预测下一个。只要重复这个过程，模型就可以从一句话扩展到一段文章。

更重要的是，Decoder-Only 不要求模型在结构上预先区分“输入”和“输出”。任务描述、示例、用户问题以及模型回答，都可以被写进同一条 token 序列：

```text
任务：判断下面文本的情感。
文本：这部电影非常精彩。
答案：
```

对模型而言，这仍然只是一个尚未结束的上下文。分类可以变成生成类别名称，翻译可以变成生成另一种语言，问答可以变成在问题后继续写出答案。Prompt、Few-Shot Learning 和 In-Context Learning 也可以自然地放进这条序列，而不需要为每一种任务改变模型结构。

不过，这并不意味着 Causal Attention 在所有方面都优于双向 Attention。预测某个位置时，GPT 不能利用尚未出现的右侧内容；如果目标是为一篇已经完整给出的文档构建表示，这种单向可见方式未必是最直接的选择。Decoder-Only 的突出特点，是用一个统一的自回归接口，把训练、任务描述和生成连接在了一起。

### Encoder–Decoder：T5 与 BART 路线

第三条路线保留了原始 Transformer 的整体结构：Encoder 双向读取输入，Decoder 从左到右生成输出，并通过 Cross-Attention 查询 Encoder 的表示。

这类模型特别适合输入与输出边界明确的任务：

```text
翻译：英文文章 → 中文文章
摘要：长文 → 摘要
改写：原句 → 改写后的句子
问答：问题与材料 → 答案
```

与 Decoder-Only 把所有内容放进一条序列不同，Encoder–Decoder 为两部分保留了不同的信息流。Encoder 可以不受因果遮罩限制，先完整分析输入；Decoder 则一边读取已经生成的结果，一边通过 Cross-Attention 回到输入中寻找当前需要的信息。这让模型能够明确地区分“生成的条件”和“需要生成的内容”。

T5 和 BART 都沿用了这套结构，但它们的预训练任务不完全相同。T5 的一种核心训练方式是 Span Corruption：从输入中遮住若干连续片段，让 Decoder 生成被移除的内容；BART 则对文本进行遮盖、删除、打乱等破坏，再让模型重建原文。两者都在构造一种从受损输入到目标输出的序列转换，让 Encoder–Decoder 可以在没有人工任务标注的大规模文本上进行预训练。

这种结构的优势是条件输入与生成输出分工清楚，尤其适合翻译、摘要等序列转换任务。代价则是模型需要维护 Encoder 和 Decoder 两套计算栈，推理流程也比单一的 Decoder 更复杂。在相同参数预算下，参数还需要分配到编码和解码两部分。

### 三条路线究竟差在哪里？

如果只看模块名称，这三种架构似乎是把 Transformer 拆成了不同的部分；如果从信息流出发，它们的区别会更清楚：

![BERT、GPT 与 T5/BART 三条路线的信息流](/images/attention-evolution-part-2/transformer-three-routes.png)

*图：BERT、GPT 与 T5/BART 的信息流。三条路线分别对应完整输入到表示、历史内容到下一个 token，以及完整输入与生成历史共同决定下一个 token。*

BERT 让每个位置看到完整输入，目标是恢复被遮住的内容；GPT 让每个位置只看到过去，目标是持续预测下一个 token；T5 和 BART 则把完整输入与生成历史分开处理，通过 Cross-Attention 连接两者。它们之间真正的分界，不只是保留了 Encoder 还是 Decoder，而是模型在预测时可以使用哪些信息、训练信号要求它学会什么，以及输入与输出是否需要被明确分开。

## 结语：Attention 之后，关键是怎样组织信息

回到文章开头的问题：同样建立在 Transformer 之上，BERT 为什么更适合构建文本表示，GPT 为什么能够持续生成，而 T5、BART 又为什么保留了 Encoder–Decoder？答案并不在某一个全新的 Attention 公式里，而在于它们对信息流做出了不同选择。

Self-Attention 与 Cross-Attention 决定信息从哪里读取，双向与因果注意力决定每个位置能够看到什么，Multi-Head Attention 则允许模型同时建立多组关系。原始 Transformer 将这些机制组合起来，完成从输入序列到输出序列的转换；BERT、GPT 和 T5/BART 又根据各自的训练目标，对这套组合进行了拆分与重组。

所以，Transformer 更适合被理解为一套组织信息流的方法，而不是一种固定架构。Encoder-Only、Decoder-Only 和 Encoder–Decoder 也不是谁彻底取代谁：完整上下文有利于学习表示，因果信息流天然适合连续生成，分开的编码与解码结构则为条件生成保留了清晰的输入输出边界。不同架构的能力倾向，正是这些选择共同作用的结果。

不过，到这里我们看到的仍然只是一张架构地图。BERT 的 Attention 在真实训练中怎样配合 `[MASK]` 工作？GPT 又怎样利用因果遮罩，把一段文本变成可以不断扩展的上下文？随着模型从 GPT-1、GPT-2 扩展到 GPT-3，并进一步走向 GPT-3.5，Attention 的使用方式、上下文组织和训练范式又经历了哪些变化？

下一篇将进入这些模型内部，具体比较 Attention 在 BERT 与 GPT 中的应用，并沿着 GPT-1 到 GPT-3.5 的发展线索，梳理其中能够从论文和公开资料确认的变化。至于未被公开的架构细节，我们也会把事实、合理推断与未知部分明确区分开来。
