---
title: "注意力演进史（三）：LLM 时代的 Attention——如何变得更快、更长、更省？"
category: "人工智能"
date: "2026.09.01"
featured: false
---

上一篇从原始 Transformer 出发，讨论了 Attention 如何被组织成三种不同的信息流。BERT 保留 Encoder，让每个位置读取左右两侧的完整上下文；GPT 保留因果 Self-Attention，让每个位置只能读取已经出现的内容；T5 和 BART 则继续使用 Encoder–Decoder，通过 Cross-Attention 连接输入与输出。

沿着这些路线继续向后看，会发现一个值得注意的现象：从 BERT 到 RoBERTa，从 GPT-1 到 GPT-2，再到规模更大的 GPT-3，模型能力虽然不断提高，Attention 本身却没有被一代代重新发明。更多时候，研究者是在改变预训练任务、增加数据与参数、延长训练，或者重新组织任务和文本的关系。BERT 的 Masked Language Modeling 和 GPT 的自回归预测，首先证明的不是哪一种新公式更强，而是同一套 Transformer 可以通过不同的训练方式释放出怎样的潜力。

当上下文从几百个 token 增长到几万、几十万 token 时，标准 Multi-Head Attention 的两个结构性成本开始变得明显：token 之间的完整连接随序列长度近似平方增长，而每个 Head 独立保留的 K/V 表示也会随历史不断增加。

于是，LLM 时代的 Attention 改进开始沿着几条不同的方向发生。Multi-Query Attention 和 Grouped-Query Attention 让多个 Query Heads 共享更少的 Key/Value Heads；Sliding Window Attention 放弃所有 token 之间的完全连接，只保留一定范围内的历史；RoPE 等位置方法则改变位置信息参与 Attention Score 的方式。到了 Multi-head Latent Attention，Key 和 Value 又不再只是被共享，而是被压缩进更紧凑的潜在表示。

这些方案不在同一个维度工作。有些改变模型中“谁可以读取谁”，有些改变多个 Heads 怎样共享 K/V，有些改变距离如何影响 Query 与 Key 的匹配，还有一些改变历史信息以什么表示被保留。它们没有抛弃 Attention 的基本思想，却开始重新设计这套机制内部原本默认不变的部分。

因此，本文不再把 BERT 与 GPT 的型号演进作为主线，而是从 LLM 规模暴露出的结构性瓶颈出发：先讨论 MQA/GQA 如何重新组织多头 K/V，再看滑动窗口怎样减少连接、位置方法怎样扩展距离表示，以及 MLA 怎样把多头历史压缩进潜在空间。在文章末尾，我们还会越过这些对标准 Attention 的改造，看看 Linear Attention 如何尝试从计算形式上绕开完整的两两关系。Attention 最初解决的是模型应该从哪里读取信息；进入 LLM 时代之后，它开始进一步改变信息怎样连接、怎样表示，又怎样被汇总。

## 从标准 MHA 到 MQA、GQA：多个 Query 是否需要同样多组 K/V？

先回到标准的 Multi-Head Attention。对于长度为 $n$ 的序列，每个 Head 都会用 Query 与 Key 计算一组 $n\times n$ 的相关性，再根据这些权重汇总 Value：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M
\right)V.
$$

因此，完整 Attention 需要处理近似 $n^2$ 组 Query–Key 关系。这个平方成本将在后面的 Sliding Window 和 Linear Attention 中继续出现；本节先关注另一个维度：Multi-Head Attention 为什么要为每个 Head 都准备一套独立的 Key 和 Value？

### MHA：每个 Head 都保存一套观察方式

在标准 Multi-Head Attention（MHA）中，第 $i$ 个 Head 拥有自己的 Query、Key 和 Value 投影：

$$
Q_i=XW_i^Q,\qquad
K_i=XW_i^K,\qquad
V_i=XW_i^V,
$$

$$
\operatorname{head}_i
=\operatorname{Attention}(Q_i,K_i,V_i).
$$

不同 Head 因而可以用不同方式提出问题、描述历史 token，并决定从中读回什么。这里，Key 可以理解为一个 token“在什么条件下容易被找到”，Value 则表示“它被关注后会传递什么内容”。同一个 token 经过不同 Head 的投影，可以产生不同的 Key 和 Value：一个 Head 可能突出它的实体信息，另一个 Head 可能突出它的句法角色。于是，各个 Head 不仅拥有不同的 Query，也拥有不同的历史索引方式和信息内容。

这种差异会同时作用于 Attention 的两个阶段。不同的 $K_i$ 会改变 Query 与历史 token 的匹配分数，从而改变权重分配；不同的 $V_i$ 又会让同一个 token 在被关注后，向不同 Head 传递不同信息。因此，MHA 的丰富性并不只来自“用多个 Query 提出不同问题”，也来自“让同一段历史以多种方式提供信息”。

这种设计给予每个 Head 最大的独立性，但它也隐含了一个假设：有多少种 Query 的提问方式，就必须有多少套 K/V 来描述和保存历史。如果模型拥有 32 个 Query Heads，那么每一层、每个历史 token 通常也会产生 32 组 Key 和 32 组 Value。

在自回归生成中，过去 token 的 K/V 会被保留下来，供之后生成的新 Query 继续读取。暂时忽略层数和 Head Dimension 等共同因素，需要保留的历史表示数量与 KV Heads 数量成正比：

$$
\text{K/V states}\propto 2\times n\times h_{kv},
$$

其中 $n$ 是历史长度，$h_{kv}$ 是 KV Heads 数量，2 代表 Key 和 Value。随着序列变长，研究者自然会追问：多个 Query Heads 真的需要完全独立的 K/V 吗？

### MQA：保留多种问题，共享同一份历史

Multi-Query Attention（MQA）给出的答案是：不一定。

MQA 仍然保留多个 Query Heads，因此不同 Head 依然能够从不同角度提出问题；但它只生成一组共享的 Key 和 Value：

$$
Q_i=XW_i^Q,\qquad
K=XW^K,\qquad
V=XW^V,
$$

$$
\operatorname{head}_i
=\operatorname{Attention}(Q_i,K,V).
$$

这并不是把 Multi-Head Attention 退化成 Single-Head Attention。各个 Head 的 Query 仍然不同，因此即使面对同一组 Key，也可以得到不同的 Attention Weights：

$$
\alpha_i
=\operatorname{softmax}\!\left(
\frac{Q_iK^{\mathsf T}}{\sqrt{d_k}}+M
\right),
\qquad
\operatorname{head}_i=\alpha_iV.
$$

不同 Head 仍然可能关注不同 token，并得到不同的加权结果；最终的多个 Head 输出也仍会被拼接并投影。被共享的，是历史 token 提供 Key 和 Value 的方式。

可以把它理解成多个读者查阅同一套索引和档案。每个读者带着不同问题，因此会从档案中选择不同内容；但系统不再为每个读者分别保存一套完整档案。如果原来有 $h$ 组 KV Heads，在其他条件相同的情况下，MQA 需要保留的 K/V 主体可以下降到 MHA 的约 $1/h$。

因此，MQA 的假设并不是“不同 Query 最终需要的信息相同”。更准确地说，它假设 MHA 中不同 Heads 对历史所做的 K/V 投影存在一定冗余：多个 Query 可以继续提出不同问题，却未必需要多套完全独立的历史编码。

这种共享也带来了明确的取舍。在 MHA 中，每个 Head 可以分别决定历史 token 应该以什么 Key 被找到，以及被找到后通过什么 Value 提供内容；MQA 只保留了 Query 端的多样性，要求所有 Heads 面对同一套历史索引和内容。需要保存的信息减少了，但各个 Head 独立表示历史的自由度也随之降低。

### GQA：在完全独立与完全共享之间分组

Grouped-Query Attention（GQA）在 MHA 与 MQA 之间增加了一个中间选项：不让每个 Query Head 都拥有独立 K/V，也不让所有 Query Heads 共用唯一一组 K/V，而是把 Query Heads 分成若干组。

假设模型有 $h_q$ 个 Query Heads 和 $h_{kv}$ 个 KV Heads，并且 $h_q>h_{kv}$。第 $i$ 个 Query Head 使用所属组 $g(i)$ 的 Key 和 Value：

$$
\operatorname{head}_i
=\operatorname{Attention}
\left(Q_i,K_{g(i)},V_{g(i)}\right).
$$

例如，8 个 Query Heads 可以共享 2 组 K/V，每 4 个 Query Heads 使用同一组历史表示。这样，组内的 Head 共享 K/V，组与组之间仍然能够用不同方式表示历史。

MHA、GQA 和 MQA 因而不是三种彼此孤立的结构，而是一条连续的设计轴：

| 形式 | Query Heads | KV Heads | 共享方式 |
|---|---:|---:|---|
| MHA | 32 | 32 | 每个 Query Head 使用独立 K/V |
| GQA | 32 | 8 | 每 4 个 Query Heads 共享一组 K/V |
| MQA | 32 | 1 | 所有 Query Heads 共享一组 K/V |

表中的数字只用于展示三种结构的关系，不对应某个特定模型。

当 $h_{kv}=h_q$ 时，GQA 就是标准 MHA；当 $h_{kv}=1$ 时，它就变成 MQA。KV Heads 越少，需要保留的历史表示越少；KV Heads 越多，各组表示历史的方式越丰富。GQA 的意义，正是让模型能够在这两者之间选择，而不必只在“全部独立”和“全部共享”之间二选一。Mistral 7B 等公开模型采用 GQA，也说明这种折中已经进入实际的 LLM 架构。

用一句话概括三者的差别：MHA 是“不同问题、不同索引、不同内容”；MQA 是“不同问题、共享索引、共享内容”；GQA 则是“不同问题、分组共享索引与内容”。

不过，MHA、MQA 和 GQA 改变的都只是 **Head 维度**：同一个历史 token 需要保存多少组 K/V。无论有 32 组、8 组还是 1 组 K/V，每一组仍然包含序列中所有可见位置。换句话说，它们减少了同一段历史被表示多少次，却没有减少一个 Query 需要面对多少个历史 token。

要改变后一个问题，就必须转向 **序列维度**：每个 token 是否真的需要在每一层读取全部历史？这正是 Sliding Window Attention 的出发点。

## Sliding Window Attention：不是每个 token 都需要读取全部历史

在标准的因果 Attention 中，第 $t$ 个 token 可以读取自己以及此前的所有位置。随着序列向后延伸，每一行的可见范围不断扩大，最终形成一个完整的下三角 Attention Matrix。

Sliding Window Attention（SWA）保留了因果方向，但进一步限制了每个位置能够回看的距离。假设窗口宽度为 $w$，第 $t$ 个 token 只读取自己和最近的 $w-1$ 个位置：

$$
M_{t,j}^{\mathrm{SWA}}=
\begin{cases}
0, & \max(1,t-w+1)\leq j\leq t,\\
-\infty, & \text{otherwise}.
\end{cases}
$$

例如，当窗口宽度为 3 时，不同位置的可见范围如下：

```text
token 1 → 1
token 2 → 1, 2
token 3 → 1, 2, 3
token 4 →    2, 3, 4
token 5 →       3, 4, 5
token 6 →          4, 5, 6
```

窗口并不是把序列切成互不相连的固定分块，而是随着当前位置一起向后滑动。`token 4` 与 `token 3` 的窗口存在重叠，`token 5` 与 `token 4` 的窗口也存在重叠，因此相邻区域仍然可以连续传递信息。

### 从完整连接到局部连接

完整的因果 Attention 需要建立的连接数量大约是：

$$
1+2+\cdots+n
=\frac{n(n+1)}{2}
=O(n^2).
$$

在 Sliding Window Attention 中，每个位置最多只保留 $w$ 条连接，因此总连接数不超过：

$$
n\times w=O(nw).
$$

当窗口宽度 $w$ 不随序列长度增长时，连接数量便随 $n$ 线性增长。这里的改进与 MQA、GQA 完全不同：MQA/GQA 保留所有历史位置，只减少同一个位置需要多少组 K/V；Sliding Window 保留原来的多头计算方式，却直接删除窗口之外的 Query–Key 连接。

这是一种明确的结构假设：对于当前 token 来说，邻近上下文通常包含最直接的信息。短语内部的搭配、相邻词之间的句法关系，以及局部语义的延续，大多不需要在每一层扫描整篇文档。模型可以把计算集中在附近区域，而不是反复比较大量距离很远的位置。

### 只能看一个窗口，模型怎样获得更远的信息？

单层中的直接可见范围虽然有限，多层堆叠仍然可以让信息逐步向外传播。继续使用窗口宽度为 3 的例子：

- 第一层中，`token 6` 可以直接读取 `token 4、5、6`；
- 第二层中，`token 6` 读取到的 `token 4` 表示已经融合了第一层的 `token 2、3、4`；
- 因此，第二层的 `token 6` 可以间接获得最远来自 `token 2` 的信息；
- 再增加一层，信息还可以继续向更早的位置传播。

如果每层窗口都包含当前位置和此前 $w-1$ 个位置，那么堆叠 $L$ 层后的理论感受野大约可以扩展到：

$$
1+L(w-1).
$$

这与卷积网络很相似：单层卷积核只处理局部区域，但多层结构可以逐渐覆盖更大的范围。Sliding Window Attention 并没有让远距离信息完全消失，而是把“任意两个位置一步直达”改成了“信息通过相邻窗口逐层传递”。

不过，理论上能够传到，并不等于模型能够无损使用。远处信息需要经过多个中间 token 和多层加权汇总；传播路径越长，某个具体细节越可能被混合或削弱。对于需要从很远位置精确复制一个名字、数字或代码变量的任务，完整 Attention 可以建立一条直接连接，Sliding Window 则必须依赖多层间接传递。

因此，窗口大小构成了一个新的取舍：更小的窗口保留更少连接，但远距离依赖需要经过更多层；更大的窗口更接近完整 Attention，也保留了更多直接读取能力。所谓“理论感受野覆盖了整段文本”，也不应被等同为模型能够同样可靠地使用其中每一个位置。

### GQA 与 Sliding Window 改变的是两个维度

Mistral 7B 同时采用了 Grouped-Query Attention 和 Sliding Window Attention。这两项技术能够组合，是因为它们回答的问题并不相同：

| 技术 | 改变的维度 | 核心问题 |
|---|---|---|
| GQA | Head 维度 | 同一个历史位置需要多少组 K/V？ |
| Sliding Window | 序列维度 | 一个 Query 需要直接读取多少个历史位置？ |

假设模型有 32 个 Query Heads、8 个 KV Heads，并使用宽度为 $w$ 的滑动窗口，那么 GQA 决定这 32 个 Query Heads 怎样分组共享 8 套 K/V；Sliding Window 则决定每套 K/V 中，当前 Query 只与最近的 $w$ 个位置建立联系。一个减少历史的重复表示，另一个减少历史的直接连接，两者可以在同一层中同时发生。

Sliding Window Attention 由此把 Attention 的改进从 Head 组织推进到了连接拓扑。但无论窗口多大，Attention 还需要知道窗口中的 token 谁先谁后、彼此相隔多远。Self-Attention 的点积本身并不包含顺序，位置必须通过另一种方式进入 Query–Key 匹配。下一节将简要建立这部分背景，但把位置编码的具体推导留给单独的文章。

## 位置怎样进入 Attention

Attention 的核心匹配来自 Query 与 Key 的点积。如果只看 token 内容，这个点积并不知道两个 token 谁先谁后，也不知道它们相隔一个位置还是一千个位置。因果 Mask 可以阻止当前位置读取未来，却只规定了信息流方向，并没有完整表达位置之间的距离。

最直接的做法，是在输入表示中加入一个位置向量，再由混合了内容和位置的表示产生 Q、K、V。现代 LLM 还常常让位置更直接地参与 Attention Score。这里不展开具体推导，只保留三种理解后文需要的思路：

| 方法 | 位置放在哪里？ | 核心直觉 |
|---|---|---|
| RoPE | 按位置旋转 Q 和 K | 两个旋转后向量的点积能够反映相对位移 |
| ALiBi | 直接修改 Query–Key Score | 距离越远，加入的线性惩罚越大 |
| Position Interpolation | 缩放输入 RoPE 的位置编号 | 把更长的位置范围压回模型训练时熟悉的区间 |

RoPE 的关键不是简单地给 token 再加一个位置向量，而是让 Q 和 K 随位置发生旋转。两个位置之间的旋转差会进入点积，因此 Attention 在比较内容的同时，也能感受到它们的相对距离。至于不同维度为什么使用不同旋转频率、这种旋转怎样影响长度外推，则留给位置编码专题。

ALiBi 采用了更直接的方式：不把位置混入 token embedding，而是在 Attention Score 中加入一个随距离增加的负偏置。它相当于告诉不同 Heads：“在其他条件相近时，更远的位置需要更强的内容匹配，才能获得同样高的权重。”

Position Interpolation 则不是重新设计一种 Attention，而是扩展已有 RoPE 模型上下文的一种方法。如果模型训练时只见过一定范围的位置，直接使用远超该范围的位置编号可能不稳定；Position Interpolation 把更长序列的位置坐标按比例缩小，使其重新落入模型熟悉的区间，再通过适配训练学习新的分辨率。

这些方法解决的是“位置怎样参与 Attention”，并不会直接减少 Query–Key 连接的数量。能够表示更远的位置，也不等于模型一定能可靠利用远处内容：连接模式决定信息能否到达，训练数据决定模型是否学会使用长距离信息，位置编码只是其中一部分。

本文对位置方法只讨论到这里。不过，RoPE 将位置信息直接作用在 Key 上，会给下一种 Attention 改造带来一个具体问题：如果模型想把多头 K/V 压缩成一份更小的潜在表示，内容投影和位置旋转应该以什么顺序组合？这正是理解 Multi-head Latent Attention 时必须保留的位置背景。

## Multi-head Latent Attention：从共享 K/V 到压缩 K/V

MQA 和 GQA 减少 K/V 的方法，是让多个 Query Heads 共用一组历史表示。KV Heads 越少，共享程度越高；但正如前面所见，这也会限制不同 Heads 独立表示历史的能力。

Multi-head Latent Attention（MLA）选择了另一条路线：它不直接删除大部分 KV Heads，而是追问多头 K/V 背后是否存在一份更紧凑的共同表示。如果能够先把历史信息压缩到低维潜在空间，再让不同 Heads 从中提取各自需要的 Key 和 Value，就有可能同时保留多头差异与较小的历史表示。

### 低秩 K/V 联合压缩

在标准 MHA 中，第 $t$ 个 token 的隐藏状态 $h_t$ 会直接投影成展开后的多头 Key 和 Value：

$$
k_t=W^Kh_t,\qquad v_t=W^Vh_t.
$$

如果有很多 Heads，$k_t$ 和 $v_t$ 就包含许多组彼此独立的向量。MLA 则先用一个降维投影，把 $h_t$ 压缩为一份较小的潜在表示：

$$
c_t^{KV}=W^{DKV}h_t,
$$

再通过不同的升维投影，从 $c_t^{KV}$ 产生多头 Key 和 Value：

$$
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}.
$$

其中，$c_t^{KV}$ 的维度远小于展开后的全部多头 K/V。它不是人工规定的摘要，也不是训练结束后额外进行的压缩；整个模型从一开始就通过这条低秩瓶颈训练，因此会学习把 Attention 真正需要的信息保留在 $c_t^{KV}$ 中。

这个结构仍然允许不同 Heads 产生不同 Key 和 Value。共同的只是底层潜在表示，不同 Head 对应的部分仍可通过 $W^{UK}$ 和 $W^{UV}$ 采用不同方式读取它。可以把 MQA 看成“所有人查阅同一套索引和档案”，而 MLA 更像是“所有信息先存进一份紧凑的底稿，再由不同 Head 用不同解码方式形成各自的索引和内容”。

### 为什么不需要每次恢复完整 K/V？

乍看之下，如果每次 Attention 都要先把 $c_t^{KV}$ 展开回完整的 $k_t^C$ 和 $v_t^C$，压缩似乎只是把工作推迟了。但线性投影可以与 Attention 中的其他运算重新组合。

对于 Key，Query 与展开后 Key 的点积可以写成：

$$
q^{\mathsf T}k_t^C
=q^{\mathsf T}W^{UK}c_t^{KV}
=\left((W^{UK})^{\mathsf T}q\right)^{\mathsf T}c_t^{KV}.
$$

也就是说，可以先把 Query 投影到与 $c_t^{KV}$ 对应的空间，再直接与压缩表示计算匹配，而不必先为每个历史 token 展开完整 Key。

对于 Value，同样可以利用线性运算的结合方式：

$$
\sum_t\alpha_t v_t^C
=\sum_t\alpha_tW^{UV}c_t^{KV}
=W^{UV}\left(\sum_t\alpha_t c_t^{KV}\right).
$$

模型可以先对压缩表示进行加权汇总，再把汇总结果投影到多头 Value 空间。于是，$c_t^{KV}$ 不只是一个临时压缩文件，而可以直接成为 Attention 实际读取的历史状态。

MLA 也对 Query 使用了单独的低秩投影，不过 Query 只属于当前正在处理的位置，不会像历史 K/V 那样持续累积。理解 MLA 的主线时，重点仍然是 K/V 的联合潜在表示。

### RoPE 为什么必须被单独处理？

上面的投影合并成立，是因为 $W^{UK}$ 是一组与位置无关的固定参数。RoPE 却会根据 token 所在位置，对 Query 和 Key 施加不同旋转。如果直接对展开后的 $k_t^C$ 使用 RoPE，位置相关的旋转就会夹在 $W^{UK}$ 与点积之间，无法再简单地把 $W^{UK}$ 合并到 Query 一侧。

MLA 因此把每个 Head 的 Query 和 Key 分成两部分：

$$
q_{t,i}=[q_{t,i}^C;q_{t,i}^R],qquad
k_{t,i}=[k_{t,i}^C;k_t^R].
$$

- 内容部分 $q^C$、$k^C$ 继续使用低秩潜在表示，并保留前面的投影合并；
- 位置部分 $q^R$、$k^R$ 单独应用 RoPE；
- 位置 Key $k^R$ 在多个 Heads 之间共享，只占据相对较小的维度。

最终的 Attention Score 同时包含内容匹配和位置关系，但需要为历史保留的主要信息，仍然是压缩后的 $c_t^{KV}$，再加上一份较小的位置 Key。这种 Decoupled RoPE 不是额外附加的技巧，而是让低秩 K/V 压缩与旋转位置编码能够同时成立的关键。

### MLA 与 MQA、GQA 的根本区别

三种方法都减少了历史 K/V 的表示量，但压缩方式并不相同：

| 形式 | 核心做法 | 保留多头差异的方式 |
|---|---|---|
| MQA | 所有 Query Heads 共享一组 K/V | 主要依靠不同 Query 产生不同权重 |
| GQA | 每组 Query Heads 共享一组 K/V | 不同 KV Groups 保留部分独立表示 |
| MLA | 多头 K/V 联合压缩到潜在表示 | 不同投影从共同 latent 中提取多头内容 |

因此，GQA 改变的是“保留几组 KV Heads”，MLA 改变的是“每个 token 究竟需要保留什么”。前者通过分组减少重复，后者通过低秩瓶颈重新定义历史表示。

这种低秩结构同样存在取舍。潜在维度太小，重要信息可能无法穿过瓶颈；潜在维度太大，压缩的意义又会减弱。内容与位置的拆分也让 MLA 比 MQA/GQA 更复杂。DeepSeek-V2 的实验表明这种结构可以在其模型配置中大幅缩小 K/V 表示并保持较强性能，但具体压缩比例来自它所选择的 Head Dimension、潜在维度和位置维度，并不是 MLA 在所有模型上的固定常数。

还需要注意，MLA 没有减少序列中的位置数量。如果仍然使用完整 Attention，当前 Query 依然要与所有历史位置对应的潜在表示进行匹配。它压缩了“每个位置保存什么”，却没有消除“需要读取多少个位置”。Linear Attention 将沿着这个问题继续前进：历史是否还必须以一串逐 token 的表示存在，还是可以被不断汇总进一个递归状态？

## Linear Attention：当历史被写进递归状态

前面讨论的方法仍然保留了标准 Attention 的一个基本形态。MQA、GQA 和 MLA 虽然缩小了每个位置的 K/V，但历史中每个 token 仍有一份可以单独读取的表示；Sliding Window 虽然只读取附近位置，也仍然会在窗口内逐一计算 Query 与 Key 的关系。

Linear Attention 提出了更激进的问题：能否不再显式生成 token 两两之间的 Attention Matrix，而是先把历史 Key 和 Value 汇总成一个大小固定的状态，再让 Query 直接读取它？

### 为什么标准 Softmax Attention 很难直接重排

以因果 Self-Attention 为例，第 $t$ 个位置的输出可以写成：

$$
y_t=
\frac{
\sum_{j\leq t}
\exp\!\left(q_t^{\mathsf T}k_j/\sqrt{d_k}\right)v_j
}{
\sum_{j\leq t}
\exp\!\left(q_t^{\mathsf T}k_j/\sqrt{d_k}\right)
}.
$$

这里，每个权重都由当前 Query $q_t$ 和某个历史 Key $k_j$ 共同决定。换一个 Query，所有历史位置的权重都要重新计算。因此，标准实现必须先求出 Query 与 Key 的两两匹配，再用这些结果汇总 Value；不能只把所有 Key 或 Value 预先相加。

Linear Attention 的关键，是把相似度换成可以分解的 Kernel：

$$
\operatorname{sim}(q,k)
=\phi(q)^{\mathsf T}\phi(k).
$$

$\phi$ 把 Query 和 Key 映射到一个新的特征空间，并通常被选择为非负映射，使相似度和归一化仍具有 Attention Weight 的含义。将这个相似度代入 Attention，可以得到：

$$
y_t=
\frac{
\sum_{j\leq t}
\phi(q_t)^{\mathsf T}\phi(k_j)v_j^{\mathsf T}
}{
\sum_{j\leq t}
\phi(q_t)^{\mathsf T}\phi(k_j)
}.
$$

由于 $\phi(q_t)$ 与求和下标 $j$ 无关，可以把它移到求和之外：

$$
y_t=
\frac{
\phi(q_t)^{\mathsf T}
\left(\sum_{j\leq t}\phi(k_j)v_j^{\mathsf T}\right)
}{
\phi(q_t)^{\mathsf T}
\left(\sum_{j\leq t}\phi(k_j)\right)
}.
$$

这个看似简单的括号变化改变了整个计算过程。模型不再需要先形成一个 $n\times n$ 的 Attention Matrix，而可以先把所有历史 Key–Value 关系汇总，再由 Query 读取汇总结果。如果特征维度相对序列长度固定，计算量便可以随序列长度 $n$ 线性增长。这里的“Linear”指复杂度关于序列长度近似线性，而不是模型只能表达线性函数。

标准 Softmax 中的指数相似度并没有一个足够小、可以直接使用的精确有限维特征映射。因此，Linear Attention 通常要么改用另一种 Kernel，要么用有限维特征去近似 Softmax；它不是把完全相同的 Softmax Attention 仅靠交换乘法顺序就免费变成线性复杂度。

### 从 Attention Matrix 到递归状态

在因果场景中，括号里的两个历史汇总量可以随着 token 到来逐步更新：

$$
S_t=S_{t-1}+\phi(k_t)v_t^{\mathsf T},
$$

$$
z_t=z_{t-1}+\phi(k_t).
$$

其中，$S_t$ 保存 Key 与 Value 的联合汇总，$z_t$ 用于归一化。当前 Query 的输出则是：

$$
y_t=
\frac{\phi(q_t)^{\mathsf T}S_t}
{\phi(q_t)^{\mathsf T}z_t}.
$$

假设一段文本先写道“小林把钥匙放进蓝色抽屉”，此时 Key 可以突出“钥匙”这一检索线索，Value 则携带“蓝色抽屉”这一内容。外积 $\phi(k_t)v_t^{\mathsf T}$ 相当于把这条对应关系写入矩阵状态 $S_t$。之后出现“钥匙放在哪里？”时，新的 Query 会激活与“钥匙”相近的方向，从同一个状态中读出与之关联的内容。

这仍然保留了 Attention 的直觉：Query 决定当前需要读取什么。但可供读取的对象不再是一排彼此独立的历史 token，而是许多 Key–Value 关系叠加形成的共同状态。

### 它为什么开始像 RNN？

经典 RNN 在每个时间步接收新输入，并把旧状态更新成新状态：

$$
h_t=F(h_{t-1},x_t),\qquad
y_t=G(h_t).
$$

Linear Attention 现在也可以写成同样的结构：$S_{t-1}$ 和 $z_{t-1}$ 是上一时刻的状态，当前 $k_t$ 与 $v_t$ 负责写入，$q_t$ 负责读取。生成一个新 token 时，对每一层、每个 Head 而言，模型只需更新维度固定的 $S_t$ 和 $z_t$，不必重新扫描或继续扩充全部历史 K/V。就自回归推理而言，一个 Linear Attention Layer 因而可以被看作拥有矩阵隐藏状态的 RNN。

二者的相似之处主要有三点：

- 历史都被浓缩进一个随时间更新的状态；
- 每一步的更新成本不再随已经生成的序列长度增长；
- 下一步输出都由当前输入与上一时刻状态共同决定。

但“Linear Transformer 是 RNN”并不意味着它退回了传统 RNN 的全部设计。经典 RNN 通常用一个隐藏向量整体编码过去，再通过非线性递归变换它；Linear Attention 的状态通常是一个矩阵，更像可由 Query 进行内容寻址的联想记忆。Transformer 的多头结构、前馈层、残差连接等也仍然存在，只是其中的 Attention Layer 获得了递归写法。

还有一个重要区别在训练方式。普通 RNN 的非线性状态必须严格按照 $h_1,h_2,\ldots,h_n$ 的顺序计算；上面的 $S_t$ 和 $z_t$ 却只是外积与向量的前缀累加。由于加法满足结合律，训练时可以并行计算这些前缀状态，生成时再切换为逐步递归更新。这正是 Linear Attention 最吸引人的性质之一：训练时保留类似 Transformer 的并行形式，推理时获得类似 RNN 的固定状态。

反过来看，标准因果 Transformer 也可以勉强写成一种递归系统：每一步把新的 K/V 追加到 Cache，再由 Query 读取它。区别在于它的“状态”会随着序列不断增长；Linear Attention 则把这份无界的 K/V 列表压缩成了大小固定的矩阵与向量。

### 固定状态并不是免费的无限记忆

固定大小状态带来了线性复杂度，也意味着不同 token 的信息会叠加在一起。在标准 Softmax Attention 中，模型可以对某个历史位置分配几乎全部权重，并直接取回它的 Value；在线性状态中，多条相似的 Key–Value 关系可能互相干扰。序列越长，需要被同一状态承载的关系越多，精确复制、远距离检索和区分相近线索就可能越困难。

因此，Linear Attention 的真正难点从“如何避免平方计算”转移成了“如何管理有限状态”：应该选择怎样的特征映射，哪些信息需要写入，旧信息应当保留多久，状态发生冲突时又该怎样遗忘。后来的许多方法开始引入衰减、门控和选择性更新；有趣的是，这些设计又与 LSTM、GRU 曾经用门控制记忆的思想重新汇合。

所以，Linear Attention 并不是 MQA、Sliding Window 或 MLA 必然演化出的终点。前面的方案仍保存可单独访问的 token 表示，只是减少它们的数量、连接或维度；Linear Attention 则用固定状态换掉显式的历史列表。它获得了关于序列长度的线性计算与常量大小的递归状态，也接受了信息叠加和近似带来的表达取舍。
