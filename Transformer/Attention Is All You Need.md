>待看：
>
>[计算机视觉中的注意力机制](https://zhuanlan.zhihu.com/p/56501461)
>
>Transformer 的地位就像编程语言中 Python 的地位，Transformer 有望统一神经网络模型界，Python 有望统一编程语言界。

# 前导知识

> - 以下内容不是Transformer中新提出的，之前就有的，Transformer使用到了它们或者对它们进行了改进。
>
> - 参考资料：https://www.zhihu.com/question/68482809/answer/264632289

## RNN

因为《Attention Is All You Need》是基于机器翻译来进行讲述的，而机器翻译一般使用RNN，所以需要先对RNN有些了解。

请看 [ NN.md -> 神经网络的搭建 -> RNN ]. 

## Encoder-Decoder 

Encoder-Decoder 框架可以看作是一种深度学习领域的研究模式，应用场景异常广泛。下图是文本处理领域里常用的 Encoder-Decoder 框架最抽象的一种表示。

![image-20211104155341455](images/image-20211104155341455.png)

可以将文本处理领域的 Encoder-Decoder 框架直观地理解为：**由一个句子（或篇章）生成另外一个句子（或篇章）的通用处理模型（即序列到序列）**。

对于句子对 <Source,Target>，我们的目标是给定输入句子 Source，期待通过 Encoder-Decoder 框架来生成目标句子 Target。Source 和 Target 分别由各自的单词序列构成：

<img src="images/image-20211104181819797.png" alt="image-20211104181819797" style="zoom:50%;" />

Encoder 顾名思义就是对输入句子 Source 进行编码，<font color="red">将输入句子通过非线性变换转化为中间语义表示 C</font>：

<img src="images/image-20211104181900960.png" alt="image-20211104181900960" style="zoom:50%;" />

对于解码器 Decoder 来说，其任务是<font color="red">根据句子 Source 的**中间语义表示 C** 和**之前已经生成的历史信息** $y_1, y_2, ..., y_{(i-1)}$ 来生成 $i$ 时刻要生成的单词 $y_i$ </font>。即：

<img src="images/image-20211104182047422.png" alt="image-20211104182047422" style="zoom:50%;" />

> 这是一种**自回归（auto-regressive）机制**——过去时刻的输出会作为当前时刻的输入。

最终整个系统根据输入句子 Source 生成了目标句子 Target。

如果 Source 是中文句子，Target 是英文句子，那么这就是解决机器翻译问题的 Encoder-Decoder 框架；如果 Source 是一篇文章，Target 是概括性的几句描述语句，那么这是文本摘要的 Encoder-Decoder 框架；如果 Source 是一句问句，Target 是一句回答，那么这是问答系统或者对话机器人的 Encoder-Decoder 框架。由此可见，在文本处理领域，Encoder-Decoder 的应用领域相当广泛。

Encoder-Decoder 框架不仅仅在文本领域广泛使用，在语音识别、图像处理等领域也经常使用。比如对于**语音识别**来说，Encoder 部分的输入是语音流，输出是对应的文本信息；对于**图像描述**任务来说，Encoder 部分的输入是一副图片、输出是该图片的中间表达形式（即抽象出来的特征），Decoder 的输出是能够描述图片语义内容的一句描述语。一般而言，文本处理和语音识别的 Encoder 部分通常采用 RNN 模型，图像处理的 Encoder 一般采用 CNN 模型。

> Encoder-Decoder 在图像上应该有很多方面的应用啊，比如下面我搜索到的这两个应用：
>
> - [Encoder+Decoder+LSTM 预测图像帧](https://blog.csdn.net/PMPWDF/article/details/101224827)  
>
> - [Encoder+Decoder 图像分割](https://blog.csdn.net/qq_37614597/article/details/105593497)  

> <font color="grey">(《花书》)</font>思路：
>
> 1. **编码器（encoder）/读取器（Reader）/输入（input）RNN** 处理输入序列。编码器输出上下文 $C$（通常是最终隐藏状态 $h_{n_x}$ 的简单函数）。【<font color="red">最终隐藏状态 $h_{n_x}$ </font>通常被当作输入的表示<font color="red"> $C$ </font>并作为**解码器 RNN** 的输入。】
> 2. **解码器（decoder）/写入器（writer）/输出（output）RNN ** 则以固定长度的向量为条件产生输出序列 $Y=(y^{(1)},...y^{(n_y)})$​。

对于一个机器翻译任务来说，训练过程如下：

![image-20211106172722832](images/image-20211106172722832.png)

训练的时候是知道目标句子的，也就是知道真正的翻译是什么，就算预测出的某个词的翻译错了，也不会影响下一个预测的翻译的词的输入。

当真实的处理现实中的翻译任务时，不知道真正的翻译应该是什么的，如果出现某个词翻译错误，就很有可能影响接下来每个词，因为 RNN 中上一个词的这个错误的翻译会被当成预测下一个词的输入。

![image-20211106172859019](images/image-20211106172859019-16361909401941.png)

> 预测第 t+1 个输出时，解码器中输入前 t 个预测结果（在自注意力中，前 t 个预测值作为 key 和 value，第 t 个预测值还作为 query）。![image-20211109204715251](images/image-20211109204715251-16364620367136.png)

## 注意力（Attention）机制

> 目前大多数注意力模型附着在 **Encoder-Decoder** 框架下，当然，其实注意力模型可以看作一种通用的思想，本身并不依赖于特定框架，这点需要注意。 

与人类学习相同，机器学习过程中我们也希望能有侧重点，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。所以就有了 Attention 的出现。

> 直观理解——”婴儿在干嘛？“：
>
> <img src="images/image-20211106142812841-16361800941322.png" alt="image-20211106142812841" style="zoom:80%;" />
>
> 对婴儿对每部分的注意力进行计算：
>
> <img src="images/image-20211106142843137.png" alt="image-20211106142843137" style="zoom:80%;" />

本节先以机器翻译作为例子讲解最常见的 Soft Attention 模型的基本原理，之后抛离 Encoder-Decoder 框架抽象出了Attention 的本质思想，然后简单介绍最近广为使用的 Self Attention 的基本思路。

### Soft Attention 模型

上一节中的 Encoder-Decoder 模型是“注意力不集中”的，为什么呢？

请观察下目标句子 Target 中每个单词的生成过程：

<img src="images/image-20211104183306926.png" alt="image-20211104183306926" style="zoom:50%;" />

其中 $f$ 是 Decoder 的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子 Source 的语义编码 C 都是一样的，没有任何区别。

而<font color="blue">语义编码 C 是由句子 Source 的每个单词经过 Encoder 编码产生的，这意味着不论是生成哪个单词，$y_1$, $y_2$ 还是 $y_3$，其实句子 Source 中任意单词对生成某个目标单词 $y_i$​ 来说影响力都是相同的</font>，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点一样。

-- --

比如输入的是英文句子：`Tom chase Jerry`，Encoder-Decoder 框架逐步生成中文单词：“汤姆”，“追逐”，“杰瑞”。在翻译“杰瑞”这个中文单词的时候，分心模型里面的每个英文单词对于翻译目标单词“杰瑞”贡献是相同的，很明显这里不太合理，因为“Jerry”对于翻译成“杰瑞”最重要，但是分心模型是无法体现这一点的，这就是为何说它没有引入注意力的原因。

没有引入注意力的模型在输入句子比较短的时候问题不大，但是如果**输入句子比较长**，此时所有语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，可想而知会**丢失很多细节信息**，这也是为何要引入注意力模型的重要原因。

> 以下内容参考自 https://zhuanlan.zhihu.com/p/53682800 .
>
> -- --
>
> **Q1：** 为什么卷积或循环神经网络不能处理长序列？
>
> **A1：** 当使用神经网络来处理一个变长的向量序列时，我们通常可以使用卷积网络或循环网络进行编码来得到一个相同长度的输出向量序列，如图所示：
>
> ![image-20211105094200381](images/image-20211105094200381-163607652271218.png)
>
> 从上图可以看出，无论卷积还是循环神经网络其实都是对变长序列的一种“**局部编码**”：卷积神经网络显然是基于 N-gram 的局部编码；而对于循环神经网络，由于梯度消失等问题也只能建立短距离依赖。
>
> -- --
>
> **Q2：** 要解决这种短距离依赖的“局部编码”问题，从而对输入序列建立长距离依赖关系，有哪些办法呢？
>
> **A2：** 使用全连接模型或自注意力模型。由下图可以看出，全连接网络虽然是一种非常直接的建模远距离依赖的模型， 但是无法处理变长的输入序列；不同的输入长度，其连接权重的大小也是不同的。自注意力模型（self-attention model）可以**利用注意力机制来“动态”地生成不同连接的权重，从而可以处理变长的信息序列**；且该模型不会因句子过长而逐渐丢失很多细节。
>
> ![img](images/v2-cd2d7f0961c669d983b73db4e93ccbdc_1440w-163607677682620.jpg)

上面的例子中，如果引入 Attention 模型的话，应该在翻译“杰瑞”的时候，体现出英文单词对于翻译当前中文单词不同的影响程度，比如给出类似下面一个概率分布值：（Tom,0.3）(Chase,0.2)  (Jerry,0.5). 每个英文单词的概率代表了<font color="blue">翻译当前单词“杰瑞”时，注意力分配模型分配给不同英文单词的注意力大小</font>。这对于正确翻译目标语单词肯定是有帮助的。

同理，目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。<font color="blue">这意味着在生成每个单词 $y_i$ 的时候，原先都是**相同的根据整个句子得到的中间语义表示 $C$ **会被替换成**根据当前生成单词而不断变化的 $C_i$** 。理解 Attention 模型的关键就是这里，即由固定的中间语义表示 $C$ 换成了根据当前输出单词来调整成加入注意力模型的变化的 $C_i$ 。</font>增加了注意力模型的 Encoder-Decoder 框架表示如下：

<img src="images/image-20211104155628503.png" alt="image-20211104155628503" style="zoom:67%;" />

仍然以翻译英文句子：`Tom chase Jerry` 为例，则此时目标句子 Target 中每个单词的生成过程变化为下面这样：

<img src="images/image-20211104184309133.png" alt="image-20211104184309133" style="zoom:67%;" />

其中每个 $C_i$ 对应着不同的源语句子单词的注意力分配概率分布。比如对于上面的英汉翻译来说，其对应的信息可能如下：

<img src="images/image-20211104184632258.png" alt="image-20211104184632258" style="zoom:67%;" />

其中，$f_2$ 函数代表 Encoder 对输入英文**单词**的某种变换函数，从而得到该单词对应的语义编码；g 代表 Encoder 根据单词的中间表示合成**整个句子**中间语义表示的变换函数，通常，g 函数就是**对构成元素加权求和**，即下列公式：

<img src="images/image-20211104185127994.png" alt="image-20211104185127994" style="zoom:67%;" />

其中，$L_x$ 代表输入句子 Source 的长度，$a_{ij}$ 代表在 Target 输出第 $i$ 个单词时 Source 输入句子中第 $j$ 个单词的注意力分配系数，而 $h_j$ 则是 Source 输入句子中第 $j$ 个单词的语义编码。假设下标 $i$ 就是上面例子所说的“ 汤姆” ，那么 $L_x = 3$ ，每个单词的语义编码分别为 $ h_1=f(Tom)$，$h2=f(Chase)$，$h3=f(Jerry)$ ，对应的注意力模型权值则分别是 0.6, 0.2, 0.2，所以 g 函数本质上就是个加权求和函数。这一具体过程可表示成下图：

<img src="images/image-20211104190611886.png" alt="image-20211104190611886" style="zoom: 67%;" />

-- --

那么如何确定 Attention 模型的输入句子中每个单词的注意力分配概率分布值呢？

为了便于说明，我们对<font color="brown">非 Attention 模型</font>的 Encoder-Decoder 框架进行细化，且令 Encoder 采用 RNN 模型，Decoder 也采用 RNN 模型（这是比较常见的一种模型配置）：

<img src="images/image-20211104190903782.png" alt="image-20211104190903782" style="zoom: 67%;" />

该<font color="brown">传统的 RNN Encoder-Decoder 模型</font>：

>  参考：[Attention in RNN](https://zhuanlan.zhihu.com/p/42724582)

- 在编码过程中， $t$ 时刻的状态 $h_t$ 由 $t-1$ 时刻的状态 $h_{t-1}$ 和  $t$ 时刻的输入数据 $x_t$ 得到，经过 $T$ 个时间片后得到长度等于隐节点数量的特征向量 $C$ .
- 在解码过程中，将特征向量 $C$ 和上个时间片预测的输出 $y_{i-1}$ 输入到 RNN 单元中得到隐藏状态 $H_{i-1}$，进一步得到输出 $y_i$，经过 $T^{’}$ 个时间片后得到输出结果 .

<font color="brown">将 Attention 应用在此 RNN 中</font>，那么注意力分配概率分布值的通用计算过程为：

<img src="images/v2-ac1c016e17a1a681a1dd7da0fb18d1e8_r.jpg" alt="preview" style="zoom:80%;" />

函数 $F(h_j, H_{i-1})$ 被称作**相似度函数**：<font color="blue">将输出句子 Target 的 $i-1$ 时刻的隐层节点状态 $H_{i-1}$ 一一和输入句子 Source 中每个单词对应的语义编码 $h_j$ 进行对比，以获得目标单词 $y_i$ 和每个输入单词对应的对齐可能性/相似度</font>。函数 $F$ 的输出经过 Softmax 进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。

绝大多数 Attention 模型都是采取上述的计算框架来计算注意力分配概率分布信息，区别只是在 $F$ 的定义上可能有所不同。

-- --

传统的统计机器翻译一般在做的过程中会专门有一个短语对齐的步骤，而注意力模型其实起的是相同的作用。下图是 Google 于 2016 年部署到线上的基于神经网络的机器翻译系统，相对传统模型翻译效果有大幅提升，翻译错误率降低了60%，其架构就是上文所述的加上 Attention 机制的 Encoder-Decoder 框架，主要区别无非是其 Encoder 和 Decoder 使用了 8 层叠加的 LSTM 模型。

![preview](images/v2-b2e651923277337fb4413c0e93ac8d55_r.jpg)

### Attention 机制的本质思想

如果把 Attention 机制从上文讲述例子中的 Encoder-Decoder 框架中剥离，并进一步做抽象，可以更容易看懂 Attention 机制的本质思想：

![img](images/v2-24927f5c33083c1322bc16fa9feb38fd_1440w-16360268257245.jpg)

<font color="blue"> 将 Source 中的构成元素想象成是由一系列的 <Key,Value> 数据对构成，此时给定 Target 中的某个元素 Query，通过计算 Query 和各个 Key 的相似性或者相关性，得到每个 Key 对应 Value 的权重系数，然后对 Value 进行加权求和，即得到了最终的 Attention 数值。</font>所以<font color="red">本质上 Attention 机制是对 Source 中元素的 Value 值进行加权求和，而给定的 Target 中的一个 Query 和 Source 中的所有 Key 用来计算对应 Value 的权重系数。</font>即可以将其**本质思想**改写为如下公式：

<img src="images/image-20211104195642355.png" alt="image-20211104195642355" style="zoom: 67%;" />

其中，$L_x=||Source||$ 表示 Source 的长度。上面一小节所举的机器翻译的例子里，因为在计算 Attention 的过程中，Source 中的 Key 和 Value 合二为一，指向的是同一个东西，也即输入句子中每个单词对应的语义编码，所以未能很好的体现出 Attention 机制的本质思想。

-- --

当然，从概念上理解，把 Attention 仍然理解为从大量信息中有选择地筛选出少量重要信息并聚焦到这些重要信息上，忽略大多不重要的信息，这种思路仍然成立。聚焦的过程体现在权重系数的计算上，权重越大越聚焦于其对应的 Value 值上，即权重代表了信息的重要性，而 Value 是其对应的信息。

Query，Key，Value 的概念取自于信息检索系统，举个简单的搜索的例子来说。在某电商平台上，平台中所有商品都有自己的 <Key, Value>，而当你搜索某件商品（如“红色薄款羽绒服”）时，输入的搜索内容便是 Query，然后搜索引擎根据 Query 为你匹配 Key（例如从商品的种类，颜色，描述等方面进行匹配），然后根据 Query 和 Key 的相似度得到匹配度最高的那部分内容。

![image-20211106142402502](images/image-20211106142402502-16361798441311.png)

-- --

Attention 机制的**具体计算**可以归纳为两个过程：

1. <font color="red">根据 Query 和 Key 计算权重系数；</font> 
   - <font color="red">根据 Query 和 Key 计算两者的相似性/相关性 </font> 
   - <font color="red">对上面的结果进行归一化处理，得到 Key 对应的 Value 的权重</font> 
2. <font color="red">根据权重系数对 Value 进行加权求和。</font> 

-- --

可以将 Attention 的计算过程抽象为下图所示的三个阶段：

![preview](images/v2-07c4c02a9bdecb23d9664992f142eaa5_r.jpg)

在第一个阶段，计算得到 Query 与 $Key_i$ 的相似性/相关性 $s_1, ... s_4$。

> 【常用的计算相似度函数】
>
> ![img](images/v2-da4a315aa6decd424ec9f83f29572107_1440w-16360284014683.jpg)

由于第一阶段产生的分值根据具体产生的方法不同其数值取值范围也不一样，所以第二阶段引入类似 SoftMax 的计算方式对第一阶段的得分进行数值转换，一方面可以进行归一化，将原始计算分值整理成所有元素权重之和为 1 的概率分布；另一方面也可以通过 SoftMax 的内在机制更加突出重要元素的权重。即一般采用如下公式计算：

<img src="images/image-20211104202140215.png" alt="image-20211104202140215" style="zoom:50%;" />

第二阶段的计算结果 $a_i$ 即为 $value_i$ 对应的权重系数，进行加权求和即可得到<font color="blue"> Query 对应的 Attention数值</font>：

<img src="images/image-20211104202325381.png" alt="image-20211104202325381" style="zoom:67%;" />

> ==【思考】==所以 Attention 的输出就是一个数值（而不是一个向量）！但是后面将 Self Attention 应用到 Transformer 中时，为什么输出的是一个向量？因为 Q、K、V 是向量啊。实际上，Q 有多少维，输出就会有多少维~  那么 Q、K、V 的维度由什么决定？那就要看  Q、K、V 是怎么出现的了——  Q、K、V 是由词向量与一个权重矩阵做矩阵乘积得来的，而这个权重矩阵是训练过程中学习出来的。所以， Q、K、V 的维度就是由词向量的维度和权重矩阵的维度决定的。

### Self Attention 模型

通过上述对 Attention 本质思想的梳理，我们可以更容易理解本节介绍的 Self Attention 模型。Self Attention 也经常被称为 **intra-Attention**，最近一年也获得了比较广泛的使用，比如 Google 最新的机器翻译模型内部大量采用了 Self Attention 模型。

在一般任务的 Encoder-Decoder 框架中，输入 Source 和输出 Target 内容是不一样的，比如对于英-中机器翻译来说，Source 是英文句子，Target 是对应的翻译出的中文句子。而 Self-Attention 中  `Target = Source` ，更进一步地，可将其表示为 `Key = Value  = Query`。其具体计算过程是一样的，只是计算对象发生了变化而已，所以此处不再赘述其计算过程细节。

如果是常规的 Target 不等于 Source 情形下的注意力计算，其物理含义正如上文所讲，比如对于机器翻译来说，本质上是目标语单词和源语单词之间的一种单词对齐机制。那么如果是 Self Attention 机制，一个很自然的问题是：**通过 Self Attention 到底学到了哪些规律或者抽取出了哪些特征呢？**或者说引入 Self Attention 有什么增益或者好处呢？答案是<font color="red"> **Self Attention 可以捕获同一个句子内部单词之间的一些句法特征或者语义特征** </font>。

示例1. 下图捕获到了有一定距离的短语结构（句法特征）——“making ... more difficult”：

![image-20211104204014001](images/image-20211104204014001-16360296162486.png)

示例2. 下图捕获到了its的指代对象Law（语义特征）：

![image-20211104204055306](images/image-20211104204055306-16360296569807.png)

显然，引入 Self Attention 后会更容易**捕获句子中长距离的相互依赖的特征**，因为如果是 RNN 或者 LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

但是<font color="red"> Self Attention 在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，从而挖掘出句子内部更多的且有效的特征。</font>除此外，Self Attention 对于增加计算的**并行性**也有直接帮助作用。这是 Self Attention 逐渐被广泛使用的主要原因。

### Multi-head Attention 模型

多头注意力（multi-head attention）是利用多个查询 $Q = [q_1, · · · , q_M$] 来平行地计算从输入信息中选取多个信息。每个注意力关注输入信息的不同部分，然后再进行拼接：

![img](images/v2-27673fff36241d6ef163c9ac1cedcce7_1440w-163607616045217.jpg)

### Attention 机制的应用

前文有述，Attention 机制在深度学习的各种应用领域都有广泛的使用场景。上文在介绍过程中我们主要以自然语言处理中的机器翻译任务作为例子，下面分别再从图像处理领域和语音识别选择典型应用实例来对其应用做简单说明。

**应用1. 图片描述（Image Caption）**

![img](images/v2-19e1f44cc6b0d1f97b022cd1281cf350_1440w-16360301364459.jpg)

图片描述（Image Caption）是一种典型的图文结合的深度学习应用，**输入一张图片，人工智能系统输出一句描述句子**，语义等价地描述图片所示内容。很明显这种应用场景也可以使用 Encoder-Decoder 框架来解决任务目标（见上图），此时 Encoder 输入部分是一张图片，一般会用 CNN 来对图片进行特征抽取，Decoder 部分使用 RNN 或者 LSTM 来输出自然语言句子。

此时如果加入 Attention 机制能够明显改善系统输出效果，Attention 模型在这里起到了类似人类视觉选择性注意的机制，在输出某个实体单词的时候会将注意力焦点聚焦在图片中相应的区域上。下图是根据给定图片生成句子 “A person is standing on a beach with a surfboard.” 过程时每个单词对应图片中的注意力聚焦区域。

![preview](images/v2-3a652b0dd6fd983cdca6d6ddc3f12212_r-163607440775211.jpg)

下图仍为图片描述的4个示例，每个例子上方左侧是输入的原图，下方句子是人工智能系统自动产生的描述语句，上方右侧图展示了当 AI 系统产生语句中划横线单词的时候，对应图片中聚焦的位置区域。比如当输出单词 dog 的时候，AI 系统会将注意力更多地分配给图片中小狗对应的位置。

![preview](images/v2-515b0f7b340d65e709fbf41dcc949ab3_r-163607458923513.jpg)

**应用2. 语音识别**

![preview](images/v2-de056a44b848c63c40ea466696115855_r-163607466613715.jpg)

语音识别的任务目标是**将语音流信号转换成文字**，所以也是 Encoder-Decoder 的典型应用场景。Encoder 部分的 Source 输入是语音流信号，Decoder 部分输出语音对应的字符串流。

上图展示了在 Encoder-Decoder 框架中加入 Attention 机制后，当用户用语音说句子 “how much would a woodchuck chuck” 时，输入部分的声音特征信号和输出字符之间的注意力分配概率分布情况，颜色越深代表分配到的注意力概率越高。从图中可以看出，在这个场景下，Attention 机制起到了将输出字符和输入语音信号进行对齐的功能。

-- --

上述内容仅仅选取了不同AI领域的几个典型 Attention 机制应用实例，Encoder-Decoder 加 Attention 架构由于其卓越的实际效果，目前在深度学习领域里得到了广泛的使用，了解并熟练使用这一架构对于解决实际问题会有极大帮助。

# Transformer

> **【Transform 的定义】**Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution。

>  参考资料：
>
> 1. [详解Transformer (Attention Is All You Need)](https://zhuanlan.zhihu.com/p/48508221) 
>
> 2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) 
>
> 3. [细讲 | Attention Is All You Need](https://cloud.tencent.com/developer/article/1377062) 

## 提出背景 / 解决的问题

在处理序列转录问题时，考虑到 RNN（或 RNN 变种—— LSTM，GRU 等）的计算为是顺序的，也就是说 RNN 相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1. 时间片 ![[公式]](https://www.zhihu.com/equation?tex=t) 的计算依赖 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 时刻的计算结果，这样限制了模型的并行能力；

2. 顺序计算的过程中信息会丢失，尽管 LSTM 等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象，LSTM 依旧无能为力。


CNN 无法解决上面两个问题。CNN 对比较长的序列难以建模，因为卷积运算每次是看一个小的窗口，如一个3×3的像素块，一旦两个像素隔得较远，就需要多次卷积才能学习到它们的特征、得到相关性。

Transformer的提出解决了上面两个问题：

- 1. Transformer  使用Attention机制，从而可以“动态”地生成不同连接的权重，处理变长的信息序列；且不会因句子过长而逐渐丢失很多细节。
- 2. Transformer 不是类似 RNN 的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

## 模型纵观

论文中的验证 Transformer 的实验室基于机器翻译的，下面我们就以机器翻译为例子详细剖析 Transformer 的结构。

在机器翻译中，Transformer 可概括为下图：

![img](images/the_transformer_3-163608219146824.png)

Transformer 的本质上是一个 Encoder-Decoder 的结构，那么上图可以表示为下图：

![img](images/The_transformer_encoders_decoders-163608224176326.png)

如论文中所设置的，编码器由 6 个编码 block 组成，同样解码器是 6 个解码 block 组成：

![img](images/The_transformer_encoder_decoder_stack-163608231872228.png)

每个 encoder 都是一样的，它可以分成两个子层：

![img](images/Transformer_encoder-163608241676330.png)

encoder 的输入首先进入到 self-attention 层；self-attention 的输出进入到前馈神经网络==（The exact same feed-forward network is independently applied to each position.）==。

每个 decoder 也都是一样的，它与 encoder 的区别在于多了一个 Encoder-Decoder Attention 。

<img src="images/image-20211105113140221.png" alt="image-20211105113140221" style="zoom:67%;" />

论文中给出的结构：

![image-20211106103730793](images/image-20211106103730793-163616625218696.png)

-- --

Now that we’ve seen the major components of the model, let’s start to look at the various vectors/tensors and how they flow between these components to turn the input of a trained model into an output.

-- --

## 各个模块の详细解读

> 1、2、4 是 encoder 里的；5、6 是 decoder 里的；7、8 是 encoder 和 decoder 共有的。

### 1 Input Embedding

通过Word2Vec等词嵌入方法将输入序列的每个词汇转化成一个向量表示（论文中使用的词嵌入的维度/向量的维度为 $d_{model} = 512$）：

![img](images/embeddings-163608340686832.png)

> **Q：**原论中编码器与解码器的 Embedding 层的权重为什么要乘以 $\sqrt{d_{model}}$ ？
>
> **A：**为了让 embedding 层的权重值不至于过小，乘以 $\sqrt{d_{model}}$ 后与位置编码的值差不多，可以保护原有向量空间不被破坏。

### 2 Positional Encoding

Recurrence/Convolution 结构能很好地捕捉到序列顺序信息，但 Transformer 没有利用到这两种结构，所以就想了其他办法来使模型具备这一能力，即位置编码（Position Embedding）。

> "无法捕捉序列顺序信息"意味着：无论句子的结构怎么打乱，学习到的结果都是一样的。

具体地说，位置编码会在词向量中加入单词的位置信息，这样 Transformer 就能区分不同位置的单词了。

![image-20211105162629835](images/image-20211105162629835-163610079143560.png)

那么怎么编码这个位置信息呢？常见的模式有：a. 根据数据学习；b. 自己设计编码规则。在这里作者采用了第二种方式，并给出如下编码方式：

<img src="images/image-20211105162828260.png" alt="image-20211105162828260" style="zoom: 67%;" />

在上式中，$pos$ 表示单词的位置， $i$ 表示单词的维度。关于位置编码的实现可在 Google 开源的算法中`get_timing_signal_1d()`函数找到对应的代码。

Position Embedding 本身是一个绝对位置的信息，但在 NLP 任务中，相对位置也很重要。Google 选择前述的位置向量公式的一个重要原因是，由于：

![image-20211106110545466](images/image-20211106110545466-1636167946544100.png)

这表明 $PE_{pos+k}$ 可以表示成 $PE_{pos}$ 的线性变换，这提供了表达相对位置信息的可能性。

If we assumed the embedding has a dimensionality of 4, the actual positional encodings would look like this:

![img](images/transformer_positional_encoding_example-163610126608665.png)

> 最底层的 encoder 的输入就是上面得到的每个单词的向量表示；对于其他的 encoder，其输入为上一个 encoder 的输出。所有 encoder 都是接收一组维度相等的向量。“一组”里到底有多少向量——这是我们唯一的超参数，通常，这个超参数设置为数据集（数据集就是很多很多个句子）中最长的那个句子的长度。流程可表示为下图：
>
> ![img](images/encoder_with_tensors_2.png)

### 3 Self-Attention 的运作

Self-Attention 是 Transformer 最核心的内容，然而作者并没有详细讲解，下面我们来补充一下作者遗漏的地方。回想 Bahdanau 等人提出的 Attention，其核心内容是**为输入向量的每个单词学习一个权重**（此即 Attention 的输出结果）。

> 论文中对 Attention 函数的描述：An attention function can be described as mapping a **query** and a set of **key-value pairs** to an output, where the query, keys, values, and output are all vectors. The <font color="red">output is</font> computed as a weighted sum of the values, where the <font color="red">weight assigned to</font> each value is computed by a **compatibility function** of the **query** with the corresponding **key**.

> Transformer 中使用的 Attention 又具体的称之为 **Scaled Dot-Product Attention**，因为在计算 Attention 时，进行了缩放（scaled）以及使用点积作为相似性的计算。
>
> <img src="images/image-20211106104602440-163616676505298.png" alt="image-20211106104602440" style="zoom:67%;" />
>
> 下面会具体讲解这一计算过程。

假设我们要翻译下面这个句子：

`The animal didn't cross the street because it was too tired. `

What does “it” in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.

When the model is processing the word “it”, **self-attention** allows it to associate “it” with “animal”.

self-attention 的计算过程就和上面 [前导知识 > 注意力机制 > Attention 机制的本质思想] 里讲到的过程一样。步骤如下：

<font color = "red"> **Step0、**</font> 前面提到的词嵌入——将每个单词表示成了维度为 512 的向量 $X$。

![image-20211105125129276](images/image-20211105125129276-163608789052039.png)

<font color = "red"> **Step1、**</font> 嵌入向量 $X$ 分别乘以三个不同的权值矩阵 $W^Q, W^K, W^V$ 得到 Query 向量 $Q$，Key 向量 $K$ 和 Value 向量 $V$ 。其中，$Q, K, V$ 尺寸均为 $d_k = 64$； $W^Q, W^K, W^V$ 尺寸均为 $512 × 64$ 。【权值矩阵是在训练过程中学习出来的；embedding 的维度可以和对应的 Q、K、V 的维度不等，但是 Q、K、V 三者的维度要保持一致】

![image-20211105123653723](images/image-20211105123653723.png)

<font color = "red"> **Step2、**</font> 为每个向量 $X$ 计算一个 score（score 就是某个单词与其他单词的相似度/相关性。。也就是考虑某个单词时，应该多大程度上也考虑其他各个单词，显然，与自己计算出来的 score 是最大的）。这里计算 score 的方式是令两向量**点积**，即 $score = q · k$：

> - 点积是计算两向量相似性的一种方法！点积是一种向量运算，但其结果为某一数值，而非向量。
>
> - 例如，$A = [a_1, ..., a_n], B=[b_1, ..., b_n]$，则：
>
>   $A · B = a_1b_1 + ... + a_nb_n = |A|·|B|·cos{\theta}$
>
>   其中，$|A|=\frac{1}{2}(a_1^2 + ... + a_n^2), |B|=\frac{1}{2}(b_1^2 + ... + b_n^2)$  ，${\theta}$ 为两向量之间的夹角。
>
> - 那么，当且仅当 $A⊥B$ 时，$A · B = 0$ , 这表明相似度为0！

![img](images/transformer_self_attention_score.png)

<font color = "red"> **Step3、**</font> ==为了梯度的稳定，Transformer 令 score 除以 $\sqrt{d_k}$==  。

<font color = "red"> **Step4、**</font> 对上一步的结果再传入 softmax 进行正规化。

![img](images/self-attention_softmax-163609273701043.png)

<font color = "red"> **Step5、**</font> 上一步得到的结果 softmax 被看做是 Value 的权重，求加权了的每个输入向量 $softmax · Values$。

<font color = "red"> **Step6、**</font> 加权求和。

![img](images/self-attention-output-163609361000445.png)

按照这一步骤计算得到每个单词的 $z$，这些结果就作为 FFNN 的输入。

-- --

上面的数据都是向量，实际计算过程中是采用基于矩阵的并行计算方式：

<font color = "red"> **Step1、**</font> 所有嵌入向量组合成一个矩阵，分别乘以 $W^Q, W^K, W^V$ 得到 Query 向量 $Q$，Key 向量 $K$ 和 Value 向量 $V$ ；

![img](images/self-attention-matrix-calculation-163609406951247.png)

<font color = "red"> **Step2、**</font> 以矩阵的形式将很容易的计算出最后结果：

![img](images/self-attention-matrix-calculation-2-163609413408349.png)

> 这个 $Z$ 的第一行就是向量 $z_1$, 第二行就是向量 $z_2$ 。Softmax做正规化时也是对每一行进行操作的。

此即论文中 self-attention 的计算。

### 4 Multi-Head Attention 的运作

> 论文中的图形描述：
>
> <img src="images/image-20211106104908278.png" alt="image-20211106104908278" style="zoom:67%;" />
>
> 下面对其进行详细解释。

Multi-Head 指的是有多个 Query/Key/Value 的权重矩阵 $W^Q, W^K, W^V$ （Transformer 中使用了8个），每一组 $W^Q, W^K, W^V$ 的值被随机初始化。每一组值都与嵌入向量 $X$ 进行运算，从而**得到不同的表示子空间**。

除此之外，Multi-Head 扩展了模型专注于不同位置的能力。在上面的示例中，$z_1$ 包含所有一些其他编码，但它可以由实际单词本身主导。如果我们只是想翻译一句 "The animal didn’t cross the street because it was too tired"，应用注意力机制是机器知道 "it" 的指代，这是很有用的。但句子中其他的联系呢？我们想知道多方面的联系。

> **对 Multi-Head 的通俗理解：**
>
> - 相当于 CNN 中使用的多个滤波器，就是为了捕获到内部具有的不同方面的联系。
> - 相当于人有多个头，有了多个头后就能一时间注意到多方面的东西了！

如下图，得到两组 Query/Key/Value ：

![img](images/transformer_attention_heads_qkv-163609856562551.png)

有多组 Query/Key/Value，那么最终将输出多个 $Z$： 

![image-20211105155031451](images/image-20211105155031451-163609863267352.png)

对这多个 $Z$ 需要近一步处理后才能传入 FFNN，因为 FFNN 的输入是单个矩阵。如何进行处理？我们的做法是：![image-20211105155409745](images/image-20211105155409745.png)

> concat 操作相比于相加取平均，能更好的保留原来的信息。

上面的过程表示成公式，即：

![image-20211106123321593](images/image-20211106123321593.png)

**【总结】**

![img](images/transformer_multi-headed_self-attention-recap-163609915097954.png)

**【对比】**

未使用 Multi-Head 时的结果：

![img](images/transformer_self-attention_visualization-163608512163035-163609954529455.png)

使用了 “2-Head” 的结果：

![img](images/transformer_self-attention_visualization_2-163609959445757.png)

一个注意力头把焦点聚集在 "the animal"，其他的头的注意力焦点在 “tired” 上。

使用更多的头，则模型结果将很难得到解释：

![img](images/transformer_self-attention_visualization_3-163609995011859.png)

### 5 Masked Multi-Head Attention 的运作

decoder 中的 self-attention 与 encoder 中的 self-attention 的运行有所区别（不然为什么多了个“masked”）：

decoder 中的 self-attention 层只允许 / 当前正在处理的序列的部分 / 的前面的部分 / 作为输入（“/” 表示断句）。This is done by **masking** future positions (setting them to `-inf`) before the softmax step in the self-attention calculation. 

Q、K、V 是根据同一个输入向量计算得到的==（这个输入向量好像是由待预测的那个单词得到的）==，所以是**属于 self-attention**。这里的输入向量的维度与 encoder 的相同，但是长度可能不一样（比如 encoder 那里一直是 n 个 d 维的向量，decoder 那里可能是 m 个 d 维的向量）。

> **PS.** 去看后面 [一个 Encoder 和 Decoder 的层数为 2 的 Transformer] 节里的第三幅图，对比底层的 encoder 和 底层的 decoder 的输入，就能明白上面所说的了。

### 6 Encoder-Decoder Attention

decoder 中多了个 encoder-cecoder attention。在 encoder-decoder attention 中， $Q$ 来自于上一个 decoder 的输出，$K$ 和 $V$ 则来自于与顶层 encoder 的输出。所以它**不叫 self-attention** 了嘛——不是由同一个向量转换得到的 Q、K、V。

### 7 Linear 层和 Softmax 层

最后的 decoder 的输出是数字类型的矢量，那么如何将其转换成最终的词汇输出呢？这就是 Linear 层和 Softmax 层要完成的工作。

首先，Linear 层将 decoder 的输出转换成一维的一个很长的向量。假设我们的模型知道 100,000 个英文单词，那么 Linear 层的输出就是一个长为 100,000 的向量，每一个元素代表可能为对应单词的得分情况，得分越大，说明最终输出就越可能为它。

Softmax 层负责对上面的向量进行归一化。

![img](images/transformer_decoder_output_softmax-163616294349980.png)

### 8 FFNN

<img src="images/image-20211106111036366.png" alt="image-20211106111036366" style="zoom:67%;" />

FFNN 包括包括两个线性变换和一个 ReLU 激活输出：

- 线性变换：$A=xW_1+b_1$
- ReLU 激活：$B = max(A)$ 
- 线性变化：$C = BW_2 + b_2$

FFNN 是**对每一个单词分别**进行操作（见下节第一幅图），不同的位置所做的线性变换是一样的。假设输入 x 的维度为 512，那么 A 的维度为 2048（扩大了四倍），C 的维度为 512（还原）。==这样做就相当于卷积运算在提取特征。。== 

### 9 Residuals | 层归一化(Add & Normalize)

Transformer 结构中每一个 sub-layers 都使用到了残差连接（防止梯度消失），紧接着是层规范化步骤。

![img](images/transformer_resideual_layer_norm-163610302444067-163610302600069.png)

计算式：<img src="images/image-20211106111442667.png" alt="image-20211106111442667" style="zoom: 50%;" />

可视化：【注意 x 和 Sublayer(x) 分别代表什么】

![img](images/transformer_resideual_layer_norm_2-163610311443671.png)

> **Layer Norm(层归一化)** 与 **Batch Norm(批量归一化)**：
>
> - Batch Norm 对每个特征/通道里的元素进行归一化 --> 不适合序列长度会变的 NLP 应用
> - Layer Norm 对每个样本里的元素进行归一化 --> 适合序列长度会变的 NLP 应用
>
> ![image-20211109201502634](images/image-20211109201502634-16364601036985.png)
>
> - 它们的出发点都是让该层参数稳定下来，避免梯度消失或者梯度爆炸，方便后续的学习。但是也有侧重点。一般来说，如果你的特征依赖于不同样本间的统计参数（CV领域），那BN更有效。因为它抹杀了不同特征之间的大小关系，但是保留了不同样本间的大小关系。而在NLP领域，LN就更加合适。因为它抹杀了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系。对于NLP或者序列任务来说，一条样本的不同特征，其实就是时序上字符取值的变化，样本内的特征关系是非常紧密的。
> - LN 的作用：允许使用更大的学习率，加速训练。有一定的抗过拟合作用，使训练过程更加平稳。
> - ==原理详解（待看）：==https://blog.csdn.net/qq_36158230/article/details/117399396

## 一个 Encoder 和 Decoder 的层数为 2 的 Transformer

![img](images/transformer_decoding_1-163616603262093.gif)

上图，最底层的 Encoder 处理输入序列，最顶层的 Encoder 的输出 Z 再次作为 K 和 V。每一个 Decoder 中的  encoder-decoder attention 层将使用 K、V，这能帮助 decoder 聚焦到序列的合适位置上。decoder 阶段的每一步都会输出序列中的元素，最终输出一个翻译词。==【第一次运作时，底层的 decoder 的输入就是一个表示开始的东西】==

重复上面的步骤，直到遍历完整个句子，最终输出一个完整的翻译句子。每一步 decoder 最终的输出还将作为底部 decoder 在下一步时需要用到的输入信息，而且也会给它安排一个 Positional Encoding 来表明它的位置。如下图：

![img](images/transformer_decoding_2.gif)

**【总体结构】**

![img](images/transformer_resideual_layer_norm_3-163610325899073.png)

## 模型训练与预测

### 训练

我们要对有标签数据集进行训练，训练结果与真实标签比较。

假设我们的输出词汇中只有 6 个单词：“a”, “am”, “i”, “thanks”, “student”, and “<eos>” (‘end of sentence’ 的缩写)。

![img](images/vocabulary-163616392161382.png)

我们使用 one-hot 的编码形式对这些单词进行编码表示。例如，单词 “am” 的表示：

![img](images/one-hot-vocabulary-example-163616399680784.png)

现在，假设我们要做一个简单的翻译任务：将 “merci” 翻译成 “thanks”（通常翻译任务是复杂的——对句子进行翻译而不是一个单词）。那么我们希望 softmax 得到的向量中对应 “thanks” 处的概率最大。

模型参数首先被随机初始化：

![img](images/transformer_logits_output_and_label-163616421991086.png)

上面两个向量会进行比较（如使用交叉熵损失函数），然后通过反向传播是模型的输出更加接近真实标签。

如果不是对单词而是对一个句子翻译，如将 “je suis étudiant” 翻译成 “i am a student”. 那么我们希望我们的输出是这样的：

![img](images/output_target_probability_distributions-163616464598188.png)

我们的目标是使训练出的模型参数尽可能实现上面的结果。

After training the model for enough time on a large enough dataset, we would hope the produced probability distributions would look like this:

![img](images/output_trained_model_probability_distributions-163616471463290-163616471554792.png)

> **【greedy search & beam search】**
>
> 如何得到翻译后的最适合的序列组合，即翻译得最接近原句子？Transformer 中使用的搜索方式是 beam search。参考 https://zhuanlan.zhihu.com/p/82829880 .

### 编码器做了什么

> 参考：https://zhuanlan.zhihu.com/p/405543591

<font color="red">**【编码器无论是训练还是测试，都是并行工作。】**</font> 

- 对于一个【B F E】的数据块（B 为一个批次的样本数、F 为一个样本中的字符数、E 为每个字符的embedding长度），其经过一层编码器后，数据块的形状仍为【B F E】。而在一个编码器中，尽管计算非常复杂，但是通过**矩阵运算**，每条样本的计算都是同时发生的，每条样本内的字符之间的自注意力分数计算也是同时发生的。

- 编码器就是为了得到句子内部每个词之间的联系，所以训练、预测时的工作都是一样的。

### 解码器做了什么

> 参考：https://zhuanlan.zhihu.com/p/405543591

<font color="red">**【解码器在训练时并行，测试时串行。】**</font> 

> 下面主要讲解的是最底层的那个 decoder 里的第一个 attention 的工作。

#### 训练时的解码器

我们都知道，<font color="brown">最底层的那个 decoder 里的第一个 attention 的输入</font>（后文直接说成“解码器的输入”了）是<font color="blue">此前得到的翻译结果</font>。例如一个中译英的机器翻译任务需要将 `我爱你`翻译为 `I love you`。那么【我爱你】就是编码器的输入，解码器的输入就是 <font color="blue">`l love you`</font>。

是不是觉得上面两处蓝色字体所表达的意思矛盾了？！不是应该先预测出 `l`，结合 `l` 再预测出 `love`，最后结合`l love` 预测出 `you` 嘛？为什么这里是直接将 `I love you` 作为输入？

是的，这就是训练时的效果。这是因为训练时，我们是已经知道真实的翻译结果了，而预测时由于不知道真实结果，所以只能一步一步来（串行）。但是，训练的过程还是要遵循"先预测出 `l`，结合 `l` 再预测出 `love`，最后结合`l love` 预测出 `you` "这一逻辑！如何做到呢？那就是使用 **mask** 啦！

假设我们对 `I` `love` `you` 的编码如下：

![img](images/v2-037f151579d82d45dc5a77e2c1a5d94a_1440w-16365980289543.jpg)

假设得到了 Q、K 矩阵，且令 Q 与 K 的转置相乘得到每个字符之间的相关性如下：

![image-20211111110318137](images/image-20211111110318137-16365998004184.png)

然后需要再乘以矩阵 V。不过，乘以 V 之前，该矩阵需要做 mask 处理。

按理说，预测 `I` 的时候，是不可以看到`love`、`you` 的信息的，预测 `love` 的时候，也不可以看到 `you` 的信息。因此该矩阵的上半部分要加上一个极小负值，因为在后续做 **softmax** 的时候，极小负值就可以变为 0 了。mask 后的矩阵如下：

<img src="images/image-20211111110619421.png" alt="image-20211111110619421" style="zoom:50%;" />

然后该 mask 过后的矩阵与 V 相乘：

![img](images/v2-f85f927a0cd47d584c80f9e95ea95b83_1440w-16366000451597.jpg)

最终得到的矩阵 B：`I` 那一行只考虑了 A 矩阵中 `I` 那一行的信息，`love` 那一行综合考虑 A 矩阵中 `I` 与 `love` 两行的信息，`you` 那一行综合考虑 A 矩阵中 `I` 、 `love`、`you` 两行的信息。

这样，在训练时使用 mask 机制就能够以并行的计算完成本该串行的计算，从而提高运算。

这里得到的矩阵 B 将作为 Q 传入到 encoder-decoder attention ，与编码器传过来的 K、V 进行运算。之后在经过一系列的运算就相当于一次完成了有 3 个分类任务的分类工作。每个分类结果（向量）中对应的值最大的就是得分最大的。

再次总结这样做的好处：

- 可以并行预测
- 前面的多分类任务并不会融合序列后面字符的信息（**mask**）
- 后面的字符可以融合前面字符的信息，并且可以保证前面的字符信息一定是对的(**teach force**)

#### 预测时的解码器

测试阶段就比较好理解了，我们必须预测出当前字符，才可以利用当前字符再去预测后面的字符，根本不会看到之后的结果，所以不需要 mask 了。就是串行的完成了整个任务的。

## 总结

> 这一部分不是来自论文，是别人对论文的评价。

**优点**：（1）虽然Transformer最终也没有逃脱传统学习的套路，Transformer也只是一个全连接（或者是一维卷积）加Attention的结合体。但是其设计已经足够有创新，因为其抛弃了在NLP中最根本的RNN或者CNN并且取得了非常不错的效果，算法的设计非常精彩，值得每个深度学习的相关人员仔细研究和品位。（2）Transformer的设计最大的带来性能提升的关键是将任意两个单词的距离是1，这对解决NLP中棘手的长期依赖问题是非常有效的。（3）Transformer不仅仅可以应用在NLP的机器翻译领域，甚至可以不局限于NLP领域，是非常有科研潜力的一个方向。（4）算法的并行性非常好，符合目前的硬件（主要指GPU）环境。

**缺点**：（1）粗暴的抛弃RNN和CNN虽然非常炫技，但是它也使模型丧失了捕捉局部特征的能力，RNN + CNN + Transformer的结合可能会带来更好的效果。（2）Transformer失去的位置信息其实在NLP中非常重要，而论文中在特征向量中加入Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷。

-- --

**李沐：**

- 标题 + 作者

- 摘要 Abstract

- 结论 Conclusion

- 导言 Introduction

  > 对摘要部分展开介绍

- 相关工作 Background

  > 主要讲跟你论文相关的那些论文是谁、跟你的联系是什么以及你跟他们的区别是什么

- 模型 Model Architecture

- 实验 Training

- 评论 (李沐)

  - Transformer开创了继MLP、CL、RNN之后的第四大模型。

  - Transformer 对 NLP领域的贡献，类似于CNN对计算机视觉领域的贡献，我们训练出一个CNN模型，使得别的任务也能够从中受益。另外，CNN给整个计算机视觉的研究者提供了一个同样的框架，使得只要学会CNN就行了，而不需要去管以前跟任务相关的那么多的专业知识，比如做特征提取、对整个任务如何建模。在Transformer之前，我们需要做各种各样的数据文本的预处理 ，然后根据NLP的任务设计不一样的架构，而现在可以直接使用Transformer这个架构就能够在各个任务上得到非常好的结果，而且其预设模型使用起来可以使训练变得很简单。不过，Transformer现在不仅在NLP领域应用得好，在图片的处理上、语音上、视频上等也都取得了很大的进展。这是一个很具有影响力的事件，因为在此之前：计算机视觉领域使用CNN，NLP领域使用RNN，然后别的领域用别的模型；而现在：所有领域竟然能够通用一个Transformer模型！未来的一个研究方向就是：混合训练图片、语音、文本等信息，训练出一个更好、更大的模型。

  - 我们对Transformer的理解还处于一个初级阶段，尽管标题叫做“Attention is all you need”，但实际上，并非仅使用的是 Attention（Attention起到的作用是把整个序列的信息聚合起来），该模型还使用到了残差连接、MLP，也都是缺一不可的。

  - 使用Transformer需要很大的训练集才能训练出很好的结果

## 问答

![image-20211110215454489](images/image-20211110215454489-16365524957431.png)

![image-20211110215700158](images/image-20211110215700158-16365526221452.png)

![image-20211110220609321](images/image-20211110220609321-16365531704263.png)

**Transformer模型的计算复杂度是多少？**

n 是序列长度，d 是 embedding 的长度。Transformer 中最大的计算量就是多头自注意力层，这里的计算量主要就是 Q、K 相乘再乘上 V，即两次矩阵相乘。Q、K 相乘是矩阵 `n x d `乘以 `d x n`，这个复杂度就是 $n^2d$。

**Transformer中三个多头自注意力层分别有什么意义与作用？**

Transformer中有三个多头自注意力层，编码器中有一个，解码器中有两个。

编码器中的多头自注意力层的作用是将原始文本序列信息做整合，转换后的文本序列中每个字符都与整个文本序列的信息相关（这也是Transformer中最创新的思想，尽管根据最新的综述研究表明，Transformer的效果非常好其实多头自注意力层并不占据绝大贡献）。示意图如下：

<img src="images/image-20211111092941780.png" alt="image-20211111092941780" style="zoom:67%;" />

解码器的第一个多头自注意力层比较特殊，原论文给其起名叫Masked Multi-Head-Attention。其一方面也有上图介绍的作用，即对输入文本做整合（对与翻译任务来说，编码器的输入是翻译前的文本，解码器的输入是翻译后的文本）。另一个任务是做掩码，防止信息泄露。拓展解释一下就是在做信息整合的时候，第一个字符其实不应该看到后面的字符，第二个字符也只能看到第一个、第二个字符的信息，以此类推。

解码器的第二个多头自注意力层与编码器的第一个多头自注意力层功能是完全一样的。不过输入需要额外强调下，我们都知道多头自注意力层是通过计算QKV三个矩阵最后完成信息整合的。在这里，Q是解码器整合后的信息，KV两个矩阵是编码器整合后的信息，是两个完全相同的矩阵。QKV矩阵相乘后，翻译前与翻译后的文本也做了充分的交互整合。至此最终得到的向量矩阵用来做后续下游工作。







![image-20211110220733718](images/image-20211110220733718-16365532547134.png)

> 参考自：https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247546392&idx=3&sn=bc00064b5ba262c01d797407324b4db6&chksm=ebb70eccdcc087da9cbd380fc5535bfa7027ac8b224eedc08880851e3cc950be2cc790e48a1e&mpshare=1&scene=23&srcid=1110Rc2r7UH8rN6GX8omUTfk&sharer_sharetime=1636552376203&sharer_shareid=faf494c081fffd893ec55dcd41b9199b#rd

## 代码学习

```
Step 1. 设置变量、参数
Step 2. 数据处理
Step 3. Transformer 模型搭建
	Step 3.1 Transformer 总体架构 —— Transformer()
		Step 3.1.1 Encoder
			Encoder Layer
				MultiHeadAttention
					Add & Norm
				Feed Forward
					???
					Add & Norm
		Step 3.1.2 Decoder
		Step 3.1.3 Linear
Step 4. 训练
Step 5. 测试
```





## PPT

<img src="images/image-20211126220044184.png" alt="image-20211126220044184" style="zoom:80%;" />

首先，我们拿到一份输入数据，这里是4个token组成的序列矩阵。

这里第一行中的11， 12， 13， 14， 15表示第一个token的第1个特征，第2个特征。类似的，21表示第二个token的第一个特征。

接下来，我们将输入映射到注意力空间中，生成我们需要的三大序列矩阵——Query、Key和Value。这里，我们采用等维映射，也就是输入的特征维度与映射空间中的序列矩阵特征维度一致。当然，是可以不等维的。

-- --

在进行接下的步骤前，我们再具体理清一下三者的含义——

Query中 每一行对应着从每一个Token中映射得到的查询信息；

Key的每一行对应着每一个Token的一个描述信息；

Value的每一行即对应着每一个Token映射过来的特征值。

因为key-value是相对应的，所以Key中的每一行可以理解为对应Value中的每一行特征值的描述信息。

-- --

之后，我们需要先计算token之间的相似度，使用的是点积模型。

在点积模型中，我们将Query与Key进行矩阵乘法，利用矩阵乘法实现并行的查询，完成每一个Query中的Token与Key中所有的Token进行一次查询，得到注意力分布矩阵。

然后将注意力分布作用于Value中的每一个Token得到注意力结果输出，此时的输出与Value的Shape等大。

-- --

具体过程，我们现在就开始演示说明——

首先，执行Query与Key的注意力分布计算，我们先将Key转置，Query保持不变，从而保证矩阵乘法能够执行。矩阵Query中的第一行token与Key矩阵中的每一列进行点积，依次得到一个结果，最后得到一个维度为4的结果。

每一个结果所表示的含义是  当前查询与Key中描述信息的相关程度，或者说突出程度。之后 将每一个Query依次作用，就会得到一个4*4的注意力分布矩阵（注意力分配系数）。

然后通过Softmax进行归一化。

-- --

下面就进行加权求和。具体的操作就是，注意力分布矩阵中的每一【行】与Value矩阵中的每一【列】进行点积。因为注意力分布矩阵中的第i【行】代表第i个token与每个token的相似度，或者说是第i个查询与每个key的匹配程度；Value矩阵中的第j【列】代表第 j 种特征。所以最终得到的注意力结果矩阵中的（第i行第j个）元素的含义应该是第 i 个Query对第 j 种特征的关注程度。

总体来看，这个最终结果就是对输入数据抽象出来的特征。

-- --

那么对于识别图片中的小猫这个问题来说，我们就可以假设把原始图像划分成四行五列，每一个patch就作为一个token，像这样进行运算，那么最终得到的注意力结果矩阵，就应该是有小猫的那部分的值最大。

-- --

最后，由于这里的QKV都来自同一个输入，所以称为 Self Attention 自注意力，可以理解为自己对自己的注意。

-- --

就是学完上面这些内容后，我的总结就是：

我刚看完Transformer的时候，以为encoder、decoder必须连起来用，但现在我明白了，其实并不是的。不管是encoder还是decoder，其内部最关键的就是attention运算，就好像卷积网络中的最核心的是卷积运算。attention运算和卷积运算，都是为了发现输入数据的特征。只不过对于卷积网络来说，需要一次又一次的做卷积运算来发现更深层次的特征；而对于Attention来说，是能够直接计算出来输入数据对每种特征的关注程度。【ViT中还提到attention相比于卷积的一个优势在于：数据集对于卷积的进一步提升预训练的性能是有限的，而对于transformer encoder来说，数据集更大、或者模型更深都还不会使模型的性能达到饱和的状态。】

至于是否使用decoder，应该根据具体的任务而定。原始的Transformer要使用decoder，是因为对于翻译任务来说，我们不仅要关注原始句子的特征，还要在后面的翻译过程中去关注已经翻译出来的内容。所以，encoder就用来得到原始句子的特征，decoder用来发现已经翻译出来的内容的特征，而且还会进一步将两种特征进行结合，最终匹配到合适的结果。



![image-20211210203836077](images/image-20211210203836077.png)
