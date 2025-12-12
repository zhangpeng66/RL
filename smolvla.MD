# <center>smolvla</center>

## 研究现状
SmolVLA 是一套轻量级视觉-语言-行动（VLA）策略：前端用小型 VLM（视觉 SigLIP + 语言 SmolLM2）做感知与理解；后端用一个“动作专家”专门预测一段连续的低层控制。它与Pi0相比，参数规模少了将近10倍只有约0.45B（450M）。

## 原理

模型结构主要有前端的VLM+后端的动作专家Action Expert组成，包括以下组件：
输入：文本指令token+视觉token（多摄像头采集的图像）+机器的状态token（关节角、传感器）。
VLM（感知端）：语音采用SmolVLM-2，VLM共有L层，但是只N=⌊L/2⌋层隐藏表示喂给动作专家，视觉采用SigLIP。
Action Expert（控制端）：一个Flow Matching Transformer，以以Cross Attention→  Self Attention→ Cross Attention的“三明治”层为基本单元，按块预测n步动作序列。
输出：一次预测长度为n的动作块，对应机器的控制指令。
![alt text](image-42.png)
SmolVLA与Pi0有很多相似之处，不过其背后有四个关键设计，分别是Layer Skipping（层跳过）、Visual tokens reduction（视觉token压缩）、动作专家交替Self-Attn与Cross-Attn、异步推理。

## 层跳过Layer Skipping
层跳过就是把感知端的VLM（视觉+语言）的解码器中间层拿来当条件特征，具体的做法就是只去$N=L/2$层的隐藏层表示送给动作专家，VLM权重冻结不训练。把“文本指令 token、图像 token、状态 token”拼接，送入解码器；在第N层获取特征信息H，然后用一个线性投影到Action expert所需的维度$d_a$作为$K/V$。   

## 视觉token  
在transformer里面，“token”就是序列里的一个位置。对图像来说，我们把一张图拆成很多小块（patch）或网格上的特征点，每个块/点用一个向量表示，这个向量就是视觉token，设patch的特征图大小为$\frac{H}{p} \times \frac{W}{p} \times d$,选一个下采样因子r (整数)，做 space-to-depth：
![alt text](image-44.png)
压缩过程如下：
输入图像尺寸为$512\times512$
patch大小为$p=16 \Rightarrow 32 \times 32$
![alt text](image-43.png)