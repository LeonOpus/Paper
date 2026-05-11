# Attention is all you need: PyTorch 实现

这是 Transformer 模型的 PyTorch 实现，基于论文 "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017)。


这是一个新颖的序列到序列框架，利用 **自注意力机制** 而非卷积操作或循环结构，在 **WMT 2014 英德翻译任务** 上取得了最先进的性能。(2017/06/12)

> 官方的 TensorFlow 实现可以在以下位置找到：[tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)。

> 要了解更多关于自注意力机制的信息，您可以阅读 "[A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)"。

<p align="center">
<img src="http://imgur.com/1krF2R6.png" width="250">
</p>


该项目现在支持模型训练和使用训练好的模型进行翻译。

请注意，该项目仍在开发中。

**BPE 相关部分尚未完全测试。**


如果有任何建议或错误，请随时提出 issue 告诉我。:)


# 使用方法

## WMT'16 多模态翻译：德英

以下是 WMT'16 多模态翻译任务 (http://www.statmt.org/wmt16/multimodal-task.html) 的训练示例。

### 0) 下载 spacy 语言模型。
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) 使用 torchtext 和 spacy 预处理数据。
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) 训练模型
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

### 3) 测试模型
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```

## [(开发中)] WMT'17 多模态翻译：德英（使用 BPE）
### 1) 使用 BPE 下载并预处理数据：

> 由于接口未统一，您需要将主函数调用从 `main_wo_bpe` 切换到 `main`。

```bash
python preprocess.py -raw_dir /tmp/raw_deen -data_dir ./bpe_deen -save_data bpe_vocab.pkl -codes codes.txt -prefix deen
```

### 2) 训练模型
```bash
python train.py -data_pkl ./bpe_deen/bpe_vocab.pkl -train_path ./bpe_deen/deen-train -val_path ./bpe_deen/deen-val -log deen_bpe -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000 -epoch 400
```

### 3) 测试模型（未完成）
- 待办：
	- 加载词汇表。
	- 翻译后执行解码。
---
# 性能
## 训练

<p align="center">
<img src="https://i.imgur.com/S2EVtJx.png" width="400">
<img src="https://i.imgur.com/IZQmUKO.png" width="400">
</p>

- 参数设置：
  - 批量大小 256 
  - 预热步数 4000 
  - 轮数 200 
  - lr_mul 0.5
  - 标签平滑 
  - 不应用 BPE 和共享词汇表
  - 目标嵌入 / softmax 前线性层权重共享。

  
## 测试
- 即将推出。
---
# 待办事项
  - 对生成文本的评估。
  - 注意力权重可视化。
---
# 致谢
- 字节对编码部分借鉴自 [subword-nmt](https://github.com/rsennrich/subword-nmt/)。
- 项目结构、一些脚本和数据集预处理步骤主要借鉴自 [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)。
- 感谢 @srush、@iamalbert、@Zessay、@JulesGM、@ZiJianZhao 和 @huanghoujing 的建议。