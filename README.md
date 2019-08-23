# Bert_comment_classify
## 概要
使用bert的pytorch版本对英文评论分成如下六类

toxic, severe_toxic, obscene, threat, insult,identity_hate

## 数据来源
数据默认路径是./data, 来源于Kaggle的一次比赛,链接:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview

## 环境

- pytorch 1.1.0
- pytorch-pretrained-bert   0.6.2

以及bert-base-uncased预训练模型


## TODOS
- 模型改进
- 从上次的iteration处继续训练

## 使用方式
1. 训练

```
python --train_size -1 --val_ratio 0.5 --do_train --do_val --max_seq_length 256 --checkpoint 3 --num_train_epochs 3
```

2. 预测

```
python --resume_path ./cache/save_1_2.pth --do_eval --predict_batch_size 8 --max_seq_length 256
```

其他未列举的参数均采用默认值