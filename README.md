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
python --do_train [--do_val] --train_batch_size 8 --learning_rate 3e-5 --num_train_epochs --max_seq_length 256
```

2. 预测

```
python --do_eval --predict_batch_size 8 --max_seq_length 256
```

其他未列举的参数均采用默认值