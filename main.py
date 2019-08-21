import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from ops import parse_args
import logging
import numpy as np
import random
from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from model import generate_model
from dataset import get_labels, get_train_examples, generate_dataset, get_test_examples
from optimizer import generate_optimizer
from train import train_epoch
from validation import val_epoch
from test import test
import pandas as pd
import os

logging.basicConfig(level=logging.INFO,
                    # filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    args.output_dir = Path(args.output_dir)
    args.model_save_dir = Path(args.model_save_dir)
    args.data_dir = Path(args.data_dir)

    args.output_dir.mkdir(exist_ok=True)
    args.model_save_dir.mkdir(exist_ok=True)
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 加载模型和tokenizer
    labels = get_labels(args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = generate_model(args, len(labels), device, n_gpu)

    if args.do_train:
        train_examples = get_train_examples(args)

        if args.do_val:
            train_data, val_data = generate_dataset(args, train_examples, tokenizer, 0.8)
            val_dataloader = DataLoader(val_data, batch_size=args.predict_batch_size)
        else:
            train_data = generate_dataset(args, train_examples, tokenizer)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        # 总共的训练步数
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        # num_train_steps为并行训练的单个次数
        optimizer, num_train_steps = generate_optimizer(args, model.named_parameters(), num_train_steps)
        args.num_train_steps = num_train_steps

    if args.resume_path:
        # 从上次训练状态恢复
        logger.info('Loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path, map_location=device.type)
        args.begin_epoch = checkpoint['epoch']
        args.begin_it = checkpoint['iteration']
        model.load_state_dict(checkpoint['state_dict'])
        if args.do_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        args.global_step = 0  # 共迭代的次数
        for epoch in range(args.begin_epoch, args.num_train_epochs):
            train_epoch(args, device, epoch, model, optimizer, train_dataloader, n_gpu)
            if args.do_val:
                val_epoch(args, device, epoch, model, val_dataloader, labels)

    if args.do_test:
        test_examples = get_test_examples(args)
        guids = []
        for example in test_examples:
            guids.append(example.guid)

        test_data = generate_dataset(args, test_examples, tokenizer)
        test_dataloader = DataLoader(test_data, batch_size=args.predict_batch_size)
        predicts = test(args, device, model, test_dataloader)

        logger.info("Writing file...")
        content = {}
        content['id'] = guids
        for i, label in enumerate(labels):
            content[label] = predicts[:, i]
        dataframe = pd.DataFrame(content)
        dataframe.to_csv(os.path.join(args.output_dir, 'predicts.csv'), index=False, sep=',')
