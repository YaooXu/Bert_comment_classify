import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model",
                        default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    # Other parameters
    parser.add_argument("--output_dir",
                        default='./output', type=str,
                        help="The output directory where the predictions will be written.")
    parser.add_argument("--model_save_dir",
                        default='./cache', type=str,
                        help="The output directory where the model will be written.")
    parser.add_argument("--data_dir",
                        default='./data', type=str,
                        help="path of data directory")
    parser.add_argument("--pretrain_path",
                        default=None, type=str,
                        help="path of pretrian model (./chche/xxx.bin)")
    parser.add_argument("--evl_freq",
                        default=2000, type=int,
                        help="The frequency of evaluate dev set")
    parser.add_argument("--do_train",
                        default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_val",
                        default=False, action='store_true', help="Whether to run eval on the val set.")
    parser.add_argument("--do_test",
                        default=False, action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size",
                        default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size",
                        default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--train_size",
                        default=-1, type=int,
                        help="How much train examples to read")
    parser.add_argument("--val_size",
                        default=-1, type=int,
                        help="How much validation examples to read")
    parser.add_argument("--test_size",
                        default=-1, type=int,
                        help="How much test examples to read")
    parser.add_argument("--learning_rate",
                        default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--max_seq_length",
                        default=512, type=int,
                        help="The maximum length of sequence")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--checkpoint',
                        default=1000,
                        type=int,
                        help='Trained model is saved at every this iteration.')
    parser.add_argument("--begin_epoch",
                        default=0, type=int,
                        help="Record the epochs of last train")
    parser.add_argument("--resume_path",
                        default=None, type=str,
                        help="Where to continue")
    parser.add_argument("--begin_it",
                        default=0, type=int,
                        help="Record the iteration of last train")
    args = parser.parse_args()
    return args
