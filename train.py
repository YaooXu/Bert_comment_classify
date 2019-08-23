import torch
import logging
from utils import AverageMeter, calculate_accuracy
from tqdm import tqdm
import time
import os

logging.basicConfig(level=logging.INFO,
                    # filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def train_epoch(args, device, epoch, model, optimizer, data_loader,
                n_gpu):
    logger.info('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    st = args.begin_it if epoch == args.begin_epoch else 0

    end_time = time.time()
    for i, batch in enumerate(data_loader):
        # 找到上次中断处....不优雅
        if i < st:
            continue

        data_time.update(time.time() - end_time)

        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        logits, loss = model(input_ids, segment_ids, input_mask, label_ids)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        acc = calculate_accuracy(logits, label_ids)
        accuracies.update(acc, input_ids.size(0))
        losses.update(loss.item(), input_ids.size(0))

        if (i + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(args.global_step / args.num_train_steps,
                                                                  args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            args.global_step += 1

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies))

        if args.global_step % args.checkpoint == 0:
            save_file_path = os.path.join(args.model_save_dir,
                                          'save_{}_{}.pth'.format(epoch, i + 1))
            states = {
                'epoch': epoch,
                'iteration': i + 1,  # 下次直接从[ (i + 1) * batch:]开始
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
            logger.info('Model:save_{}_{}.pth has been saved'.format(epoch, i + 1))
