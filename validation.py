import torch
import logging
from utils import AverageMeter, calculate_accuracy
from tqdm import tqdm
import time
import os
from sklearn.metrics import roc_curve, auc
import numpy as np

logging.basicConfig(level=logging.INFO,
                    # filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def val_epoch(args, device, epoch, model, val_dataloader, labels):
    logger.info('train at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    all_logits = None
    all_labels = None

    end_time = time.time()
    for batch in tqdm(val_dataloader, desc='VAL'):
        data_time.update(time.time() - end_time)

        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            logits, loss = model(input_ids, segment_ids, input_mask, label_ids)

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

        acc = calculate_accuracy(logits, label_ids)
        accuracies.update(acc, input_ids.size(0))
        losses.update(loss.item(), input_ids.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    # logger.info({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    #     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    FPR = dict()
    TPR = dict()
    roc_auc = dict()

    for i in range(len(labels)):
        FPR[i], TPR[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(FPR[i], TPR[i])

    # Compute micro-average ROC curve and ROC area
    FPR["micro"], TPR["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(FPR["micro"], TPR["micro"])

    result = {'eval_loss': losses.avg,
              'eval_accuracy': accuracies.avg,
              #               'loss': tr_loss/nb_tr_steps,
              'roc_auc': roc_auc}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))
    return result
