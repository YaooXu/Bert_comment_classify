import torch
import logging
from utils import AverageMeter, calculate_accuracy
from tqdm import tqdm
import time
import os
from sklearn.metrics import roc_curve, auc
import numpy as np


def test(args, device, model, dataloader):
    all_logits = None
    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, _ = batch
        logits = model(input_ids, segment_ids, input_mask).sigmoid()
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
    return all_logits
