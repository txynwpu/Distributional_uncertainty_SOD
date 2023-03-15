import os
import numpy as np
import torch
import pandas as pd
from copy import deepcopy

def IoU(mask1, mask2):
    mask1[mask1 >= 0.5] = 1
    mask1[mask1 < 0.5] = 0
    mask2[mask2 >= 0.5] = 1
    mask2[mask2 < 0.5] = 0

    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


def accuracy(mask1, mask2):
    mask1[mask1 >= 0.5] = 1
    mask1[mask1 < 0.5] = 0
    mask2[mask2 >= 0.5] = 1
    mask2[mask2 < 0.5] = 0

    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    F = F_score(precisions, recalls, betta_sq)
    return F.mean(dim=0).max().item()


def get_curve(known, novel):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    # if method == 'row':
    #     threshold = -0.5
    # else:
    threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


class MyDict(dict):
    def __str__(self):
        return str(pd.Series(self))


class Metrics:
    def __init__(self):
        self.precisions = []
        self.recalls = []
        self.ious = []
        self.accs = []
        self.fpr95 = []
        self.auroc = []
        self.prob_bins = 255

    def update(self, pred, uncert, gt):
        # iou
        self.ious.append(IoU(gt, pred))

        # acc
        self.accs.append(accuracy(gt, pred))
        
        # precision, recall
        p,r = [], []
        splits = 2.0 * pred.mean(dim=0) if self.prob_bins is None else np.arange(0.0, 1.0, 1.0 / self.prob_bins)
        for split in splits:
            pr = precision_recall(gt, pred >= split)
            p.append(pr[0])
            r.append(pr[1])
        self.precisions.append(p)
        self.recalls.append(r)

        # uncertainty
        binary_pred = deepcopy(pred)
        binary_pred[binary_pred >= 0.5] = 1
        binary_pred[binary_pred < 0.5] = 0

        right = binary_pred == gt
        wrong = ~right

        right_uncert = uncert[right].cpu().numpy()
        wrong_uncert = uncert[wrong].cpu().numpy()

        # tp, fp, fpr_at_tpr95 = get_curve(right_uncert, wrong_uncert) # wrong order
        tp, fp, fpr_at_tpr95 = get_curve(wrong_uncert, right_uncert)
        self.fpr95.append(fpr_at_tpr95)

        tpr = np.concatenate([[1.], tp/tp[0], [0.]])
        fpr = np.concatenate([[1.], fp/fp[0], [0.]])
        auroc = -np.trapz(1.-fpr, tpr)
        self.auroc.append(auroc)

    
    def get_results(self):
        result_dict = {
            'F_max': F_max(torch.tensor(self.precisions), torch.tensor(self.recalls)),
            'IoU': np.mean(self.ious),
            'Accuracy': np.mean(self.accs),
            'FPR95': np.mean(self.fpr95),
            'AUROC': np.mean(self.auroc)
            }
        return MyDict(result_dict)
