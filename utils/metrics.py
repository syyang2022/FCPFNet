import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_mpa(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    #area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred

    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    area_FN = area_pred - area_inter
    area_FP = area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy(), area_FN.cpu().numpy(), area_FP.cpu().numpy()
    #return area_inter.cpu().numpy(), area_union.cpu().numpy()


def compute_target(target):
    target = target.view(-1)
    #SUM = 0
    # Compute the confusion matrix
    #for t in target:
    #    SUM += 1
    #print(len(target))
    return len(target)

def eval_metrics(output, target, num_class):
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    inter, union, FN, FP = batch_intersection_union(predict, target, num_class, labeled)
    #inter, union = batch_intersection_union(predict, target, num_class, labeled)
    a, b = batch_mpa(predict, target, num_class, labeled)
    SUM = compute_target(target)
    # Compute confusion matrix
    #confusion_matrix = compute_confusion_matrix(predict, target, num_class)
    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5), np.round(a, 5), np.round(b, 5), np.round(FN, 5), np.round(FP, 5), np.round(SUM, 5)]
    #return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5), np.round(union, 5), np.round(a, 5), np.round(b, 5)]