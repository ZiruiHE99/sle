import numpy as np
import torch
import math

device = torch.device("cuda")

def flat_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(preds_flat == labels_flat) / len(labels_flat)

def map_to_lower(x,p1,p2):
    if x < p1:
        return 0  
    else:
        return math.log((x+0.2),p2)

def smooth_label(y_true,y_bias):
    bias_scores = []
    for i in range(len(y_true)):
        num = torch.max(y_bias[i])
        des_score = map_to_lower(num.item(),0.88,1.48)
        ind = torch.argmax(y_true[i]).item()
        y_true[i][ind] = y_true[i][ind] - des_score
        bias_scores.append(des_score)
    bias_scores = torch.tensor(bias_scores)
    bias_scores = bias_scores.unsqueeze(1)
    bias_scores = bias_scores.to(device)
    smoothed_label = torch.concat([y_true,bias_scores],dim=1)
    return smoothed_label

