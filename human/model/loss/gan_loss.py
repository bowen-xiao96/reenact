import torch
import torch.nn as nn
import torch.nn.functional as F


# discriminator hinge loss
def loss_dsc_hinge(d_score_hat, d_score):
    loss = F.relu(1.0 - d_score)
    loss_hat = F.relu(1.0 + d_score_hat)

    loss_all = torch.mean(loss + loss_hat)  # average over batch size
    return loss_all


# feature matching loss
def loss_fm(features, features_hat):
    loss_all = list()

    for feat, feat_hat in zip(features, features_hat):
        loss_all.append(F.l1_loss(feat, feat_hat))

    loss_all = torch.sum(torch.stack(loss_all))
    return loss_all


# adversarial loss
def loss_adv(d_score_hat):
    return -torch.mean(d_score_hat)
