import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg_model.vgg_network import *


# discriminator
def Loss_DSC(D_score_hat, D_score):
    loss = F.relu(1.0 - D_score)
    loss_hat = F.relu(1.0 + D_score_hat)

    loss_all = torch.mean(loss + loss_hat)  # average over batch size
    return loss_all


# embedder and generator
class Loss_EG_meta(nn.Module):
    def __init__(self, vgg19_layers, vggface_layers, vgg19_weight_file, vggface_weight_file,
                 vgg19_loss_weight, vggface_loss_weight, fm_loss_weight, mch_loss_weight, normalize):

        super(Loss_EG_meta, self).__init__()

        self.vgg19_loss_weight = vgg19_loss_weight
        self.vggface_loss_weight = vggface_loss_weight
        self.fm_loss_weight = fm_loss_weight
        self.mch_loss_weight = mch_loss_weight
        self.normalize = normalize

        # VGG19 model
        vgg19_model = VGGNetwork('vgg19', vgg19_config, 1000)
        vgg19_model.load_state_dict(torch.load(vgg19_weight_file))
        self.vgg19_activation = VGGActivation(vgg19_model, vgg19_layers, normalize)

        # VGGFace model
        vggface_model = VGGNetwork('vgg16', vgg16_config, 2622)
        vggface_model.load_state_dict(torch.load(vggface_weight_file))
        self.vggface_activation = VGGActivation(vggface_model, vggface_layers, normalize)

    def forward(self, x, x_hat,  # content loss
                D_features, D_features_hat,  # feature matching loss
                D_score_hat,  # adversarial loss
                e_hat, w):  # matching loss

        # content loss of VGG19
        vgg19_feature = self.vgg19_activation(x)
        vgg19_feature_hat = self.vgg19_activation(x_hat)
        vgg19_loss = list()

        for x_feat, x_hat_feat in zip(vgg19_feature, vgg19_feature_hat):
            # reduced over both batch size and feature dimension
            vgg19_loss.append(F.l1_loss(x_feat, x_hat_feat))

        vgg19_loss = torch.sum(torch.stack(vgg19_loss))

        # content loss of VGGFace
        vggface_feature = self.vggface_activation(x)
        vggface_feature_hat = self.vggface_activation(x_hat)
        vggface_loss = list()

        for x_feat, x_hat_feat in zip(vggface_feature, vggface_feature_hat):
            vggface_loss.append(F.l1_loss(x_feat, x_hat_feat))

        vggface_loss = torch.sum(torch.stack(vggface_loss))

        content_loss = vgg19_loss * self.vgg19_loss_weight + vggface_loss * self.vggface_loss_weight

        # adversarial loss
        adv_loss = -torch.mean(D_score_hat)

        # feature matching loss
        fm_loss = list()

        for D_feature, D_feature_hat in zip(D_features, D_features_hat):
            fm_loss.append(F.l1_loss(D_feature, D_feature_hat))

        fm_loss = torch.sum(torch.stack(fm_loss))

        # matching loss
        w = torch.unsqueeze(w, 1).expand(e_hat.size())
        mch_loss = F.l1_loss(w, e_hat)

        total_loss = content_loss + adv_loss + fm_loss * self.fm_loss_weight + mch_loss * self.mch_loss_weight

        return total_loss, vgg19_loss, vggface_loss, content_loss, adv_loss, fm_loss, mch_loss  # for debug


class Loss_EG_finetune(nn.Module):
    def __init__(self, vgg19_layers, vggface_layers, vgg19_weight_file, vggface_weight_file,
                 vgg19_loss_weight, vggface_loss_weight, fm_loss_weight, normalize):

        super(Loss_EG_finetune, self).__init__()

        self.vgg19_loss_weight = vgg19_loss_weight
        self.vggface_loss_weight = vggface_loss_weight
        self.fm_loss_weight = fm_loss_weight
        self.normalize = normalize

        # VGG19 model
        vgg19_model = VGGNetwork('vgg19', vgg19_config, 1000)
        vgg19_model.load_state_dict(torch.load(vgg19_weight_file))
        self.vgg19_activation = VGGActivation(vgg19_model, vgg19_layers, normalize)

        # VGGFace model
        vggface_model = VGGNetwork('vgg16', vgg16_config, 2622)
        vggface_model.load_state_dict(torch.load(vggface_weight_file))
        self.vggface_activation = VGGActivation(vggface_model, vggface_layers, normalize)

    def forward(self, x, x_hat,  # content loss
                D_features, D_features_hat,  # feature matching loss
                D_score_hat):  # adversarial loss

        # content loss of VGG19
        vgg19_feature = self.vgg19_activation(x)
        vgg19_feature_hat = self.vgg19_activation(x_hat)
        vgg19_loss = list()

        for x_feat, x_hat_feat in zip(vgg19_feature, vgg19_feature_hat):
            # reduced over both batch size and feature dimension
            vgg19_loss.append(F.l1_loss(x_feat, x_hat_feat))

        vgg19_loss = torch.sum(torch.stack(vgg19_loss))

        # content loss of VGGFace
        vggface_feature = self.vggface_activation(x)
        vggface_feature_hat = self.vggface_activation(x_hat)
        vggface_loss = list()

        for x_feat, x_hat_feat in zip(vggface_feature, vggface_feature_hat):
            vggface_loss.append(F.l1_loss(x_feat, x_hat_feat))

        vggface_loss = torch.sum(torch.stack(vggface_loss))

        content_loss = vgg19_loss * self.vgg19_loss_weight + vggface_loss * self.vggface_loss_weight

        # adversarial loss
        adv_loss = -torch.mean(D_score_hat)

        # feature matching loss
        fm_loss = list()

        for D_feature, D_feature_hat in zip(D_features, D_features_hat):
            fm_loss.append(F.l1_loss(D_feature, D_feature_hat))

        fm_loss = torch.sum(torch.stack(fm_loss))

        total_loss = content_loss + adv_loss + fm_loss * self.fm_loss_weight

        return total_loss, vgg19_loss, vggface_loss, content_loss, adv_loss, fm_loss  # for debug
