import os, sys
import math
import shutil
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn

torch.backends.cudnn.benchmark = True

from torch.utils.data.dataloader import DataLoader

import config

from model.network import *
from model.loss import *
from dataset.utils import to_pil_image
from dataset.voxceleb import VoxCeleb

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M_meta')

    log_path = os.path.join('log', run_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = open(os.path.join(log_path, 'log.txt'), 'w')

    print('The ID of this run: ' + run_id)
    print('Log directory: ' + log_path)

    shutil.copy('config.py', os.path.join(log_path, 'config.py'))

    # dataset
    dataset = VoxCeleb(config.train_dataset, config.input_size,
                       config.metatrain_T + 1, config.input_normalize, config.random_flip)

    dataloader = DataLoader(dataset, batch_size=config.metatrain_batch_size, shuffle=config.dataset_shuffle,
                            num_workers=config.num_worker, pin_memory=True, drop_last=False)

    video_count = len(dataset)
    batch_count = len(dataloader)
    print('Video count: %d, batch size: %d, batch count: %d' % (video_count, config.metatrain_batch_size, batch_count))

    # network
    G = Generator(config.G_config, config.embedding_dim, config.input_normalize)
    G = G.train()
    G = G.cuda()
    print('Generator:')
    print(str(G) + '\n')

    E = Embedder(config.E_config, config.embedding_dim)
    E = E.train()
    E = E.cuda()
    print('Embedder:')
    print(str(E) + '\n')

    D = Discriminator(config.V_config, video_count, config.embedding_dim)
    D = D.train()
    D = D.cuda()
    print('Discriminator:')
    print(str(D) + '\n')

    # loss
    L_EG = Loss_EG_meta(config.vgg19_layers, config.vggface_layers,
                        config.vgg19_weight_file, config.vggface_weight_file,
                        config.vgg19_loss_weight, config.vggface_loss_weight,
                        config.fm_loss_weight, config.mch_loss_weight,
                        config.input_normalize)

    L_EG = L_EG.eval()
    L_EG = L_EG.cuda()
    set_grad_enabled(L_EG, False)  # fix the two VGG networks inside
    print('Loss EG:')
    print(str(L_EG) + '\n')

    # parameters
    G_param_count = initialize_param(G)
    E_param_count = initialize_param(E)
    D_param_count = initialize_param(D)
    print('Parameter count for generator, embedder and discriminator: ')
    print('%d, %d, %d' % (G_param_count, E_param_count, D_param_count))

    # optimizer
    eg_parameters = list(E.parameters()) + list(G.parameters())
    optim_EG = optim.Adam(eg_parameters, lr=config.lr_EG, betas=config.adam_betas)
    optim_D = optim.Adam(list(D.parameters()), lr=config.lr_D, betas=config.adam_betas)

    print('EG learning rate: %.5f, D learning rate: %.5f' % (config.lr_EG, config.lr_D))
    print('VGG19 loss weight: %.4f, VGGFace loss weight: %.4f' %
          (config.vgg19_loss_weight, config.vggface_loss_weight))
    print('FM loss weight: %.1f, MCH loss weight: %.1f' % (config.fm_loss_weight, config.mch_loss_weight))

    # training loop
    print('Start training...')

    total_step = 0
    start_time = datetime.now()

    output_format_1 = 'L_VGG19: %.2f, \tL_VGGFace: %.2f, \tL_CNT: %.4f, \tL_ADV: %.4f, \tL_FM: %.4f, \tL_MCH: %.4f'
    output_format_2 = 'L_EG: %.4f,    \tL_DSC1: %.4f,    \tL_DSC2: %.4f'
    output_format_3 = 'D_SCORE: %.4f, \tD_SCORE_HAT: %.4f'

    for epoch in range(config.max_epoch):
        print('Epoch: ' + str(epoch))

        for step, (video_idx, x, y, x_t, y_t) in enumerate(dataloader):

            video_idx = video_idx.long().cuda()  # b
            x = x.cuda()  # b * T * c * h * w
            x_t = x_t.cuda()  # b * c * h * w
            y = y.cuda()
            y_t = y_t.cuda()

            optim_EG.zero_grad()
            optim_D.zero_grad()

            b, _, c, h, w = x.size()

            x = x.view(b * config.metatrain_T, c, h, w)
            y = y.view(b * config.metatrain_T, c, h, w)
            x_t = x_t.view(b, c, h, w)  # batch of single image
            y_t = y_t.view(b, c, h, w)  # batch of single image

            e_hat = E(torch.cat((x, y), dim=1))
            e_hat = e_hat.view(b, config.metatrain_T, config.embedding_dim)
            e_hat_mean = torch.mean(e_hat, dim=1, keepdim=False)  # averaging the embedding vectors
            x_hat = G(y_t, e_hat_mean)  # b * c * h * w

            # train G and D
            d_output = D(torch.cat((x_t, y_t), dim=1), video_idx)
            d_output_hat = D(torch.cat((x_hat, y_t), dim=1), video_idx)

            d_features = d_output[:-2]
            d_features_hat = d_output_hat[:-2]
            d_score = d_output[-2]
            d_score_hat = d_output_hat[-2]
            ww = d_output[-1]  # the video-specific embedding vector

            l_eg, l_vgg19, l_vggface, l_cnt, l_adv, l_fm, l_mch = \
                L_EG(x_t, x_hat, d_features, d_features_hat, d_score_hat, e_hat, ww)

            l_d = Loss_DSC(d_score_hat, d_score)
            loss = l_eg + l_d
            loss.backward()
            optim_EG.step()
            optim_D.step()

            # train D again
            optim_D.zero_grad()
            x_hat = x_hat.detach()  # do not need to train the generator

            d_output = D(torch.cat((x_t, y_t), dim=1), video_idx)
            d_output_hat = D(torch.cat((x_hat, y_t), dim=1), video_idx)

            d_score = d_output[-2]
            d_score_mean = torch.mean(d_score)
            d_score_hat = d_output_hat[-2]
            d_score_hat_mean = torch.mean(d_score_hat)

            l_d2 = Loss_DSC(d_score_hat, d_score)
            l_d2.backward()
            optim_D.step()

            # display
            if total_step and total_step % config.display_every == 0:
                end_time = datetime.now()
                duration = (end_time - start_time).seconds
                duration_step = 1.0 * duration / config.display_every

                start_time = datetime.now()

                print('Epoch: %d, total step: %d, progress: %d/%d, time: %.2fs' %
                      (epoch, total_step, step, len(dataloader), duration_step))

                print(output_format_1 % (l_vgg19.data.cpu().item(), l_vggface.data.cpu().item(),
                                         l_cnt.data.cpu().item(), l_adv.data.cpu().item(),
                                         l_fm.data.cpu().item(), l_mch.data.cpu().item()))

                print(output_format_2 % (l_eg.data.cpu().item(), l_d.data.cpu().item(), l_d2.data.cpu().item()))
                print(output_format_3 % (d_score_mean.data.cpu().item(), d_score_hat_mean.data.cpu().item()))

            # checkpoint
            if total_step and total_step % config.check_every == 0:
                save_path = os.path.join(log_path, 'check_%d' % total_step)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for i in range(config.metatrain_batch_size):
                    # save for each image in the batch
                    to_pil_image(x_hat[i, ...].data.cpu(), config.input_normalize)\
                        .save(os.path.join(save_path, 'x_hat_%d.jpg' % i))
                    to_pil_image(x_t[i, ...].data.cpu(), config.input_normalize)\
                        .save(os.path.join(save_path, 'x_t_%d.jpg' % i))
                    to_pil_image(y_t[i, ...].data.cpu(), config.input_normalize)\
                        .save(os.path.join(save_path, 'y_t_%d.jpg' % i))

                # periodically call this to boost training
                torch.cuda.empty_cache()

            # save model
            if total_step and total_step % config.save_every == 0:
                save_filename = os.path.join(log_path, 'save_%d.pkl' % total_step)
                save_data = [epoch, step, total_step, G.state_dict(), E.state_dict(), D.state_dict()]

                if config.save_optim_state:
                    save_data.extend([optim_EG.state_dict(), optim_D.state_dict()])

                torch.save(save_data, save_filename)

            # write log
            all_losses = (epoch, step, total_step,
                          l_vgg19.data.cpu().item(), l_vggface.data.cpu().item(), l_cnt.data.cpu().item(),
                          l_adv.data.cpu().item(), l_fm.data.cpu().item(), l_mch.data.cpu().item(),
                          l_eg.data.cpu().item(), l_d.data.cpu().item(), l_d2.data.cpu().item(),
                          d_score_mean.data.cpu().item(), d_score_hat_mean.data.cpu().item())

            log_text = '%d, %d, %d, %.2f, %.2f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n' % all_losses
            log_file.write(log_text)
            log_file.flush()

            total_step += 1

    log_file.close()


if __name__ == '__main__':
    main()
