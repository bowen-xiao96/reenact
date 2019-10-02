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
from torch.utils.tensorboard import SummaryWriter

import config

import model.model1 as model1
from dataset.human36m.human36m import Human36m
from dataset.human36m.dataset_utils import to_pil_image

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


def main():
    run_id = datetime.now().strftime('%Y%m%d_%H%M_meta')

    log_path = os.path.join('log', run_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # tensorboard
    sw = SummaryWriter(os.path.join(log_path, 'tensorboard'))

    print('The ID of this run: ' + run_id)
    print('Log directory: ' + log_path)

    # make backup of config files
    shutil.copy('config.py', os.path.join(log_path, 'config.py'))
    shutil.copy(os.path.join('model', 'model1.py'), os.path.join(log_path, 'model1.py'))

    # dataset
    dataset = Human36m(config.train_dataset, model1.input_modality, config.metatrain_T + 1,
                       model1.input_size, model1.input_normalize, random_flip=config.random_flip,
                       random_crop=config.random_crop)

    dataloader = DataLoader(dataset, batch_size=config.metatrain_batch_size, shuffle=config.dataset_shuffle,
                            num_workers=config.num_worker, pin_memory=True, drop_last=False)

    video_count = len(dataset)
    batch_count = len(dataloader)
    print('Video count: %d, batch size: %d, batch count: %d' %
          (video_count, config.metatrain_batch_size, batch_count))

    # network
    G = model1.Generator()
    G = G.train()
    G = G.cuda()
    print('Generator:')
    print(str(G) + '\n')

    E = model1.Embedder()
    E = E.train()
    E = E.cuda()
    print('Embedder:')
    print(str(E) + '\n')

    D = model1.Discriminator(video_count)
    D = D.train()
    D = D.cuda()
    print('Discriminator:')
    print(str(D) + '\n')

    # loss
    L_EG = model1.Loss_EG()
    L_EG = L_EG.eval()
    L_EG = L_EG.cuda()
    print('Loss_EG:')
    print(str(L_EG) + '\n')

    L_DSC = model1.Loss_DSC()
    L_DSC = L_DSC.eval()
    L_DSC = L_DSC.cuda()
    print('Loss_DSC:')
    print(str(L_DSC) + '\n')

    # paramater initialization
    G_param_count, E_param_count, D_param_count = model1.initialize(G, E, D, L_EG, L_DSC)
    print('Parameter count for generator, embedder and discriminator: ')
    print('%d, %d, %d' % (G_param_count, E_param_count, D_param_count))

    # optimizer
    eg_parameters = list(E.parameters()) + list(G.parameters())
    optim_EG = optim.Adam(eg_parameters, lr=config.lr_EG, betas=config.adam_betas)
    optim_D = optim.Adam(list(D.parameters()), lr=config.lr_D, betas=config.adam_betas)

    print('EG learning rate: %.5f, D learning rate: %.5f' % (config.lr_EG, config.lr_D))
    for k, v in model1.loss_weight.items():
        print('Weight for %s: %.4f' % (k, v))

    # training loop
    print('Start training...')

    total_step = 0
    start_time = datetime.now()

    for epoch in range(config.max_epoch):
        print('Epoch: ' + str(epoch))

        for step, (video_idx, x, y, x_t, y_t) in enumerate(dataloader):

            video_idx = video_idx.long().cuda()  # b
            x = x.cuda()
            x_t = x_t.cuda()
            y = y.cuda()  # b * T * c * h * w or b * T * c * d * h * w
            y_t = y_t.cuda()  # b * c * h * w or b * c * d * h * w

            optim_EG.zero_grad()
            optim_D.zero_grad()

            e_output = E(x, y)
            g_output = G(y_t, e_output)

            # train G and D
            d_output = D(x_t, y_t, video_idx)
            d_output_hat = D(g_output, y_t, video_idx)

            loss_eg_all = L_EG(x_t, g_output, e_output, d_output, d_output_hat)
            loss_eg = loss_eg_all['eg_loss']
            loss_dsc_all = L_DSC(d_output, d_output_hat)
            loss_dsc = loss_dsc_all['dsc_loss']

            loss = loss_eg + loss_dsc
            loss.backward()
            optim_EG.step()
            optim_D.step()

            loss_record = list(loss_eg_all.items())
            for k, v in loss_dsc_all.items():
                loss_record.append((k + '_1', v.data.cpu().item()))

            for i in range(config.d_step - 1):
                optim_D.zero_grad()
                g_output = g_output.detach()  # do not need to train the generator

                d_output = D(x_t, y_t, video_idx)
                d_output_hat = D(g_output, y_t, video_idx)

                loss_dsc_all = L_DSC(d_output, d_output_hat)
                loss_dsc = loss_dsc_all['dsc_loss']

                loss_dsc.backward()
                optim_D.step()

                for k, v in loss_dsc_all.items():
                    loss_record.append(('%s_%d' % (k, i + 2), v.data.cpu().item()))

            # write to tensorboard
            for k, v in loss_record:
                sw.add_scalar(k, v, global_step=total_step)

            # display
            if total_step and total_step % config.display_every == 0:
                end_time = datetime.now()
                duration = (end_time - start_time).seconds
                duration_step = 1.0 * duration / config.display_every

                start_time = datetime.now()

                print('Epoch: %d, total step: %d, progress: %d/%d, time: %.2fs' %
                      (epoch, total_step, step, len(dataloader), duration_step))

                loss_record.sort(key=lambda x: x[0])
                for i, (k, v) in enumerate(loss_record):
                    print('%s: %.4f' % (k, v), end='')
                    if i and i % 3 == 0:
                        print('\n', end='')
                    else:
                        print('    ', end='')

            # checkpoint
            if total_step and total_step % config.check_every == 0:
                save_path = os.path.join(log_path, 'check_%d' % total_step)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for i in range(config.metatrain_batch_size):
                    sw.add_images('x_%d' % i, x[i].data.cpu(), global_step=total_step)
                    sw.add_image('x_hat_%d' % i, g_output[i].data.cpu(), global_step=total_step)
                    sw.add_image('x_t_%d' % i, x_t[i].data.cpu(), global_step=total_step)
                    sw.add_image('y_t_%d' % i, y_t[i].data.cpu(), global_step=total_step)

                    # save for each image in the batch
                    to_pil_image(g_output[i, ...].data.cpu(), model1.input_normalize) \
                        .save(os.path.join(save_path, 'x_hat_%d.jpg' % i))
                    to_pil_image(x_t[i, ...].data.cpu(), model1.input_normalize) \
                        .save(os.path.join(save_path, 'x_t_%d.jpg' % i))
                    to_pil_image(y_t[i, ...].data.cpu(), model1.input_normalize) \
                        .save(os.path.join(save_path, 'y_t_%d.jpg' % i))

                    for j in range(config.metatrain_T):
                        to_pil_image(x[i, j, ...].data.cpu(), model1.input_normalize) \
                            .save(os.path.join(save_path, 'x_%d_%d.jpg' % (i, j)))

                # periodically call this to boost training
                torch.cuda.empty_cache()

            # save model
            if total_step and total_step % config.save_every == 0:
                save_filename = os.path.join(log_path, 'save_%d.pkl' % total_step)
                save_data = [epoch, step, total_step, G.state_dict(), E.state_dict(), D.state_dict()]

                if config.save_optim_state:
                    save_data.extend([optim_EG.state_dict(), optim_D.state_dict()])

                torch.save(save_data, save_filename)

            total_step += 1

    sw.flush()


if __name__ == '__main__':
    main()
