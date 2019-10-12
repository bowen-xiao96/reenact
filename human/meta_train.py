import os, sys
import shutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader
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
    shutil.copy('meta_train.py', os.path.join(log_path, 'meta_train.py'))
    shutil.copy(os.path.join('model', 'model1.py'), os.path.join(log_path, 'model1.py'))

    # dataset
    return_all = config.metatrain_T + 1
    test_sample = config.test_episode * return_all

    train_dataset = Human36m(
        config.train_dataset,
        model1.input_modality,
        return_all,
        model1.input_size,
        model1.input_normalize,
        True,
        None,
        random_flip=config.random_flip,
        random_crop=config.random_crop
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.metatrain_batch_size,
        shuffle=config.dataset_shuffle,
        num_workers=config.num_worker,
        pin_memory=True,
        drop_last=False
    )

    test_dataset = Human36m(
        config.test_dataset,
        model1.input_modality,
        return_all,
        model1.input_size,
        model1.input_normalize,
        False,
        test_sample,  # sample count
        extend_ratio=0.1,
        random_flip=False,
        random_crop=False
    )

    train_video_count = len(train_dataset)
    train_batch_count = len(train_dataloader)
    test_video_count = len(test_dataset)
    print('Training video count: %d, batch size: %d, batch count: %d' %
          (train_video_count, config.metatrain_batch_size, train_batch_count))
    print('Testing video count: %d' % test_video_count)

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

    D = model1.Discriminator(train_video_count)
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
    d_parameters = D.parameters()
    eg_parameters = list(E.parameters()) + list(G.parameters())
    optim_D = optim.Adam(d_parameters, lr=config.lr_D, betas=config.adam_betas)
    optim_EG = optim.Adam(eg_parameters, lr=config.lr_EG, betas=config.adam_betas)

    print('EG learning rate: %.5f, D learning rate: %.5f' % (config.lr_EG, config.lr_D))
    for k, v in model1.loss_weight.items():
        print('Weight for %s: %.4f' % (k, v))

    # training loop
    print('Start training...')

    total_step = 0
    start_time = datetime.now()

    for epoch in range(config.max_epoch):
        print('Epoch: ' + str(epoch))

        for step, (video_idx, x, y, x_t, y_t) in enumerate(train_dataloader):
            G = G.train()
            E = E.train()

            video_idx = video_idx.long().cuda()  # b
            x = x.cuda()
            x_t = x_t.cuda()
            y = y.cuda()  # b * T * c * h * w
            y_t = y_t.cuda()  # b * c * h * w

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

            # train D for more times
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
                duration_step = float(duration) / config.display_every

                start_time = datetime.now()

                print('Epoch: %d, total step: %d, progress: %d/%d, time: %.3fs' %
                      (epoch, total_step, step, len(train_dataloader), duration_step))

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
                    image_path = os.path.join(save_path, str(i))
                    if not os.path.exists(image_path):
                        os.makedirs(image_path)

                    # save each image in the batch
                    to_pil_image(g_output[i, ...].data.cpu(), model1.input_normalize) \
                        .save(os.path.join(image_path, 'x_hat.jpg'))
                    to_pil_image(x_t[i, ...].data.cpu(), model1.input_normalize) \
                        .save(os.path.join(image_path, 'x_t.jpg'))
                    to_pil_image(y_t[i, ...].data.cpu(), model1.input_normalize) \
                        .save(os.path.join(image_path, 'y_t.jpg'))

                    for j in range(config.metatrain_T):
                        to_pil_image(x[i, j, ...].data.cpu(), model1.input_normalize) \
                            .save(os.path.join(image_path, 'x_%d.jpg' % j))

                sw.add_images('g_output', g_output.data.cpu(), global_step=total_step)

                # periodically call this to boost training
                torch.cuda.empty_cache()

            # save model
            if total_step and total_step % config.save_every == 0:
                save_path = os.path.join(log_path, 'save_%d' % total_step)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_filename = os.path.join(save_path, 'step_%d.pkl' % total_step)
                save_data = [epoch, step, total_step, G.state_dict(), E.state_dict(), D.state_dict()]

                if config.save_optim_state:
                    save_data.extend([optim_EG.state_dict(), optim_D.state_dict()])

                torch.save(save_data, save_filename)

                # run on test set
                G = G.eval()
                E = E.eval()
                torch.set_grad_enabled(False)

                for k in range(config.test_episode):
                    for _video_idx, _x, _y, _x_t, _y_t in test_dataset:
                        video_name = test_dataset.all_videos[_video_idx]
                        image_path = os.path.join(save_path, '%s_%d' % (video_name, k))
                        if not os.path.exists(image_path):
                            os.makedirs(image_path)

                        _x = _x.cuda()
                        _y = _y.cuda()  # T * c * h * w
                        _y_t = _y_t.cuda()  # c * h * w

                        # feed into network
                        _e_output = E(torch.unsqueeze(_x, dim=0), torch.unsqueeze(_y, dim=0))
                        _g_output = G(torch.unsqueeze(_y_t, dim=0), _e_output)

                        _g_output = torch.squeeze(_g_output, dim=0)

                        # save each image in the batch
                        to_pil_image(_g_output.data.cpu(), model1.input_normalize) \
                            .save(os.path.join(image_path, 'x_hat.jpg'))
                        to_pil_image(_x_t.data.cpu(), model1.input_normalize) \
                            .save(os.path.join(image_path, 'x_t.jpg'))
                        to_pil_image(_y_t.data.cpu(), model1.input_normalize) \
                            .save(os.path.join(image_path, 'y_t.jpg'))

                        for j in range(config.metatrain_T):
                            to_pil_image(_x[j, ...].data.cpu(), model1.input_normalize) \
                                .save(os.path.join(image_path, 'x_%d.jpg' % j))

                torch.set_grad_enabled(True)

            total_step += 1

    sw.close()


if __name__ == '__main__':
    main()
