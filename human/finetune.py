import os, sys
import shutil
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

import config

import model.model1_finetune as model1
from model.utils import set_grad_enabled
from dataset.human36m.human36m import Human36m
from dataset.human36m.dataset_utils import to_pil_image

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


def main(model_file):
    run_id = datetime.now().strftime('%Y%m%d_%H%M_finetune')

    log_path = os.path.join('log', run_id)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print('The ID of this run: ' + run_id)
    print('Log directory: ' + log_path)

    # dataset
    return_all = config.finetune_T + 1
    test_sample = config.test_episode * return_all

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

    test_video_count = len(test_dataset)
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

    D = model1.Discriminator()
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

    # model initialization
    model1.initialize(G, E, D, L_EG, L_DSC)

    save_data = torch.load(model_file)
    _, _, _, G_state_dict, E_state_dict, D_state_dict = save_data[:6]
    E.load_state_dict(E_state_dict)

    print('EG learning rate: %.5f, D learning rate: %.5f' % (config.lr_EG, config.lr_D))
    for k, v in model1.loss_weight.items():
        print('Weight for %s: %.4f' % (k, v))

    for k in range(config.test_episode):
        for _video_idx, _x, _y, _x_t, _y_t in test_dataset:
            video_name = test_dataset.all_videos[_video_idx]
            print('%d: %s' % (k, video_name))

            image_path = os.path.join(log_path, '%s_%d' % (video_name, k))
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            dataset = TensorDataset(_x, _y)
            dataloader = DataLoader(dataset, batch_size=config.finetune_batch_size, shuffle=config.dataset_shuffle,
                                    num_workers=0, pin_memory=True, drop_last=False)

            _x = _x.cuda()
            _y = _y.cuda()  # T * c * h * w
            _y_t = _y_t.cuda()  # c * h * w

            with torch.no_grad():
                e_hat = E(torch.unsqueeze(_x, dim=0), torch.unsqueeze(_y, dim=0))
                e_hat = e_hat.view(1, -1, model1.embedding_dim)
                e_hat_mean = torch.mean(e_hat, dim=1, keepdim=False)

                G_state_dict_new = G_state_dict.copy()
                P = G_state_dict_new['P.weight']
                adain = torch.matmul(e_hat_mean, torch.transpose(P, 0, 1))
                del G_state_dict_new['P.weight']
                adain = adain.view(1, -1, 2)
                assert adain.size(1) == G.adain_param_count
                G_state_dict_new['adain'] = adain.data
                G.load_state_dict(G_state_dict_new)

                D_state_dict_new = D_state_dict.copy()
                del D_state_dict_new['embedding.weight']
                w0 = D_state_dict_new['w0']
                w = w0 + e_hat_mean
                del D_state_dict_new['w0']
                D_state_dict_new['w'] = w.data
                D.load_state_dict(D_state_dict_new)

                # initial output before finetuning
                x_hat = G(torch.unsqueeze(_y_t, dim=0))
                to_pil_image(x_hat.data.cpu(), model1.input_normalize) \
                    .save(os.path.join(image_path, 'x_hat.jpg'))

            G = G.train()
            set_grad_enabled(G, True)
            D = D.train()
            set_grad_enabled(D, True)

            optim_EG = optim.Adam(G.parameters(), lr=config.lr_EG, betas=config.adam_betas)
            optim_D = optim.Adam(D.parameters(), lr=config.lr_D, betas=config.adam_betas)

            for epoch in range(config.finetune_epoch):
                for _, (xx, yy) in enumerate(dataloader):
                    xx = xx.cuda()
                    yy = yy.cuda()

                    optim_EG.zero_grad()
                    optim_D.zero_grad()

                    g_output = G(yy)

                    # train G and D
                    d_output = D(xx, yy)
                    d_output_hat = D(g_output, yy)

                    loss_eg_all = L_EG(xx, g_output, d_output, d_output_hat)
                    loss_eg = loss_eg_all['eg_loss']
                    loss_dsc_all = L_DSC(d_output, d_output_hat)
                    loss_dsc = loss_dsc_all['dsc_loss']

                    loss = loss_eg + loss_dsc
                    loss.backward()
                    optim_EG.step()
                    optim_D.step()

                    # train D for more times
                    for i in range(config.d_step - 1):
                        optim_D.zero_grad()
                        g_output = g_output.detach()  # do not need to train the generator

                        d_output = D(xx, yy)
                        d_output_hat = D(g_output, yy)

                        loss_dsc_all = L_DSC(d_output, d_output_hat)
                        loss_dsc = loss_dsc_all['dsc_loss']

                        loss_dsc.backward()
                        optim_D.step()

                with torch.no_grad():
                    x_hat_ii = G(torch.unsqueeze(_y_t, dim=0))
                    to_pil_image(x_hat_ii.data.cpu(), model1.input_normalize) \
                        .save(os.path.join(image_path, 'x_hat_%d.jpg' % epoch))

            # save image
            to_pil_image(_x_t.data.cpu(), model1.input_normalize) \
                .save(os.path.join(image_path, 'x_t.jpg'))
            to_pil_image(_y_t.data.cpu(), model1.input_normalize) \
                .save(os.path.join(image_path, 'y_t.jpg'))

            for j in range(config.finetune_T):
                to_pil_image(_x[j].data.cpu(), model1.input_normalize) \
                    .save(os.path.join(image_path, 'x_%d.jpg' % j))


if __name__ == '__main__':
    assert len(sys.argv) > 1

    model_file = sys.argv[1].strip()
    main(model_file)
