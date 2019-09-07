import os, sys
import math
import random
from datetime import datetime, timedelta
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

from model.network_finetune import *
from model.loss import *
from dataset.utils import to_pil_image, to_tensor
from dataset.landmark import get_detector, extract_landmark, plot_landmarks

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)


def sample_frames(person, k):
    # sample k frames from one video of the specified person
    (video, ) = random.sample(os.listdir(os.path.join(config.test_dataset, person)), 1)
    video_root = os.path.join(config.test_dataset, person, video)

    all_frames = list()
    for clip in os.listdir(video_root):
        clip_root = os.path.join(video_root, clip)
        if os.path.isdir(clip_root):
            all_frames.extend([os.path.join(clip_root, f) for f in os.listdir(clip_root)])

    selected_frames = random.sample(all_frames, k)
    return selected_frames


def main(model_file):
    run_id = datetime.now().strftime('%Y%m%d_%H%M_finetune')

    output_path = os.path.join('output', run_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('The ID of this run: ' + run_id)
    print('Output directory: ' + output_path)

    all_people = os.listdir(config.test_dataset)
    people_count = len(all_people)
    print('People count: %d' % people_count)

    for i, person in enumerate(all_people):
        print('Progress: %d/%d' % (i, people_count))

        # T training images should come from the same video
        xx = sample_frames(person, config.finetune_T)

        person_t = person
        while person_t == person:
            (person_t, ) = random.sample(all_people, 1)

        xx_t = sample_frames(person_t, 1)

        xx_all = xx_t + xx

        x = list()
        y = list()

        detector = get_detector('cuda')

        for filename in xx_all:
            img = Image.open(filename).convert('RGB')
            img = img.resize((config.input_size, config.input_size), Image.LANCZOS)
            x.append(to_tensor(img, config.input_normalize))

            arr = np.array(img)
            landmarks = extract_landmark(detector, arr)

            rendered = plot_landmarks(config.input_size, landmarks)
            y.append(to_tensor(rendered, config.input_normalize))

        del detector
        torch.set_grad_enabled(True)

        x_t = torch.unsqueeze(x[0], dim=0)
        y_t = torch.unsqueeze(y[0], dim=0)
        y_t = y_t.cuda()

        x = torch.stack(x[1:])  # n * c * h * w
        y = torch.stack(y[1:])

        # sanity check
        assert x.size(0) == config.finetune_T

        # load models
        save_data = torch.load(model_file)
        _, _, _, G_state_dict, E_state_dict, D_state_dict = save_data[:6]

        G = Generator(config.G_config, config.input_normalize)
        G = G.eval()
        G = G.cuda()

        E = Embedder(config.E_config, config.embedding_dim)
        E = E.eval()
        E = E.cuda()

        D = Discriminator(config.V_config, config.embedding_dim)
        D = D.eval()
        D = D.cuda()

        with torch.no_grad():
            E.load_state_dict(E_state_dict)
            E_input = torch.cat((x, y), dim=1)
            E_input = E_input.cuda()
            e_hat = E(E_input)
            e_hat = e_hat.view(1, -1, config.embedding_dim)
            e_hat_mean = torch.mean(e_hat, dim=1, keepdim=False)
            del E

            P = G_state_dict['P.weight']
            adain = torch.matmul(e_hat_mean, torch.transpose(P, 0, 1))
            del G_state_dict['P.weight']
            adain = adain.view(1, -1, 2)
            assert adain.size(1) == G.adain_param_count
            G_state_dict['adain'] = adain.data
            G.load_state_dict(G_state_dict)

            del D_state_dict['embedding.weight']
            w0 = D_state_dict['w0']
            w = w0 + e_hat_mean
            del D_state_dict['w0']
            D_state_dict['w'] = w.data
            D.load_state_dict(D_state_dict)

            x_hat_0 = G(y_t)
            x_hat_0_img = to_pil_image(x_hat_0, config.input_normalize)
            del x_hat_0

        G = G.train()
        set_grad_enabled(G, True)
        D = D.train()
        set_grad_enabled(D, True)

        # loss
        L_EG = Loss_EG_finetune(config.vgg19_layers, config.vggface_layers,
                                config.vgg19_weight_file, config.vggface_weight_file,
                                config.vgg19_loss_weight, config.vggface_loss_weight,
                                config.fm_loss_weight, config.input_normalize)

        L_EG = L_EG.eval()
        L_EG = L_EG.cuda()
        set_grad_enabled(L_EG, False)

        optim_EG = optim.Adam(G.parameters(), lr=config.lr_EG, betas=config.adam_betas)
        optim_D = optim.Adam(D.parameters(), lr=config.lr_D, betas=config.adam_betas)

        # dataset
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=config.finetune_batch_size, shuffle=config.dataset_shuffle,
                                num_workers=config.num_worker, pin_memory=True, drop_last=False)

        # finetune
        for epoch in range(config.finetune_epoch):
            for _, (xx, yy) in enumerate(dataloader):
                xx = xx.cuda()
                yy = yy.cuda()

                optim_EG.zero_grad()
                optim_D.zero_grad()

                x_hat = G(yy)

                d_output = D(torch.cat((xx, yy), dim=1))
                d_output_hat = D(torch.cat((x_hat, yy), dim=1))

                d_features = d_output[:-1]
                d_features_hat = d_output_hat[:-1]
                d_score = d_output[-1]
                d_score_hat = d_output_hat[-1]

                l_eg, l_vgg19, l_vggface, l_cnt, l_adv, l_fm = \
                    L_EG(xx, x_hat, d_features, d_features_hat, d_score_hat)

                l_d = Loss_DSC(d_score_hat, d_score)
                loss = l_eg + l_d
                loss.backward()
                optim_EG.step()
                optim_D.step()

                # train D again
                optim_D.zero_grad()
                x_hat = x_hat.detach()  # do not need to train the generator

                d_output = D(torch.cat((xx, yy), dim=1))
                d_output_hat = D(torch.cat((x_hat, yy), dim=1))

                d_score = d_output[-1]
                d_score_hat = d_output_hat[-1]

                l_d2 = Loss_DSC(d_score_hat, d_score)
                l_d2.backward()
                optim_D.step()

        # after finetuning
        with torch.no_grad():
            x_hat_1 = G(y_t)
            x_hat_1_img = to_pil_image(x_hat_1, config.input_normalize)
            del x_hat_1

        # save image
        training_img = Image.new('RGB', (config.finetune_T * config.input_size, config.input_size))
        for j in range(config.metatrain_T):
            img = to_pil_image(x[j], config.input_normalize)
            training_img.paste(img, (j * config.input_size, 0))

        training_img.save(os.path.join(output_path, 't_%d.jpg' % i))

        x_t_img = to_pil_image(x_t, config.input_normalize)
        y_t_img = to_pil_image(y_t, config.input_normalize)

        output_img = Image.new('RGB', (4 * config.input_size, config.input_size))
        output_img.paste(x_hat_0_img, (0, 0))
        output_img.paste(x_hat_1_img, (config.input_size, 0))
        output_img.paste(x_t_img, (2 * config.input_size, 0))
        output_img.paste(y_t_img, (3 * config.input_size, 0))

        output_img.save(os.path.join(output_path, 'o_%d.jpg' % i))


if __name__ == '__main__':
    assert len(sys.argv) > 1

    model_file = sys.argv[1].strip()
    main(model_file)
