import numpy as np
import torch
import os
import argparse
import random

from ltsm.data_provider.data_factory import get_data_loader, get_data_loaders
from ltsm.training import train
from ltsm.testing import test

from transformers import TrainingArguments


def get_args():
    parser = argparse.ArgumentParser(description='LTSM')

    parser.add_argument('--model_id', type=str, required=True, default='test_run')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--data_path', type=str, default='dataset/weather.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default="h")
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--percent', type=int, default=10)

    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)

    parser.add_argument('--decay_fac', type=float, default=0.75)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=25)

    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=-1)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--tmax', type=int, default=10)

    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--cos', type=int, default=0)

    args = parser.parse_args()

    return args


def seed_all(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)


def run(config):
    mses = []
    maes = []
    training_args = TrainingArguments("test-trainer")

    for ii in range(config.itr):

        setting = '{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(
            config.model_id,
            config.pred_len,
            config.d_model,
            config.n_heads,
            config.e_layers,
            config.gpt_layers,
            config.d_ff,
            config.embed,
            ii,
        )
        save_dir = os.path.join(config.checkpoints, setting)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Load data
        train_loader = get_data_loader(config, 'train')
        vali_loader = get_data_loader(config, 'val')
        test_loader = get_data_loader(config, 'test')

        # train_loader, vali_loader, test_loader = get_data_loaders(config)
        print("Data loaded!")

        device = torch.device(config.device)

        # Train
        model = train(train_loader, vali_loader, save_dir, config, training_args, device, ii)
        print("Training done!")

        # Test
        if training_args.local_rank == 0:
            best_model_path = os.path.join(save_dir, 'checkpoint.pth')
            model.load_state_dict(torch.load(best_model_path))
            print("------------------------------------")
            mse, mae = test(model, test_loader, config, device, ii)
            mses.append(mse)
            maes.append(mae)

    if training_args.local_rank == 0:
        mses = np.array(mses)
        maes = np.array(maes)
        print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
        print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))


if __name__ == "__main__":
    args = get_args()
    seed_all(args.seed)
    run(args)
