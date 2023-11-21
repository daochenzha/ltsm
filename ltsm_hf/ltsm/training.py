from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
from ltsm.utils.metrics import metric

from accelerate import Accelerator

def train(
    train_loader,
    vali_loader,
    save_dir,
    config,
    training_args,
    device,
    iters
):
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    device = accelerator.device

    if config.model == 'PatchTST':
        from ltsm.models.PatchTST import PatchTST
        model = PatchTST(config, device)
        model.to(device)
    elif config.model == 'DLinear':
        from ltsm.models.DLinear import DLinear
        model = DLinear(config, device)
        model.to(device)
    else:
        from ltsm.models.ltsm_model import LTSM
        model = LTSM(config, device)

    time_now = time.time()
    train_steps = len(train_loader)

    model_optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    save_iters = save_iters_class(verbose=True)
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=config.tmax, eta_min=1e-8)

    model, model_optim, train_loader, vali_loader = accelerator.prepare(model, model_optim, train_loader, vali_loader)


    for epoch in range(config.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()


        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float() # .to(device)
            batch_y = batch_y.float() # .to(device)

            # The following two are not used, but could be useful
            # batch_x_mark = batch_x_mark.float().to(device)
            # batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, iters)
            loss = criterion(outputs, batch_y)

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((config.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
                save_iters(train_loss, model, save_dir, i + 1, training_args.local_rank)
                mse, mae = vali_metric(accelerator, model, vali_loader, config, device, 0)

            accelerator.backward(loss)
            model_optim.step()

            all_outputs, all_batch_y = accelerator.gather_for_metrics((outputs.contiguous(), batch_y))
            with torch.no_grad():
                all_loss = criterion(all_outputs, all_batch_y)
            train_loss.append(all_loss.item())

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(
            accelerator,
            model,
            vali_loader,
            criterion,
            config,
            device,
            0
        )
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if config.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, config)

        early_stopping(vali_loss, model, save_dir, training_args.local_rank)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model

def vali(
    accelerator,
    model,
    vali_loader,
    criterion,
    config,
    device,
    iters
):

    total_loss = []
    if model == "LTSM":
        model.in_layer.eval()
        model.out_layer.eval()
    else:
        model.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float() # .to(device)
            batch_y = batch_y.float()
            # batch_x_mark = batch_x_mark.float().to(device)
            # batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, iters)

            all_outputs, all_batch_y = accelerator.gather_for_metrics((outputs.contiguous(), batch_y))
            with torch.no_grad():
                all_loss = criterion(all_outputs.contiguous(), all_batch_y)
            total_loss.append(all_loss.item())


    total_loss = np.average(total_loss)
    if model == "LTSM":
        model.in_layer.train()
        model.out_layer.train()
    else:
        model.train()
    return total_loss


def adjust_learning_rate(optimizer, epoch,config):
    if config.lradj =='type1':
        lr_adjust = {epoch: config.learning_rate if epoch < 3 else config.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif config.lradj =='type2':
        lr_adjust = {epoch: config.learning_rate * (config.decay_fac ** ((epoch - 1) // 1))}
    elif config.lradj =='type4':
        lr_adjust = {epoch: config.learning_rate * (config.decay_fac ** ((epoch) // 1))}
    else:
        config.learning_rate = 1e-4
        lr_adjust = {epoch: config.learning_rate if epoch < 3 else config.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def vali_metric(accelerator, model, test_loader, config, device, iters):
    preds = []
    trues = []
    # mases = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float() # .to(device)
            batch_y = batch_y.float()

            outputs = model(batch_x[:, -config.seq_len:, :], iters)
            # pred = outputs.detach().cpu().numpy()
            # true = batch_y.detach().cpu().numpy()

            all_outputs, all_batch_y = accelerator.gather_for_metrics((outputs.contiguous(), batch_y))

            preds.append(all_outputs.cpu().numpy())
            trues.append(all_batch_y.cpu().numpy())

    preds = np.array(preds).reshape(-1)
    trues = np.array(trues).reshape(-1)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

class save_iters_class:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.iter = 0

    def __call__(self, val_loss, model, path, iters, local_rank=0):
        if iters % 500 == 0 and local_rank == 0:
            self.iter = iters
            print("Saving iteration model of {}".format(iters))
            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            torch.save(model.state_dict(), path + '/' + 'checkpoint_'+ str(self.iter) + '.pth')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, local_rank=0):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if local_rank == 0:
                self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            if local_rank == 0:
                self.save_checkpoint(val_loss, model, path)

            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


