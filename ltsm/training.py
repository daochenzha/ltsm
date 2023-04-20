from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
 

def train(
    train_loader,
    vali_loader,
    save_dir,
    config,
    device,
):
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
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=config.tmax, eta_min=1e-8)

    for epoch in range(config.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            
            # The following two are not used, but could be useful
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((config.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(
            model,
            vali_loader,
            criterion,
            config,
            device,
        )
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if config.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, config)
        early_stopping(vali_loss, model, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model

def vali(
    model,
    vali_loader,
    criterion,
    config,
    device,
):

    total_loss = []
    if model == "LTSM":
        model.in_layer.eval()
        model.out_layer.eval()
    else:
        model.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x)
            
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


