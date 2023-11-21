from tqdm import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
from tsbench.utils.metrics import metric
from .utils.tools import *


class BaseRunner:
    def __init__(self, model, model_args, save_dir, device):
        pass

    def train(self,train_loader, val_loader):
        pass

    def test(self, test_loader, processor):
        pass


class StandardRunner(BaseRunner):

    def __init__(self, model_factory, model_args, save_dir, logger, device):
        self.save_dir = save_dir
        self.logger = logger
        self.device = device

        self.args = model_args
        self.model = model_factory(model_args).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean') # Loss

    def train(self,train_loader, val_loader):
        train_steps = len(train_loader)
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)


        for epoch in range(self.args.train_epochs):
            train_loss = []
            epoch_start_time = time.time()

            # Get Training Loss
            print("Training...")
            for batch_x, batch_y in tqdm(train_loader):

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                train_loss.append(loss.item())

            cost_time = time.time() - epoch_start_time
            train_loss = np.average(train_loss)
            print("Validation...")
            val_loss = self._validation(val_loader)

            log_info = (epoch + 1, train_loss, val_loss, cost_time)
            self.logger.log("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Time: {3:.7f}".format(*log_info))
            self.logger.log_performance(*log_info)

            if self.args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            early_stopping(val_loss, self.model, self.save_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return self.model

    def test(self, test_loader, processor):
        # Get Test Loss
        mse = nn.MSELoss(reduction="mean") # MSE Loss
        mae = nn.L1Loss(reduction="mean") # MAE Loss

        test_outputs, test_y = self._predict(test_loader)
        test_mse = mse(test_outputs, test_y).item()
        test_mae = mae(test_outputs, test_y).item()

        test_outputs_inversed = torch.from_numpy(processor.inverse_process(test_outputs.cpu().numpy()))
        test_y_inversed = torch.from_numpy(processor.inverse_process(test_y.cpu().numpy()))
        test_mse_raw = mse(test_outputs_inversed, test_y_inversed).item()
        test_mae_raw = mae(test_outputs_inversed, test_y_inversed).item()

        return test_mse, test_mae, test_mse_raw, test_mae_raw

    def _validation(self, val_loader):
        val_outputs, val_y = self._predict(val_loader)
        return self.criterion(val_outputs, val_y).item()

    def _predict(self, loader):
        outputs, y = [], []
        for batch_x, batch_y in tqdm(loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            with torch.no_grad():
                output = self.model(batch_x)
            y.append(batch_y)
            outputs.append(output)

        y = torch.cat(y, 0)
        outputs = torch.cat(outputs, 0)

        return outputs, y


