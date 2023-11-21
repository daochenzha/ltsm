from tqdm import tqdm
import numpy as np
import torch

from ltsm.utils.metrics import metric

def test(model, test_loader, config, device, iters):
    preds = []
    trues = []
    # mases = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            outputs = model(batch_x[:, -config.seq_len:, :], iters)
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            preds.append(pred)
            trues.append(true)


    preds = np.array(preds).reshape(-1)
    trues = np.array(trues).reshape(-1)
    # mases = np.mean(np.array(mases))
    print('test shape:', preds.shape, trues.shape)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    # print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}, mases:{:.4f}'.format(mae, mse, rmse, smape, mases))
    print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, smape:{:.4f}'.format(mae, mse, rmse, smape))

    return mse, mae

