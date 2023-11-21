from base_wrapper import TimeSeriesDomainAdaptationWrapper
import torch
from torch import nn, optim


def calc_attn(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    attn_logits = torch.exp(
        query.transpose(1, 2) @ key
        - (query.transpose(1, 2) @ key)
        * torch.eye(query.shape[-1], device=query.device)
        / math.sqrt(query.shape[-1])
    )
    attn_scores = torch.softmax(attn_logits, dim=-1)
    attn_values = (attn_scores @ value.transpose(1, 2)).transpose(1, 2)
    return attn_values


class DAF(TimeSeriesDomainAdaptationWrapper):
    def __init__(self, source_model, target_model, configs):
        super(DAF, self).__init__()
        self.device = configs.device
        self.source_model = source_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.source_Value = nn.Linear(configs.dim1, configs.vdim).to(self.device)
        self.target_Value = nn.Linear(configs.dim1, configs.vdim).to(self.device)
        self.Key = nn.Linear(configs.dim1, configs.vdim).to(self.device)
        self.Query = nn.Linear(configs.dim1, configs.vdim).to(self.device)
        self.dense = nn.Linear(configs.vdim, configs.dim2).to(self.device)
        self.domain_discriminator = nn.Linear(configs.vdim * 2, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.tradeoff = configs.tradeoff
        self.models = [self.source_model, self.target_model, self.source_Value, self.target_Value, 
                      self.Key, self.Query, self.dense, self.domain_discriminator]
            
    def new_train_forward(self, source_x, target_x):
        # source_x & target_x: (N * C * L)
        for _ in self.models:
            _.train()
        encoded_x_source, encoded_x_target = self.source_model.encoder(self.source_model.preprocess(x)), 
                                             self.target_model.encoder(self.target_model.preprocess(x))
        v_source, v_target = self.source_Value(encoded_x_source), self.target_Value(encoded_x_target)
        source_k, source_q, target_k, target_q = self.Key(encoded_x_source), self.Query(encoded_x_source), 
                                                 self.Key(encoded_x_target), self.Query(encoded_x_target)
        rep_in_source, rep_in_target = self.dense(calc_attn(source_q, source_k, v_source)),
                                       self.dense(calc_attn(target_q, target_k, v_target))
        source_pred, target_pred = self.source_model.postprocess(self.source_model.decoder(rep_in_source)), 
                                   self.target_model.postprocess(self.target_model.decoder(rep_in_target))
        source_kq = torch.flatten(torch.cat((source_k, source_q), -1), start_dim=0, end_dim=1)
        target_kq = torch.flatten(torch.cat((target_k, target_q), -1), start_dim=0, end_dim=1)
        kq = torch.cat((source_kq, target_kq), 0)
        domain_pred = torch.squeeze(self.sigmoid(self.domain_discriminator(kq)), -1)
        domain_y = torch.cat((torch.ones(source_kq.shape[0]), torch.zeros(target_kq.shape[0])), 0)
        return source_pred, target_pred, domain_pred, domain_y

    def new_inference_forward(self, x, domain_flag):
        # x: (N * C * L)
        for _ in self.models:
            _.eval()
        if domain_flag == 0:
            # source domain
            encoded_x = self.source_model.encoder(self.source_model.preprocess(x))
            v = self.source_Value(encoded_x)
            k, q = self.Key(encoded_x), self.Query(encoded_x)
            rep_in = self.dense(calc_attn(q, k, v))
            return self.source_model.postprocess(self.source_model.decoder(rep_in))
        else:
            # target domain
            encoded_x = self.target_model.encoder(self.target_model.preprocess(x))
            v = self.target_Value(encoded_x)
            k, q = self.Key(encoded_x), self.Query(encoded_x)
            rep_in = self.dense(calc_attn(q, k, v))
            return self.target_model.postprocess(self.target_model.decoder(rep_in))

    def loss(self, source_pred, target_pred, source_y, target_y):
        # source_pred & target_pred & source_y & target_y: (N * C * H)
        mse, bce = nn.MSELoss(), nn.BCELoss()
        seq_loss = mse(source_y, source_pred).mean() + mse(target_y, target_pred).mean()
        return seq_loss

    def train(self, data_loader, lr, n_epoch):
        att_optim = optim.Adam(
            list(self.Key.enc.parameters())
            + list(self.Query.dec.parameters())
            + list(self.dense.enc.parameters()),
            lr=lr,
        )
        gen_optim = optim.Adam(
            list(self.source_model.enc.parameters())
            + list(self.target_model.dec.parameters())
            + list(self.source_Value.enc.parameters())
            + list(self.target_Value.dec.parameters()),
            lr=lr,
        )
        dis_optim = optim.Adam(self.domain_discriminator.parameters(), lr=lr)
        for epoch in range(n_epoch):
            one_epoch_train_loss = []
            for i, (source_x, target_x, source_y, target_y) in enumerate(data_loader):
                s_pred, t_pred, domain_pred, domain_y = self.new_train_forward(self, source_x, target_x)
                seq_loss = self.loss(s_pred, t_pred, s_y, t_y)
                bce = nn.BCELoss()
                dom_loss = bce(domain_pred, domain_y)
                loss = seq_loss - self.tradeoff * dom_loss
                one_epoch_train_loss.append(loss.item())
                att_optim.zero_grad()
                gen_optim.zero_grad()
                loss.backward()
                att_optim.step()
                gen_optim.step()
                dis_optim.zero_grad()
                dom_loss.backward()
                dis_optim.step()
            print('Epoch: [{}/{}], Average Loss: {:.9f}'.format(epoch+1, 
                                                                n_epoch, 
                                                                sum(one_epoch_train_loss) / len(one_epoch_train_loss)))
