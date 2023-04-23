
from collections import defaultdict
import os
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam


def pad_col(input, val=0, where='end'):
    if input.ndim != 2:
        raise ValueError("Only works for 2-D tensors.")
    pad = torch.zeros_like(input[:, :1]) + val
    return torch.cat([input, pad] if where == 'end' else [pad, input], dim=1)


"""
Reference: Wang, Z., & Sun, J. (2022, August 7). Survtrace: Transformers for survival analysis with competing events. University of Illinois Urbana-Champaign
"""

class NLLPCHazardLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, interval_frac: Tensor,
                reduction: str = 'mean') -> Tensor:
        """
        Computes the negative log-likelihood of the piecewise-constant hazard model.

        References:
        [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
            with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
            https://arxiv.org/pdf/1910.06724.pdf
        """
        if events.dtype == torch.bool:
            events = events.float()

        idx_durations = idx_durations.view(-1, 1)
        events = events.view(-1)
        interval_frac = interval_frac.view(-1)

        keep = idx_durations.view(-1) >= 0
        phi = phi[keep, :]
        idx_durations = idx_durations[keep, :]
        events = events[keep]
        interval_frac = interval_frac[keep]

        log_h_e = torch.nn.functional.softplus(phi.gather(1, idx_durations).view(-1)).log().mul(events)
        haz = F.softplus(phi)
        scaled_h_e = haz.gather(1, idx_durations).view(-1).mul(interval_frac)
        haz = pad_col(haz, where='start')
        sum_haz = haz.cumsum(1).gather(1, idx_durations).view(-1) 
        loss = - log_h_e.sub(scaled_h_e).sub(sum_haz)

        return loss.mean()



############################
# trainer #
############################

class Trainer:
    def __init__(self, model, metrics=None):
        '''metrics must start from NLLPCHazardLoss, then be others
        '''
        self.model = model
        if metrics is None:
            self.metrics = [NLLPCHazardLoss(),]

        self.train_logs = defaultdict(list)
        self.get_target = lambda df: (df['duration'].values, df['event'].values)
        self.use_gpu = True if torch.cuda.is_available() else False
        ckpt_dir = os.path.dirname(model.config['checkpoint'])
        self.ckpt = model.config['checkpoint']
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    def fit(self, 
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):
        df_train, df_y_train = train_set
        durations_train, events_train = self.get_target(df_y_train)

        df_val, df_y_val = val_set
        durations_val, events_val = self.get_target(df_y_val)
        tensor_val = torch.tensor(val_set[0].values)
        tensor_y_val = torch.tensor(val_set[1].values)

        # assign no weight decay on these parameters
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, 
            learning_rate, 
            weight_decay_rate=weight_decay, 
            )

        num_train_batch = int(np.ceil(len(df_y_train) / batch_size))
        train_loss_list, val_loss_list = [], []
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            df_train = train_set[0].sample(frac=1)
            df_y_train = train_set[1].loc[df_train.index]

            tensor_train = torch.tensor(df_train.values)
            tensor_y_train = torch.tensor(df_y_train.values)

            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()

                batch_train = tensor_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_y_train = tensor_y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                
                batch_x_cat = batch_train[:, :self.model.config.num_categorical_feature].long()
                batch_x_num = batch_train[:, self.model.config.num_categorical_feature:].float()

                phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num)

                if len(self.metrics) == 1: # only NLLPCHazardLoss is asigned
                    batch_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,1].long(), batch_y_train[:,2].float(), reduction="mean")

                else:
                    raise NotImplementedError

                batch_loss.backward()
                optimizer.step()

                epoch_loss += batch_loss.item()

            train_loss_list.append(epoch_loss / (batch_idx+1))

            self.model.eval()
            with torch.no_grad():
                phi_val = self.model.predict(tensor_val, val_batch_size)
            
            val_loss = self.metrics[0](phi_val, tensor_y_val[:,0].long(), tensor_y_val[:,1].long(), tensor_y_val[:,2].float())
            print("[Train-{}]: {}".format(epoch, epoch_loss))
            print("[Val-{}]: {}".format(epoch, val_loss.item()))
            val_loss_list.append(val_loss.item())

        return train_loss_list, val_loss_list