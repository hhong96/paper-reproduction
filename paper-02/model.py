import torch
import numpy as np
import torch.nn.functional as F
from modeling_bert import BaseModel, BertEmbeddings, BertEncoder, BertCLS
from train_utils import pad_col

class SurvTraceSingle(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertCLS(config)
        self.config = config
        self.init_weights()

    def forward(self, input_ids, input_nums):
        embedding_output = self.embeddings(input_ids=input_ids, input_x_num=input_nums)
        encoder_outputs = self.encoder(embedding_output)
        sequence_output = encoder_outputs[1]
        predict_logits = self.cls(encoder_outputs[0])
        
        return sequence_output, predict_logits


    def predict(self, x_input, batch_size=None):
        if not isinstance(x_input, torch.Tensor):
            x_input_cat = x_input.iloc[:, :self.config.num_categorical_feature]
            x_input_num = x_input.iloc[:, self.config.num_categorical_feature:]
            x_cat = torch.tensor(x_input_cat.values).long()
            x_num = torch.tensor(x_input_num.values).float()
        else:
            x_cat = x_input[:, :self.config.num_categorical_feature].long()
            x_num = x_input[:, self.config.num_categorical_feature:].float()

        num_sample = len(x_num)
        self.eval()
        with torch.no_grad():
            if batch_size is None:
                preds = self.forward(x_cat, x_num)[1]
            else:
                preds = []
                num_batch = int(np.ceil(num_sample / batch_size))
                for idx in range(num_batch):
                    batch_x_num = x_num[idx*batch_size:(idx+1)*batch_size]
                    batch_x_cat = x_cat[idx*batch_size:(idx+1)*batch_size]
                    batch_pred = self.forward(batch_x_cat,batch_x_num)
                    preds.append(batch_pred[1])
                preds = torch.cat(preds)

        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        surv = hazard.cumsum(1).mul(-1).exp()
        return preds, surv
