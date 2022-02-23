import torch
import json
import os
import numpy as np
import transformers
import torch.nn as nn
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def loss_fn(output, target):
    weights = [0.75, 1.0, 1.0, 1.0, 1.0]
    class_weights = torch.FloatTensor(weights).cuda()
    lfn = nn.CrossEntropyLoss(weight=class_weights)
    output = output.transpose(0, 1)
    target = target.transpose(0, 1)
    un_pad_mask = target != -100

    output_unpad = output[un_pad_mask]
    target_unpad = target[un_pad_mask]

    output_unpad = output_unpad.reshape(output.size(0), -1).t()
    target_unpad = target_unpad.reshape(target.size(0), -1).t()
    loss = lfn(output_unpad,torch.argmax(target_unpad,dim=1))
    # print("End of loss function!!\n\n")
    return loss

def read_config(config_file):
    # abspath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    basepath = os.path.join(basepath, "config")
    with open(os.path.join(basepath, config_file), 'r') as f:
        config = json.load(f)
    return basepath, config

def get_model_params(params):
    print('The BERT model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

def create_parameter(*size):
    out = nn.Parameter(torch.empty(*size, dtype=torch.float),requires_grad=True)
    if len(size) > 1:
        torch.nn.init.xavier_uniform_(out)
    else:
        torch.nn.init.uniform_(out)
    return out

class Bilinear2DAttention(nn.Module):
    def __init__(self, dim):
        super(Bilinear2DAttention,self).__init__()
        self.label_U_diag = create_parameter(dim, dim)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        # (batch x label x seq x seq)
        return torch.einsum('bij,jk,blk->bil', (head, self.label_U_diag, dep))

class BilinearLabelAttention(nn.Module):
    def __init__(self, dim, n_labels):
        super(BilinearLabelAttention,self).__init__()
        self.label_U_diag = create_parameter(n_labels, dim)

    def forward(self, head, dep):
        # head, dep: [batch x sequence x mlp]
        # (batch x label x seq x seq)
        return torch.einsum("bij,lj,boj->blio", (head, self.label_U_diag, dep))

class EntityModel(nn.Module):
    def __init__(self, dim_emb, label_count, CONFIG_NAME):
        super(EntityModel, self).__init__()
        self.dim_emb = dim_emb
        self.label_count = label_count
        _, config = read_config(CONFIG_NAME)
        self.context_len = 0
        if "CONTEXT" in config:
            for i in config["CONTEXT"]:
                if config["CONTEXT"][i] == "yes":
                    with open(config["DATA"]["SAVE_VOCAB"]+i+".json", 'r') as f:
                        self.context_len += len(json.load(f))

        bert_config = transformers.BertConfig.from_json_file(os.path.join(config["MODEL"]["MODEL_PATH"], 'config.json'))
        self.bert = transformers.BertModel.from_pretrained(os.path.join(config["MODEL"]["MODEL_PATH"], 'pytorch_model.bin'), config=bert_config)
        # get_model_params(list(self.bert.named_parameters()))
        ## the dropout layer makes random channels or input tensors to zeros to regularization
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bilstm1 = torch.nn.LSTM(self.context_len, 64, num_layers=2, dropout=0.3,
                                    bidirectional=True, batch_first=True)
        self.bilstm2 = torch.nn.LSTM(self.context_len, 64, num_layers=2, dropout=0.3,
                                     bidirectional=True, batch_first=True)
        self.head_tag = nn.Linear(768+128, self.dim_emb)
        self.dep_tag = nn.Linear(768+128, self.dim_emb)
        self.label_attention = BilinearLabelAttention(self.dim_emb, self.label_count)

        ## adding multiple attentions for different layers..
        # self.attention_none = Bilinear2DAttention(self.dim_emb)
        # self.attention_exp_pos = Bilinear2DAttention(self.dim_emb)
        # self.attention_exp_neg = Bilinear2DAttention(self.dim_emb)
        # self.attention_src = Bilinear2DAttention(self.dim_emb)
        # self.attention_tar = Bilinear2DAttention(self.dim_emb)


    def forward(self, ids, mask, context_vec, target_tag, opt):
        o1, _ = self.bert(ids, attention_mask=mask)
        bo_tag = self.bert_drop_1(o1)
        # lstm_out1, _ = self.bilstm1(torch.cat((bo_tag, context_vec), 2))
        # lstm_out2, _ = self.bilstm2(torch.cat((bo_tag, context_vec), 2))
        lstm_out1, _ = self.bilstm1(context_vec)
        lstm_out2, _ = self.bilstm2(context_vec)
        mlp_out1 = self.head_tag(torch.cat((bo_tag, lstm_out1), 2))
        mlp_out2 = self.dep_tag(torch.cat((bo_tag, lstm_out2), 2))
        tag = self.label_attention(mlp_out1, mlp_out2)

        ## multiple attentions forward
        # tag_none = self.attention_none(mlp_out1, mlp_out2)
        # tag_exp_pos = self.attention_exp_pos(mlp_out1, mlp_out2)
        # tag_exp_neg = self.attention_exp_neg(mlp_out1, mlp_out2)
        # tag_src = self.attention_src(mlp_out1, mlp_out2)
        # tag_tar = self.attention_tar(mlp_out1, mlp_out2)
        # tag = torch.stack([tag_none, tag_exp_pos, tag_exp_neg, tag_tar, tag_src],dim=0).transpose(0, 1)

        if opt == "train":
            loss_tag = loss_fn(tag, target_tag)
            return tag, loss_tag
        else:
            return tag