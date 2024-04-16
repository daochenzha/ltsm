import numpy as np
import torch
import torch.nn as nn
from torch import optim
import ipdb

from transformers.models.gpt2.modeling_gpt2 import GPT2Model,GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange

from ltsm.models.embed import DataEmbedding, DataEmbedding_wo_time

from transformers.modeling_utils import PreTrainedModel
from .config import LTSMConfig

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass



class Chronos(PreTrainedModel):

    config_class = LTSMConfig

    # To load the LTSM model from pretrained weight, Run:
    # LTSM.from_pretrained("/home/sl237/ltsm/ltsm_hf/output/ltsm_debug")

    def __init__(self, configs, device=torch.device("cpu")):
        super().__init__(configs)
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len + configs.prompt_len - self.patch_size) // self.stride + 1
        self.d_type = torch.bfloat16
        self.pred_len = configs.pred_len    
        self.seq_len = configs.seq_len

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_gpt:
            # if configs.local_pretrain != "None":
            #     print("------------------Load pretrain from {}-----------------\n".format(configs.local_pretrain))
            #     self.gpt2 = GPT2Model.from_pretrained(configs.local_pretrain, output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2-medium',output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------\n")
                self.gpt2 = GPT2Model(GPT2Config())
            print("gpt2 = {}".format(self.gpt2))

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        # self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        

        


        # if configs.freeze and configs.pretrain:
        #     for i, (name, param) in enumerate(self.gpt2.named_parameters()):
        #         if 'ln' in name or 'wpe' in name:
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False
        #
        # for layer in (self.gpt2, self.in_layer, self.out_layer):
        #     layer.to(device=device)
        #     layer.train()

        # self.cnt = 0


    def forward(self, x, scale,  iters=None):
        B, L, M = x.shape
        # ipdb.set_trace()
        # means = x.mean(1, keepdim=True).detach()
        # ipdb.set_trace()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
        # x /= stdev
        # x = rearrange(x, 'b l m -> b m l')
        x = x.reshape(B*M, L)
        # x = self.padding_patch_layer(x)
        # x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # x = rearrange(x, 'b m n p -> (b m) n p')

        # outputs = self.in_layer(x)
        print('x.shape:',x.shape)
        token_ids, attention_mask, scale = self.tokenizer.input_transform(x)
        
        print("token_ids = {}".format(token_ids))
        print(token_ids.shape)
        
        if self.is_gpt:
            outputs = self.gpt2(input_ids=token_ids, attention_mask=attention_mask ).last_hidden_state

        
        token_ids_argmax = torch.argmax(outputs, dim=-1)
        print('outputs.shape',outputs.shape)
        print('token_ids_argmax.shape',token_ids_argmax.shape)
        print('token_ids_argmax',token_ids_argmax)
        token_ids_argmax = token_ids_argmax.reshape(B, 1, L)
        prediction = self.tokenizer.output_transform(
            token_ids_argmax, scale
        )
        
        outputs = prediction.reshape(B, L,M).to(x.device)   
        print('outputs.shape',outputs.shape)
        
        # outputs = self.out_layer(outputs.reshape(B*M, -1))
        # outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        
        

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs[:, :self.pred_len, :]
