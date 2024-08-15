# Copyright 2022 CircuitNet. All rights reserved.

import pred_model
import torch

def build_model(opt):
    model = pred_model.__dict__[opt['model_type']](**opt)
    model.init_weights(**opt)
    if opt['test_mode']:
        model.eval()
    elif not opt['test_mode'] and opt['pretrained'] is not None:
        model.load_state_dict(torch.load(opt['pretrained'])['state_dict'])
    return model
