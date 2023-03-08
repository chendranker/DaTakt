import torch
import numpy as np
import os

from .bakt_time import BAKTTime
device = "cpu" if not torch.cuda.is_available() else "cuda"

import pandas as pd

def init_model(model_name, model_config, data_config, emb_type):
    print(f"in init_model, model_name: {model_name}")
    if model_name == "bakt_time":
        model = BAKTTime(data_config["num_c"], data_config["num_q"], data_config["num_rgap"], data_config["num_sgap"], data_config["num_pcount"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    else:
        print(f"The wrong model name: {model_name} was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    print(f"in load model! model name: {model_name}")
    model = init_model(model_name, model_config, data_config, emb_type)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.ckpt"))
    model.load_state_dict(net)
    return model
