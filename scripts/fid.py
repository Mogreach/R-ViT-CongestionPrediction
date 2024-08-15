from pytorch_fid import fid_score as fi
from os import system
import torch
def FID(data_root,task):
    feature_path=[f"{data_root}feature",f"{data_root}fake_feature"]
    label_path=[f"{data_root}label",f"{data_root}fake_label"]
    if task=="congestion_gpdl":
        fi.init(feature_path,"RGB",1)
        f_fid=fi.cal_fid()
        fi.init(label_path,"L",1)
        l_fid=fi.cal_fid()
    elif task=="drc_routenet":
        sum=0
        for k in range(3):
            fi.init(feature_path, "RGB", k+1)
            sum+=fi.cal_fid()
        f_fid=sum/3
        fi.init(label_path,"L",1)
        l_fid=fi.cal_fid()
    else:
        fi.init(feature_path, "RGB", 1)
        f_fid=fi.cal_fid()
        fi.init(label_path,"L",1)
        l_fid=fi.cal_fid()
    torch.cuda.empty_cache()
    return f_fid,l_fid