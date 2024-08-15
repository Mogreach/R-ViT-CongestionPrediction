import csv
from test import test_model
from arg.config import Paraser
import json
import config
from utils import torch as ptu
import os
import time
from train import training
from scripts import Generate , fid
def main(arg):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    arg_dict = vars(arg)
    arg_dict['test_mode'] = False
    arg_dict['ann_file'] = arg_dict['ann_file_train']
    print(f"Dataset:{arg.ann_file_train}")


    # set up configuration
    cfg = config.load_config()
    # start distributed mode
    ptu.set_gpu_mode(True)

    record_data=[arg_dict['Experiment_name'],arg_dict['order'],time.asctime(time.localtime()),0,0,
                 arg_dict['pre_max_iters'],arg_dict['fin_max_iters']]

    if arg.pretrained:
        # 预训练 以生成数据为训练样本
        # 生成数据
        # G = Generate.Generate_data(arg_dict)
        # G.main()
        # 计算FID值
        # f_fid,l_fid=fid.FID(arg_dict["dataroot"],arg_dict["task"])
        f_fid=0
        l_fid=0
        record_data[3]=f_fid
        record_data[4]=l_fid
        print(f"Feature FID:{f_fid}")
        print(f"Label FID:{l_fid}")
        arg_dict['ann_file']='files/fake.csv'
        hour,min,second = training(arg,arg_dict,cfg)
    else:
        #正式训练
        hour,min,second = training(arg,arg_dict,cfg)
    train_time = f"{hour}h {min}min {second}second"
    record_data.append(train_time)

    #测试结果
    result=test_model(model_dir_path=os.path.join(arg.save_path,arg.order),model_name=f"Segmodel_{arg.epoch}ep.pth")

    for each in result:
        record_data.append(each)

    record_data.append(arg_dict["Experiment_note"])
    #记录数据
    if arg_dict["task"]=="congestion_gpdl":
        record=open("../record_congestion.csv", 'a', newline='')
    elif arg_dict["task"]=="drc_routenet":
        record = open("../record_drc.csv", 'a', newline='')
    else:
        record = open("../record_irdrop.csv", 'a', newline='')
    writer = csv.writer(record)
    writer.writerow(record_data)
    record.close()



if __name__ == '__main__':
    argp = Paraser()
    arg = argp.parser.parse_args()
    # arg.order = str(79)
    # arg.load = False
    # arg.Experiment_note = "CircuitNet-28数集，tinyeh=20，加载预训练模型,采用bias_loss"
    arg.test_mode = True
    if arg.test_mode == False:
        arg.ann_file = arg.ann_file_train
        main(arg)
    else:
        arg_dict =vars(arg)

        record_data = [arg_dict['Experiment_name'], arg_dict['order'], time.asctime(time.localtime()), 0, 0,
                       arg_dict['pre_max_iters'], arg_dict['fin_max_iters']]
        train_time = "None"
        record_data.append(train_time)
        result=test_model(model_dir_path=os.path.join(arg.save_path,arg.order),model_name=f"Segmodel_{arg.epoch}ep.pth")
        for each in result:
            record_data.append(each)
        record_data.append(arg_dict["Experiment_note"])
        # 记录数据
        if arg_dict["task"] == "congestion_gpdl":
            record = open("../record_congestion.csv", 'a', newline='')
        elif arg_dict["task"] == "drc_routenet":
            record = open("../record_drc.csv", 'a', newline='')
        else:
            record = open("../record_irdrop.csv", 'a', newline='')
        writer = csv.writer(record)
        writer.writerow(record_data)
        record.close()