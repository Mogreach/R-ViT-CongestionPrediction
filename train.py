import os.path
import torch
import time
from utils import torch as ptu
from model.utils import num_params
from engine import build_model , build_optimizer
import torch.optim as optim
from math import cos, pi
from datasets.build_dataset import build_dataset
from losses import build_loss,HuberLoss
import json
import csv
from test import test_per_epoch
from model import SegNet
class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights=[1],
                 min_lr=None,
                 min_lr_ratio=None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                      end: float,
                      factor: float,
                      weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                         f'cumulative_periods {cumulative_periods}')

    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
                            ]

def timing(time):
    hours=int(time/3600)
    mins=int((time-hours*3600)/60)
    second=time-hours*3600-mins*60
    return hours,mins,second

def logger(epoch,loss,result,logs):
    with open(logs,'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch,loss,result])

def training(arg,arg_dict,cfg):
    # 1. 设置随机种子
    # SEED = int(arg.order)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    if not os.path.exists("work_dir"):
        os.mkdir("work_dir")

    save_path = os.path.join(arg.save_path,arg.order)
    train_logs = os.path.join(save_path,"train_loss.csv")
    validation_logs = os.path.join(save_path, "validation_loss.csv")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise ValueError("Experiment save path exited!")

    if not os.path.exists(train_logs):
        with open(train_logs,'wt') as f:
            writer = csv.writer(f)
            # writer.writerow(["epoch","loss_val","PSNR","SSIM","EMD","NRMS","ROC","PRC"])
            writer.writerow(["epoch","loss_val","MAE"])
    if not os.path.exists(validation_logs):
        with open(validation_logs,'wt') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch","loss_val","MAE"])

    with open(os.path.join(save_path, 'arg.json'), 'wt') as f:
        json.dump(arg_dict, f, indent=4)
    #建立模型
    model = build_model(arg,cfg)
    # model = SegNet.SegNet(150).to("cuda")

    if arg.load :
        # 预训练加载
        data = torch.load(f"./pretrain_model/{arg.task}/{arg.pretrain_type+arg.backbone}.pth", map_location=ptu.device)
        checkpoint = data["model"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not (k.startswith('conv') or k.startswith('decoder'))}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    #导入数据集
    train_loader=build_dataset(arg_dict)
    arg_dict['ann_file'] = "./files/test_validation.csv"
    arg_dict['test_mode'] = True
    validation_dataset=build_dataset(arg_dict)
    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False
    # Build loss
    criterion = build_loss(arg_dict)    #MSE
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    # criterion = WeightedFocalLoss()

    #建立optimizer
    if arg.optimizer == "Adam":
        # optimizer = optim.AdamW(list(model.parameters())+list(criterion.parameters()), lr=arg_dict['lr'], betas=(0.9, 0.999),
        #                         weight_decay=arg_dict['weight_decay'])
        optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'], betas=(0.9, 0.999),
                                weight_decay=arg_dict['weight_decay'])
    elif arg.optimizer == "sgd":
        optimizer = build_optimizer(arg,len(train_loader),model)
    else:
        raise ValueError

    # train
    start_epoch = 0
    iter=0
    num_epochs = arg.epoch
    batch = int(len(train_loader)/arg.batch_size)

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [num_epochs*(len(train_loader))/arg.batch_size], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    # print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    # print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    #断点恢复训练
    if os.path.exists(os.path.join(save_path, "Segmodel.pth")):
        checkpoint = torch.load(os.path.join(save_path, "Segmodel.pth"))
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"]
        iter = start_epoch * len(train_loader)

    T1=time.time()
    for epoch in range(start_epoch, num_epochs):
        loss_val=0
        train_mae=0
        validation_loss_total=0
        mae_total=0
        for feature, label, _ in train_loader:
            input, target = feature.cuda(), label.cuda()

            prediction = model(input)
            loss=criterion(prediction,target)
            train_mae+=torch.mean(torch.abs(prediction.cpu() - target.cpu())).item()
            # loss = criterion(prediction.squeeze(), target.squeeze())
            loss_val+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter+=1
            if iter % batch ==0 :
                break
        # caculate validation loss
        for val_feature,val_label,_ in validation_dataset:
            validation_prediction = model(val_feature.cuda())
            validation_loss=criterion(validation_prediction,val_label.cuda())
            validation_loss_total+=validation_loss.item()
            mae_total  += torch.mean(torch.abs(validation_prediction.cpu() - val_label)).item()

        loss_mean = loss_val/len(train_loader)
        train_mae_mean = train_mae/len(train_loader)

        torch.save({'model':model.state_dict(),'epoch':epoch},os.path.join(save_path,"Segmodel.pth"))
        print(f"第{epoch+1}次epoch，平均loss：{loss_mean}")
        train_result=[0,0,0,0,0,0]
        # train_result = test_per_epoch(model_dir_path=os.path.join(arg.save_path, arg.order), model=model,
        #                         arg_dict=arg_dict,ann_file="./files/train_validation.csv")
        validation_loss_mean = validation_loss_total/len(validation_dataset)
        mae_mean = mae_total/len(validation_dataset)
        # validation_result = test_per_epoch(model_dir_path=os.path.join(arg.save_path, arg.order), model=model,
        #                         arg_dict=arg_dict,ann_file="./files/test_validation.csv")

        if (epoch + 1 ) % arg.save_freq == 0 :
            # result = test_per_epoch(model_dir_path=os.path.join(arg.save_path, arg.order), model=model,
            #                         arg_dict=arg_dict)
            torch.save(model,os.path.join(save_path,f"Segmodel_{epoch+1}ep.pth"))
        logger(epoch,loss_mean,train_mae_mean,train_logs)
        logger(epoch,validation_loss_mean,mae_mean,validation_logs)
        T2=time.time()
    hour,min,second=timing(T2-T1)
    print("Having trained for {} second".format(T2-T1))
    print("Having trained for : {} h {} min {} second".format(hour,min,second))
    return (hour,min,second)

