import argparse
import os
import sys

sys.path.append(os.getcwd())


class Paraser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--Experiment_name', default='SegmTransformer',help="实验文件名称")
        self.parser.add_argument('--Experiment_note', default='测试132，阈值设置为0.1')
        self.parser.add_argument('--order', default='132',help="1")
        self.parser.add_argument('--load', default=False ,type=bool)   #是否加载已下载的模型
        self.parser.add_argument('--task', default='congestion_gpdl',help="congestion_gpdl/drc_routenet/irdrop_mavi")
        self.parser.add_argument('--pretrained', default=False , type=bool) #是否用生成数据进行预训练
        self.parser.add_argument('--gan_max_iters', default=50000,type=int,help="GAN's iters")
        self.parser.add_argument('--pre_max_iters', default=50000, type=int, help="pretrain's iters")
        self.parser.add_argument('--fin_max_iters', default=50000, type=int, help="fine tune's iters")
        self.parser.add_argument('--save_as_npy', action='store_true')
        self.parser.add_argument('--arg_file', default=None)
        self.parser.add_argument('--multiple', default=5,type=float)
        self.parser.add_argument('--result_path', default=None)
        self.parser.add_argument('--z_dim', default=50)

        # training hyper-parameters
        self.parser.add_argument('--num_workers', type=int, default=16)#在drc_dataset设置,16
        self.parser.add_argument("--log_dir", default="seg_tiny_mask", type=str, help="logging directory")
        self.parser.add_argument("--dataset", default="drc", type=str)
        self.parser.add_argument("--im_size", default=None, type=int, help="dataset resize size")
        self.parser.add_argument("--crop_size", default=None, type=int)
        self.parser.add_argument("--window_size", default=None, type=int)
        self.parser.add_argument("--window_stride", default=None, type=int)
        self.parser.add_argument("--pretrain_type", default='', type=str)#填fake则加载以生成数据为主的预训练模型
        self.parser.add_argument("--backbone", default="vit_tiny_patch16_384", type=str)
        self.parser.add_argument("--decoder", default="mask_transformer", type=str)
        self.parser.add_argument("--optimizer", default="Adam", type=str)
        self.parser.add_argument("--scheduler", default="polynomial", type=str)
        self.parser.add_argument("--weight_decay", default=1e-4, type=float)
        self.parser.add_argument("--dropout", default=0.1, type=float)
        self.parser.add_argument("--drop_path", default=0.1, type=float)
        self.parser.add_argument("--batch_size", default=8, type=int)

        self.parser.add_argument("--epoch", default=20, type=int)
        self.parser.add_argument("--save_freq", default=5, type=int)
        self.parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
        self.parser.add_argument("--normalization", default=None, type=str)
        self.parser.add_argument("--eval_freq", default=None, type=int)
        self.parser.add_argument("--amp/--no-amp", default=False)
        self.parser.add_argument("--resume/--no-resume", default=True)
        self.parser.add_argument("--n_cls", default=150,type=int)
        self.parser.add_argument('--ann_file_train', default='files/CircuitNet-28_train.csv')
        self.parser.add_argument('--ann_file_test', default='files/CircuitNet-28_test.csv')
        self.parser.add_argument('--test_mode',default=False)
        self.get_remainder()

    def get_remainder(self):
        if self.parser.parse_args().task == 'drc_routenet':
            self.parser.add_argument('--save_path', default='work_dir/DRC/')
            self.parser.add_argument('--logs_path', default='work_dir/DRC/logs/')
            self.parser.add_argument('--dataroot', default='../CircuitNet-28/DRC/')
            self.parser.add_argument('--dataset_type', default='DRCDataset')
            self.parser.add_argument('--aug_pipeline', default=['Flip'])
            self.parser.add_argument('--model_type', default='RouteNet')
            self.parser.add_argument('--in_channels', default=9)
            self.parser.add_argument('--out_channels', default=1)
            # self.parser.add_argument('--weight_decay', default=1e-4)
            self.parser.add_argument('--loss_type', default='MixLoss')
            self.parser.add_argument('--eval-metric', default=['PSNR', 'SSIM','NRMS'])
            self.parser.add_argument('--threashold', default=0.1)


        elif self.parser.parse_args().task == 'congestion_gpdl':
            self.parser.add_argument('--save_path', default='work_dir/congestion/')
            self.parser.add_argument('--logs_path', default='work_dir/congestion/logs/')
            self.parser.add_argument('--dataroot', default='../CircuitNet-28/congestion/')
            self.parser.add_argument('--dataset_type', default='CongestionDataset')
            self.parser.add_argument('--aug_pipeline', default=['Flip'])

            self.parser.add_argument('--model_type', default='GPDL')
            self.parser.add_argument('--in_channels', default=3)
            self.parser.add_argument('--out_channels', default=1)
            # self.parser.add_argument('--weight_decay', default=0)
            self.parser.add_argument('--loss_type', default='HuberLoss')
            self.parser.add_argument('--eval-metric', default=['PSNR', 'SSIM', 'EMD', 'NRMS'])
            self.parser.add_argument('--threashold', default=0.15)



        elif self.parser.parse_args().task == 'irdrop_mavi':
            self.parser.add_argument('--save_path', default='work_dir/IR_drop/')
            self.parser.add_argument('--logs_path', default='work_dir/IR_drop/logs/')
            self.parser.add_argument('--dataroot', default='../CircuitNet-28/IR_drop/')
            self.parser.add_argument('--in_channals', type=int, default=5)
            self.parser.add_argument('--ann_file_train', default='files/train.csv')
            self.parser.add_argument('--ann_file_test', default='files/test.csv')
            self.parser.add_argument('--dataset_type', default='IRDropDataset')
            self.parser.add_argument('--batch_size', default=2)
            self.parser.add_argument('--model_type', default='MAVI')
            self.parser.add_argument('--in_channels', default=1)
            self.parser.add_argument('--out_channels', default=4)
            self.parser.add_argument('--lr', default=2e-4)
            # self.parser.add_argument('--weight_decay', default=1e-2)
            self.parser.add_argument('--loss_type', default='L1Loss')
            self.parser.add_argument('--eval_metric', default=['PSNR', 'SSIM'])
            self.parser.add_argument('--threashold', default=0.05)


        else:
            raise ValueError
