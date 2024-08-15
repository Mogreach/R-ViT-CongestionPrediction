import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
import csv
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from datasets.build_dataset import build_dataset
import matplotlib.pyplot as plt
from engine import build_model
import config
from arg.config import Paraser
import json
from torchstat import stat
import time
"""
match函数：
功能：判断给定数值，与（0，1）四等分的阈值比较，返回一组向量，反映该数值高于哪个阈值
输入：数值
输出：1x5向量
说明：向量组划分阈值依次为0.25，0.5，0.75，可划分为4个区间，0表示不属于该区间，1表示属于该区间
     向量组第一个表示是否为0，若数值为0，则返回[1,0,0,0,0]
     如：若给定0.8，则返回[0,0,0,0,1]
"""
def match(pixel):
    result=np.zeros(5)
    if pixel == 0:
        result[0]=1
        return result
    elif pixel < 0.25:
        result[1]=1
        return result
    elif pixel < 0.5:
        result[2]=1
        return result
    elif pixel < 0.75:
        result[3]=1
        return result
    else:
        result[4]=1
        return result
"""
normlization函数：
功能：给定预测数组，将所有小于0的数值置为0
输入：（HxWxC）数组
输出：（HxWxC）数组
"""
def normlization(predict):
    for i in range(predict.shape[0]):
        for k in range(predict.shape[1]):
            if predict[i][k][0] < 0:
                predict[i][k][0]=0
    return predict
"""
cal_cls_n_array函数：
功能：一个热点图样本，历遍每个像素，通过match函数，计算各自的分类向量，再累加，返回该样本分类向量;同时根据给定阈值，得出对应的分类图
输入：（HxWx1）数组 x2；（1x3）数组；(1x3)阈值数组
输出：（1x5）向量组 x2；（3xHxW）矩阵
说明：分类向量为[a,b,c,d,e]，代表该样本共有a个像素=0，b个像素大于0但小于0.25，c个像素大于等于0.25但小于0.5...
     one-hot矩阵，若大于等于阈值则为1，小于则为0，对应3个阈值
"""
def cal_cls_n_array(label,prediction,threasold):
    threasold_num=len(threasold)
    label_cls=np.zeros(threasold_num+2)
    prediction_cls=np.zeros(threasold_num+2)
    label_onehot=np.zeros((label.shape[0],label.shape[1],threasold_num))
    prediction_onehot = np.zeros((label.shape[0], label.shape[1], threasold_num))
    for i in range(label.shape[0]):
        for k in range(label.shape[1]):
            label_cls+=match(label[i][k][0])
            prediction_cls+=match(prediction[i][k][0])

            for j,thre in enumerate(threasold):
                if label[i][k][0] >= thre:
                    label_onehot[i][k][j]=1
                if prediction[i][k][0] >= thre:
                    prediction_onehot[i][k][j]=1

    return label_cls,prediction_cls,label_onehot,prediction_onehot



def main(model):
    with open("./files/test.csv") as f:
        reader=csv.reader(f)
        threasold=np.array([0.1,0.5,0.75])
        for each in reader:

            feature_pth="../CircuitNet/DRC/" + each[0]
            label_pth = "../CircuitNet/DRC/" + each[1]
            feature=np.load(feature_pth)
            label=np.load(label_pth)
            input = feature.transpose(2,0,1).astype(np.float32)
            input = torch.tensor(input).cuda().unsqueeze(0)

            output = model(input)
            output=torch.sum(output,dim=1,keepdim=True)
            prediction = output[0,:,:].cpu().data.numpy()
            prediction=prediction.transpose(1,2,0)
            prediction=normlization(prediction)

            label_cls,prediction_cls,label_onehot,prediction_onehot=cal_cls_n_array(label,prediction,threasold)
            plt.imshow(prediction[:,:,0])
            plt.show()
            confmatrix=confusion_matrix(label_onehot[:,:,0].flatten(),prediction_onehot[:,:,0].flatten(),labels=[0,1])

            # sns.heatmap(confmatrix, annot=True, fmt="d")
            plt.plot(confmatrix)
            print("next")
"""测试混淆矩阵"""
def test_confusion(file_path,model_name):
    time_sum=0
    argp = Paraser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    model_dir_path = file_path
    # model_dir_path = "work_dir/congestion/12/"
    print(f"=========================================================================>>\
            Current model path : {model_dir_path}\n")
    arg.arg_file =  model_dir_path + "/arg.json"
    #arg_file 参数文件
    # if arg.arg_file is not None:
    #     with open(arg.arg_file, 'rt') as f:
    #         arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test']
    arg_dict['test_mode'] = True
    conf_matrix_sum = np.array([0,0,0,0]).astype('float64')
    # model_path = os.path.join(model_dir_path,"Segmodel_20ep.pth")
    model_path = model_dir_path
    # model_path = "./work_dir/model_cmp/CGAN2.pth"
    model = torch.load(model_path)
    # threshold_label = arg.threashold
    threshold_label = 0.1
    dataset=build_dataset(arg_dict)
    size = len(dataset)
    for feature, label, label_path in dataset:
        input, target = feature.cuda(), label.cuda()
        start_time = time.time()
        prediction = model(input)
        end_time = time.time()
        time_sum += (end_time - start_time)
        target_test = (prediction.reshape(256, 256).squeeze().cpu().detach().numpy() >= threshold_label).astype(int)
        target_probabilities = (target.reshape(256, 256).squeeze().cpu().detach().numpy() >= threshold_label).astype(int)
        #tn, fp, fn, tp
        temp = confusion_matrix(y_true=target_probabilities.flatten(), y_pred=target_test.flatten()).ravel()
        conf_matrix_sum += (temp/size)
    print(conf_matrix_sum)
    print(f"Total samples:{size}")
    print(f"Accuracy:{(conf_matrix_sum[0] + conf_matrix_sum[3]) / (256 * 256)}")
    print(f"Precision:{conf_matrix_sum[3]/(conf_matrix_sum[1]+conf_matrix_sum[3])}")
    print(f"Time consumed:{time_sum}")
    data_to_save = {
    'Model': [file_path],  
    'Size': [size],  
    'TN': [int(conf_matrix_sum[0])],
    'FP': [int(conf_matrix_sum[1])],
    'FN': [int(conf_matrix_sum[2])],
    'TP': [int(conf_matrix_sum[3])], 
    'Accuracy': [round((conf_matrix_sum[0] + conf_matrix_sum[3]) / (256 * 256),4)],  
    'Precision': [round(conf_matrix_sum[3] / (conf_matrix_sum[1] + conf_matrix_sum[3]),4)],  
    'Time consumed': [round(time_sum,2)],
    'Average time consumed': [round(time_sum/size,7)] 
    } 
    # 创建一个DataFrame  
    df = pd.DataFrame(data_to_save)  
    
    # 将DataFrame保存到CSV文件  
    csv_filename = f'work_dir/model_cmp_supplement/{model_name}_results.csv'  # 你可以更改文件名和路径 
     # 检查文件是否已存在并包含标题行  
    if os.path.exists(csv_filename):  
        # 如果文件存在，则追加数据  
        with open(csv_filename, mode='a', newline='') as file:  
            writer = csv.writer(file)  
            # 注意：这里我们假设你的CSV文件已经有一个标题行，所以我们不写入标题  
            # 写入新数据行  
            for row in df.itertuples(index=False, name=None):  
                writer.writerow(row)  
    else:  
        # 如果文件不存在，则创建文件并写入标题和数据     
        df.to_csv(csv_filename, index=False)  
    
    print(f"Results saved to {csv_filename}") 

def test_param():
    argp = Paraser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    model_dir_path = "work_dir/congestion/140/"
    arg.arg_file = model_dir_path + "/arg.json"
    # arg_file 参数文件
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test']
    arg_dict['test_mode'] = True
    model_path = os.path.join(model_dir_path, "Segmodel_20ep.pth")
    # model_path = "./work_dir/model_cmp/FCN.pth"
    model = torch.load(model_path)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # stat(model.to("cpu"), (3, 256, 256))
# argp = Paraser()
# arg = argp.parser.parse_args()
# arg_dict = vars(arg)
# cfg = config.load_config()
# model = build_model(arg,cfg)
# model_path="./work_dir//Segmodel60.pth"
# model.load_state_dict(torch.load(model_path))
# model = build_model(arg,cfg)
# main(model)
#==============================================
# index_list = [63,65,66,67]
# model_list =["FCN","CGAN","SegNet","ConvNexT","ViT"]
# for j in range(3):
#     for i in model_list:
#         # test_confusion(f"work_dir/congestion/{i}/")
#         test_confusion(f"work_dir/model_cmp_supplement/{i}.pth",i)
#==============================================
test_param()