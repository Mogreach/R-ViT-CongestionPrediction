import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import config
from arg.config import Paraser
from scipy.optimize import fsolve
#统计最大，最小，平均绝对误差样本
def plot_scater():

    model_path = "./work_dir/congestion/60/Segmodel_40ep.pth"
    model = torch.load(model_path)
    mean_sample=np.array([])
    max_sample=np.array([])
    min_sample=np.array([])
    with open("./files/CircuitNet-28_test.csv") as f:
        reader = csv.reader(f)
        sample = np.array([])
        for each in reader:
            feature_pth = "../CircuitNet-28/congestion/" + each[0]
            label_pth = "../CircuitNet-28/congestion/" + each[1]
            feature = np.load(feature_pth)
            label = np.load(label_pth)
            input = feature.transpose(2, 0, 1).astype(np.float32)
            input = torch.tensor(input).cuda().unsqueeze(0)

            output = model(input)
            # output=torch.sum(output,dim=1,keepdim=True)
            prediction = output[0, :, :].cpu().data.numpy()
            prediction = prediction.transpose(1, 2, 0)

            a_erro = abs(label-prediction)
            tmean = np.mean(a_erro)
            tmax = np.max(a_erro)
            tmin = np.min(a_erro)

            mean_sample = np.append(mean_sample,tmean)
            max_sample = np.append(max_sample, tmax)
            min_sample = np.append(min_sample, tmin)
            sample = np.append(sample,a_erro.flatten())
        info_dict={'平均最大误差': max_sample,'平均最小误差':min_sample,'平均误差':mean_sample}
        sam_dict = {'样本':sample}
        df = pd.DataFrame(data=info_dict)
        df2 = pd.DataFrame(data=sam_dict)
        df.to_csv("./fake_mean_sample.csv")
        df2.to_csv("./fake_sample.csv")
        # plt.scatter(np.arange(0,len(sample)),sample)
        # plt.show()
#显示总样本分布区间
def plot_hist():
    pf = pd.read_csv("./fake_sample.csv")
    data = np.array(pf['样本'])
    hist,bins = np.histogram(data,bins=8)
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.65)
    plt.xlabel('Value Range')
    plt.ylabel('Frequency')
    plt.title('Histogram of Mae Sample')
    plt.savefig("绝对误差直方图.svg")
    plt.show()


def show_scater():
    pf=pd.read_csv("./mean_sample.csv")
    # max_sample = pf['平均最大误差']
    # min_sample = pf['平均最小误差']
    mean_sample = pf['平均误差']
    length = len(mean_sample)
    x = np.arange(length)
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.scatter(x,pf[pf.columns[i+1]])
    plt.show()
def plot_loss(k,b):
    x=np.linspace(0,1,1000)
    y=25/(1+np.exp(-k*(abs(x)-b)))
    dy=25*(np.exp(-k*(x-b)))/(1+(2*np.exp(-k*(x-b)))+np.exp(-2*k*(x-b)))
    # y = 1 / (1 + np.exp( -10*x))
    print((10 / (1 + np.exp(-k * (abs(0.2) - b)))))
    print((10 / (1 + np.exp(-k * (abs(0.1) - b)))))
    print((10/(1+np.exp(-k*(abs(0.05)-b)))))
    plt.scatter(x,y,label="Weight Loss")
    plt.scatter(x,dy,label="Gradient Loss")
    plt.show()

def func(paramlist):
    x,y=paramlist[0],paramlist[1]
    return [ 1/(1+math.exp(1*x*(0.2-y)))-0.8,
             1 / (1 + math.exp(1 * x * (0.05 - y))) - 0.001]
# s=fsolve(func,[0,0])
# print(s)
# plot_loss(s[0],s[1])
# plot_scater()

# show_scater()
plot_hist()
# plot_loss(20,0.4)
