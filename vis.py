import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from PyQt5.QtCore import QLibraryInfo
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from roc import roc_prc
from sklearn.metrics import f1_score

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)
def show_feature_imgs():
    path="F:\CircuitNet\DRC\\feature"
    files=os.listdir(path)
    ofiles=[0 for i in range(10370)]
    col=0
    row=0
    layout=np.zeros((45*256,45*256))
    array = np.ones((256, 256))
    for filename in files:
        index = int(filename.split("-")[0])
        ofiles[index-1]=filename

    for file in ofiles:
        if type(file)==type('a'):
            name=file.split("-")
        else:
            continue
        if name[1]=="RISCY" and name[2]=="a":
            temp=np.load(path+"\\"+file)
            array=temp[:,:,0]
            if col<=44:
                layout[(row*256):((row+1)*256),(col*256):((col+1)*256)]=array
                col+=1
            else:
                row+=1
                col=0
    plt.imshow(layout)
    plt.show()
def tpr(tp, fn):
    return tp/(tp+fn)

def fpr(fp, tn):
    return fp/(fp+tn)

def precision(tp, fp):
    return tp/(tp+fp)
#展示roc_prc.csv内容，绘制roc曲线
def show_roc(csv_path):
    tpr_sum_List = []
    fpr_sum_List = []
    precision_sum_List = []
    threshold_remain_list = []
    count=0
    num = 0
    tpr_sum = 0
    fpr_sum = 0
    precision_sum = 0
    fp_thredsold_sum=0
    tp_thredsold_sum = 0
    fn_thredsold_sum=0
    tn_thredsold_sum=0
    first_flag = False
    csv_file = open(os.path.join(csv_path,"roc_prc.csv"), 'r')
    for line in csv_file:
        #修复bug，csv隔行有回车，需剔除
        if line=='\n':
            continue
        threshold, idx, tn, fp, fn, tp = line.strip().split(',')
        if threshold == "0.1":
            fp_thredsold_sum+=int(fp)
            tp_thredsold_sum+=int(tp)
            fn_thredsold_sum+=int(fn)
            tn_thredsold_sum+=int(tn)
        if threshold not in threshold_remain_list:
            if first_flag:
                if num != 0:
                    tpr_sum_List.append(tpr_sum / num)
                    fpr_sum_List.append(fpr_sum / num)
                    precision_sum_List.append(precision_sum / num)
            threshold_remain_list.append(threshold)
            tpr_sum = 0
            fpr_sum = 0
            precision_sum = 0
            num = 0
            first_flag = True

        if int(fp) == 0 and int(tn) == 0:
            continue
        elif int(tp) == 0 and int(fn) == 0:
            continue
        elif int(tp) == 0 and int(fp) == 0:
            continue
        else:
            tpr_sum += tpr(int(tp), int(fn))
            fpr_sum += fpr(int(fp), int(tn))
            precision_sum += precision(int(tp), int(fp))
            num += 1
    if num != 0:
        tpr_sum_List.append(tpr_sum / num)
        fpr_sum_List.append(fpr_sum / num)
        precision_sum_List.append(precision_sum / num)
    #计算面积
    roc_metric, prc_metric = roc_prc(csv_path)
    tpr_sum_List, fpr_sum_List, precision_sum_List=np.array(tpr_sum_List),np.array(fpr_sum_List),np.array(precision_sum_List)
    print(f"tpr:{tp_thredsold_sum/(tp_thredsold_sum+fn_thredsold_sum)}")
    print(f"fpr:{fp_thredsold_sum/(fp_thredsold_sum+tn_thredsold_sum)}")

    #ROC
    plt.subplot(1,2,1)
    plt.title("ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr_sum_List,tpr_sum_List,'-k',lw=1.5,label="AUC of Roc=%.3f"%roc_metric)
    plt.axis("square")
    plt.legend()
    #PRC
    plt.subplot(1,2,2)
    plt.title("PRC")
    plt.xlabel("Sensitivity")
    plt.ylabel("Precision")
    plt.plot(tpr_sum_List,precision_sum_List,'-b',lw=1.5,label="AUC of Prc=%.3f"%prc_metric)
    plt.axis("square")
    plt.legend()

    plt.tight_layout(pad=1.08)
    plt.ylim(ymin=0,ymax=1)
    plt.show()

#展示一个csv多个指标随epoch变化曲线图，均标准化
def show_loss(loss_csv,eval):
    data = pd.read_csv(loss_csv)
    epoch = data["epoch"][0:100]+1
    plt.title(f"{loss_csv}")
    plt.xlabel("Epoch")
    plt.ylabel("value")
    for each in eval:
        metric = np.array(data[each][0:100])
        for i, value in enumerate(metric):
            if value <= 0 or value >= 1:
                metric[i] = metric[i - 1]
        metric=normalization(metric)
        metric=moving_average(metric,10)
        plt.plot(epoch,metric)
        print(f"{each}:{metric.mean()}")
    plt.legend(eval)
    plt.show()
def show_val(loss_csv,val):
    data = pd.read_csv(loss_csv)
    metric = np.array(data[val])[0:100]
    for i,value in enumerate(metric):
        if value<=0 or value>=1:
            metric[i]=metric[i-1]
    epoch = data["epoch"][0:100]+1
    plt.title(f"{loss_csv}-{val}")
    plt.xlabel("Epoch")
    plt.ylabel("value")
    plt.plot(epoch,metric)
    print(metric.mean())
    plt.show()
def show_train_vs_valid(train_csv,valid_csv):
    metric=["loss_val","ROC"]
    train = pd.read_csv(train_csv)
    valid = pd.read_csv(valid_csv)
    epoch = train["epoch"]
    for i , m in enumerate(metric):
        t = train[m]
        v = valid[m]
        t = normalization(np.array(train[m]))
        t = moving_average(t,(15+i*20))
        v = normalization(np.array(valid[m]))
        v = moving_average(v,(15+i*20))
        plt.plot(epoch,t)
        plt.plot(epoch,v)

    # 设置新罗马字体
    plt.xlabel("Epoch", fontname="Times New Roman", fontsize=12)
    plt.ylabel("Value", fontname="Times New Roman", fontsize=12)

    # 设置图例并使用新罗马字体
    legend_labels = ["Train loss", "Validation loss", "Train ROC", "Validation ROC"]
    plt.legend(legend_labels)
    # 设置坐标轴刻度的字体为新罗马
    plt.xticks(fontname="Times New Roman", fontsize=10)
    plt.yticks(fontname="Times New Roman", fontsize=10)
    # plt.show()
    plt.savefig("early_stop.svg")
def normalization(x):
    out = (x-x.min())/(x.max()-x.min())
    return out
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re



if __name__ == '__main__':
    csv_path="./work_dir/congestion/130/"
    eval=["loss_val","NRMS","SSIM","ROC","PRC"]
    loss_csv = f"./work_dir/congestion/53/validation_loss.csv"
    train_csv = "./work_dir/congestion/53/train_loss.csv"
    valid_csv = "./work_dir/congestion/53/validation_loss.csv"
    show_train_vs_valid(train_csv,valid_csv)
    # show_loss(train_csv,eval)
    # show_val(loss_csv,"ROC")
    # show_roc(csv_path)
    # for  i in range(10,12):
    #     loss_csv = f"./work_dir/congestion/{i}/loss.csv"
    #     show_loss(loss_csv,["loss_val"])
