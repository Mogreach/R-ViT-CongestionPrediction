import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns
import csv
from PyQt5.QtCore import QLibraryInfo
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from roc import roc_prc
from sklearn.metrics import f1_score

def plot9():
    # 生成示例数据
    data=np.load("../CircuitNet-28/congestion/feature/17-RISCY-a-1-c2-u0.7-m3-p4-f0.npy")
    # 获取图像的大小
    width = data.shape[0]
    height = data.shape[1]
    # 显示图像
    plt.figure(figsize=(10, 8))
    plt.imshow(data)
    # 将图像切分成9等份
    for i in range(1, 4):
        plt.axhline(i * height / 4, color='white', lw=10)
        plt.axvline(i * width / 4, color='white', lw=10)
    # plt.title('Image split into 9 parts')
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig("16等分.svg",dpi=300)
    # count=0
    # for j in range(4):
    #     for k in range(4):
    #         count+=1
    #         plt.imshow(data[j*64:(j+1)*64-1,k*64:(k+1)*64-1,:])
    #         plt.axis('off')
    #         plt.savefig(f"{count}.svg")

def plot_real_vs_fake():
    feature=np.load("../CircuitNet-28/congestion/feature/17-RISCY-a-1-c2-u0.7-m3-p4-f0.npy")
    label=np.load("../CircuitNet-28/congestion/label/17-RISCY-a-1-c2-u0.7-m3-p4-f0.npy")
    fake_feature = np.load("../CircuitNet-28/congestion/fake_feature/1000.npy")
    fake_label = np.load("../CircuitNet-28/congestion/fake_label/1000.npy")
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2)

    # Remove ticks and labels for all subplots
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Plot the real feature data in the first subplot
    axs[0, 0].imshow(feature)
    axs[0, 0].set_title('Real data')
    # axs[0, 0].set_ylabel('Feature')

    # Plot the fake feature data in the second subplot
    axs[0, 1].imshow(fake_feature)
    axs[0, 1].set_title('Fake data')

    # Plot the real label data in the third subplot
    axs[1, 0].imshow(label)
    # axs[1, 0].set_ylabel('Label')

    # Plot the fake label data in the fourth subplot
    axs[1, 1].imshow(fake_label)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.savefig("真实生成数据对比图.svg", format='svg', dpi=1200, bbox_inches='tight')
    # plt.show()
def tpr(tp, fn):
    return tp/(tp+fn)

def fpr(fp, tn):
    return fp/(fp+tn)

def precision(tp, fp):
    return tp/(tp+fp)
def plot_roc(csv_path):
    tpr_sum_List = []
    fpr_sum_List = []
    precision_sum_List = []
    threshold_remain_list = []
    count = 0
    num = 0
    tpr_sum = 0
    fpr_sum = 0
    precision_sum = 0
    fp_thredsold_sum = 0
    tp_thredsold_sum = 0
    fn_thredsold_sum = 0
    tn_thredsold_sum = 0
    first_flag = False
    csv_file = open(os.path.join(csv_path, "roc_prc.csv"), 'r')
    for line in csv_file:
        # 修复bug，csv隔行有回车，需剔除
        if line == '\n':
            continue
        threshold, idx, tn, fp, fn, tp = line.strip().split(',')
        if threshold == "0.1":
            fp_thredsold_sum += int(fp)
            tp_thredsold_sum += int(tp)
            fn_thredsold_sum += int(fn)
            tn_thredsold_sum += int(tn)
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
    # 计算面积
    # roc_metric, prc_metric = roc_prc(csv_path)
    tpr_sum_List, fpr_sum_List, precision_sum_List = np.array(tpr_sum_List), np.array(fpr_sum_List), np.array(
        precision_sum_List)
    return tpr_sum_List,fpr_sum_List,precision_sum_List
def plot_4roc():
    data = ["MAE", "MSE", "Huber", "Adaptive Huber"]

    # 创建一个大图，其中包含两个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for each in data:
        tpr_sum_List, fpr_sum_List, precision_sum_List = plot_roc("./roc_curve/" + each)

        # ROC
        axes[0].plot(fpr_sum_List, tpr_sum_List, '.-', lw=2, label=each)  # Increased linewidth
        axes[0].set_title("ROC", fontname="Times New Roman", fontsize=16)
        axes[0].set_xlabel("FPR", fontname="Times New Roman", fontsize=14)
        axes[0].set_ylabel("TPR", fontname="Times New Roman", fontsize=14)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        for label in axes[0].get_xticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(12)
        for label in axes[0].get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(12)

        # PRC
        axes[1].plot(tpr_sum_List, precision_sum_List, '.-', lw=2, label=each)  # Increased linewidth
        axes[1].set_title("PRC", fontname="Times New Roman", fontsize=16)
        axes[1].set_xlabel("TPR", fontname="Times New Roman", fontsize=14)
        axes[1].set_ylabel("Precision", fontname="Times New Roman", fontsize=14)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        for label in axes[1].get_xticklabels():
            label.set_fontname("Times New Roman")
            # label.set_fontsize(12)
        for label in axes[1].get_yticklabels():
            label.set_fontname("Times New Roman")
            # label.set_fontsize(12)

    # 添加图例
    # axes[0].legend(fontsize=12)
    # axes[1].legend(fontsize=12)
    axes[0].legend(loc='lower center', bbox_to_anchor=(0.7, 0.03))  # 添加 loc 参数
    axes[1].legend(loc='lower center', bbox_to_anchor=(0.7, 0.03))  # 添加 loc 参数
    # 调整布局以确保不会有重叠的部分
    plt.tight_layout()
    # plt.show()
    plt.savefig("ROC_PRC对比图.svg")

#绘制不同模型预测可视化对比图
def vis_model(model_path):
    data_path=["10006-zero-riscy-b-3-c2-u0.85-m2-p6-f1.npy","9782-zero-riscy-b-2-c5-u0.85-m1-p5-f0.npy",
               "9774-zero-riscy-b-2-c5-u0.8-m3-p3-f0.npy","10183-zero-riscy-b-3-c5-u0.75-m2-p4-f1.npy"]
    for i,path in enumerate(model_path):
        model = torch.load(path, map_location=torch.device('cpu'))
        for k in range(4):
            feature = np.load(f"../CircuitNet-28/congestion/feature/{data_path[k]}")
            label = np.load(f"../CircuitNet-28/congestion/label/{data_path[k]}")
            input = feature.transpose(2,0,1).astype(np.float32)
            input = torch.tensor(input).unsqueeze(0)
            output = model(input)
            prediction = output[0,:,:].cpu().data.numpy()
            prediction=prediction.transpose(1,2,0)
            plt.subplot(4, 6, (k*6)+1)
            # plt.imshow((label * 255).astype(np.uint8))
            plt.imshow(label,cmap='coolwarm')
            plt.axis('off')
            num = i+2+k*6
            plt.subplot(4, 6, num)
            plt.title(f"MAE = {(abs(prediction-label).sum()/(256*256)):.3f}", fontname="Times New Roman")
            plt.imshow(prediction,cmap='coolwarm')
            plt.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig("预测可视化.svg")
def plot_trade_off_time_n_acc():
    scatter = {"FCN" :[(0.0054,68.36),(0.0059,67.57),(0.0051,66.43),(0.0052,67.78),(0.0050,68.04)],
               "CGAN":[(0.0050,68.59),(0.0051,68.43),(0.00506,67.88),(0.00497,67.68),(0.00567,68.67)],
               "SegNet":[(0.0082,59.71),(0.0083,57.67),(0.0085,58.46),(0.0077,57.23),(0.0091,58.28)],
               "ConvNeXt":[(0.0198,61.93),(0.0203,60.67),(0.0200,61.43),(0.0194,62.31),(0.0259,62.04)],
               "Ours":[(0.0183,74.77),(0.0180,75.98),(0.0185,76.37),(0.0184,75.78),(0.0182,75.34)]}
    # 初始化一个图形和坐标轴  
    plt.figure(figsize=(10, 6))  # 设置图形大小  
    
    # 为每个模型绘制散点图，并指定不同的颜色和图标  
    markers = ['o', 's', '^', 'D', 'p']  # 选择不同的图标  
    colors = ['b', 'g', 'r', 'c', 'm']  # 选择不同的颜色  
    for i, (model, data) in enumerate(scatter.items()):  
        plt.scatter([x for x, _ in data], [y for _, y in data], label=model, marker=markers[i], color=colors[i])  
    
    # 添加图例  
    plt.legend(loc='upper right', ncol=2,fontsize=16)  # 调整图例位置，并设置列数以优化显示  
    
    # 添加坐标轴标签  
    plt.xlabel('Prediction time(s/sample)',fontsize=16)  # 根据实际情况替换为X轴标签  
    plt.ylabel('Accuracy(%)',fontsize=16)  # 根据实际情况替换为Y轴标签  
    
    # 显示图形  
    plt.show()
def plot_huber_loss():
    def huber_loss(y_true, y_pred, delta):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.where(is_small_error, squared_loss, linear_loss)

    # 设置不同的delta值
    deltas = [0.5, 1.0, 2.0, 5.0]
    y_true = 0  # 固定真实值
    y_pred = np.linspace(-10, 10, 400)  # 预测值范围

    plt.figure(figsize=(10, 6))

    # 为每个delta值绘制Huber loss曲线
    for delta in deltas:
        loss = huber_loss(y_true, y_pred, delta)
        plt.plot(y_pred, loss, label=f'Delta = {delta}')

    plt.xlabel('Prediction Error (y_true - y_pred)')
    plt.ylabel('Huber Loss')
    plt.title('Huber Loss for Different Delta Values')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # vis_model(["./work_dir/model_cmp/FCN.pth","./work_dir/model_cmp/CGAN.pth","./work_dir/model_cmp/SegNet.pth","./work_dir/model_cmp/ConvNexT.pth","./work_dir/model_cmp/ViT.pth"])
    # plot_real_vs_fake()
    # plot9()
    # plot_4roc()
    # plot_trade_off_time_n_acc()
    plot_huber_loss()