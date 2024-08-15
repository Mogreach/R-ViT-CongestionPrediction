import os
import numpy as np
import csv
"""
match函数：
功能：判断给定数值，与（0，1）四等分的阈值比较，返回一组向量，反映该数值高于哪个阈值
输入：数值 , 阈值向量（由小往大排列）
输出：1x5向量
说明：向量组划分阈值依次为0.25，0.5，0.75，可划分为4个区间，0表示不属于该区间，1表示属于该区间
     向量组第一个表示是否为0，若数值为0，则返回[1,0,0,0,0]
     如：若给定0.8，则返回[0,0,0,0,1]
"""
def match(pixel,threasold):
    result=np.zeros(len(threasold)+2)
    if pixel == 0:
        result[0]=1
        return result
    if pixel >= threasold[-1]:
        result[-1] = 1
        return result
    for i,thre in enumerate(threasold):
        if pixel < thre:
            result[i+1]=1
            return result


"""
cal_cls_n_array函数：
功能：一个热点图样本，历遍每个像素，通过match函数，计算各自的分类向量，再累加，返回该样本分类向量;同时根据给定阈值，得出对应的分类图
输入：（HxWx1）数组 ；（1x3）数组；(1x3)阈值数组
输出：（1x5）向量组 ；（3xHxW）矩阵
说明：分类向量为[a,b,c,d,e]，代表该样本共有a个像素=0，b个像素大于0但小于0.25，c个像素大于等于0.25但小于0.5...
     one-hot矩阵，若大于等于阈值则为1，小于则为0.
"""
def cal_cls_n_array(label,threasold):

    threasold_num=len(threasold)
    label_cls=np.zeros(threasold_num+2)
    label_onehot=np.zeros((label.shape[0],label.shape[1],threasold_num+1))

    for i in range(label.shape[0]):
        for k in range(label.shape[1]):
            label_cls+=match(label[i][k][0],threasold)

            #小于0.01的one-hot编码图
            if label[i][k][0] < 0.1:
                label_onehot[i][k][0] = 1

            for j,thre in enumerate(threasold):
                if label[i][k][0] >= thre:
                    label_onehot[i][k][j+1]=1
    return label_cls,label_onehot
"""
main函数：
功能：将标签图按阈值分割多个（thresold_len+1）独立的one-hot编码图，并记录每个样本像素的分布情况，并保存文件
"""
def main():
    data_root="../CircuitNet/DRC/"
    thresold = np.array([0.1])
    channels = len(thresold)+1
    save_pth = data_root + f"oh{channels}_label"
    cls_save_pth = data_root + "cls_statistics"
    cls_head = ["样本名称","阈值=0","区间1"]

    if not os.path.exists(cls_save_pth):
        os.mkdir(cls_save_pth)
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)

    cls_name=""
    for i,th in enumerate(thresold):
        cls_name+="_"+str(th)
        cls_head.append(f"区间{i+2}")


    file=open(f"./files/oh{channels}_train.csv",'w', newline='')
    cls_file=open(cls_save_pth+f"/{cls_name}.csv",'w', newline='')
    writer=csv.writer(file)
    writer_cls=csv.writer(cls_file)
    writer_cls.writerows(cls_head)
    with open("./files/train.csv") as f:
        reader=csv.reader(f)
        for each in reader:
            cls=[each[1]]
            #写入one-hot编码图的csv读取文件，仅在label路径作改动
            oh_pth=[each[0],f"oh{channels}_"+each[1]]
            label_pth = data_root + each[1]
            save_pth = data_root + f"oh{channels}_"+each[1]
            label=np.load(label_pth)
            #获取one-hot编码图，及像素取值相关统计
            label_cls,label_onehot=cal_cls_n_array(label,thresold)
            for c in label_cls:
                cls.append(int(c))
            writer.writerow(oh_pth)
            writer_cls.writerow(cls)
            np.save(save_pth,label_onehot)
    file.close()
    cls_file.close()
            #可视化
            # plt.imshow(label_onehot[:,:,1]+label_onehot[:,:,0])
            # plt.show()



if __name__ == '__main__':
    main()