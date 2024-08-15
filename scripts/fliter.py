def cal_mse(a,b):
    value=(a-b)**2
    return value
def cal_mae(a,b):
    value=abs(a-b)
    return value

def cal_each(label,pred,w,h):
    mse_val = 0  # 存储一张图像的所有像素点的误差平方值
    mae_val = 0
    for i in range(w):
        for k in range(h):
            mse_val += cal_mse(label[i][k], pred[i][k])
            mae_val += cal_mae(label[i][k], pred[i][k])
    return mse_val,mae_val

def trans2np(t):
    n = t.cpu().data.squeeze(0).numpy()
    n = n.transpose(1, 2, 0)
    return n

def flit(fake_label,pred_fake_label):
    # mae_criteria=0.0027451335581061176+1.75*0.008369182133137408    #drc
    mae_criteria = 0.025451335581061176 + 1.75 * 0.008369182133137408
    # mae_criteria = 0.1                                          #congestion

    (width, height,c) = fake_label.shape
    mse_val,mae_val=cal_each(fake_label,pred_fake_label,width,height)

    mse=mse_val/fake_label.size
    mae=mae_val/fake_label.size

    print(f"MSE平均值：{mse}")
    print(f"MAE平均值：{mae}")

    if mae<=mae_criteria:
        return 0
    else:
        return 1