import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import config
from arg.config import Paraser
from pred_model.build_model import build_model
import os
from PyQt5.QtCore import QLibraryInfo
from sklearn.metrics import mean_absolute_error,f1_score,roc_curve,precision_recall_curve
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
def vis(model):
    # palette = [[0, 0, 0], [192, 224, 224]]
    f1=0
    count=0
    with open("./files/CircuitNet-28_test.csv") as f:
        reader=csv.reader(f)
        for each in reader:
            feature_pth="../CircuitNet-28/congestion/" + each[0]
            label_pth = "../CircuitNet-28/congestion/" + each[1]
            feature=np.load(feature_pth)
            label=np.load(label_pth)
            input = feature.transpose(2,0,1).astype(np.float32)
            input = torch.tensor(input).cuda().unsqueeze(0)

            output = model(input)
            # output=torch.sum(output,dim=1,keepdim=True)
            prediction = output[0,:,:].cpu().data.numpy()
            prediction=prediction.transpose(1,2,0)
            # prediction=normlization(prediction)

            # prediction=np.reshape(prediction[:,:,1],(256,256,1))
            # cls,prediction=cal_cls_n_array(prediction,np.array([0.04]))

            # prediction = onehot_to_mask(prediction,palette)
            # print(mean_absolute_error(label.reshape(256,256),np.array(prediction).reshape(256,256)))
            threshold = 0.15
            binary_label = (label.reshape(256,256) >= threshold).astype(int)
            binary_prediction = (np.array(prediction).reshape(256,256) >= threshold).astype(int)
            f1+=f1_score(binary_label,binary_prediction,average='weighted')
            count+=1
            plt.subplot(2 , 2 , 1)
            plt.imshow(feature[:,:,2])
            plt.subplot(2, 2, 2)
            plt.imshow(label)
            # plt.imshow(binary_label)
            plt.subplot(2, 2, 3)
            plt.imshow(prediction)
            # plt.imshow(binary_prediction)
            plt.subplot(2,2,4)
            attention = model.get_attention_map_enc(input, 11).squeeze().cpu().detach().numpy().transpose(1, 2, 0)[:, :,2]
            plt.imshow(attention)
            plt.show()
            print(f"F1 Score:{f1/count}")
            # print(1)
            # plt.savefig("./test_result.png")
            # break
def normlization(predict):
    for i in range(predict.shape[0]):
        for k in range(predict.shape[1]):
            if predict[i][k][0] < 0:
                predict[i][k][0]=0
    return predict

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x
def main(model_path):
    argp = Paraser()
    arg = argp.parser.parse_args()
    cfg = config.load_config()
    arg_dict=vars(arg)
    arg_dict["test_mode"]=True
    # model = build_model(arg,cfg)
    # model_path="./work_dir/Segmodel.pth"
    # model.load_state_dict(torch.load(model_path)['model'])
    model =torch.load((model_path))
    vis(model)
if __name__ == "__main__":
    # main("../GAN_FCN/work_dir/congestion/17/fine_tune/model_iters_55ep.pth")
    # main("./work_dir/congestion/60/Segmodel_40ep.pth")#扩充两倍预训练
    # main("./work_dir/congestion/63/Segmodel_20ep.pth")#ever best
    # main("./work_dir/congestion/66/Segmodel_20ep.pth")
    # main("./work_dir/congestion/114/Segmodel_20ep.pth")#loss
    main("./work_dir/congestion/130/Segmodel_40ep.pth")#反卷积核
    #绘制每层注意力图
    # for i in range(11):
    #     plt.imshow(
    #         model.get_attention_map_enc(input, i).reshape(3, 257, 257).cpu().detach().numpy().transpose(1, 2, 0)[:, :,
    #         2])
    #     plt.show()