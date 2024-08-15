from roc import build_metric, build_roc_prc_metric
import numpy as np
import torch
from arg.config import Paraser
import os
import os.path as osp
import json
from datasets.build_dataset import build_dataset
from sklearn.metrics import f1_score
import time
argp = Paraser()
arg = argp.parser.parse_args()
arg_dict = vars(arg)
def test_model(model_dir_path,model_name):
    arg.arg_file = model_dir_path + "/arg.json"
    #arg_file 参数文件
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test']
    arg_dict['test_mode'] = True
    arg_dict['result_path'] = model_dir_path

    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    model_path =os.path.join(model_dir_path,model_name)
    print(model_path)
    model = torch.load(model_path)
    dataset=build_dataset(arg_dict)
    f1=0
    result=[]
    count=0
    save_path = osp.join(arg_dict['result_path'], 'test_result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for feature, label, label_path in dataset:
        input, target = feature.cuda(), label.cuda()
        prediction = model(input)
        for metric, metric_func in metrics.items():
            save_path = osp.join(arg_dict['result_path'], 'test_result')
            if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

        file_name = osp.splitext(osp.basename(label_path[0]))[0]
        save_path = osp.join(save_path, f'{file_name}.npy')
        output_final = prediction.float().detach().cpu().numpy()
        np.save(save_path, output_final)
        # threshold = arg.threashold
        # binary_label = ((label.reshape(256, 256).squeeze().cpu().detach().numpy()) >= threshold).astype(int)
        # binary_prediction = ((prediction.reshape(256, 256).squeeze().cpu().detach().numpy()) >= threshold).astype(int)
        # f1 += f1_score(binary_label, binary_prediction, average='weighted')
        count+=1

    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))
        result.append(avg_metric/len(dataset))
    roc_metric, prc_metric = build_roc_prc_metric(**arg_dict)
    print("\n===> AUC of ROC. {:.4f}".format(roc_metric))
    print("===> AUC of PR. {:.4f}".format(prc_metric))
    result.append(roc_metric)
    result.append(prc_metric)
    result.append(f1/count)
    return result

def test_per_epoch(model_dir_path,model,arg_dict,ann_file):
    arg_dict['ann_file'] = ann_file
    arg_dict['test_mode'] = True
    arg_dict['result_path'] = model_dir_path

    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    dataset=build_dataset(arg_dict)

    result=[]
    save_path = osp.join(arg_dict['result_path'], 'test_result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for feature, label, label_path in dataset:
        input, target = feature.cuda(), label.cuda()
        prediction = model(input)
        for metric, metric_func in metrics.items():
            save_path = osp.join(arg_dict['result_path'], 'test_result')
            if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

        file_name = osp.splitext(osp.basename(label_path[0]))[0]
        save_path = osp.join(save_path, f'{file_name}.npy')
        output_final = prediction.float().detach().cpu().numpy()
        np.save(save_path, output_final)

    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset)))
        result.append(avg_metric/len(dataset))
    roc_metric, prc_metric = build_roc_prc_metric(**arg_dict)
    print("\n===> AUC of ROC. {:.4f}".format(roc_metric))
    print("===> AUC of PR. {:.4f}".format(prc_metric))
    result.append(roc_metric)
    result.append(prc_metric)
    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False
    arg_dict['result_path'] = None
    os.system(f"rm -rf {save_path}")
    return result


# record_data = [arg_dict['Experiment_name'], arg_dict['order'], time.asctime(time.localtime()), 0, 0,
#                arg_dict['pre_max_iters'], arg_dict['fin_max_iters']]
# result = test_model("./work_dir/congestion/75/","Segmodel_20ep.pth")
# for each in result:
#     record_data.append(each)
# record_data.append(arg_dict["Experiment_note"])
# # 记录数据
# if arg_dict["task"] == "congestion_gpdl":
#     record = open("../record_congestion.csv", 'a', newline='')
# elif arg_dict["task"] == "drc_routenet":
#     record = open("../record_drc.csv", 'a', newline='')
# else:
#     record = open("../record_irdrop.csv", 'a', newline='')
# writer = csv.writer(record)
# writer.writerow(record_data)
# record.close()