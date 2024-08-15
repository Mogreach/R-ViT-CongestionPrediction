import numpy as np
import csv
import torch
from .fliter import flit
#7078
#generate fake data into train set of drc_routenet
#用于生成数据，输入参数为argparser参数字典，GAN训练序号
class Generate_data():
    def __init__(self,arg_dict):
        self.arg=arg_dict
        self.modelf_path = f"./pretrain_model/{arg_dict['task']}/GAN_f.pth"
        self.modell_path = f"./pretrain_model/{arg_dict['task']}/GAN_l.pth"
        self.pred_path = f"./pretrain_model/{arg_dict['task']}/predict.pth"
        self.filename=arg_dict['ann_file_train']
        self.multiple=self.arg["multiple"]
    def main(self):
        Gf = torch.load(self.modelf_path)
        Gl = torch.load(self.modell_path)
        pred = torch.load(self.pred_path)
        feature_path = []
        label_path = []
        fake_feature_path = []
        fake_label_path = []
        with open(self.filename) as f:
            reader = csv.reader(f)
            test = open("./files/enrich.csv", 'w', newline='')
            fake = open("./files/fake.csv", 'w', newline='')
            writer = csv.writer(test)
            fake_writer = csv.writer(fake)
            #read real data path
            for row in reader:
                feature_path.append(row[0])
                label_path.append(row[1])
            #create fake data path
            for c in range(int(self.multiple*7078)):
                fake_feature_path.append(f"fake_feature/{c}.npy")
                fake_label_path.append(f"fake_label/{c}.npy")
            for k in range(0,len(feature_path)):
                writer.writerow([feature_path[k],label_path[k]])
            #generate fake data
            for i in range(0,len(fake_feature_path)):
                writer.writerow([fake_feature_path[i],fake_label_path[i]])
                fake_writer.writerow([fake_feature_path[i],fake_label_path[i]])
                flag=1
                while flag:

                    z = torch.randn(1, self.arg["z_dim"], 1, 1).cuda()
                    fake_feature = Gf(z)
                    fake_label = Gl(fake_feature)
                    pred_fake_label = pred(fake_feature)
                    #tensor 1 9 256 256 -> numpy 9 256 256
                    fake_feature = fake_feature.cpu().data.squeeze(0).numpy()
                    fake_feature = fake_feature.transpose(1, 2, 0)
                    fake_label = fake_label.cpu().data.squeeze(0).numpy()
                    fake_label = fake_label.transpose(1, 2, 0)
                    pred_fake_label = pred_fake_label.cpu().data.squeeze(0).numpy().transpose(1, 2, 0)

                    flag=flit(fake_label,pred_fake_label)


                np.save(self.arg['dataroot']+fake_feature_path[i], fake_feature)
                np.save(self.arg['dataroot']+fake_label_path[i], fake_label)
            f.close()
            fake.close()
            test.close()
        torch.cuda.empty_cache()
