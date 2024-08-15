import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FastRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FastRCNN, self).__init__()

        self.features = torchvision.models.vgg16(pretrained=True).features
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes + 1)
        )
        self.bbox = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, (num_classes + 1) * 4)
        )

    def forward(self, x, rois):
        x = self.features(x)
        rois = torch.cat([rois[:, :1], rois[:, 1:] - rois[:, :1] + 1], dim=1)
        rois = self.roi_pool(F.roi_align(x, [rois], output_size=(7, 7)))
        rois = rois.view(rois.size(0), -1)
        cls_scores = self.classifier(rois)
        bbox_preds = self.bbox(rois)

        return cls_scores, bbox_preds