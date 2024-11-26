import json
import os
import random

import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from OD.model.faster_rcnn.engine import train_one_epoch, evaluate
import OD.model.faster_rcnn.utils as utils
# 解析YOLO的annotation文件，将YOLO格式的bbox转换为[x_min, y_min, x_max, y_max]


def train_faster_rcnn(epochs, round, save_data_dir, yaml_path, dataset_name, device_gpu, AL_method):

    def set_seed(seed):
        seed_n = seed
        g = torch.Generator()
        g.manual_seed(seed_n)
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        # torch.backends.cudnn.deterministic=True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        # torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed_n)

    set_seed(round)


    def load_yaml_config(yaml_file):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    yolo_config = load_yaml_config(yaml_path)
    train_txt = yolo_config['train']
    val_txt = yolo_config['val']
    num_classes = yolo_config['nc']

    class MarkDataset(torch.utils.data.Dataset):
        def __init__(self, list_file, transforms=None):
            """
            list_file: 包含图片文件路径的txt文档路径
            transforms: 数据增强与变换
            """
            self.transforms = transforms

            # 读取文件中的图片路径
            with open(list_file, 'r') as f:
                self.img_paths = [line.strip() for line in f.readlines()]

            # 确定每个图片的标签路径（假设标签文件名与图片文件名一致）
            self.label_paths = [os.path.join(os.path.dirname(img_path).replace('images', 'labels'),
                                             os.path.basename(img_path).replace('.jpg', '.txt'))
                                for img_path in self.img_paths]
            # 过滤掉没有对应标注的图片
            self.img_paths = [img_path for img_path, label_path in zip(self.img_paths, self.label_paths) if
                              os.path.exists(label_path)]
            self.label_paths = [label_path for label_path in self.label_paths if os.path.exists(label_path)]

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            label_path = self.label_paths[idx]

            img = Image.open(img_path).convert("RGB")

            boxes = []
            labels = []

            # 读取YOLO标签
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    label_data = line.strip().split()
                    label = int(label_data[0])  # 类别ID
                    x_center, y_center, width, height = map(float, label_data[1:])

                    # 将YOLO的归一化坐标转换为像素坐标
                    img_width, img_height = img.size
                    xmin = (x_center - width / 2) * img_width
                    ymin = (y_center - height / 2) * img_height
                    xmax = (x_center + width / 2) * img_width
                    ymax = (y_center + height / 2) * img_height

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)

            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = idx
            target["area"] = area
            target["iscrowd"] = iscrowd

            # 应用图片和边界框的变换
            img, target = self.transforms(img, target)

            return img, target

        def __len__(self):
            return len(self.img_paths)

    def get_object_detection_model(num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def get_transform():
        return T.Compose([
            T.Resize((640, 640)),  # 将图片缩放到320×320
            T.ToTensor()  # 将图片转换为Tensor
        ])

    def transform_with_bboxes(img, target):
        transform = get_transform()
        # 获取原图尺寸
        original_width, original_height = img.size

        # 执行图片的变换（缩放和转换为Tensor）
        img = transform(img)

        # 计算宽高缩放比例
        scale_x = 640 / original_width
        scale_y = 640 / original_height

        # 缩放边界框
        if "boxes" in target:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x  # 缩放 x 坐标
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y  # 缩放 y 坐标
            target["boxes"] = boxes

        return img, target


    # use our dataset and defined transformations
    dataset = MarkDataset(train_txt, transform_with_bboxes)
    dataset_test = MarkDataset(val_txt, transform_with_bboxes)

    # split the dataset in train and test set
    # 我的数据集一共有492张图，差不多训练验证4:1
    dataset = torch.utils.data.Subset(dataset, list(range(len(dataset))))
    dataset_test = torch.utils.data.Subset(dataset_test, list(range(len(dataset_test))))

    # define training and validation data loaders
    # 在jupyter notebook里训练模型时num_workers参数只能为0，不然会报错，这里就把它注释掉了
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True,  # num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,  # num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_object_detection_model(num_classes)  # 或get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device_gpu)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader, device_gpu, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device_gpu)
    save_path = os.path.join(save_data_dir, f'faster_rcnn_trained_epoch_{AL_method}_{round}.pth')
    torch.save(model.state_dict(), save_path)