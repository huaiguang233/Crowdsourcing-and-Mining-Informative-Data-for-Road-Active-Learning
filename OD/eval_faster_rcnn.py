import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import OD.model.faster_rcnn.utils as utils
from OD.model.faster_rcnn.coco_eval import CocoEvaluator
from OD.model.faster_rcnn.coco_utils import get_coco_api_from_dataset
from OD.model.faster_rcnn.engine import _get_iou_types
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

def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in tqdm(metric_logger.log_every(data_loader, 10000, header)):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = [model(images)[0]]

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


    class_results = {}
    for idx, class_id in enumerate(coco_evaluator.coco_eval['bbox'].params.catIds):
        ap = coco_evaluator.coco_eval['bbox'].eval['precision'][:, :, idx, 0, 2].mean()  # AP@IoU=0.5:0.95, maxDet=100
        ar = coco_evaluator.coco_eval['bbox'].eval['recall'][:, idx, 0, 2].mean()  # AR@IoU=0.5:0.95, maxDet=100
        class_results[class_id] = {'AP': ap, 'AR': ar}

    torch.set_num_threads(n_threads)
    return coco_evaluator


def load_model(model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型参数
    model.to(device)
    return model


def evaluate_all_models(models_folder, data_loader, device):
    results = {}
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pth')]

    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        print(f"Evaluating model: {model_file}")
        model = load_model(model_path, device)

        coco_evaluator = evaluate(model, data_loader, device)
        results[model_file] = coco_evaluator.coco_eval['bbox'].stats  # 假设评估为bbox，存储结果
    return results


# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.train()  # 设置为训练模式以输出loss
#     total_loss = 0.0
#     num_batches = 0
#
#     for images, targets in data_loader:
#         images = list(img.to(device) for img in images)
#         targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
#
#         with torch.no_grad():  # 禁用梯度
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             total_loss += losses.item()
#             num_batches += 1
#
#     avg_loss = total_loss / num_batches
#     print(f"Validation Loss: {avg_loss:.4f}")
#     torch.set_num_threads(n_threads)
#
#     return avg_loss

def evaluate_all_models(models_folder, data_loader, device):
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pth')]
    loss_results = {}

    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        print(f"Evaluating model: {model_file}")
        model = load_model(model_path, device)
        avg_loss = evaluate(model, data_loader, device)
        loss_results[model_file] = avg_loss  # 记录每个模型的平均损失
    return loss_results



val_txt = 'H:/final_code/deepALplus-master/data/Waymo/val.txt'
dataset_test = MarkDataset(val_txt, transform_with_bboxes)
dataset_test = torch.utils.data.Subset(dataset_test, list(range(len(dataset_test))))
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, # num_workers=4,
    collate_fn=utils.collate_fn)


model_path = 'D:\\aliyundownload\\22'
device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = evaluate_all_models(model_path, data_loader_test, device_gpu)