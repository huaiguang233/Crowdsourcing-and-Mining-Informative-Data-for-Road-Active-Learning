import copy
import json
import os
import random
import shutil
import time
from types import MethodType
import torch.nn.functional as F

import torchvision
import yaml
from sklearn.metrics import pairwise_distances, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN
from torchvision.ops import box_iou
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.spatial.distance import euclidean, cdist
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from OD.faster_rcnn_train_test import train_faster_rcnn
from OD.model.train_RetinaNet import train_retinanet
from OD.yolo_modify import _predict_once, non_max_suppression
from submodular_maximization import FacilityLocation_with_M
import torchvision.transforms as T
import faster_rcnn_modify



def init_train_txt(folder_path, output_txt):
    # 打开输出的txt文件
    with open(output_txt, 'w') as f:
        # 遍历文件夹中的所有文件
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 获取文件的绝对路径
                file_path = os.path.join(root, file)
                # 将文件的绝对路径和文件名写入txt
                f.write(f"{file_path}\n")


def append_index_to_txt(filenames, output_txt):
    # 打开txt文件，模式为'a'表示追加内容
    with open(output_txt, 'a') as f:
        # 遍历文件名列表
        for filename in filenames:
            f.write(f"{filename}\n")


def clear_txt(file_path):
    # 以写模式打开文件，相当于清空文件
    with open(file_path, 'w') as f:
        pass  # 不写入任何内容，文件会被清空


class CustomDataset(Dataset):
    def __init__(self, txt_file, names, transforms=None):
        self.txt_file = txt_file
        self.transforms = transforms
        self.names = names
        with open(txt_file, 'r') as f:
            self.image_paths = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图片路径
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # 读取对应的标注文件路径，假设标注文件和图片文件名称相同
        label_path = img_path.replace(".jpg", ".xml").replace("images", "Annotations")

        # 检查标注文件是否存在
        if not os.path.exists(label_path):
            # 如果找不到标注文件，返回None并在外部处理
            return None

        # 加载标注信息
        target = self.load_annotation(label_path)

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def load_annotation(self, annotation_path):
        boxes = []
        labels = []

        # 解析 XML 文件
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # 获取文件中的所有对象
        for obj in root.findall('object'):
            # 获取边界框信息
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

            # 获取类别标签
            label = obj.find('name').text
            labels.append(self.class_name_to_id(label))  # 假设有一个函数将类别名转为ID

        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

    def class_name_to_id(self, class_name):
        # 将类别名转换为ID的示例函数，需根据你的类别进行调整
        class_dict = {name: index for index, name in enumerate(self.names)}
        return class_dict.get(class_name, 0)  # 默认返回0（背景类）


def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])


def compute_map(predictions, targets, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = [False] * len(gt_boxes)

        for i, (iou_row, pred_label) in enumerate(zip(ious, pred_labels)):
            if pred_scores[i] < 0.5:  # 忽略低置信度的预测
                continue

            # 寻找与该预测框IoU最大的真实框
            max_iou, max_idx = torch.max(iou_row, dim=0)

            if max_iou >= iou_threshold and not matched_gt[max_idx] and gt_labels[max_idx] == pred_label:
                true_positives += 1
                matched_gt[max_idx] = True
            else:
                false_positives += 1

        false_negatives += len(gt_boxes) - sum(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

# def fasterrcnn_resnet50_fpn_feature(num_classes=91, pretrained_backbone=True, **kwargs):
#     # Wrap the backbone with FPN
#     backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained_backbone).features
#
#     # Define the layers to return and the corresponding in_channels
#     return_layers = {
#         '0': '0',  # Initial layer
#         '3': '1',  # First block
#         '6': '2',  # Second block
#         '11': '3',  # Last block
#     }
#
#     # Specify the input channels for the layers returned by MobileNet V3
#     in_channels_list = [16, 24, 40, 96]  # Corresponding to the layers above
#     out_channels = 256  # Output channels for FPN
#
#     # Wrap the backbone with FPN
#     backbone = BackboneWithFPN(backbone, return_layers=return_layers,
#                                in_channels_list=in_channels_list, out_channels=out_channels)
#
#     # backbone = resnet_fpn_backbone('resnet18', pretrained_backbone)
#     model = FRCNN_Feature(backbone, num_classes, **kwargs)
#     return model


def yolo_train(model_type, yaml_path, epochs, round, save_data_dir, dataset_name, device_gpu, AL_method):
    if model_type == 'yolo':
        yolo_model = YOLO('yolov8s.pt')
        # 开始训练
        yolo_model.train(data = yaml_path, epochs=epochs, batch=32, imgsz=640, device=device_gpu, workers = 0, verbose=False, val=False)
        # 保存和读取训练好的模型
        save_path = f'{save_data_dir}\\yolov8s_trained_epoch_{round}.pt'
        yolo_model.save(save_path)
        # results = yolo_model.val()
        # if hasattr(results, 'confusion_matrix'):
        #     cm = results.confusion_matrix.matrix  # 获取混淆矩阵数据
        #     class_accuracies = cm.diagonal() / cm.sum(axis=1)
        #
        #     for i, acc in enumerate(class_accuracies):
        #         print(f"类别 {i} 的准确率: {acc:.2f}")
        # else:
        #     print("Confusion matrix 不可用")


    if model_type == 'faster_rcnn':
        train_faster_rcnn(epochs, round, save_data_dir, yaml_path, dataset_name, device_gpu, AL_method)

    if model_type == 'RetinaNet':
        # 使用coco格式的标注
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        # 访问path和train
        train_path = config['train']
        val_path = config['val']
        nc = config['nc']
        names = config['names']
        root_path = config['path']

        train_retinanet(epochs, train_path, val_path, nc, root_path, dataset_name)


def get_top_n_models(model_type, folder_path, n=3):
    """
    Get the top N models with the highest epoch from the given folder.

    Args:
    folder_path (str): Path to the folder containing the models.
    n (int): Number of top models to retrieve.

    Returns:
    list: A list of the top N model file paths, sorted by epoch in descending order.
    """
    model_files = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        if model_type == 'faster_rcnn':
            if file_name.startswith(f'faster_rcnn_trained') and file_name.endswith('.pth'):
                # Extract the epoch number from the file name
                epoch_num = int(file_name.split('_')[-1].split('.')[0])
                model_files.append((epoch_num, os.path.join(folder_path, file_name)))
        if model_type == 'yolo':
            if file_name.startswith(f'yolov8s_trained_epoch_') and file_name.endswith('.pt'):
                # Extract the epoch number from the file name
                epoch_num = int(file_name.split('_')[-1].split('.')[0])
                model_files.append((epoch_num, os.path.join(folder_path, file_name)))

    # Sort the models by epoch in descending order and get the top N
    model_files = sorted(model_files, key=lambda x: x[0], reverse=True)[:n]

    # Return only the file paths of the top N models
    return [file_path for _, file_path in model_files]


def get_object_features(feat_list, idxs):
    obj_feats = []
    for fid, feat in enumerate(feat_list):
        if fid == 0:
            temp = feat.permute(0, 2, 3, 1).reshape(-1, 128, feat.shape[1] // 128).mean(dim=-1)
            temp_resized = F.interpolate(temp.unsqueeze(1), size=512, mode='linear', align_corners=False).squeeze(1)
            obj_feats.append(temp_resized)

        if fid == 1:
            temp = feat.permute(0, 2, 3, 1).reshape(-1, 256, feat.shape[1] // 256).mean(dim=-1)
            temp_resized = F.interpolate(temp.unsqueeze(1), size=512, mode='linear', align_corners=False).squeeze(1)
            obj_feats.append(temp_resized)

        if fid == 2:
            obj_feats.append(feat.permute(0, 2, 3, 1).reshape(-1, 512, feat.shape[1] // 512).mean(dim=-1))
    obj_feats = torch.cat(obj_feats, dim=0)
    return obj_feats[idxs]


def detect_objects(model_type, model, image_path, device_gpu):
    """
    Function to perform object detection using a YOLOv5 model and extract features of the detected objects using ResNet18.

    Args:
    yolov5_model_path (str): Path to the YOLOv5 model.
    image_path (str): Path to the image file.

    Returns:
    list: A list of dictionaries containing detection results with bbox, class, confidence score, and features.
    """

    detection_results = []
    img = cv2.imread(image_path)
    # Perform detection using the YOLO model

    if model_type == 'yolo':
        model.model._predict_once = MethodType(_predict_once, model.model)
        _ = model(source="1.jpg", save=False, embed=[15, 18, 21, 22], verbose=False)
        prepped = model.predictor.preprocess([img])
        result = model.predictor.inference(prepped)
        results, idxs = non_max_suppression(result[-1][0], in_place=False)
        obj_feats = get_object_features(result[:3], idxs[0].tolist())

        for result in results:
            for idx, instance in enumerate(result):
                bbox = instance[:4].tolist()  # Bounding box [x_min, y_min, x_max, y_max]
                cls = int(instance[5].item())  # Class ID
                conf = instance[4].item()  # Confidence score
                classconf = instance[6:].cpu().numpy()
                feature = obj_feats[idx].cpu().numpy()
                detection_results.append({
                    "bbox": bbox,
                    "class": cls,
                    "confidence": conf,
                    "classconf": classconf,
                    "embedding": feature,
                })


    if model_type == 'faster_rcnn':
        model.eval()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 进行图像的预处理，如转换为tensor并调整大小
        transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])
        img_tensor = transform(img_rgb).unsqueeze(0)  # 扩展维度，因为模型期望输入batch格式
        # 将图像移到设备上
        img_tensor = img_tensor.to(device_gpu)
        # 进行推理
        with torch.no_grad():
            outputs = model(img_tensor)
        # 处理模型输出，提取预测边界框、标签和置信度
        output = outputs[0]
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        embeddings = output['embedding'].cpu().numpy()
        class_logits = output['class_logits'].cpu().numpy()

        threshold = 0.2
        filtered_boxes = boxes[scores > threshold]
        filtered_labels = labels[scores > threshold]
        filtered_scores = scores[scores > threshold]
        filtered_embeddings = embeddings[scores > threshold]
        filtered_class_logits = class_logits[scores > threshold]
        for box, label, score, embedding, class_logit in zip(filtered_boxes, filtered_labels, filtered_scores, filtered_embeddings, filtered_class_logits):
            detection_results.append({
                "bbox": box.tolist(),
                "class": label,
                "confidence": score,
                "classconf": class_logit,
                "embedding": embedding,
            })
    return detection_results

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
    box1 (list): Bounding box [x_min, y_min, x_max, y_max].
    box2 (list): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    float: IoU score.
    """
    box1_tensor = torch.tensor([box1])
    box2_tensor = torch.tensor([box2])

    iou = box_iou(box1_tensor, box2_tensor).item()
    return iou

def pre_select_bbox(det_results, iou_ = 0.5):
    """
    Filter detection results across multiple models by IoU threshold and remove duplicate boxes across models.

    Args:
    all_detections (list): List of detection results from different models.
    iou_threshold (float): IoU threshold to filter out overlapping boxes.

    Returns:
    list: Filtered detection results with bbox, class, confidence, and the model from which the detection came.
    """
    final_detections = []

    # Iterate through models
    for model_idx, detections in enumerate(det_results):
        for detection in detections:
            bbox = detection['bbox']
            cls = detection['class']
            conf = detection['confidence']
            classconf = detection['classconf']
            embedding = detection['embedding']

            should_add = True

            # Compare only with detections from previous models
            for prev_model_idx, prev_detections in enumerate(final_detections):
                prev_bbox = prev_detections['bbox']
                prev_cls = prev_detections['class']
                prev_conf = prev_detections['confidence']

                # Check if they belong to the same class and IoU exceeds the threshold
                if cls == prev_cls:
                    iou = compute_iou(bbox, prev_bbox)
                    if iou > iou_:
                        # Keep the box with higher confidence
                        if conf > prev_conf:
                            # Update the existing detection with higher confidence bbox
                            prev_detections['bbox'] = bbox
                            prev_detections['confidence'] = conf
                            prev_detections['model'] = model_idx
                            prev_detections['classconf'] = classconf
                            prev_detections['embedding'] = embedding
                        should_add = False
                        break

            # If it's not redundant, add the new detection
            if should_add:
                final_detections.append({
                    "bbox": bbox,
                    "class": cls,
                    "confidence": conf,
                    "classconf": classconf,
                    "model": model_idx,
                    "embedding": embedding
                })

    return final_detections

def calculate_js_divergence(classconf1, classconf2):
    """
    Calculate Jensen-Shannon (JS) divergence between two class confidence distributions.

    Args:
    classconf1 (np.array): Class confidence distribution for detection 1.
    classconf2 (np.array): Class confidence distribution for detection 2.

    Returns:
    float: JS divergence score.
    """
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(classconf1, classconf2)

def calculate_detection_scores(final_detections, all_detections, iou_):
    """
    Calculate the M score for each detection in final_detections by comparing with original all_detections,
    while removing identical bounding boxes from the comparisons.

    Args:
    final_detections (list): List of filtered detection results.
    all_detections (list): Original list of detections from all models.
    iou_threshold (float): IoU threshold for matching detections.

    Returns:
    list: Updated final_detections with added M scores.
    """
    for final_det in final_detections:
        matched = False  # To track if there is a match with IoU > iou_threshold
        M = 0.0  # Initialize M value for current final detection
        matched_count = 0

        # Compare the final detection with all the original detections, excluding identical boxes
        for idx, model_detections in enumerate(all_detections):
            for det in model_detections:
                # Skip identical boxes (if they are exactly the same in terms of coordinates)
                if final_det["bbox"] == det["bbox"]:
                    continue  # Skip comparison for identical detections

                # Compute IoU
                iou = compute_iou(final_det["bbox"], det["bbox"])
                if (iou > iou_) and (final_det["class"] == det["class"]):  # If IoU exceeds the threshold
                    js1 = calculate_js_divergence(final_det.get("classconf", []), det.get("classconf", []))
                    M += 2**(idx) * ((1 - iou) + js1) + (1 - final_det["confidence"])
                    matched_count += 1
                    matched = True  # Found a match with IoU > iou_threshold

        # If no match was found with IoU > iou_threshold, apply the alternative M value
        if not matched:
            M = 2 - final_det["confidence"]
        else:
            # Calculate average of M for the matched detections
            M = M / matched_count if matched_count > 0 else M

        # Save the M score in the final detection
        final_det["M_score"] = M

    return final_detections


def calculate_embedding_distance(embedding1, embedding2):
    """
    Calculate the Euclidean distance between two embeddings.

    Args:
    embedding1 (np.array): Embedding of the first bounding box.
    embedding2 (np.array): Embedding of the second bounding box.

    Returns:
    float: Euclidean distance between two embeddings.
    """
    return euclidean(embedding1, embedding2)

def greedy_select_images(all_image_results, n=10, lambda1=0.5):
    image_files = list(all_image_results.keys())
    num_images = len(image_files)

    # Collect all embeddings for each image
    image_bboxes = {image: np.array([bbox['embedding'] for bbox in bboxes]) for image, bboxes in
                    all_image_results.items()}

    image_scores = np.array([sum(bbox['M_score'] for bbox in bboxes) for image, bboxes in all_image_results.items()])
    # Initialize distance matrix
    distance_matrix = np.zeros((num_images, num_images))

    # Compute pairwise distances between images using cdist for batch processing
    for i in range(num_images):
        for j in range(i + 1, num_images):
            if i != j:
                bboxes1 = image_bboxes[image_files[i]]
                bboxes2 = image_bboxes[image_files[j]]
                # Use cdist to compute pairwise distances between bounding boxes
                distances = cdist(bboxes1, bboxes2, metric='euclidean')
                # Sum of distances between all bounding boxes
                distance_matrix[i][j] = distance_matrix[j][i] = np.sum(distances)

    selected_indices = facility_location_greedy(distance_matrix, image_scores, n, lambda1)
    selected_images = [image_files[i] for i in selected_indices]
    return selected_images


def facility_location_greedy(distance_matrix, scores, num_select=5, lambda1=0.5):
    # 一共有多少张图片
    num_images = distance_matrix.shape[0]
    selected = []

    remaining_indices = list(range(num_images))

    # 选评分最大的图像
    selected.append(np.argmax(scores))
    # 初始化最小距离数组
    max_dis_to_selected = distance_matrix[selected[0]]
    # 在原来的距离数据中删除筛选过的数据点
    distance_matrix = np.delete(distance_matrix, selected[-1], 0)
    # 更新remaining_indices，保证后续的映射正常
    remaining_indices.remove(selected[-1])
    # 贪心的选择
    for _ in range(num_select-1):
        # 假设每一个点加入之后，带来的距离减小程度
        temp = np.minimum(distance_matrix, max_dis_to_selected)
        # 输出每一个数据点加入后的距离增益，减少了多少距离
        gain_distance = np.sum(max_dis_to_selected) - temp.sum(axis=1)

        # 综合考虑相似度增益和得分
        gain = lambda1 * gain_distance + (1-lambda1) * scores[remaining_indices]

        # print(lambda1)
        # print("gain_distance")
        # print(lambda1 * gain_distance)
        # print("scores[remaining_indices]")
        # print((1-lambda1) * scores[remaining_indices])

        # 贪心的选择增益最高的数据点
        id = np.argmax(gain)

        # 把id映射回原来列表的索引
        selected.append(remaining_indices[id])

        # 选出来后在原相似度矩阵中删除选择的数据点对应的数据
        distance_matrix = np.delete(distance_matrix, id, 0)

        # 更新索引映射
        remaining_indices.pop(id)
        # 更新最大相似度矩阵
        max_dis_to_selected = temp[id]

    return selected

def greedy_select_images_fal(all_image_embedding, n=10):
    distances = pairwise_distances(all_image_embedding, metric="euclidean")
    M = 1 / (1 + 0.01 * distances)
    index = np.arange(0, 200)

    M, M_max = create_M_max_M(M, all_image_embedding)

    sampling_policy = FacilityLocation_with_M(
        M, M_max, index, n
    )
    select_final = sampling_policy.sample_caches("Interactive")

    return select_final


def k_center_greedy(all_image_embeddings, n):
    # all_image_embeddings: [num_images, embedding_dim] 的张量
    num_images = all_image_embeddings.size(0)

    # 随机选择第一个中心点
    centers = [np.random.randint(0, num_images)]

    min_distances = torch.cdist(all_image_embeddings[centers], all_image_embeddings).squeeze(0)

    # 迭代选择剩下的 n-1 个中心点
    for _ in range(n - 1):
        # 找到距离当前中心点最远的点
        new_center = torch.argmax(min_distances).item()
        centers.append(new_center)

        # 计算新中心点到其他点的距离，并更新最小距离
        new_distances = torch.cdist(all_image_embeddings[new_center].unsqueeze(0), all_image_embeddings).squeeze(0)
        min_distances = torch.minimum(min_distances, new_distances)

    return centers

def create_M_max_M(M, all_image_embedding):
    obs_mask = list(np.arange(len(all_image_embedding) - 190, len(all_image_embedding)))
    #得到观察样本在所有样本中的索引
    base_mask = list(np.arange(0, len(all_image_embedding) - 190))
    #得到在云端服务器的样本在所有样本中的索引
    M_max = np.max(M[obs_mask][:, base_mask], axis=1).reshape(-1, 1)
    # 得到一个列向量，具体来说，对于每个观测数据，
    # M_max的相应元素是 观测数据与所有base数据之间距离的最大值
    M = M[np.ix_(obs_mask, obs_mask)]

    return M, M_max


class Dual_dpp():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device
        self.covariance_inv = None
        self.cov_inv_scaling = 1000

    def set_dim(self, dim):
        self.covariance_inv = self.cov_inv_scaling * torch.eye(dim).to(self.gpu_device)

    def inf_replace(self, mat):
        mat[torch.where(torch.isinf(mat))] = torch.sign(mat[torch.where(torch.isinf(mat))]) * np.finfo('float32').max
        return mat

    def dual_dpp(self, samps):
        samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        u_norm = 0
        samps = torch.tensor(samps, dtype=torch.float32).to(self.gpu_device)
        for i, u in enumerate(samps):
            u = u.view(-1, 1)
            u_norm += torch.abs(u.t() @ self.covariance_inv @ u).item()
        return u_norm/samps.shape[0]

    def update_covariance_inv(self, samps):
        samps = samps.reshape((samps.shape[0], 1, samps.shape[1]))
        rank = samps.shape[-2]
        samps = samps.clone().detach().to(self.gpu_device).float()

        for i, u in enumerate(samps):

            inner_inv = torch.inverse(torch.eye(rank).to(self.gpu_device) + u @ self.covariance_inv @ u.t())
            inner_inv = self.inf_replace(inner_inv)
            self.covariance_inv = self.covariance_inv - self.covariance_inv @ u.t() @ inner_inv @ u @ self.covariance_inv

def get_top_k_keys(my_dict, k):
    # 按照字典的值降序排序，选择前k个值对应的键
    top_k_keys = sorted(my_dict, key=my_dict.get, reverse=True)[:k]
    return top_k_keys


def cal_duplicate_samples(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        from collections import Counter

        line_counts = Counter(lines)

        # 找到重复的行并计算重复的数量
        duplicate_lines = {line: count for line, count in line_counts.items() if count > 1}
        num_duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)

        # print(f"重复的行: {duplicate_lines}")
        # print(f"重复的总行数: {num_duplicate_lines}")


def clear_folder(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # 判断是否是文件，删除文件
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 判断是否是文件夹，删除文件夹及其内容
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除 {file_path} 时发生错误: {e}")
    else:
        print(f"文件夹 {folder_path} 不存在")