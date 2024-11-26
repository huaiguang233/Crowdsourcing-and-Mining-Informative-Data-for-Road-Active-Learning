import os
import random
import time
from types import MethodType
import torchvision.transforms as T

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.models as models
from ultralytics.data import YOLODataset
from ultralytics.utils import yaml_load

from OD.yolo_modify import _predict_once
from submodular_maximization import FacilityLocation_with_M

from yolo_function import detect_objects, pre_select_bbox, calculate_detection_scores, \
    get_top_n_models, greedy_select_images, Dual_dpp, get_top_k_keys, clear_txt, \
    append_index_to_txt, calculate_js_divergence, greedy_select_images_fal, k_center_greedy, init_train_txt


def get_class_distrbution(yaml_path, img_path_txt):
    if 'HW' in yaml_path:
        labels_cnt = {i: 0 for i in range(6)}
    if 'VOC2012' in yaml_path:
        labels_cnt = {i: 0 for i in range(20)}
    if 'Waymo' in yaml_path:
        labels_cnt = {i: 0 for i in range(5)}
    if 'BDD100K' in yaml_path:
        labels_cnt = {i: 0 for i in range(10)}
    if 'TJU-DHD' in yaml_path:
        labels_cnt = {i: 0 for i in range(5)}

    dataset = YOLODataset(data=yaml_load(yaml_path), task="detect", img_path=img_path_txt)
    # 更新字典中的类别计数
    for label in dataset.labels:
        for cls_id in label["cls"].reshape(-1):
            labels_cnt[cls_id] += 1

    class_distribution_curr = np.array(
        [labels_cnt[key] / sum(labels_cnt.values()) for key in sorted(labels_cnt.keys())])
    print(class_distribution_curr)
    return class_distribution_curr, dataset


def DAL_method(windows_b, iou_, lambda1, b, imgs, class_distribution_curr, dpp, model_type, dataset_name, device_gpu):

    selected_images, all_image_results = Stage1(windows_b, iou_, lambda1, imgs, b, model_type, dataset_name, device_gpu)
    # sorted_file_names = sorted(selected_images, key=lambda x: int(x.split('_')[1].split('.')[0]))
    # 往下开始，记得改下2b
    image_score = dict()
    # Concatenate all bbox embeddings from all images
    for image_name, bboxes in all_image_results.items():
        if image_name in selected_images:
            # Extract embeddings and concatenate them
            embeddings = [bbox['embedding'] for bbox in bboxes]  # List of embeddings

            # 不考虑js散度
            # normalized_weighted_embeddings = [
            #     (embedding if np.linalg.norm(embedding) != 0 else embedding)
            #     for embedding in embeddings
            # ]

            # 考虑js散度
            js_score = [calculate_js_divergence(bbox['classconf'], class_distribution_curr) for bbox in bboxes]

            normalized_weighted_embeddings = [
                (embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) != 0 else embedding) * js
                for embedding, js in zip(embeddings, js_score)
            ]

            if normalized_weighted_embeddings:  # Check if the list is not empty
                all_embeddings = torch.stack(
                    [torch.tensor(e) for e in normalized_weighted_embeddings])  # Stack into a single tensor
                # Use the dual_dpp function with the concatenated embeddings
                image_score[image_name] = dpp.dual_dpp(all_embeddings)
    image_names = get_top_k_keys(image_score, b)

    # print(sorted(image_names, key=lambda x: int(x.split('_')[1].split('.')[0])))
    for image_name in image_names:
        bboxes = all_image_results[image_name]

        embeddings = [bbox['embedding'] for bbox in bboxes]  # List of embeddings

        # 不考虑js散度
        # normalized_weighted_embeddings = [
        #     (embedding if np.linalg.norm(embedding) != 0 else embedding)
        #     for embedding in embeddings
        # ]

        # 考虑js散度
        js_score = [calculate_js_divergence(bbox['classconf'], class_distribution_curr) for bbox in bboxes]

        normalized_weighted_embeddings = [
            (embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) != 0 else embedding) * js
            for embedding, js in zip(embeddings, js_score)
        ]

        if normalized_weighted_embeddings:  # Check if the list is not empty
            all_embeddings = torch.stack(
                [torch.tensor(e) for e in normalized_weighted_embeddings])  # Stack into a single tensor
            # Use the dual_dpp function with the concatenated embeddings
            dpp.update_covariance_inv(all_embeddings)
    return image_names


def Stage1(windows_b, iou_, lambda1, imgs, b, model_type, dataset_name, device_gpu):
    dataset_name_class_num = {
        'HW': 6,
        'VOC2012':20,
        'Waymo':5,
        'BDD100K':10,
        'TJU-DHD': 5,
    }
    all_image_results = {}
    model_folder_path  ='../yolo_model'
    model_path_list = get_top_n_models(model_type, model_folder_path, n = windows_b)

    model_list = []
    for model_path in model_path_list:
        if model_type == 'yolo':
            model = YOLO(model_path)
            # model.model._predict_once = MethodType(_predict_once, model.model)
            # _ = model(source="1.jpg", save=False, embed=[15, 18, 21, 22])
            model_list.append(model)
        if model_type == 'faster_rcnn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset_name_class_num[dataset_name])

            model.load_state_dict(torch.load(model_path))
            model.to(device=device_gpu)
            model_list.append(model)

    for image_filename in imgs:
        # Step 1: Detect objects in the image using all models
        all_detections = []
        for model in model_list:
            # 调整模型设定
            all_detections.append(detect_objects(model_type, model, image_filename, device_gpu))
        # Step 2: Filter the bounding boxes across models based on IOU threshold
        pre_selected_bbox = pre_select_bbox(all_detections, iou_=iou_)

        # Step 3: Calculate detection scores (optional, depending on your specific needs)
        pre_selected_bbox = calculate_detection_scores(pre_selected_bbox, all_detections, iou_)

        # Step 4: Extract feature embeddings for each bbox using ResNet18
        # pre_selected_bbox = get_box_embedding(pre_selected_bbox, image_filename, embedding_model)

        if len(pre_selected_bbox) != 0:
            # Store the results for this image
            all_image_results[image_filename] = pre_selected_bbox

    # 选择前2*b个图片
    selected_images = greedy_select_images(all_image_results, n=2*b, lambda1=lambda1)
    return selected_images, all_image_results

    # 没有第一阶段， 消融实验
    # return imgs, all_image_results


def Leastconf_method(model_type, b, imgs, dataset_name, device_gpu):
    model_folder_path  ='../yolo_model'
    all_det = []
    all_conf = []
    all_images = []

    dataset_name_class_num = {
        'HW': 6,
        'VOC2012':20,
        'Waymo':5,
        'BDD100K':10
    }

    if model_type == 'faster_rcnn':
        model_path = get_top_n_models(model_type, model_folder_path, n=1)[0]
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset_name_class_num[dataset_name])
        model.load_state_dict(torch.load(model_path))
        model.to(device=device_gpu)
        model.eval()

        for image_filename in tqdm(imgs):
            img = cv2.imread(image_filename)
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
                all_images.append(image_filename)
                output = model(img_tensor)[0]
                scores = output['scores'].cpu().numpy()
                threshold = 0.2
                filtered_scores = scores[scores > threshold]
                if len(filtered_scores) == 0:
                    all_conf.append(100000)
                else:
                    all_conf.append(filtered_scores.mean())

    if model_type == 'yolo':
        model_path = get_top_n_models(model_type, model_folder_path, n=1)[0]
        model = YOLO(model_path)
        for image_filename in tqdm(imgs):
            all_images.append(image_filename)
            all_det.append(detect_objects('yolo', model, image_filename, device_gpu))
        for det in all_det:
            conf_sum = 0
            if len(det) > 0:  # 如果有检测结果
                for d in det:
                    conf_sum += d['confidence']
                conf_mean = conf_sum / len(det)  # 计算置信度的均值
            else:
                conf_mean = 100000  # 如果没有检测到任何目标, 赋一个极大值，后来不选择他
            all_conf.append(conf_mean)

    all_conf = np.array(all_conf)
    indices = np.argsort(all_conf)[:b]
    image_names = np.array(all_images)[indices]
    return image_names



def Random_method(all_images, b):

    selected_images = random.sample(list(all_images), b)
    return selected_images

def Coreset_method(model_type, b, imgs, dataset_name, device_gpu):

    model_folder_path  ='../yolo_model'
    all_image_embeddings = []
    all_images = []
    dataset_name_class_num = {
        'HW': 6,
        'VOC2012':20,
        'Waymo':5,
        'BDD100K':10
    }

    if model_type == 'faster_rcnn':
        model_list = []
        model_path = get_top_n_models(model_type, model_folder_path, n = 1)[0]
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset_name_class_num[dataset_name])

        model.load_state_dict(torch.load(model_path))
        model.to(device=device_gpu)
        model.eval()
        model_list.append(model)

        for image_filename in tqdm(imgs):
            img = cv2.imread(image_filename)
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
                outputs = model(img_tensor)[1]
                all_images.append(image_filename)
                all_image_embeddings.append(outputs.cpu())
        all_image_embeddings = torch.stack(all_image_embeddings)

    if model_type =='yolo':
        model_path = get_top_n_models(model_type, model_folder_path, n = 1)[0]
        model = YOLO(model_path)
        model.model._predict_once = MethodType(_predict_once, model.model)
        _ = model(source="1.jpg", save=False, embed=[15, 18, 21, 22])

        for image_filename in tqdm(imgs):
            img = cv2.imread(image_filename)
            if dataset_name == 'VOC2012':
                img = cv2.resize(img, (640, 480))
            prepped = model.predictor.preprocess([img])
            result = model.predictor.inference(prepped)
            feat = result[2]
            fea = feat.permute(0, 2, 3, 1).reshape(-1, 64, feat.shape[1] // 64).mean(dim=-1).view(-1)
            all_images.append(image_filename)
            all_image_embeddings.append(fea.cpu())

        all_image_embeddings = torch.stack(all_image_embeddings)

    image_names = k_center_greedy(all_image_embeddings, b)
    image_names = [all_images[i] for i in image_names]

    return image_names

def FAL_method(model_type, img_path_txt, b, imgs, device_gpu, dataset_name):

    model_folder_path  ='../yolo_model'
    all_image_embeddings = []
    all_images = []

    if model_type == 'faster_rcnn':

        dataset_name_class_num = {
            'HW': 6,
            'VOC2012': 20,
            'Waymo': 5,
            'BDD100K': 10,
            'TJU-DHD': 5
        }

        model_path = get_top_n_models(model_type, model_folder_path, n = 1)[0]
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset_name_class_num[dataset_name])

        model.load_state_dict(torch.load(model_path))
        model.to(device=device_gpu)
        model.eval()

        for image_filenames in tqdm(imgs):
            for image_filename in image_filenames:
                img = cv2.imread(image_filename)
                if dataset_name == 'VOC2012':
                    img = cv2.resize(img, (640, 480))
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
                    outputs = model(img_tensor)[1]
                    all_images.append(image_filename)
                    all_image_embeddings.append(outputs.cpu())

                with open(img_path_txt, 'r', encoding='utf-8') as file:
                    base_image_path = file.readlines()
                    base_image_path = [path.strip() for path in base_image_path]

        for base_images in base_image_path:

            img = cv2.imread(base_images)
            if dataset_name == 'VOC2012':
                img = cv2.resize(img, (640, 480))
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
                outputs = model(img_tensor)[1]
                all_images.append(base_images)
                all_image_embeddings.append(outputs.cpu())

        all_image_embeddings = torch.stack(all_image_embeddings)

    if model_type =='yolo':
        model_path = get_top_n_models(model_type, model_folder_path, n = 1)[0]
        model = YOLO(model_path)
        model.model._predict_once = MethodType(_predict_once, model.model)
        _ = model(source="1.jpg", save=False, embed=[15, 18, 21, 22])

        for image_filenames in tqdm(imgs):
            for image_filename in image_filenames:
                img = cv2.imread(image_filename)
                if dataset_name == 'VOC2012':
                    img = cv2.resize(img, (640, 480))
                prepped = model.predictor.preprocess([img])
                result = model.predictor.inference(prepped)
                feat = result[2]
                fea = feat.permute(0, 2, 3, 1).reshape(-1, 64, feat.shape[1] // 64).mean(dim=-1).view(-1)
                all_images.append(image_filename)
                all_image_embeddings.append(fea.cpu())

        with open(img_path_txt, 'r', encoding='utf-8') as file:
            base_image_path = file.readlines()
            base_image_path = [path.strip() for path in base_image_path]

        for base_images in base_image_path:

            img = cv2.imread(base_images)
            if dataset_name == 'VOC2012':
                img = cv2.resize(img, (640, 480))
            prepped = model.predictor.preprocess([img])
            result = model.predictor.inference(prepped)
            feat = result[2]
            fea = feat.permute(0, 2, 3, 1).reshape(-1, 64, feat.shape[1] // 64).mean(dim=-1).view(-1)

            all_image_embeddings.append(fea.cpu())

        all_image_embeddings = torch.stack(all_image_embeddings)


    distances = pairwise_distances(all_image_embeddings, metric="euclidean")

    index_all = np.concatenate((imgs.flatten(), np.array(base_image_path)))

    index_base = base_image_path

    M = 1 / (1 + 0.01 * distances)

    M, M_max = create_M_max_M(
        M, index_all, imgs, index_base
    )
    sampling_policy = FacilityLocation_with_M(
        M, M_max, imgs, b, imgs.shape[1]
    )
    select_final = sampling_policy.sample_caches("Interactive")

    return select_final


def create_M_max_M(M, all_inds, obs_inds, base_inds):

    all_inds = all_inds.tolist()
    obs_ind = np.concatenate(obs_inds, axis=0)

    obs_mask = [all_inds.index(i) for i in obs_ind]

    #得到观察样本在所有样本中的索引
    base_mask = [all_inds.index(i) for i in base_inds]
    #得到在云端服务器的样本在所有样本中的索引
    M_max = np.max(M[obs_mask][:, base_mask], axis=1).reshape(-1, 1)
    # 得到一个列向量，具体来说，对于每个观测数据，
    # M_max的相应元素是 观测数据与所有base数据之间距离的最大值
    # 扩充到20000×20000，因为之前有一步去重
    M = M[np.ix_(obs_mask, obs_mask)]

    return M, M_max