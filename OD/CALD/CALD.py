import time
from types import MethodType

import cv2
import scipy.stats

import torch.utils.data
import torchvision
from torch import nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy, nms_rotated
from OD.faster_rcnn_train_test import train_faster_rcnn
from OD.CALD.cald_helper import *
from OD.yolo_function import get_top_n_models, detect_objects
from OD.yolo_modify import _predict_once
import OD.faster_rcnn_modify


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    This version returns confidences for all classes.

    Args:
        (... same as before ...)

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_classes + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, class_conf_1, class_conf_2, ..., mask1, mask2, ...).
    """
    import torchvision

    # Checks and initialization (same as before)
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size  1
    nc = nc or (prediction.shape[1] - 4)  # number of classes 12
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    xk = torch.tensor([list(range(len(i))) for i in xc], device=prediction.device)


    # Settings
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    xk = xk.transpose(-1, -2)

    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nc + nm), device=prediction.device)] * bs
    feati = [torch.zeros((0, 1), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference

        filt = xc[xi]
        x = x[filt]  # confidence
        xk = xk[filt]  # indices update

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx(4 + nc + nm) (xyxy, class_conf, cls, masks)
        box, cls_conf, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls_conf > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            xk = xk[i] # indices update
        else:  # best class only
            conf, j = cls_conf.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), cls_conf, mask), 1)[filt]
            xk = xk[filt] # indices update

        # Confidence thresholding
        # conf, j = cls_conf.max(1, keepdim=True)
        # x = torch.cat((box, conf, j.float(), cls_conf, mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]
            x = x[filt]
            xk = xk[filt] # indices update

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]  # sort by confidence and remove excess boxes
            xk = xk[filt] # indices update

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        feati[xi] = xk[i].reshape(-1)

        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output


def get_uncertainty(task_model, unlabeled_loader, augs, num_cls, device_gpu, model_type):

    for aug in augs:
        if aug not in ['flip', 'multi_ga', 'color_adjust', 'color_swap', 'multi_color_adjust', 'multi_sp', 'cut_out',
                       'multi_cut_out', 'multi_resize', 'larger_resize', 'smaller_resize', 'rotation', 'ga', 'sp']:
            print('{} is not in the pre-set augmentations!'.format(aug))

    with torch.no_grad():
        consistency_all = []
        mean_all = []
        cls_all = []

        for image in tqdm(unlabeled_loader):
            torch.cuda.synchronize()

            # only support 1 batch size
            aug_images = []
            aug_boxes = []

            if model_type == 'yolo':
                image = cv2.imread(image)

                # 用于初始化preprocess
                _ = task_model(source="1.jpg", save=False, verbose=False)
                #

                task_model.model._predict_once = MethodType(_predict_once, task_model.model)
                _ = task_model(source="1.jpg", save=False, embed=[15, 18, 21, 22], verbose=False)
                prepped = task_model.predictor.preprocess([image])
                result = task_model.predictor.inference(prepped)
                results = non_max_suppression(result[-1][0], in_place=False)
                detection={
                        "boxes": [],
                        "labels": [],
                        "prob_max": [],
                        "scores_cls": []
                    }

                for result in results:
                    for idx, instance in enumerate(result):
                        bbox = instance[:4].tolist()  # Bounding box [x_min, y_min, x_max, y_max]
                        cls = int(instance[5].item())  # Class ID
                        conf = instance[4].item()  # Confidence score
                        classconf = instance[6:].cpu().numpy()

                        detection["boxes"].append(bbox)
                        detection["labels"].append(cls)
                        detection["prob_max"].append(conf)
                        detection["scores_cls"].append(classconf)

                detection["boxes"] = torch.tensor(detection["boxes"])
                detection["scores_cls"] = torch.tensor(detection["scores_cls"])
                detection["labels"] = torch.tensor(detection["labels"])
                detection["prob_max"] = torch.tensor(detection["prob_max"])

                ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = detection["boxes"], detection["prob_max"], detection["scores_cls"], detection["labels"], torch.ones(len(detection["labels"]))

                # ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = output[0]['boxes'], output[0][
                #     'prob_max'], output[0]['scores_cls'], output[0]['labels'], output[0]['scores']

                if len(ref_scores) > 40:
                    inds = np.round(np.linspace(0, len(ref_scores) - 1, 50)).astype(int)
                    ref_boxes, ref_scores_cls, ref_labels, ref_scores = ref_boxes[inds], ref_scores_cls[inds], ref_labels[inds], ref_scores[inds]
                cls_corr = [0] * (num_cls - 1)
                for s, l in zip(ref_scores, ref_labels):
                    cls_corr[l - 1] = max(cls_corr[l - 1], s.item())
                cls_corrs = [cls_corr]
                if detection["boxes"].shape[0] == 0:
                    consistency_all.append(0.0)
                    cls_all.append(np.mean(cls_corrs, axis=0))
                    continue
                # start augment

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 将 numpy 数组转换为 PIL 图像
                image = Image.fromarray(image)

                if 'flip' in augs:
                    flip_image, flip_boxes = HorizontalFlip(image, ref_boxes)
                    aug_images.append(flip_image)
                    aug_boxes.append(flip_boxes)
                if 'cut_out' in augs:
                    cutout_image = cutout(image, ref_boxes, ref_labels, 2)
                    aug_images.append(cutout_image)
                    aug_boxes.append(ref_boxes)
                if 'smaller_resize' in augs:
                    resize_image, resize_boxes = resize(image, ref_boxes, 0.8)
                    aug_images.append(resize_image)
                    aug_boxes.append(resize_boxes)
                if 'rotation' in augs:
                    rot_image, rot_boxes = rotate(image, ref_boxes, 5)
                    aug_images.append(rot_image)
                    aug_boxes.append(rot_boxes)
                outputs = []

                for aug_image in aug_images:
                    prepped = task_model.predictor.preprocess([aug_image])
                    result_ = task_model.predictor.inference(prepped)
                    results_ = non_max_suppression(result_[-1][0], in_place=False)
                    output = {
                        "boxes": [],
                        "labels": [],
                        "prob_max": [],
                        "scores_cls": [],
                        "scores": []
                    }

                    for result in results_:
                        for idx, instance in enumerate(result):
                            bbox = instance[:4].tolist()  # Bounding box [x_min, y_min, x_max, y_max]
                            cls = int(instance[5].item())  # Class ID
                            conf = instance[4].item()  # Confidence score
                            classconf = instance[6:].cpu().numpy()

                            output["boxes"].append(bbox)
                            output["labels"].append(cls)
                            output["prob_max"].append(conf)
                            output["scores_cls"].append(classconf)

                    output["boxes"] = torch.tensor(output["boxes"])
                    output["scores_cls"] = torch.tensor(output["scores_cls"])
                    output["scores"] = torch.ones(len(output["labels"]))

                    outputs.append(output)

            if model_type == 'faster_rcnn':
                img = cv2.imread(image)
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
                    outputs = task_model(img_tensor)
                # 处理模型输出，提取预测边界框、标签和置信度
                output = outputs[0]
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                class_logits = output['class_logits'].cpu().numpy()
                threshold = 0.2

                filtered_boxes = boxes[scores > threshold]
                filtered_labels = labels[scores > threshold]
                filtered_scores = scores[scores > threshold]
                filtered_class_logits = class_logits[scores > threshold]
                filtered_class_probabilities = torch.softmax(torch.tensor(filtered_class_logits), dim=1)
                if filtered_class_probabilities.size(0) > 0:
                    max_probabilities, max_classes = torch.max(filtered_class_probabilities, dim=1)
                else:
                    max_probabilities = torch.tensor([])


                ref_boxes, prob_max, ref_scores_cls, ref_labels, ref_scores = torch.tensor(filtered_boxes), torch.tensor(max_probabilities), torch.tensor(filtered_class_probabilities), torch.tensor(filtered_labels), torch.tensor(filtered_scores)

                if len(ref_scores) > 40:
                    inds = np.round(np.linspace(0, len(ref_scores) - 1, 50)).astype(int)
                    ref_boxes, ref_scores_cls, ref_labels, ref_scores = ref_boxes[inds], ref_scores_cls[inds], ref_labels[inds], ref_scores[inds]
                cls_corr = [0] * (num_cls - 1)
                for s, l in zip(ref_scores, ref_labels):
                    cls_corr[l - 1] = max(cls_corr[l - 1], s.item())
                cls_corrs = [cls_corr]
                if ref_boxes.shape[0] == 0:
                    consistency_all.append(0.0)
                    cls_all.append(np.mean(cls_corrs, axis=0))
                    continue

                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 将 numpy 数组转换为 PIL 图像
                image = Image.fromarray(image)
                # start augment

                if 'flip' in augs:
                    flip_image, flip_boxes = HorizontalFlip(image, ref_boxes)
                    aug_images.append(flip_image)
                    aug_boxes.append(flip_boxes)
                if 'cut_out' in augs:
                    cutout_image = cutout(image, ref_boxes, ref_labels, 2)
                    aug_images.append(cutout_image)
                    aug_boxes.append(ref_boxes)
                if 'smaller_resize' in augs:
                    resize_image, resize_boxes = resize(image, ref_boxes, 0.8)
                    aug_images.append(resize_image)
                    aug_boxes.append(resize_boxes)
                if 'rotation' in augs:
                    rot_image, rot_boxes = rotate(image, ref_boxes, 5)
                    aug_images.append(rot_image)
                    aug_boxes.append(rot_boxes)
                outputs = []

                for aug_image in aug_images:

                    transform = T.Compose([
                        T.ToPILImage(),
                        T.ToTensor()
                    ])
                    img_tensor = transform(aug_image).unsqueeze(0)  # 扩展维度，因为模型期望输入batch格式
                    # 将图像移到设备上
                    img_tensor = img_tensor.to(device_gpu)
                    # 进行推理
                    with torch.no_grad():
                        output_ = task_model(img_tensor)
                    output_ = output_[0]
                    boxes = output_['boxes'].cpu().numpy()
                    labels = output_['labels'].cpu().numpy()
                    scores = output_['scores'].cpu().numpy()
                    class_logits = output_['class_logits'].cpu().numpy()
                    threshold = 0.2

                    filtered_boxes = boxes[scores > threshold]
                    filtered_labels = labels[scores > threshold]
                    filtered_scores = scores[scores > threshold]
                    filtered_class_logits = class_logits[scores > threshold]
                    filtered_class_probabilities = torch.softmax(torch.tensor(filtered_class_logits), dim=1)
                    if filtered_class_probabilities.size(0) > 0:
                        max_probabilities, max_classes = torch.max(filtered_class_probabilities, dim=1)
                    else:
                        max_probabilities = torch.tensor([])

                    output = {
                        "boxes": torch.tensor(filtered_boxes),
                        "labels": torch.tensor(filtered_labels),
                        "prob_max": max_probabilities,
                        "scores_cls": filtered_class_probabilities,
                        "scores": torch.tensor(filtered_scores)
                    }
                    outputs.append(output)

            consistency_aug = []
            mean_aug = []
            for output, aug_box, aug_image in zip(outputs, aug_boxes, aug_images):
                consistency_img = 1.0
                mean_img = []
                boxes, scores_cls, pm, labels, scores = output['boxes'], output['scores_cls'], output['prob_max'], \
                                                        output['labels'], output['scores']
                cls_corr = [0] * (num_cls - 1)
                for s, l in zip(scores, labels):
                    cls_corr[l - 1] = max(cls_corr[l - 1], s.item())
                cls_corrs.append(cls_corr)
                if len(boxes) == 0:
                    consistency_aug.append(0.0)
                    mean_aug.append(0.0)
                    continue
                for ab, ref_score_cls, ref_pm, ref_score in zip(aug_box, ref_scores_cls, prob_max, ref_scores):
                    width = torch.min(ab[2], boxes[:, 2]) - torch.max(ab[0], boxes[:, 0])
                    height = torch.min(ab[3], boxes[:, 3]) - torch.max(ab[1], boxes[:, 1])
                    Aarea = (ab[2] - ab[0]) * (ab[3] - ab[1])
                    Barea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    iner_area = width * height
                    iou = iner_area / (Aarea + Barea - iner_area)
                    iou[width < 0] = 0.0
                    iou[height < 0] = 0.0
                    p = ref_score_cls.cpu().numpy()
                    q = scores_cls[torch.argmax(iou)].cpu().numpy()
                    m = (p + q) / 2
                    js = 0.5 * scipy.stats.entropy(p, m) + 0.5 * scipy.stats.entropy(q, m)
                    if js < 0:
                        js = 0
                    # consistency_img.append(torch.abs(
                    #     torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - args.bp).item())
                    consistency_img = min(consistency_img, torch.abs(
                        torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)]) - 1.3).item())
                    mean_img.append(torch.abs(
                        torch.max(iou) + 0.5 * (1 - js) * (ref_pm + pm[torch.argmax(iou)])).item())
                consistency_aug.append(np.mean(consistency_img))
                mean_aug.append(np.mean(mean_img))
            consistency_all.append(np.mean(consistency_aug))
            mean_all.append(mean_aug)
            cls_corrs = np.mean(np.array(cls_corrs), axis=0)
            cls_all.append(cls_corrs)
    mean_aug = np.mean(mean_all, axis=0)
    return consistency_all, cls_all


def cls_kldiv(labeled_loader, cls_corrs, budget):
    cls_inds = []
    result = []

    for label in labeled_loader.labels:
        for cls_id in label["cls"].reshape(-1):
            cls_corr = [0] * cls_corrs[0].shape[0]
            cls_corr[int(cls_id)-1] += 1
            result.append(cls_corr)

    # for _, targets in labeled_loader:
    #     for target in targets:
    #         cls_corr = [0] * cls_corrs[0].shape[0]
    #         for l in target['labels']:
    #             cls_corr[l - 1] += 1
    #         result.append(cls_corr)


    for a in list(np.where(np.sum(cls_corrs, axis=1) == 0)[0]):
        print(a)
        cls_inds.append(a)
        # result.append(cls_corrs[a])

    while len(cls_inds) < budget:
        # batch cls_corrs together to accelerate calculating
        KLDivLoss = nn.KLDivLoss(reduction='none')
        _cls_corrs = torch.tensor(cls_corrs)
        _result = torch.tensor(np.mean(np.array(result), axis=0)).unsqueeze(0)

        p = torch.nn.functional.softmax(_result, -1)
        q = torch.nn.functional.softmax(_cls_corrs, -1)
        log_mean = ((p + q) / 2).log()
        jsdiv = torch.sum(KLDivLoss(log_mean, p), dim=1) / 2 + torch.sum(KLDivLoss(log_mean, q), dim=1) / 2
        jsdiv[cls_inds] = -1
        max_ind = torch.argmax(jsdiv).item()
        cls_inds.append(max_ind)
    # result.append(cls_corrs[max_ind])
    return cls_inds


def CALD_method(model_type, b, imgs, dataset_name, device_gpu, labeled_dataloader):

    dataset_name_class_num = {
        'HW': 6,
        'VOC2012':20,
        'Waymo':5,
        'BDD100K':10,
        'TJU-DHD': 5,
    }

    model_folder_path  ='../yolo_model'
    if model_type == 'yolo':
        model_path = get_top_n_models(model_type, model_folder_path, n=1)[0]
        task_model = YOLO(model_path)
        task_model.to(device_gpu)


    if model_type == 'faster_rcnn':

        model_path = get_top_n_models(model_type, model_folder_path, n=1)[0]
        task_model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = task_model.roi_heads.box_predictor.cls_score.in_features
        task_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset_name_class_num[dataset_name])
        task_model.load_state_dict(torch.load(model_path))
        task_model.to(device=device_gpu)
        task_model.eval()



    # 图像增强的选项，默认FCDR
    augs = []
    # FCDR
    augs.append('flip')
    augs.append('cut_out')
    augs.append('smaller_resize')
    augs.append('rotation')

    # 获取不确定
    uncertainty, _cls_corrs = get_uncertainty(task_model, imgs, augs, dataset_name_class_num[dataset_name], device_gpu, model_type)

    arg = np.argsort(np.array(uncertainty))

    # 选择1.2倍的预算
    cls_corrs_set = arg[:int(1.2 * b)]

    # 得到每一个候选样本的类别得分
    cls_corrs = [_cls_corrs[i] for i in cls_corrs_set]

    # 得到KL散度并且根据KL散度过滤多余的0.2倍的样本
    tobe_labeled_set = cls_kldiv(labeled_dataloader, cls_corrs, b)
    image_names = [imgs[i] for i in cls_corrs_set[tobe_labeled_set]]
    print(len(image_names))
    return image_names