import copy
from typing import List, Dict
import torch.nn.functional as F

import torch
import torchvision
from torch import tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_loss, maskrcnn_inference, \
    keypointrcnn_loss, keypointrcnn_inference
from torchvision.ops import boxes as box_ops

# def postprocess_detections(
#         self,
#         class_logits,  # type: Tensor
#         box_regression,  # type: Tensor
#         proposals,  # type: List[Tensor]
#         image_shapes,  # type: List[Tuple[int, int]]
# ):
#     # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
#     device = class_logits.device
#     num_classes = class_logits.shape[-1]
#
#     boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
#     # [150]
#     pred_boxes = self.box_coder.decode(box_regression, proposals)
#
#     pred_scores = F.softmax(class_logits, -1)
#
#     pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
#     pred_scores_list = pred_scores.split(boxes_per_image, 0)
#
#     all_boxes = []
#     all_scores = []
#     all_labels = []
#     for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
#         boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
#
#         # create labels for each prediction
#         labels = torch.arange(num_classes, device=device)
#         labels = labels.view(1, -1).expand_as(scores)
#         # remove predictions with the background label
#         boxes = boxes[:, 1:]
#
#         scores = scores[:, 1:]
#         labels = labels[:, 1:]
#
#         # batch everything, by making every class prediction be a separate instance
#         boxes = boxes.reshape(-1, 4)
#         scores = scores.reshape(-1)
#         labels = labels.reshape(-1)
#
#         # remove low scoring boxes
#         inds = torch.where(scores > self.score_thresh)[0]
#         boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
#
#         # remove empty boxes
#         keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
#         boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
#
#         # non-maximum suppression, independently done per class
#         keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
#         # keep only topk scoring predictions
#         keep = keep[: self.detections_per_img]
#         boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
#
#         all_boxes.append(boxes)
#         all_scores.append(scores)
#         all_labels.append(labels)
#
#     return all_boxes, all_scores, all_labels


def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
    device = class_logits.device

    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_indices = []  # To store the indices of the final boxes in the original boxes_per_image

    for i, (boxes, scores, image_shape) in enumerate(zip(pred_boxes_list, pred_scores_list, image_shapes)):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # Create initial indices for the boxes before reshape
        original_indices = torch.arange(boxes.shape[0], device=device).view(-1, 1).expand(-1, num_classes)

        # Remove background boxes (index 0) and keep foreground boxes (index 1:)
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        original_indices = original_indices[:, 1:]

        # Reshape boxes, scores, labels and indices (class-wise flattening)
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        original_indices = original_indices.reshape(-1)

        # Remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels, original_indices = boxes[inds], scores[inds], labels[inds], original_indices[inds]

        # Remove small boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels, original_indices = boxes[keep], scores[keep], labels[keep], original_indices[keep]

        # Non-maximum suppression (NMS)
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

        # Keep only top-k scoring boxes
        keep = keep[:self.detections_per_img]
        boxes, scores, labels, original_indices = boxes[keep], scores[keep], labels[keep], original_indices[keep]

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
        all_indices.append(original_indices)  # Append the final indices for this image

    return all_boxes, all_scores, all_labels, all_indices, pred_scores


def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
):
    # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
    """
    Args:
        features (List[Tensor])
        proposals (List[Tensor[N, 4]])
        image_shapes (List[Tuple[H, W]])
        targets (List[Dict])
    """
    if targets is not None:
        for t in targets:
            # TODO: https://github.com/pytorch/pytorch/issues/26731
            floating_point_types = (torch.float, torch.double, torch.half)
            if not t["boxes"].dtype in floating_point_types:
                raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
            if not t["labels"].dtype == torch.int64:
                raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
            if self.has_keypoint():
                if not t["keypoints"].dtype == torch.float32:
                    raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

    if self.training:
        proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
    else:
        labels = None
        regression_targets = None
        matched_idxs = None

    # torch.Size([150, 256, 7, 7])
    box_features = self.box_roi_pool(features, proposals, image_shapes)
    if self.training:
        pass
    else:
        features_temp = features['1'].clone()
        features_map = features_temp.view(features_temp.shape[0], features_temp.shape[1]//32, 32, features_temp.shape[2], features_temp.shape[3]).mean(dim=2).squeeze(0).permute(1, 2, 0)
        features_map = features_map.flatten()

    box_features = self.box_head(box_features)
    class_logits, box_regression = self.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    losses = {}
    if self.training:
        if labels is None:
            raise ValueError("labels cannot be None")
        if regression_targets is None:
            raise ValueError("regression_targets cannot be None")
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    else:
        boxes, scores, labels, box_ind, cls_logits = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                    "class_logits" : cls_logits[box_ind],
                    "embedding": box_features[box_ind]
                }
            )
        result.append(features_map)

    if self.has_mask():
        mask_proposals = [p["boxes"] for p in result]
        if self.training:
            if matched_idxs is None:
                raise ValueError("if in training, matched_idxs should not be None")

            # during training, only focus on positive boxes
            num_images = len(proposals)
            mask_proposals = []
            pos_matched_idxs = []
            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                mask_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)
        else:
            raise Exception("Expected mask_roi_pool to be not None")

        loss_mask = {}
        if self.training:
            if targets is None or pos_matched_idxs is None or mask_logits is None:
                raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

            gt_masks = [t["masks"] for t in targets]
            gt_labels = [t["labels"] for t in targets]
            rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
            loss_mask = {"loss_mask": rcnn_loss_mask}
        else:
            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        losses.update(loss_mask)

    # keep none checks in if conditional so torchscript will conditionally
    # compile each branch
    if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
    ):
        keypoint_proposals = [p["boxes"] for p in result]
        if self.training:
            # during training, only focus on positive boxes
            num_images = len(proposals)
            keypoint_proposals = []
            pos_matched_idxs = []
            if matched_idxs is None:
                raise ValueError("if in trainning, matched_idxs should not be None")

            for img_id in range(num_images):
                pos = torch.where(labels[img_id] > 0)[0]
                keypoint_proposals.append(proposals[img_id][pos])
                pos_matched_idxs.append(matched_idxs[img_id][pos])
        else:
            pos_matched_idxs = None

        keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        loss_keypoint = {}
        if self.training:
            if targets is None or pos_matched_idxs is None:
                raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

            gt_keypoints = [t["keypoints"] for t in targets]
            rcnn_loss_keypoint = keypointrcnn_loss(
                keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
            )
            loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
        else:
            if keypoint_logits is None or keypoint_proposals is None:
                raise ValueError(
                    "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                )

            keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps
        losses.update(loss_keypoint)

    return result, losses

torchvision.models.detection.roi_heads.RoIHeads.forward = forward
torchvision.models.detection.roi_heads.RoIHeads.postprocess_detections = postprocess_detections