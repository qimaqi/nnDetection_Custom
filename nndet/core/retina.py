# Modifications licensed under:
# SPDX-FileCopyrightText: 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# SPDX-License-Identifier: Apache-2.0
#
# Parts of this code are from torchvision (https://github.com/pytorch/vision) licensed under
# SPDX-FileCopyrightText: 2016 Soumith Chintala 
# SPDX-License-Identifier: BSD-3-Clause


import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Tuple, Dict, Any, Optional, Union

from nndet.arch.abstract import AbstractModel
from nndet.core import boxes as box_utils
from nndet.arch.encoder.abstract import EncoderType
from nndet.arch.decoder.base import DecoderType
from nndet.arch.heads.segmenter import SegmenterType
from nndet.arch.heads.comb import HeadType
from nndet.core.boxes.anchors import AnchorGeneratorType


class BaseRetinaNet(AbstractModel):
    def __init__(self,
                 dim: int,
                 # modules
                 encoder: EncoderType,
                 decoder: DecoderType,
                 head: HeadType,
                 num_classes: int,
                 anchor_generator: AnchorGeneratorType,
                 matcher: box_utils.MatcherType,
                 decoder_levels: tuple = (2, 3, 4, 5),
                 # post-processing
                 score_thresh: float = None,
                 detections_per_img: int = 100,
                 topk_candidates: int = 10000,
                 remove_small_boxes: float = 1e-2,
                 nms_thresh: float = 0.9,
                 # optional
                 segmenter: Optional[SegmenterType] = None,
                 ):
        """
        Base Retina(U)Net
        Can be subclasses to add specific configurations to it

        Args:
            dim: number of spatial dimensions
            encoder: encoder module
            decoder: decoder module
            head: head module
            num_classes: number of foreground classes
            anchor_generator: generate anchors
            matcher: match ground truth boxes and anchors
            decoder_levels: decoder levels to use for detection prediciton
            score_thresh: minimum output probability
            detections_per_img: max detections per image
            topk_candidates: select only topk candidates for nms computation
            remove_small_boxes: remove small bounding boxes
            nms_thresh: non maximum suppression threshold
            segmenter: segmentation module
        """
        super().__init__()
        assert dim in [2, 3]
        self.dim = dim
        self.decoder_levels = decoder_levels

        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.num_foreground_classes = num_classes

        self.anchor_generator = anchor_generator
        self.proposal_matcher = matcher

        self.score_thresh = score_thresh
        self.topk_candidates = topk_candidates
        self.detections_per_img = detections_per_img
        self.remove_small_boxes = remove_small_boxes
        self.nms_thresh = nms_thresh

        self.segmenter = segmenter

    def train_step(self,
                   images: Tensor,
                   targets: dict,
                   evaluation: bool,
                   batch_num: int,
                   ) -> Tuple[
            Dict[str, torch.Tensor], Optional[Dict]]:
        """
        Perform a single training step (forward pass + loss computation)

        Args:
            images: batch of images
            targets: labels for training
                `target_boxes` (List[Tensor]): ground truth bounding boxes
                    (x1, y1, x2, y2, (z1, z2))[X, dim * 2], X= number of ground
                    truth boxes in image
                `target_classes` (List[Tensor]): ground truth class per box
                    (classes start from 0) [X], X= number of ground truth
                    boxes in image
                `target_seg`(Tensor): segmentation ground truth
                    (only needed if :param:`segmenter`
                    was provided in init) (classes start from 1, 0 background)
            evaluation (bool): compute final predictions (includes detection
                postprocessing)
            batch_num (int): batch index inside epoch

        Returns:
            torch.Tensor: final loss for back propagation
            Dict: predictions for metric calculation
                'pred_boxes': List[Tensor]: predicted bounding boxes for each
                    image List[[R, dim * 2]]
                'pred_scores': List[Tensor]: predicted probability for the
                    class List[[R]]
                'pred_labels': List[Tensor]: predicted class List[[R]]
                'pred_seg': Tensor: predicted segmentation [N, dims]
            Dict[str, torch.Tensor]: scalars for logging (e.g. individual
                loss components)
        """
        # import napari
        # with napari.gui_qt():
        #     viewer = napari.view_image(images.detach().cpu().numpy())
        #     viewer.add_labels(seg_targets[:, None].detach().cpu().numpy())

        target_boxes: List[Tensor] = targets["target_boxes"]
        target_classes: List[Tensor] = targets["target_classes"]
        target_seg: Tensor = targets["target_seg"]

        pred_detection, anchors, pred_seg = self(images)

        # print("train result debug", pred_detection['box_deltas'].min(), pred_detection['box_deltas'].max(), pred_detection['box_deltas'].dtype)
        # print("anchors", anchors[0].min(), anchors[0].max(), anchors[0].dtype)
        # print("seg", pred_seg['seg_logits'].min(), pred_seg['seg_logits'].max(), pred_seg['seg_logits'].dtype)


        labels, matched_gt_boxes = self.assign_targets_to_anchors(
            anchors, target_boxes, target_classes)

        losses = {}
        head_losses, pos_idx, neg_idx = self.head.compute_loss(
            pred_detection, labels, matched_gt_boxes, anchors)
        losses.update(head_losses)

        if self.segmenter is not None:
            losses.update(self.segmenter.compute_loss(pred_seg, target_seg))

        if evaluation:
            print("before crop len of pred_detection", len(pred_detection['box_deltas']), len(pred_detection['box_logits']))

            prediction = self.postprocess_for_inference(
                images=images,
                pred_detection=pred_detection,
                pred_seg=pred_seg,
                anchors=anchors,
            )

            print("after crop len of pred_detection", len(prediction['pred_boxes']), len(prediction['pred_scores']))
        else:
            prediction = None

        # self.save_matched_anchors(images=images, target_boxes=target_boxes,
        #                             anchors=anchors, pos_idx=pos_idx,
        #                             neg_idx=neg_idx, seg=seg_targets)

        return losses, prediction

    @torch.no_grad()
    def postprocess_for_inference(self,
                                  images: torch.Tensor,
                                  pred_detection: Dict[str, torch.Tensor],
                                  pred_seg: Dict[str, torch.Tensor],
                                  anchors: List[torch.Tensor],
                                  ) -> Dict[str, Union[List[Tensor], Tensor]]:
        """
        Postprocess predictions for inference

        Args:
            images: input images
            pred_detection: detection predictions
            pred_seg: segmentation predictions
            anchors: anchors

        Returns:
            Dict: post processed predictions
                'pred_boxes': List[Tensor]: predicted bounding boxes for each
                    image List[[R, dim * 2]]
                'pred_scores': List[Tensor]: predicted probability for
                    the class List[[R]]
                'pred_labels': List[Tensor]: predicted class List[[R]]
                'pred_seg': Tensor: predicted segmentation [N, C, dims]
        """
        image_shapes = [images.shape[2:]] * images.shape[0]
        boxes, probs, labels = self.postprocess_detections(
            pred_detection=pred_detection,
            anchors=anchors,
            image_shapes=image_shapes,
        )
        prediction = {"pred_boxes": boxes, "pred_scores": probs, "pred_labels": labels}

        if self.segmenter is not None:
            prediction["pred_seg"] = self.segmenter.postprocess_for_inference(pred_seg)["pred_seg"]
        return prediction

    def forward(self,
                inp: torch.Tensor,
                ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute predicted bounding boxes, scores and segmentations

        Args:
            inp (torch.Tensor): batch of input images

        Returns:
            dict: predictions from head. Typically includes:
                `box_deltas`(Tensor): bounding box offsets
                    [Num_Anchors_Batch, (dim * 2)]
                `box_logits`(Tensor): classification logits
                    [Num_Anchors_Batch, (num_classes)]
            List[torch.Tensor]: list of anchors (for each image inside the
                batch)
            dict: segmentation prediction. None if retina net is configured.
                Typically includes:
                    `seg_logits`: segmentation logits
        """
        # print("inp", inp.shape, inp.min(), inp.max())
 

        features_maps_all = self.decoder(self.encoder(inp))
        feature_maps_head = [features_maps_all[i] for i in self.decoder_levels]

        # for step, feat_i in enumerate(feature_maps_head):
        #     print("feat_i", feat_i.shape, "step", step)


        pred_detection = self.head(feature_maps_head)
        anchors = self.anchor_generator(inp, feature_maps_head)

        pred_seg = self.segmenter(features_maps_all) if self.segmenter is not None else None
        return pred_detection, anchors, pred_seg

    @torch.no_grad()
    def assign_targets_to_anchors(self,
                                  anchors: List[torch.Tensor],
                                  target_boxes: List[torch.Tensor],
                                  target_classes: List[torch.Tensor]) -> Tuple[
                                      List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute labels and matched ground truth for each anchor
        Adapted from torchvision https://github.com/pytorch/vision

        Args:
            anchors (List[torch.Tensor[float]]): anchors (!)per image(!)
                List[[N, dim * 2]], N=number of anchors per image
            target_boxes (List[torch.Tensor[float]]): ground truth boxes
                (!)per image(!)
                List[[X, dim * 2]], X=number of gt per image
            target_classes (List[torch.Tensor): ground truth classes
                (!)per image(!) (classes start from 0)
                List[[X]], X=number of gt per image

        Returns:
            List[torch.Tensor]: labels ([1, K]: foreground classes, 0: background,
                -1: between) List[[N]], N=number of anchors per image
            List[torch.Tensor]: matched gt box List[[N, dim *  2]],
                N=number of anchors per image
        """
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, gt_boxes, gt_classes in zip(anchors, target_boxes, target_classes):
            # indices of ground truth box for each proposal
            match_quality_matrix, matched_idxs = self.proposal_matcher(
                gt_boxes, anchors_per_image,
                num_anchors_per_level=self.anchor_generator.get_num_acnhors_per_level(),
                num_anchors_per_loc=self.anchor_generator.num_anchors_per_location()[0])

            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            if match_quality_matrix.numel() > 0:
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # Positive (negative indices can be ignored because they are overwritten in the next step)
                # this influences how background class is handled in the input!!!! (here +1 for background)
                labels_per_image = gt_classes[matched_idxs.clamp(min=0)].to(dtype=anchors_per_image.dtype)
                labels_per_image = labels_per_image + 1
            else:
                num_anchors_per_image = anchors_per_image.shape[0]
                # no ground truth => no matches, all background
                matched_gt_boxes_per_image = torch.zeros_like(anchors_per_image)
                labels_per_image = torch.zeros(num_anchors_per_image).to(anchors_per_image)

            # Background (negative examples)
            bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def postprocess_detections(self,
                               pred_detection: Dict[str, Tensor],
                               anchors: List[Tensor],
                               image_shapes: List[Tuple[int]],
                               ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Postprocess bounding box deltas and logits to generate final boxes and
        scores
        Adapted from torchvision https://github.com/pytorch/vision

        Args:
            pred_detection: detection predictions for loss computation
                `box_logits`: classification logits for each anchor [N]
                `box_deltas`: offsets for each anchor
                    (x1, y1, x2, y2, (z1, z2))[N, dim * 2]
            anchors: proposals for each image
            image_shapes: shape of each image
        
        Returns:
            List[Tensor]: final boxes [R, dim * 2]
            List[Tensor]: final scores (for final class) [R]
            List[Tensor]: final class label [R]
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in anchors]
        pred_detection = self.head.postprocess_for_inference(pred_detection, anchors)
        pred_boxes, pred_probs = pred_detection["pred_boxes"], pred_detection["pred_probs"]

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_probs = pred_probs.split(boxes_per_image, 0)

        all_boxes, all_probs, all_labels = [], [], []
        # iterate over images
        for boxes, probs, image_shape in zip(pred_boxes, pred_probs, image_shapes):
            boxes, probs, labels = self.postprocess_detections_single_image(boxes, probs, image_shape)
            all_boxes.append(boxes)
            all_probs.append(probs)
            all_labels.append(labels)
        return all_boxes, all_probs, all_labels

    def postprocess_detections_single_image(
        self, 
        boxes: Tensor, 
        probs: Tensor,
        image_shape: Tuple[int],
        ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Postprocess bounding box deltas and probabilities for a single image
        Adapted from torchvision https://github.com/pytorch/vision

        Args:
            boxes: predicted deltas for proposals [N, dim * 2]
            probs: predicted logits for boxes [N, C]
            image_shape: shape of image

        Returns:
            Tensor: final boxes [R, dim * 2]
            Tensor: final scores (for final class) [R]
            Tensor: final class label [R]
        """
        assert boxes.shape[0] == probs.shape[0]
        boxes = box_utils.clip_boxes_to_image_(boxes, image_shape)
        probs = probs.flatten()

        if self.topk_candidates is not None:
            num_topk = min(self.topk_candidates, boxes.size(0))
            probs, idx = probs.sort(descending=True)
            probs, idx = probs[:num_topk], idx[:num_topk]
        else:
            idx = torch.arange(probs.numel())

        if self.score_thresh is not None:
            keep_idxs = probs > self.score_thresh
            probs, idx = probs[keep_idxs], idx[keep_idxs]

        anchor_idxs = torch.div(idx, self.num_foreground_classes, rounding_mode="floor")
        labels = idx % self.num_foreground_classes
        boxes = boxes[anchor_idxs]

        if self.remove_small_boxes is not None:
            keep = box_utils.remove_small_boxes(boxes, min_size=self.remove_small_boxes)
            boxes, probs, labels = boxes[keep], probs[keep], labels[keep]

        keep = box_utils.batched_nms(boxes, probs, labels, self.nms_thresh)
        
        if self.detections_per_img is not None:
            keep = keep[:self.detections_per_img]
        return boxes[keep], probs[keep], labels[keep]

    # @torch.no_grad()
    # def save_matched_anchors(self, **kwargs):
    #     logger = get_logger("mllogger")
    #     logger.save_pickle("anchor_matching",
    #                        to_device(kwargs, device="cpu", detach=True))

    @torch.no_grad()
    def inference_step(self,
                       images: Tensor,
                       **kwargs,
                       ) -> Dict[str, Any]:
        """
        Perform inference for a batch of images

        Args:
            images: batch of input images [N, C, W, H, (D)]

        Returns:
            Dict:
                'pred_boxes': List[Tensor]: predicted bounding boxes for each
                    image List[[R, dim * 2]]
                'pred_scores': List[Tensor]: predicted probability for
                    the class List[[R]]
                'pred_labels': List[Tensor]: predicted class List[[R]]
                'pred_seg': Tensor: predicted segmentation [N, C, dims]
        """
        # print("images", images.shape, images.min(), images.max(), images.dtype)

        with torch.cuda.amp.autocast(enabled=False):
            pred_detection, anchors, pred_seg = self(images)

        # assert not inf or NaN and pred_detection['box_deltas'] 
        assert not torch.isnan(pred_detection['box_deltas']).any()
        assert not torch.isinf(pred_detection['box_deltas']).any()
        # print("test result debug", pred_detection['box_deltas'].min(), pred_detection['box_deltas'].max(), pred_detection['box_deltas'].dtype)
        # print("anchors", anchors[0].min(), anchors[0].max(), anchors[0].dtype)
        # print("seg", pred_seg['seg_logits'].min(), pred_seg['seg_logits'].max(), pred_seg['seg_logits'].dtype)
        # raise ValueError('stop here')

        print("Box shape before postprocess", len(pred_detection['box_deltas']), len(pred_detection['box_logits']))
        prediction = self.postprocess_for_inference(
            images=images,
            pred_detection=pred_detection,
            pred_seg=pred_seg,
            anchors=anchors,
        )
        print("Box shape after postprocess", len(prediction['pred_boxes']), len(prediction['pred_scores']))
        return prediction

# np torch.Size([4, 1, 16, 224, 224]) tensor(-2.1874, device='cuda:0') tensor(1.6805, device='cuda:0')                  | 0/104 [00:00<?, ?it/s]
# test result debug {'box_deltas': tensor([[   -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [   -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [   -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
#         ...,
#         [-0.0844, -0.0249, -0.0569, -0.0828, -0.0355, -0.0009],
#         [-0.0933, -0.0017, -0.0524, -0.0375, -0.0268, -0.1099],
#         [-0.0822, -0.0042, -0.0320,  0.0188,  0.0045, -0.1328]],
#        device='cuda:0', dtype=torch.float16), 'box_logits': tensor([[-inf],
#         [-inf],
#         [-inf],
#         ...,
#         [-inf],
#         [-inf],
#         [-inf]], device='cuda:0', dtype=torch.float16)}


# train result debug {'box_deltas': tensor([[ 0.0868,  0.0029,  0.0204,  0.0401, -0.0095,  0.0198],
#         [ 0.0840, -0.0041,  0.0139,  0.0226, -0.0097, -0.1011],
#         [ 0.0745,  0.0082,  0.0107,  0.0393, -0.0116, -0.2100],
#         ...,
#         [-0.1264, -0.0394, -0.0198,  0.0786, -0.0776,  0.2166],
#         [-0.1241, -0.0115, -0.0026,  0.1290, -0.0841,  0.1109],
#         [-0.1144, -0.0252,  0.0317,  0.1842, -0.0229,  0.0680]],
#        device='cuda:0', grad_fn=<ReshapeAliasBackward0>), 'box_logits': tensor([[-3.2835],
#         [-3.2921],
#         [-3.2927],
#         ...,
#         [-3.9670],
#         [-4.0268],
#         [-4.1811]], device='cuda:0', grad_fn=<ReshapeAliasBackward0>)}

# feat_i torch.Size([4, 128, 8, 112, 112]) tensor(-66467.7578, device='cuda:0', grad_fn=<MinBackward1>) tensor(75487.1641, device='cuda:0', grad_fn=<MaxBackward1>)
# feat_i torch.Size([4, 128, 4, 56, 56]) tensor(-26676.1172, device='cuda:0', grad_fn=<MinBackward1>) tensor(29650.8730, device='cuda:0', grad_fn=<MaxBackward1>)
# feat_i torch.Size([4, 128, 4, 28, 28]) tensor(-13842.2520, device='cuda:0', grad_fn=<MinBackward1>) tensor(10250.9951, device='cuda:0', grad_fn=<MaxBackward1>)
# feat_i torch.Size([4, 128, 4, 14, 14]) tensor(-741.7260, device='cuda:0', grad_fn=<MinBackward1>) tensor(865.0156, device='cuda:0', grad_fn=<MaxBackward1>)

# anchors [tensor([[ -2.0000,  -3.5000,   2.0000,   3.5000,  -3.5000,   3.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -4.5000,   4.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -5.5000,   5.5000],
#         ...,
#         [  4.0000, 176.0000,  20.0000, 240.0000, 180.0000, 236.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 172.0000, 244.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 164.0000, 252.0000]],
#        device='cuda:0'), tensor([[ -2.0000,  -3.5000,   2.0000,   3.5000,  -3.5000,   3.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -4.5000,   4.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -5.5000,   5.5000],
#         ...,
#         [  4.0000, 176.0000,  20.0000, 240.0000, 180.0000, 236.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 172.0000, 244.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 164.0000, 252.0000]],
#        device='cuda:0'), tensor([[ -2.0000,  -3.5000,   2.0000,   3.5000,  -3.5000,   3.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -4.5000,   4.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -5.5000,   5.5000],
#         ...,
#         [  4.0000, 176.0000,  20.0000, 240.0000, 180.0000, 236.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 172.0000, 244.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 164.0000, 252.0000]],
#        device='cuda:0'), tensor([[ -2.0000,  -3.5000,   2.0000,   3.5000,  -3.5000,   3.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -4.5000,   4.5000],
#         [ -2.0000,  -3.5000,   2.0000,   3.5000,  -5.5000,   5.5000],
#         ...,
#         [  4.0000, 176.0000,  20.0000, 240.0000, 180.0000, 236.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 172.0000, 244.0000],
#         [  4.0000, 176.0000,  20.0000, 240.0000, 164.0000, 252.0000]],



# test
# Transform:   0%|                                                                                                        | 0/1 [00:00<?, ?it/sfeat_i torch.Size([4, 128, 8, 112, 112]) tensor(-56896., device='cuda:0', dtype=torch.float16) tensor(58784., device='cuda:0', dtype=torch.float16) tensor(978., device='cuda:0', dtype=torch.float16)
# feat_i torch.Size([4, 128, 4, 56, 56]) tensor(-21232., device='cuda:0', dtype=torch.float16) tensor(22432., device='cuda:0', dtype=torch.float16) tensor(-100., device='cuda:0', dtype=torch.float16)
# feat_i torch.Size([4, 128, 4, 28, 28]) tensor(-9392., device='cuda:0', dtype=torch.float16) tensor(6904., device='cuda:0', dtype=torch.float16) tensor(26.7812, device='cuda:0', dtype=torch.float16)
# feat_i torch.Size([4, 128, 4, 14, 14]) tensor(-697.5000, device='cuda:0', dtype=torch.float16) tensor(792.5000, device='cuda:0', dtype=torch.float16) tensor(23.2656, device='cuda:0', dtype=torch.float16)
# test result debug {'box_deltas': tensor([[   -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [   -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
#         [   -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
#         ...,
#         [-0.0844, -0.0249, -0.0569, -0.0828, -0.0355, -0.0009],
#         [-0.0933, -0.0017, -0.0524, -0.0375, -0.0268, -0.1099],
#         [-0.0822, -0.0042, -0.0320,  0.0188,  0.0045, -0.1328]],
#        device='cuda:0', dtype=torch.float16), 'box_logits': tensor([[-inf],
#         [-inf],
#         [-inf],
#         ...,
#         [-inf],
#         [-inf],
#         [-inf]], device='cuda:0', dtype=torch.float16)}



# feat_i torch.Size([4, 128, 8, 112, 112]) tensor(-67769.7578, device='cuda:0', grad_fn=<MinBackward1>) tensor(73758.4141, device='cuda:0', grad_fn=<MaxBackward1>) tensor(1017.4573, device='cuda:0', grad_fn=<MeanBackward0>) torch.float32
# feat_i torch.Size([4, 128, 4, 56, 56]) tensor(-34232.6758, device='cuda:0', grad_fn=<MinBackward1>) tensor(37332.7031, device='cuda:0', grad_fn=<MaxBackward1>) tensor(-122.2526, device='cuda:0', grad_fn=<MeanBackward0>) torch.float32
# feat_i torch.Size([4, 128, 4, 28, 28]) tensor(-12806.6895, device='cuda:0', grad_fn=<MinBackward1>) tensor(11468.0576, device='cuda:0', grad_fn=<MaxBackward1>) tensor(21.2834, device='cuda:0', grad_fn=<MeanBackward0>) torch.float32
# feat_i torch.Size([4, 128, 4, 14, 14]) tensor(-834.0378, device='cuda:0', grad_fn=<MinBackward1>) tensor(971.3677, device='cuda:0', grad_fn=<MaxBackward1>) tensor(16.0727, device='cuda:0', grad_fn=<MeanBackward0>) torch.float32
# train result debug tensor(-2.0145, device='cuda:0', grad_fn=<MinBackward1>) tensor(12.3476, device='cuda:0', grad_fn=<MaxBackward1>) torch.float32
# anchors tensor(-44., device='cuda:0') tensor(252., device='cuda:0') torch.float32
# seg tensor(-134.3690, device='cuda:0', grad_fn=<MinBackward1>) tensor(417.7444, device='cuda:0', grad_fn=<MaxBackward1>) torch.float32
# Epoch 46:   0%|▏ 