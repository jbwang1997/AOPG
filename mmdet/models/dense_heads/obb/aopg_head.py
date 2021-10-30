import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, Scale

from mmdet.ops import DeformConv, arb_batched_nms, DeformConvPack

from mmdet.core import (get_bbox_dim, build_assigner, build_bbox_coder,
                        build_sampler, multi_apply, mintheta_obb, distance2bbox,
                        build_anchor_generator, images_to_levels, obb2hbb,
                        multiclass_arb_nms)
from mmdet.models.builder import build_loss
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.rpn_test_mixin import RPNTestMixin
from .obb_anchor_head import OBBAnchorHead

INF = 1e8


class AdaptiveConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AdaptiveConv, self).__init__()
        self.conv = DeformConv(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv, std=0.01)

    def forward(self, x, offset):
        N, _, H, W = x.shape
        assert offset.shape[0] == N * H * W
        offset = offset.reshape(N, H, W, -1)
        offset = offset.permute(0, 3, 1, 2)
        x = self.relu(self.conv(x, offset))
        return x


class CLM(nn.Module):

    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((-1, 45), (45, 90), (90, 180), (180, 360),
                               (360, INF)),
                 sigma=0.8,
                 theta_std=1.,
                 point_generator=dict(type='PointGenerator'),
                 loss_shape=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_theta=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_centerness=dict(
                     type='FocalLoss', use_sigmoid=True, loss_weight=1.0)):
        super(CLM, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.strides = strides
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.theta_std = theta_std

        self.point_generator = build_anchor_generator(point_generator)
        if loss_shape['type'] in ['IoULoss', 'GIoULoss', 'BoundedIoULoss']:
            self.reg_decoded_bbox = True
        else:
            self.reg_decoded_bbox = False

        self.loss_shape = build_loss(loss_shape)
        self.loss_theta = build_loss(loss_theta)
        self.loss_centerness = build_loss(loss_centerness)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.conv_shape = nn.Conv2d(self.feat_channels, 4, 1)
        self.conv_theta = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 1)

    def init_weights(self):
        normal_init(self.feat_conv, std=0.01)
        normal_init(self.conv_shape, std=0.01)
        normal_init(self.conv_theta, std=0.01)
        normal_init(self.conv_centerness, std=0.01)

    def forward_single(self, x, inds):
        x = self.relu(self.feat_conv(x))
        shape_pred = self.conv_shape(x)
        theta_pred = self.conv_theta(x)
        centerness = self.conv_centerness(x)

        num_imgs = x.size(0)
        points = self.point_generator.grid_points(
            shape_pred.size()[-2:], self.strides[inds], shape_pred.device)
        points = points.repeat(num_imgs, 1)

        anchors = self.get_anchor_single(
            points, shape_pred, theta_pred, self.base_edge_list[inds])
        offsets = self.get_offset_single(points, anchors)
        return shape_pred, theta_pred, centerness, anchors, offsets

    @torch.no_grad()
    def get_anchor_single(self, points, shape_pred, theta_pred, base_len):
        shape_pred = shape_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        shape_pred = shape_pred.exp() * base_len
        theta_pred = theta_pred.permute(0, 2, 3, 1).reshape(-1)
        theta_pred = theta_pred * self.theta_std

        r_x_offset = (shape_pred[:, 2] - shape_pred[:, 0]) / 2
        r_y_offset = (shape_pred[:, 3] - shape_pred[:, 1]) / 2
        Cos, Sin = torch.cos(theta_pred), torch.sin(theta_pred)
        x_offset = Cos * r_x_offset + Sin * r_y_offset
        y_offset = -Sin * r_x_offset + Cos * r_y_offset
        offset = torch.stack([x_offset, y_offset], axis=1)
        center = points[:, :2] + offset
        wh = shape_pred[:, :2] + shape_pred[:, 2:]

        anchors = torch.cat([center, wh, theta_pred[..., None]], dim=-1)
        anchors = mintheta_obb(anchors)
        return anchors

    def get_offset_single(self, points, anchors):
        kernel_size = 3
        strides = points[:, 2]
        x, y, w, h, theta = anchors.unbind(1)

        c_x_offset = (x - points[:, 0]) / strides
        c_y_offset = (y - points[:, 1]) / strides

        inds = torch.arange(-1, 2, dtype=anchors.dtype,
                            device=anchors.device)
        yy, xx = torch.meshgrid(inds, inds)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        dw, dh = w / (kernel_size - 1), h / (kernel_size - 1)
        dx, dy = dw[:, None] * xx, dh[:, None] * yy

        Cos, Sin = torch.cos(theta), torch.sin(theta)
        rx = Cos[:, None] * dx + Sin[:, None] * dy
        ry = -Sin[:, None] * dx + Cos[:, None] * dy
        rx, ry = rx / strides[:, None], ry / strides[:, None]
        s_x_offset = rx - xx
        s_y_offset = ry - yy

        offset_x = c_x_offset[:, None] + s_x_offset
        offset_y = c_y_offset[:, None] + s_y_offset
        offset = torch.stack([offset_x, offset_y], dim=-1)
        offset = offset.view(offset.size(0), -1)
        return offset

    def loss(self,
             shape_preds,
             theta_preds,
             centernesses,
             gt_bboxes):
        featmap_sizes = [featmap.size()[-2:] for featmap in shape_preds]
        points = [self.point_generator.grid_points(
            size, stride, shape_preds[0].device) for size, stride
            in zip(featmap_sizes, self.strides)]
        num_imgs = shape_preds[0].size(0)
        flatten_shape_preds = [
            shape_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for shape_pred in shape_preds
        ]
        flatten_theta_preds = [
            theta_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for theta_pred in theta_preds
        ]
        flatten_ctr_preds = [
            ctr.permute(0, 2, 3, 1).reshape(-1, 1)
            for ctr in centernesses
        ]
        flatten_shape_preds = torch.cat(flatten_shape_preds)
        flatten_theta_preds = torch.cat(flatten_theta_preds)
        flatten_ctr_preds = torch.cat(flatten_ctr_preds)
        flatten_shape_targets, flatten_theta_targets, flatten_ctr_targets = \
                self.get_targets(gt_bboxes, featmap_sizes, points)

        for base_len, p in zip(self.base_edge_list, points):
            p[:, 2] = base_len
        flatten_points = torch.cat(
            [p.repeat(num_imgs, 1) for p in points])
        pos_inds = (flatten_ctr_targets == 0).nonzero().view(-1)
        num_pos = len(pos_inds)
        loss_ctr = self.loss_centerness(
            flatten_ctr_preds,
            flatten_ctr_targets,
            avg_factor=flatten_ctr_targets.size(0)/200)
        if num_pos > 0:
            pos_points = flatten_points[pos_inds]
            pos_shape_preds = flatten_shape_preds[pos_inds]
            pos_shape_targets = flatten_shape_targets[pos_inds]
            if self.reg_decoded_bbox:
                loss_shape = self.loss_shape(
                    self.get_pseudo_bboxes(pos_points, pos_shape_preds),
                    self.get_pseudo_bboxes(pos_points, pos_shape_targets))
            else:
                loss_shape = self.loss_shape(
                    pos_shape_preds,
                    pos_shape_targets)
            pos_theta_preds = flatten_theta_preds[pos_inds]
            pos_theta_targets = flatten_theta_targets[pos_inds]
            loss_theta = self.loss_theta(
                pos_theta_preds,
                pos_theta_targets)
        else:
            loss_shape = flatten_shape_preds.sum() * 0.
            loss_theta = flatten_theta_preds.sum() * 0.
            loss_ctr = flatten_ctr_preds.sum() * 0.
        return dict(loss_aog_shape=loss_shape, loss_aog_theta=loss_theta,
                    loss_aog_ctr=loss_ctr)

    def get_pseudo_bboxes(self, points, shape):
        position, base_len = torch.split(points, [2, 1], dim=1)
        shape = shape.exp() * base_len

        x1 = position[:, 0] - shape[:, 0]
        y1 = position[:, 1] - shape[:, 1]
        x2 = position[:, 0] + shape[:, 2]
        y2 = position[:, 1] + shape[:, 3]
        return torch.stack([x1, y1, x2, y2], -1)

    def get_targets(self, gt_bbox_list, featmap_sizes, points):
        shape_target_list, theta_target_list, ctr_target_list = multi_apply(
            self._get_target_single,
            gt_bbox_list,
            featmap_size_list=featmap_sizes,
            point_list=points)
        flatten_shape_targets = [
            torch.cat(shape_targets_lvl)
            for shape_targets_lvl in zip(*shape_target_list)
        ]
        flatten_theta_targets = [
            torch.cat(theta_targets_lvl)
            for theta_targets_lvl in zip(*theta_target_list)
        ]
        flatten_ctr_targets = [
            torch.cat(ctr_targets_lvl)
            for ctr_targets_lvl in zip(*ctr_target_list)
        ]
        flatten_shape_targets = torch.cat(flatten_shape_targets)
        flatten_theta_targets = torch.cat(flatten_theta_targets)
        flatten_ctr_targets = torch.cat(flatten_ctr_targets)
        return flatten_shape_targets, flatten_theta_targets, flatten_ctr_targets

    def _get_target_single(self,
                           gt_bboxes,
                           featmap_size_list=None,
                           point_list=None):
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_areas = torch.sqrt(gt_bboxes[:, 2] * gt_bboxes[:, 3])
        shape_target_list = []
        theta_target_list = []
        ctr_target_list = []
        for base_len, (lower_bound, upper_bound), featmap_size, points \
                in zip(self.base_edge_list, self.scale_ranges,
                       featmap_size_list, point_list):
            if points.size(-1) == 3:
                points = points[:, :-1]
            shape_targets = gt_bboxes.new(
                featmap_size[0]*featmap_size[1], 4) + 1
            theta_targets = gt_bboxes.new_zeros(
                featmap_size[0]*featmap_size[1], 1)
            # ctr_targets = gt_bboxes.new_full(
                # (featmap_size[0]*featmap_size[1], ), -1).long()
            ctr_targets = gt_bboxes.new_ones(
                (featmap_size[0]*featmap_size[1], )).long()

            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                shape_target_list.append(torch.log(shape_targets))
                theta_target_list.append(theta_targets)
                ctr_target_list.append(ctr_targets)
                continue

            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes_lvl = gt_bboxes[hit_indices, :]
            num_gts = gt_bboxes_lvl.size(0)
            num_points = points.size(0)

            gt_center, gt_wh, gt_theta = torch.split(
                gt_bboxes_lvl, [2, 2, 1], dim=1)
            Cos, Sin = torch.cos(gt_theta), torch.sin(gt_theta)
            gt_Matrix = torch.cat([Cos, -Sin, Sin, Cos], dim=1)
            gt_Matrix = gt_Matrix.view(num_gts, 2, 2)

            points_epd = points[:, None].expand(num_points, num_gts, 2)
            points_ctr = (points_epd - gt_center).permute(1, 0, 2)
            rpoints_ctr = torch.matmul(points_ctr, gt_Matrix.transpose(-1, -2))
            rpoints_ctr = rpoints_ctr.permute(1, 0, 2)

            for i in range(num_gts):
                rpts_ctr = rpoints_ctr[:, i, :]
                theta = gt_theta[i]
                W, H = gt_wh[i]
                W_shrink, H_shrink = gt_wh[i] * (1 - self.sigma)

                inside_shrink_mask = ((torch.abs(rpts_ctr[:, 0]) < W_shrink/2)
                                      & (torch.abs(rpts_ctr[:, 1]) < H_shrink/2))
                outside_shrink_mask = ~inside_shrink_mask
                inside_bbox_mask = ((torch.abs(rpts_ctr[:, 0]) < W/2)
                                    & (torch.abs(rpts_ctr[:, 1]) < H/2))
                inside_rpts_ctr = rpts_ctr[inside_shrink_mask]
                left = W/2 + inside_rpts_ctr[..., 0]
                right = W/2 - inside_rpts_ctr[..., 0]
                top = H/2 + inside_rpts_ctr[..., 1]
                bottom = H/2 - inside_rpts_ctr[..., 1]

                shape_targets[inside_shrink_mask] = torch.stack(
                    [left, top, right, bottom], dim=-1) / base_len
                theta_targets[inside_shrink_mask] = theta
                ctr_targets[inside_shrink_mask] = 0
                # ctr_targets[inside_bbox_mask & outside_shrink_mask] = 1
            shape_targets = shape_targets.clamp(min=1. /16, max=16.)
            shape_target_list.append(torch.log(shape_targets))
            theta_target_list.append(theta_targets / self.theta_std)
            ctr_target_list.append(ctr_targets)
        return shape_target_list, theta_target_list, ctr_target_list


@HEADS.register_module()
class AOPGHead(OBBAnchorHead, RPNTestMixin):

    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((-1, 45), (45, 90), (90, 180), (180, 360),
                               (360, INF)),
                 sigma=0.8,
                 theta_std=1.,
                 point_generator=dict(type='PointGenerator'),
                 bbox_coder=dict(
                     type='OBB2OBBDeltaXYWHTCoder',
                     target_means=[.0, .0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 loss_shape=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_theta=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_centerness=dict(
                     type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=True,loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(OBBAnchorHead, self).__init__()
        self.num_classes = 1
        self.bbox_type = 'obb'
        self.reg_dim = 5
        self.reg_decoded_bbox = False
        self.background_label = 0
        self.num_anchors = 1
        self.fp16_enabled = False

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.clm = CLM(in_channels,
                       feat_channels,
                       strides,
                       base_edge_list,
                       scale_ranges,
                       sigma,
                       theta_std,
                       point_generator,
                       loss_shape,
                       loss_theta,
                       loss_centerness)

        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self._init_layers()

    def _init_layers(self):
        self.feature_adaption = AdaptiveConv(
            self.in_channels, self.feat_channels)
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.reg_dim, 1)

    def init_weights(self):
        self.clm.init_weights()
        self.feature_adaption.init_weights()
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x, inds):
        shape_pred, theta_pred, centerness, anchors, offsets = \
                self.clm.forward_single(x, inds)
        x = self.feature_adaption(x, offsets)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return (cls_score, bbox_pred, shape_pred, theta_pred,
                centerness, anchors)

    def forward(self, feats):
        return multi_apply(
            self.forward_single, feats, list(range(len(feats))))

    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             theta_preds,
             centernesses,
             anchors,
             gt_bboxes,
             img_metas,
             gt_bboxes_ignore=None):
        loss = dict()
        loss.update(self.clm.loss(shape_preds, theta_preds, centernesses, gt_bboxes))

        num_imgs = cls_scores[0].size(0)
        anchor_list = [[] for _ in range(num_imgs)]
        valid_flag_list = [[] for _ in range(num_imgs)]
        for lvl_anchors in anchors:
            lvl_anchors = torch.chunk(lvl_anchors, num_imgs, dim=0)
            for img_id, lvl_img_anchors in enumerate(lvl_anchors):
                anchor_list[img_id].append(lvl_img_anchors)
                valid_flags = anchors[0].new_ones(
                    (lvl_img_anchors.size(0), ), dtype=torch.bool)
                pad_shape = img_metas[img_id]['pad_shape']
                pad_h, pad_w = pad_shape[:2]
                valid_flags[lvl_img_anchors[:, 0] > pad_w] = 0
                valid_flags[lvl_img_anchors[:, 1] > pad_h] = 0
                valid_flag_list[img_id].append(valid_flags)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        loss.update(dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox))
        return loss

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   shape_preds,
                   theta_preds,
                   centernesses,
                   anchors,
                   img_metas,
                   cfg=None,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        num_imgs = len(img_metas)

        anchor_list = [[] for _ in range(num_imgs)]
        for anchors_lvl in anchors:
            anchors_lvl = torch.chunk(anchors_lvl, num_imgs, dim=0)
            for i in range(num_imgs):
                anchor_list[i].append(anchors_lvl[i])

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)]
            img_anchors = anchor_list[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_list, img_anchors,
                                                img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_centerness = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            rpn_centerness = centernesses[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]\
                    == rpn_centerness.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # we set FG labels to [0, num_class-1] and BG label to
                # num_class in other heads since mmdet v2.0, However we
                # keep BG label as 0 and FG label as 1 in rpn head
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_dim)
            rpn_centerness = rpn_centerness.permute(1, 2, 0).reshape(-1)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                rpn_centerness = rpn_centerness[topk_inds]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_centerness.append(rpn_centerness)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        rpn_centerness = torch.cat(mlvl_centerness)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w, h = proposals[:, 2], proposals[:, 3]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                rpn_centerness = rpn_centerness[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        hproposals = obb2hbb(proposals)
        scores = (scores + torch.sigmoid(rpn_centerness)) / 2
        nms_cfg = dict(type='nms', iou_thr=cfg.nms_thr)
        _, keep = arb_batched_nms(hproposals, scores, ids, nms_cfg)

        dets = torch.cat([proposals, scores[:, None]], dim=1)
        dets = dets[keep]
        return dets[:cfg.nms_post]
