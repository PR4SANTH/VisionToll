"""run_stage1_forensics.py

Forensic diagnostic script for VisionTollNet Stage 1 failure analysis.
Instruments model, data, assignments, loss decomposition, prediction statistics,
and gradient norms across Stage 0, Stage 1 (best), and Stage 1 (last).
"""

from __future__ import annotations
import sys
import os
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.utils.data import DataLoader

# Project paths
ROOT = Path("d:/VisionToll")
sys.path.insert(0, str(ROOT / "src"))

from models.vision_toll_net.vision_toll_net import VisionTollNet
from models.vision_toll_net.dataset import VisionTollDataset, collate_fn
from models.vision_toll_net.assigner import task_aligned_assigner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "results" / "diagnostics" / "visiontollnet_stage1_forensics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(stage: str, weights_path: Path | None = None) -> VisionTollNet:
    """Instantiate and optionally load checkpoint weights."""
    model = VisionTollNet(num_classes=5, pretrained=False, stage=stage).to(DEVICE)
    if weights_path is not None and weights_path.exists():
        ckpt = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {weights_path.name} (epoch {ckpt.get('epoch', 'N/A')}, mAP50-95: {ckpt.get('best_metric', 'N/A')})")
    return model


def get_sample_batches():
    """Retrieve one deterministic train batch and one val batch."""
    data_dir = ROOT / "data" / "processed" / "vision_toll_yolo"
    train_img_dir = data_dir / "images" / "train"
    train_lbl_dir = data_dir / "labels" / "train"
    val_img_dir = data_dir / "images" / "val"
    val_lbl_dir = data_dir / "labels" / "val"

    train_ds = VisionTollDataset(train_img_dir, train_lbl_dir, img_size=640)
    val_ds = VisionTollDataset(val_img_dir, val_lbl_dir, img_size=640)

    torch.manual_seed(42)
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    return train_batch, val_batch


def instrument_assignments(model: VisionTollNet, images: torch.Tensor, targets: list[dict], batch_name: str):
    """Analyze assignment distribution across feature levels P2, P3, P4, P5."""
    B, _, H, W = images.shape
    device = images.device

    backbone_feats = model.backbone(images)
    pyramid_feats = model._c_to_p(backbone_feats)
    fused_feats = model.fusion(pyramid_feats, use_adaptive=model.use_adaptive_fusion)

    with torch.amp.autocast("cuda", enabled=False):
        fused_fp32 = {k: v.float() for k, v in fused_feats.items()}
        head_out = model.head(fused_fp32)

        cls_lvl = []
        reg_lvl = []
        level_point_counts = {}
        for lvl in model.levels:
            cls = head_out["cls"][lvl]
            reg = head_out["reg"][lvl]
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 5)
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 5)
            cls_lvl.append(cls)
            reg_lvl.append(reg)
            level_point_counts[lvl] = cls.shape[1]

        cls_pred = torch.cat(cls_lvl, dim=1)
        reg_pred = torch.cat(reg_lvl, dim=1)
        points, stride_vec, level_vec = model._generate_grid_points(H, W, device)
        decoded_boxes = model.decode_boxes(points, stride_vec, reg_pred[..., :4])
        pred_scores = torch.sigmoid(cls_pred)

    total_gt = sum(len(t["boxes"]) for t in targets)
    assignments_per_level = {lvl: 0 for lvl in model.levels}
    iou_scores = []
    task_scores = []
    gt_matched_count = 0
    gt_total_count = total_gt

    # Track per-image assignments
    for i in range(B):
        gt = targets[i]
        gt_b = gt["boxes"].to(device).float()
        gt_l = gt["labels"].to(device)
        M_i = len(gt_b)

        res = task_aligned_assigner(
            points=points,
            stride_vec=stride_vec,
            pred_boxes=decoded_boxes[i].detach(),
            pred_scores=pred_scores[i].detach(),
            gt_boxes=gt_b,
            gt_labels=gt_l,
            topk=10,
            alpha=0.5,
            beta=2.0,
            basic_assignment=model.basic_assignment,
        )

        fg_mask = res["fg_mask"]
        assigned_boxes = res["assigned_boxes"]
        pos_levels = level_vec[fg_mask]

        for lvl_idx, lvl in enumerate(model.levels):
            assignments_per_level[lvl] += (pos_levels == lvl_idx).sum().item()

        if fg_mask.any():
            ious = ops.box_iou(decoded_boxes[i][fg_mask], assigned_boxes[fg_mask])
            diag_ious = torch.diagonal(ious)
            iou_scores.extend(diag_ious.cpu().tolist())

        # Check matched GTs
        assigned_labels = res["assigned_labels"]
        for j in range(M_i):
            if (assigned_labels == gt_l[j]).any():
                gt_matched_count += 1

    total_candidates = len(points) * B
    total_positives = sum(assignments_per_level.values())
    total_negatives = total_candidates - total_positives

    return {
        "batch_name": batch_name,
        "stage": model.stage,
        "total_gt": total_gt,
        "gt_matched": gt_matched_count,
        "total_candidates": total_candidates,
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "pos_neg_ratio": f"1:{total_negatives / max(1, total_positives):.1f}",
        "assignments_per_level": assignments_per_level,
        "level_point_counts_per_img": level_point_counts,
        "iou_mean": float(torch.tensor(iou_scores).mean().item()) if iou_scores else 0.0,
        "iou_median": float(torch.tensor(iou_scores).median().item()) if iou_scores else 0.0,
        "iou_min": float(torch.tensor(iou_scores).min().item()) if iou_scores else 0.0,
        "iou_max": float(torch.tensor(iou_scores).max().item()) if iou_scores else 0.0,
    }


def instrument_loss_breakdown(model: VisionTollNet, images: torch.Tensor, targets: list[dict]):
    """Detailed loss decomposition by feature level and pos/neg components."""
    B, _, H, W = images.shape
    device = images.device

    backbone_feats = model.backbone(images)
    pyramid_feats = model._c_to_p(backbone_feats)
    fused_feats = model.fusion(pyramid_feats, use_adaptive=model.use_adaptive_fusion)

    with torch.amp.autocast("cuda", enabled=False):
        fused_fp32 = {k: v.float() for k, v in fused_feats.items()}
        head_out = model.head(fused_fp32)

        cls_lvl = []
        reg_lvl = []
        level_slices = {}
        curr_idx = 0
        for lvl in model.levels:
            cls = head_out["cls"][lvl]
            reg = head_out["reg"][lvl]
            cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 5)
            reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 5)
            cls_lvl.append(cls)
            reg_lvl.append(reg)
            num_pts = cls.shape[1]
            level_slices[lvl] = (curr_idx, curr_idx + num_pts)
            curr_idx += num_pts

        cls_pred = torch.cat(cls_lvl, dim=1)
        reg_pred = torch.cat(reg_lvl, dim=1)
        points, stride_vec, level_vec = model._generate_grid_points(H, W, device)
        decoded_boxes = model.decode_boxes(points, stride_vec, reg_pred[..., :4])
        pred_scores = torch.sigmoid(cls_pred)
        pred_quality = reg_pred[..., 4]

        all_target_scores = []
        all_fg_masks = []
        all_assigned_boxes = []
        total_positives = 0

        for i in range(B):
            gt = targets[i]
            gt_b = gt["boxes"].to(device).float()
            gt_l = gt["labels"].to(device)

            res = task_aligned_assigner(
                points=points,
                stride_vec=stride_vec,
                pred_boxes=decoded_boxes[i].detach(),
                pred_scores=pred_scores[i].detach(),
                gt_boxes=gt_b,
                gt_labels=gt_l,
                topk=10,
                alpha=0.5,
                beta=2.0,
                basic_assignment=model.basic_assignment,
            )
            all_target_scores.append(res["target_scores"])
            all_fg_masks.append(res["fg_mask"])
            all_assigned_boxes.append(res["assigned_boxes"])
            total_positives += res["fg_mask"].sum().item()

        batch_target_scores = torch.stack(all_target_scores, dim=0)
        batch_fg_masks = torch.stack(all_fg_masks, dim=0)
        batch_assigned_boxes = torch.stack(all_assigned_boxes, dim=0)

        # Focal loss per point
        focal_loss_all = ops.sigmoid_focal_loss(
            cls_pred.clamp(min=-30.0, max=30.0), batch_target_scores, alpha=0.25, gamma=2.0, reduction="none"
        )
        focal_loss_all = torch.nan_to_num(focal_loss_all, nan=0.0, posinf=10.0, neginf=0.0)
        norm = max(float(total_positives), float(B * 10))
        total_cls_loss = focal_loss_all.sum() / norm

        # Separate positive vs negative focal loss
        # Point is positive if fg_mask is True for any class target > 0
        fg_mask_expanded = batch_fg_masks.unsqueeze(-1).expand_as(batch_target_scores)
        pos_focal_loss = (focal_loss_all * (batch_target_scores > 0).float()).sum() / norm
        neg_focal_loss = (focal_loss_all * (batch_target_scores == 0).float()).sum() / norm

        # Per level classification loss breakdown
        level_cls_loss = {}
        level_pos_loss = {}
        level_neg_loss = {}
        for lvl, (start, end) in level_slices.items():
            lvl_focal = focal_loss_all[:, start:end, :]
            lvl_targets = batch_target_scores[:, start:end, :]
            level_cls_loss[lvl] = float((lvl_focal.sum() / norm).item())
            level_pos_loss[lvl] = float(((lvl_focal * (lvl_targets > 0).float()).sum() / norm).item())
            level_neg_loss[lvl] = float(((lvl_focal * (lvl_targets == 0).float()).sum() / norm).item())

        # Box regression breakdown
        pred_pos_boxes = decoded_boxes[batch_fg_masks]
        target_pos_boxes = batch_assigned_boxes[batch_fg_masks]
        if total_positives > 0:
            ciou_loss = ops.complete_box_iou_loss(pred_pos_boxes, target_pos_boxes, reduction="none", eps=1e-7)
            ciou_loss = torch.nan_to_num(ciou_loss, nan=1.0, posinf=2.0, neginf=0.0)
            total_reg_loss = ciou_loss.mean()
        else:
            ciou_loss = torch.empty(0, device=device)
            total_reg_loss = torch.tensor(0.0, device=device)

        # Per level regression loss
        level_reg_loss = {}
        pos_level_vec = level_vec.unsqueeze(0).expand(B, -1)[batch_fg_masks]
        for lvl_idx, lvl in enumerate(model.levels):
            mask_lvl = (pos_level_vec == lvl_idx)
            if mask_lvl.any():
                level_reg_loss[lvl] = float(ciou_loss[mask_lvl].mean().item())
            else:
                level_reg_loss[lvl] = 0.0

    return {
        "norm": norm,
        "total_positives": total_positives,
        "total_cls_loss": float(total_cls_loss.item()),
        "pos_focal_loss": float(pos_focal_loss.item()),
        "neg_focal_loss": float(neg_focal_loss.item()),
        "neg_to_pos_loss_ratio": float((neg_focal_loss / max(pos_focal_loss, 1e-6)).item()),
        "level_cls_loss": level_cls_loss,
        "level_pos_loss": level_pos_loss,
        "level_neg_loss": level_neg_loss,
        "total_reg_loss": float(total_reg_loss.item()),
        "level_reg_loss": level_reg_loss,
    }


def inspect_prediction_stats(model: VisionTollNet, images: torch.Tensor):
    """Extract distribution statistics for logits, probabilities, and box dimensions."""
    B, _, H, W = images.shape
    device = images.device

    model.eval()
    with torch.no_grad():
        backbone_feats = model.backbone(images)
        pyramid_feats = model._c_to_p(backbone_feats)
        fused_feats = model.fusion(pyramid_feats, use_adaptive=model.use_adaptive_fusion)

        with torch.amp.autocast("cuda", enabled=False):
            fused_fp32 = {k: v.float() for k, v in fused_feats.items()}
            head_out = model.head(fused_fp32)

            stats = {}
            for lvl in model.levels:
                cls_raw = head_out["cls"][lvl]  # (B, 5, H_l, W_l)
                reg_raw = head_out["reg"][lvl]  # (B, 5, H_l, W_l)
                prob = torch.sigmoid(cls_raw)

                # Quality logit
                qual_logit = reg_raw[:, 4, :, :]
                qual_prob = torch.sigmoid(qual_logit)

                # Box regression
                stride = {"P2": 4, "P3": 8, "P4": 16, "P5": 32}[lvl]
                offsets = F.softplus(reg_raw[:, :4, :, :]) * stride
                pred_w = offsets[:, 0, :, :] + offsets[:, 2, :, :]
                pred_h = offsets[:, 1, :, :] + offsets[:, 3, :, :]

                stats[lvl] = {
                    "logit_min": float(cls_raw.min().item()),
                    "logit_max": float(cls_raw.max().item()),
                    "logit_mean": float(cls_raw.mean().item()),
                    "logit_std": float(cls_raw.std().item()),
                    "prob_min": float(prob.min().item()),
                    "prob_max": float(prob.max().item()),
                    "prob_mean": float(prob.mean().item()),
                    "pct_prob_gt_0_05": float((prob > 0.05).float().mean().item() * 100.0),
                    "pct_prob_gt_0_25": float((prob > 0.25).float().mean().item() * 100.0),
                    "box_w_mean": float(pred_w.mean().item()),
                    "box_w_max": float(pred_w.max().item()),
                    "box_h_mean": float(pred_h.mean().item()),
                    "box_h_max": float(pred_h.max().item()),
                    "quality_logit_mean": float(qual_logit.mean().item()),
                    "quality_prob_mean": float(qual_prob.mean().item()),
                }

    return stats


def measure_gradient_norms(model: VisionTollNet, images: torch.Tensor, targets: list[dict]):
    """Measure gradient norms across backbone, fusion, head, and per-level head branches."""
    model.train()
    model.zero_grad()

    # Register hooks on head per-level outputs to isolate per-level gradient norms
    per_level_grads = {}

    def get_hook(name):
        def hook(grad):
            per_level_grads[name] = grad.detach().norm(2).item()
        return hook

    B, _, H, W = images.shape
    device = images.device

    backbone_feats = model.backbone(images)
    pyramid_feats = model._c_to_p(backbone_feats)
    fused_feats = model.fusion(pyramid_feats, use_adaptive=model.use_adaptive_fusion)

    with torch.amp.autocast("cuda", enabled=False):
        fused_fp32 = {k: v.float() for k, v in fused_feats.items()}
        head_out = model.head(fused_fp32)

        # Attach hooks to per-level predictions
        for lvl in model.levels:
            head_out["cls"][lvl].register_hook(get_hook(f"cls_{lvl}"))
            head_out["reg"][lvl].register_hook(get_hook(f"reg_{lvl}"))

        cls_lvl = []
        reg_lvl = []
        for lvl in model.levels:
            cls = head_out["cls"][lvl].permute(0, 2, 3, 1).reshape(B, -1, 5)
            reg = head_out["reg"][lvl].permute(0, 2, 3, 1).reshape(B, -1, 5)
            cls_lvl.append(cls)
            reg_lvl.append(reg)

        cls_pred = torch.cat(cls_lvl, dim=1)
        reg_pred = torch.cat(reg_lvl, dim=1)
        points, stride_vec, level_vec = model._generate_grid_points(H, W, device)
        decoded_boxes = model.decode_boxes(points, stride_vec, reg_pred[..., :4])
        pred_scores = torch.sigmoid(cls_pred)

        all_target_scores = []
        all_fg_masks = []
        all_assigned_boxes = []
        total_positives = 0

        for i in range(B):
            gt = targets[i]
            res = task_aligned_assigner(
                points=points,
                stride_vec=stride_vec,
                pred_boxes=decoded_boxes[i].detach(),
                pred_scores=pred_scores[i].detach(),
                gt_boxes=gt["boxes"].to(device).float(),
                gt_labels=gt["labels"].to(device),
                topk=10,
                alpha=0.5,
                beta=2.0,
                basic_assignment=model.basic_assignment,
            )
            all_target_scores.append(res["target_scores"])
            all_fg_masks.append(res["fg_mask"])
            all_assigned_boxes.append(res["assigned_boxes"])
            total_positives += res["fg_mask"].sum().item()

        batch_target_scores = torch.stack(all_target_scores, dim=0)
        batch_fg_masks = torch.stack(all_fg_masks, dim=0)
        batch_assigned_boxes = torch.stack(all_assigned_boxes, dim=0)

        focal_loss_all = ops.sigmoid_focal_loss(
            cls_pred.clamp(min=-30.0, max=30.0), batch_target_scores, alpha=0.25, gamma=2.0, reduction="none"
        )
        focal_loss_all = torch.nan_to_num(focal_loss_all, nan=0.0, posinf=10.0, neginf=0.0)
        norm = max(float(total_positives), float(B * 10))
        cls_loss = focal_loss_all.sum() / norm

        pred_pos_boxes = decoded_boxes[batch_fg_masks]
        target_pos_boxes = batch_assigned_boxes[batch_fg_masks]
        if total_positives > 0:
            ciou_loss = ops.complete_box_iou_loss(pred_pos_boxes, target_pos_boxes, reduction="none", eps=1e-7)
            ciou_loss = torch.nan_to_num(ciou_loss, nan=1.0, posinf=2.0, neginf=0.0)
            reg_loss = ciou_loss.mean()
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = cls_loss + 2.0 * reg_loss

    total_loss.backward()

    # Collect parameter gradient norms by module
    def calc_module_grad_norm(module: nn.Module) -> float:
        total_norm_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm_sq += param_norm ** 2
        return total_norm_sq ** 0.5

    grad_norms = {
        "backbone": calc_module_grad_norm(model.backbone),
        "backbone_layer1_C2": calc_module_grad_norm(model.backbone.layer1),
        "fusion": calc_module_grad_norm(model.fusion),
        "fusion_conv_P2": calc_module_grad_norm(model.fusion.conv["P2"]) if "P2" in model.fusion.conv else 0.0,
        "fusion_conv_P3": calc_module_grad_norm(model.fusion.conv["P3"]),
        "cls_head": calc_module_grad_norm(model.head.cls_head),
        "reg_head": calc_module_grad_norm(model.head.reg_head),
        "per_level_activations": per_level_grads,
    }

    return grad_norms


def verify_p2_mechanics(model: VisionTollNet, images: torch.Tensor, targets: list[dict]):
    """Rigorous programmatic assertions for P2 forward, assign, loss, and backward."""
    B, _, H, W = images.shape
    device = images.device

    # 1. Forward check
    model.eval()
    with torch.no_grad():
        out = model(images)
        p2_cls_out = out["cls"]["P2"]
        p2_reg_out = out["reg"]["P2"]
        p2_has_pred = (p2_cls_out.shape == (B, 5, 160, 160)) and (p2_reg_out.shape == (B, 5, 160, 160))

    # 2. Assignment check
    points, stride_vec, level_vec = model._generate_grid_points(H, W, device)
    res = task_aligned_assigner(
        points=points,
        stride_vec=stride_vec,
        pred_boxes=out["decoded_boxes"][0].detach(),
        pred_scores=out["pred_scores"][0].detach(),
        gt_boxes=targets[0]["boxes"].to(device).float(),
        gt_labels=targets[0]["labels"].to(device),
        topk=10,
        alpha=0.5,
        beta=2.0,
        basic_assignment=model.basic_assignment,
    )
    p2_pos_count = (level_vec[res["fg_mask"]] == 0).sum().item()

    # 3. Loss & Backward check
    model.train()
    model.zero_grad()
    loss_dict = model(images, targets)
    total_loss = sum(loss_dict.values())
    total_loss.backward()

    p2_conv_grad = model.fusion.conv["P2"].weight.grad
    p2_has_grad = p2_conv_grad is not None and torch.isfinite(p2_conv_grad).all() and (p2_conv_grad.norm().item() > 0)

    return {
        "p2_forward_valid": p2_has_pred,
        "p2_cls_shape": list(p2_cls_out.shape),
        "p2_reg_shape": list(p2_reg_out.shape),
        "p2_sample_assigned_positives": p2_pos_count,
        "p2_fusion_conv_has_finite_nonzero_grad": bool(p2_has_grad),
        "p2_fusion_conv_grad_norm": float(p2_conv_grad.norm().item()) if p2_has_grad else 0.0,
    }


def main():
    print("=" * 70)
    print("VISIONTOLLNET STAGE 1 FORENSIC DIAGNOSIS")
    print("=" * 70)

    # 1. Load Batches
    print("\n1. Loading deterministic sample batches from train and val splits...")
    train_batch, val_batch = get_sample_batches()
    train_imgs, train_tgts = train_batch[0].to(DEVICE), train_batch[1]
    val_imgs, val_tgts = val_batch[0].to(DEVICE), val_batch[1]
    print(f"Train batch: {train_imgs.shape}, {sum(len(t['boxes']) for t in train_tgts)} GT boxes across 4 images")
    print(f"Val batch: {val_imgs.shape}, {sum(len(t['boxes']) for t in val_tgts)} GT boxes across 4 images")

    # 2. Load Models
    print("\n2. Loading Model Checkpoints:")
    stage0_best_path = ROOT / "results" / "experiments" / "visiontollnet_stage0" / "weights" / "best.pt"
    stage1_best_path = ROOT / "results" / "experiments" / "visiontollnet_stage1_p2" / "weights" / "best.pt"
    stage1_last_path = ROOT / "results" / "experiments" / "visiontollnet_stage1_p2" / "weights" / "last.pt"

    m_stage0_best = load_model("stage0", stage0_best_path)
    m_stage1_best = load_model("stage1", stage1_best_path)
    m_stage1_last = load_model("stage1", stage1_last_path)

    # 3. Assignment Analysis
    print("\n3. Running Assignment Instrumentation...")
    assign_s0_val = instrument_assignments(m_stage0_best, val_imgs, val_tgts, "val_stage0_best")
    assign_s1_best_val = instrument_assignments(m_stage1_best, val_imgs, val_tgts, "val_stage1_best")
    assign_s1_last_val = instrument_assignments(m_stage1_last, val_imgs, val_tgts, "val_stage1_last")
    assign_s1_best_trn = instrument_assignments(m_stage1_best, train_imgs, train_tgts, "trn_stage1_best")

    # 4. Loss Decompositions
    print("\n4. Running Loss Decompositions...")
    loss_s0_val = instrument_loss_breakdown(m_stage0_best, val_imgs, val_tgts)
    loss_s1_best_val = instrument_loss_breakdown(m_stage1_best, val_imgs, val_tgts)
    loss_s1_last_val = instrument_loss_breakdown(m_stage1_last, val_imgs, val_tgts)
    loss_s1_best_trn = instrument_loss_breakdown(m_stage1_best, train_imgs, train_tgts)

    # 5. Prediction Statistics
    print("\n5. Extracting Prediction Statistics (logits, probabilities, boxes)...")
    pred_stats_s0 = inspect_prediction_stats(m_stage0_best, val_imgs)
    pred_stats_s1_best = inspect_prediction_stats(m_stage1_best, val_imgs)
    pred_stats_s1_last = inspect_prediction_stats(m_stage1_last, val_imgs)

    # 6. Gradient Norms
    print("\n6. Measuring Gradient Norms Across Modules and Levels...")
    grads_s0 = measure_gradient_norms(m_stage0_best, train_imgs, train_tgts)
    grads_s1_best = measure_gradient_norms(m_stage1_best, train_imgs, train_tgts)
    grads_s1_last = measure_gradient_norms(m_stage1_last, train_imgs, train_tgts)

    # 7. Explicit P2 Mechanics Verification
    print("\n7. Verifying P2 Mechanics (forward, assign, loss, backward)...")
    p2_check = verify_p2_mechanics(m_stage1_best, train_imgs, train_tgts)

    # Compile comprehensive forensic data dict
    forensics_data = {
        "date": "2026-10-01",
        "assignment_instrumentation": {
            "stage0_best_val": assign_s0_val,
            "stage1_best_val": assign_s1_best_val,
            "stage1_last_val": assign_s1_last_val,
            "stage1_best_train": assign_s1_best_trn,
        },
        "loss_decomposition": {
            "stage0_best_val": loss_s0_val,
            "stage1_best_val": loss_s1_best_val,
            "stage1_last_val": loss_s1_last_val,
            "stage1_best_train": loss_s1_best_trn,
        },
        "prediction_statistics": {
            "stage0_best_val": pred_stats_s0,
            "stage1_best_val": pred_stats_s1_best,
            "stage1_last_val": pred_stats_s1_last,
        },
        "gradient_norms": {
            "stage0_best": grads_s0,
            "stage1_best": grads_s1_best,
            "stage1_last": grads_s1_last,
        },
        "p2_mechanics_verification": p2_check,
    }

    # Save JSON
    json_path = OUT_DIR / "stage1_forensics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(forensics_data, f, indent=2)
    print(f"\nSaved forensics data to: {json_path}")

    # Save CSV: assignments
    csv_assign_path = OUT_DIR / "batch_assignment_stats.csv"
    with open(csv_assign_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment_Batch", "Stage", "Total_GT", "GT_Matched", "Total_Candidates", "Total_Positives", "Total_Negatives", "Pos_Neg_Ratio", "P2_Pos", "P3_Pos", "P4_Pos", "P5_Pos", "IoU_Mean", "IoU_Max"])
        for k, v in forensics_data["assignment_instrumentation"].items():
            lvl = v["assignments_per_level"]
            writer.writerow([
                k, v["stage"], v["total_gt"], v["gt_matched"], v["total_candidates"], v["total_positives"], v["total_negatives"], v["pos_neg_ratio"],
                lvl.get("P2", 0), lvl.get("P3", 0), lvl.get("P4", 0), lvl.get("P5", 0),
                f"{v['iou_mean']:.4f}", f"{v['iou_max']:.4f}"
            ])
    print(f"Saved assignment CSV to: {csv_assign_path}")

    # Save CSV: loss breakdown
    csv_loss_path = OUT_DIR / "level_loss_breakdown.csv"
    with open(csv_loss_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment_Batch", "Norm", "Total_Positives", "Total_Cls_Loss", "Pos_Focal_Loss", "Neg_Focal_Loss", "Neg_Pos_Ratio", "P2_Cls", "P3_Cls", "P4_Cls", "P5_Cls", "Total_Reg_Loss", "P2_Reg", "P3_Reg", "P4_Reg", "P5_Reg"])
        for k, v in forensics_data["loss_decomposition"].items():
            cls_lvl = v["level_cls_loss"]
            reg_lvl = v["level_reg_loss"]
            writer.writerow([
                k, v["norm"], v["total_positives"], f"{v['total_cls_loss']:.4f}", f"{v['pos_focal_loss']:.4f}", f"{v['neg_focal_loss']:.4f}", f"{v['neg_to_pos_loss_ratio']:.2f}",
                f"{cls_lvl.get('P2', 0):.4f}", f"{cls_lvl.get('P3', 0):.4f}", f"{cls_lvl.get('P4', 0):.4f}", f"{cls_lvl.get('P5', 0):.4f}",
                f"{v['total_reg_loss']:.4f}", f"{reg_lvl.get('P2', 0):.4f}", f"{reg_lvl.get('P3', 0):.4f}", f"{reg_lvl.get('P4', 0):.4f}", f"{reg_lvl.get('P5', 0):.4f}"
            ])
    print(f"Saved loss breakdown CSV to: {csv_loss_path}")

    # Print summary of results to stdout for inspection
    print("\n--- Summary of Findings ---")
    print(f"Stage 0 Val: Total points = {assign_s0_val['total_candidates']}, Positives = {assign_s0_val['total_positives']}, Negatives = {assign_s0_val['total_negatives']} ({assign_s0_val['pos_neg_ratio']})")
    print(f"Stage 1 Val: Total points = {assign_s1_best_val['total_candidates']}, Positives = {assign_s1_best_val['total_positives']}, Negatives = {assign_s1_best_val['total_negatives']} ({assign_s1_best_val['pos_neg_ratio']})")
    print(f"Stage 1 Assignments per level: P2={assign_s1_best_val['assignments_per_level'].get('P2')}, P3={assign_s1_best_val['assignments_per_level'].get('P3')}, P4={assign_s1_best_val['assignments_per_level'].get('P4')}, P5={assign_s1_best_val['assignments_per_level'].get('P5')}")
    print(f"Stage 0 Val Loss: Total Cls={loss_s0_val['total_cls_loss']:.4f}, Pos Focal={loss_s0_val['pos_focal_loss']:.4f}, Neg Focal={loss_s0_val['neg_focal_loss']:.4f} (Neg/Pos ratio: {loss_s0_val['neg_to_pos_loss_ratio']:.2f})")
    print(f"Stage 1 Best Val Loss: Total Cls={loss_s1_best_val['total_cls_loss']:.4f}, Pos Focal={loss_s1_best_val['pos_focal_loss']:.4f}, Neg Focal={loss_s1_best_val['neg_focal_loss']:.4f} (Neg/Pos ratio: {loss_s1_best_val['neg_to_pos_loss_ratio']:.2f})")
    print(f"Stage 1 Last Val Loss: Total Cls={loss_s1_last_val['total_cls_loss']:.4f}, Pos Focal={loss_s1_last_val['pos_focal_loss']:.4f}, Neg Focal={loss_s1_last_val['neg_focal_loss']:.4f} (Neg/Pos ratio: {loss_s1_last_val['neg_to_pos_loss_ratio']:.2f})")
    print("\nStage 1 Best vs Last Logit & Prob Means on Val:")
    for lvl in ["P2", "P3", "P4", "P5"]:
        print(f"  {lvl} Best: logit mean={pred_stats_s1_best[lvl]['logit_mean']:.3f}, prob mean={pred_stats_s1_best[lvl]['prob_mean']:.4f}, % > 0.05={pred_stats_s1_best[lvl]['pct_prob_gt_0_05']:.2f}%")
        print(f"  {lvl} Last: logit mean={pred_stats_s1_last[lvl]['logit_mean']:.3f}, prob mean={pred_stats_s1_last[lvl]['prob_mean']:.4f}, % > 0.05={pred_stats_s1_last[lvl]['pct_prob_gt_0_05']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
