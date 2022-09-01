import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def _reconstruction_loss(target, pred, mask, norm_pix_loss=True):
    # MSE loss
    # return torch.Tensor([0])
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def _constrastive_loss(student_output, teacher_output, temperature=0.1):
    # InfoNCE loss
    preds = student_output
    targets = teacher_output
    loss = -nn.CosineSimilarity()(preds, targets).sum().mean() / temperature
    for p_idx, pred in enumerate(preds):
        des = 0.0
        for t_idx, targ in enumerate(targets):
            s = nn.CosineSimilarity(dim=0)(pred, targ)
            des = des + torch.exp(s / temperature)
        loss = loss + torch.log(des)
    
    return loss / len(targets)


class CMAELoss(nn.Module):
    def __init__(self, args):
        super(CMAELoss, self).__init__()
        self.norm_pix_loss = args.norm_pix_loss
        self.temperature = args.temperature

    def forward(self, student_inputs, teacher_inputs,
                student, teacher, args):

        student_features, mask = student.foward_features(
            student_inputs, mask_ratio=0.75)
        teacher_features, _ = teacher.foward_features(
            teacher_inputs, mask_ratio=0.0)

        # Reconstruct loss 
        reconstruct_pred = student.forward_pixel_decoder(student_features)
        reconstruct_target = teacher.patchify(student_inputs)
        reconstruct_loss = _reconstruction_loss(
            reconstruct_target, reconstruct_pred, mask, self.norm_pix_loss)

        # Contrastive loss
        student_features = student.forward_feature_decoder(student_features)
        teacher_features = teacher.forward_feature_decoder(teacher_features)
        student_prob = student.project_predictor(student_features)
        teacher_prob = teacher.project_predictor(teacher_features)
        constractive_loss = _constrastive_loss(
            student_prob, teacher_prob, self.temperature)
        constractive_loss = constractive_loss.to(reconstruct_loss.device)
        
        total_loss = reconstruct_loss + constractive_loss
        return reconstruct_loss, constractive_loss, total_loss
