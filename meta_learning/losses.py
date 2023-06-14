from torch import nn
from torch.nn import functional as F
import torch


def approximate_iou(outputs, masks, smooth=1):
    outputs = nn.Sigmoid()(outputs)
    outputs = outputs.squeeze(1)
    masks = masks.squeeze(1)
    intersection = (outputs * masks).sum()
    total_overlap = (outputs + masks).sum()
    union = total_overlap - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def approximate_dice(outputs, masks, smooth=1):
    inputs = nn.Sigmoid()(outputs)
    inputs = inputs.view(-1)
    targets = masks.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice


class IoULoss(nn.Module):
    """IoU Loss implementation that follows http: // cs.umanitoba.ca / ~ywang / papers / isvc16.pdf"""

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, outputs, masks, smooth=1):
        iou = approximate_iou(outputs, masks, smooth)
        return 1 - iou.mean()


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss approximation. https://arxiv.org/pdf/1706.05721.pdf
    Larger beta places more emphasis on false negatives -> more importance to recall.
    """

    def __init__(self):
        super(TverskyLoss, self).__init__()

    def forward(self, outputs, masks, alpha=0.2, beta=0.8, smooth=1):
        outputs = nn.Sigmoid()(outputs)
        outputs = outputs.squeeze(1)
        masks = masks.squeeze(1)
        true_pos = (outputs * masks).sum()
        false_pos = ((1 - masks) * outputs).sum()
        fals_neg = (masks * (1 - outputs)).sum()
        tversky = (true_pos + smooth) / (true_pos + alpha *
                                         false_pos + beta * fals_neg + smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined loss that combines IoU and BCE loss"""

    def __init__(self):
        super(CombinedLoss, self).__init__()

    def forward(self, outputs, masks, smooth=1, alpha=0.5):
        iou = approximate_iou(outputs, masks, smooth)
        iou_loos = 1 - iou
        bce_loss = nn.BCELoss()(outputs, masks)
        combined_loss = alpha * iou_loos + (1 - alpha) * bce_loss
        return combined_loss

class CombinedLoss3(nn.Module):
    """Combined loss that combines Dice and BCE loss"""

    def __init__(self):
        super(CombinedLoss3, self).__init__()

    def forward(self, outputs, masks, smooth=1, alpha=0.5):
        dice_loss = approximate_dice(outputs, masks, smooth)
        bce_loss = nn.BCELoss()(outputs, masks)
        combined_loss = alpha * dice_loss + (1 - alpha) * bce_loss
        return combined_loss


class CombinedLoss2(nn.Module):
    def __init__(self, pos_weight):
        super(CombinedLoss2, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, outputs, masks, smooth=1):
        iou = approximate_iou(outputs, masks, smooth)
        modified_dice = (2 * iou) / (iou + 1)
        bce = F.binary_cross_entropy_with_logits(outputs, masks,
                                                 pos_weight=self.pos_weight)
        combined = bce - torch.log(modified_dice)
        return combined

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, eps=1e-9):

        #flatten label and prediction tensors
        dice = approximate_dice(inputs, targets)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        print(out.isnan())
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)

        return combo
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
LOSSES = {"bce": nn.BCELoss, "bce_weighted": nn.BCEWithLogitsLoss, "iou": IoULoss, "tversky": TverskyLoss,
          "combined": CombinedLoss, "combined2": CombinedLoss2, "dice": DiceLoss, "combined_dice_bce": DiceBCELoss, "combo_loss": ComboLoss}
