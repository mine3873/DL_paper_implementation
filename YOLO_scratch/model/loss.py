import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    loss = lambda_coord * sum_{i=0}^{S^2} sum_{j=0}^{B} 1_{ij}^{obj} * [(x_i - x_hat_i)^2 + (y_i - y_hat_i)^2]
        + lambda_coord * sum_{i=0}^{S^2}sum_{j=0}^{B} 1_{ij}^{obj} * [(sqrt(w_i) - sqrt(w_hat_i))^2 + (sqrt(h_i) - sqrt(h_hat_i))^2]
        + sum_{i=0}^{S^2}sum_{j=0}^{B} 1_{ij}^{obj} * (C_i - C_hat_i)^2
        + lambda_noobj * sum_{i=0}^{S^2}sum_{j=0}^{B} 1_{ij}^{noobj} * (C_i - C_hat_i)^2
        + sum_{i=0}^{S^2} 1_{i}^{obj} * sum_{c in classes} (p_i(c) - p_hat_i(c))^2
        
    - C : Confidence score
        if there is thing, C = 1 * IoU, otherwise C = 0
    
    - p_i(c) : class probabilities
        when there is thing, the probabilities C=c
    
    - x,y : center coordinate of each boundary box
    
    - w,h : width, height of each boundary box 
    """
    
    def __init__(self, config):
        super(YOLOLoss, self).__init__()
        self.config = config
        
        self.S = config.S
        self.B = config.B
        self.C = config.C
        
        self.eps = config.eps
        
        self.lambda_coord = config.lambda_coord
        self.lambda_noobj = config.lambda_noobj
        
        # sum(error^2)
        self.mse = nn.MSELoss(reduction="sum")
        
    def forward(self, prediction, targets):
        """
        predictions, tagets: (batch_size, S, S, B * 5 + C)
        
        boxxes: tensor(batch_size, S, S, B, (x, y, w, h, c))
        """
        boxxes = prediction[..., :self.B * 5].reshape(-1, self.S, self.S, self.B, 5)
        
        classes = prediction[..., self.B * 5:]
        target_classes = targets[..., self.B * 5:]
        
        
        # select bestbox (have largest IoU)
        ious = []
        for b in range(self.B):
            ious += [self.returnIoU(
                boxxes[..., b, :4], targets[..., :4]
            ).unsqueeze(0)]
        ious = torch.cat(ious, dim=0) # (B, batch, S, S)
        iou_maxval, bestbox = torch.max(ious, dim=0) # (batch, S, S), (batch, S, S) 
        best_idx = bestbox.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 5)
        pred_best = torch.gather(boxxes, dim=3, index=best_idx) # (:, :, :, best_idx, :)
        pred_best = pred_best.squeeze(3) # (batch, S, S, 5)
        
        exists_box_i = targets[..., 4] == 1 # (batch, S, S)
        
        p_best = pred_best[exists_box_i] # (N, 5)
        t_best = targets[exists_box_i] # (N, 5)
        
        if p_best.numel() == 0:
            return torch.tensor(0.0, requires_grad=True).to(prediction.device)
        
        
        
        exists_box_i_mask = exists_box_i.unsqueeze(-1) # (batch, S, S, 1)
        
        # Coordinate Loss
        #pred_xy = pred_best[..., :2]
        #target_xy = targets[..., :2]
        
        pred_xy = p_best[..., :2]
        targets_xy = t_best[..., :2]
        
        #pred_wh = torch.sign(pred_best[..., 2:4]) * torch.sqrt(torch.abs(pred_best[..., 2:4]).clamp(min = self.eps))
        #targets_wh = torch.sqrt(targets[..., 2:4])
        
        pred_wh = torch.sign(p_best[..., 2:4]) * torch.sqrt(torch.abs(p_best[..., 2:4]).clamp(min = self.eps))
        targets_wh = torch.sqrt(t_best[..., 2:4])
        
        #coord_loss = self.mse(
        #    exists_box_i_mask * pred_xy, exists_box_i_mask * target_xy
        #) + self.mse(
        #    exists_box_i_mask * pred_wh, exists_box_i_mask * targets_wh
        #)
        
        coord_loss = self.mse(pred_xy, targets_xy) + self.mse(pred_wh, targets_wh)
        coord_loss *= self.lambda_coord
        
        # Object Loss
        
        # pred_obj_confidence = pred_best[..., 4:5]
        #target_obj_confidence = iou_maxval.unsqueeze(-1)
        pred_obj_confidence = p_best[..., 4:5]
        target_obj_confidence = iou_maxval[exists_box_i].unsqueeze(-1)
        
        #obj_loss = self.mse(
        #    exists_box_i_mask * pred_obj_confidence,
        #    exists_box_i_mask * target_obj_confidence
        #)
        obj_loss = self.mse(pred_obj_confidence, target_obj_confidence)
        
        # No object Loss
        all_pred_confidence = boxxes[..., 4:5]
        
        # only at bestbox, set 1, the others 0
        one_hot_best = torch.zeros_like(all_pred_confidence).scatter_(3, bestbox.unsqueeze(-1).unsqueeze(-1).long(), 1)
        
        # multiply origin mask with one_hot_best, then only 1 at bestbox, not at all boxes have confidence = 1
        noobj_mask = 1 - (exists_box_i_mask.unsqueeze(3) * one_hot_best)
        
        noobj_loss = self.lambda_noobj * self.mse(
            noobj_mask * all_pred_confidence,
            torch.zeros_like(all_pred_confidence) 
            # because if the box detect nonthing, then targets' confidence must be 0.
        )
        
        # Class Loss
        pred_classes = classes[exists_box_i]
        target_classes = target_classes[exists_box_i]
        
        #class_loss = self.mse(
        #    exists_box_i_mask * classes,
        #    exists_box_i_mask * target_classes
        #)
        class_loss = self.mse(pred_classes, target_classes)
        
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        return total_loss
        
    def returnIoU(self, box1, box2):
        """
        box: tensor(batch_size, S, S)
        
        return: 
        tensor(batch_size, S, S)
        """
        
        x1, y1, w1, h1 = [box1[..., i] for i in range(4)]
        x2, y2, w2, h2 = [box2[..., i] for i in range(4)]
        
        inter_x1 = torch.max(x1 - w1/2, x2 - w2/2)
        inter_x2 = torch.min(x1 + w1/2, x2 + w2/2)
        inter_y1 = torch.max(y1 - h1/2, y2 - h2/2)
        inter_y2 = torch.min(y1 + h1/2, y2 + h2/2)   
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        area1 = (w1.clamp(min = self.eps) * h1.clamp(min = self.eps))
        area2 = (w2 * h2)
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + self.eps)
            
        
        
        
        