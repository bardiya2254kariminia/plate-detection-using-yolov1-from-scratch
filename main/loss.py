import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    """
    calculating the loss as the paper say's
    """

    def __init__(self, S =7 , B = 2 , C = 20):
        super(YoloLoss , self).__init__()
        # in the original paper we use just the summation of predictions not the mean of them
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        # in the papers we have the lambda's  for every parts of it
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self,  predictions:torch.tensor , targets:torch.tensor):
        # predictions are reshaped
        predictions  = predictions.reshape(-1 , self.S , self.S , self.C +self.B * 5)

        # calculating the best iou from the each anchor boxes
        iou_b1 = intersection_over_union(boxes_perds=predictions[... , self.C+1:self.C+5] , boxes_labels=targets[... , self.C+1:self.C+5])
        iou_b2 = intersection_over_union(boxes_perds=predictions[... , self.C+6:self.C+10] , boxes_labels=targets[... , self.C+1:self.C+5])
        # we unsqueeze the ious form the 0's dimensions and then concat them through the 0's dim
        # then we found the  max of them and return them   
        ious = torch.cat([iou_b1.unsqueeze(0) , iou_b2.unsqueeze(0)], dim=0)
        # get the max of them and the indexof it
        iou_maxes , bestbox = torch.max(ious , dim = 0)
        exists_box = targets[... , self.C].unsqueeze(3)


        box_predictions = exists_box * (
            (
                bestbox * predictions[... , self.C+6:self.C+10] +
                (1 - bestbox) * predictions[... , self.C+1:self.C+5]
            )
        )


        box_targets = exists_box * targets[... , self.C+1:self.C+5]

        box_predictions[...,2:4] = torch.sign(box_predictions[... , 2:4]) * torch.sqrt(
            torch.abs(box_predictions[... , 2:4] + 1e-6)
        )
        box_targets[... , 2:4]= torch.sqrt(box_targets[... , 2:4])
        # (N , S , S, 4) -> (N*S*S , 4) for box_predictions
        box_loss = self.mse(
            torch.flatten(box_predictions , end_dim=-2),
            torch.flatten(box_targets , end_dim=-2)
        )

        pred_box = (
            bestbox * predictions[... , self.C+5:self.C+6] + (1 - bestbox) * predictions[... , self.C:self.C+1]
        )
        # (N * S * S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box) ,
            torch.flatten(exists_box * targets[... , self.C:self.C+1])
        )

        # (N , S , S ,1) -> (N , S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box)* predictions[... , self.C:self.C+1] , start_dim=1),
            torch.flatten((1-exists_box)* targets[... , self.C:self.C+1] , start_dim=1)
        )
        no_object_loss +=  self.mse(
            torch.flatten((1-exists_box) * predictions[...,self.C+5:self.C+6] , start_dim=1),
            torch.flatten((1-exists_box)* targets[... , self.C:self.C+1] , start_dim=1)
        )

        # (N , S , S , 20) -> (N*S*S , 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[... , :self.C] , end_dim=-2),
            torch.flatten(exists_box * targets[... , :self.C] , end_dim=-2)
        )

        loss = (self.lambda_coord  * box_loss) + object_loss + (self.lambda_noobj * no_object_loss) + class_loss

        return loss
