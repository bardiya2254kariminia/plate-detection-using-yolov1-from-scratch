import torch
import torch.nn
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm  
from torch.utils.data import DataLoader
from plate_notplate_dataset import plate_notplate_dataset
from utils import *
from loss import YoloLoss
from model import Yolov1
import os, sys
import matplotlib.pyplot as plt
from sampler import get_sampler

seed = 123
torch.manual_seed(seed= seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
WEIGHT_DECAY = 0
EPOCHES = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_ADDRESS = "/content/weights/weights.pth"
PLATE_LABELS_PATH = "/content/plates_labels"
PLATE_IMAGES_PATH = "/content/plates_image"
NOTPLATE_LABELS_PATH = "/content/notplates_labels"
NOTPLATE_IMAGES_PATH = "/content/notplates_image"
ANNOTATION_PATH  = "/content/annotation.csv"


class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self , img , bboxes):
        for t in self.transforms:
            img , bboxes = t(img), bboxes
        return   img , bboxes
    
# we have to reshape the pictures to 448 * 448 * 3 as the paper say's

transform = Compose([transforms.Resize((448,448)) , transforms.ToTensor()])


def train_fn(train_loader , model , optimizer , loss_fn):
    loop = tqdm(train_loader , leave=True)
    mean_loss= []
    for batch_idx , (x ,  y) in enumerate(loop):
        x , y = x.to(DEVICE) , y.to(DEVICE)
        pred = model(x)
        loss = loss_fn(pred , y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss is {sum(mean_loss) / len(mean_loss)}")


def main():
    model = Yolov1(
        num_boxes=2 , num_classes=1 , split_size=7
    ).to(DEVICE)
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(
      S = 7 ,
      B = 2,
      C = 1
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_ADDRESS) , model , optimizer)
    train_dataset = plate_notplate_dataset(
        csv_file=ANNOTATION_PATH,
        plate_image_dir=PLATE_IMAGES_PATH,
        plate_label_dir=PLATE_LABELS_PATH,
        notplate_image_dir=NOTPLATE_IMAGES_PATH,
        notplate_label_dir=NOTPLATE_LABELS_PATH,
        S = 7,
        C = 1,
        B = 2,
        transform=transform
    )
    sampler = get_sampler(dataset=train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset ,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        # num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    for epoch  in range(EPOCHES):
        if epoch % 4 == 3:
                
            pred_boxes, target_boxes = get_bboxes(
                loader=train_loader , model=model , iou_threshold=0.5 , threshold=0.4
            )

            mean_average_prec = mean_average_precision(
                pred_boxes=pred_boxes , true_boxes=target_boxes , iou_threshold=0.5 , box_format="midpoint"
            )
            print(f"Train mAP is :{mean_average_prec}")
            if mean_average_prec >0.82:
                torch.save(model.state_dict() , "plate_detection.pth")
                sys.exit()
        print(f"epoch num :{epoch}")
        if epoch == 20:
            torch.save(model.state_dict() , "plate_detection.pth")
            sys.exit()

        train_fn(train_loader=train_loader , model=model , optimizer=optimizer,loss_fn=loss_fn)
main()

