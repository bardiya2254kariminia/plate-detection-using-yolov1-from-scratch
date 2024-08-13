import torch
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import pandas as pd
import os

class plate_notplate_dataset(Dataset):
    def __init__(
            self ,csv_file , plate_image_dir ,plate_label_dir , notplate_label_dir, notplate_image_dir ,S = 7 , C = 1 , B =2 , transform = None
        ):
        self.annotation = pd.read_csv(csv_file)
        self.plate_image_dir = plate_image_dir
        self.notplate_image_dir = notplate_image_dir
        self.plate_label_dir = plate_label_dir
        self.notplate_label_dir = notplate_label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        print(len(self.annotation))
        return len((self.annotation))
    
    def __getitem__(self, index):
        file_name = self.annotation.iloc[index ,  0]
        boxes = []
        image = None
        if "Car" in file_name:
            label_path = os.path.join(self.plate_label_dir , self.annotation.iloc[index ,  1])
            with open(label_path) as f:
                for label in f.readlines():
                    class_label , x , y , width ,height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n" , "").split()
                ]
                    boxes.append([class_label , x , y , width , height])
                image_path = os.path.join(self.plate_image_dir , self.annotation.iloc[index , 0])
                image = Image.open(image_path).convert("RGB")
                boxes = torch.tensor(boxes) 
        else:
            label_path =   os.path.join(self.notplate_label_dir , self.annotation.iloc[index ,1])
            boxes.append([0 , 0 , 0 , 0 ,0])
            image_path = os.path.join(self.notplate_image_dir , self.annotation.iloc[index, 0])
            image = Image.open(image_path).convert("RGB")
            boxes = torch.tensor(boxes)
        if self.transform:
            image , boxes = self.transform(image, boxes)
        label_matrix = torch.zeros((self.S , self.S , self.C + self.B * 5))
        for box in boxes :
            class_label , x , y , width , height = box.tolist()
            class_label = int(class_label)
            i , j = int(self.S * y)  , int(self.S * x)
            x_cell , y_cell = self.S *  x - j , self.S * y -i
            
            width_cell , height_cell = (
                width * self.S,
                height  * self.S
            )

            if label_matrix[i , j, self.C] == 0:
                if "Car" in file_name:
                    # set the existency of plates
                    label_matrix[i , j , self.C] = 1
                    # set the one hot encode for the class label
                    label_matrix[i,j,self.C-1] =1
                else:
                    # no object as background
                    label_matrix[i , j , self.C] = 0                    
                    # set the one hot encode for the class label
                    label_matrix[i,j,self.C-1] = 0

                # box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i,j,self.C+1:self.C+5] = box_coordinates

        return image , label_matrix
