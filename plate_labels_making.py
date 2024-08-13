import torch
import os
import xml.etree.ElementTree as ET
import pandas as pd

plates_info_path = "C:\\Users\\Asus\\Desktop\\Projects\\licence plate detection\\plate_labels_xml"

plates_annotations_dir= os.listdir(plates_info_path)
plate_image_label = []
for p in plates_annotations_dir:
    xml_path = os.path.join(plates_info_path , p)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = {
        "label" :[],
        "x" :[],
        "y" :[],
        "w" :[],
        "h" :[]
    }
    labels = pd.DataFrame(labels)
    for child in root:
        if child.tag == "object":
            image_file_name = root[1].text
            width,height = float(root[2][0].text) , float(root[2][1].text)
            xmin_norm , ymin_norm , xmax_norm , ymax_norm = float(child[5][0].text) /width  , float(child[5][1].text) / height , float(child[5][2].text)/width , float(child[5][3].text) / height
            x = (xmin_norm + xmax_norm)/2
            y = (ymin_norm + ymax_norm)/2
            w = xmax_norm - xmin_norm
            h = ymax_norm - ymin_norm
            box  = ["1", x , y , w , h]
            print(box)
            labels.loc[len(labels.index)] = box
    plate_path  = path_or_buf=p.split(".")[0] + ".txt"
    plate_path = os.path.join("C:\\Users\\Asus\\Desktop\\Projects\\licence plate detection\\plate_labels" , plate_path)
    labels.to_csv(plate_path , index=False , header=False , sep=" ")
