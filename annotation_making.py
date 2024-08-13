import os
import pandas as pd

plates_info_path = "C:\\Users\\Asus\\Desktop\\Projects\\licence plate detection\\plate_labels"
notplates_info_path = "C:\\Users\\Asus\\Desktop\\Projects\\licence plate detection\\utills_csv\\test.csv"
plates_dir= os.listdir(plates_info_path)

df = pd.read_csv(notplates_info_path)
for filename in plates_dir:
    df.loc[len(df.index)] = [filename.split(".")[0] + ".png" , filename.split(".")[0] + ".txt"]
df.to_csv("annotation.csv"  , index=False)