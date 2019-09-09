import pandas as pd
import os
import numpy as np
train_old = pd.DataFrame()
for file in os.listdir(r"./outputcsv/train/class-0"):

    train = pd.read_csv(os.path.join(r"./outputcsv/train/class-0", file), index_col=0)
    mystring = (r".\output\train\class-0\{}.png".format(file[:-4]))
    train["filepath"] = str(mystring)
    train = train[["filepath", "xmin", "ymin", "xmax", "ymax", "class" ]]

    train[["xmin", "ymin", "xmax", "ymax"]] = train[["xmin", "ymin", "xmax", "ymax"]].astype(int)
    train_old = train_old.append(train)



    
train_old.to_csv(r"./outputtxt/train/class-0/{}.txt".format(file[:-4]), header=None, index=0, sep=' ')
input("stop")
