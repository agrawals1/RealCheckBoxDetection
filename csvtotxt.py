import pandas as pd
import os
import numpy as np
train_old = pd.DataFrame()
#for file in os.listdir(r"./data/splitcsv/train/class-0"):
for file in os.listdir(r"./data/afterAugBB/xminymin"):
    #train = pd.read_csv(os.path.join(r"./splitcsv/train/class-0", file), index_col=0)
    train = pd.read_csv(os.path.join(r"./data/afterAugBB/xminymin", file))
    #mystring = (r".\output\train\class-0\{}.png".format(file[:-4]))
    mystring = (r".\data\augImgs\class-0\{}.png".format(file[:-4]))    
    train["filepath"] = str(mystring)
    print(train.columns)
    train = train[["filepath", "xmin", "ymin", "xmax", "ymax", "class"]]

    train[["xmin", "ymin", "xmax", "ymax"]] = train[["xmin", "ymin", "xmax", "ymax"]].astype(int)
    
    train_old = train_old.append(train)
    print(train_old)



    
train_old.to_csv(r"./data/annotate.txt", header=None, index=0, sep=' ')

