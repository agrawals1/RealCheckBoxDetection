import shutil
import os
valimagelist = os.listdir(r"./data/splitImages/val/class-0")
trainimagelist = os.listdir(r"./data/splitImages/train/class-0")
testimagelist = os.listdir(r"./data/splitImages/test/class-0")

for file in os.listdir(r"./data/afterAugBB/xminymin"):
    for temp in valimagelist:
        if file[:-4] == temp[:-4]:
           shutil.move(r"./data/afterAugBB/xminymin/{}".format(file), r"./data/splitcsv/val/class-0/{}".format(file)) 
           
for file in os.listdir(r"./data/afterAugBB/xminymin"):
    for temp in trainimagelist:
        if file[:-4] == temp[:-4]:
           shutil.move(r"./data/afterAugBB/xminymin/{}".format(file), r"./data/splitcsv/train/class-0/{}".format(file)) 

for file in os.listdir(r"./data/afterAugBB/xminymin"):
    for temp in testimagelist:
        if file[:-4] == temp[:-4]:
           shutil.move(r"./data/afterAugBB/xminymin/{}".format(file), r"./data/splitcsv/test/class-0/{}".format(file)) 
           
    

