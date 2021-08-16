import shutil
import os
fp=open("train.csv")
ofp=open("small_train.csv","w")
h=next(fp)
for line in fp:
    arr=line.strip().split(",")
    #print(arr)
    print(arr[9])
    basename=os.path.basename(arr[9])
    barcode=arr[-1]
    target="./small_data/"+basename
    shutil.copyfile(arr[9],target)
    arr[9]=target
    #ofp.write(barcode+","+target+"\n")
    ofp.write(",".join(arr)+"\n")
