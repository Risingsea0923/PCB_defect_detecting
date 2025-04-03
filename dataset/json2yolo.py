import os
import json

def findAllFile(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith('.json'):
                yield f[:-5]

# 使用字典对于标签进行存储
label_dict=dict()
label_num = 0

base = "./original_data"

for name in findAllFile(base+"/json"):
    # 获取json数据
    with open(f"{base}/json/{name}.json", 'r') as jf:
        data = json.load(jf)
    height = data["imageHeight"]
    width = data["imageWidth"]

    s="" # 用来储藏txt中的内容
    for item in data["shapes"]: # 遍历数据集中每一个分割子类
        # 对于标签进行收集和处理
        if (label:=item["label"]) not in label_dict.keys():
            label_dict[label]=label_num
            label_num+=1

        s = s+str(label_dict[item["label"]])+" "

        points = item["points"]
        for point in points:
            s=s+str(point[0]/width)+" "
            s=s+str(point[1]/height)+" "
        s = s[:-1]+"\n"

    # 将数据集进行写入labels文件夹下
    with open(f"{base}/labels/{name}.txt", 'w') as tf:
        tf.write(s)

with open("./dataset.yaml","w") as f:
    f.write(f"path: {base}\n")
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("test: \n\n")
    f.write("names:\n")
    for key,num in label_dict.items():
        f.write(f"  {num}:{key}\n")

import os
import json

def findAllFile(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            if f.endswith('.json'):
                yield f[:-5]

# 使用字典对于标签进行存储
label_dict=dict()
label_num = 0

base = "./original_data"

for name in findAllFile(base+"/json"):
    # 获取json数据
    with open(f"{base}/json/{name}.json", 'r') as jf:
        data = json.load(jf)
    height = data["imageHeight"]
    width = data["imageWidth"]

    s="" # 用来储藏txt中的内容
    for item in data["shapes"]: # 遍历数据集中每一个分割子类
        # 对于标签进行收集和处理
        if (label:=item["label"]) not in label_dict.keys():
            label_dict[label]=label_num
            label_num+=1

        s = s+str(label_dict[item["label"]])+" "

        points = item["points"]
        for point in points:
            s=s+str(point[0]/width)+" "
            s=s+str(point[1]/height)+" "
        s = s[:-1]+"\n"

    # 将数据集进行写入labels文件夹下
    with open(f"{base}/labels/{name}.txt", 'w') as tf:
        tf.write(s)

with open("./dataset.yaml","w") as f:
    f.write(f"path: {base}\n")
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("test: \n\n")
    f.write("names:\n")
    for key,num in label_dict.items():
        f.write(f"  {num}:{key}\n")

