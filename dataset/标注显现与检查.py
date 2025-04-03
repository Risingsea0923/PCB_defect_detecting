import os
import cv2
import matplotlib.pyplot as plt

# 图像与标注所在路径，根据实际情况修改
IMAGE_DIR = "D:/030923/dataset/images"
LABEL_DIR = "D:/030923/dataset/labels"


# 获取所有标注文件列表（.txt）
label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]

for label_file in label_files:
    # 1. 拼出对应图像的路径
    txt_path = os.path.join(LABEL_DIR, label_file)
    # 假设图像文件名与标注文件名同名，只是后缀不同
    image_name = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(IMAGE_DIR, image_name)

    # 如果图片不存在则跳过
    if not os.path.exists(img_path):
        print(f"Warning: image {img_path} not found for {label_file}")
        continue
    
    # 2. 读取图像
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]

    # 3. 读取标注文件，每一行解析YOLO格式
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # YOLO格式: class_id x_center_norm y_center_norm width_norm height_norm
        class_id_str, x_center_norm_str, y_center_norm_str, w_norm_str, h_norm_str = line.split()
        
        # 转成浮点数
        class_id = int(class_id_str)
        x_center_norm = float(x_center_norm_str)
        y_center_norm = float(y_center_norm_str)
        w_norm = float(w_norm_str)
        h_norm = float(h_norm_str)
        
        # 4. 把归一化坐标变换回图像像素坐标
        box_w = w_norm * img_width
        box_h = h_norm * img_height
        box_x_center = x_center_norm * img_width
        box_y_center = y_center_norm * img_height
        
        # 左上角 (x1, y1) ，右下角 (x2, y2)
        x1 = int(box_x_center - box_w / 2)
        y1 = int(box_y_center - box_h / 2)
        x2 = int(box_x_center + box_w / 2)
        y2 = int(box_y_center + box_h / 2)

        # 5. 在图像上绘制矩形框
        # BGR颜色可以任意设置，这里只给一个大概示例；线条宽度=2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        # 如果需要在框上方显示类别ID，可用以下方式
        text = f"{class_id}"
        cv2.putText(
            img, text, 
            (x1, y1 - 5),    # 在框上方一点
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,             # 字体大小
            (0,255,0),       # 字体颜色
            2                # 字体线宽
        )
    
    # 6. 用matplotlib显示该图像（也可以cv2.imshow直接显示，取决于你的环境）
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Image with YOLO boxes: {image_name}")
    plt.axis("off")
    plt.show()
