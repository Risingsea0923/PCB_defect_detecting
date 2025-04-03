import os
import json
import cv2
import random
import math
import numpy as np
import shutil
from skimage.util import random_noise
from skimage import exposure
from PIL import Image

# -------------------------------
# 固定的缺陷类型映射（原始映射，供字符串标签使用）
# -------------------------------
defect_mapping = {
    "missing_hole": 0,       # 缺焊
    "mouse_bite": 1,
    "open_circuit": 2,
    "short": 3,
    "spurious_copper": 4,
    "spur": 5,
    "misalignment": 6,       # 歪斜
    "insufficient_tin": 7,   # 少锡
    "lifted_edge": 8         # 翘边
}
# -------------------------------
# 数字到字符串的反向映射（若 JSON 中直接给出数字）
# -------------------------------
defect_mapping_int_to_str = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spurious_copper",
    5: "spur",
    6: "misalignment",
    7: "insufficient_tin",
    8: "lifted_edge"
}

# -------------------------------
# 数据增强类（基于 image_enhence.py）
# -------------------------------
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=30,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5,
                 cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate
        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold

    def _addNoise(self, img):
        return random_noise(img, mode='gaussian', clip=True) * 255

    def _changeLight(self, img):
        flag = random.uniform(0.5, 1.5)  # >1调暗，<1调亮
        return exposure.adjust_gamma(img, flag)

    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.0):
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        rot_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            # 采用四个角点进行旋转
            corners = np.array([
                [xmin, ymin, 1],
                [xmax, ymin, 1],
                [xmax, ymax, 1],
                [xmin, ymax, 1]
            ])
            rotated_corners = np.dot(rot_mat, corners.T).T  # 4x2
            r_xmin = np.min(rotated_corners[:, 0])
            r_ymin = np.min(rotated_corners[:, 1])
            r_xmax = np.max(rotated_corners[:, 0])
            r_ymax = np.max(rotated_corners[:, 1])
            rot_bboxes.append([r_xmin, r_ymin, r_xmax, r_ymax])
        return rot_img, rot_bboxes

    def _crop_img_bboxes(self, img, bboxes):
        w = img.shape[1]
        h = img.shape[0]
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)
        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        crop_bboxes = []
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min])
        return crop_img, crop_bboxes

    def _shift_pic_bboxes(self, img, bboxes):
        w = img.shape[1]
        h = img.shape[0]
        x_min = min(b[0] for b in bboxes)
        y_min = min(b[1] for b in bboxes)
        x_max = max(b[2] for b in bboxes)
        y_max = max(b[3] for b in bboxes)
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max
        x_shift = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y_shift = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shift_img = cv2.warpAffine(img, M, (w, h))
        shift_bboxes = []
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x_shift, bbox[1]+y_shift, bbox[2]+x_shift, bbox[3]+y_shift])
        return shift_img, shift_bboxes

    def _filp_pic_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        flip_img = cv2.flip(img, 1)
        flip_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        change_num = 0
        # 保证至少进行一种数据增强
        while change_num < 1:
            if random.random() < self.crop_rate:
                img, bboxes = self._crop_img_bboxes(img, bboxes)
                change_num += 1
            if random.random() > self.rotation_rate:
                angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
                change_num += 1
            if random.random() < self.shift_rate:
                img, bboxes = self._shift_pic_bboxes(img, bboxes)
                change_num += 1
            if random.random() > self.change_light_rate:
                img = self._changeLight(img)
                change_num += 1
            if random.random() < self.add_noise_rate:
                img = self._addNoise(img)
                change_num += 1
        return img, bboxes

# -------------------------------
# 从 Labelme JSON 文件中解析标注
# -------------------------------
def parse_labelme_json(json_path):
    """
    读取 JSON 文件，提取所有 bounding box 和标签
    bounding box 格式为 [xmin, ymin, xmax, ymax]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    width = data.get("imageWidth", None)
    height = data.get("imageHeight", None)
    bboxes = []
    labels = []
    for shape in data["shapes"]:
        points = shape["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        bboxes.append([x_min, y_min, x_max, y_max])
        labels.append(shape["label"])
    return bboxes, labels, width, height, data.get("imagePath", None)

# -------------------------------
# YOLO 格式转换函数（类别号为原始数字+1）
# -------------------------------
def convert_to_yolo(bboxes, labels, img_width, img_height):
    """
    将 bounding box 转换为 YOLO 格式（归一化）：
      每行：class_index x_center y_center box_width box_height
    若标签为数字或数字字符串，则直接转换，输出类别号为原始数字（从0开始）；
    若为字符串，则通过 defect_mapping 查找对应数字（从0开始）。
    """
    lines = []
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height
        try:
            label_num = int(label)
        except:
            if label not in defect_mapping:
                print(f"警告: {label} 不在 defect_mapping 中")
                continue
            label_num = defect_mapping[label]
        # 直接使用原始索引，不再+1
        cls_id = label_num
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return "\n".join(lines)

# -------------------------------
# 主流程：遍历 images 文件夹中的所有图片并更新缺陷后边的编号
# -------------------------------
def main():
    # 路径设置（根据实际情况修改）
    base = "D:/030923/data/train"
    #base = "D:/030923/dataset"
    json_dir = os.path.join(base, "json")
    image_dir = os.path.join(base, "images_0")
    output_image_dir = os.path.join(base, "images")
    output_label_dir = os.path.join(base, "labels")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    need_aug_num = 4  # 每个文件生成增强图数量
    # 固定板子编号为 "013"（后续可扩展板子编号逻辑）
    board_id = "013"
    # 用 file_counter 遍历 images 中的图片更新缺陷编号（文件编号）
    file_counter = 1
    
    # 遍历 images 文件夹中所有 jpg/png 图片
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # 可选：排序
    
    # 用于记录所有生成的文件名（不含扩展名）
    all_generated_files = []

    for image_file in image_files:
        file_index = str(file_counter).zfill(2)
        file_counter += 1

        base_name = os.path.splitext(image_file)[0]
        json_file = base_name + ".json"
        json_path = os.path.join(json_dir, json_file)
        if not os.path.exists(json_path):
            print(f"未找到 {json_path} 对应的 JSON 文件，跳过 {image_file}")
            continue

        # 解析 JSON 得到 bounding box 和标签
        bboxes, labels, json_width, json_height, image_path_in_json = parse_labelme_json(json_path)
        if not labels:
            print(f"文件 {json_file} 没有标注信息，保存为无缺陷图像，不进行增强。")
            # 保存原始图片作为无缺陷样本
            image_path_full = os.path.join(image_dir, image_file)
            img = cv2.imread(image_path_full)
            if img is None:
                print(f"读取图片 {image_path_full} 失败，跳过。")
                continue
                
            # 使用特殊标识"no_defect"保存无缺陷图像
            original_img_name = f"{board_id}_no_defect_{file_index}_0.jpg"
            original_name_base = f"{board_id}_no_defect_{file_index}_0"
            cv2.imwrite(os.path.join(output_image_dir, original_img_name), img)
            # 创建一个空的标注文件
            original_label_name = f"{board_id}_no_defect_{file_index}_0.txt"
            with open(os.path.join(output_label_dir, original_label_name), 'w') as f:
                f.write("")  # 写入空内容，表示没有缺陷
            print(f"保存无缺陷图片：{original_img_name}")
            all_generated_files.append(original_name_base)
            continue
        
        # 处理第一个标注的缺陷类型用于命名
        defect = labels[0]
        try:
            defect_num = int(defect)
            if defect_num not in defect_mapping_int_to_str:
                print(f"缺陷类型 {defect} 不在预定义列表中，跳过文件 {json_file}")
                continue
            defect_str = defect_mapping_int_to_str[defect_num]
        except:
            if defect not in defect_mapping:
                print(f"缺陷类型 {defect} 不在预定义列表中，跳过文件 {json_file}")
                continue
            defect_str = defect

        image_path_full = os.path.join(image_dir, image_file)
        img = cv2.imread(image_path_full)
        if img is None:
            print(f"读取图片 {image_path_full} 失败，跳过。")
            continue
        if json_width is None or json_height is None:
            json_height, json_width = img.shape[:2]

        # 保存原始图片及其标注（子编号 0）
        original_img_name = f"{board_id}_{defect_str}_{file_index}_0.jpg"
        original_name_base = f"{board_id}_{defect_str}_{file_index}_0"
        cv2.imwrite(os.path.join(output_image_dir, original_img_name), img)
        original_yolo = convert_to_yolo(bboxes, labels, json_width, json_height)
        original_label_name = f"{board_id}_{defect_str}_{file_index}_0.txt"
        with open(os.path.join(output_label_dir, original_label_name), 'w') as f:
            f.write(original_yolo)
        print(f"保存原始图片和标注：{original_img_name} / {original_label_name}")
        all_generated_files.append(original_name_base)

        # 对原图进行数据增强（只对有缺陷的图像进行增强）
        for i in range(need_aug_num):
            augmenter = DataAugmentForObjectDetection()
            aug_img, aug_bboxes = augmenter.dataAugment(img, bboxes)
            new_height, new_width = aug_img.shape[:2]
            yolo_annotation = convert_to_yolo(aug_bboxes, labels, new_width, new_height)
            aug_img_name = f"{board_id}_{defect_str}_{file_index}_{i+1}.jpg"
            aug_name_base = f"{board_id}_{defect_str}_{file_index}_{i+1}"
            cv2.imwrite(os.path.join(output_image_dir, aug_img_name), aug_img)
            aug_label_name = f"{board_id}_{defect_str}_{file_index}_{i+1}.txt"
            with open(os.path.join(output_label_dir, aug_label_name), 'w') as f:
                f.write(yolo_annotation)
            print(f"保存增强图片和标注：{aug_img_name} / {aug_label_name}")
            all_generated_files.append(aug_name_base)

    # 划分训练集、验证集和测试集
    print(f"总共生成 {len(all_generated_files)} 个文件")
    random.shuffle(all_generated_files)  # 随机打乱文件列表
    
    
    # 划分比例：训练集70%，验证集20%，测试集10%
    val_split = int(len(all_generated_files) * 0.2)
    test_split = int(len(all_generated_files) * 0.1)
    
    test_files = all_generated_files[:test_split]
    val_files = all_generated_files[test_split:test_split+val_split]
    train_files = all_generated_files[test_split+val_split:]
    
    # 创建数据集划分目录结构
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base, "labels", split), exist_ok=True)
    
    # 将文件移动到对应的训练/验证/测试目录
    def move_files_to_split(file_list, split_name):
        for file_base in file_list:
            src_img = os.path.join(output_image_dir, f"{file_base}.jpg")
            src_label = os.path.join(output_label_dir, f"{file_base}.txt")
            
            dst_img = os.path.join(output_image_dir, split_name, f"{file_base}.jpg")
            dst_label = os.path.join(output_label_dir, split_name, f"{file_base}.txt")
            
            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
    
    move_files_to_split(train_files, "train")
    move_files_to_split(val_files, "val")
    move_files_to_split(test_files, "test")
    print(f"数据集划分完成: 训练集 {len(train_files)}张, 验证集 {len(val_files)}张, 测试集 {len(test_files)}张")
    
    # 生成 dataset.yaml 文件 - 使用从0开始的类别索引
    dataset_yaml = os.path.join(base, "dataset.yaml")
    with open(dataset_yaml, 'w') as f:
        f.write(f"path: {base}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write("names:\n")
        for defect, idx in defect_mapping.items():
            f.write(f"  {idx}: {defect}\n")  # 从0开始的索引
    print("dataset.yaml 文件已生成，类别索引从0开始")   

if __name__ == "__main__":
    main()  