import cv2
import os

# -----------请自行修改以下路径-----------
input_dir = r'\\tsclient\E\HW5502\Photo'      # 原始图像文件夹
output_dir = r'D:\030923\dataset\images_0'    # 处理后图像保存文件夹
os.makedirs(output_dir, exist_ok=True)

from PIL import Image
'''
image_path = r'\\tsclient\E\HW5502\Photo\2000_1201_015120_001.jpg'  # 替换为实际的图像路径
with Image.open(image_path) as img:
    width, height = img.size  # PIL返回 (width, height)
    print(f"图像分辨率：{width} x {height}")
'''


# 目标分辨率（可根据需要灵活调整）
#new_size = (3000, 2000)
new_size = (9000, 6000)

# 第一步：压缩质量（JPEG），数值越低文件体积越小，但画质也越差
intermediate_quality = 90

# 第二步：滤波 / 去噪操作参数
#  - fastNlMeansDenoisingColored常用参数:
denoise_h = 5
denoise_hColor = 5
templateWindowSize = 7
searchWindowSize = 21

# 高斯模糊参数
gaussian_kernel = (3, 3)
sigma = 0

# 最终输出的JPEG质量
final_quality = 90

# -------------------------------------

def process_image(img_path, out_dir):
    file_name = os.path.basename(img_path)
    
    # 1. 读取原始图像
    image = cv2.imread(img_path)
    if image is None:
        print(f"无法读取: {img_path}")
        return
    
    # 2. 先降低分辨率
    #    将大图直接resize到较小尺寸，从而减少后续滤波、去噪的计算量
    resized = cv2.resize(image, new_size)
    
    # 3. 先对 resized 图像进行一次JPEG压缩，以减少其占用的存储大小
    #    这一步是可选的，如果仅仅是想在内存中做后续操作，也可跳过实际写入过程
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), intermediate_quality]
    success, encoded_img = cv2.imencode('.jpg', resized, encode_param)
    if not success:
        print(f"编码失败: {img_path}")
        return
    
    # 将压缩后的图像重新解码到内存中，这样接下来就对“压缩后的小图”做滤波、去噪
    compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    
    # 4. 对图像做滤波、去噪等操作
    # 4.1 去噪
    denoised = cv2.fastNlMeansDenoisingColored(
        compressed_img, None,
        h=denoise_h,
        hColor=denoise_hColor,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize
    )
    
    # 4.2 高斯模糊/平滑
    blurred = cv2.GaussianBlur(denoised, gaussian_kernel, sigma)
    
    # 5. 最后以期望质量保存到输出文件夹
    #    按照最终的JPEG质量（例如70）写出
    out_path = os.path.join(out_dir, file_name)
    cv2.imwrite(out_path, blurred, [int(cv2.IMWRITE_JPEG_QUALITY), final_quality])
    
    print(f"处理完成: {img_path} -> {out_path}")


if __name__ == "__main__":
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    for file_name in os.listdir(input_dir):
        lower_name = file_name.lower()
        if any(lower_name.endswith(ext) for ext in valid_exts):
            img_path = os.path.join(input_dir, file_name)
            process_image(img_path, output_dir)
    
    print("全部图像处理完毕!")
