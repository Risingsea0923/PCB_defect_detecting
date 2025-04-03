# image_processing.py
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 让Matplotlib正常显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_histogram_with_cdf(image, title, ax):
    """绘制灰度直方图 + CDF (用于可视化)"""
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    ax.hist(image.flatten(), 256, [0, 256], color='r', alpha=0.6, label='histogram')
    ax.plot(cdf_normalized, color='b', label='cdf')
    ax.set_xlim([0, 256])
    ax.set_title(title)
    ax.legend(loc='upper left')

def manual_hist_equalize(gray_img):
    """手动实现的直方图均衡化"""
    H, W = gray_img.shape
    hist, _ = np.histogram(gray_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    N = H * W

    T = np.round((cdf - cdf_min) / (N - cdf_min) * 255)
    T[T < 0] = 0
    T[T > 255] = 255
    T = T.astype(np.uint8)
    return T[gray_img]

def apply_filter(gray_img, method="nlm", kernel_size=(3,3),
                 h=10, templateWindowSize=7, searchWindowSize=21):
    """
    对灰度图进行滤波:
      - method: "mean", "median", "gaussian", "nlm"
    """
    if method == "mean":
        return cv2.blur(gray_img, kernel_size)
    elif method == "median":
        k = kernel_size[0]
        return cv2.medianBlur(gray_img, k)
    elif method == "gaussian":
        return cv2.GaussianBlur(gray_img, kernel_size, 0)
    elif method == "nlm":
        return cv2.fastNlMeansDenoising(gray_img, None, h, templateWindowSize, searchWindowSize)
    else:
        return gray_img

def do_sharpen(gray_img, method="laplacian", weight=1.0):
    """
    对灰度图进行锐化:
      - method: "gradient" or "laplacian"
      - weight: 锐化叠加系数
    """
    if method == "gradient":
        sobelx = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=3)
        absx = cv2.convertScaleAbs(sobelx)
        absy = cv2.convertScaleAbs(sobely)
        edges = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        sharpened = cv2.addWeighted(gray_img, 1.0, edges, weight, 0)
    else:
        lap = cv2.Laplacian(gray_img, cv2.CV_16S, ksize=3)
        lap_abs = cv2.convertScaleAbs(lap)
        sharpened = cv2.addWeighted(gray_img, 1.0, lap_abs, weight, 0)
    return sharpened

def do_equalize(gray_img, method="manual", clipLimit=2.0, tileGridSize=(8,8)):
    """
    对灰度图进行直方图均衡:
      - method: "manual", "cv2", "clahe"
    """
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    if method == "manual":
        return manual_hist_equalize(gray_img)
    elif method == "cv2":
        return cv2.equalizeHist(gray_img)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(gray_img)
    else:
        return gray_img

def do_edge_extraction(gray_img, method="canny"):
    """可选的边缘提取 (Canny / Sobel / Laplacian)"""
    if method == "canny":
        return cv2.Canny(gray_img, 100, 200)
    elif method == "sobel":
        sobelx = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1, ksize=3)
        absx = cv2.convertScaleAbs(sobelx)
        absy = cv2.convertScaleAbs(sobely)
        return cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    elif method == "laplacian":
        lap = cv2.Laplacian(gray_img, cv2.CV_16S, ksize=3)
        return cv2.convertScaleAbs(lap)
    else:
        return gray_img
def threshold_segment(gray_img, method="manual", val=65):
    """
    对灰度图进行阈值分割:
      - method=="manual": 使用手动阈值 val
      - method=="otsu": 使用 OTSU 自适应阈值
    返回二值图(0/255)
    """
    if method=="otsu":
        _, out = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, out = cv2.threshold(gray_img, val, 255, cv2.THRESH_BINARY)
    return out

def show_histogram(img):
    """
    显示图像的灰度直方图和CDF。
    若图像不是灰度图，则先转换为灰度。
    """
    if img is None:
        print("图像为空")
        return
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    plt.figure(figsize=(8,4))
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    ax1.imshow(gray, cmap='gray')
    ax1.set_title("原图")
    ax1.axis("off")
    # 计算直方图和CDF
    hist, bins = np.histogram(gray.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    ax2.hist(gray.flatten(), 256, [0,256], color='r', alpha=0.6, label='histogram')
    ax2.plot(cdf_normalized, color='b', label='CDF')
    ax2.set_xlim([0,256])
    ax2.set_title("直方图+CDF")
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def optimal_threshold(img):
    """
    遍历0~255的手动阈值，计算类间方差（Between-Class Variance），
    返回使类间方差最大的阈值和对应的方差值。
    """
    if img is None:
        print("图像为空")
        return None, 0
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    total_pixels = gray.size
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    prob = hist / total_pixels
    omega = np.cumsum(prob)       # 类0的累计概率
    mu = np.cumsum(np.arange(256) * prob)  # 类0的累计均值
    mu_total = mu[-1]

    best_thresh = 0
    max_between = 0
    for t in range(256):
        if omega[t] == 0 or omega[t] == 1:
            continue
        between = ((mu_total * omega[t] - mu[t]) ** 2) / (omega[t] * (1 - omega[t]) + 1e-6)
        if between > max_between:
            max_between = between
            best_thresh = t
    return best_thresh, max_between


def morphology_erosion(bin_img, ksize=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(bin_img, kernel, iterations=iterations)

def morphology_dilation(bin_img, ksize=3, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(bin_img, kernel, iterations=iterations)

def morphology_open(bin_img, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

def morphology_close(bin_img, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

def concatenate_images_horizontally(img_list, target_height):
    resized_imgs = []
    for img in img_list:
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height))
        resized_imgs.append(resized)
    if len(resized_imgs)==0:
        return None
    return cv2.hconcat(resized_imgs)

def morph_compare(img, thresh_method="manual", manual_val=65, morph_method="none", ksize=3, iterations=1):
    """
    对输入图像进行阈值分割，然后依次对其使用四种形态学运算（腐蚀、膨胀、开运算、闭运算），
    最后将四个结果横向拼接为一张图返回。
    """
    bin_img = threshold_segment(img, method=thresh_method, val=manual_val)
    results = []
    methods = ["erosion", "dilation", "open", "close"]
    for method in methods:
        if method == "erosion":
            proc = morphology_erosion(bin_img, ksize=ksize, iterations=iterations)
        elif method == "dilation":
            proc = morphology_dilation(bin_img, ksize=ksize, iterations=iterations)
        elif method == "open":
            proc = morphology_open(bin_img, ksize=ksize)
        elif method == "close":
            proc = morphology_close(bin_img, ksize=ksize)
        else:
            proc = bin_img
        # 在每个处理结果上加上文字标注
        cv2.putText(proc, method, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        results.append(proc)
    composite = concatenate_images_horizontally(results, target_height=img.shape[0])
    return composite


def optimal_threshold_curve(gray_img):
    """
    遍历阈值 0～255，计算每个阈值下的类间方差，
    绘制阈值与类间方差的曲线，并返回使类间方差最大的阈值及其方差值。
    """
    if len(gray_img.shape) != 2:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()
    total = gray.size
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    prob = hist / total
    omega = np.cumsum(prob)       # 累计概率
    mu = np.cumsum(np.arange(256)*prob)  # 累计均值
    mu_total = mu[-1]
    between_var = np.zeros(256)
    for t in range(256):
        if omega[t] * (1 - omega[t]) == 0:
            between_var[t] = 0
        else:
            between_var[t] = ((mu_total * omega[t] - mu[t]) ** 2) / (omega[t]*(1-omega[t]) + 1e-6)
    best_thresh = np.argmax(between_var)
    
    # 绘制曲线
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(256), between_var, label="类间方差")
    plt.xlabel("阈值")
    plt.ylabel("类间方差")
    plt.title("最佳阈值曲线")
    plt.axvline(x=best_thresh, color='r', linestyle='--', label=f"最佳阈值={best_thresh}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_thresh, between_var[best_thresh]


def adaptive_threshold(gray_img, blockSize=31, C=5, method="mean"):
    """
    自适应阈值：
      - gray_img: 灰度图
      - blockSize: 邻域块大小，必须为奇数
      - C: 从均值中减去的常数
      - method: "mean" 或 "gaussian"
    返回二值图
    """
    if len(gray_img.shape) != 2:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()
        
    if method == "gaussian":
        adapt_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        adapt_method = cv2.ADAPTIVE_THRESH_MEAN_C

    bin_img = cv2.adaptiveThreshold(gray, 255, adapt_method,
                                    cv2.THRESH_BINARY, blockSize, C)
    return bin_img
def threshold_segment(img, method="manual", val=65, blockSize=31, C=5):
    """
    对图像进行阈值分割：
      - method=="none": 不做阈值分割，直接返回灰度图
      - method=="manual": 使用固定阈值 val
      - method=="otsu": 使用 OTSU 自适应阈值
      - method=="adaptive-mean": 使用自适应均值阈值，需 blockSize 和 C
      - method=="adaptive-gauss": 使用自适应高斯阈值，需 blockSize 和 C
    """
    import cv2
    if img is None:
        return None
    if len(img.shape) != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if method == "none":
        return gray
    elif method == "manual":
        _, out = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    elif method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive-mean":
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, blockSize, C)
    elif method == "adaptive-gauss":
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, blockSize, C)
    else:
        _, out = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    return out


def remove_small_components(bin_img, min_area=50):
    """
    通过连通域分析过滤掉面积小于 min_area 的噪声区域。
    参数：
      - bin_img: 输入的二值图（0/255）
      - min_area: 最小连通域面积（像素数），低于该值的连通域被视为噪声并去除
    返回：
      - 处理后的二值图
    """
    import cv2
    import numpy as np
    if bin_img is None:
        return None
    out = bin_img.copy()
    # 使用 connectedComponents 对二值图进行连通域分析
    num_labels, labels = cv2.connectedComponents(out, connectivity=8)
    for label in range(1, num_labels):
        mask = (labels == label)
        area = int(np.sum(mask))
        if area < min_area:
            out[mask] = 0
    return out


def multi_scale_filter(gray_img, scales=[1, 2, 3, 4]):
    """
    对灰度图在多个尺度下进行高斯滤波，并将结果横向拼接进行对比。
    参数：
      - gray_img: 输入的灰度图
      - scales: sigma 值列表，表示不同的高斯平滑尺度
    返回：
      - 拼接后的对比图
    """
    import cv2
    import numpy as np
    filtered_images = []
    for sigma in scales:
        # 采用高斯滤波，kernel size 自动根据 sigma 确定（这里使用 (0,0)）
        filtered = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # 添加文字标注显示当前 sigma 值
        cv2.putText(filtered, f"sigma={sigma}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
        filtered_images.append(filtered)
    composite = concatenate_images_horizontally(filtered_images, target_height=gray_img.shape[0])
    return composite

def concatenate_images_horizontally(img_list, target_height):
    """
    将图像列表横向拼接到一起，所有图像统一缩放到 target_height 高度。
    """
    import cv2
    import numpy as np
    resized = []
    for img in img_list:
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_height)))
    if len(resized) == 0:
        return None
    return cv2.hconcat(resized)
def optimal_threshold_curve(gray_img, method="manual"):
    """
    遍历0~255阈值，计算每个阈值下的类间方差，并绘制阈值-类间方差曲线，
    返回最佳阈值和对应的最大类间方差。

    参数：
      - gray_img: 灰度图
      - method: 阈值分割方法。支持 "manual"、"adaptive-mean"、"adaptive-gauss"（仅这些方法执行遍历），
                对于 "otsu" 或 "none" 则不进行遍历，直接返回 None。
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    if len(gray_img.shape) != 2:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img.copy()

    # 如果选择的是 otsu 或 none，则不进行阈值遍历
    if method in ["otsu", "none"]:
        print("当前阈值方法为 {}, 不执行阈值遍历".format(method))
        return None, None

    total_pixels = gray.size
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    prob = hist / total_pixels
    omega = np.cumsum(prob)
    mu = np.cumsum(np.arange(256) * prob)
    mu_total = mu[-1]

    between_var = np.zeros(256)
    for t in range(256):
        if omega[t] == 0 or omega[t] == 1:
            between_var[t] = 0
        else:
            between_var[t] = ((mu_total * omega[t] - mu[t]) ** 2) / (omega[t]*(1-omega[t]) + 1e-6)

    best_thresh = int(np.argmax(between_var))
    max_between = between_var[best_thresh]

    plt.figure()
    plt.plot(range(256), between_var, marker='o', label="类间方差")
    plt.xlabel("阈值")
    plt.ylabel("类间方差")
    plt.title(f"{method} 阈值分割类间方差曲线")
    plt.axvline(x=best_thresh, color='r', linestyle='--', label=f"最佳阈值={best_thresh}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_thresh, max_between
def analyze_components(bin_img, min_area=50):
    """
    对二值图 bin_img 进行连通域分析，返回:
      - marked_img: 在原二值图或者彩色图上标记出的连通域
      - components_info: 列表，每个元素包含 {id, area, perimeter, centroid=(cx,cy)}

    参数:
      bin_img: 二值图(0/255)
      min_area: 连通域的最小面积，低于此面积的将会被跳过
    """
    import cv2
    import numpy as np

    # 先将二值图复制并转成三通道，用来画标记
    if len(bin_img.shape) == 2:
        marked_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    else:
        marked_img = bin_img.copy()

    # 调用 connectedComponentsWithStats 进行连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    components_info = []
    obj_id = 1  # 连通域编号计数

    # 从1开始，0 通常是背景
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_area:
            continue  # 面积太小就跳过

        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        cX, cY = centroids[label_idx]

        # 计算连通域周长(轮廓)
        mask = (labels == label_idx).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = 0.0
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)

        # 在 marked_img 上画出矩形框和质心
        color = (0, 0, 255)  # 红色
        cv2.rectangle(marked_img, (x, y), (x + w, y + h), color, 2)
        cv2.circle(marked_img, (int(cX), int(cY)), 4, (0, 255, 0), -1)
        cv2.putText(marked_img, f"ID:{obj_id}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        components_info.append({
            "id": obj_id,
            "area": float(area),
            "perimeter": float(perimeter),
            "centroid": (float(cX), float(cY))
        })
        obj_id += 1

    return marked_img, components_info
