# image_registration.py
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
def extract_keypoints_harris(gray_img, blockSize=2, ksize=3, k=0.04, thresh=0.01):
    """
    Harris角点检测。为了后续能和SIFT/SURF一样进行匹配，需要再用某种
    描述子（如BRIEF、SIFT等）给这些角点生成描述子。
    """
    # 1) 做Harris检测
    harris = cv2.cornerHarris(np.float32(gray_img), blockSize, ksize, k)
    harris_dilate = cv2.dilate(harris, None)
    max_r = harris_dilate.max()

    # 2) 收集角点 KeyPoint
    kp = []
    h, w = gray_img.shape[:2]
    for y in range(h):
        for x in range(w):
            if harris_dilate[y,x] > thresh*max_r:
                kp.append(cv2.KeyPoint(float(x), float(y), 3))

    # 3) 给这些 Harris角点生成描述子。可用BRIEF或ORB/SIFT描述子
    #   这里用ORB示例
    orb = cv2.ORB_create()
    # ORB的compute需要先传关键点列表
    kp, des = orb.compute(gray_img, kp)

    return kp, des
def extract_keypoints_sift(gray_img, nfeatures=500):
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures)
    except:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray_img, None)
    return kp, des
def extract_keypoints_surf(gray_img, hessianThreshold=400):
    """
    强制使用 SURF，不再退化到 SIFT。
    如果本地没有 xfeatures2d.SURF_create，就会报错。
    """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    kp, des = surf.detectAndCompute(gray_img, None)
    return kp, des

def extract_keypoints_orb(gray_img, nfeatures=500):
    """
    使用ORB提取特征, nfeatures可调
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(gray_img, None)
    return kp, des
def match_features(des1, des2, crossCheck=True, method="BF"):
    """
    匹配两组特征描述符
    参数:
        des1, des2: 特征描述符
        crossCheck: 是否使用交叉检查
        method: 匹配方法，"BF"或"FLANN"
    返回:
        matches: 匹配结果
    """
    # 检查描述符是否为空或无效
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []
    
    # 确保描述符格式正确
    des1 = np.float32(des1) if des1.dtype != np.float32 else des1
    des2 = np.float32(des2) if des2.dtype != np.float32 else des2
    
    try:
        if method == "BF":
            # 根据描述符类型选择距离度量
            if des1.dtype == np.uint8:
                # 二进制描述符使用汉明距离
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
            else:
                # 浮点描述符使用L2距离
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
            
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
        elif method == "FLANN":
            # FLANN参数
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            matches = good_matches
        else:
            raise ValueError(f"不支持的匹配方法: {method}")
        
        return matches
    except Exception as e:
        print(f"特征匹配失败: {str(e)}")
        return []
def draw_matches(img1, kp1, img2, kp2, matches, max_draw=50):
    """绘制特征点匹配结果"""
    # 获取图像尺寸，考虑彩色图像的情况
    if len(img1.shape) == 3:
        h1, w1, _ = img1.shape
    else:
        h1, w1 = img1.shape
        
    if len(img2.shape) == 3:
        h2, w2, _ = img2.shape
    else:
        h2, w2 = img2.shape
    
    """将两张灰度图左右拼接并用线连接匹配点"""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    H = max(h1, h2)
    W = w1 + w2
    combo = np.zeros((H, W, 3), dtype=np.uint8)

    # 左图(灰度->BGR)
    combo[:h1, :w1, 0] = img1
    combo[:h1, :w1, 1] = img1
    combo[:h1, :w1, 2] = img1

    # 右图
    combo[:h2, w1:w1+w2, 0] = img2
    combo[:h2, w1:w1+w2, 1] = img2
    combo[:h2, w1:w1+w2, 2] = img2

    for m in matches[:max_draw]:
        idx1 = m.queryIdx
        idx2 = m.trainIdx
        x1,y1 = kp1[idx1].pt
        x2,y2 = kp2[idx2].pt
        x2_off = x2 + w1
        cv2.circle(combo, (int(x1), int(y1)), 3, (0,0,255), -1)
        cv2.circle(combo, (int(x2_off), int(y2)), 3, (0,0,255), -1)
        cv2.line(combo, (int(x1), int(y1)), (int(x2_off), int(y2)), (0,255,0), 1)
    return combo

def do_registration(std_img, test_img, detector="SIFT", orb_nfeatures=500, harris_thresh=0.01): 
    """
    根据 detector ("SIFT", "ORB", "Harris") 进行特征检测与匹配，并返回匹配拼接图、
    配准后的 test_img 及匹配得分。
    对于ORB，参数 orb_nfeatures 可调；对于 Harris，阈值 harris_thresh 可调。
    
    返回：
    - combo: 标准图 & 待测图的特征匹配拼接图
    - aligned_test_img: 变换后的待测图（已对齐到标准图坐标）
    - score: 匹配得分
    """
    if detector.upper() == "SIFT":
        kp1, des1 = extract_keypoints_sift(std_img, nfeatures=500)
        kp2, des2 = extract_keypoints_sift(test_img, nfeatures=500)
    elif detector.upper() == "ORB":
        kp1, des1 = extract_keypoints_orb(std_img, nfeatures=orb_nfeatures)
        kp2, des2 = extract_keypoints_orb(test_img, nfeatures=orb_nfeatures)
    elif detector.upper() == "HARRIS":
        kp1, des1 = extract_keypoints_harris(std_img, thresh=harris_thresh)
        kp2, des2 = extract_keypoints_harris(test_img, thresh=harris_thresh)
    else:
        return None, None, 0.0  # **返回None，表示失败**

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return None, None, 0.0

    matches = match_features(des1, des2)
    if not matches:
        return None, None, 0.0

    avg_dist = np.mean([m.distance for m in matches])
    good = [m for m in matches if m.distance < 0.7 * avg_dist]
    score = len(good) / len(matches)

    # 绘制匹配拼接图
    combo = draw_matches(std_img, kp1, test_img, kp2, matches)

    # **计算 Homography 矩阵**
    if len(good) > 4:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            h, w = std_img.shape[:2]
            aligned_test_img = cv2.warpPerspective(test_img, M, (w, h))
        else:
            aligned_test_img = None
    else:
        aligned_test_img = None

    return combo, aligned_test_img, score  # **返回对齐后的图**

def concatenate_images_horizontally(img_list, target_height=300):
    """
    将列表中的图像按横向拼接，先将所有图像调整为相同高度。
    """
    resized_imgs = []
    for img in img_list:
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height))
        resized_imgs.append(resized)
    if not resized_imgs:
        return None
    composite = cv2.hconcat(resized_imgs)
    return composite

# 修改 extract_keypoints_harris 函数，确保输入图像类型正确
def extract_keypoints_harris(img, thresh=0.01, blockSize=2, ksize=3, k=0.04):
    """使用Harris角点检测提取关键点"""
    # 确保输入图像是灰度图
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    
    # 确保图像类型为 uint8
    gray_img = gray_img.astype(np.uint8)
    
    # 执行Harris角点检测
    harris = cv2.cornerHarris(gray_img, blockSize, ksize, k)
    
    # 阈值处理
    harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX)
    harris_norm = harris_norm.astype(np.uint8)
    _, harris_thresh = cv2.threshold(harris_norm, int(thresh * 255), 255, cv2.THRESH_BINARY)
    
    # 找到角点坐标
    coords = np.column_stack(np.where(harris_thresh > 0))
    
    # 创建KeyPoint对象
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in coords]
    
    # 计算描述符 (简单使用坐标作为描述符)
    descriptors = np.array([np.array([kp.pt[0], kp.pt[1]], dtype=np.float32) for kp in keypoints])
    
    return keypoints, descriptors
def compare_detectors(std_img, test_img, method='sift', sift_nfeatures=500, orb_nfeatures=500, harris_thresh=0.01):
    """特征对比主函数（修正版）"""
    # 统一图像为灰度图
    std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY) if len(std_img.shape) == 3 else std_img
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img

    results = []
    composite = []
    
    try:
        # 根据选择的配准方法执行对应算法
        if method == 'sift':
            sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
            kp1, des1 = sift.detectAndCompute(std_gray, None)
            kp2, des2 = sift.detectAndCompute(test_gray, None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                results.append({
                    'method': 'SIFT',
                    'matches': len(good),
                    'std_kp': len(kp1),
                    'test_kp': len(kp2)
                })
                if len(good) > 10:
                    composite = cv2.drawMatches(std_img, kp1, test_img, kp2, good[:50], None)

        elif method == 'orb':
            orb = cv2.ORB_create(nfeatures=orb_nfeatures)
            kp1, des1 = orb.detectAndCompute(std_gray, None)
            kp2, des2 = orb.detectAndCompute(test_gray, None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                good = sorted(matches, key=lambda x: x.distance)[:100]
                results.append({
                    'method': 'ORB',
                    'matches': len(good),
                    'std_kp': len(kp1),
                    'test_kp': len(kp2)
                })
                if len(good) > 10:
                    composite = cv2.drawMatches(std_img, kp1, test_img, kp2, good[:50], None)

        # 无匹配结果时生成错误提示图像
        if not composite:
            h, w = std_img.shape[:2]
            composite = np.zeros((max(480, h), w*2, 3), dtype=np.uint8)
            cv2.putText(composite, "未找到有效匹配", (w//2-100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        return composite, results
        
    except Exception as e:
        error_img = np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(error_img, f"配准错误: {str(e)}", (50,240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return error_img, [{'method': 'Error', 'matches': 0, 'std_kp': 0, 'test_kp': 0}]
def create_matches_composite(img1, img2, keypoints1_list, keypoints2_list, desc1_list, desc2_list, detector_names):
    """
    创建一个包含多个检测器匹配结果的复合图像
    
    参数:
        img1, img2: 输入图像
        keypoints1_list, keypoints2_list: 各检测器的关键点列表
        desc1_list, desc2_list: 各检测器的描述符列表
        detector_names: 检测器名称列表
        
    返回:
        composite: 复合图像
    """
    # 计算复合图像的尺寸
    h, w = img1.shape[:2]
    
    # 创建复合图像 - 三种检测器，每种占一行
    composite = np.zeros((h*3, w*2, 3), dtype=np.uint8)
    
    # 转换灰度图为BGR以便绘制彩色线条
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
        
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    
    # 为每个检测器绘制匹配结果
    for i, (kp1, kp2, des1, des2, name) in enumerate(zip(keypoints1_list, keypoints2_list, desc1_list, desc2_list, detector_names)):
        # 计算当前检测器结果的位置
        y_offset = i * h
        
        # 绘制左侧图像
        composite[y_offset:y_offset+h, 0:w] = img1_color
        
        # 绘制右侧图像
        composite[y_offset:y_offset+h, w:w*2] = img2_color
        
        # 进行特征匹配
        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            try:
                # 根据检测器类型选择匹配方法
                if name == "ORB" or name == "Harris":
                    # ORB和Harris使用汉明距离
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                else:
                    # SIFT使用L2距离
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda m: m.distance)
                
                # 绘制匹配线条
                max_matches = min(50, len(matches))  # 限制显示的匹配数量
                for m in matches[:max_matches]:
                    pt1 = tuple(map(int, kp1[m.queryIdx].pt))
                    pt2 = tuple(map(int, [kp2[m.trainIdx].pt[0] + w, kp2[m.trainIdx].pt[1]]))
                    cv2.line(composite[y_offset:y_offset+h, :], pt1, pt2, (0, 255, 0), 1)
                
                # 添加匹配数量
                cv2.putText(composite, f"{name}: {len(matches)}匹配", (10, y_offset + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as e:
                cv2.putText(composite, f"{name}: 匹配失败 ({str(e)[:20]})", (10, y_offset + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(composite, f"{name}: 无特征点或描述符", (10, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return composite

def plot_sift_curve(std_img, test_img):
    """
    在此文件中进行绘图：
    遍历一定范围的 SIFT nfeatures，对每个值提取并匹配特征，计算匹配率，然后绘图。
    """
    if std_img is None or test_img is None:
        print("标准图或待测图为空，无法绘制SIFT曲线")
        return
    sift_rates = []
    n_values = list(range(100, 2100, 200))
    for n in n_values:
        kp1, des1 = extract_keypoints_sift(std_img, nfeatures=n)
        kp2, des2 = extract_keypoints_sift(test_img, nfeatures=n)
        if des1 is None or des2 is None or len(kp1)==0:
            rate = 0
        else:
            matches = match_features(des1, des2)
            rate = len(matches) / len(kp1) if len(kp1)>0 else 0
        sift_rates.append(rate)

    plt.figure()
    plt.plot(n_values, sift_rates, marker='o')
    plt.xlabel("SIFT nfeatures")
    plt.ylabel("匹配率")
    plt.title("SIFT匹配率曲线")
    plt.grid(True)
    plt.show()

def plot_surf_curve(std_img, test_img):
    """
    遍历一定范围的 SURF hessianThreshold，对每个值提取并匹配特征，计算匹配率，然后绘图。
    """
    if std_img is None or test_img is None:
        print("标准图或待测图为空，无法绘制SURF曲线")
        return
    surf_rates = []
    hessian_values = list(range(50, 1050, 100))
    for h in hessian_values:
        kp1, des1 = extract_keypoints_surf(std_img, hessianThreshold=h)
        kp2, des2 = extract_keypoints_surf(test_img, hessianThreshold=h)
        if des1 is None or des2 is None or len(kp1)==0:
            rate = 0
        else:
            matches = match_features(des1, des2)
            rate = len(matches) / len(kp1) if len(kp1)>0 else 0
        surf_rates.append(rate)

    plt.figure()
    plt.plot(hessian_values, surf_rates, marker='o')
    plt.xlabel("SURF hessianThreshold")
    plt.ylabel("匹配率")
    plt.title("SURF匹配率曲线")
    plt.grid(True)
    plt.show()
def plot_orb_curve(std_img, test_img):
    if std_img is None or test_img is None:
        print("图像为空，无法绘制ORB曲线")
        return
    orb_rates = []
    n_values = list(range(100, 2100, 200))
    for n in n_values:
        kp1, des1 = extract_keypoints_orb(std_img, nfeatures=n)
        kp2, des2 = extract_keypoints_orb(test_img, nfeatures=n)
        if des1 is None or len(kp1)==0:
            rate = 0
        else:
            matches = match_features(des1, des2)
            rate = len(matches) / len(kp1)
        orb_rates.append(rate)
    plt.figure()
    plt.plot(n_values, orb_rates, marker='o')
    plt.xlabel("ORB nfeatures")
    plt.ylabel("匹配率")
    plt.title("ORB匹配率曲线")
    plt.grid(True)
    plt.show()

def plot_harris_curve(std_img, test_img):
    if std_img is None or test_img is None:
        print("图像为空，无法绘制Harris曲线")
        return
    harris_rates = []
    thresh_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    for t in thresh_values:
        kp1, des1 = extract_keypoints_harris(std_img, thresh=t)
        kp2, des2 = extract_keypoints_harris(test_img, thresh=t)
        if des1 is None or len(kp1)==0:
            rate = 0
        else:
            matches = match_features(des1, des2)
            rate = len(matches) / len(kp1)
        harris_rates.append(rate)
    plt.figure()
    plt.plot(thresh_values, harris_rates, marker='o')
    plt.xlabel("Harris 阈值")
    plt.ylabel("匹配率")
    plt.title("Harris匹配率曲线")
    plt.grid(True)
    plt.show()
def align_and_overlay(std_img, test_img, detector="SIFT", nfeatures=500, h_thresh=0.01):
    # 预处理：统一转换为灰度图处理
    if len(std_img.shape) == 3:
        std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
        std_display = std_img.copy()
    else:
        std_gray = std_img.copy()
        std_display = cv2.cvtColor(std_gray, cv2.COLOR_GRAY2BGR)
    
    if len(test_img.shape) == 3:
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_display = test_img.copy()
    else:
        test_gray = test_img.copy()
        test_display = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR)

    # 1. 提取特征点
    if detector.upper() == "SIFT":
        kp1, des1 = extract_keypoints_sift(std_img, nfeatures=nfeatures)
        kp2, des2 = extract_keypoints_sift(test_img, nfeatures=nfeatures)
        norm = cv2.NORM_L2
    elif detector.upper() == "ORB":
        kp1, des1 = extract_keypoints_orb(std_img, nfeatures=nfeatures)
        kp2, des2 = extract_keypoints_orb(test_img, nfeatures=nfeatures)
        norm = cv2.NORM_HAMMING
    elif detector.upper() == "HARRIS":
        kp1, des1 = extract_keypoints_harris(std_img, thresh=h_thresh)
        kp2, des2 = extract_keypoints_harris(test_img, thresh=h_thresh)
        norm = cv2.NORM_HAMMING
    else:
        return None, None, 0.0

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return None, None, 0.0

    # 2. 进行特征匹配
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        return None, None, 0.0
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = matches[:max(int(len(matches)*0.3), 4)]

    # 3. 计算透视变换
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None, None, 0.0
    # 修改这部分：处理对齐后的图像通道
    aligned = cv2.warpPerspective(test_gray, M, (std_img.shape[1], std_img.shape[0]))
    # 4. 颜色通道区分
    # 确保 std_display 和 aligned 都是三通道
    if len(std_display.shape) == 2:
        std_display = cv2.cvtColor(std_display, cv2.COLOR_GRAY2BGR)
    aligned_display = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    
    # 颜色通道处理
    std_display[:, :, 1] = 0      # 绿色通道设为0
    std_display[:, :, 2] = 0      # 红色通道设为0
    aligned_display[:, :, 0] = 0  # 蓝色通道设为0
    aligned_display[:, :, 1] = 0  # 绿色通道设为0



    # 5. 叠加显示
    overlay = cv2.addWeighted(std_display, 0.5, aligned_display, 0.5, 0)

    # 6. 计算匹配率
    inliers = mask.ravel().tolist().count(1)
    score = inliers / len(good_matches)

    # === 修改处：多返回一个 aligned ===
    return overlay, aligned, score
 
def plot_registration_curve(std_img, test_img, detector="SIFT", param_range=None, title=None):
    """
    绘制不同参数下的配准效果曲线
    
    参数:
        std_img: 标准图像
        test_img: 待测图像
        detector: 特征检测器类型，可选 "SIFT", "ORB", "Harris"
        param_range: 参数范围，如果为None则使用默认范围
        title: 图表标题，如果为None则自动生成
    """
    if param_range is None:
        if detector == "SIFT":
            param_range = [100, 200, 300, 500, 1000, 2000]
        elif detector == "ORB":
            param_range = [100, 200, 300, 500, 1000, 2000]
        elif detector == "Harris":
            param_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    scores = []
    times = []
    
    for param in param_range:
        start_time = time.time()
        # 修改这里，正确处理返回值
        result = align_and_overlay(std_img, test_img, detector=detector, nfeatures=param)
        
        # 根据返回值类型进行处理
        if isinstance(result, tuple):
            if len(result) >= 2:
                overlay = result[0]  # 第一个元素是叠加图像
                score = result[1]    # 第二个元素是评分
            else:
                # 如果返回值元组长度不足，设置默认值
                overlay = None
                score = 0.0
        else:
            # 如果返回值不是元组，可能直接返回了图像
            overlay = result
            score = 0.0
            
        end_time = time.time()
        process_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        scores.append(score)
        times.append(process_time)
    
    # 绘制评分曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(param_range, scores, 'o-', color='blue')
    plt.xlabel('参数值')
    plt.ylabel('配准评分')
    if title:
        plt.title(f"{title} - 评分曲线")
    else:
        plt.title(f"{detector}配准评分曲线")
    plt.grid(True)
    
    # 绘制时间曲线
    plt.subplot(1, 2, 2)
    plt.plot(param_range, times, 'o-', color='red')
    plt.xlabel('参数值')
    plt.ylabel('处理时间 (ms)')
    plt.title('处理时间曲线')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()