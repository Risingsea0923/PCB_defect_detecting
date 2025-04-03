#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：PCB缺陷检测
@File    ：detector.py
'''

import cv2
import torch
import numpy as np
from ultralytics import YOLO

class PCBDetector:
    def __init__(self):
        self.model = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.method = "深度学习"  # 可选: "传统方法", "深度学习", "两种方法结合"
        self.review_threshold = 0.6  # 人工复检阈值

    @torch.no_grad()
    def load_model(self, weights):
        """加载模型"""
        try:
            self.model = YOLO(weights)
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False

    def detect_video_frame(self, frame):
        """处理单帧视频"""
        try:
            if self.model is None:
                return {'success': False, 'error': '模型未加载'}

            # 保存原始帧
            original = frame.copy()
            
            # 执行检测
            results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)
            result = results[0]
            
            # 在图像上绘制检测结果
            annotated_frame = result.plot()
            
            # 提取检测信息
            defect_info = []
            for box in result.boxes:
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 获取置信度
                conf = float(box.conf[0])
                
                # 获取类别
                cls = int(box.cls[0])
                cls_name = result.names[cls]
                
                defect_info.append({
                    'type': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

            return {
                'success': True,
                'image': annotated_frame,
                'original': original,
                'info': defect_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def detect_image(self, image, conf=0.25, iou=0.45):
        """检测单张图像"""
        try:
            if self.model is None:
                return {'success': False, 'error': '模型未加载'}

            # 保存原始图像
            original = image.copy()
            
            # 执行检测
            results = self.model(image, conf=conf, iou=iou)
            result = results[0]
            
            # 在图像上绘制检测结果
            annotated_frame = result.plot()
            
            # 提取检测信息
            defect_info = []
            for box in result.boxes:
                # 获取坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 获取置信度
                conf = float(box.conf[0])
                
                # 获取类别
                cls = int(box.cls[0])
                cls_name = result.names[cls]
                
                defect_info.append({
                    'type': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

            return {
                'success': True,
                'image': annotated_frame,
                'original': original,
                'info': defect_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    # 以下是为并行处理器添加的方法
    
    def traditional_detect(self, std_img, test_img):
        """使用传统方法检测缺陷"""
        try:
            # 图像对齐
            aligned = self._align_images(std_img, test_img)
            
            # 图像差分
            if len(std_img.shape) == 3:
                std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
            else:
                std_gray = std_img.copy()
                
            if len(aligned.shape) == 3:
                aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            else:
                aligned_gray = aligned.copy()
                
            diff = cv2.absdiff(std_gray, aligned_gray)
            
            # 二值化
            _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 连通域分析
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制结果
            result_img = test_img.copy()
            if len(result_img.shape) == 2:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                
            defects = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 50:  # 过滤小面积
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                # 添加缺陷信息
                defects.append({
                    "type": "传统检测缺陷",
                    "confidence": 1.0,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })
            
            return {
                "success": True,
                "image": result_img,
                "defects": defects
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def dl_detect(self, img):
        """使用深度学习方法检测缺陷，适配并行处理器接口"""
        try:
            # 调用现有的detect_image方法
            result = self.detect_image(img, conf=self.conf_thres, iou=self.iou_thres)
            
            if not result['success']:
                return result
            
            # 转换结果格式以适配并行处理器
            defects = []
            for info in result['info']:
                x1, y1, x2, y2 = info['bbox']
                defects.append({
                    "type": info['type'],
                    "confidence": info['confidence'],
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1
                })
            
            return {
                "success": True,
                "image": result['image'],
                "defects": defects
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def need_manual_review(self, defects):
        """判断是否需要人工复检"""
        for defect in defects:
            conf = defect.get('confidence', 1.0)
            # 如果有任何缺陷的置信度低于复检阈值，则需要复检
            if conf < self.review_threshold:
                return True
        return False
    
    def _align_images(self, std_img, test_img):
        """对齐两张图像"""
        try:
            # 确保图像是灰度的
            if len(std_img.shape) == 3:
                std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
            else:
                std_gray = std_img.copy()
                
            if len(test_img.shape) == 3:
                test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            else:
                test_gray = test_img.copy()
            
            # 使用ORB特征检测和匹配
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(std_gray, None)
            kp2, des2 = orb.detectAndCompute(test_gray, None)
            
            # 如果没有足够的特征点，直接返回原图
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return test_img
            
            # 特征匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)
            
            # 如果匹配点太少，直接返回原图
            if len(matches) < 4:
                return test_img
            
            # 获取匹配点坐标
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
            
            # 计算变换矩阵
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # 应用变换
            h, w = std_gray.shape
            aligned = cv2.warpPerspective(test_img, M, (w, h))
            
            return aligned
            
        except Exception as e:
            print(f"图像对齐失败: {str(e)}")
            return test_img
    
    def _is_overlap(self, rect1, rect2, threshold=0.5):
        """判断两个矩形是否重叠"""
        x1 = max(rect1['x'], rect2['x'])
        y1 = max(rect1['y'], rect2['y'])
        x2 = min(rect1['x'] + rect1['w'], rect2['x'] + rect2['w'])
        y2 = min(rect1['y'] + rect1['h'], rect2['y'] + rect2['h'])
        
        if x1 >= x2 or y1 >= y2:
            return False
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = rect1['w'] * rect1['h']
        area2 = rect2['w'] * rect2['h']
        
        overlap_ratio = intersection / min(area1, area2)
        
        return overlap_ratio > threshold
    
    def integrated_detection(self, std_img, test_img):
        """结合传统方法和深度学习的检测"""
        try:
            # 1. 传统方法检测
            diff = cv2.absdiff(std_img, test_img)
            if len(diff.shape) == 3:
                diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 2. 连通域分析获取ROI区域
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. 在ROI区域上执行深度学习检测
            result_img = test_img.copy()
            if len(result_img.shape) == 2:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
            
            combined_defects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # 最小面积阈值
                    continue
                    
                # 获取ROI区域
                x, y, w, h = cv2.boundingRect(contour)
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(result_img.shape[1], x + w + padding)
                y2 = min(result_img.shape[0], y + h + padding)
                
                roi = result_img[y1:y2, x1:x2]
                
                # 在ROI区域上运行深度学习检测
                results = self.model(roi, conf=self.conf_thres)
                result = results[0]
                
                # 处理检测结果
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]
                    
                    # 将相对坐标转换为全局坐标
                    box_x1, box_y1, box_x2, box_y2 = map(int, box.xyxy[0])
                    global_x1 = x1 + box_x1
                    global_y1 = y1 + box_y1
                    global_x2 = x1 + box_x2
                    global_y2 = y1 + box_y2
                    
                    # 添加到缺陷信息列表
                    combined_defects.append({
                        "type": name,
                        "confidence": conf,
                        "x": global_x1,
                        "y": global_y1,
                        "w": global_x2 - global_x1,
                        "h": global_y2 - global_y1
                    })
            
            # 在结果图像上标注缺陷
            if combined_defects:
                for defect in combined_defects:
                    x, y = defect['x'], defect['y']
                    w, h = defect['w'], defect['h']
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(result_img, f"{defect['type']}", (x, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            return result_img, combined_defects
            
        except Exception as e:
            print(f"集成检测失败: {str(e)}")
            return None, None