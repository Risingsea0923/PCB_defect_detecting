import os
import cv2
import numpy as np
import time
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

class ParallelProcessor:
    """PCB缺陷检测的并行处理器"""
    
    def __init__(self, detector, max_workers=None):
        """
        初始化并行处理器
        
        参数:
            detector: 缺陷检测器实例
            max_workers: 最大工作线程/进程数，默认为CPU核心数
        """
        self.detector = detector
        self.max_workers = max_workers
        self.progress_callback = None
        self.cancel_flag = False
        self.result_queue = queue.Queue()
        
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def cancel(self):
        """取消处理"""
        self.cancel_flag = True
        
    def process_image(self, image_path, std_img, save_dir):
        """处理单张图像的函数，将被并行调用"""
        if self.cancel_flag:
            return None
            
        try:
            # 加载图像
            img = cv2.imread(image_path)
            if img is None:
                return {
                    'path': image_path,
                    'success': False,
                    'error': '无法加载图像'
                }
            
            # 执行检测
            filename = os.path.basename(image_path)
            method = self.detector.method
            
            if method == "传统方法":
                result = self.detector.traditional_detect(std_img, img)
            elif method == "深度学习":
                result = self.detector.dl_detect(img)
            else:  # 两种方法结合
                result = self.detector.combined_detect(std_img, img)
                
            # 保存结果图像
            if result['success']:
                result_img_path = os.path.join(save_dir, f"result_{filename}")
                cv2.imwrite(result_img_path, result['image'])
                
                # 保存结果信息
                result_info = {
                    'path': image_path,
                    'success': True,
                    'defects': result['defects'],
                    'result_path': result_img_path
                }
                
                # 检查是否需要人工复检
                if self.detector.need_manual_review(result['defects']):
                    result_info['need_review'] = True
                else:
                    result_info['need_review'] = False
                    
                return result_info
            else:
                return {
                    'path': image_path,
                    'success': False,
                    'error': result.get('error', '未知错误')
                }
                
        except Exception as e:
            return {
                'path': image_path,
                'success': False,
                'error': str(e)
            }
    
    def batch_process_thread(self, image_paths, std_img, save_dir, use_process_pool=False):
        """在单独的线程中执行批量处理，避免阻塞UI"""
        try:
            total = len(image_paths)
            processed = 0
            results = []
            
            # 选择线程池或进程池
            executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
            
            with executor_class(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_path = {
                    executor.submit(self.process_image, path, std_img, save_dir): path 
                    for path in image_paths
                }
                
                # 处理完成的任务
                for future in future_to_path:
                    if self.cancel_flag:
                        executor.shutdown(wait=False)
                        break
                        
                    result = future.result()
                    if result:
                        results.append(result)
                        
                    processed += 1
                    if self.progress_callback:
                        self.progress_callback(processed, total, os.path.basename(future_to_path[future]))
            
            # 处理完成后，将结果放入队列
            self.result_queue.put(results)
            
            # 回调通知处理完成
            if self.progress_callback:
                self.progress_callback(total, total, "完成")
                
        except Exception as e:
            # 处理异常
            if self.progress_callback:
                self.progress_callback(-1, total, str(e))
    
    def start_batch_process(self, image_paths, std_img, save_dir, use_process_pool=False):
        """启动批量处理线程"""
        self.cancel_flag = False
        thread = threading.Thread(
            target=self.batch_process_thread,
            args=(image_paths, std_img, save_dir, use_process_pool)
        )
        thread.daemon = True
        thread.start()
        return thread
    
    def get_results(self, timeout=None):
        """获取处理结果，如果还没有完成则等待"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def generate_report(self, results, save_path):
        """生成处理报告"""
        # 创建报告数据
        report_data = []
        
        for result in results:
            if not result['success']:
                report_data.append({
                    '文件名': os.path.basename(result['path']),
                    '处理状态': '失败',
                    '错误信息': result.get('error', '未知错误'),
                    '缺陷数量': 0,
                    '需要复检': 'N/A'
                })
                continue
                
            defects = result['defects']
            
            # 基本信息
            base_info = {
                '文件名': os.path.basename(result['path']),
                '处理状态': '成功',
                '错误信息': '',
                '缺陷数量': len(defects),
                '需要复检': '是' if result.get('need_review', False) else '否'
            }
            
            # 如果没有缺陷，添加一行基本信息
            if not defects:
                report_data.append(base_info)
                continue
                
            # 如果有缺陷，为每个缺陷添加一行
            for i, defect in enumerate(defects):
                if i == 0:
                    # 第一个缺陷行包含基本信息
                    row = base_info.copy()
                else:
                    # 后续缺陷行只填充缺陷信息
                    row = {
                        '文件名': '',
                        '处理状态': '',
                        '错误信息': '',
                        '缺陷数量': '',
                        '需要复检': ''
                    }
                
                # 添加缺陷信息
                row.update({
                    '缺陷ID': i + 1,
                    '缺陷类型': defect.get('type', 'unknown'),
                    '置信度': defect.get('confidence', 'N/A'),
                    'X': defect.get('x', 0),
                    'Y': defect.get('y', 0),
                    '宽': defect.get('w', 0),
                    '高': defect.get('h', 0)
                })
                
                report_data.append(row)
        
        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(report_data)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        return save_path


class VisualizationComparer:
    """可视化对比工具"""
    
    def __init__(self, parent):
        self.parent = parent
        
    def show_comparison_window(self, results_dict):
        """
        显示对比窗口
        
        参数:
            results_dict: 包含不同方法结果的字典，格式为:
                {
                    "方法名": {
                        "image": 结果图像,
                        "defects": 缺陷信息列表
                    },
                    ...
                }
        """
        # 创建对比窗口
        self.comparison_window = tk.Toplevel(self.root)
        self.comparison_window.title("检测方法对比")
        self.comparison_window.geometry("1200x800")
        
        # 创建上下分栏
        top_frame = ttk.Frame(self.comparison_window)
        top_frame.pack(fill="both", expand=True)
        
        bottom_frame = ttk.Frame(self.comparison_window)
        bottom_frame.pack(fill="both", expand=True)
        
        # 配置网格权重，减少空白
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=1)
        
        # 在上部显示图像结果
        for i, (method, result) in enumerate(results_dict.items()):
            frame = ttk.LabelFrame(top_frame, text=method)
            frame.grid(row=0, column=i, padx=2, pady=2, sticky="nsew")
            
            canvas = ZoomableCanvas(frame, width=380, height=300)
            canvas.pack(fill="both", expand=True, padx=2, pady=2)
            canvas.set_image(result["image"])
        
        # 在下部显示统计信息
        for i, (method, result) in enumerate(results_dict.items()):
            stats_frame = ttk.LabelFrame(bottom_frame, text=f"{method}统计信息")
            stats_frame.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            
            # 创建Text控件显示统计信息
            text = tk.Text(stats_frame, height=15, width=40)
            text.pack(fill="both", expand=True, padx=2, pady=2)
            
            # 添加统计信息
            defects = result["defects"]
            text.insert("end", f"检测到的缺陷数量: {len(defects)}\n\n")
            
            # 统计缺陷类型分布
            type_counts = {}
            for defect in defects:
                defect_type = defect.get("type", "未知")
                type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
            
            text.insert("end", "缺陷类型分布:\n")
            for t, count in type_counts.items():
                percentage = count / len(defects) * 100
                text.insert("end", f"- {t}: {count}个 ({percentage:.1f}%)\n")
            
            # 添加置信度信息
            if "confidence" in defects[0]:
                confidences = [d.get("confidence", 0) for d in defects]
                avg_conf = sum(confidences) / len(confidences)
                text.insert("end", f"\n平均置信度: {avg_conf:.4f}")
            
            text.config(state="disabled")
        
        # 添加底部按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        
        # 添加导出按钮
        ttk.Button(btn_frame, text="导出对比结果", 
                  command=lambda: self._export_comparison(results_dict)).pack(side="right", padx=5)
        
        # 添加关闭按钮
        ttk.Button(btn_frame, text="关闭", 
                  command=window.destroy).pack(side="right", padx=5)