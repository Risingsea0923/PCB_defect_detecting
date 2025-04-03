# 在文件顶部添加调试信息
import sys
import os
print(f"Python路径: {sys.executable}")
print(f"模块搜索路径:")
for p in sys.path:
    print(f"  {p}")

# 检查ultralytics是否在site-packages目录中
site_packages = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages')
print(f"检查site-packages目录: {site_packages}")
if os.path.exists(site_packages):
    print("site-packages目录存在")
    # 列出site-packages目录中的内容
    print("site-packages目录内容:")
    for item in os.listdir(site_packages):
        if "ultra" in item.lower():
            print(f"  找到可能的ultralytics相关项: {item}")

# 尝试导入ultralytics并打印版本
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"成功导入ultralytics，版本: {ultralytics.__version__}")
except ImportError as e:
    print(f"导入ultralytics失败: {e}")
    # 尝试手动添加site-packages路径
    if os.path.exists(site_packages):
        print(f"尝试添加路径: {site_packages}")
        sys.path.append(site_packages)
        try:
            from ultralytics import YOLO
            import ultralytics
            print(f"第二次尝试成功，版本: {ultralytics.__version__}")
        except ImportError as e2:
            print(f"第二次尝试仍然失败: {e2}")
            
            # 尝试使用pip检查ultralytics是否已安装
            import subprocess
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
                print("已安装的包列表:")
                for line in result.stdout.split('\n'):
                    if "ultra" in line.lower():
                        print(f"  {line}")
            except Exception as e3:
                print(f"无法获取已安装的包列表: {e3}")

# 在文件顶部添加 ttk 导入
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import os
import numpy as np
from agent.yolo_agent.agent import YOLOAgent, YOLOAgentGUI  # 添加导入
from PIL import Image, ImageTk
import glob
import json
import pandas as pd
import sys  
import shutil  # 用于查找可执行文件路径
import threading
import time
from ultralytics import YOLO  # 添加YOLO导入
from PIL import Image, ImageDraw, ImageFont
import image_processing as ip
import image_registration as ir
from detector import PCBDetector
from parallel_processor import ParallelProcessor, VisualizationComparer
from statistics_analyzer import StatisticsAnalyzer
# 添加JSON序列化辅助函数
def numpy_json_encoder(obj):
    """处理NumPy类型的JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj




    def __init__(self):
        self.defects = []
        self.type_stats = {}
        self.confidence_stats = {}
        
    def load_single_result(self, defects):
        self.defects = defects
        self._calculate_stats()
        
    def _calculate_stats(self):
        # 计算类型统计
        self.type_stats = {}
        for defect in self.defects:
            defect_type = defect.get('type', '未知')
            self.type_stats[defect_type] = self.type_stats.get(defect_type, 0) + 1
            
        # 计算置信度统计
        confidences = [d.get('confidence', 0) for d in self.defects]
        if confidences:
            self.confidence_stats = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
    
    def compare_with(self, other):
        """
        与另一个分析器进行比较
        
        参数:
            other: 另一个StatisticsAnalyzer实例
        """
        # 比较缺陷数量
        total_diff = len(self.defects) - len(other.defects)
        
        # 比较类型分布
        all_types = set(self.type_stats.keys()) | set(other.type_stats.keys())
        type_diffs = {}
        for t in all_types:
            count1 = self.type_stats.get(t, 0)
            count2 = other.type_stats.get(t, 0)
            type_diffs[t] = count1 - count2
        
        # 比较置信度
        conf_diff = {}
        if self.confidence_stats and other.confidence_stats:
            conf_diff = {
                'mean_diff': self.confidence_stats['mean'] - other.confidence_stats['mean'],
                'std_diff': self.confidence_stats['std'] - other.confidence_stats['std']
            }
        
        return {
            'total_diff': total_diff,
            'type_diffs': type_diffs,
            'confidence_diffs': conf_diff
        }
###########################################
# ZoomableCanvas: 支持缩放/拖拽
###########################################
class ZoomableCanvas(tk.Canvas):
    def __init__(self, parent, width=400, height=300, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.image_id = None
        self.original_img = None
        self.tk_img = None
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        
        # 添加标注相关的属性初始化
        self.annotation_start = None
        self.current_annotation = None

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<MouseWheel>", self.on_wheel)  # Windows
        self.bind("<Button-4>", self.on_wheel)    # Linux上滚
        self.bind("<Button-5>", self.on_wheel)    # Linux下滚


    def set_image(self, img):
        """img: 灰度或BGR numpy数组(或RGB)"""
        if img is None:
            return
        if len(img.shape) == 2:
            # 如果输入是灰度图，转为三通道显示
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # 如果是彩色图，直接转RGB显示，不进行灰度化
            disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.original_img = disp
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._update()

    def _update(self):
        if self.original_img is None:
            return
        h, w = self.original_img.shape[:2]
        new_w = int(w * self.zoom_scale)
        new_h = int(h * self.zoom_scale)

        # 保证最小尺寸至少1x1
        if new_w < 1:
            new_w = 1
        if new_h < 1:
            new_h = 1

        resized = cv2.resize(self.original_img, (new_w, new_h))
        pil = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(pil)

        if self.image_id is None:
            self.image_id = self.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)
        else:
            self.itemconfig(self.image_id, image=self.tk_img)
            self.coords(self.image_id, self.offset_x, self.offset_y)

    def on_press(self, event):
        self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self.drag_start = (event.x, event.y)
        self._update()

    def on_wheel(self, event):
        if event.delta > 0 or event.num == 4:  # 滚轮上
            self.zoom_scale *= 1.1
        else:  # 滚轮下
            self.zoom_scale *= 0.9

        # 避免缩放比例过小
        if self.zoom_scale < 0.05:
            self.zoom_scale = 0.05

        self._update()

    def update_annotation(self, x, y):
        """更新标注框"""
        if self.annotation_start:
            if self.current_annotation:
                self.delete(self.current_annotation)
            
            # 计算矩形坐标
            x1, y1 = self.annotation_start
            x2, y2 = x, y
            
            # 创建矩形标注
            self.current_annotation = self.create_rectangle(
                x1, y1, x2, y2,
                outline='red',
                width=2
            )
    
    def finish_annotation(self):
        """完成标注并返回标注区域"""
        if self.annotation_start and self.current_annotation:
            coords = self.coords(self.current_annotation)
            self.annotation_start = None
            self.current_annotation = None
            return coords
        return None
    
    def start_annotation(self, x, y):
        """开始标注"""
        self.annotation_start = (x, y)
        self.current_annotation = None

class ROISelectableCanvas(ZoomableCanvas):
    def __init__(self, parent, img, main_gui, **kwargs):
        """
        img: 结果图像
        main_gui: 引用 IntegratedGUI 对象，方便回调
        """
        super().__init__(parent, **kwargs)
        self.main_gui = main_gui
        self.display_img = img

        self.roi_start = None
        self.roi_rect_id = None
        
        # 添加鼠标右键选区功能
        self.bind("<ButtonPress-3>", self.on_roi_press)
        self.bind("<B3-Motion>", self.on_roi_drag)
        self.bind("<ButtonRelease-3>", self.on_roi_release)

    def set_image(self, img):
        """重写 set_image 方法，保存 display_img"""
        self.display_img = img
        super().set_image(img)

    def on_roi_press(self, event):
        if not self.main_gui.selection_mode:
            return
        x = (event.x - self.offset_x) / self.zoom_scale
        y = (event.y - self.offset_y) / self.zoom_scale
        self.roi_start = (x, y)
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
            self.roi_rect_id = None

    def on_roi_drag(self, event):
        if not self.main_gui.selection_mode or self.roi_start is None:
            return
        # 获取相对于当前缩放和偏移的坐标
        x = (event.x - self.offset_x) / self.zoom_scale
        y = (event.y - self.offset_y) / self.zoom_scale
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
        # 创建矩形时考虑缩放和偏移
        x1 = self.roi_start[0] * self.zoom_scale + self.offset_x
        y1 = self.roi_start[1] * self.zoom_scale + self.offset_y
        x2 = event.x
        y2 = event.y
        self.roi_rect_id = self.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    def on_roi_release(self, event):
        if not self.main_gui.selection_mode or self.roi_start is None:
            return
        # 计算真实的ROI坐标
        x = (event.x - self.offset_x) / self.zoom_scale
        y = (event.y - self.offset_y) / self.zoom_scale
        x1, y1 = self.roi_start
        x2, y2 = x, y
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        
        if self.display_img is not None:
            h, w = self.display_img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
    
            # 提取ROI区域
            roi = self.display_img[y1:y2, x1:x2].copy()
            if roi.size == 0:
                return
    
            # 保存原始ROI（不转灰度）
            self.main_gui.original_roi_color = roi.copy()
            
            # 如果是彩色图，保存灰度版本用于处理
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi.copy()
    
            # 保存ROI坐标和灰度图
            self.main_gui.roi_coords = (x1, y1, x2, y2)
            self.main_gui.selected_subroi = roi.copy()  # 保存原始ROI（可能是彩色的）
            self.main_gui.selected_subroi_gray = roi_gray  # 保存灰度ROI用于处理
    
            # 在结果区域显示原始ROI（保持颜色）
            self.set_image(roi)
    
            # 清理矩形标记
            if self.roi_rect_id:
                self.delete(self.roi_rect_id)
                self.roi_rect_id = None
            self.roi_start = None
    
            # 在历史记录输出
            self.main_gui.add_history(
                f"已选择ROI: ({x1},{y1})-({x2},{y2}), size={roi.shape}"
            )
###########################################
# IntegratedGUI: 整合标准图、待测图、结果图及参数
###########################################
class IntegratedGUI:
    def __init__(self, root):

        self.root = root
        root.title("PCB缺陷检测系统")
        root.state('zoomed')  # 最大化窗口
        self.enable_processing = False  # 初始为“关闭”图像处理
        #self.root.geometry("1600x900")
        # 数据初始化
        self.history = []
        self.test_images = {}
        self.std_img = None
        self.cur_test_img = None
        self.result_img = None
        self.selection_mode = False
        self.selected_subroi = None
        self.roi_coords = None
        self.enable_processing = False
        
        # 图像预处理顺序控制
        self.order_step1 = tk.StringVar(value="滤波")
        self.order_step2 = tk.StringVar(value="锐化")
        self.order_step3 = tk.StringVar(value="均衡化")
 
        # 创建标签页控件
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # 第一个标签页 - 图像处理
        self.tab1 = tk.Frame(self.notebook)
        self.notebook.add(self.tab1, text='图像处理与缺陷检测')
        # 先创建左侧面板（包含历史记录列表）
        self.build_left_panel()
        
        # 初始化时获取images_0中的图像列表并记录
        self.initial_images = self.get_images_list()
        self.add_history(f"程序启动: 检测到 {len(self.initial_images)} 张原始图像")
        
        # 输出图像列表详情
        if self.initial_images:
            self.add_history("原始图像列表:")
            for i, img_name in enumerate(sorted(self.initial_images)):
                if i < 10:  # 只显示前10张，避免列表过长
                    self.add_history(f"  {i+1}. {img_name}")
                elif i == 10:
                    self.add_history(f"  ... 以及其他 {len(self.initial_images)-10} 张图像")
                    break
        else:
            self.add_history("原始图像目录为空")
        # 初始化文件记录字典，用于跟踪各目录下的初始文件
        self.initial_files = {
            "images_0": [],
            "images": [],
            "images/train": [],
            "images/val": [],
            "json": [],
            "labels": [],
            "labels/train": [],
            "labels/val": []
        }
        
        # 记录初始文件列表
        self.record_initial_files()

        # 第二个标签页 - 视频检测与标注
        self.tab2 = tk.Frame(self.notebook)
        self.notebook.add(self.tab2, text='视频检测与标注')
        # 第三个标签页 - YOLO Agent
        self.tab3 = tk.Frame(self.notebook)
        self.notebook.add(self.tab3, text='自动检测与模型训练')
        # 设置第一个标签页的布局
        self.setup_tab1()
        
        # 设置第二个标签页的布局
        self.setup_tab2()
        # 设置第三个标签页的布局
        self.setup_tab3()

        # 是否处于选区模式
        self.selection_mode = False
        # 选区的坐标与选区子图
        self.roi_coords = None
        self.selected_subroi = None
        self.model_path_var = tk.StringVar()
    def setup_tab1(self):
        """设置第一个标签页 - 图像处理"""
        # 设置tab1的网格布局
        self.tab1.rowconfigure(1, weight=1)
        self.tab1.rowconfigure(2, weight=1)
        self.tab1.columnconfigure(0, weight=0)
        self.tab1.columnconfigure(1, weight=1)
        self.tab1.columnconfigure(2, weight=0)
        self.tab1.columnconfigure(3, weight=0)
        self.tab1.columnconfigure(4, weight=0)
        
        # 在tab1中构建UI
        self.build_top_bar()
       #self.build_left_panel()
        self.build_center_canvases()
        self.build_right_panel()
        self.build_extra_panel()
        self.build_right2_panel()
            
    def setup_tab2(self):
        """设置第二个标签页 - 视频检测与标注"""
        # 创建视频检测区域
        self.video_frame = tk.Frame(self.tab2)
        self.video_frame.pack(side="left", fill="both", expand=True)
        
        # 创建视频显示区域
        self.video_display = tk.Canvas(self.video_frame, bg="black")
        self.video_display.pack( fill="both", expand=True, padx=10, pady=10)
        
        # 创建控制面板
        self.video_control_frame = tk.Frame(self.tab2, width=300)
        self.video_control_frame.pack(side="right", fill="y", padx=10, pady=10)
        
        # 视频源选择
        tk.Label(self.video_control_frame, text="视频源:").pack(pady=5)
        self.video_source_var = tk.StringVar(value="0")
        tk.Entry(self.video_control_frame, textvariable=self.video_source_var).pack(fill="x", padx=5)
        
        # 模型路径
        tk.Label(self.video_control_frame, text="模型路径:").pack(pady=5)
        self.video_model_path_var = tk.StringVar(value="d:/030923/v9/yolov9-42-pcb/42_demo/best.pt")
        tk.Entry(self.video_control_frame, textvariable=self.video_model_path_var).pack(fill="x", padx=5)
        tk.Button(self.video_control_frame, text="浏览", command=self.browse_video_model_path).pack(pady=2)
        
        # 置信度阈值
        tk.Label(self.video_control_frame, text="置信度阈值:").pack(pady=5)
        self.video_conf_threshold_var = tk.DoubleVar(value=0.25)
        tk.Scale(self.video_control_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                variable=self.video_conf_threshold_var).pack(fill="x", padx=5)
        
        # 视频检测按钮
        tk.Button(self.video_control_frame, text="开始视频检测", 
                command=self.start_video_detection).pack(pady=5, fill="x")
        tk.Button(self.video_control_frame, text="停止视频检测", 
                command=self.stop_video_detection).pack(pady=5, fill="x")
        
        # 添加保存视频帧按钮
        tk.Button(self.video_control_frame, text="保存当前帧", 
                command=self.save_video_frame).pack(pady=5, fill="x")
        
        # 标注功能
        tk.Label(self.video_control_frame, text="人工标注", font=("Arial", 10, "bold")).pack(pady=10)
        
        # 标注图像路径
        tk.Label(self.video_control_frame, text="图像文件夹:").pack(pady=5)
        self.annotation_folder_var = tk.StringVar()
        tk.Entry(self.video_control_frame, textvariable=self.annotation_folder_var).pack(fill="x", padx=5)
        tk.Button(self.video_control_frame, text="浏览", 
                command=self.browse_annotation_folder).pack(pady=2)
        
        # 启动标注工具按钮
        tk.Button(self.video_control_frame, text="启动Labelme标注工具", 
                command=self.start_labelme).pack(pady=5, fill="x")


    def setup_tab3(self):
        """设置第三个标签页 - YOLO Agent"""
        try:
            # 导入必要的类
            from agent.yolo_agent.agent import YOLOAgent, YOLOAgentGUI
            
            # 首先创建YOLOAgent实例
            default_model_path = "d:/030923/Model/best.pt"
            default_data_yaml = "d:/030923/dataset/dataset.yaml"
            
            # 检查默认模型文件是否存在
            if not os.path.exists(default_model_path):
                self.add_history(f"警告：默认模型文件不存在: {default_model_path}")
                default_model_path = ""
            
            # 创建YOLOAgent实例
            yolo_agent = YOLOAgent(model_path=default_model_path, data_yaml=default_data_yaml)
            
            # 使用master参数而不是root
            self.yolo_agent_gui = YOLOAgentGUI(agent=yolo_agent, master=self.tab3)
            
            # 设置回调函数，用于将日志信息添加到主界面的历史记录中
            def log_callback(message):
                self.add_history(message)
            
            # 设置日志回调
            if hasattr(self.yolo_agent_gui, 'set_log_callback'):
                self.yolo_agent_gui.set_log_callback(log_callback)
            
            # 初始化GUI
            if hasattr(self.yolo_agent_gui, 'setup_gui'):
                self.yolo_agent_gui.setup_gui()
            
            self.add_history("YOLO Agent 标签页已加载")
            
        except Exception as e:
            messagebox.showerror("错误", f"初始化YOLO Agent标签页失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")



    def browse_annotation_folder(self):
        """浏览选择标注图像文件夹"""
        folder_path = filedialog.askdirectory(title="选择标注图像文件夹")
        if folder_path:
            self.annotation_folder_var.set(folder_path)
            self.add_history(f"已选择标注图像文件夹: {folder_path}")

    def start_labelme(self):
        """启动Labelme标注工具"""
        try:
            folder_path = self.annotation_folder_var.get()
            if not folder_path:
                messagebox.showwarning("提示", "请先选择标注图像文件夹")
                return
                
            if not os.path.exists(folder_path):
                messagebox.showerror("错误", f"文件夹不存在: {folder_path}")
                return
            
            # 尝试使用Python模块方式启动labelme
            self.add_history("尝试使用Python模块方式启动labelme...")
            
            # 获取当前Python解释器路径
            python_exe = sys.executable
            
            # 构建命令
            cmd = f'"{python_exe}" -m labelme "{folder_path}"'
            self.add_history(f"执行命令: {cmd}")
            
            # 使用subprocess执行命令
            import subprocess
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 检查是否启动成功
            returncode = process.poll()
            if returncode is not None and returncode != 0:
                # 获取错误输出
                _, stderr = process.communicate()
                raise Exception(f"启动失败，返回码: {returncode}, 错误: {stderr}")
            
            self.add_history("labelme启动命令已执行")
            
        except Exception as e:
            self.add_history(f"启动labelme失败: {str(e)}")
            
            # 提供安装指导
            install_msg = (
                "labelme可能未安装，请尝试以下步骤:\n"
                f"1. 打开命令提示符\n"
                f"2. 执行命令: {sys.executable} -m pip install labelme\n"
                f"3. 安装完成后重试"
            )
            messagebox.showerror("错误", f"启动Labelme失败: {str(e)}\n\n{install_msg}")
            
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def save_video_frame(self):
        """保存当前视频帧"""
        if not hasattr(self, 'tk_img') or self.tk_img is None:
            messagebox.showwarning("提示", "没有可保存的视频帧")
            return
            
        filename = filedialog.asksaveasfilename(
            title="保存视频帧",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not filename:
            return
            
        try:
            # 获取当前帧的numpy数组
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                cv2.imwrite(filename, self.current_frame)
                self.add_history(f"已保存视频帧到: {filename}")
            else:
                messagebox.showwarning("提示", "当前没有可用的视频帧数据")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")


    # 添加一个JSON序列化辅助函数
    def numpy_json_encoder(obj):
        """处理NumPy类型的JSON序列化"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def browse_video_model_path(self):
        """浏览选择视频检测模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.video_model_path_var.set(file_path)
            self.add_history(f"已选择视频检测模型文件: {file_path}")

    def start_video_detection(self):
        """开始视频检测"""
        try:
            # 获取视频源
            source = self.video_source_var.get()
            if source.isdigit():
                source = int(source)
            
            # 创建检测器实例
            self.detector = PCBDetector()
            
            # 加载模型
            model_path = self.video_model_path_var.get()
            if not os.path.exists(model_path):
                messagebox.showerror("错误", f"模型文件不存在: {model_path}")
                return
                
            if not self.detector.load_model(model_path):
                messagebox.showerror("错误", "模型加载失败")
                return
            
            # 打开视频源
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开视频源")
                return
            
            # 创建停止事件
            self.stopEvent = threading.Event()
            self.stopEvent.clear()
            
            # 创建并启动视频检测线程
            self.video_thread = threading.Thread(target=self.video_detection_thread)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            self.add_history("视频检测已启动")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动视频检测失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def video_detection_thread(self):
        """视频检测线程"""
        try:
            while not self.stopEvent.is_set():
                # 读取一帧
                ret, frame = self.cap.read()
                if not ret:
                    self.add_history("视频读取结束或出错")
                    break
                
                # 执行检测
                result = self.detector.detect_video_frame(frame)
                if result['success']:
                    # 保存当前帧供后续使用
                    self.current_frame = result['original']
                    
                    # 转换为PIL图像以在Tkinter中显示
                    pil_img = Image.fromarray(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
                    
                    # 调整大小以适应画布
                    canvas_w = self.video_display.winfo_width()
                    canvas_h = self.video_display.winfo_height()
                    
                    if canvas_w > 0 and canvas_h > 0:
                        # 计算缩放比例
                        h, w = result['image'].shape[:2]
                        scale = min(canvas_w / w, canvas_h / h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        
                        # 调整大小
                        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                    
                    # 转换为Tkinter可用的图像
                    self.tk_img = ImageTk.PhotoImage(pil_img)
                    
                    # 在主线程中更新UI
                    self.root.after(1, self.update_video_frame, result['info'])
                
                # 添加延迟以控制帧率
                time.sleep(0.03)  # 约30fps
                
        except Exception as e:
            self.add_history(f"视频处理出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def stop_video_detection(self):
        """停止视频检测"""
        try:
            if hasattr(self, 'detector'):
                # 停止视频检测
                if hasattr(self, 'stopEvent'):
                    self.stopEvent.set()
                
                # 释放视频资源
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
                
                # 清空画布
                self.video_display.delete("all")
                
                self.add_history("视频检测已停止")
                
        except Exception as e:
            messagebox.showerror("错误", f"停止视频检测失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def update_video_frame(self, defect_info):
        """在主线程中更新视频帧"""
        try:
            # 清除画布
            self.video_display.delete("all")
            
            # 显示图像
            self.video_display.create_image(
                self.video_display.winfo_width() // 2,
                self.video_display.winfo_height() // 2,
                image=self.tk_img, anchor="center"
            )
            
            # 更新缺陷信息
            if defect_info:
                info_text = f"检测到 {len(defect_info)} 个缺陷"
                self.video_display.create_text(10, 10, text=info_text, fill="red", anchor="nw")
                
                # 在状态栏显示详细信息
                details = []
                for i, defect in enumerate(defect_info):
                    details.append(f"{i+1}. {defect['type']}({defect['confidence']:.2f})")
                
                detail_text = ", ".join(details)
                if len(detail_text) > 50:
                    detail_text = detail_text[:47] + "..."
                    
                self.video_display.create_text(
                    10, 30, text=detail_text, 
                    fill="yellow", anchor="nw", font=("Arial", 9)
                )
        except Exception as e:
            self.add_history(f"更新视频帧出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def get_images_list(self):
        """获取images_0目录中的图像列表"""
        try:
            import os
            # 获取并输出当前工作目录
            current_dir = os.getcwd()
            self.add_history(f"当前工作目录: {current_dir}")
            
            # 使用相对路径
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
            self.add_history(f"数据集目录: {base_dir}")
            input_dir = os.path.join(base_dir, "images_0")    


            #base_dir = "d:/030923/dataset"
            #input_dir = os.path.join(base_dir, "images_0")
            
            self.add_history(f"检查图像目录: {input_dir}")
            
            if not os.path.exists(input_dir):
                os.makedirs(input_dir, exist_ok=True)
                self.add_history(f"创建图像目录: {input_dir}")
                return []
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 检查JSON目录
            json_dir = os.path.join(base_dir, "json")
            if os.path.exists(json_dir):
                json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
                self.add_history(f"JSON标注目录: {json_dir}，包含 {len(json_files)} 个标注文件")
                
                # 检查图像和JSON文件的匹配情况
                matched_count = 0
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    json_file = base_name + ".json"
                    if json_file in json_files:
                        matched_count += 1
                
                self.add_history(f"图像与JSON匹配: {matched_count}/{len(image_files)}")
            else:
                self.add_history(f"JSON标注目录不存在: {json_dir}")
                
            return image_files
            
        except Exception as e:
            import traceback
            self.add_history(f"获取图像列表出错: {str(e)}")
            self.add_history(traceback.format_exc())
            return []

    def augment_dataset(self):
        """调用img_en.py进行数据增强，只处理新增的图像"""
        try:
            # 默认数据集基础目录
            default_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
            self.add_history(f"数据集基础目录: {default_base_dir}")
            # 备选路径（注释掉）
            # default_base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')
            # 检查默认目录是否存在
            # 默认数据集基础目录
            # default_base_dir = "D:/030923/dataset"
            #default_base_dir = "D:/030923/data/train" #一边采集一边检测的工作流            
            # 检查默认目录是否存在
            if not os.path.exists(default_base_dir):
                os.makedirs(default_base_dir, exist_ok=True)
                self.add_history(f"创建默认数据集目录: {default_base_dir}")
            
            # 检查images_0目录是否存在
            default_image_dir = os.path.join(default_base_dir, "images_0")
            if not os.path.exists(default_image_dir):
                os.makedirs(default_image_dir, exist_ok=True)
                self.add_history(f"创建原始图像目录: {default_image_dir}")


            if not os.path.exists(default_base_dir):
                os.makedirs(default_base_dir, exist_ok=True)
                self.add_history(f"创建默认数据集目录: {default_base_dir}")
            
            # 检查images_0目录是否存在
            default_image_dir = os.path.join(default_base_dir, "images_0")
            if not os.path.exists(default_image_dir):
                os.makedirs(default_image_dir, exist_ok=True)
                self.add_history(f"创建原始图像目录: {default_image_dir}")
            
            # 检查json目录是否存在
            default_json_dir = os.path.join(default_base_dir, "json")
            if not os.path.exists(default_json_dir):
                os.makedirs(default_json_dir, exist_ok=True)
                self.add_history(f"创建JSON标注目录: {default_json_dir}")
            
            # 获取当前images_0目录中的图像列表（转换为小写进行比较）
            current_images = set()
            self.add_history(f"尝试读取目录: {default_image_dir}")
            if os.path.exists(default_image_dir):
                files = os.listdir(default_image_dir)
                self.add_history(f"目录中的所有文件: {files}")
                current_images = set([f.lower() for f in files 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                self.add_history(f"筛选后的图像文件: {current_images}")
            else:
                self.add_history(f"警告：目录不存在: {default_image_dir}")
            
            self.add_history(f"当前原始图像目录中有 {len(current_images)} 张图像")
            
            # 确保initial_images属性存在
            if not hasattr(self, 'initial_images'):
                self.initial_images = []
            
            # 将initial_images转换为小写的集合
            initial_images_set = set([f.lower() for f in self.initial_images])
            
            # 计算新增的图像（当前图像减去初始图像）
            new_images = current_images - initial_images_set
            new_images_list = list(new_images)
            
            # 调试输出
            self.add_history(f"程序启动时记录的图像数量: {len(initial_images_set)} 张")
            self.add_history(f"检测到的新增图像数量: {len(new_images)} 张")
            
            if new_images:
                self.add_history("新增图像列表:")
                for i, img in enumerate(sorted(new_images)):
                    if i < 10:  # 只显示前10张
                        self.add_history(f"  {i+1}. {img}")
                    elif i == 10:
                        self.add_history(f"  ... 以及其他 {len(new_images)-10} 张图像")
                        break
            
            # 询问用户是否使用默认目录
            use_default = messagebox.askyesno("确认", 
                f"是否使用默认数据集目录?\n{default_base_dir}\n\n"
                f"当前包含 {len(current_images)} 张原始图像\n"
                f"其中 {len(new_images)} 张为新增图像")

            if use_default:
                base_dir = default_base_dir
            else:
                # 选择数据集基础目录
                base_dir = filedialog.askdirectory(title="选择数据集基础目录", 
                                                initialdir=default_base_dir)
                if not base_dir:
                    return
            
            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("数据增强进度")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # 添加进度信息标签
            info_label = tk.Label(progress_window, text="正在准备数据增强...")
            info_label.pack(pady=10)
            
            # 添加图像计数标签
            count_label = tk.Label(progress_window, text="")
            count_label.pack(pady=5)
            
            # 添加进度条
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
            progress_bar.pack(pady=10, padx=20)
            
            # 更新进度窗口
            progress_window.update()
            
            # 检查必要的目录结构
            json_dir = os.path.join(base_dir, "json")
            image_dir = os.path.join(base_dir, "images_0")
            output_image_dir = os.path.join(base_dir, "images")
            output_label_dir = os.path.join(base_dir, "labels")
            
            if not os.path.exists(json_dir):
                os.makedirs(json_dir, exist_ok=True)
                self.add_history(f"创建JSON目录: {json_dir}")
            
            if not os.path.exists(image_dir):
                os.makedirs(image_dir, exist_ok=True)
                self.add_history(f"创建原始图像目录: {image_dir}")
            
            # 更新进度信息
            count_label.config(text=f"发现 {len(current_images)} 张图像，其中 {len(new_images)} 张为新增")
            
            if len(new_images) == 0:
                info_label.config(text="没有发现新增图像，无需处理")
                progress_var.set(100)
                self.add_history("没有发现新增图像，无需进行数据增强")
                messagebox.showinfo("提示", "没有发现新增图像，无需进行数据增强")
                progress_window.after(2000, progress_window.destroy)
                return
            
            info_label.config(text=f"正在对 {len(new_images)} 张新增图像执行数据增强...")
            progress_window.update()
            
            # 创建一个线程来执行数据增强
            def run_augmentation():
                try:
                    # 导入img_en模块
                    import importlib.util
                    #spec = importlib.util.spec_from_file_location("img_en", "d:/030923/dataset/img_en.py")
                    # 使用相对路径导入img_en模块
                    img_en_path = os.path.join(os.path.dirname(__file__), 'dataset', 'img_en.py')
                    spec = importlib.util.spec_from_file_location("img_en", img_en_path)
                    img_en = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(img_en)
                    
                    # 修改img_en中的路径设置
                    img_en.base = base_dir
                    
                    # 创建临时目录来存放新增图像
                    temp_image_dir = os.path.join(base_dir, "temp_images_0")
                    temp_json_dir = os.path.join(base_dir, "temp_json")
                    os.makedirs(temp_image_dir, exist_ok=True)
                    os.makedirs(temp_json_dir, exist_ok=True)
                    
                    # 复制新增图像及其JSON文件到临时目录
                    for img_file in new_images_list:
                        src_img = os.path.join(image_dir, img_file)
                        dst_img = os.path.join(temp_image_dir, img_file)
                        shutil.copy2(src_img, dst_img)
                        
                        # 复制对应的JSON文件
                        base_name = os.path.splitext(img_file)[0]
                        json_file = base_name + ".json"
                        src_json = os.path.join(json_dir, json_file)
                        if os.path.exists(src_json):
                            dst_json = os.path.join(temp_json_dir, json_file)
                            shutil.copy2(src_json, dst_json)
                    
                    # 修改img_en的路径设置为临时目录
                    original_base = img_en.base
                    img_en.base = base_dir
                    
                    # 保存原始目录路径
                    original_image_dir = os.path.join(base_dir, "images_0")
                    original_json_dir = os.path.join(base_dir, "json")
                    
                    # 临时修改目录结构，使img_en只处理新增图像
                    os.rename(original_image_dir, original_image_dir + "_backup")
                    os.rename(original_json_dir, original_json_dir + "_backup")
                    os.rename(temp_image_dir, original_image_dir)
                    os.rename(temp_json_dir, original_json_dir)
                    
                    try:
                        # 调用img_en.py中的main函数
                        img_en.main()
                    finally:
                        # 恢复原始目录结构
                        os.rename(original_image_dir, temp_image_dir)
                        os.rename(original_json_dir, temp_json_dir)
                        os.rename(original_image_dir + "_backup", original_image_dir)
                        os.rename(original_json_dir + "_backup", original_json_dir)
                        
                        # 清理临时目录
                        shutil.rmtree(temp_image_dir, ignore_errors=True)
                        shutil.rmtree(temp_json_dir, ignore_errors=True)
                    
                    # 完成后更新UI
                    self.root.after(0, lambda: info_label.config(text="数据增强完成!"))
                    self.root.after(0, lambda: progress_var.set(100))
                    self.root.after(2000, progress_window.destroy)
                    
                    # 获取生成的图像数量
                    generated_images_count = 0
                    if os.path.exists(output_image_dir):
                        generated_images_count = len([f for f in os.listdir(output_image_dir) 
                                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    # 添加到历史记录
                    self.add_history(f"数据增强完成，共处理 {len(new_images)} 张新增图像")
                    self.add_history(f"生成了新的增强图像，保存在 {output_image_dir}")
                    self.add_history(f"对应的YOLO格式标注保存在 {output_label_dir}")
                    
                    # 显示成功消息
                    messagebox.showinfo("成功", 
                        f"数据增强完成!\n\n"
                        f"处理了 {len(new_images)} 张新增图像\n"
                        f"结果保存在:\n{output_image_dir}")
                    
                    # 更新initial_images，将新增的图像添加到记录中
                    self.initial_images.extend(new_images_list)
                    self.add_history(f"已更新程序记录的图像列表，现在包含 {len(self.initial_images)} 张图像")
                    
                except Exception as e:
                    self.root.after(0, lambda: progress_window.destroy())
                    self.add_history(f"数据增强失败: {str(e)}")
                    messagebox.showerror("错误", f"数据增强失败: {str(e)}")
                    import traceback
                    self.add_history(f"错误详情: {traceback.format_exc()}")
            
            # 启动线程
            threading.Thread(target=run_augmentation, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("错误", f"启动数据增强失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def record_initial_files(self):
        """记录程序启动时各目录下的文件列表"""
        try:
            #base_dir = "D:/030923/dataset"
            # 使用相对路径
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
            self.add_history(f"数据集基础目录: {base_dir}")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
                self.add_history(f"创建默认数据集目录: {base_dir}")
                return
            
            # 检查并记录各目录下的文件
            for dir_key in self.initial_files.keys():
                dir_path = os.path.join(base_dir, dir_key)
                if os.path.exists(dir_path):
                    if "images" in dir_key:
                        self.initial_files[dir_key] = [
                            f.lower() for f in os.listdir(dir_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        ]
                    elif "json" in dir_key:
                        self.initial_files[dir_key] = [
                            f.lower() for f in os.listdir(dir_path) 
                            if f.lower().endswith('.json')
                        ]
                    elif "labels" in dir_key:
                        self.initial_files[dir_key] = [
                            f.lower() for f in os.listdir(dir_path) 
                            if f.lower().endswith('.txt')
                        ]
                    else:
                        self.initial_files[dir_key] = [f.lower() for f in os.listdir(dir_path)]
                    
                    self.add_history(f"记录初始文件: {dir_key} 目录下有 {len(self.initial_files[dir_key])} 个文件")
            
        except Exception as e:
            self.add_history(f"记录初始文件失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")


    def delete_dataset(self):
        """删除指定目录的图像数据与标注(json与txt)"""
        try:
            # 默认数据集基础目录
            #default_base_dir = "D:/030923/dataset"
            #default_base_dir = "D:/030923/data/train"
            # 默认数据集基础目录
            default_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
            self.add_history(f"数据集基础目录: {default_base_dir}")
            
            # 检查默认目录是否存在
            if not os.path.exists(default_base_dir):
                messagebox.showwarning("提示", f"默认数据集目录不存在: {default_base_dir}")
                return
            
            
            # 询问用户是否使用默认目录
            use_default = messagebox.askyesno("确认", 
                f"是否使用默认数据集目录?\n{default_base_dir}")

            if use_default:
                base_dir = default_base_dir
            else:
                # 选择数据集基础目录
                base_dir = filedialog.askdirectory(title="选择数据集基础目录", 
                                                initialdir=default_base_dir)
                if not base_dir:
                    return
            
            # 定义可能的数据目录
            image_dirs = {
                "原始图像": os.path.join(base_dir, "images_0"),
                "增强图像": os.path.join(base_dir, "images"),
                "JSON标注": os.path.join(base_dir, "json"),
                "YOLO标注": os.path.join(base_dir, "labels")
            }
            
            # 检查训练/验证集目录
            train_dir = os.path.join(base_dir, "images", "train")
            val_dir = os.path.join(base_dir, "images", "val")
            train_label_dir = os.path.join(base_dir, "labels", "train")
            val_label_dir = os.path.join(base_dir, "labels", "val")
            
            if os.path.exists(train_dir):
                image_dirs["训练集图像"] = train_dir
            if os.path.exists(val_dir):
                image_dirs["验证集图像"] = val_dir
            if os.path.exists(train_label_dir):
                image_dirs["训练集标注"] = train_label_dir
            if os.path.exists(val_label_dir):
                image_dirs["验证集标注"] = val_label_dir
            
            # 检查哪些目录存在并获取当前文件列表
            existing_dirs = {}
            current_files = {}
            
            for name, path in image_dirs.items():
                if os.path.exists(path):
                    # 确定目录类型和对应的初始文件列表键
                    if name == "原始图像":
                        dir_key = "images_0"
                    elif name == "增强图像":
                        dir_key = "images"
                    elif name == "训练集图像":
                        dir_key = "images/train"
                    elif name == "验证集图像":
                        dir_key = "images/val"
                    elif name == "JSON标注":
                        dir_key = "json"
                    elif name == "YOLO标注":
                        dir_key = "labels"
                    elif name == "训练集标注":
                        dir_key = "labels/train"
                    elif name == "验证集标注":
                        dir_key = "labels/val"
                    else:
                        dir_key = None
                    
                    # 获取当前文件列表
                    if name in ["原始图像", "增强图像", "训练集图像", "验证集图像"]:
                        current_files[name] = [
                            f.lower() for f in os.listdir(path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                        ]
                    elif name == "JSON标注":
                        current_files[name] = [
                            f.lower() for f in os.listdir(path) 
                            if f.lower().endswith('.json')
                        ]
                    elif name in ["YOLO标注", "训练集标注", "验证集标注"]:
                        current_files[name] = [
                            f.lower() for f in os.listdir(path) 
                            if f.lower().endswith('.txt')
                        ]
                    else:
                        current_files[name] = [f.lower() for f in os.listdir(path)]
                    
                    # 计算新增文件
                    new_files_count = 0
                    if dir_key and dir_key in self.initial_files:
                        initial_set = set(self.initial_files[dir_key])
                        current_set = set(current_files[name])
                        new_files = current_set - initial_set
                        new_files_count = len(new_files)
                    
                    existing_dirs[name] = (path, len(current_files[name]), new_files_count)
            
            if not existing_dirs:
                messagebox.showwarning("提示", f"在 {base_dir} 中没有找到任何数据目录")
                return
            
            # 创建选择对话框
            delete_window = tk.Toplevel(self.root)
            delete_window.title("选择要删除的数据")
            delete_window.geometry("500x500")  # 增加高度以容纳更多选项
            delete_window.transient(self.root)
            delete_window.grab_set()
            
            # 添加说明标签
            tk.Label(delete_window, text="请选择要删除的数据类型:", font=("Arial", 12)).pack(pady=10)
            
            # 创建滚动区域以容纳所有选项
            canvas = tk.Canvas(delete_window)
            scrollbar = ttk.Scrollbar(delete_window, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            scrollbar.pack(side="right", fill="y")
            
            # 创建复选框变量
            checkboxes = {}
            for name, (path, total_count, new_count) in existing_dirs.items():
                var = tk.BooleanVar(value=False)
                checkboxes[name] = var
                tk.Checkbutton(scrollable_frame, text=f"{name} (共{total_count}个文件, 新增{new_count}个)", 
                            variable=var, font=("Arial", 10)).pack(anchor="w", padx=20, pady=5)
            
            # 添加删除模式选择
            tk.Label(scrollable_frame, text="删除模式:", font=("Arial", 12)).pack(pady=10)
            delete_mode = tk.StringVar(value="all")
            tk.Radiobutton(scrollable_frame, text="删除所有文件", variable=delete_mode, 
                        value="all", font=("Arial", 10)).pack(anchor="w", padx=20, pady=2)
            tk.Radiobutton(scrollable_frame, text="仅删除新增文件", variable=delete_mode, 
                        value="new", font=("Arial", 10)).pack(anchor="w", padx=20, pady=2)
            
            # 添加文件名过滤器
            tk.Label(scrollable_frame, text="文件名过滤器 (可选，使用通配符):", font=("Arial", 10)).pack(pady=5)
            filter_var = tk.StringVar()
            tk.Entry(scrollable_frame, textvariable=filter_var, width=40).pack(pady=5)
            tk.Label(scrollable_frame, text="例如: *.jpg 或 image_*.png", font=("Arial", 8)).pack()
            
            # 添加单测生成选项
            tk.Label(scrollable_frame, text="单测生成:", font=("Arial", 12)).pack(pady=10)
            generate_test = tk.BooleanVar(value=False)
            tk.Checkbutton(scrollable_frame, text="生成单测数据", 
                        variable=generate_test, font=("Arial", 10)).pack(anchor="w", padx=20, pady=5)
            
            # 单测比例设置
            tk.Label(scrollable_frame, text="训练集比例 (%):", font=("Arial", 10)).pack(pady=2)
            train_ratio = tk.IntVar(value=80)
            tk.Scale(scrollable_frame, from_=50, to=95, orient="horizontal",
                    variable=train_ratio).pack(fill="x", padx=20)
            
            # 添加按钮
            button_frame = tk.Frame(scrollable_frame)
            button_frame.pack(pady=20)
            
            def on_confirm():
                # 检查是否选择了生成单测
                if generate_test.get():
                    # 执行单测生成
                    self.generate_train_val_split(train_ratio.get() / 100.0)
                    return
                
                # 获取选中的目录
                selected_dirs = {name: (path, total, new) for name, (path, total, new) in existing_dirs.items() 
                                if checkboxes[name].get()}
                
                if not selected_dirs:
                    messagebox.showwarning("提示", "请至少选择一个数据类型")
                    return
                
                # 确认删除
                mode_text = "新增" if delete_mode.get() == "new" else "所有"
                filter_text = f"匹配 '{filter_var.get()}' 的" if filter_var.get() else ""
                dirs_text = ", ".join(selected_dirs.keys())
                
                confirm_msg = f"确定要删除以下目录中的{mode_text}{filter_text}文件吗?\n\n{dirs_text}\n\n此操作不可恢复!"
                if not messagebox.askyesno("确认删除", confirm_msg, icon="warning"):
                    return
                
                # 执行删除
                try:
                    total_deleted = 0
                    
                    # 处理每个选中的目录
                    for name, (path, _, _) in selected_dirs.items():
                        # 确定目录类型和对应的初始文件列表键
                        if name == "原始图像":
                            dir_key = "images_0"
                        elif name == "增强图像":
                            dir_key = "images"
                        elif name == "训练集图像":
                            dir_key = "images/train"
                        elif name == "验证集图像":
                            dir_key = "images/val"
                        elif name == "JSON标注":
                            dir_key = "json"
                        elif name == "YOLO标注":
                            dir_key = "labels"
                        elif name == "训练集标注":
                            dir_key = "labels/train"
                        elif name == "验证集标注":
                            dir_key = "labels/val"
                        else:
                            dir_key = None
                        
                        # 获取目录中的所有文件
                        all_files = os.listdir(path)
                        
                        # 应用过滤器
                        if filter_var.get():
                            import fnmatch
                            filtered_files = fnmatch.filter(all_files, filter_var.get())
                        else:
                            filtered_files = all_files
                        
                        # 应用新增文件过滤
                        if delete_mode.get() == "new" and dir_key in self.initial_files:
                            # 只保留不在initial_files中的文件
                            initial_set = set([f.lower() for f in self.initial_files[dir_key]])
                            filtered_files = [f for f in filtered_files if f.lower() not in initial_set]
                        
                        # 删除文件
                        for file in filtered_files:
                            file_path = os.path.join(path, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                total_deleted += 1
                                
                                # 如果删除的是训练/验证集图像，同时删除对应的标注文件
                                if name == "训练集图像" and "训练集标注" in selected_dirs:
                                    base_name = os.path.splitext(file)[0]
                                    label_path = os.path.join(selected_dirs["训练集标注"][0], f"{base_name}.txt")
                                    if os.path.exists(label_path):
                                        os.remove(label_path)
                                        total_deleted += 1
                                
                                elif name == "验证集图像" and "验证集标注" in selected_dirs:
                                    base_name = os.path.splitext(file)[0]
                                    label_path = os.path.join(selected_dirs["验证集标注"][0], f"{base_name}.txt")
                                    if os.path.exists(label_path):
                                        os.remove(label_path)
                                        total_deleted += 1
                    
                    # 更新历史记录
                    self.add_history(f"已删除 {total_deleted} 个文件")
                    
                    # 如果删除了原始图像，更新initial_files
                    if delete_mode.get() == "all":
                        for name, (_, _, _) in selected_dirs.items():
                            if name == "原始图像":
                                self.initial_files["images_0"] = []
                            elif name == "增强图像":
                                self.initial_files["images"] = []
                            elif name == "训练集图像":
                                self.initial_files["images/train"] = []
                            elif name == "验证集图像":
                                self.initial_files["images/val"] = []
                            elif name == "JSON标注":
                                self.initial_files["json"] = []
                            elif name == "YOLO标注":
                                self.initial_files["labels"] = []
                            elif name == "训练集标注":
                                self.initial_files["labels/train"] = []
                            elif name == "验证集标注":
                                self.initial_files["labels/val"] = []
                        
                        self.add_history("已清空程序记录的相应文件列表")
                    
                    messagebox.showinfo("成功", f"已成功删除 {total_deleted} 个文件")
                    delete_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("错误", f"删除文件时出错: {str(e)}")
                    import traceback
                    self.add_history(f"错误详情: {traceback.format_exc()}")
            
            def on_cancel():
                delete_window.destroy()
            
            # 添加执行按钮
            tk.Button(button_frame, text="确认删除", command=on_confirm, 
                    bg="#ff6b6b", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=10)
            tk.Button(button_frame, text="取消", command=on_cancel, 
                    font=("Arial", 10)).pack(side="left", padx=10)
            
        except Exception as e:
            messagebox.showerror("错误", f"删除数据集操作失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
    
    def generate_train_val_split(self, train_ratio=0.8):
        """生成训练集和验证集的单测数据"""
        try:
            # default_base_dir = "D:/030923/dataset"
            # 使用相对路径
            default_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))
            self.add_history(f"数据集基础目录: {default_base_dir}")
            
            # 检查默认目录是否存在
            if not os.path.exists(default_base_dir):
                messagebox.showwarning("提示", f"默认数据集目录不存在: {default_base_dir}")
                return
            
            # 询问用户是否使用默认目录
            use_default = messagebox.askyesno("确认", 
                f"是否使用默认数据集目录?\n{default_base_dir}")

            if use_default:
                base_dir = default_base_dir
            else:
                # 选择数据集基础目录
                base_dir = filedialog.askdirectory(title="选择数据集基础目录", 
                                                initialdir=default_base_dir)
                if not base_dir:
                    return
            
            # 检查必要的目录
            images_dir = os.path.join(base_dir, "images")
            labels_dir = os.path.join(base_dir, "labels")
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                messagebox.showwarning("提示", f"未找到必要的目录: {images_dir} 或 {labels_dir}")
                return
            
            # 创建训练集和验证集目录
            train_img_dir = os.path.join(images_dir, "train")
            val_img_dir = os.path.join(images_dir, "val")
            train_label_dir = os.path.join(labels_dir, "train")
            val_label_dir = os.path.join(labels_dir, "val")
            
            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(train_label_dir, exist_ok=True)
            os.makedirs(val_label_dir, exist_ok=True)
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(images_dir, f))]
            
            if not image_files:
                messagebox.showwarning("提示", f"在 {images_dir} 中没有找到图像文件")
                return
            
            # 随机打乱文件列表
            import random
            random.shuffle(image_files)
            
            # 计算训练集大小
            train_size = int(len(image_files) * train_ratio)
            
            # 分割为训练集和验证集
            train_files = image_files[:train_size]
            val_files = image_files[train_size:]
            
            # 创建进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("生成单测数据")
            progress_window.geometry("400x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # 添加进度信息标签
            info_label = tk.Label(progress_window, text="正在生成训练集和验证集...")
            info_label.pack(pady=10)
            
            # 添加图像计数标签
            count_label = tk.Label(progress_window, 
                                  text=f"总共 {len(image_files)} 张图像，训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")
            count_label.pack(pady=5)
            
            # 添加进度条
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100, length=350)
            progress_bar.pack(pady=10, padx=20)
            
            # 更新进度窗口
            progress_window.update()
            
            # 创建一个线程来执行数据分割
            def run_split():
                try:
                    total_processed = 0
                    
                    # 处理训练集
                    for i, file in enumerate(train_files):
                        # 复制图像文件
                        src_img = os.path.join(images_dir, file)
                        dst_img = os.path.join(train_img_dir, file)
                        shutil.copy2(src_img, dst_img)
                        
                        # 复制对应的标注文件
                        base_name = os.path.splitext(file)[0]
                        label_file = f"{base_name}.txt"
                        src_label = os.path.join(labels_dir, label_file)
                        if os.path.exists(src_label):
                            dst_label = os.path.join(train_label_dir, label_file)
                            shutil.copy2(src_label, dst_label)
                        
                        total_processed += 1
                        progress = (total_processed / len(image_files)) * 100
                        self.root.after(0, lambda p=progress: progress_var.set(p))
                        
                        if i % 10 == 0:
                            self.root.after(0, lambda i=i: info_label.config(
                                text=f"正在处理训练集... ({i+1}/{len(train_files)})"))
                            progress_window.update()
                    
                    # 处理验证集
                    for i, file in enumerate(val_files):
                        # 复制图像文件
                        src_img = os.path.join(images_dir, file)
                        dst_img = os.path.join(val_img_dir, file)
                        shutil.copy2(src_img, dst_img)
                        
                        # 复制对应的标注文件
                        base_name = os.path.splitext(file)[0]
                        label_file = f"{base_name}.txt"
                        src_label = os.path.join(labels_dir, label_file)
                        if os.path.exists(src_label):
                            dst_label = os.path.join(val_label_dir, label_file)
                            shutil.copy2(src_label, dst_label)
                        
                        total_processed += 1
                        progress = (total_processed / len(image_files)) * 100
                        self.root.after(0, lambda p=progress: progress_var.set(p))
                        
                        if i % 10 == 0:
                            self.root.after(0, lambda i=i: info_label.config(
                                text=f"正在处理验证集... ({i+1}/{len(val_files)})"))
                            progress_window.update()
                    
                    # 完成后更新UI
                    self.root.after(0, lambda: info_label.config(text="单测数据生成完成!"))
                    self.root.after(0, lambda: progress_var.set(100))
                    self.root.after(2000, progress_window.destroy)
                    
                    # 添加到历史记录
                    self.add_history(f"单测数据生成完成，训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")
                    self.add_history(f"训练集图像保存在: {train_img_dir}")
                    self.add_history(f"训练集标注保存在: {train_label_dir}")
                    self.add_history(f"验证集图像保存在: {val_img_dir}")
                    self.add_history(f"验证集标注保存在: {val_label_dir}")
                    
                    # 显示成功消息
                    messagebox.showinfo("成功", 
                        f"单测数据生成完成!\n\n"
                        f"训练集: {len(train_files)} 张图像\n"
                        f"验证集: {len(val_files)} 张图像\n\n"
                        f"数据已保存到相应目录")
                    
                    # 更新initial_files，记录新生成的文件
                    self.record_initial_files()
                    
                except Exception as e:
                    self.root.after(0, lambda: progress_window.destroy())
                    self.add_history(f"生成单测数据失败: {str(e)}")
                    messagebox.showerror("错误", f"生成单测数据失败: {str(e)}")
                    import traceback
                    self.add_history(f"错误详情: {traceback.format_exc()}")
            
            # 启动线程
            threading.Thread(target=run_split, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("错误", f"生成单测数据失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

            



    def on_process_roi_button_click(self):
        """处理选区按钮点击事件"""
        if self.selection_mode and self.selected_subroi is not None:
            # 设置处理请求标志
            self._processing_requested = True
            # 执行图像处理
            self.update_result_image()
            self.add_history("已处理选区")
        else:
            self.add_history("没有选区可处理")
        """处理选区按钮点击事件"""
        if self.selection_mode and self.selected_subroi is not None:
            # 设置处理请求标志
            self._processing_requested = True
            # 执行图像处理
            self.update_result_image()
            self.add_history("已处理选区")
        else:
            self.add_history("没有选区可处理")
    def merge_roi_back(self):
        if self.selected_subroi is None or self.roi_coords is None:
            self.add_history("无已编辑的ROI，不需要合并")
            if self.result_img is not None:
                self.canvas_result.update_image(self.result_img)
            return
        x1, y1, x2, y2 = self.roi_coords
        # 将选区覆盖到 result_img
        self.result_img[y1:y2, x1:x2] = self.selected_subroi
        self.add_history(f"ROI合并完成：({x1},{y1})-({x2},{y2})")
        # 恢复显示整张 result_img
        self.canvas_result.update_image(self.result_img)
        # 清空选区信息
        self.roi_coords = None
        self.selected_subroi = None

    ###########################################
    # 顶部
    ###########################################
    def build_top_bar(self):
        top_frame = tk.Frame(self.tab1)  # 修改这里，使用self.tab1而不是self.root
        top_frame.grid(row=0, column=0, columnspan=4, sticky="ew")
    
        btn_load_std = tk.Button(top_frame, text="加载标准图", command=self.load_standard)
        btn_load_std.pack(side="left", padx=5)
    
        btn_load_folder = tk.Button(top_frame, text="加载文件夹(待测图)", command=self.load_folder)
        btn_load_folder.pack(side="left", padx=5)
    
        btn_load_params_file = tk.Button(top_frame, text="加载PCB参数文件", command=self.load_pcb_parameters)
        btn_load_params_file.pack(side="left", padx=5)
    
        self.save_batch_var = tk.StringVar(value="保存...")
        save_batch_menu = tk.OptionMenu(top_frame, self.save_batch_var,
                                        "保存标准图", "保存待测图", "保存结果图",
                                        "批量处理待测图", "批量处理结果图",
                                        command=self.on_save_batch_select)
        save_batch_menu.pack(side="left", padx=10)
    
        btn_save_params = tk.Button(top_frame, text="保存参数设置", command=self.save_current_parameters)
        btn_save_params.pack(side="left", padx=5)
        btn_load_params = tk.Button(top_frame, text="加载参数设置", command=self.load_parameters)
        btn_load_params.pack(side="left", padx=5)
    
        btn_roi = tk.Button(top_frame, text="选区", command=self.toggle_selection_mode)
        btn_roi.pack(side="left", padx=5)
    
        self.btn_toggle_processing = tk.Button(top_frame, text="图像处理：关闭",
                                               command=self.on_toggle_processing)
        self.btn_toggle_processing.pack(side="left", padx=5)
    
        # 将原来的多个辅助功能按钮合并到一个下拉菜单中
        self.tools_var = tk.StringVar(value="绘制图像")
        tools_options = ["最优阈值曲线", "显示灰度直方图", "形态学效果对比", "绘制配准匹配率曲线"]
        tools_menu = tk.OptionMenu(top_frame, self.tools_var, *tools_options, command=self.on_tools_select)
        tools_menu.pack(side="left", padx=5)
        # 添加对齐方法选择
        tk.Label(top_frame, text="差分对齐方法:").pack(side="left", padx=5)
        self.align_method_var = tk.StringVar(value="透视变换")
        tk.OptionMenu(top_frame, self.align_method_var, 
                    "透视变换", "薄板样条变换", "多尺度特征匹配", "局部分块对齐").pack(side="left", padx=5)
    
    def on_tools_select(self, choice):
        if choice == "最优阈值曲线":
            self.on_optimal_threshold_curve()
        elif choice == "显示灰度直方图":
            self.on_show_histogram()
        elif choice == "形态学效果对比":
            self.on_morph_compare()
        elif choice == "绘制配准匹配率曲线":
            self.on_plot_reg_curve()
        # 重置选项为"辅助功能"
        self.tools_var.set("绘制图像")

    def load_pcb_parameters(self):
        """加载PCB参数文件"""
        file_path = filedialog.askopenfilename(title="选择PCB参数文件",
                                            filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'cp936', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    self.add_history(f"使用 {encoding} 编码成功读取文件")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                messagebox.showerror("错误", "无法解码文件，请检查文件编码")
                return
                
            # 打印列名以便调试
            self.add_history(f"CSV文件列名: {list(df.columns)}")
            
            self.pcb_params = df
            self.add_history("PCB参数文件预览：")
            # 输出前5行到历史记录
            self.add_history(df.head().to_string())
            
            # 检查必要的列是否存在
            required_cols = ['Reference', 'X', 'Y', 'Rotation', 'Width', 'Height']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.add_history(f"警告: 缺少列: {missing_cols}")
                
            # 显示参数
            self.add_history("各元器件参数：")
            for idx, row in df.iterrows():
                ref = row.get('Reference', f'组件{idx}')
                x = row.get('X', 'N/A')
                y = row.get('Y', 'N/A')
                rotation = row.get('Rotation', 'N/A')
                width = row.get('Width', 'N/A')
                height = row.get('Height', 'N/A')
                info = f"组件 {ref}: X={x}, Y={y}, Rotation={rotation}, Width={width}, Height={height}"
                self.add_history(info)
        except Exception as e:
            messagebox.showerror("错误", f"读取参数文件失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
    def toggle_selection_mode(self):
        """切换选区模式"""
        if not hasattr(self, "selection_mode"):
            self.selection_mode = False
            
        if not self.selection_mode:
            # 进入选区模式
            self.selection_mode = True
            self.add_history("进入选区模式")
            # 清除处理请求标志，确保不会自动处理
            self._processing_requested = False
            # 通知用户已进入选区模式
            messagebox.showinfo("提示", "已进入选区模式，请在结果图上拖动鼠标选择区域")
        else:
            # 退出选区模式，合并 ROI
            self.selection_mode = False
            if hasattr(self, "selected_subroi") and self.selected_subroi is not None and hasattr(self, "roi_coords") and self.roi_coords is not None:
                x1, y1, x2, y2 = self.roi_coords
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                # 如果 self.result_img 还不存在，则根据处理对象去拿
                if self.result_img is None:
                    target = self.target_var.get()
                    if target == "标准图":
                        if self.std_img is None:
                            self.add_history("标准图为空，无法合并ROI")
                            return
                        self.result_img = self.std_img.copy()
                    elif target == "待测图":
                        if self.cur_test_img is None:
                            self.add_history("待测图为空，无法合并ROI")
                            return
                        self.result_img = self.cur_test_img.copy()

                # 处理通道数不匹配的问题
                if self.result_img is not None:
                    # 检查通道数是否匹配
                    if len(self.result_img.shape) == 3 and len(self.selected_subroi.shape) == 2:
                        # 如果结果图是彩色的，但选区是灰度的，将选区转为彩色
                        colored_roi = cv2.cvtColor(self.selected_subroi, cv2.COLOR_GRAY2BGR)
                        self.result_img[y1:y2, x1:x2] = colored_roi
                    elif len(self.result_img.shape) == 2 and len(self.selected_subroi.shape) == 3:
                        # 如果结果图是灰度的，但选区是彩色的，将选区转为灰度
                        gray_roi = cv2.cvtColor(self.selected_subroi, cv2.COLOR_BGR2GRAY)
                        self.result_img[y1:y2, x1:x2] = gray_roi
                    else:
                        # 通道数匹配，直接复制
                        self.result_img[y1:y2, x1:x2] = self.selected_subroi.copy()
                    

                    self.canvas_result.set_image(self.result_img)
                    self.add_history(f"ROI合并回result_img: ({x1},{y1})-({x2},{y2})")

                self.selected_subroi = None
                self.roi_coords = None
                # 清除原始彩色ROI
                if hasattr(self, "original_roi_color"):
                    self.original_roi_color = None
            else:
                self.add_history("无选区需要合并")
            
            # 通知用户已退出选区模式
            messagebox.showinfo("提示", "已退出选区模式")
    def on_plot_reg_curve(self):
        """
        遍历当前选定的配准方法参数范围，绘制匹配率参数曲线，并输出最佳参数到历史记录。
        """
        if self.std_img is None or self.cur_test_img is None:
            messagebox.showwarning("提示", "请先加载标准图和待测图")
            return
        reg_method = self.reg_method_var.get()
        if reg_method == "none":
            messagebox.showwarning("提示", "请选择配准方法")
            return
        # 设置参数遍历范围
        if reg_method.upper() in ["SIFT", "ORB"]:
            param_range = list(range(100, 2100, 200))
        elif reg_method.upper() == "HARRIS":
            param_range = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        else:
            param_range = [0]
        try:
            if self.std_img is None or self.cur_test_img is None:
                messagebox.showwarning("提示", "请先加载标准图和待测图")
                return
                
            # 修改为接收3个返回值
            result = ir.plot_registration_curve(self.std_img, self.cur_test_img,
                                            detector=self.reg_method_var.get())
            if result is not None:
                curve_img, scores, params = result  # 正确解包3个返回值
                self.canvas_result.set_image(curve_img)
                
                # 显示详细信息
                self.add_history("\n=== 配准率曲线分析 ===")
                self.add_history(f"配准方法: {self.reg_method_var.get()}")
                self.add_history(f"参数范围: {min(params)}-{max(params)}")
                self.add_history(f"最佳分数: {max(scores):.4f}")
                self.add_history(f"最佳参数: {params[np.argmax(scores)]}")
                
        except Exception as e:
            self.add_history(f"绘制配准率曲线失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def on_optimal_threshold_curve(self):
        """
        根据当前选定的阈值分割方法，对当前处理对象（标准图/待测图/结果图）计算类间方差曲线，
        如果方法为 "manual"、"adaptive-mean" 或 "adaptive-gauss"，则绘制曲线图并输出最佳阈值到历史记录；
        如果方法为 "otsu" 或 "none"，则不执行遍历。
        """
        target = self.target_var.get()
        if target == "标准图":
            base_img = self.std_img
        elif target == "待测图":
            base_img = self.cur_test_img
        else:
            base_img = self.result_img
        if base_img is None:
            messagebox.showwarning("提示", f"{target}为空")
            return
        method = self.thresh_method_var.get()
        best_thresh, max_between = ip.optimal_threshold_curve(base_img, method=method)
        if best_thresh is not None:
            self.add_history(f"{target} 最优阈值: {best_thresh} (类间方差: {max_between:.2f})")
        else:
            self.add_history(f"{target} 阈值方法为 {method}，未执行阈值遍历")
    def optimal_threshold_curve(gray_img, method="manual"):
        """
        遍历0~255阈值，计算每个阈值下的类间方差，并绘制阈值-类间方差曲线，
        返回最佳阈值和对应的最大类间方差。
        
        参数：
          gray_img: 灰度图
          method: 阈值方法。支持：
                 "manual"、"adaptive-mean"、"adaptive-gauss"：执行遍历计算
                 "otsu" 或 "none"：不执行遍历，直接返回 None
        """
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
    
        # 如果图像不是灰度图，则转换
        if len(gray_img.shape) != 2:
            gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_img.copy()
    
        # 对于 OTSU 或 "none" 不进行遍历
        if method in ["otsu", "none"]:
            print("当前阈值方法为 {}，不执行阈值遍历".format(method))
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
                between_var[t] = ((mu_total * omega[t] - mu[t]) ** 2) / (omega[t] * (1 - omega[t]) + 1e-6)
    
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

    def on_show_histogram(self):
        target = self.target_var.get()
        if target == "标准图":
            base = self.std_img
        elif target == "待测图":
            base = self.cur_test_img
        else:
            base = self.result_img
        if base is None:
            messagebox.showwarning("提示", f"{target}为空")
            return
        ip.show_histogram(base)

    def on_morph_compare(self):
        """
        调用 image_processing.morph_compare() 对当前处理对象进行阈值分割和形态学处理对比，
        并将拼接结果显示到结果Canvas，同时更新历史记录。
        """
        target = self.target_var.get()
        if target == "标准图":
            base_img = self.std_img
        elif target == "待测图":
            base_img = self.cur_test_img
        else:
            base_img = self.result_img
        if base_img is None:
            messagebox.showwarning("提示", f"{target}为空")
            return
        thresh_method = self.thresh_method_var.get()
        manual_val = self.manual_thresh_val.get()
        morph_method = self.morph_method_var.get()
        ksize = self.morph_kernel.get()
        iterations = self.morph_iter.get()
        comp = ip.morph_compare(base_img, thresh_method=thresh_method, manual_val=manual_val,
                                morph_method=morph_method, ksize=ksize, iterations=iterations)
        if comp is not None:
            self.canvas_result.set_image(comp)
        self.add_history(f"{target} 阈值分割({thresh_method}:{manual_val}) + 形态学对比 "
                         f"({morph_method}, 核={ksize}, 迭代={iterations})")

    def on_plot_curve_select(self, choice):
        if self.std_img is None or self.cur_test_img is None:
            messagebox.showwarning("提示", "请先加载标准图和待测图")
            self.plot_curve_var.set("绘制匹配曲线")
            return
        if choice == "绘制SIFT匹配曲线":
            ir.plot_sift_curve(self.std_img, self.cur_test_img)
        elif choice == "绘制ORB匹配曲线":
            ir.plot_orb_curve(self.std_img, self.cur_test_img)
        elif choice == "绘制Harris匹配曲线":
            ir.plot_harris_curve(self.std_img, self.cur_test_img)
        self.plot_curve_var.set("绘制匹配曲线")

    def load_standard(self):
        path = filedialog.askopenfilename(title="选择标准图",
                                        filetypes=[("Image files","*.jpg *.png *.bmp")])
        if not path:
            return
        # 修改为读取彩色图像
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误","无法读取标准图")
            return
        self.std_img = img
        self.add_history(f"加载标准图: {os.path.basename(path)}, size={img.shape}")
        self.canvas_std.set_image(self.std_img)
        self.update_result_image()
    def load_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        paths = glob.glob(os.path.join(folder, "*.jpg")) + \
                glob.glob(os.path.join(folder, "*.png")) + \
                glob.glob(os.path.join(folder, "*.bmp"))
        if not paths:
            messagebox.showinfo("提示", "该文件夹下没有找到常见格式图像")
            return
        self.test_images.clear()
        self.test_listbox.delete(0, tk.END)
        for p in paths:
            fn = os.path.basename(p)
            # 修改为读取彩色图像，不进行灰度化
            img = cv2.imread(p)
            if img is not None:
                self.test_images[fn] = img
                self.test_listbox.insert(tk.END, fn)
        self.add_history(f"加载待测图文件夹: {folder}, 共{len(self.test_images)} 张")
    def on_save_batch_select(self, choice):
        self.save_batch_var.set("保存...")
        if choice.startswith("保存"):
            if choice == "保存标准图":
                self.save_image("标准图")
            elif choice == "保存待测图":
                self.save_image("待测图")
            else:
                self.save_image("结果图")
        else:
            if choice == "批量处理待测图":
                self.do_batch("待测图")
            else:
                self.do_batch("结果图")


    def save_image(self, target_type):
        """保存图像到指定路径"""
        if target_type == "标准图" and self.std_img is None:
            messagebox.showwarning("提示", "标准图为空")
            return
        elif target_type == "待测图" and self.cur_test_img is None:
            messagebox.showwarning("提示", "待测图为空")
            return
        elif target_type == "结果图" and self.result_img is None:
            messagebox.showwarning("提示", "结果图为空")
            return

        filename = filedialog.asksaveasfilename(
            title=f"保存{target_type}",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not filename:
            return

        try:
            # 获取要保存的图像
            if target_type == "标准图":
                img = self.std_img
            elif target_type == "待测图":
                img = self.cur_test_img
            else:
                img = self.result_img
                
            # 检查图像的颜色空间并进行转换
            if len(img.shape) == 3:  # 彩色图像
                # OpenCV使用BGR，而我们的显示使用RGB，所以保存前需要转换
                img_to_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:  # 灰度图像
                img_to_save = img
                
            # 保存图像
            cv2.imwrite(filename, img_to_save)
            self.add_history(f"已保存{target_type}到: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
    def do_batch(self, target_type):
        """批量处理并保存图像"""
        if not self.test_images:
            messagebox.showwarning("提示", "请先加载待测图文件夹")
            return
        
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return

        total = len(self.test_images)
        processed = 0
        
        for fn, img in self.test_images.items():
            # 设置当前处理的图像
            self.cur_test_img = img.copy()
            # 执行处理
            self.update_result_image()
            # 保存处理结果
            if self.result_img is not None:
                save_path = os.path.join(save_dir, f"processed_{fn}")
                cv2.imwrite(save_path, self.result_img)
                processed += 1
                
        self.add_history(f"批量处理完成: {processed}/{total} 张图像已保存到 {save_dir}")
    def on_toggle_processing(self):
        """
        切换 self.enable_processing 的值，并更新按钮文字。
        """
        if self.enable_processing:
            # 目前是开启状态 -> 切换为关闭
            self.enable_processing = False
            self.btn_toggle_processing.config(text="图像处理：关闭")
            self.add_history("已关闭图像处理功能")
        else:
            # 目前是关闭状态 -> 切换为开启
            self.enable_processing = True
            self.btn_toggle_processing.config(text="图像处理：开启")
            self.add_history("已开启图像处理功能")

    ###########################################
    # 左侧
    ###########################################
    def build_left_panel(self):
        left_frame = tk.Frame(self.tab1, bg="#f0f0f0")  # 修改这里
        left_frame.grid(row=1, column=0, rowspan=2, sticky="nsw")
        # 历史记录
        tk.Label(left_frame, text="历史记录:", bg="#ccc").pack(fill="x")
        hist_frame = tk.Frame(left_frame)
        hist_frame.pack(fill="both", expand=True)
        hist_frame.rowconfigure(0, weight=1)
        hist_frame.columnconfigure(0, weight=1)
        self.hist_list = tk.Listbox(hist_frame, width=28)
        self.hist_list.grid(row=0, column=0, sticky="nsew")
        s_hist_y = tk.Scrollbar(hist_frame, orient="vertical", command=self.hist_list.yview)
        s_hist_y.grid(row=0, column=1, sticky="ns")
        s_hist_x = tk.Scrollbar(hist_frame, orient="horizontal", command=self.hist_list.xview)
        s_hist_x.grid(row=1, column=0, sticky="ew")
        self.hist_list.config(yscrollcommand=s_hist_y.set, xscrollcommand=s_hist_x.set)
        # 待测图列表
        tk.Label(left_frame, text="待测图列表:", bg="#ccc").pack(fill="x")
        test_frame = tk.Frame(left_frame)
        test_frame.pack(fill="both", expand=True)
        test_frame.rowconfigure(0, weight=1)
        test_frame.columnconfigure(0, weight=1)
        self.test_listbox = tk.Listbox(test_frame, width=28)
        self.test_listbox.bind("<ButtonRelease-1>", self.on_select_test)
        self.test_listbox.grid(row=0, column=0, sticky="nsew")
        s_test_y = tk.Scrollbar(test_frame, orient="vertical", command=self.test_listbox.yview)
        s_test_y.grid(row=0, column=1, sticky="ns")
        s_test_x = tk.Scrollbar(test_frame, orient="horizontal", command=self.test_listbox.xview)
        s_test_x.grid(row=1, column=0, sticky="ew")
        self.test_listbox.config(yscrollcommand=s_test_y.set, xscrollcommand=s_test_x.set)

    def add_history(self, msg):
        self.history.append(msg)
        self.hist_list.insert(tk.END, msg)
        self.hist_list.see(tk.END)

    def on_select_test(self, event):
        sel = self.test_listbox.curselection()
        if not sel:
            return
        fn = self.test_listbox.get(sel[0])
        if fn not in self.test_images:
            return
        self.cur_test_img = self.test_images[fn]
        self.add_history(f"切换待测图: {fn}, size={self.cur_test_img.shape}")
        self.canvas_test.set_image(self.cur_test_img)
        
        # 只有在启用图像处理时才更新结果图
        if self.enable_processing:
            self.update_result_image()
        else:
            # 不处理，直接显示原图
            self.canvas_result.set_image(self.cur_test_img)
    def on_compare_features(self):
        try:
            if self.std_img is None or self.cur_test_img is None:
                messagebox.showwarning("提示", "请先加载标准图和待测图")
                return
                
            self.add_history("=== 开始特征对比 ===")
            
            # 新增调试信息
            self.add_history(f"输入图像信息 - 标准图 shape: {self.std_img.shape} dtype: {self.std_img.dtype}")
            self.add_history(f"输入图像信息 - 待测图 shape: {self.cur_test_img.shape} dtype: {self.cur_test_img.dtype}")
            
            # 执行特征对比
            result = ir.compare_detectors(
                self.std_img, 
                self.cur_test_img,
                sift_nfeatures=self.sift_nfeatures.get(),
                orb_nfeatures=self.orb_nfeatures.get(),
                harris_thresh=self.harris_thresh.get()
            )
            # 新的返回值处理逻辑
            if isinstance(result, list):
                valid_images = []
                for i, img in enumerate(result):
                    if isinstance(img, np.ndarray):
                        # 自动转换灰度图为BGR格式
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        valid_images.append(img)
                        self.add_history(f"列表项{i}有效 | 尺寸：{img.shape} | 通道：{img.shape[2] if len(img.shape)>2 else 1}")
                
                # 图像拼接处理
                if valid_images:
                    try:  # 优先尝试垂直拼接
                        composite = np.vstack(valid_images)
                    except Exception as e:
                        self.add_history(f"垂直拼接失败：{str(e)}，尝试水平拼接")
                        composite = np.hstack(valid_images)
                    self.add_history(f"最终拼接尺寸：{composite.shape}")
                else:
                    self.add_history("警告：无有效图像，生成空白画布")
                    composite = np.zeros((480,640,3), dtype=np.uint8)
                    cv2.putText(composite, "无有效图像", (160,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                results = []  # 确保results初始化为空列表
                self.add_history(f"已处理列表类型返回值 | 包含{len(valid_images)}张有效图像")
            
            elif isinstance(result, tuple) and len(result)>=2:
                composite, results = result
                self.add_history(f"元组类型返回值 | 图像类型：{type(composite)} | 结果数量：{len(results)}")
            
            else:
                self.add_history(f"无法处理的返回值类型：{type(result)}")
                return
                    
            # 显示结果
            self.canvas_result.set_image(composite)             
            self.add_history("=== 特征对比 ===")
            
            # 处理结果信息
            if isinstance(results, list):
                for i, r in enumerate(results):
                    # 新增特征点数量校验
                    if not isinstance(r, dict):
                        self.add_history(f"结果{i}格式错误，不是字典类型")
                        continue
                        
                    required_keys = ['method', 'std_kp', 'test_kp', 'matches']
                    missing_keys = [k for k in required_keys if k not in r]
                    if missing_keys:
                        self.add_history(f"结果{i}缺少必要字段: {missing_keys}")
                        continue
                        
                    # 新增空特征点警告
                    msg = (f"方法={r['method']} | 标准图kp={r['std_kp'] if isinstance(r['std_kp'], int) else len(r['std_kp'])} | "
                        f"待测图kp={r['test_kp'] if isinstance(r['test_kp'], int) else len(r['test_kp'])} | "
                        f"匹配点={r['matches'] if isinstance(r['matches'], int) else len(r['matches'])}")
                    
                    if len(r['std_kp']) == 0:
                        self.add_history(f"警告：方法{r['method']}在标准图中未检测到特征点")
                    if len(r['test_kp']) == 0:
                        self.add_history(f"警告：方法{r['method']}在待测图中未检测到特征点")
                    if len(r['matches']) == 0:
                        self.add_history(f"警告：方法{r['method']}未找到有效匹配，可能原因：1.图像模糊 2.参数设置不当 3.图像未对齐")
            else:
                self.add_history(f"警告：results不是列表类型，类型={type(results)}")
                    
            # 新增组合图像调试信息
            if composite.size == 0:
                self.add_history("错误：生成空图像（尺寸为0）")
            elif composite.shape[0] < 10 or composite.shape[1] < 10:  # 最小尺寸校验
                self.add_history(f"错误：生成图像尺寸过小（{composite.shape}）")
            else:
                self.add_history(f"成功生成组合图像，尺寸={composite.shape}")
                    
            self.add_history("=== 对比结束 ===")
            
        except Exception as e:
            error_msg = f"特征对比失败: {str(e)}"
            self.add_history(error_msg)
            # 新增错误类型判断
            if "queryIdx" in str(e):
                self.add_history("可能原因：特征匹配时索引越界，请检查特征点提取结果")
            elif "descriptors" in str(e):
                self.add_history("可能原因：特征描述符生成失败，请检查输入图像")
            messagebox.showerror("错误", error_msg)
            
            # 记录完整堆栈信息
            import traceback
            full_trace = traceback.format_exc()
            self.add_history(f"错误堆栈:\n{full_trace}")  
        try:
            self.add_history("=== 开始特征对比 ===")
            
            # 新增调试信息
            self.add_history(f"输入图像信息 - 标准图 shape: {self.std_img.shape} dtype: {self.std_img.dtype}")
            self.add_history(f"输入图像信息 - 待测图 shape: {self.cur_test_img.shape} dtype: {self.cur_test_img.dtype}")
            
            # 执行特征对比 - 新增method参数
            result = ir.compare_detectors(
                self.std_img, 
                self.cur_test_img,
                method=self.reg_method_var.get(),  # 获取当前选择的配准方法
                sift_nfeatures=self.sift_nfeatures.get(),
                orb_nfeatures=self.orb_nfeatures.get(),
                harris_thresh=self.harris_thresh.get()
            )
            # 检查返回值类型
            if result is None:
                self.add_history("错误：特征对比返回空结果")
                return
                
            # 正确解包返回值
            if isinstance(result, tuple) and len(result) >= 2:
                composite, results = result
                self.add_history(f"成功获取返回值：composite类型={type(composite)}, results类型={type(results)}")
            else:
                self.add_history(f"警告：返回值格式异常，类型={type(result)}")
                if isinstance(result, np.ndarray):
                    # 如果直接返回了图像
                    composite = result
                    results = []
                else:
                    raise TypeError(f"无法处理的返回值类型: {type(result)}")
                    
            # 检查composite是否为有效图像
            if isinstance(composite, list):
                self.add_history(f"检测到composite是列表类型，尝试转换为图像")
                # 尝试将列表中的第一个元素作为图像
                if len(composite) > 0 and isinstance(composite[0], np.ndarray):
                    self.add_history(f"使用列表中的第一个元素作为图像，shape={composite[0].shape}")
                    composite = composite[0]
                else:
                    # 尝试垂直拼接列表中的所有图像
                    valid_images = [img for img in composite if isinstance(img, np.ndarray)]
                    if valid_images:
                        self.add_history(f"尝试垂直拼接{len(valid_images)}个有效图像")
                        try:
                            composite = np.vstack(valid_images)
                        except Exception as e:
                            self.add_history(f"拼接失败: {str(e)}，尝试水平拼接")
                            try:
                                composite = np.hstack(valid_images)
                            except Exception as e:
                                self.add_history(f"水平拼接也失败: {str(e)}")
                                composite = None
                    else:
                        self.add_history("列表中没有有效图像")
                        composite = None
                        
            if not isinstance(composite, np.ndarray):
                self.add_history(f"错误：无法获取有效图像，类型={type(composite)}")
                return
                    
            # 显示结果
            self.canvas_result.set_image(composite)             
            self.add_history("=== 特征对比 ===")
            if not isinstance(results, list):
                raise TypeError("compare_detectors应返回结果列表")
                
            for i, r in enumerate(results):
                # 新增特征点数量校验
                if not isinstance(r, dict):
                    self.add_history(f"结果{i}格式错误，不是字典类型")
                    continue
                    
                required_keys = ['method', 'std_kp', 'test_kp', 'matches']
                missing_keys = [k for k in required_keys if k not in r]
                if missing_keys:
                    self.add_history(f"结果{i}缺少必要字段: {missing_keys}")
                    continue
                    
                # 新增空特征点警告
                msg = (f"方法={r['method']} | 标准图kp={len(r['std_kp'])} | 待测图kp={len(r['test_kp'])} | "
                    f"匹配数={len(r['matches'])} | 用时={r.get('time_ms', 'N/A')}ms")
                self.add_history(msg)
                
                if len(r['std_kp']) == 0:
                    self.add_history(f"警告：方法{r['method']}在标准图中未检测到特征点")
                if len(r['test_kp']) == 0:
                    self.add_history(f"警告：方法{r['method']}在待测图中未检测到特征点")
                if len(r['matches']) == 0:
                    self.add_history(f"警告：方法{r['method']}未找到有效匹配，可能原因：1.图像模糊 2.参数设置不当 3.图像未对齐")
                    
            # 新增组合图像调试信息
            if composite is None:
                self.add_history("错误：组合图像生成失败（返回值为None）")
            elif composite.size == 0:
                self.add_history("错误：生成空图像（尺寸为0）")
            elif composite.shape[0] < 10 or composite.shape[1] < 10:  # 最小尺寸校验
                self.add_history(f"错误：生成图像尺寸过小（{composite.shape}）")
            else:
                self.canvas_result.set_image(composite)
                
            self.add_history("=== 对比结束 ===")
            
        except Exception as e:
            error_msg = f"特征对比失败: {str(e)}"
            self.add_history(error_msg)
            # 新增错误类型判断
            if "queryIdx" in str(e):
                self.add_history("可能原因：特征匹配时索引越界，请检查特征点提取结果")
            elif "descriptors" in str(e):
                self.add_history("可能原因：特征描述符生成失败，请检查输入图像")
            messagebox.showerror("错误", error_msg)
            
            # 记录完整堆栈信息
            import traceback
            full_trace = traceback.format_exc()
            self.add_history(f"错误堆栈:\n{full_trace}")
    ###########################################
    # 中间
    ###########################################
    def build_center_canvases(self):
        # 创建三个画布，使用位置参数传递父窗口
        self.canvas_std = ZoomableCanvas(self.tab1)  # 使用位置参数
        self.canvas_std.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.canvas_test = ZoomableCanvas(self.tab1)
        self.canvas_test.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        
        # 修改这里，添加 img=None 参数
        self.canvas_result = ROISelectableCanvas(self.tab1, img=None, main_gui=self)
        self.canvas_result.grid(row=2, column=1, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # 使用configure方法设置背景色
        try:
            self.canvas_std.configure(bg="#E0E0E0")
            self.canvas_test.configure(bg="#E0E0E0")  # 浅灰色背景
            self.canvas_result.configure(bg="#E0E0E0")
        except Exception as e:
            self.add_history(f"设置Canvas背景色失败: {str(e)}")
    def update_std_canvas(self):
        if self.std_img is not None:
            self.canvas_std.set_image(self.std_img)
    def update_result_image(self):
        """根据当前参数设置更新结果图像"""
        target = self.target_var.get()
        
        if target == "标准图":
            base_img = self.std_img
        elif target == "待测图":
            base_img = self.cur_test_img
        else:
            base_img = self.result_img
            
        if base_img is None:
            messagebox.showwarning("提示", f"{target}为空")
            return
            
        # 应用当前的图像处理参数
        try:
            self.add_history("=== 开始阈值差分调试 ===")
            self.add_history(f"初始图像信息: shape={base_img.shape} dtype={base_img.dtype} channels={base_img.shape[2] if len(base_img.shape)==3 else 1}")
            
            # 转换为灰度图
            if len(base_img.shape) == 3:
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
                self.add_history(f"灰度转换后: shape={gray.shape} dtype={gray.dtype} (通道数=1)")
            else:
                gray = base_img.copy()
                self.add_history("图像已经是灰度格式")
                
            # 应用滤波
            filter_method = self.filter_var.get()
            if filter_method != "none":
                filter_h = self.filter_h.get()
                filter_k = self.filter_k.get()
                
                self.add_history(f"应用滤波 - 方法={filter_method} h={filter_h} k={filter_k}")
                
                if filter_method == "gaussian":
                    gray = cv2.GaussianBlur(gray, (filter_k, filter_k), 0)
                elif filter_method == "median":
                    gray = cv2.medianBlur(gray, filter_k)
                elif filter_method == "bilateral":
                    gray = cv2.bilateralFilter(gray, filter_k, filter_h, filter_h)
                elif filter_method == "nlm":
                    gray = cv2.fastNlMeansDenoising(gray, None, filter_h, filter_k, 7)
                    
                self.add_history(f"滤波后图像信息: shape={gray.shape} dtype={gray.dtype} 极值=[{gray.min()}, {gray.max()}]")
            
            # 应用直方图均衡化
            eq_method = self.eq_var.get()
            if eq_method != "none":
                self.add_history(f"应用直方图均衡化 - 方法={eq_method}")
                
                if eq_method == "global":
                    gray = cv2.equalizeHist(gray)
                elif eq_method == "clahe":
                    clip_limit = self.eq_clip.get()
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    self.add_history(f"CLAHE参数: clip_limit={clip_limit} tile_grid=(8,8)")
                
                self.add_history(f"均衡化后图像信息: shape={gray.shape} dtype={gray.dtype} 极值=[{gray.min()}, {gray.max()}]")
            
            # 应用锐化
            sharpen_method = self.sharpen_var.get()
            if sharpen_method != "none":
                sharpen_w = self.sharpen_w.get()
                self.add_history(f"应用锐化 - 方法={sharpen_method} 权重={sharpen_w}")
                
                if sharpen_method == "laplacian":
                    kernel = np.array([[0, -1, 0], [-1, 4 + sharpen_w, -1], [0, -1, 0]], np.float32)
                    gray = cv2.filter2D(gray, -1, kernel)
                elif sharpen_method == "unsharp":
                    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
                    gray = cv2.addWeighted(gray, 1 + sharpen_w, gaussian, -sharpen_w, 0)
                
                self.add_history(f"锐化后图像信息: shape={gray.shape} dtype={gray.dtype} 极值=[{gray.min()}, {gray.max()}]")
            
            # 应用阈值分割
            thresh_method = self.thresh_method_var.get()
            if thresh_method == "manual":
                thresh_val = self.manual_thresh_val.get()
                _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                self.add_history(f"手动阈值分割: 阈值={thresh_val} 有效像素统计: 前景={np.sum(binary==255)} 背景={np.sum(binary==0)}")
            elif thresh_method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.add_history(f"Otsu自动阈值: 计算阈值={_} 有效像素统计: 前景={np.sum(binary==255)} 背景={np.sum(binary==0)}")
            elif thresh_method == "adaptive":
                block_size = self.adapt_blockSize.get()
                c = self.adapt_C.get()
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, block_size, c)
                self.add_history(f"自适应阈值: block_size={block_size} C={c} 有效像素统计: 前景={np.sum(binary==255)} 背景={np.sum(binary==0)}")
            else:
                binary = gray
                self.add_history("跳过阈值分割步骤")
            
            # 应用形态学操作
            morph_method = self.morph_method_var.get()
            if morph_method != "none":
                kernel_size = self.morph_kernel.get()
                iterations = self.morph_iter.get()
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                self.add_history(f"应用形态学操作 - 方法={morph_method} 核大小={kernel_size}x{kernel_size} 迭代次数={iterations}")
                
                if morph_method == "erode":
                    binary = cv2.erode(binary, kernel, iterations=iterations)
                elif morph_method == "dilate":
                    binary = cv2.dilate(binary, kernel, iterations=iterations)
                elif morph_method == "open":
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
                elif morph_method == "close":
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                elif morph_method == "gradient":
                    binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
                
                self.add_history(f"形态学操作后图像信息: shape={binary.shape} dtype={binary.dtype} 极值=[{binary.min()}, {binary.max()}]")
            
            # 显示结果
            if len(binary.shape) == 2:
                result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                self.add_history(f"灰度转BGR后: shape={result.shape} channels={result.shape[2]}")
            else:
                result = binary
                
            self.result_img = result
            self.add_history(f"最终结果图像信息: shape={result.shape} dtype={result.dtype} channels={result.shape[2] if len(result.shape)==3 else 1}")
            self.canvas_result.set_image(result)
            
        except Exception as e:
            self.add_history(f"图像处理出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
            messagebox.showerror("错误", f"图像处理失败: {str(e)}")

    ###########################################
    # 右侧：参数调节区域（实时更新）
    ###########################################
    def build_right_panel(self):
        right_frame = tk.Frame(self.tab1, bg="#e0e0e0", width=250)  # 修改这里
        right_frame.grid(row=1, column=3, rowspan=2, sticky="ns")
        right_frame.config(width=250)
        right_frame.grid_propagate(False)

        tk.Label(right_frame, text="处理对象:", bg="#e0e0e0").pack(pady=3)
        self.target_var = tk.StringVar(value="待测图")
        tk.OptionMenu(right_frame, self.target_var, "标准图", "待测图", "结果图",
                      command=lambda _: self.update_result_image()).pack()

        tk.Button(right_frame, text="设置预处理顺序", command=self.show_order_dialog).pack(pady=5)

        tk.Label(right_frame, text="滤波方法:", bg="#e0e0e0").pack(pady=2)
        self.filter_var = tk.StringVar(value="nlm")
        tk.OptionMenu(right_frame, self.filter_var, "nlm", "mean", "median", "gaussian", "none",
                      command=lambda _: self.update_result_image()).pack()

        tk.Label(right_frame, text="NLM h:", bg="#e0e0e0").pack()
        self.filter_h = tk.IntVar(value=10)
        tk.Scale(right_frame, from_=0, to=30, orient="horizontal", variable=self.filter_h,
                 command=lambda _: self.update_result_image()).pack(fill="x")

        tk.Label(right_frame, text="Kernel大小:", bg="#e0e0e0").pack()
        self.filter_k = tk.IntVar(value=3)
        tk.Scale(right_frame, from_=1, to=11, resolution=2, orient="horizontal", variable=self.filter_k,
                 command=lambda _: self.update_result_image()).pack(fill="x")

        tk.Label(right_frame, text="锐化方法:", bg="#e0e0e0").pack()
        self.sharpen_var = tk.StringVar(value="laplacian")
        tk.OptionMenu(right_frame, self.sharpen_var, "laplacian", "gradient", "none",
                      command=lambda _: self.update_result_image()).pack()

        tk.Label(right_frame, text="锐化权重:", bg="#e0e0e0").pack()
        self.sharpen_w = tk.DoubleVar(value=1.0)
        tk.Scale(right_frame, from_=0.0, to=2.0, resolution=0.1, orient="horizontal", variable=self.sharpen_w,
                 command=lambda _: self.update_result_image()).pack(fill="x")

        tk.Label(right_frame, text="均衡化:", bg="#e0e0e0").pack()
        self.eq_var = tk.StringVar(value="clahe")
        tk.OptionMenu(right_frame, self.eq_var, "manual", "cv2", "clahe", "none",
                      command=lambda _: self.update_result_image()).pack()

        tk.Label(right_frame, text="CLAHE clip:", bg="#e0e0e0").pack()
        self.eq_clip = tk.DoubleVar(value=2.0)
        tk.Scale(right_frame, from_=1.0, to=10.0, resolution=0.5, orient="horizontal", variable=self.eq_clip,
                 command=lambda _: self.update_result_image()).pack(fill="x")

        tk.Label(right_frame, text="边缘检测:", bg="#e0e0e0").pack()
        self.edge_var = tk.StringVar(value="none")
        tk.OptionMenu(right_frame, self.edge_var, "none", "canny", "sobel", "laplacian",
                      command=lambda _: self.update_result_image()).pack()

        tk.Label(right_frame, text="配准方法:", bg="#e0e0e0").pack()
        self.reg_method_var = tk.StringVar(value="none")
        tk.OptionMenu(right_frame, self.reg_method_var, "none", "SIFT", "ORB", "Harris",
                      command=lambda _: self.update_result_image()).pack()

        tk.Button(right_frame, text="特征对比", command=self.on_compare_features).pack(pady=5)

        tk.Button(right_frame, text="图像对齐叠加", command=self.on_align_overlay).pack(pady=5)

    def show_order_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("预处理顺序设置")
        dialog.grab_set()

        tk.Label(dialog, text="步骤1:").grid(row=0, column=0, padx=5, pady=5)
        order1 = tk.StringVar(value=self.order_step1.get())
        tk.OptionMenu(dialog, order1, "none", "滤波", "锐化", "均衡化").grid(row=0, column=1, padx=5, pady=5)

        tk.Label(dialog, text="步骤2:").grid(row=1, column=0, padx=5, pady=5)
        order2 = tk.StringVar(value=self.order_step2.get())
        tk.OptionMenu(dialog, order2, "none", "滤波", "锐化", "均衡化").grid(row=1, column=1, padx=5, pady=5)

        tk.Label(dialog, text="步骤3:").grid(row=2, column=0, padx=5, pady=5)
        order3 = tk.StringVar(value=self.order_step3.get())
        tk.OptionMenu(dialog, order3, "none", "滤波", "锐化", "均衡化").grid(row=2, column=1, padx=5, pady=5)

        def on_confirm():
            self.order_step1.set(order1.get())
            self.order_step2.set(order2.get())
            self.order_step3.set(order3.get())
            self.update_result_image()
            dialog.destroy()

        tk.Button(dialog, text="确定", command=on_confirm).grid(row=3, column=0, columnspan=2, pady=10)

    def on_align_overlay(self):
        try:
            if self.std_img is None or self.cur_test_img is None:
                messagebox.showwarning("提示", "请先加载标准图和待测图")
                return
            
            # 新增输入验证
            self.add_history("=== 开始图像对齐调试 ===")
            self.add_history(f"标准图信息: shape={self.std_img.shape} dtype={self.std_img.dtype}")
            self.add_history(f"待测图信息: shape={self.cur_test_img.shape} dtype={self.cur_test_img.dtype}")
            
            reg_method = self.reg_method_var.get()
            self.add_history(f"当前配准方法: {reg_method}")
            
            # 参数验证
            if reg_method == "none":
                messagebox.showwarning("提示", "请选择配准方法")
                return
            elif reg_method not in ["SIFT", "ORB", "Harris"]:
                messagebox.showerror("错误", f"不支持的配准方法: {reg_method}")
                return
        
            # 参数收集调试
            params = {
                "detector": reg_method,
                "nfeatures": self.sift_nfeatures.get() if reg_method in ["SIFT", "ORB"] else 0,
                "h_thresh": self.harris_thresh.get() if reg_method == "Harris" else 0.0
            }
            self.add_history(f"调用参数: {params}")
            
            # 调用对齐函数
            overlay, aligned, score = ir.align_and_overlay(
                self.std_img,
                self.cur_test_img,
                **params
            )
            # === 新增通道数校验 ===
            if aligned is not None and len(aligned.shape) == 3 and aligned.shape[2] == 3:
                self.add_history("检测到对齐结果为BGR格式，自动转换为灰度图")
                aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                aligned = aligned_gray.copy()
            # 结果验证
            if overlay is None or aligned is None:
                error_msg = "图像对齐失败：返回空结果"
                self.add_history(error_msg)
                if self.std_img.shape != self.cur_test_img.shape:
                    self.add_history("图像尺寸不匹配可能导致对齐失败")
                messagebox.showerror("错误", error_msg)
                return
                
            self.add_history(f"对齐结果: overlay_shape={overlay.shape} aligned_shape={aligned.shape} score={score:.2f}")
            
            # === 新增通道数记录 ===
            self.add_history(f"对齐结果: overlay_shape={overlay.shape}（通道数{overlay.shape[2] if len(overlay.shape)==3 else 1}）")
            self.add_history(f"对齐结果: aligned_shape={aligned.shape}（通道数{len(aligned.shape)}）")



            # 显示结果
            self.canvas_result.set_image(overlay)
            self.aligned_test_img = aligned.copy()
            self.add_history(f"图像对齐叠加成功: 方法={reg_method}, 匹配率={score:.2f}")

        except Exception as e:
            error_msg = f"图像对齐异常: {str(e)}"
            self.add_history(error_msg)
            # 常见错误类型判断
            if "non-empty" in str(e):
                self.add_history("可能原因：输入图像为空或无效尺寸")
            elif "data type" in str(e):
                self.add_history("可能原因：图像数据类型应为uint8")
            messagebox.showerror("错误", error_msg)
            
            # 记录完整堆栈信息
            import traceback
            full_trace = traceback.format_exc()
            self.add_history(f"错误堆栈:\n{full_trace}")

    ###########################################
    # 右侧第二个扩展面板
    ###########################################
    def build_extra_panel(self):
        """构建右侧扩展面板，包含特征检测参数和阈值分割参数"""
        self.extra_frame = tk.Frame(self.tab1, bg="#d0d0d0", width=250)  # 修改这里
        self.extra_frame.grid(row=1, column=4, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.extra_frame.grid_propagate(False)
        self.morph_method_var = tk.StringVar(value="none")

        # 特征检测参数
        tk.Label(self.extra_frame, text="SIFT nfeatures:", bg="#d0d0d0").pack(pady=2)
        self.sift_nfeatures = tk.IntVar(value=500)
        tk.Scale(self.extra_frame, from_=100, to=2000, orient="horizontal",
                variable=self.sift_nfeatures, 
                command=lambda x: self.update_result_image()).pack(fill="x")

        tk.Label(self.extra_frame, text="ORB nfeatures:", bg="#d0d0d0").pack(pady=2)
        self.orb_nfeatures = tk.IntVar(value=500)
        tk.Scale(self.extra_frame, from_=100, to=2000, orient="horizontal",
                variable=self.orb_nfeatures, 
                command=lambda x: self.update_result_image()).pack(fill="x")

        tk.Label(self.extra_frame, text="Harris 阈值:", bg="#d0d0d0").pack(pady=2)
        self.harris_thresh = tk.DoubleVar(value=0.01)
        tk.Scale(self.extra_frame, from_=0.005, to=0.03, resolution=0.001, orient="horizontal",
                variable=self.harris_thresh, 
                command=lambda x: self.update_result_image()).pack(fill="x")

        # 阈值分割参数
        tk.Label(self.extra_frame, text="阈值分割方法:", bg="#d0d0d0").pack(pady=2)
        self.thresh_method_var = tk.StringVar(value="none")
        thresh_menu = tk.OptionMenu(self.extra_frame, self.thresh_method_var,
                    "none", "manual", "otsu", "adaptive-mean", "adaptive-gauss")
        thresh_menu.pack(fill="x", padx=5)
        # 添加跟踪变量变化的回调
        self.thresh_method_var.trace_add("write", lambda *args: self.do_threshold_segmenting())

        tk.Label(self.extra_frame, text="手动阈值:", bg="#d0d0d0").pack(pady=2)
        self.manual_thresh_val = tk.IntVar(value=65)
        tk.Scale(self.extra_frame, from_=0, to=255, orient="horizontal",
                variable=self.manual_thresh_val,
                command=lambda x: self.do_threshold_segmenting()).pack(fill="x", padx=5)

        tk.Label(self.extra_frame, text="自适应 blockSize:", bg="#d0d0d0").pack(pady=2)
        self.adapt_blockSize = tk.IntVar(value=31)
        tk.Scale(self.extra_frame, from_=3, to=101, resolution=2, orient="horizontal",
                variable=self.adapt_blockSize,
                command=lambda x: self.do_threshold_segmenting()).pack(fill="x", padx=5)

        tk.Label(self.extra_frame, text="自适应 C 值:", bg="#d0d0d0").pack(pady=2)
        self.adapt_C = tk.IntVar(value=5)
        tk.Scale(self.extra_frame, from_=-20, to=20, orient="horizontal",
                variable=self.adapt_C,
                command=lambda x: self.do_threshold_segmenting()).pack(fill="x", padx=5)
        #阈值差分
        btn_thresh_diff = tk.Button(self.extra_frame, text="阈值差分", command=self.on_thresh_difference).pack(pady=5)


        # 形态学操作参数
        tk.Label(self.extra_frame, text="形态学操作:", bg="#d0d0d0").pack(pady=2)
        morph_menu = tk.OptionMenu(self.extra_frame, self.morph_method_var,
                    "none", "erosion", "dilation", "open", "close")
        morph_menu.pack(fill="x", padx=5)
        # 添加跟踪变量变化的回调
        self.morph_method_var.trace_add("write", lambda *args: self.do_threshold_segmenting())

        tk.Label(self.extra_frame, text="核大小:", bg="#d0d0d0").pack(pady=2)
        self.morph_kernel = tk.IntVar(value=3)
        tk.Scale(self.extra_frame, from_=1, to=11, resolution=2, orient="horizontal",
                variable=self.morph_kernel,
                command=lambda x: self.do_threshold_segmenting()).pack(fill="x", padx=5)

        tk.Label(self.extra_frame, text="迭代次数:", bg="#d0d0d0").pack(pady=2)
        self.morph_iter = tk.IntVar(value=1)
        tk.Scale(self.extra_frame, from_=1, to=5, orient="horizontal",
                variable=self.morph_iter,
                command=lambda x: self.do_threshold_segmenting()).pack(fill="x", padx=5)

    def do_threshold_segmenting(self):
        """
        根据当前扩展面板中设置的阈值分割方法和参数，对当前处理对象进行阈值分割，
        并实时将分割结果显示在结果 Canvas 上。
        """
        import image_processing as ip
        
        # 选区模式下的处理
        if self.selection_mode and self.selected_subroi is not None:
            # 使用选区进行处理
            if hasattr(self, "original_roi_color") and self.original_roi_color is not None:
                base_img = self.original_roi_color.copy()
            else:
                base_img = self.selected_subroi.copy()
            is_roi_mode = True
        else:
            # 非选区模式下的原有逻辑保持不变
            target = self.target_var.get()
            if target == "标准图":
                if self.std_img is None: return
                base_img = self.std_img.copy()
            elif target == "待测图":
                if self.cur_test_img is None: return
                base_img = self.cur_test_img.copy()
            else:
                if self.result_img is None: return
                base_img = self.result_img.copy()
            is_roi_mode = False
        
    
        if base_img is None:
            return
    
        # [保持原来的阈值、形态学处理逻辑不变...]
        method = self.thresh_method_var.get()
        manual_val = self.manual_thresh_val.get()
        blockSize = self.adapt_blockSize.get()
        C = self.adapt_C.get()
        bin_img = ip.threshold_segment(base_img, method=method, val=manual_val, blockSize=blockSize, C=C)
    
        morph_method = self.morph_method_var.get()
        ksize = self.morph_kernel.get()
        iterations = self.morph_iter.get()
        
        if morph_method == "erosion":
            proc_img = ip.morphology_erosion(bin_img, ksize=ksize, iterations=iterations)
        elif morph_method == "dilation":
            proc_img = ip.morphology_dilation(bin_img, ksize=ksize, iterations=iterations)
        elif morph_method == "open":
            proc_img = ip.morphology_open(bin_img, ksize=ksize)
        elif morph_method == "close":
            proc_img = ip.morphology_close(bin_img, ksize=ksize)
        else:
            proc_img = bin_img.copy()
    
        # [保持原来逻辑：若是选区模式就更新 selected_subroi，否则更新整图]
        if is_roi_mode:
            self.selected_subroi = proc_img.copy()
            self.canvas_result.set_image(self.selected_subroi)
            self.add_history("选区阈值分割: " + 
                        f"方法={method}, manual_val={manual_val}, blockSize={blockSize}, C={C}; " +
                        f"形态学操作: {morph_method}, kernel={ksize}, iterations={iterations}")
        else:
            self.result_img = proc_img.copy()
            self.canvas_result.set_image(self.result_img)
            self.add_history(f"{target} 阈值分割: " + 
                        f"方法={method}, manual_val={manual_val}, blockSize={blockSize}, C={C}; " +
                        f"形态学操作: {morph_method}, kernel={ksize}, iterations={iterations}")

    def on_thresh_difference(self):
        """对标准图和待测图进行差分处理"""
        import image_processing as ip
        import cv2
        import numpy as np
        import time
        
        # 检查输入图像
        if self.std_img is None or self.cur_test_img is None:
            messagebox.showwarning("提示", "请先加载标准图和待测图")
            return
        
        try:
            self.add_history("=== 开始阈值差分处理 ===")
            
            # 修改：每次都重新执行对齐，不再检查aligned_test_img是否存在
            # 获取配准方法参数
            reg_method = self.reg_method_var.get()
            
            # 从顶部菜单获取对齐方法
            if hasattr(self, "align_method_var") and self.align_method_var.get():
                align_method = self.align_method_var.get()
            else:
                align_method = "透视变换"  # 默认方法
            
            self.add_history(f"使用配准方法: {reg_method}, 对齐方法: {align_method}")
            self.add_history(f"标准图尺寸: {self.std_img.shape}, 待测图尺寸: {self.cur_test_img.shape}")
            
            # 记录当前选择的所有参数
            self.add_history(f"当前界面参数 - 特征点数量: SIFT={self.sift_nfeatures.get()}, ORB={self.orb_nfeatures.get()}")
            self.add_history(f"当前界面参数 - Harris阈值: {self.harris_thresh.get()}")
            
            if reg_method == "none":
                self.add_history("不进行对齐，直接使用原始图像")
                self.aligned_test_img = self.cur_test_img.copy()
            else:
                # 记录对齐开始时间
                start_time = time.time()
                self.add_history(f"开始执行{align_method}对齐...")
                
                try:
                    # 根据选择的对齐方法执行不同的对齐算法
                    if align_method == "透视变换":
                        # 使用原有的透视变换方法
                        self.add_history(f"执行透视变换对齐，使用{reg_method}特征点")
                        self.aligned_test_img = self.perform_perspective_alignment(reg_method)
                    elif align_method == "薄板样条变换":
                        # 使用TPS变换
                        self.add_history(f"执行薄板样条变换对齐，使用{reg_method}特征点")
                        self.aligned_test_img = self.perform_tps_alignment(reg_method)
                    elif align_method == "多尺度特征匹配":
                        # 使用多尺度特征匹配
                        self.add_history(f"执行多尺度特征匹配对齐，使用{reg_method}特征点")
                        self.aligned_test_img = self.perform_multiscale_alignment(reg_method)
                    elif align_method == "局部分块对齐":
                        # 使用局部分块对齐
                        self.add_history(f"执行局部分块对齐，使用{reg_method}特征点")
                        self.aligned_test_img = self.perform_local_alignment(reg_method)
                    else:
                        # 默认使用透视变换
                        self.add_history(f"未知对齐方法: {align_method}，默认使用透视变换")
                        self.aligned_test_img = self.perform_perspective_alignment(reg_method)
                    
                    # 记录对齐结束时间和结果信息
                    elapsed_time = time.time() - start_time
                    self.add_history(f"对齐完成，耗时: {elapsed_time:.2f}秒")
                    self.add_history(f"对齐结果尺寸: {self.aligned_test_img.shape}, 类型: {self.aligned_test_img.dtype}")
                    
                    # 检查对齐结果是否有效
                    if self.aligned_test_img is None:
                        raise ValueError("对齐结果为空")
                    
                    # 检查对齐结果尺寸是否与标准图一致
                    if self.aligned_test_img.shape[:2] != self.std_img.shape[:2]:
                        self.add_history(f"警告: 对齐结果尺寸({self.aligned_test_img.shape[:2]})与标准图尺寸({self.std_img.shape[:2]})不一致")
                        # 调整尺寸以匹配标准图
                        self.aligned_test_img = cv2.resize(self.aligned_test_img, (self.std_img.shape[1], self.std_img.shape[0]))
                        self.add_history(f"已调整对齐结果尺寸为: {self.aligned_test_img.shape}")
                
                except Exception as e:
                    self.add_history(f"对齐过程出错: {str(e)}")
                    import traceback
                    self.add_history(f"错误堆栈: {traceback.format_exc()}")
                    self.add_history("使用原始图像作为备选")
                    self.aligned_test_img = self.cur_test_img.copy()

            # 2. 阈值分割处理
            method = self.thresh_method_var.get()
            manual_val = self.manual_thresh_val.get()
            blockSize = self.adapt_blockSize.get()
            C = self.adapt_C.get()
            
            # 分别对标准图和对齐后的待测图做阈值分割
            bin_std = ip.threshold_segment(self.std_img, method=method, val=manual_val,
                                        blockSize=blockSize, C=C)
            bin_test = ip.threshold_segment(self.aligned_test_img, method=method, val=manual_val,
                                        blockSize=blockSize, C=C)
            
            # 3. 计算差异
            diff = cv2.bitwise_xor(bin_std, bin_test)
            
            # 4. 形态学处理
            morph_method = self.morph_method_var.get()
            kernel_size = self.morph_kernel.get()
            iterations = self.morph_iter.get()
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if morph_method == "erosion":
                diff = cv2.erode(diff, kernel, iterations=iterations)
            elif morph_method == "dilation":
                diff = cv2.dilate(diff, kernel, iterations=iterations)
            elif morph_method == "open":
                diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif morph_method == "close":
                diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # 5. 连通域分析和过滤
            min_area = self.min_area_var.get()
            if min_area > 0:
                # 标记连通域
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=8)
                
                # 创建掩码，只保留大于最小面积的连通域
                mask = np.zeros_like(diff)
                for i in range(1, num_labels):  # 跳过背景（标签0）
                    if stats[i, cv2.CC_STAT_AREA] >= min_area:
                        mask[labels == i] = 255
                
                diff = mask
            
            # 6. 结果可视化
            # 获取连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=8)
            
            # 创建彩色标记图像
            rgb_label = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
            
            # 为每个连通域随机分配颜色
            for i in range(1, num_labels):
                mask = (labels == i)
                rgb_label[mask] = np.random.randint(0, 255, size=3)
            
            # 叠加显示
            golden_resized = cv2.resize(self.std_img, (diff.shape[1], diff.shape[0]))
            biaoji = cv2.addWeighted(golden_resized, 0.6, rgb_label, 0.4, 0)
            
            # 水平拼接显示
            diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            result = np.hstack((diff_color, biaoji))
            
            # 7. 显示结果
            self.result_img = result
            self.canvas_result.set_image(self.result_img)
            
            # 8. 统计信息
            diff_pixels = np.sum(diff > 0)
            total_pixels = diff.size
            diff_ratio = (diff_pixels / total_pixels) * 100
            self.add_history(f"差异统计: 差异像素={diff_pixels} 总像素={total_pixels} 差异率={diff_ratio:.2f}%")
            
        except Exception as e:
            self.add_history(f"差分处理出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
            messagebox.showerror("错误", f"差分处理失败: {str(e)}")

    def perform_perspective_alignment(self, reg_method):
        """执行透视变换对齐"""
        import cv2
        import numpy as np
        
        self.add_history("执行透视变换对齐...")
        
        # 使用当前配准参数进行对齐
        if reg_method == "ORB":
            nfeatures = self.orb_nfeatures.get()
            detector = cv2.ORB_create(nfeatures=nfeatures)
        elif reg_method == "SIFT":
            nfeatures = self.sift_nfeatures.get()
            detector = cv2.SIFT_create(nfeatures=nfeatures)
        else:
            raise ValueError(f"不支持的配准方法: {reg_method}")

        # 转换为灰度图并进行特征匹配
        std_gray = cv2.cvtColor(self.std_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(self.cur_test_img, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = detector.detectAndCompute(std_gray, None)
        kp2, des2 = detector.detectAndCompute(test_gray, None)
        
        if des1 is None or des2 is None:
            raise ValueError("特征提取失败")
        
        # 特征匹配
        if reg_method == "ORB":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv2.BFMatcher()
        matches = bf.match(des1, des2)
        
        # 提取匹配点对
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # RANSAC求解变换矩阵
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            raise ValueError("无法计算变换矩阵")
        
        # 对齐待测图
        h, w = self.std_img.shape[:2]
        aligned_img = cv2.warpPerspective(self.cur_test_img, M, (w, h))
        self.add_history(f"完成透视变换对齐: 使用{reg_method}方法, 找到{len(matches)}对匹配点")
        
        return aligned_img

    def perform_tps_alignment(self, reg_method):
        """执行薄板样条变换(TPS)对齐"""
        import cv2
        import numpy as np
        from scipy import interpolate
        
        self.add_history("执行薄板样条变换(TPS)对齐...")
        
        # 特征提取
        if reg_method == "ORB":
            nfeatures = self.orb_nfeatures.get() * 2  # 增加特征点数量
            detector = cv2.ORB_create(nfeatures=nfeatures)
        elif reg_method == "SIFT":
            nfeatures = self.sift_nfeatures.get() * 2  # 增加特征点数量
            detector = cv2.SIFT_create(nfeatures=nfeatures)
        else:
            raise ValueError(f"不支持的配准方法: {reg_method}")
        
        # 转换为灰度图
        std_gray = cv2.cvtColor(self.std_img, cv2.COLOR_BGR2GRAY) if len(self.std_img.shape) == 3 else self.std_img
        test_gray = cv2.cvtColor(self.cur_test_img, cv2.COLOR_BGR2GRAY) if len(self.cur_test_img.shape) == 3 else self.cur_test_img
        
        # 提取特征点和描述符
        kp1, des1 = detector.detectAndCompute(std_gray, None)
        kp2, des2 = detector.detectAndCompute(test_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.add_history("特征提取失败或特征点数量不足，尝试使用透视变换替代")
            return self.perform_perspective_alignment(reg_method)
        
        # 特征匹配
        if reg_method == "ORB":
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            # 按距离排序
            matches = sorted(matches, key=lambda x: x.distance)
            # 选择最佳匹配点
            matches = matches[:min(100, len(matches))]
        else:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            # 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            matches = good_matches[:min(100, len(good_matches))]
        
        if len(matches) < 10:
            self.add_history(f"匹配点数量不足: {len(matches)}，尝试使用透视变换替代")
            return self.perform_perspective_alignment(reg_method)
        
        # 提取匹配点对
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches])  # 源图像点
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches])  # 目标图像点
        
        # 检查点是否有重复
        src_unique = np.unique(src_pts, axis=0)
        dst_unique = np.unique(dst_pts, axis=0)
        
        if len(src_unique) < len(src_pts) or len(dst_unique) < len(dst_pts):
            self.add_history(f"警告: 检测到重复点，从{len(src_pts)}减少到{len(src_unique)}和{len(dst_unique)}")
            # 使用唯一点
            src_pts = src_unique
            dst_pts = dst_unique[:len(src_unique)] if len(dst_unique) > len(src_unique) else dst_unique
        
        # 确保点数足够
        if len(src_pts) < 5 or len(dst_pts) < 5:
            self.add_history("TPS变换失败: 唯一匹配点数量不足，尝试使用透视变换替代")
            return self.perform_perspective_alignment(reg_method)
        
        try:
            # 添加四个角点以确保边界对齐
            h, w = self.std_img.shape[:2]
            corner_pts = np.array([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]], dtype=np.float32)
            
            # 确保角点不与现有点重复
            dst_pts_with_corners = np.vstack([dst_pts, corner_pts])
            src_pts_with_corners = np.vstack([src_pts, corner_pts])
            
            # 添加随机扰动避免奇异性
            epsilon = 1e-5
            noise = np.random.normal(0, epsilon, src_pts_with_corners.shape)
            src_pts_with_corners += noise
            
            # 使用更稳定的插值函数
            try:
                # 尝试使用multiquadric函数，通常更稳定
                tps_x = interpolate.Rbf(dst_pts_with_corners[:, 0], dst_pts_with_corners[:, 1], 
                                       src_pts_with_corners[:, 0], function='multiquadric', epsilon=2)
                tps_y = interpolate.Rbf(dst_pts_with_corners[:, 0], dst_pts_with_corners[:, 1], 
                                       src_pts_with_corners[:, 1], function='multiquadric', epsilon=2)
            except np.linalg.LinAlgError:
                # 如果仍然失败，尝试使用gaussian函数
                self.add_history("multiquadric插值失败，尝试使用gaussian插值")
                tps_x = interpolate.Rbf(dst_pts_with_corners[:, 0], dst_pts_with_corners[:, 1], 
                                       src_pts_with_corners[:, 0], function='gaussian', epsilon=5)
                tps_y = interpolate.Rbf(dst_pts_with_corners[:, 0], dst_pts_with_corners[:, 1], 
                                       src_pts_with_corners[:, 1], function='gaussian', epsilon=5)
            
            # 创建网格并应用变换
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            
            # 使用try-except捕获可能的插值错误
            try:
                map_x = tps_x(x_coords, y_coords)
                map_y = tps_y(x_coords, y_coords)
                
                # 检查映射是否有效
                if np.isnan(map_x).any() or np.isnan(map_y).any():
                    raise ValueError("插值结果包含NaN值")
                
                # 创建映射矩阵
                map_x_32 = map_x.astype(np.float32)
                map_y_32 = map_y.astype(np.float32)
                
                # 应用映射进行重映射
                aligned_img = cv2.remap(self.cur_test_img, map_x_32, map_y_32, 
                                       interpolation=cv2.INTER_LINEAR, 
                                       borderMode=cv2.BORDER_CONSTANT)
                
                self.add_history(f"完成薄板样条变换对齐: 使用{reg_method}方法, 找到{len(matches)}对匹配点")
                return aligned_img
                
            except Exception as e:
                self.add_history(f"TPS映射应用失败: {str(e)}，尝试使用透视变换替代")
                return self.perform_perspective_alignment(reg_method)
                
        except np.linalg.LinAlgError as e:
            self.add_history(f"TPS变换失败: {str(e)}，尝试使用透视变换替代")
            return self.perform_perspective_alignment(reg_method)
        except Exception as e:
            self.add_history(f"TPS变换异常: {str(e)}，尝试使用透视变换替代")
            return self.perform_perspective_alignment(reg_method)

    def perform_multiscale_alignment(self, reg_method):
        """执行多尺度特征匹配对齐"""
        import cv2
        import numpy as np
        
        self.add_history("执行多尺度特征匹配对齐...")
        
        # 创建不同尺度的图像金字塔
        scales = [1.0, 0.75, 0.5, 0.25]
        best_matches = []
        best_homography = None
        max_inliers = 0
        
        std_img = self.std_img.copy()
        test_img = self.cur_test_img.copy()
        
        for scale in scales:
            # 缩放图像
            width = int(std_img.shape[1] * scale)
            height = int(std_img.shape[0] * scale)
            std_scaled = cv2.resize(std_img, (width, height))
            test_scaled = cv2.resize(test_img, (width, height))
            
            # 特征检测和匹配
            if reg_method == "ORB":
                detector = cv2.ORB_create(nfeatures=self.orb_nfeatures.get())
            else:  # SIFT
                detector = cv2.SIFT_create(nfeatures=self.sift_nfeatures.get())
                
            # 提取特征点和描述符
            kp1, des1 = detector.detectAndCompute(std_scaled, None)
            kp2, des2 = detector.detectAndCompute(test_scaled, None)
            
            if des1 is None or des2 is None:
                continue
                
            # 特征匹配
            if reg_method == "ORB":
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
            else:
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                # 应用比率测试
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                matches = good_matches
                
            if len(matches) < 4:
                continue
                
            # 提取匹配点对
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                continue
                
            # 统计内点数量
            inliers = np.sum(mask)
            if inliers > max_inliers:
                max_inliers = inliers
                best_homography = H / scale  # 调整回原始尺度
                best_matches = matches
        
        if best_homography is None:
            raise ValueError("未能在任何尺度下找到有效的变换")
        
        # 应用最佳变换矩阵
        h, w = self.std_img.shape[:2]
        aligned_img = cv2.warpPerspective(self.cur_test_img, best_homography, (w, h))
        
        self.add_history(f"完成多尺度特征匹配对齐: 使用{reg_method}方法, 最佳匹配数={len(best_matches)}")
        
        return aligned_img


    def perform_local_alignment(self, reg_method):
        """
        执行局部分块对齐
        
        参数:
            reg_method: 特征检测方法
            
        返回:
            aligned_img: 对齐后的图像
        """
        try:
            # 使用类的成员变量作为输入图像
            std_img = self.std_img
            test_img = self.cur_test_img
            
            # 图像分块大小
            block_size = 200
            overlap = 50  # 重叠区域大小
            
            h, w = std_img.shape[:2]
            aligned_img = np.zeros_like(test_img)
            
            # 计算分块数量
            num_blocks_h = (h - overlap) // (block_size - overlap)
            num_blocks_w = (w - overlap) // (block_size - overlap)
            
            self.add_history(f"执行局部分块对齐: {num_blocks_h}x{num_blocks_w} 个分块")
            
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # 计算当前块的坐标
                    y1 = i * (block_size - overlap)
                    x1 = j * (block_size - overlap)
                    y2 = min(y1 + block_size, h)
                    x2 = min(x1 + block_size, w)
                    
                    # 提取当前块
                    std_block = std_img[y1:y2, x1:x2]
                    test_block = test_img[y1:y2, x1:x2]
                    
                    # 对当前块进行特征匹配和对齐
                    if reg_method.lower() == "sift":
                        sift = cv2.SIFT_create()
                        kp1, des1 = sift.detectAndCompute(std_block, None)
                        kp2, des2 = sift.detectAndCompute(test_block, None)
                    else:  # 默认使用ORB
                        orb = cv2.ORB_create()
                        kp1, des1 = orb.detectAndCompute(std_block, None)
                        kp2, des2 = orb.detectAndCompute(test_block, None)
                    
                    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                        # 特征匹配
                        bf = cv2.BFMatcher(cv2.NORM_L2 if reg_method.lower() == "sift" else cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        
                        if len(matches) >= 4:
                            # 获取匹配点坐标
                            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                            
                            # 计算变换矩阵
                            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                            
                            if M is not None:
                                # 对当前块进行变换
                                aligned_block = cv2.warpPerspective(test_block, M, (block_size, block_size))
                                
                                # 将对齐后的块放回结果图像
                                if overlap > 0:
                                    # 使用加权平均处理重叠区域
                                    mask = np.ones_like(aligned_block) * 255
                                    if i > 0:  # 上边重叠
                                        mask[:overlap, :] = np.linspace(0, 255, overlap)[:, None]
                                    if j > 0:  # 左边重叠
                                        mask[:, :overlap] *= np.linspace(0, 255, overlap)[None, :]
                                    
                                    # 应用加权平均
                                    aligned_img[y1:y2, x1:x2] = cv2.addWeighted(
                                        aligned_img[y1:y2, x1:x2], 
                                        1 - mask/255, 
                                        aligned_block, 
                                        mask/255, 
                                        0
                                    )
                                else:
                                    aligned_img[y1:y2, x1:x2] = aligned_block
                            else:
                                # 如果变换失败，使用原始块
                                aligned_img[y1:y2, x1:x2] = test_block
                        else:
                            # 如果匹配点不足，使用原始块
                            aligned_img[y1:y2, x1:x2] = test_block
                    else:
                        # 如果特征提取失败，使用原始块
                        aligned_img[y1:y2, x1:x2] = test_block
            
            self.add_history("局部分块对齐完成")
            return aligned_img
            
        except Exception as e:
            self.add_history(f"局部分块对齐失败: {str(e)}")
            return test_img

    ###########################################
    # 右侧第二功能区
    ###########################################

    def build_right2_panel(self):
        """构建右侧第二个面板"""
        self.right2_frame = tk.Frame(self.tab1, bg="#d8d8d8", width=250)
        self.right2_frame.grid(row=1, column=5, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.right2_frame.grid_propagate(False)
        
        # 连通域分析部分
        tk.Label(self.right2_frame, text="最小连通域面积:", bg="#e8e8e8").pack(pady=2)
        self.min_area_var = tk.IntVar(value=50)
        tk.Scale(self.right2_frame, from_=1, to=500, orient="horizontal",
                variable=self.min_area_var).pack(fill="x", padx=5)
        tk.Button(self.right2_frame, text="执行连通域分析", command=self.on_conn_components).pack(pady=5)


        # PCB缺陷检测部分
        tk.Label(self.right2_frame, text="PCB缺陷检测", bg="#e8e8e8", font=("Arial", 10, "bold")).pack(pady=5)
        
        # 检测方法选择
        tk.Label(self.right2_frame, text="检测方法:", bg="#e8e8e8").pack(pady=2)
        self.defect_method_var = tk.StringVar(value="深度学习")
        tk.OptionMenu(self.right2_frame, self.defect_method_var, 
                    "传统方法", "深度学习", "两种方法结合").pack(fill="x", padx=5)
        
        # 置信度阈值设置
        tk.Label(self.right2_frame, text="置信度阈值:", bg="#e8e8e8").pack(pady=2)
        self.confidence_threshold_var = tk.DoubleVar(value=0.25)
        tk.Scale(self.right2_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                variable=self.confidence_threshold_var).pack(fill="x", padx=5)
        
        # 人工复检阈值设置
        tk.Label(self.right2_frame, text="人工复检阈值:", bg="#e8e8e8").pack(pady=2)
        self.review_threshold_var = tk.DoubleVar(value=0.6)
        tk.Scale(self.right2_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                variable=self.review_threshold_var).pack(fill="x", padx=5)
        
        # YOLOv8模型路径
        tk.Label(self.right2_frame, text="模型路径:", bg="#e8e8e8").pack(pady=2)
        self.model_path_var = tk.StringVar(value="D:/030923/Model/best.pt")
        model_path_entry = tk.Entry(self.right2_frame, textvariable=self.model_path_var)
        model_path_entry.pack(fill="x", padx=5, pady=2)
        tk.Button(self.right2_frame, text="浏览", 
                command=self.browse_model_path).pack(pady=2)
        
        # 检测按钮
        tk.Button(self.right2_frame, text="执行缺陷检测", 
                command=self.run_defect_detection).pack(pady=5)
        
        # 批量检测按钮
        tk.Button(self.right2_frame, text="批量缺陷检测", 
                command=self.batch_defect_detection).pack(pady=5)
        
        # 人工复检按钮
        tk.Button(self.right2_frame, text="人工复检", 
                command=self.open_manual_review).pack(pady=5)
        # 添加统计分析按钮
        tk.Button(self.right2_frame, text="检测结果统计分析", 
                command=self.show_statistics_analysis).pack(pady=5)
        # 添加检测方法对比按钮
        tk.Button(self.right2_frame, text="检测方法对比", 
                command=self.compare_detection_methods).pack(pady=5)           
        # 导出报告按钮
        tk.Button(self.right2_frame, text="导出检测报告", 
                command=self.export_detection_report).pack(pady=5)
        # 添加数据集扩增按钮
        tk.Button(self.right2_frame, text="数据集扩增", 
                command=self.augment_dataset).pack(pady=5)
        # 添加删除数据按钮
        tk.Button(self.right2_frame, text="删除图像与标注", 
                command=self.delete_dataset).pack(fill="x", padx=5, pady=5)


    def on_conn_components(self):

        target = self.target_var.get()
        if target == "标准图":
            base_img = self.std_img
        elif target == "待测图":
            base_img = self.cur_test_img
        else:
            base_img = self.result_img
        if base_img is None:
            messagebox.showwarning("提示", f"{target}为空")
            return

        method = self.thresh_method_var.get()
        manual_val = self.manual_thresh_val.get()
        blockSize = self.adapt_blockSize.get()
        C = self.adapt_C.get()

        bin_img = ip.threshold_segment(base_img, method=method, val=manual_val,
                                       blockSize=blockSize, C=C)

        try:
            min_area = self.min_area_var.get()
            result = ip.analyze_components(bin_img, min_area=min_area)
            if isinstance(result, tuple) and len(result) == 2:
                marked_img, components_info = result
            else:
                raise ValueError("analyze_components 返回值格式不正确")

            self.canvas_result.set_image(marked_img)

            self.add_history(f"=== {target} 连通域分析结果 ===")
            self.add_history(f"检测到 {len(components_info)} 个连通域（最小面积={min_area}）")
            for comp in components_info:
                self.add_history(f"连通域 #{comp['id']}: 面积={comp['area']:.1f}, "
                                 f"周长={comp['perimeter']:.1f}, "
                                 f"质心=({comp['centroid'][0]:.1f}, {comp['centroid'][1]:.1f})")
            self.add_history("=== 分析结束 ===")
        except Exception as e:
            messagebox.showerror("错误", f"连通域分析失败: {str(e)}")

    # 添加统计分析方法
    def show_statistics_analysis(self):
        """显示检测结果的统计分析"""
        # 检查是否有检测结果
        if not hasattr(self, 'current_defect_info') or not self.current_defect_info:
            # 尝试查找最近的批量检测结果
            report_path = filedialog.askopenfilename(
                title="选择检测报告CSV文件",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
            )
            if not report_path:
                messagebox.showwarning("提示", "请先执行缺陷检测或选择检测报告文件")
                return
                
            # 创建统计分析器
            analyzer = StatisticsAnalyzer()
            
            # 加载CSV文件
            if analyzer.load_batch_results(report_path):
                # 显示分析窗口
                analyzer.show_analysis_window(self.root)
                self.add_history(f"已加载检测报告进行统计分析: {report_path}")
            else:
                messagebox.showerror("错误", "无法加载检测报告文件")
        else:
            # 使用当前检测结果
            analyzer = StatisticsAnalyzer()
            analyzer.load_single_result(self.current_defect_info)
            analyzer.show_analysis_window(self.root)
            self.add_history("已对当前检测结果进行统计分析")

    # 改进导出检测报告方法
    def export_detection_report(self):
        """导出检测报告"""
        if not hasattr(self, 'current_defect_info') or not self.current_defect_info:
            messagebox.showwarning("提示", "请先执行缺陷检测")
            return
            
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存检测报告",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # 准备报告数据
            data = []
            for i, defect in enumerate(self.current_defect_info):
                row = {
                    "ID": i+1,
                    "类型": defect.get('type', 'unknown'),
                    "置信度": defect.get('confidence', 0),
                    "X坐标": defect.get('x', 0),
                    "Y坐标": defect.get('y', 0),
                    "宽度": defect.get('w', 0),
                    "高度": defect.get('h', 0),
                    "检测时间": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                data.append(row)
                
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 根据文件扩展名保存
            if file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                
            self.add_history(f"检测报告已导出到: {file_path}")
            messagebox.showinfo("成功", f"检测报告已导出到: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出报告失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    # 添加一个方法来保存当前检测结果
    def save_current_detection_result(self):
        """保存当前检测结果"""
        if not hasattr(self, 'current_defect_info') or not self.current_defect_info:
            messagebox.showwarning("提示", "没有可保存的检测结果")
            return
            
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="保存检测结果",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # 准备保存数据
            save_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "defects": self.current_defect_info,
                "method": self.defect_method_var.get() if hasattr(self, 'defect_method_var') else "未知",
                "confidence_threshold": self.confidence_threshold_var.get() if hasattr(self, 'confidence_threshold_var') else 0.25
            }
            
            # 保存为JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
            self.add_history(f"检测结果已保存到: {file_path}")
            messagebox.showinfo("成功", f"检测结果已保存到: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存结果失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def browse_model_path(self):
        """浏览选择YOLOv8模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择YOLOv8模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.add_history(f"已选择模型文件: {file_path}")

    def run_defect_detection(self):
        """执行缺陷检测"""
        try:
            # 获取检测方法
            method = self.defect_method_var.get()
            self.add_history(f"使用{method}进行检测...")

            # 检查输入图像 - 支持选区模式
            if self.selection_mode and self.selected_subroi is not None:
                # 使用选区进行检测
                if hasattr(self, "original_roi_color") and self.original_roi_color is not None:
                    input_img = self.original_roi_color.copy()
                else:
                    input_img = self.selected_subroi.copy()
                self.add_history("使用选区进行缺陷检测")
            else:
                # 使用整图进行检测
                if self.cur_test_img is None:
                    messagebox.showwarning("提示", "请先选择待测图")
                    return
                input_img = self.cur_test_img.copy()

            # 根据不同方法执行检测
            if method == "传统方法":
                # 检查是否有标准图
                if self.std_img is None:
                    messagebox.showwarning("提示", "传统方法需要先加载标准图")
                    return
                    
                # 执行传统方法检测
                result_img, defect_info = self.traditional_defect_detection()
                
                if result_img is not None and defect_info:
                    self.result_img = result_img
                    self.current_defect_info = defect_info
                    
            elif method == "深度学习":
                # 获取模型路径并检查
                model_path = self.model_path_var.get().replace('/', '\\')  # 增加路径格式转换
                default_path = "D:\\030923\\Model\\best.pt"
                
                # 添加调试日志
                self.add_history(f"[DEBUG] 尝试加载模型路径: {model_path}")
                self.add_history(f"[DEBUG] 默认模型路径: {default_path}")
                # 如果当前路径不存在，尝试使用默认路径
                if not os.path.exists(model_path):
                    self.add_history(f"指定的模型文件不存在: {model_path}")
                    if os.path.exists(default_path):
                        model_path = default_path
                        self.model_path_var.set(default_path)
                        self.add_history(f"自动切换默认模型路径: {default_path}")
                    else:
                        # 添加更详细的错误信息
                        error_msg = (
                            f"模型文件检查失败\n"
                            f"当前路径: {os.path.abspath(model_path)}\n"
                            f"默认路径: {os.path.abspath(default_path)}\n"
                            f"工作目录: {os.getcwd()}"
                        )
                        messagebox.showerror("路径错误", error_msg)
                        return
                    
                # 获取参数
                conf_threshold = self.confidence_threshold_var.get()
                iou_threshold = 0.45
                
                # 加载模型
                self.add_history("正在加载深度学习模型...")
                model = YOLO(model_path)
                
                # 检查并转换图像格式
                if len(input_img.shape) == 2:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
                # 开始计时
                start_time = time.time()
                # 执行检测
                results = model(input_img, conf=conf_threshold, iou=iou_threshold)
                result = results[0]
                # 结束计时
                end_time = time.time()
                detection_time = end_time - start_time
                
                # 输出检测时间
                self.add_history(f"检测耗时: {detection_time:.3f} 秒")
                
                # 处理结果
                self.result_img = result.plot()
                boxes = result.boxes
                defect_info = []
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    defect_info.append({
                        "type": name,
                        "confidence": conf,
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1
                    })
                
                self.current_defect_info = defect_info
                
            else:  # 两种方法结合
                if self.std_img is None:
                    messagebox.showwarning("提示", "结合方法需要先加载标准图")
                    return
                    
                # 1. 先执行传统方法的图像差分和连通域分析
                if len(self.cur_test_img.shape) == 3:
                    test_gray = cv2.cvtColor(self.cur_test_img, cv2.COLOR_BGR2GRAY)
                else:
                    test_gray = self.cur_test_img.copy()
                    
                if len(self.std_img.shape) == 3:
                    std_gray = cv2.cvtColor(self.std_img, cv2.COLOR_BGR2GRAY)
                else:
                    std_gray = self.std_img.copy()

                # 使用统一的图像对齐方法
                aligned_test_img = self.align_images(self.std_img, self.cur_test_img)
                if aligned_test_img is None:
                    aligned_test_img = self.cur_test_img.copy()
                    self.add_history("警告: 图像对齐失败，使用原始图像")
                                # 转换为灰度图进行处理
                if len(aligned_test_img.shape) == 3:
                    aligned_test = cv2.cvtColor(aligned_test_img, cv2.COLOR_BGR2GRAY)
                else:
                    aligned_test = aligned_test_img.copy()
            

                # 图像差分
                diff = cv2.absdiff(std_gray, aligned_test)
                
                # 阈值分割
                thresh_method = self.thresh_method_var.get()
                if thresh_method == "manual":
                    thresh_val = self.manual_thresh_val.get()
                    _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
                else:
                    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 形态学操作
                kernel = np.ones((3, 3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                # 2. 连通域分析获取ROI区域
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = self.min_area_var.get()
                
                # 3. 在ROI区域上执行深度学习检测
                model_path = self.model_path_var.get()
                model = YOLO(model_path)
                conf_threshold = self.confidence_threshold_var.get()
                
                result_img = self.cur_test_img.copy()
                if len(result_img.shape) == 2:
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2BGR)
                
                combined_defects = []
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                        
                    # 获取ROI区域
                    x, y, w, h = cv2.boundingRect(contour)
                    # 扩大ROI区域以包含更多上下文
                    padding = 10
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(result_img.shape[1], x + w + padding)
                    y2 = min(result_img.shape[0], y + h + padding)
                    
                    roi = result_img[y1:y2, x1:x2]
                    
                    # 在ROI区域上运行深度学习检测
                    results = model(roi, conf=conf_threshold)
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
                
                # 4. 在结果图像上标注缺陷
                if combined_defects:
                    # 使用PIL绘制中文
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(result_img_rgb)
                    draw = ImageDraw.Draw(img_pil)
                    font_path = "C:/Windows/Fonts/simhei.ttf"
                    font = ImageFont.truetype(font_path, 20)
                    
                    for i, defect in enumerate(combined_defects):
                        x, y = defect['x'], defect['y']
                        w, h = defect['w'], defect['h']
                        # 先在原始图像上画矩形
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        # 在PIL图像上添加文本
                        text = f"{defect['type']} ({defect['confidence']:.2f})"
                        draw.text((x, max(0, y-25)), text, font=font, fill=(255, 0, 0))
                    
                    # 转换回OpenCV格式
                    self.result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 显示结果
            if self.selection_mode and self.selected_subroi is not None:
                self.selected_subroi = self.result_img
                self.canvas_result.set_image(self.selected_subroi)
            else:
                self.canvas_result.set_image(self.result_img)

            # 检查是否需要人工复检
            if hasattr(self, 'current_defect_info') and self.current_defect_info:
                self.add_history(f"检测完成，共发现 {len(self.current_defect_info)} 个缺陷")
                need_review = False
                for defect in self.current_defect_info:
                    if defect.get('confidence', 1.0) < self.review_threshold_var.get():
                        need_review = True
                        break
                
                if need_review:
                    if messagebox.askyesno("提示", "检测到低置信度缺陷，是否进行人工复检？"):
                        self.open_manual_review()
            else:
                self.add_history("未检测到任何缺陷")
                messagebox.showinfo("提示", "未检测到缺陷")
                
        except Exception as e:
            self.add_history(f"缺陷检测出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
            messagebox.showerror("错误", f"缺陷检测失败: {str(e)}")

    def align_images(self, std_img, test_img):
        """统一的图像对齐方法，整合所有对齐逻辑"""
        self.add_history("\n=== 开始图像对齐 ===")
        
        # 获取配准方法参数
        reg_method = self.reg_method_var.get()
        align_method = self.align_method_var.get() if hasattr(self, "align_method_var") else "透视变换"
        
        self.add_history(f"配准方法: {reg_method}")
        self.add_history(f"对齐方法: {align_method}")
        self.add_history(f"输入图像尺寸 - 标准图: {std_img.shape}, 待测图: {test_img.shape}")
        
        if reg_method == "none":
            self.add_history("不进行图像对齐")
            return test_img.copy()
        else:
            # 根据选择的对齐方法执行不同的对齐算法
            try:
                # 添加对齐方法的详细参数信息
                if align_method == "透视变换":
                    self.add_history("\n透视变换参数:")
                    if reg_method == "sift":
                        nfeatures = self.sift_nfeatures.get() if hasattr(self, 'sift_nfeatures') else 500
                        self.add_history(f"- SIFT特征点数量: {nfeatures}")
                    elif reg_method == "orb":
                        self.add_history("- 使用ORB特征检测器")
                    elif reg_method == "ecc":
                        self.add_history("- 使用ECC配准算法")
                    
                elif align_method == "薄板样条变换":
                    self.add_history("\n薄板样条变换参数:")
                    self.add_history("- 控制点数量: 自适应")
                    self.add_history("- 正则化参数: 0.0")
                    
                elif align_method == "多尺度特征匹配":
                    self.add_history("\n多尺度特征匹配参数:")
                    self.add_history("- 尺度层数: 3")
                    self.add_history("- 每层特征点数: 1000")
                    
                elif align_method == "局部分块对齐":
                    self.add_history("\n局部分块对齐参数:")
                    self.add_history("- 分块大小: 200x200")
                    self.add_history("- 重叠率: 0.2")

                # 执行对齐操作
                start_time = time.time()
                
                # ... 原有的对齐方法执行代码 ...
                
                # 添加执行时间和内存使用信息
                end_time = time.time()
                self.add_history(f"\n对齐耗时: {end_time - start_time:.2f}秒")
                
                # 添加对齐结果的详细评估
                if aligned_img is not None:
                    h1, w1 = test_img.shape[:2]
                    h2, w2 = aligned_img.shape[:2]
                    self.add_history(f"\n对齐结果评估:")
                    self.add_history(f"- 原始图像尺寸: {w1}x{h1}")
                    self.add_history(f"- 对齐后尺寸: {w2}x{h2}")
                    
                    # 计算对齐前后的差异
                    if len(aligned_img.shape) == len(std_img.shape):
                        if len(aligned_img.shape) == 3:
                            aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
                            std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY) if len(std_img.shape) == 3 else std_img
                        else:
                            aligned_gray = aligned_img
                            std_gray = std_img
                        
                        # 计算多个评估指标
                        before_diff = cv2.absdiff(std_gray, cv2.resize(test_img, (std_gray.shape[1], std_gray.shape[0])) if len(test_img.shape) == 2 else cv2.cvtColor(cv2.resize(test_img, (std_gray.shape[1], std_gray.shape[0])), cv2.COLOR_BGR2GRAY))
                        after_diff = cv2.absdiff(std_gray, aligned_gray)
                        
                        # 平均差异分数
                        before_score = np.mean(before_diff)
                        after_score = np.mean(after_diff)
                        
                        # 结构相似性指标 (SSIM)
                        before_ssim = cv2.compareSSIM(std_gray, cv2.resize(test_img, (std_gray.shape[1], std_gray.shape[0])) if len(test_img.shape) == 2 else cv2.cvtColor(cv2.resize(test_img, (std_gray.shape[1], std_gray.shape[0])), cv2.COLOR_BGR2GRAY))
                        after_ssim = cv2.compareSSIM(std_gray, aligned_gray)
                        
                        self.add_history(f"\n定量评估指标:")
                        self.add_history(f"- 对齐前平均差异: {before_score:.2f}")
                        self.add_history(f"- 对齐后平均差异: {after_score:.2f}")
                        self.add_history(f"- 差异改善率: {(before_score - after_score) / before_score * 100:.2f}%")
                        self.add_history(f"- 对齐前SSIM: {before_ssim:.4f}")
                        self.add_history(f"- 对齐后SSIM: {after_ssim:.4f}")
                        self.add_history(f"- SSIM改善率: {(after_ssim - before_ssim) / before_ssim * 100:.2f}%")
                
                return aligned_img
                
            except Exception as e:
                self.add_history(f"\n图像对齐失败: {str(e)}")
                import traceback
                self.add_history(f"错误详情: {traceback.format_exc()}")
                return test_img.copy()


    def save_detection_results(self):
        """保存当前检测结果"""
        if not hasattr(self, "current_defect_info") or not self.current_defect_info:
            messagebox.showwarning("提示", "没有可保存的检测结果")
            return
            
        if self.result_img is None:
            messagebox.showwarning("提示", "没有检测结果图像")
            return
            
        # 保存结果图像
        img_filename = filedialog.asksaveasfilename(
            title="保存结果图像",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPG文件", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if img_filename:
            cv2.imwrite(img_filename, cv2.cvtColor(self.result_img, cv2.COLOR_RGB2BGR))
            self.add_history(f"结果图像已保存至: {img_filename}")
            
        # 保存缺陷信息
        info_filename = filedialog.asksaveasfilename(
            title="保存缺陷信息",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if info_filename:
            with open(info_filename, 'w', encoding='utf-8') as f:
                json.dump(self.current_defect_info, f, indent=4, ensure_ascii=False)
            self.add_history(f"缺陷信息已保存至: {info_filename}")

    def compare_detection_methods(self):
        """对比不同检测方法的结果"""
        if self.std_img is None or self.cur_test_img is None:
            messagebox.showwarning("提示", "请先加载标准图和待测图")
            return
            
        self.add_history("开始执行检测方法对比...")
        
        # 创建主对比窗口
        compare_window = tk.Toplevel(self.root)
        compare_window.title("检测方法对比")
        compare_window.geometry("1200x800")
        
        # 创建上下分栏
        top_frame = ttk.Frame(compare_window)
        top_frame.pack(fill="both", expand=True)
        
        middle_frame = ttk.Frame(compare_window)
        middle_frame.pack(fill="both", expand=True)
        
        bottom_frame = ttk.Frame(compare_window)
        bottom_frame.pack(fill="both", expand=True)
        
        # 配置网格权重，减少空白
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=1)
        
        # 创建中间过程图像列表
        process_images = []
        
        # 添加原始图像
        original_img = self.cur_test_img.copy()
        img_pil = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)  # 调小字体大小
        draw.text((10, 10), "原始图像", font=font, fill=(255, 0, 0))
        labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        process_images.append(labeled_img)
        
        # 添加图像差分结果
        if self.std_img is not None and self.cur_test_img is not None:
            # 确保两个图像尺寸和通道数一致
            if len(self.std_img.shape) == 3:
                std_proc = self.std_img.copy()
            else:
                std_proc = cv2.cvtColor(self.std_img.copy(), cv2.COLOR_GRAY2BGR)
                
            if len(self.cur_test_img.shape) == 3:
                test_proc = self.cur_test_img.copy()
            else:
                test_proc = cv2.cvtColor(self.cur_test_img.copy(), cv2.COLOR_GRAY2BGR)
            
            # 取尺寸交集
            min_h = min(std_proc.shape[0], test_proc.shape[0])
            min_w = min(std_proc.shape[1], test_proc.shape[1])
            
            std_proc = std_proc[:min_h, :min_w]
            test_proc = test_proc[:min_h, :min_w]
            
            # 保存当前状态
            original_std_img = self.std_img
            original_cur_test_img = self.cur_test_img
            original_result_img = self.result_img
            original_aligned_test_img = self.aligned_test_img if hasattr(self, "aligned_test_img") else None
            
            # 设置当前处理的图像
            self.std_img = std_proc
            self.cur_test_img = test_proc
            if hasattr(self, "aligned_test_img"):
                self.aligned_test_img = None
            
            # 调用阈值差分方法
            self.on_thresh_difference()
            
            # 获取差分结果
            if self.result_img is not None:
                diff_img = self.result_img.copy()  # 保存差分结果供后续使用
                img_pil = Image.fromarray(cv2.cvtColor(self.result_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10), "阈值差分", font=font, fill=(255, 0, 0))
                labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                process_images.append(labeled_img)
            
            # 恢复原始状态
            self.std_img = original_std_img
            self.cur_test_img = original_cur_test_img
            self.result_img = original_result_img
            if original_aligned_test_img is not None:
                self.aligned_test_img = original_aligned_test_img
            
            # 添加形态学操作后的结果
            # 转为灰度图
            diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
            # 二值化
            _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # 转为彩色以便添加文字
            morph_color = cv2.cvtColor(morph_img, cv2.COLOR_GRAY2BGR)
            img_pil = Image.fromarray(cv2.cvtColor(morph_color, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), "形态学处理", font=font, fill=(255, 0, 0))
            labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            process_images.append(labeled_img)
            
            # 添加感兴趣区域
            roi_img = test_proc.copy()
            # 找到轮廓
            contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 绘制轮廓
            cv2.drawContours(roi_img, contours, -1, (0, 255, 0), 2)
            img_pil = Image.fromarray(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), "感兴趣区域", font=font, fill=(255, 0, 0))
            labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            process_images.append(labeled_img)
        
        # 执行三种方法的检测
        # 传统方法
        traditional_img, traditional_defects = self.traditional_defect_detection()
        if traditional_img is not None:
            # 添加中文标签
            img_pil = Image.fromarray(cv2.cvtColor(traditional_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), "传统方法检测结果", font=font, fill=(255, 0, 0))
            labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            process_images.append(labeled_img)
        
        # 深度学习方法
        dl_defects = self.dl_defect_detection()
        dl_img = self.result_img.copy() if self.result_img is not None else None
        if dl_img is not None:
            img_pil = Image.fromarray(cv2.cvtColor(dl_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), "深度学习方法检测结果", font=font, fill=(255, 0, 0))
            labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            process_images.append(labeled_img)
        
        # 结合方法
        combined_img = None  # 添加初始化
        combined_defects = self.combined_defect_detection()
        
        if combined_defects:
            # 在原图上绘制结合后的检测结果
            combined_img = self.cur_test_img.copy()
            img_pil = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            for defect in combined_defects:
                # 绘制矩形框
                draw.rectangle([
                    (defect['x'], defect['y']),
                    (defect['x'] + defect['w'], defect['y'] + defect['h'])
                ], outline=(0, 255, 0), width=2)
                
                # 添加中文标签 - 调整位置和字体大小
                label_text = f"{defect['type']}"
                # 确保标签在图像内
                text_y = max(defect['y'] - 20, 5)
                draw.text((defect['x'], text_y), label_text, 
                        font=font, fill=(0, 255, 0))
            
            labeled_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            process_images.append(labeled_img)
            combined_img = labeled_img.copy()
        
        # 将所有中间过程图像水平拼接并显示在中间frame上
        if process_images:
            # 调整所有图像到相同高度
            max_height = max(img.shape[0] for img in process_images)
            resized_images = []
            for img in process_images:
                aspect_ratio = img.shape[1] / img.shape[0]
                new_width = int(max_height * aspect_ratio)
                resized = cv2.resize(img, (new_width, max_height))
                resized_images.append(resized)
            
            # 水平拼接
            process_display = np.hstack(resized_images)
            
            # 显示在中间frame上
            middle_canvas = ZoomableCanvas(middle_frame, width=1100, height=300)
            middle_canvas.pack(fill="both", expand=True, padx=5, pady=5)
            middle_canvas.set_image(process_display)
            
            # 保存结果图像供后续使用
            self.result_img = process_display.copy()
        
        # 创建结果字典
        results_dict = {}
        
        if traditional_img is not None and traditional_defects:
            results_dict["传统方法"] = {
                "image": traditional_img,
                "defects": traditional_defects
            }
            
        if dl_img is not None and dl_defects:
            results_dict["深度学习方法"] = {
                "image": dl_img,
                "defects": dl_defects
            }
            
        if combined_img is not None and combined_defects:
            results_dict["结合方法"] = {
                "image": combined_img,
                "defects": combined_defects
            }
        
        # 如果没有任何有效结果
        if not results_dict:
            messagebox.showinfo("提示", "所有方法均未检测到缺陷")
            return
        
        # 在上部显示图像结果
        canvas1 = ZoomableCanvas(top_frame, width=380, height=300)
        canvas1.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        
        canvas2 = ZoomableCanvas(top_frame, width=380, height=300)
        canvas2.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
        
        canvas3 = ZoomableCanvas(top_frame, width=380, height=300)
        canvas3.grid(row=0, column=2, padx=2, pady=2, sticky="nsew")
        
        # 添加标签并显示图像
        if "传统方法" in results_dict:
            canvas1.set_image(results_dict["传统方法"]["image"])
            
        if "深度学习方法" in results_dict:
            canvas2.set_image(results_dict["深度学习方法"]["image"])
            
        if "结合方法" in results_dict:
            canvas3.set_image(results_dict["结合方法"]["image"])
        
        # 在下部显示统计信息
        def create_stats_frame(parent, title, defects):
            frame = ttk.LabelFrame(parent, text=title)
            frame.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            
            # 创建Text控件显示统计信息
            text = tk.Text(frame, height=15, width=40)
            text.pack(fill="both", expand=True, padx=2, pady=2)
            
            # 添加统计信息
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
            if defects and "confidence" in defects[0]:
                confidences = [d.get("confidence", 0) for d in defects]
                avg_conf = sum(confidences) / len(confidences)
                text.insert("end", f"\n平均置信度: {avg_conf:.4f}")
            
            text.config(state="disabled")
        
        # 创建三个统计信息窗口
        if "传统方法" in results_dict:
            create_stats_frame(bottom_frame, "传统方法统计", results_dict["传统方法"]["defects"])
        
        if "深度学习方法" in results_dict:
            create_stats_frame(bottom_frame, "深度学习方法统计", results_dict["深度学习方法"]["defects"])
        
        if "结合方法" in results_dict:
            create_stats_frame(bottom_frame, "结合方法统计", results_dict["结合方法"]["defects"])
        
        # 统计分析
        self.add_history("\n检测方法对比统计:")
        
        # 统计各方法检测到的缺陷数量
        for method, result in results_dict.items():
            defects = result["defects"]
            self.add_history(f"{method}: 检测到 {len(defects)} 个缺陷")
            
            # 统计缺陷类型分布
            type_counts = {}
            for defect in defects:
                defect_type = defect["type"]
                if defect_type in type_counts:
                    type_counts[defect_type] += 1
                else:
                    type_counts[defect_type] = 1
            
            # 输出类型分布
            self.add_history(f"  缺陷类型分布:")
            for t, count in type_counts.items():
                percentage = count / len(defects) * 100
                self.add_history(f"    - {t}: {count}个 ({percentage:.1f}%)")
            
            # 计算平均置信度
            if "confidence" in defects[0]:
                confidences = [d.get("confidence", 0) for d in defects]
                avg_conf = sum(confidences) / len(confidences)
                self.add_history(f"  平均置信度: {avg_conf:.4f}")
        
        # 创建详细的统计分析报告
        if len(results_dict) > 1:
            self.add_history("\n创建详细统计分析报告...")
            
            # 创建统计分析器实例
            analyzers = {}
            for method, result in results_dict.items():
                analyzer = StatisticsAnalyzer()
                analyzer.load_single_result(result["defects"])
                analyzers[method] = analyzer
            
        # 如果有多个方法，进行方法间比较
            if len(analyzers) >= 2:
                methods = list(analyzers.keys())
                
                # 比较第一个和第二个方法
                self.add_history(f"比较 {methods[0]} 和 {methods[1]}")
                analyzers[methods[0]].compare_with(analyzers[methods[1]])
                
                # 如果有第三个方法，也进行比较
                if len(methods) >= 3:
                    self.add_history(f"比较 {methods[0]} 和 {methods[2]}")
                    analyzers[methods[0]].compare_with(analyzers[methods[2]])
                    
                    self.add_history(f"比较 {methods[1]} 和 {methods[2]}")
                    analyzers[methods[1]].compare_with(analyzers[methods[2]])
        
        self.add_history("检测方法对比完成")
        
        # 添加导出报告按钮
        export_button = ttk.Button(compare_window, text="导出对比报告", 
                                command=lambda: self.export_comparison_report(results_dict))
        export_button.pack(pady=10)

    def export_comparison_report(self, results_dict):
        """导出检测方法对比报告"""
        try:
            # 选择保存目录
            save_dir = filedialog.askdirectory(title="选择保存对比结果的目录")
            self.add_history(f"选择的保存目录: {save_dir}")
            
            if save_dir:
                # 创建保存目录
                report_dir = os.path.join(save_dir, f"comparison_{time.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(report_dir, exist_ok=True)
                self.add_history(f"创建报告目录: {report_dir}")
                
                # 保存各方法的结果图像
                for method, result in results_dict.items():
                    img_path = os.path.join(report_dir, f"{method}_result.png")
                    cv2.imwrite(img_path, result["image"])
                    
                    # 保存缺陷信息为JSON
                    json_path = os.path.join(report_dir, f"{method}_defects.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result["defects"], f, indent=4, ensure_ascii=False, default=self.numpy_json_encoder)
                
                # 创建对比HTML报告
                html_path = os.path.join(report_dir, "comparison_report.html")
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <title>PCB缺陷检测方法对比报告</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            .header {{ text-align: center; margin-bottom: 20px; }}
                            .method-section {{ margin-bottom: 30px; }}
                            .method-title {{ background-color: #f2f2f2; padding: 10px; }}
                            .result-image {{ text-align: center; margin: 20px 0; }}
                            .result-image img {{ max-width: 100%; }}
                            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            tr:nth-child(even) {{ background-color: #f9f9f9; }}
                            .comparison {{ margin-top: 30px; }}
                            .chart {{ margin: 20px 0; }}
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>PCB缺陷检测方法对比报告</h1>
                            <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </div>
                    """)
                    
                    # 添加各方法的结果
                    for method, result in results_dict.items():
                        defects = result["defects"]
                        
                        f.write(f"""
                        <div class="method-section">
                            <h2 class="method-title">{method}</h2>
                            
                            <div class="result-image">
                                <h3>检测结果图像</h3>
                                <img src="{method}_result.png" alt="{method}检测结果">
                            </div>
                            
                            <h3>检测到的缺陷 ({len(defects)}个)</h3>
                            <table>
                                <tr>
                                    <th>序号</th>
                                    <th>缺陷类型</th>
                                    <th>置信度</th>
                                    <th>位置</th>
                                    <th>尺寸</th>
                                </tr>
                        """)
                        
                        for i, defect in enumerate(defects):
                            f.write(f"""
                                <tr>
                                    <td>{i+1}</td>
                                    <td>{defect['type']}</td>
                                    <td>{defect.get('confidence', 'N/A'):.2f}</td>
                                    <td>({defect['x']}, {defect['y']})</td>
                                    <td>{defect['w']} x {defect['h']}</td>
                                </tr>
                            """)
                        
                        f.write("""
                            </table>
                        </div>
                        """)
                    
                    # 添加方法对比表格
                    f.write("""
                    <div class="comparison">
                        <h2>方法对比</h2>
                        <table>
                            <tr>
                                <th>方法</th>
                                <th>缺陷总数</th>
                                <th>平均置信度</th>
                            </tr>
                    """)
                    
                    for method, result in results_dict.items():
                        defects = result["defects"]
                        avg_conf = sum([d.get('confidence', 0) for d in defects]) / len(defects) if defects else 0
                        
                        f.write(f"""
                            <tr>
                                <td>{method}</td>
                                <td>{len(defects)}</td>
                                <td>{avg_conf:.4f}</td>
                            </tr>
                        """)
                    
                    f.write("""
                        </table>
                    </div>
                    """)
                    
                    f.write("""
                    </body>
                    </html>
                    """)
                
                # 打开报告
                import webbrowser
                webbrowser.open(f"file://{html_path}")
                
                self.add_history(f"对比报告已导出至: {report_dir}")
                messagebox.showinfo("导出成功", f"对比报告已导出至:\n{report_dir}")
                
        except Exception as e:
            self.add_history(f"导出对比报告出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
            messagebox.showerror("错误", f"导出对比报告失败: {str(e)}")



    def open_manual_review(self):
        """打开人工复检窗口"""
        if not hasattr(self, "current_defect_info") or not self.current_defect_info:
            messagebox.showwarning("提示", "没有可供复检的缺陷信息")
            return
        
        if self.result_img is None:
            messagebox.showwarning("提示", "没有检测结果图像")
            return
        
        # 创建复检窗口
        review_window = tk.Toplevel(self.root)
        review_window.title("人工复检")
        review_window.geometry("1200x800")
        
        # 创建上下分栏
        top_frame = ttk.Frame(review_window)
        top_frame.pack(side="top", fill="both", expand=True)
        
        bottom_frame = ttk.Frame(review_window)
        bottom_frame.pack(side="bottom", fill="x", pady=10)
        
        # 创建图像显示区域
        canvas_frame = ttk.Frame(top_frame)
        canvas_frame.pack(fill="both", expand=True)
        
        # 使用ZoomableCanvas显示图像
        canvas = ZoomableCanvas(canvas_frame, width=800, height=600)
        canvas.pack(fill="both", expand=True)
        
        # 显示图像
        canvas.set_image(self.result_img)
        
        # 显示缺陷信息
        info_frame = ttk.LabelFrame(top_frame, text="检测到的缺陷信息")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_text = tk.Text(info_frame, height=5, width=80)
        info_text.pack(fill="x", padx=5, pady=5)
        
        # 填充缺陷信息
        for i, defect in enumerate(self.current_defect_info):
            confidence = defect.get('confidence', 'N/A')
            if isinstance(confidence, (int, float)):
                confidence_str = f"{confidence:.2f}"
            else:
                confidence_str = str(confidence)
                
            info_text.insert("end", f"缺陷 #{i+1}: 类型={defect['type']}, 置信度={confidence_str}, "
                            f"位置=({defect['x']},{defect['y']})-({defect['x']+defect['w']},{defect['y']+defect['h']})\n")
        
        info_text.config(state="disabled")  # 设为只读
        
        # 添加检测合格和检测不合格按钮
        def on_pass():
            """检测合格按钮点击事件"""
            try:
                # 创建保存目录
                images_dir = r"D:\030923\dataset\images_0"
                json_dir = r"D:\030923\dataset\json"
                
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(json_dir, exist_ok=True)
                
                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_filename = f"pass_{timestamp}.jpg"
                json_filename = f"pass_{timestamp}.json"
                
                img_path = os.path.join(images_dir, img_filename)
                json_path = os.path.join(json_dir, json_filename)
                
                # 保存原始图像，而不是标注后的图像
                if hasattr(self, "cur_test_img") and self.cur_test_img is not None:
                    # 保存原始图像
                    cv2.imwrite(img_path, self.cur_test_img)
                else:
                    # 如果没有原始图像，则保存结果图像
                    cv2.imwrite(img_path, self.result_img)
                
                # 创建labelme格式的JSON
                json_data = self.create_labelme_json(img_filename, self.current_defect_info)
                
                # 保存JSON
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                self.add_history(f"人工复检合格: 图像已保存至 {img_path}")
                self.add_history(f"标注信息已保存至 {json_path}")
                
                # 关闭窗口
                review_window.destroy()
                
            except Exception as e:
                self.add_history(f"处理人工复检合格出错: {str(e)}")
                import traceback
                self.add_history(f"错误详情: {traceback.format_exc()}")
                messagebox.showerror("错误", f"保存失败: {str(e)}")
                
        def on_fail():
            """检测不合格按钮点击事件"""
            try:
                # 创建保存原始图像的目录
                images_dir = r"D:\030923\dataset\images_0"
                os.makedirs(images_dir, exist_ok=True)
                
                # 生成文件名
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_filename = f"review_{timestamp}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                
                # 保存原始图像，而不是标注后的图像
                if hasattr(self, "cur_test_img") and self.cur_test_img is not None:
                    # 保存原始图像到dataset目录
                    cv2.imwrite(img_path, self.cur_test_img)
                    self.add_history(f"原始图像已保存至: {img_path}")
                else:
                    # 如果没有原始图像，则保存结果图像
                    cv2.imwrite(img_path, self.result_img)
                    self.add_history(f"结果图像已保存至: {img_path}")
                
                # 关闭窗口
                review_window.destroy()
                
                # 启动labelme
                self.add_history(f"启动labelme进行人工标注: {img_path}")
                
                # 使用subprocess启动labelme
                try:
                    import subprocess
                    # 检查labelme是否在PATH中
                    labelme_path = shutil.which("labelme")
                    
                    if labelme_path:
                        # 直接使用找到的labelme路径
                        cmd = f'"{labelme_path}" "{img_path}"'
                    else:
                        # 尝试使用Python模块方式启动
                        cmd = f'"{sys.executable}" -m labelme "{img_path}"'
                    
                    self.add_history(f"执行命令: {cmd}")
                    subprocess.Popen(cmd, shell=True)
                    
                except Exception as e:
                    self.add_history(f"启动labelme失败: {str(e)}")
                    messagebox.showerror("错误", f"启动labelme失败: {str(e)}")
                    
                    # 提供备用方案
                    fallback_msg = f"请手动打开labelme并加载图像: {img_path}"
                    self.add_history(fallback_msg)
                    messagebox.showinfo("备用方案", fallback_msg)
                
            except Exception as e:
                self.add_history(f"处理人工复检不合格出错: {str(e)}")
                import traceback
                self.add_history(f"错误详情: {traceback.format_exc()}")
                messagebox.showerror("错误", f"处理失败: {str(e)}")

        # 创建按钮
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="检测合格", width=20, command=on_pass).pack(side="left", padx=20)
        ttk.Button(btn_frame, text="检测不合格", width=20, command=on_fail).pack(side="left", padx=20)
    
    def create_labelme_json(self, img_filename, defects):
        """创建labelme格式的JSON数据"""
        # 获取图像尺寸
        if self.result_img is not None:
            h, w = self.result_img.shape[:2]
        else:
            h, w = 600, 800  # 默认尺寸
        
        # 创建基本JSON结构
        json_data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": img_filename,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }
        
        # 添加检测到的缺陷
        for defect in defects:
            x, y = defect['x'], defect['y']
            width, height = defect['w'], defect['h']
            
            # 创建矩形标注，使用两个点表示矩形（左上角和右下角）
            shape = {
                "label": str(defect['type']),  # 确保标签是字符串类型
                "points": [
                    [float(x), float(y)],  # 左上角
                    [float(x + width), float(y + height)]  # 右下角
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            
            json_data["shapes"].append(shape)
        
        return json_data
    def dl_defect_detection(self):
        """
        使用深度学习方法进行缺陷检测
        """
        # 检查是否有待测图
        if self.cur_test_img is None:
            messagebox.showwarning("提示", "请先加载待测图")
            return None
        
        # 检查模型路径
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("错误", f"模型文件不存在: {model_path}")
            return None
        
        self.add_history("开始执行深度学习缺陷检测...")
        
        try:
            # 加载模型
            self.add_history("正在加载深度学习模型...")
            model = YOLO(model_path)
            self.add_history("模型加载成功")
            
            # 准备输入图像
            input_img = self.cur_test_img.copy()
            if len(input_img.shape) == 2:  # 如果是灰度图
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            
            # 执行推理
            conf_threshold = self.confidence_threshold_var.get()
            iou_threshold = 0.45
            self.add_history("正在执行深度学习检测...")
            results = model(input_img, conf=conf_threshold, iou=iou_threshold)
            
            # 处理结果
            result = results[0]
            boxes = result.boxes
            
            # 创建结果图像
            result_img = result.plot()
            
            # 存储检测到的缺陷信息
            defect_info = []
            
            # 处理检测结果
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 添加到缺陷信息列表
                defect_info.append({
                    "type": name,
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1
                })
            
            # 显示结果
            self.result_img = result_img
            self.canvas_result.set_image(self.result_img)
            
            # 保存检测结果供后续使用
            self.current_defect_info = defect_info
            
            # 输出检测结果
            self.add_history(f"深度学习缺陷检测完成，发现 {len(defect_info)} 个缺陷")
            for i, defect in enumerate(defect_info):
                self.add_history(f"缺陷 #{i+1}: 类型={defect['type']}, 置信度={defect['confidence']:.2f}, "
                                f"位置=({defect['x']},{defect['y']})-({defect['x']+defect['w']},{defect['y']+defect['h']})")
            
            # 如果没有检测到缺陷
            if not defect_info:
                self.add_history("未检测到任何缺陷")
                messagebox.showinfo("提示", "未检测到缺陷")
            
            return defect_info
            
        except Exception as e:
            self.add_history(f"深度学习缺陷检测出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
            messagebox.showerror("错误", f"深度学习缺陷检测失败: {str(e)}")
            return None
    
    def export_detection_report(self):
        """
        导出检测报告，包括图像和详细信息
        """
        if not hasattr(self, "current_defect_info") or not self.current_defect_info:
            messagebox.showwarning("提示", "没有可导出的检测结果")
            return
            
        if self.result_img is None:
            messagebox.showwarning("提示", "没有检测结果图像")
            return
        
        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存报告的目录")
        if not save_dir:
            return
            
        try:
            # 创建报告目录
            report_dir = os.path.join(save_dir, f"pcb_report_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(report_dir, exist_ok=True)
            
            # 保存结果图像
            img_path = os.path.join(report_dir, "result.png")
            cv2.imwrite(img_path, cv2.cvtColor(self.result_img, cv2.COLOR_RGB2BGR))
            
            # 保存缺陷信息为JSON
            json_path = os.path.join(report_dir, "defects.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_defect_info, f, indent=4, ensure_ascii=False)
            
            # 生成HTML报告
            html_path = os.path.join(report_dir, "report.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>PCB缺陷检测报告</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .header {{ text-align: center; margin-bottom: 20px; }}
                        .result-image {{ text-align: center; margin-bottom: 20px; }}
                        .result-image img {{ max-width: 100%; }}
                        table {{ width: 100%; border-collapse: collapse; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>PCB缺陷检测报告</h1>
                        <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="result-image">
                        <h2>检测结果图像</h2>
                        <img src="result.png" alt="检测结果">
                    </div>
                    
                    <h2>检测到的缺陷 ({len(self.current_defect_info)}个)</h2>
                    <table>
                        <tr>
                            <th>序号</th>
                            <th>缺陷类型</th>
                            <th>置信度</th>
                            <th>位置</th>
                            <th>尺寸</th>
                        </tr>
                """)
                
                for i, defect in enumerate(self.current_defect_info):
                    f.write(f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{defect['type']}</td>
                            <td>{defect.get('confidence', 'N/A'):.2f}</td>
                            <td>({defect['x']}, {defect['y']})</td>
                            <td>{defect['w']} x {defect['h']}</td>
                        </tr>
                    """)
                
                f.write("""
                    </table>
                </body>
                </html>
                """)
            
            # 打开报告
            import webbrowser
            webbrowser.open(f"file://{html_path}")
            
            self.add_history(f"检测报告已导出至: {report_dir}")
            messagebox.showinfo("导出成功", f"检测报告已导出至:\n{report_dir}")
            
        except Exception as e:
            self.add_history(f"导出报告出错: {str(e)}")
            messagebox.showerror("错误", f"导出报告失败: {str(e)}")
    

    def batch_defect_detection(self):
        """批量执行缺陷检测"""
        if not self.test_images:
            messagebox.showwarning("提示", "请先加载待测图文件夹")
            return
        
        
        # 选择保存结果的目录
        save_dir = filedialog.askdirectory(title="选择保存检测结果的目录")
        if not save_dir:
            return
        
        try:
            # 导入缺陷检测模块
            from defect_detection import DefectDetector
            
            # 创建检测器实例
            detector = DefectDetector()
            detector.conf_threshold = self.confidence_threshold_var.get()
            detector.manual_review_threshold = self.review_threshold_var.get()

            # 创建人工复检队列和需要复检的图像列表
            review_queue = []
            need_review_images = []  # 添加这一行初始化need_review_images
            # 如果选择了深度学习或结合方法，加载模型
            method = self.defect_method_var.get()
            if method in ["深度学习", "两种方法结合"]:
                model_path = self.model_path_var.get()
                if not os.path.exists(model_path):
                    # 尝试使用start_window.py中的模型路径
                    alt_model_path = "D:/030923/Xiangmuxuexi/PCBjiance/mpdiou/yolov8-pcb/weights/base_model/weights/best.pt"
                    if os.path.exists(alt_model_path):
                        model_path = alt_model_path
                        self.add_history(f"使用替代模型路径: {model_path}")
                    else:
                        messagebox.showerror("错误", f"模型文件不存在: {model_path}")
                        return
                
                success = detector.load_dl_model(model_path)
                if not success:
                    messagebox.showerror("错误", "加载深度学习模型失败")
                    return
                else:
                    self.add_history(f"成功加载模型: {model_path}")
            
            # 创建结果目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 创建人工复检队列
            review_queue = []
            
            # 批量处理
            total = len(self.test_images)
            processed = 0
            detected = 0          
            # 创建进度条窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("批量检测进度")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # 添加进度信息标签
            info_label = tk.Label(progress_window, text=f"正在处理: 0/{total}")
            info_label.pack(pady=10)
            
            # 添加进度条
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=total, length=350)
            progress_bar.pack(pady=10, padx=20)
            
            # 添加当前处理文件名标签
            file_label = tk.Label(progress_window, text="")
            file_label.pack(pady=5)
            
            # 添加取消按钮
            cancel_var = tk.BooleanVar(value=False)
            
            def cancel_process():
                if messagebox.askyesno("确认", "确定要取消处理吗?"):
                    cancel_var.set(True)
                    
            cancel_btn = tk.Button(progress_window, text="取消", command=cancel_process)
            cancel_btn.pack(pady=10)
            
            # 更新进度条
            progress_window.update()
            
            from concurrent.futures import ThreadPoolExecutor
            import multiprocessing
            
            def process_single_image(args):
                fn, img = args
                try:
                    # 直接调用 combined_defect_detection
                    result_img, defect_info = detector.combined_defect_detection(self.std_img, img)
                    return fn, (result_img, defect_info)
                except Exception as e:
                    return fn, (None, None, str(e))

            # 创建CSV结果文件
            csv_path = os.path.join(save_dir, "detection_results.csv")
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("文件名,缺陷类型,置信度,X,Y,宽,高\n")
            
            for fn, img in self.test_images.items():
                if cancel_var.get():
                    self.add_history("用户取消了批量处理")
                    break
                    
                # 更新进度显示
                file_label.config(text=f"当前处理: {fn}")
                info_label.config(text=f"进度: {processed}/{total}, 已检测到: {detected}个缺陷")
                progress_var.set(processed)
                progress_window.update()
                
                try:
                    if method == "传统方法":
                        defect_mask, defect_info = detector.traditional_detection(self.std_img, img)
                        if defect_mask is not None and defect_info:
                            # 在图像上标注缺陷
                            result_img = img.copy()
                            if len(result_img.shape) == 2:
                                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                            
                            for defect in defect_info:
                                x, y, w, h = defect["x"], defect["y"], defect["w"], defect["h"]
                                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(result_img, f"{defect['type']}", (x, y-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                            # 保存结果
                            result_path = os.path.join(save_dir, f"result_{fn}")
                            cv2.imwrite(result_path, result_img)
                            
                            # 保存缺陷信息
                            info_path = os.path.join(save_dir, f"info_{os.path.splitext(fn)[0]}.json")
                            with open(info_path, 'w') as f:
                                json.dump(defect_info, f, indent=4, default=numpy_json_encoder)
                            
                            # 添加到CSV
                            with open(csv_path, 'a', encoding='utf-8') as f:
                                for defect in defect_info:
                                    f.write(f"{fn},{defect['type']},{defect.get('confidence', 'N/A')},"
                                            f"{defect['x']},{defect['y']},{defect['w']},{defect['h']}\n")
                            
                            detected += 1
                            # 添加到人工复检队列
                            review_queue.append({
                                'filename': fn,
                                'result': {
                                    'image': result_img,
                                    'defects': defect_info
                                },
                                'reason': '检出缺陷'
                            })
                    
                    elif method == "深度学习":
                        result_img, defect_info = detector.dl_detection(img)
                        if defect_info:
                            # 保存结果
                            result_path = os.path.join(save_dir, f"result_{fn}")
                            cv2.imwrite(result_path, result_img)
                            
                            # 保存缺陷信息
                            info_path = os.path.join(save_dir, f"info_{os.path.splitext(fn)[0]}.json")
                            with open(info_path, 'w') as f:
                                json.dump(defect_info, f, indent=4, default=numpy_json_encoder)
                            
                            # 添加到CSV
                            with open(csv_path, 'a', encoding='utf-8') as f:
                                for defect in defect_info:
                                    f.write(f"{fn},{defect['type']},{defect.get('confidence', 0.0)},"
                                            f"{defect['x']},{defect['y']},{defect['w']},{defect['h']}\n")
                            
                            detected += 1
                            # 检查是否需要人工复检
                            need_review = False
                            for defect in defect_info:
                                if defect.get('confidence', 1.0) < detector.manual_review_threshold:
                                    need_review = True
                                    break
                            # 添加到人工复检队列
                            if need_review:
                                need_review_images.append((fn, result_path, info_path))
                                review_queue.append({
                                    'filename': fn,
                                    'result': {
                                        'image': result_img,
                                        'defects': defect_info
                                    },
                                    'reason': '低置信度'
                                })
                    
                    else:  # 两种方法结合
                        result_img, defect_info, need_review = detector.combined_defect_detection(
                            self.std_img, img)
                        
                        if defect_info:
                            # 保存结果
                            result_path = os.path.join(save_dir, f"result_{fn}")
                            cv2.imwrite(result_path, result_img)
                            #保存缺陷信息                           
                            info_path = os.path.join(save_dir, f"info_{os.path.splitext(fn)[0]}.json")
                            with open(info_path, 'w') as f:
                                json.dump(defect_info, f, indent=4, default=numpy_json_encoder)
                            
                            # 添加到CSV
                            with open(csv_path, 'a', encoding='utf-8') as f:
                                for defect in defect_info:
                                    f.write(f"{fn},{defect['type']},{defect.get('confidence', 0.0)},"
                                            f"{defect['x']},{defect['y']},{defect['w']},{defect['h']}\n")
                            
                            detected += 1
                            # 检查是否需要人工复检
                            need_review = False
                            for defect in defect_info:
                                if defect.get('confidence', 1.0) < detector.manual_review_threshold:
                                    need_review = True
                                    break
                            
                            if need_review:
                                need_review_images.append((fn, result_path, info_path))
                                review_queue.append({
                                    'filename': fn,
                                    'result': {
                                        'image': result_img,
                                        'defects': defect_info
                                    },
                                    'reason': '低置信度'
                                })
                            
                    
                    processed += 1
                    
                except Exception as e:
                    self.add_history(f"处理 {fn} 时出错: {str(e)}")
                    import traceback
                    self.add_history(f"错误详情: {traceback.format_exc()}")
            

            # 关闭进度条窗口
            progress_window.destroy()
            
            # 显示结果摘要
            summary = f"批量检测完成: 处理 {processed}/{total} 张图像，检测到 {detected} 张有缺陷"
            self.add_history(summary)
            messagebox.showinfo("处理完成", summary + f"\n结果已保存到: {save_dir}")
            
            # 处理完成后，如果有需要复检的图片，显示复检界面
            if review_queue:
                self.add_history(f"有 {len(review_queue)} 张图片需要人工复检")
                if messagebox.askyesno("提示", f"有 {len(review_queue)} 张图像需要人工复检，是否立即进行?"):
                    self.show_review_dialog(review_queue, save_dir)
                    # 在历史记录中添加复检完成的提示
                    for item in review_queue:
                        self.add_history(f"完成图片 {item['filename']} 的人工复检")
            else:
                self.add_history("没有图片需要人工复检")

        except Exception as e:
            messagebox.showerror("错误", f"批量检测失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

        """执行缺陷检测"""
        try:
            # 获取模型路径
            model_path = self.model_path_var.get()
            self.add_history("正在加载模型: " + model_path)
            
            if not os.path.exists(model_path):
                messagebox.showerror("错误", f"模型文件不存在: {model_path}")
                return
                
            # 先获取置信度阈值和IOU阈值
            conf_threshold = self.confidence_threshold_var.get()
            iou_threshold = 0.45
            
            # 再添加参数检查的日志
            self.add_history(f"模型路径: {model_path}")
            self.add_history(f"置信度阈值: {conf_threshold}")
            self.add_history(f"IOU阈值: {iou_threshold}")
            
            # 加载YOLO模型
            self.add_history("正在初始化YOLO模型...")
            from ultralytics import YOLO
            model = YOLO(model_path)
            self.add_history("YOLO模型加载成功")
            
            # 检查输入图像
            if self.cur_test_img is None:
                messagebox.showwarning("提示", "请先选择待测图")
                return
            
            # 检查并转换图像格式
            if len(self.cur_test_img.shape) == 2:  # 如果是灰度图
                self.add_history(f"原始图像为灰度图，尺寸: {self.cur_test_img.shape}")
                input_img = cv2.cvtColor(self.cur_test_img, cv2.COLOR_GRAY2BGR)
            else:
                self.add_history(f"原始图像为彩色图，尺寸: {self.cur_test_img.shape}, 通道数: {self.cur_test_img.shape[2]}")
                input_img = self.cur_test_img.copy()
            
            # 执行检测 - 完全使用与 start_window.py 相同的方式
            self.add_history("开始执行目标检测...")
            results = model(input_img)  # 使用模型默认值
            result = results[0]
            self.add_history("目标检测完成")
            
            # 使用 YOLO 的 plot 方法绘制结果，与 start_window.py 保持一致
            # 设置绘图参数，使其与 start_window.py 中的显示效果一致（蓝色框+置信度）
            result_img = result.plot(line_width=2, font_size=1, pil=False, 
                                    labels=True, conf=True, boxes=True)
            
            # 解析检测结果
            self.add_history("正在解析检测结果...")
            boxes = result.boxes
            self.add_history(f"检测到 {len(boxes)} 个目标")
            
            defect_info = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                label = result.names[cls]
                
                self.add_history(f"发现缺陷: {label}, 置信度: {conf:.2f}, 位置: ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")
                
                defect_info.append({
                    "type": label,
                    "confidence": float(conf),
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2 - x1),
                    "h": int(y2 - y1)
                })
            
            # 如果没有检测到任何缺陷
            if len(defect_info) == 0:
                self.add_history("未检测到任何缺陷")
                messagebox.showinfo("提示", "未检测到缺陷")
                return
            
            # 显示结果
            self.add_history("正在更新显示结果...")
            self.result_img = result_img
            self.canvas_result.set_image(self.result_img)
            self.add_history(f"检测完成: 发现 {len(defect_info)} 个缺陷")
            
            # 显示详细信息
            for i, defect in enumerate(defect_info):
                self.add_history(f"缺陷 {i+1}: {defect['type']} (置信度: {defect['confidence']:.2f})")
            
            # 保存检测结果供后续使用
            self.current_defect_info = defect_info
            self.add_history("检测结果已保存")
                
        except Exception as e:
            self.add_history(f"发生错误: {str(e)}")
            messagebox.showerror("错误", f"缺陷检测失败: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")

    def show_review_dialog(self, review_queue, save_dir):
        """显示人工复检对话框"""
        if not review_queue:
            return
            
        # 创建复检窗口
        review_window = tk.Toplevel(self.root)
        review_window.title("人工复检")
        review_window.geometry("1000x700")
        
        # 创建分割窗口
        paned = ttk.PanedWindow(review_window, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)
        
        # 左侧列表
        left_frame = ttk.Frame(paned, width=300)
        paned.add(left_frame, weight=1)
        
        # 右侧图像和详情
        right_frame = ttk.Frame(paned, width=700)
        paned.add(right_frame, weight=3)
        
        # 创建列表
        ttk.Label(left_frame, text="待复检图片:").pack(anchor="w", padx=5, pady=5)
        
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        columns = ("文件名", "原因", "缺陷数")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80)
        
        # 添加数据
        for i, item in enumerate(review_queue):
            filename = os.path.basename(item['filename'])
            reason = item['reason']
            defect_count = len(item['result'].get('defects', []))
            tree.insert("", "end", iid=i, values=(filename, reason, defect_count))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        
        # 右侧图像显示
        image_frame = ttk.Frame(right_frame)
        image_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 图像画布
        canvas = tk.Canvas(image_frame, bg="black")
        canvas.pack(fill="both", expand=True)
        
        # 详情区域
        detail_frame = ttk.Frame(right_frame, height=200)
        detail_frame.pack(fill="x", padx=5, pady=5)
        
        # 缺陷列表
        ttk.Label(detail_frame, text="检测到的缺陷:").pack(anchor="w")
        
        defect_frame = ttk.Frame(detail_frame)
        defect_frame.pack(fill="both", expand=True)
        
        defect_columns = ("序号", "类型", "置信度", "位置", "尺寸")
        defect_tree = ttk.Treeview(defect_frame, columns=defect_columns, show="headings", height=5)
        
        for col in defect_columns:
            defect_tree.heading(col, text=col)
            defect_tree.column(col, width=100)
        
        defect_scrollbar = ttk.Scrollbar(defect_frame, orient="vertical", command=defect_tree.yview)
        defect_scrollbar.pack(side="right", fill="y")
        defect_tree.configure(yscrollcommand=defect_scrollbar.set)
        defect_tree.pack(side="left", fill="both", expand=True)
        
        # 按钮区域
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        # 当前选中的项目
        current_item = {"index": -1}
        
        # 显示选中图片
        def show_selected_item(event):
            selected = tree.selection()
            if not selected:
                return
                
            idx = int(selected[0])
            current_item["index"] = idx
            item = review_queue[idx]
            
            # 显示图像
            if 'result' in item and 'image' in item['result']:
                img = item['result']['image']
                if img is not None:
                    # 调整图像大小以适应画布
                    h, w = img.shape[:2]
                    canvas_w = canvas.winfo_width()
                    canvas_h = canvas.winfo_height()
                    
                    if canvas_w > 0 and canvas_h > 0:
                        scale = min(canvas_w / w, canvas_h / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        
                        # 调整图像大小
                        img_resized = cv2.resize(img, (new_w, new_h))
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                        
                        # 清除画布并显示新图像
                        canvas.delete("all")
                        canvas.create_image(canvas_w//2, canvas_h//2, image=img_tk, anchor="center")
                        canvas.image = img_tk  # 保持引用
            
            # 显示缺陷信息
            defect_tree.delete(*defect_tree.get_children())
            if 'result' in item and 'defects' in item['result']:
                defects = item['result']['defects']
                for i, defect in enumerate(defects):
                    pos = f"({defect['x']},{defect['y']})"
                    size = f"{defect['w']}x{defect['h']}"
                    conf = f"{defect.get('confidence', 'N/A'):.2f}" if isinstance(defect.get('confidence', 'N/A'), (int, float)) else 'N/A'
                    defect_tree.insert("", "end", values=(i+1, defect['type'], conf, pos, size))
        
        # 绑定选择事件
        tree.bind("<<TreeviewSelect>>", show_selected_item)
        
        # 确认按钮
        def confirm_review():
            if current_item["index"] >= 0:
                # 在这里可以添加保存修改后的缺陷信息的代码
                item = review_queue[current_item["index"]]
                self.add_history(f"确认复检: {os.path.basename(item['filename'])}")
                
                # 如果需要保存修改后的结果
                # filename = item['filename']
                # defects = item['result']['defects']
                # info_path = os.path.join(save_dir, f"info_{os.path.splitext(os.path.basename(filename))[0]}.json")
                # with open(info_path, 'w') as f:
                #     json.dump(defects, f, indent=4, default=numpy_json_encoder)
            
            # 移动到下一项
            next_idx = current_item["index"] + 1
            if next_idx < len(review_queue):
                tree.selection_set(next_idx)
                tree.see(next_idx)
                show_selected_item(None)
            else:
                messagebox.showinfo("完成", "所有图片已复检完成")
                review_window.destroy()
        
        # 跳过按钮
        def skip_review():
            # 移动到下一项
            next_idx = current_item["index"] + 1
            if next_idx < len(review_queue):
                tree.selection_set(next_idx)
                tree.see(next_idx)
                show_selected_item(None)
            else:
                messagebox.showinfo("完成", "所有图片已复检完成")
                review_window.destroy()
        
        # 完成按钮
        def finish_review():
            if messagebox.askyesno("确认", "确定要结束复检吗?"):
                review_window.destroy()
        
        # 添加按钮
        ttk.Button(button_frame, text="确认", command=confirm_review).pack(side="left", padx=5)
        ttk.Button(button_frame, text="跳过", command=skip_review).pack(side="left", padx=5)
        ttk.Button(button_frame, text="完成", command=finish_review).pack(side="right", padx=5)
        
        # 自动选择第一项
        if review_queue:
            tree.selection_set(0)
            tree.focus(0)
            # 等待窗口完全加载后显示第一项
            review_window.update()
            show_selected_item(None)
        
        # 等待窗口关闭
        review_window.wait_window()
        
        # 记录复检完成
        self.add_history("人工复检完成")


    def on_conn_components(self):
        target = self.target_var.get()
        if target == "标准图":
            base_img = self.std_img
        elif target == "待测图":
            base_img = self.cur_test_img
        else:
            base_img = self.result_img
        if base_img is None:
            messagebox.showwarning("提示", f"{target}为空")
            return

        method = self.thresh_method_var.get()
        manual_val = self.manual_thresh_val.get()
        blockSize = self.adapt_blockSize.get()
        C = self.adapt_C.get()

        bin_img = ip.threshold_segment(base_img, method=method, val=manual_val,
                                       blockSize=blockSize, C=C)

        try:
            min_area = self.min_area_var.get()
            result = ip.analyze_components(bin_img, min_area=min_area)
            if isinstance(result, tuple) and len(result) == 2:
                marked_img, components_info = result
            else:
                raise ValueError("analyze_components 返回值格式不正确")

            self.canvas_result.set_image(marked_img)

            self.add_history(f"=== {target} 连通域分析结果 ===")
            self.add_history(f"检测到 {len(components_info)} 个连通域（最小面积={min_area}）")
            for comp in components_info:
                self.add_history(f"连通域 #{comp['id']}: 面积={comp['area']:.1f}, "
                                 f"周长={comp['perimeter']:.1f}, "
                                 f"质心=({comp['centroid'][0]:.1f}, {comp['centroid'][1]:.1f})")
            self.add_history("=== 分析结束 ===")
        except Exception as e:
            messagebox.showerror("错误", f"连通域分析失败: {str(e)}")    




    def traditional_defect_detection(self):
            """使用传统方法进行缺陷检测"""
            if self.std_img is None or self.cur_test_img is None:
                messagebox.showwarning("提示", "请先加载标准图和待测图")
                return None, []
            
            self.add_history("开始执行传统方法缺陷检测...")

                
            try:
                # 1. 图像预处理 - 使用当前面板设置的参数
                self.add_history("步骤1: 图像预处理")
                # 确保两张图像尺寸一致
                if self.std_img.shape[:2] != self.cur_test_img.shape[:2]:
                    h, w = self.std_img.shape[:2]
                    self.cur_test_img = cv2.resize(self.cur_test_img, (w, h))
                    self.add_history(f"调整待测图尺寸为 {w}x{h} 以匹配标准图")
                # 获取标准图和待测图的副本
                std_img_proc = self.std_img.copy()
                test_img_proc = self.cur_test_img.copy()

                # 转换为灰度图
                if len(std_img_proc.shape) == 3:
                    std_gray = cv2.cvtColor(std_img_proc, cv2.COLOR_BGR2GRAY)
                else:
                    std_gray = std_img_proc.copy()
                    
                if len(test_img_proc.shape) == 3:
                    test_gray = cv2.cvtColor(test_img_proc, cv2.COLOR_BGR2GRAY)
                else:
                    test_gray = test_img_proc.copy()
        
                
                # 获取预处理顺序
                preprocess_order = []
                if hasattr(self, 'order_step1') and self.order_step1.get() != "none":
                    preprocess_order.append(self.order_step1.get())
                if hasattr(self, 'order_step2') and self.order_step2.get() != "none":
                    preprocess_order.append(self.order_step2.get())
                if hasattr(self, 'order_step3') and self.order_step3.get() != "none":
                    preprocess_order.append(self.order_step3.get())
                
                self.add_history(f"预处理顺序: {preprocess_order}")
                
                # 如果没有设置预处理顺序，使用默认顺序
                if not preprocess_order:
                    preprocess_order = ["滤波", "均衡化", "阈值分割", "形态学"]
                    self.add_history("使用默认预处理顺序")
                
                # 按顺序执行预处理
                for step in preprocess_order:
                    if step == "滤波":
                        # 应用滤波
                        filter_method = self.filter_var.get()
                        if filter_method != "none":
                            self.add_history(f"应用滤波: {filter_method}")
                            filter_h = self.filter_h.get()
                            filter_k = self.filter_k.get()
                            
                            if filter_method == "gaussian":
                                std_gray = cv2.GaussianBlur(std_gray, (filter_k, filter_k), 0)
                                test_gray = cv2.GaussianBlur(test_gray, (filter_k, filter_k), 0)
                                self.add_history(f"高斯滤波: kernel={filter_k}")
                            elif filter_method == "median":
                                std_gray = cv2.medianBlur(std_gray, filter_k)
                                test_gray = cv2.medianBlur(test_gray, filter_k)
                                self.add_history(f"中值滤波: kernel={filter_k}")
                            elif filter_method == "bilateral":
                                std_gray = cv2.bilateralFilter(std_gray, filter_k, filter_h, filter_h)
                                test_gray = cv2.bilateralFilter(test_gray, filter_k, filter_h, filter_h)
                                self.add_history(f"双边滤波: d={filter_k}, sigma={filter_h}")
                            elif filter_method == "nlm":
                                std_gray = cv2.fastNlMeansDenoising(std_gray, None, filter_h, filter_k, 7)
                                test_gray = cv2.fastNlMeansDenoising(test_gray, None, filter_h, filter_k, 7)
                                self.add_history(f"非局部均值滤波: h={filter_h}, templateWindowSize={filter_k}")
                    
                    elif step == "均衡化":
                        # 应用直方图均衡化
                        eq_method = self.eq_var.get()
                        if eq_method != "none":
                            self.add_history(f"应用直方图均衡化: {eq_method}")
                            
                            if eq_method == "global":
                                std_gray = cv2.equalizeHist(std_gray)
                                test_gray = cv2.equalizeHist(test_gray)
                                self.add_history("全局直方图均衡化")
                            elif eq_method == "clahe":
                                clip_limit = self.eq_clip.get()
                                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                                std_gray = clahe.apply(std_gray)
                                test_gray = clahe.apply(test_gray)
                                self.add_history(f"CLAHE均衡化: clipLimit={clip_limit}")
                    
                    elif step == "锐化":
                        # 应用锐化
                        sharpen_method = self.sharpen_var.get()
                        if sharpen_method != "none":
                            self.add_history(f"应用锐化: {sharpen_method}")
                            sharpen_w = self.sharpen_w.get()
                            
                            if sharpen_method == "laplacian":
                                kernel = np.array([[0, -1, 0], [-1, 4 + sharpen_w, -1], [0, -1, 0]], np.float32)
                                std_gray = cv2.filter2D(std_gray, -1, kernel)
                                test_gray = cv2.filter2D(test_gray, -1, kernel)
                                self.add_history(f"拉普拉斯锐化: weight={sharpen_w}")
                            elif sharpen_method == "unsharp":
                                gaussian = cv2.GaussianBlur(std_gray, (5, 5), 0)
                                std_gray = cv2.addWeighted(std_gray, 1 + sharpen_w, gaussian, -sharpen_w, 0)
                                
                                gaussian = cv2.GaussianBlur(test_gray, (5, 5), 0)
                                test_gray = cv2.addWeighted(test_gray, 1 + sharpen_w, gaussian, -sharpen_w, 0)
                                self.add_history(f"USM锐化: weight={sharpen_w}")
                
                # 2. 图像对齐
                self.add_history("步骤2: 图像对齐")
                
                # 使用选择的配准方法和对齐方法
                reg_method = self.reg_method_var.get()
                align_method = self.align_method_var.get() if hasattr(self, "align_method_var") else "透视变换"
                
                self.add_history(f"使用配准方法: {reg_method}, 对齐方法: {align_method}")
                
                # 记录原始尺寸
                self.add_history(f"标准图像尺寸: {std_gray.shape}, 待测图像尺寸: {test_gray.shape}")
                
                if reg_method == "none":
                    aligned_test = test_gray
                    self.add_history("不进行图像对齐")
                else:
                    try:
                        # 确保图像类型正确
                        std_gray = std_gray.astype(np.uint8)
                        test_gray = test_gray.astype(np.uint8)
                        
                        # 根据选择的对齐方法执行不同的对齐算法
                        if align_method == "透视变换":
                            # 使用原有的透视变换方法
                            if reg_method == "sift":
                                nfeatures = self.sift_nfeatures.get() if hasattr(self, 'sift_nfeatures') else 500
                                _, aligned_test, score = ir.align_and_overlay(std_gray, test_gray, detector="SIFT", nfeatures=nfeatures)
                                self.add_history(f"SIFT特征配准+透视变换: nfeatures={nfeatures}, score={score:.2f}")
                            elif reg_method == "orb":
                                _, aligned_test, score = ir.align_and_overlay(std_gray, test_gray, detector="ORB")
                                self.add_history(f"ORB特征配准+透视变换: score={score:.2f}")
                            elif reg_method == "ecc":
                                _, aligned_test, score = ir.do_registration(std_gray, test_gray, detector="ECC")
                                self.add_history(f"ECC配准+透视变换: score={score:.2f}")
                            else:
                                # 默认使用SIFT
                                _, aligned_test, score = ir.align_and_overlay(std_gray, test_gray, detector="SIFT")
                                self.add_history(f"默认SIFT配准+透视变换: score={score:.2f}")
                        elif align_method == "薄板样条变换":
                            # 使用TPS变换
                            aligned_test = self.perform_tps_alignment(reg_method)
                            self.add_history(f"完成薄板样条变换对齐")
                        elif align_method == "多尺度特征匹配":
                            # 使用多尺度特征匹配
                            aligned_test = self.perform_multiscale_alignment(reg_method)
                            self.add_history(f"完成多尺度特征匹配对齐")
                        elif align_method == "局部分块对齐":
                            # 使用局部分块对齐
                            aligned_test = self.perform_local_alignment(reg_method, std_gray, test_gray)
                            self.add_history(f"完成局部分块对齐")
                        else:
                            # 默认使用透视变换
                            if reg_method == "sift":
                                nfeatures = self.sift_nfeatures.get() if hasattr(self, 'sift_nfeatures') else 500
                                _, aligned_test, score = ir.align_and_overlay(std_gray, test_gray, detector="SIFT", nfeatures=nfeatures)
                                self.add_history(f"SIFT特征配准+默认透视变换: nfeatures={nfeatures}, score={score:.2f}")
                            else:
                                _, aligned_test, score = ir.align_and_overlay(std_gray, test_gray, detector="SIFT")
                                self.add_history(f"默认SIFT配准+默认透视变换: score={score:.2f}")
                        
                        # 添加调试信息
                        if aligned_test is not None:
                            h1, w1 = test_gray.shape[:2]
                            h2, w2 = aligned_test.shape[:2]
                            self.add_history(f"原始图像尺寸: {w1}x{h1}, 对齐后尺寸: {w2}x{h2}")
                            
                            # 计算对齐前后的差异
                            before_diff = cv2.absdiff(std_gray, cv2.resize(test_gray, (std_gray.shape[1], std_gray.shape[0])))
                            after_diff = cv2.absdiff(std_gray, aligned_test)
                            
                            before_score = np.sum(before_diff) / (before_diff.shape[0] * before_diff.shape[1])
                            after_score = np.sum(after_diff) / (after_diff.shape[0] * after_diff.shape[1])
                            
                            self.add_history(f"对齐前差异分数: {before_score:.2f}, 对齐后差异分数: {after_score:.2f}")
                            self.add_history(f"对齐改善率: {(before_score - after_score) / before_score * 100:.2f}%")
                        
                        if aligned_test is None:
                            self.add_history("图像对齐失败，使用原始图像继续")
                            aligned_test = test_gray
                        else:
                            self.add_history(f"对齐后图像尺寸: {aligned_test.shape}")
                    except Exception as e:
                        self.add_history(f"图像对齐出错: {str(e)}")
                        import traceback
                        self.add_history(f"错误详情: {traceback.format_exc()}")
                        aligned_test = test_gray
                
                # 确保两个图像尺寸一致 - 取交集
                min_h = min(std_gray.shape[0], aligned_test.shape[0])
                min_w = min(std_gray.shape[1], aligned_test.shape[1])
                
                # 裁剪图像到相同尺寸
                std_gray_cropped = std_gray[:min_h, :min_w]
                aligned_test_cropped = aligned_test[:min_h, :min_w]
                
                self.add_history(f"裁剪后图像尺寸: {std_gray_cropped.shape}")
                
                # 3. 图像差分
                self.add_history("步骤3: 图像差分")
                # 使用裁剪后的图像进行差分
                diff = cv2.absdiff(std_gray_cropped, aligned_test_cropped)
                
                # 保存差分图像用于显示
                diff_vis = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
                
                # 4. 阈值分割
                self.add_history("步骤4: 阈值分割")
                thresh_method = self.thresh_method_var.get()
                self.add_history(f"使用阈值方法: {thresh_method}")
                
                if thresh_method == "manual":
                    thresh_val = self.manual_thresh_val.get()
                    self.add_history(f"手动阈值: {thresh_val}")
                    _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
                elif thresh_method == "otsu":
                    self.add_history("Otsu自适应阈值")
                    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                elif thresh_method == "adaptive":
                    block_size = self.adapt_blockSize.get()
                    c = self.adapt_C.get()
                    self.add_history(f"自适应阈值: blockSize={block_size}, C={c}")
                    binary = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, block_size, c)
                else:
                    # 默认使用手动阈值
                    thresh_val = 30
                    self.add_history(f"默认手动阈值: {thresh_val}")
                    _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
                
                # 5. 形态学操作
                self.add_history("步骤5: 形态学操作")
                morph_method = self.morph_method_var.get()
                
                if morph_method != "none":
                    kernel_size = self.morph_kernel.get()
                    iterations = self.morph_iter.get()
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    self.add_history(f"形态学操作: {morph_method}, kernel={kernel_size}, iterations={iterations}")
                    
                    if morph_method == "erode":
                        binary = cv2.erode(binary, kernel, iterations=iterations)
                    elif morph_method == "dilate":
                        binary = cv2.dilate(binary, kernel, iterations=iterations)
                    elif morph_method == "open":
                        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
                    elif morph_method == "close":
                        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                    elif morph_method == "gradient":
                        binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
                
                # 显示中间结果
                # 创建一个包含多个处理步骤的可视化图像
                h, w = std_gray_cropped.shape[:2]  # 使用裁剪后的尺寸
                vis_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                
                # 转换灰度图为彩色以便显示
                std_vis = cv2.cvtColor(std_gray_cropped, cv2.COLOR_GRAY2BGR)  # 使用裁剪后的图像
                test_vis = cv2.cvtColor(aligned_test_cropped, cv2.COLOR_GRAY2BGR)  # 使用裁剪后的图像
                diff_vis = cv2.applyColorMap(diff, cv2.COLORMAP_JET)  # 使用热力图显示差分
                binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                
                # 组合图像
                vis_img[0:h, 0:w] = std_vis
                vis_img[0:h, w:w*2] = test_vis
                vis_img[h:h*2, 0:w] = diff_vis
                vis_img[h:h*2, w:w*2] = binary_vis
                
                # 添加标签 - 使用更明显的颜色和更大的字体
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                font_color = (0, 255, 255)  # 黄色
                
                cv2.putText(vis_img, "标准图", (10, 30), font, font_scale, font_color, font_thickness)
                cv2.putText(vis_img, "对齐后的待测图", (w+10, 30), font, font_scale, font_color, font_thickness)
                cv2.putText(vis_img, "差分图", (10, h+30), font, font_scale, font_color, font_thickness)
                cv2.putText(vis_img, "二值图", (w+10, h+30), font, font_scale, font_color, font_thickness)
                
                # 显示处理过程可视化图像
                self.canvas_result.set_image(vis_img)



                # 6. 连通域分析
                self.add_history("步骤6: 连通域分析")
                # 设置最小面积过滤
                min_area = self.min_area_var.get() if hasattr(self, 'min_area_var') else 50
                self.add_history(f"最小连通域面积: {min_area}")
                
                # 查找轮廓
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                self.add_history(f"找到 {len(contours)} 个初始轮廓")
                
                # 创建结果图像 - 使用原始待测图像的裁剪部分
                if len(self.cur_test_img.shape) == 3:
                    result_img = self.cur_test_img.copy()[:min_h, :min_w]
                    if result_img.shape[2] != 3:  # 确保是3通道
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                else:
                    result_img = cv2.cvtColor(self.cur_test_img.copy()[:min_h, :min_w], cv2.COLOR_GRAY2BGR)
                
                # 创建感兴趣区域可视化图像
                roi_vis = result_img.copy()
                
                # 存储缺陷信息
                defect_info = []
                valid_contours = 0

                # 处理每个轮廓
                for i, contour in enumerate(contours):
                    # 计算面积
                    area = cv2.contourArea(contour)
                    
                    # 过滤小面积
                    if area < min_area:
                        continue
                        
                    valid_contours += 1
                    
                    # 获取边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 计算周长
                    perimeter = cv2.arcLength(contour, True)
                    
                    # 计算圆形度
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # 提取ROI区域的特征
                    roi = diff[y:y+h, x:x+w]
                    mean_intensity = np.mean(roi)
                    std_intensity = np.std(roi)
                    
                    # 提取标准图和待测图中对应的ROI区域
                    roi_std = std_gray[y:y+h, x:x+w]
                    roi_test = aligned_test[y:y+h, x:x+w]
                    roi_mask_local = binary[y:y+h, x:x+w]
                    
                    # 提取缺陷特征并进行分类
                    features = self.extract_defect_features(roi_std, roi_test, roi_mask_local)
                    defect_type, confidence = self.classify_defect(features)
                    
                    # 在结果图像上标记缺陷
                    color = (0, 0, 255)  # 红色
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(result_img, f"{defect_type} ({confidence:.2f})", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 在ROI可视化图像上标记感兴趣区域
                    cv2.rectangle(roi_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(roi_vis, f"#{valid_contours}", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 添加缺陷信息
                    defect_info.append({
                        "id": valid_contours,
                        "type": defect_type,
                        "confidence": confidence,  # 添加置信度
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "area": area,
                        "perimeter": perimeter,
                        "circularity": circularity,
                        "mean_intensity": mean_intensity,
                        "std_intensity": std_intensity,
                        "features": features
                    })
                    
                    # 输出详细信息
                    self.add_history(f"缺陷 #{valid_contours}: 类型={defect_type}, 置信度={confidence:.2f}, 位置=({x},{y}), 尺寸={w}x{h}, 面积={area:.1f}, "
                                f"周长={perimeter:.1f}, 圆形度={circularity:.2f}, "
                                f"平均强度={mean_intensity:.1f}, 标准差={std_intensity:.1f}")
                
                # 显示中间结果
                # 创建一个包含多个处理步骤的可视化图像
                h, w = std_gray.shape[:2]
                vis_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                
                # 转换灰度图为彩色以便显示
                std_vis = cv2.cvtColor(std_gray, cv2.COLOR_GRAY2BGR)
                test_vis = cv2.cvtColor(aligned_test, cv2.COLOR_GRAY2BGR)
                binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                
                # 组合图像
                vis_img[0:h, 0:w] = std_vis
                vis_img[0:h, w:w*2] = test_vis
                vis_img[h:h*2, 0:w] = diff_vis
                vis_img[h:h*2, w:w*2] = binary_vis
                
                # 添加标签
                cv2.putText(vis_img, "标准图", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_img, "对齐后的待测图", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_img, "差分图", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_img, "二值图", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示处理过程可视化图像
                self.canvas_result.set_image(vis_img)
                
                # 输出检测结果
                if valid_contours > 0:
                    self.add_history(f"传统方法检测完成，发现 {valid_contours} 个缺陷")
                    
                    # 显示最终结果
                    self.result_img = result_img
                    
                    # 创建一个显示原始图像和结果图像的组合图
                    combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
                    combined_img[0:h, 0:w] = cv2.cvtColor(self.cur_test_img, cv2.COLOR_BGR2RGB) if len(self.cur_test_img.shape) == 3 else cv2.cvtColor(self.cur_test_img, cv2.COLOR_GRAY2RGB)
                    combined_img[0:h, w:w*2] = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB) if len(result_img.shape) == 3 else cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
    
                    # 在这里插入 PIL 绘制代码 ↓
                    result_img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(result_img_pil)
                    
                    # 加载中文字体
                    font_path = "C:/Windows/Fonts/simhei.ttf"  # 使用黑体
                    font = ImageFont.truetype(font_path, 20)
                    
                    # 为每个检测到的缺陷添加标注
                    for defect in defect_info:
                        x, y = defect['x'], defect['y']
                        w, h = defect['w'], defect['h']
                        defect_type = defect['type']
                        confidence = defect.get('confidence', 'N/A')
                        
                        # 绘制矩形框
                        draw.rectangle([(x, y), (x+w, y+h)], outline=(255, 0, 0), width=2)
                        
                        # 准备标注文本
                        if isinstance(confidence, float):
                            text = f"{defect_type} ({confidence:.2f})"
                        else:
                            text = f"{defect_type}"
                        
                        # 计算文本背景
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # 绘制文本背景
                        draw.rectangle([(x, y-text_height-4), (x+text_width, y-2)], 
                                    fill=(255, 255, 255), outline=(255, 0, 0))
                        
                        # 绘制文本
                        draw.text((x, y-text_height-2), text, font=font, fill=(255, 0, 0))
                    
                    # 将 PIL 图像转回 OpenCV 格式
                    result_img = cv2.cvtColor(np.array(result_img_pil), cv2.COLOR_RGB2BGR)
                    # PIL 绘制代码结束 ↑
                    
                    # 显示最终结果
                    self.result_img = result_img
                    self.canvas_result.set_image(combined_img)
                    # 保存当前缺陷信息
                    self.current_defect_info = defect_info
                    

                    # 创建感兴趣区域图像
                    roi_img = self.cur_test_img.copy()
                    if len(roi_img.shape) == 2:
                        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
                        
                    # 在感兴趣区域图像上标记ROI
                    for defect in defect_info:
                        x, y, w, h = defect["x"], defect["y"], defect["w"], defect["h"]
                        cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(roi_img, f"ROI #{defect['id']}", (x, y-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 保存感兴趣区域图像供后续使用
                    self.roi_img = roi_img
                else:
                    self.add_history("传统方法未检测到缺陷")
                    
                    # 显示ROI可视化图像
                    self.result_img = roi_vis
                    self.canvas_result.set_image(roi_vis)
                    
                    # 添加调试信息
                    self.add_history("调试信息:")
                    self.add_history(f"差分图像平均值: {np.mean(diff):.2f}")
                    self.add_history(f"差分图像标准差: {np.std(diff):.2f}")
                    self.add_history(f"差分图像最大值: {np.max(diff)}")
                    self.add_history(f"差分图像最小值: {np.min(diff)}")
                    self.add_history(f"二值图像非零像素数: {np.count_nonzero(binary)}")
                    
                    # 尝试使用更低的阈值
                    lower_thresh = thresh_val // 2 if thresh_method == "manual" else 15
                    self.add_history(f"尝试使用更低的阈值 {lower_thresh} 重新检测")
                    _, lower_binary = cv2.threshold(diff, lower_thresh, 255, cv2.THRESH_BINARY)
                    lower_binary = cv2.morphologyEx(lower_binary, cv2.MORPH_OPEN, kernel)
                    
                    # 查找轮廓
                    lower_contours, _ = cv2.findContours(lower_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    self.add_history(f"使用更低阈值找到 {len(lower_contours)} 个轮廓")
                    
                    # 计算有效轮廓数量
                    valid_lower_contours = sum(1 for c in lower_contours if cv2.contourArea(c) >= min_area)
                    self.add_history(f"其中有效轮廓数量: {valid_lower_contours}")
                    
                    # 如果使用更低阈值找到了有效轮廓，建议用户调整阈值
                    if valid_lower_contours > 0:
                        self.add_history("建议: 尝试降低阈值或调整其他参数以检测到缺陷")
                
                return result_img, defect_info
                
            except Exception as e:
                self.add_history(f"传统方法缺陷检测出错: {str(e)}")
                import traceback
                self.add_history(f"错误详情: {traceback.format_exc()}")
                messagebox.showerror("错误", f"传统方法缺陷检测失败: {str(e)}")
                return None, []







    def extract_defect_features(self, roi_std, roi_test, roi_mask):
        """
        提取缺陷特征，用于决策树分类
        参数:
            roi_std: 标准图中的ROI区域
            roi_test: 待测图中的ROI区域
            roi_mask: ROI掩码
        返回:
            features: 特征字典
        """
        # 计算标准图和待测图中的像素数量
        std_pixels = cv2.countNonZero(roi_std)
        test_pixels = cv2.countNonZero(roi_test)
        
        # 计算ROI区域的总像素数
        total_pixels = cv2.countNonZero(roi_mask)
        
        # 计算特征r（像素比例）
        r = min(std_pixels, test_pixels) / max(std_pixels, test_pixels) if max(std_pixels, test_pixels) > 0 else 0
        
        # 计算周长和面积
        contours_std, _ = cv2.findContours(roi_std, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_test, _ = cv2.findContours(roi_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算标准图的周长和面积
        perimeter_std = sum(cv2.arcLength(cnt, True) for cnt in contours_std) if contours_std else 0
        area_std = sum(cv2.contourArea(cnt) for cnt in contours_std) if contours_std else 0
        
        # 计算待测图的周长和面积
        perimeter_test = sum(cv2.arcLength(cnt, True) for cnt in contours_test) if contours_test else 0
        area_test = sum(cv2.contourArea(cnt) for cnt in contours_test) if contours_test else 0
        
        # 计算周长比和面积比
        R_P = perimeter_std / perimeter_test if perimeter_test > 0 else 0
        R_D = area_std / area_test if area_test > 0 else 0
        
        # 计算连通域数量
        num_labels_std, _ = cv2.connectedComponents(roi_std)
        num_labels_test, _ = cv2.connectedComponents(roi_test)
        N_P = num_labels_std - 1  # 减去背景
        N_S = num_labels_test - 1  # 减去背景
        
        # 计算形状复杂度
        S_P = perimeter_std**2 / (4 * np.pi * area_std) if area_std > 0 else 0
        S_D = perimeter_test**2 / (4 * np.pi * area_test) if area_test > 0 else 0
        
        # 检测是否存在错位
        misalign_flag = False
        if contours_std and contours_test:
            # 计算标准图和待测图的质心
            M_std = cv2.moments(contours_std[0])
            M_test = cv2.moments(contours_test[0])
            
            if M_std['m00'] > 0 and M_test['m00'] > 0:
                cx_std = int(M_std['m10'] / M_std['m00'])
                cy_std = int(M_std['m01'] / M_std['m00'])
                cx_test = int(M_test['m10'] / M_test['m00'])
                cy_test = int(M_test['m01'] / M_test['m00'])
                
                # 如果质心距离超过阈值，则认为存在错位
                dist = np.sqrt((cx_std - cx_test)**2 + (cy_std - cy_test)**2)
                misalign_flag = dist > 5  # 阈值可调整
        
        # 计算孔洞比
        holes_std = self.count_holes(roi_std)
        holes_test = self.count_holes(roi_test)
        R_h = holes_std / holes_test if holes_test > 0 else 0
        
        # 计算梯度特征
        gray_std = cv2.GaussianBlur(roi_std, (3, 3), 0)
        gray_test = cv2.GaussianBlur(roi_test, (3, 3), 0)
        
        grad_x_std = cv2.Sobel(gray_std, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_std = cv2.Sobel(gray_std, cv2.CV_64F, 0, 1, ksize=3)
        grad_x_test = cv2.Sobel(gray_test, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_test = cv2.Sobel(gray_test, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        grad_mag_std = cv2.magnitude(grad_x_std, grad_y_std)
        grad_mag_test = cv2.magnitude(grad_x_test, grad_y_test)
        
        # 计算梯度比
        grad_ratio = np.mean(grad_mag_std) / np.mean(grad_mag_test) if np.mean(grad_mag_test) > 0 else 0
        
        # 返回特征字典
        features = {
            "r": r,                      # 像素比例
            "R_P": R_P,                  # 周长比
            "R_D": R_D,                  # 面积比
            "N_P": N_P,                  # 标准图连通域数
            "N_S": N_S,                  # 待测图连通域数
            "S_P": S_P,                  # 标准图形状复杂度
            "S_D": S_D,                  # 待测图形状复杂度
            "misalign": misalign_flag,   # 是否错位
            "R_h": R_h,                  # 孔洞比
            "grad_ratio": grad_ratio,    # 梯度比
            "std_pixels": std_pixels,    # 标准图像素数
            "test_pixels": test_pixels,  # 待测图像素数
            "total_pixels": total_pixels # 总像素数
        }
        
        return features

    def count_holes(self, binary_img):
        """计算二值图像中的孔洞数量"""
        # 反转图像（将前景变为背景，背景变为前景）
        inverted = cv2.bitwise_not(binary_img)
        
        # 标记连通域
        num_labels, _ = cv2.connectedComponents(inverted)
        
        # 减去1（背景）和1（外部区域），得到孔洞数
        holes = max(0, num_labels - 2)
        
        return holes
    def classify_defect(self, features):
        """
        基于特征使用决策树规则对缺陷进行分类
        参数:
            features: 特征字典
        返回:
            defect_type: 缺陷类型
            confidence: 置信度
        """
        # 提取关键特征
        r = features["r"]
        R_P = features["R_P"]
        R_D = features["R_D"]
        N_P = features["N_P"]
        N_S = features["N_S"]
        misalign = features["misalign"]
        std_pixels = features["std_pixels"]
        test_pixels = features["test_pixels"]
        
        # 初始化置信度
        confidence = 0.5  # 默认置信度
        
        # 决策树规则（基于classical_method.py）
        # 1. 首先检查是否为错位
        if misalign:
            return "错位", 0.85
        
        # 2. 检查是否为缺失
        if std_pixels > 0 and test_pixels == 0:
            return "缺失", 0.9
        
        # 3. 检查是否为多余
        if std_pixels == 0 and test_pixels > 0:
            return "多余", 0.9
        
        # 4. 检查是否为短路
        if N_P > N_S and R_D < 0.8:
            confidence = 0.8 + min(0.15, (1 - R_D) * 0.5)
            return "短路", confidence
        
        # 5. 检查是否为开路
        if N_P < N_S and R_D < 0.8:
            confidence = 0.8 + min(0.15, (1 - R_D) * 0.5)
            return "开路", confidence
        
        # 6. 检查是否为针孔
        if 0.8 < r < 0.95 and 0.8 < R_P < 0.95:
            confidence = 0.75 + (0.95 - r) * 0.5
            return "针孔", confidence
        
        # 7. 检查是否为凸起
        if 0.7 < r < 0.9 and R_P > 1.1:
            confidence = 0.7 + min(0.2, (R_P - 1.1) * 0.5)
            return "凸起", confidence
        
        # 8. 检查是否为缩小
        if 0.7 < r < 0.9 and R_D < 0.9:
            confidence = 0.7 + min(0.2, (0.9 - R_D) * 0.5)
            return "缩小", confidence
        
        # 9. 检查是否为扩大
        if 0.7 < r < 0.9 and R_D > 1.1:
            confidence = 0.7 + min(0.2, (R_D - 1.1) * 0.5)
            return "扩大", confidence
        
        # 10. 如果没有匹配任何规则，则标记为未知缺陷而不是无缺陷
        return "未知缺陷", 0.6
    
    def combined_defect_detection(self):
        try:
            # 1. 首先使用传统方法进行图像差分，获取感兴趣区域
            self.add_history("步骤1: 使用传统方法进行图像差分")
            diff_img, traditional_defects = self.traditional_defect_detection()
            
            if diff_img is None:
                messagebox.showerror("错误", "传统方法差分失败")
                return []
            
            # 2. 加载深度学习模型
            self.add_history("步骤2: 加载深度学习模型")
            model_path = self.model_path_var.get() if hasattr(self, 'model_path_var') else None
            
            if not model_path or not os.path.exists(model_path):
                messagebox.showerror("错误", "请先选择有效的模型文件")
                return []
                
            model = YOLO(model_path)
            conf_threshold = self.confidence_threshold_var.get()
            iou_threshold = 0.45
            self.add_history(f"模型加载成功: {model_path}")
            # 初始化缺陷列表
            combined_defects = []
            # 3. 在整张图上进行深度学习检测
            input_img = self.cur_test_img.copy()
            if len(input_img.shape) == 2:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            
            # 执行深度学习检测
            results = model(input_img, conf=conf_threshold, iou=iou_threshold)
            result = results[0]
            
            # 创建传统方法ROI的掩码
            roi_mask = np.zeros(input_img.shape[:2], dtype=np.uint8)
            for defect in traditional_defects:
                x, y, w, h = defect['x'], defect['y'], defect['w'], defect['h']
                roi_mask[y:y+h, x:x+w] = 255
            
            # 处理深度学习检测结果
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls]
                
                # 获取检测框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                
                # 创建当前检测框的掩码
                detect_mask = np.zeros_like(roi_mask)
                detect_mask[y1:y2, x1:x2] = 255
                
                # 遍历所有传统方法检测到的ROI区域
                for defect in traditional_defects:
                    roi_x, roi_y, roi_w, roi_h = defect['x'], defect['y'], defect['w'], defect['h']
                    
                    # 创建单个ROI区域的掩码
                    single_roi_mask = np.zeros_like(roi_mask)
                    single_roi_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
                    
                    # 计算与单个ROI的交集
                    intersection = cv2.bitwise_and(single_roi_mask, detect_mask)
                    intersection_area = np.count_nonzero(intersection)
                    roi_area = np.count_nonzero(single_roi_mask)
                    
                    # 如果交集面积大于感兴趣区域面积的50%，则保留该检测结果
                    if intersection_area > 0 and roi_area > 0 and intersection_area / roi_area > 0.5:
                        combined_defects.append({
                            "type": name,
                            "confidence": conf,
                            "x": x1,
                            "y": y1,
                            "w": w,
                            "h": h,
                            "detection_method": "combined",
                            "intersection_ratio": intersection_area / roi_area,
                            "roi_x": roi_x,
                            "roi_y": roi_y,
                            "roi_w": roi_w,
                            "roi_h": roi_h
                        })
                        self.add_history(f"接受检测结果: {name}, 与ROI({roi_x},{roi_y},{roi_w},{roi_h})交集比例: {intersection_area / roi_area:.2f}")
                        break  # 一旦找到匹配的ROI就跳出循环
                else:
                    # 如果没有找到匹配的ROI，记录拒绝信息
                    self.add_history(f"拒绝检测结果: {name}, 未找到足够交集的ROI区域")
            # 5. 创建结果可视化
            result_img = input_img.copy()
            
            # 绘制检测结果
            for defect in combined_defects:
                x, y = defect['x'], defect['y']
                w, h = defect['w'], defect['h']
                conf = defect['confidence']
                defect_type = defect['type']
                
                # 使用PIL进行中文文本绘制
                img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
                
                # 绘制矩形框
                draw.rectangle([(x, y), (x+w, y+h)], outline=(0, 255, 0), width=2)
                
                # 添加标签文本
                label = f"{defect_type} ({conf:.2f})"
                draw.text((x, y-25), label, font=font, fill=(0, 255, 0))
                
                result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 6. 显示结果
            self.result_img = result_img
            self.canvas_result.set_image(result_img)
            
            # 7. 保存检测结果
            self.current_defect_info = combined_defects
            
            # 8. 输出检测统计
            self.add_history(f"\n结合方法检测完成，共发现 {len(combined_defects)} 个缺陷")
            for i, defect in enumerate(combined_defects):
                self.add_history(f"缺陷 #{i+1}: {defect['type']} "
                               f"(置信度: {defect['confidence']:.2f})")
            
            return combined_defects
            
        except Exception as e:
            self.add_history(f"结合方法缺陷检测出错: {str(e)}")
            import traceback
            self.add_history(f"错误详情: {traceback.format_exc()}")
            messagebox.showerror("错误", f"结合方法缺陷检测失败: {str(e)}")
            return []


    def create_defect_statistics_visualization(self, defect_info):
        """创建缺陷统计可视化图表"""
        if not defect_info:
            return
            
        # 创建统计窗口
        stats_window = tk.Toplevel(self.root)
        stats_window.title("缺陷检测统计")
        stats_window.geometry("800x600")
        
        # 创建选项卡
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 缺陷类型分布选项卡
        type_tab = ttk.Frame(notebook)
        notebook.add(type_tab, text="缺陷类型分布")
        
        # 置信度分析选项卡
        conf_tab = ttk.Frame(notebook)
        notebook.add(conf_tab, text="置信度分析")
        
        # 空间分布选项卡
        spatial_tab = ttk.Frame(notebook)
        notebook.add(spatial_tab, text="空间分布")
        
        # 创建缺陷类型分布图
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 统计缺陷类型
        defect_types = {}
        for defect in defect_info:
            defect_type = defect['type']
            if defect_type in defect_types:
                defect_types[defect_type] += 1
            else:
                defect_types[defect_type] = 1
        
        # 饼图
        ax1.pie(defect_types.values(), labels=defect_types.keys(), autopct='%1.1f%%')
        ax1.set_title('缺陷类型分布')
        
        # 条形图
        ax2.bar(defect_types.keys(), defect_types.values())
        ax2.set_title('缺陷类型计数')
        ax2.set_ylabel('数量')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas1 = FigureCanvasTkAgg(fig1, type_tab)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建置信度分析图
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 提取置信度数据
        confidences = [defect['confidence'] for defect in defect_info]
        types = [defect['type'] for defect in defect_info]
        
        # 直方图
        ax3.hist(confidences, bins=10)
        ax3.set_title('置信度分布')
        ax3.set_xlabel('置信度')
        ax3.set_ylabel('频率')
        
        # 按类型的箱线图
        type_conf = {}
        for t, c in zip(types, confidences):
            if t not in type_conf:
                type_conf[t] = []
            type_conf[t].append(c)
        
        box_data = [type_conf[t] for t in type_conf]
        ax4.boxplot(box_data, labels=type_conf.keys())
        ax4.set_title('各类型置信度分布')
        ax4.set_ylabel('置信度')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas2 = FigureCanvasTkAgg(fig2, conf_tab)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建空间分布图
        fig3, ax5 = plt.subplots(figsize=(8, 6))
        
        # 提取坐标数据
        x_coords = [defect['x'] + defect['w']/2 for defect in defect_info]
        y_coords = [defect['y'] + defect['h']/2 for defect in defect_info]
        
        # 散点图，不同类型用不同颜色
        unique_types = list(set(types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, t in enumerate(unique_types):
            indices = [j for j, x in enumerate(types) if x == t]
            ax5.scatter([x_coords[j] for j in indices], 
                      [y_coords[j] for j in indices], 
                      color=colors[i], label=t)
        
        ax5.set_title('缺陷空间分布')
        ax5.set_xlabel('X坐标')
        ax5.set_ylabel('Y坐标')
        ax5.legend()
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas3 = FigureCanvasTkAgg(fig3, spatial_tab)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加统计信息表格
        stats_frame = ttk.Frame(stats_window)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        # 创建统计信息表格
        columns = ("指标", "值")
        tree = ttk.Treeview(stats_frame, columns=columns, show="headings", height=8)
        
        # 设置列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # 添加统计数据
        tree.insert("", "end", values=("总缺陷数", len(defect_info)))
        tree.insert("", "end", values=("缺陷类型数", len(defect_types)))
        tree.insert("", "end", values=("平均置信度", f"{np.mean(confidences):.4f}"))
        tree.insert("", "end", values=("最高置信度", f"{np.max(confidences):.4f}"))
        tree.insert("", "end", values=("最低置信度", f"{np.min(confidences):.4f}"))
        
        # 计算面积
        areas = [defect['w'] * defect['h'] for defect in defect_info]
        tree.insert("", "end", values=("平均面积", f"{np.mean(areas):.1f}像素"))
        tree.insert("", "end", values=("最大面积", f"{np.max(areas):.1f}像素"))
        tree.insert("", "end", values=("最小面积", f"{np.min(areas):.1f}像素"))
        
        tree.pack(fill="x", expand=True)
        
        # 添加导出按钮
        btn_frame = ttk.Frame(stats_window)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        def export_statistics():
            # 创建统计分析器实例
            analyzer = StatisticsAnalyzer()
            # 加载当前检测结果
            analyzer.load_single_result(defect_info)
            # 显示摘要对话框
            analyzer.show_summary_dialog(stats_window)
        
        ttk.Button(btn_frame, text="导出统计报告", 
                  command=export_statistics).pack(side="right", padx=5)


    ##################################################################
    # 图像处理参数保存与调用
    ###################################################################
    def save_current_parameters(self):
        """
        保存当前所有参数设置到一个 JSON 文件中。
        保存内容包括目标选择、滤波、锐化、均衡、阈值分割、形态学、配准及预处理顺序等。
        """
        # 收集所有参数
        params = {
            "target": self.target_var.get(),
            "filter_method": self.filter_var.get(),
            "filter_h": self.filter_h.get(),
            "filter_k": self.filter_k.get(),
            "sharpen_method": self.sharpen_var.get(),
            "sharpen_w": self.sharpen_w.get(),
            "eq_method": self.eq_var.get(),
            "eq_clip": self.eq_clip.get(),
            "thresh_method": self.thresh_method_var.get(),
            "manual_thresh": self.manual_thresh_val.get(),
            "adapt_blockSize": self.adapt_blockSize.get(),
            "adapt_C": self.adapt_C.get(),
            "morph_method": self.morph_method_var.get(),
            "morph_kernel": self.morph_kernel.get(),
            "morph_iter": self.morph_iter.get(),
            "reg_method": self.reg_method_var.get(),
            "sift_nfeatures": self.sift_nfeatures.get() if hasattr(self, "sift_nfeatures") else None,
            "harris_thresh": self.harris_thresh.get() if hasattr(self, "harris_thresh") else None,
            "order_step1": self.order_step1.get() if hasattr(self, "order_step1") else None,
            "order_step2": self.order_step2.get() if hasattr(self, "order_step2") else None,
            "order_step3": self.order_step3.get() if hasattr(self, "order_step3") else None
        }
        # 弹出保存文件对话框，用户选择保存文件名
        filename = filedialog.asksaveasfilename(title="保存参数设置", defaultextension=".json",
                                                filetypes=[("JSON Files", "*.json")])
        if not filename:
            return
        with open(filename, "w") as f:
            json.dump(params, f, indent=4)
        self.add_history(f"保存参数设置到: {filename}")

    def load_parameters(self):
        """
        从 JSON 文件中加载参数设置，并将其应用到各个 GUI 控件上，然后更新处理结果。
        """
        filename = filedialog.askopenfilename(title="加载参数设置", filetypes=[("JSON Files", "*.json")])
        if not filename:
            return
        with open(filename, "r") as f:
            params = json.load(f)
        # 应用参数设置
        self.target_var.set(params.get("target", "待测图"))
        self.filter_var.set(params.get("filter_method", "nlm"))
        self.filter_h.set(params.get("filter_h", 10))
        self.filter_k.set(params.get("filter_k", 3))
        self.sharpen_var.set(params.get("sharpen_method", "laplacian"))
        self.sharpen_w.set(params.get("sharpen_w", 1.0))
        self.eq_var.set(params.get("eq_method", "clahe"))
        self.eq_clip.set(params.get("eq_clip", 2.0))
        self.thresh_method_var.set(params.get("thresh_method", "manual"))
        self.manual_thresh_val.set(params.get("manual_thresh", 65))
        self.adapt_blockSize.set(params.get("adapt_blockSize", 31))
        self.adapt_C.set(params.get("adapt_C", 5))
        self.morph_method_var.set(params.get("morph_method", "none"))
        self.morph_kernel.set(params.get("morph_kernel", 3))
        self.morph_iter.set(params.get("morph_iter", 1))
        self.reg_method_var.set(params.get("reg_method", "none"))

        if hasattr(self, "sift_nfeatures") and params.get("sift_nfeatures") is not None:
            self.sift_nfeatures.set(params.get("sift_nfeatures"))
        if hasattr(self, "harris_thresh") and params.get("harris_thresh") is not None:
            self.harris_thresh.set(params.get("harris_thresh"))
        if hasattr(self, "order_step1") and params.get("order_step1") is not None:
            self.order_step1.set(params.get("order_step1"))
        if hasattr(self, "order_step2") and params.get("order_step2") is not None:
            self.order_step2.set(params.get("order_step2"))
        if hasattr(self, "order_step3") and params.get("order_step3") is not None:
            self.order_step3.set(params.get("order_step3"))

        self.update_result_image()
        self.add_history(f"加载参数设置: {filename}")


def main():
    root = tk.Tk()
    app = IntegratedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
