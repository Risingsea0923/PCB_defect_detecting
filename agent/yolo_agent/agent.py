import os
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'torch_cache')
os.environ['YOLO_CACHE_DIR'] = os.path.join(os.path.dirname(__file__), '..', 'yolo_cache') 
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.path.dirname(__file__), '..', 'yolo_cache')
os.environ['XDG_CACHE_HOME'] = os.path.join(os.path.dirname(__file__), '..', 'cache')
#os.environ['TORCH_HOME'] = 'D:/030923/agent/torch_cache'
#os.environ['YOLO_CACHE_DIR'] = 'D:/030923/agent/yolo_cache'
#os.environ['YOLO_CONFIG_DIR'] = 'D:/030923/agent/yolo_cache'
#os.environ['XDG_CACHE_HOME'] = 'D:/030923/agent/cache'
import time
import cv2
import threading
import logging
import torch 
import numpy as np
from datetime import datetime, timedelta
from datetime import datetime
from tkinter import Tk, Toplevel, Label, Button, Frame, Listbox, Scrollbar, END, messagebox, filedialog, StringVar, DoubleVar,Entry,Scale,Checkbutton, IntVar,Text
import tkinter as tk  
from tkinter import ttk  # 用于 Combobox、Progressbar 等控件
from PIL import Image, ImageTk  # 用于图像显示
from ultralytics import YOLO  # YOLOv8/v9 模型接口（假定已安装 ultralytics 库）
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tkinter import IntVar  # 修复 IntVar 问题
# YOLOv9 PCB 缺陷检测 Agent 类
class YOLOAgent:
    def __init__(self, model_path, data_yaml):
        """
        初始化 YOLO Agent：加载模型，设置目录路径和参数，初始化状态和历史记录。
        model_path: 预训练模型权重路径
        data_yaml: YOLO 数据集配置文件路径（包含训练/验证集路径和类别名）
        """
        self.model_path = model_path
        # 加载模型（默认使用提供的模型权重作为初始最佳模型）
        self.model = YOLO(model_path)
        # 初始化预测模型
        self.predict_model = YOLO(model_path)
        self.model_lock = threading.Lock()  # 用于线程安全的模型更新
         # 项目目录结构设置
        base_dir = os.getcwd()
         # 数据库设置
        self.db_path = os.path.join(base_dir, "detection_records.db")
        self.db_enabled = True
        self.data_yaml = data_yaml  # 数据集配置文件路径
                
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("YOLOAgent")
        self.logger.info(f"YOLO Agent initialized. Model loaded from {model_path}")
        
        # 初始化检测线程相关属性
        self.known_files = set()  # 用于跟踪已知文件
        self.detection_interval = 3  # 检测间隔时间（秒）
        self.detection_thread = None  # 检测线程
        self.stop_detection = False  # 停止检测标志
                
        # 初始化数据库
        self._init_database()
        
        # 设置设备（GPU/CPU）
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        # 获取模型类别名列表
        self.class_names = self.model.names
        # 模型性能指标（mAP）用于跟踪最佳模型
        self.best_map = None    # 当前最佳模型的 mAP50-95 指标（若未知则为 None）
        self.last_map = None    # 最近一次训练模型的 mAP50-95 指标
        # 当前最佳模型和最近模型的路径
        self.best_model_path = model_path
        self.last_model_path = None

        self.detection_interval = 5  # 检测间隔时间（秒）
        self.confidence_threshold = 0.6  # 检测置信度阈值
        self.current_image_index = 0  # 当前显示的图像索引
        self.current_images = []  # 当前检测的图像列表

        self.auto_detection = False  # 启动时是否自动检测
        self.detection_timer = None  # 定期检测定时器
        
        # 数据增强设置
        self.augmentation_enabled = True  # 是否启用数据增强
        self.augmentation_count = 3  # 每张图像增强的数量
        
        # 标注工具路径
        self.labelme_path = "labelme"  # labelme可执行文件路径


        # 设置默认路径为网络共享路径
        self.default_network_path = r"\\DESKTOP-1IKPNNH\checkphotos"
        self.new_data_dir = self.default_network_path  # 新的（待检测）PCB图像目录        
        #self.new_data_dir = os.path.join(base_dir, "data", "new")          # 新的（待检测）PCB图像目录
        self.processed_dir = os.path.join(base_dir, "data", "processed")   # 已检测处理过的图像目录
        self.train_images_dir = os.path.join(base_dir, "data", "train", "images")  # 训练用图像保存目录
        self.train_labels_dir = os.path.join(base_dir, "data", "train", "labels")  # 训练用标注保存目录
        


        # 确保以上目录存在
        for d in [self.new_data_dir, self.processed_dir, self.train_images_dir, self.train_labels_dir]:
            os.makedirs(d, exist_ok=True)
        # 检测结果历史记录列表
        self.detections_history = []  # 每个元素为一个字典，包含一次检测的结果和人工审核情况
        # 统计指标初始化
        self.total_images = 0            # 总检测图像数
        self.reviewed_images = 0         # 已人工复检图像数
        self.correct_detections = 0      # 检测正确的图像数（无误检且无漏检）
        self.false_positive_images = 0   # 存在误检的图像数
        self.missed_defect_images = 0    # 存在漏检的图像数
        # 模型训练超参数配置（可自适应调整）
        self.hyperparams = {
            "epochs": 50,
            "batch": 4,
            "imgsz": 640,
            # 可根据需要添加其他超参数（学习率、优化器等）
        }
        self.use_staged_training = False  # 默认使用分阶段训练
        self.use_transfer_learning = False  # 默认使用迁移学习
        self.use_semi_supervised = False  # 默认不使用半监督学习
        self.unlabeled_data_dir = os.path.join(os.getcwd(), "data", "unlabeled")
        self.pseudo_label_conf = 0.7  # 伪标签置信度阈值
        # 状态标志
        self.training = False   # 是否正在训练中
        self.training_paused = False  # 是否暂停训练
        self.detecting = False  # 是否正在进行检测
        # 检测运行计数（用于命名 YOLO 输出文件夹）
        self.det_run_count = 0
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("YOLOAgent")
        self.logger.info(f"YOLO Agent initialized. Model loaded from {model_path}")
        # 加载历史检测记录（如果存在记录文件）
        self.history_file = "detection_history.csv"
        if os.path.exists(self.history_file):
            try:
                import csv
                with open(self.history_file, newline='',encoding='utf-8') as hf:
                    reader = csv.DictReader(hf)
                    for row in reader:
                        # 更新统计计数
                        self.total_images += 1
                        if row.get("reviewed") == "True":
                            self.reviewed_images += 1
                        if row.get("correct") == "True":
                            self.correct_detections += 1
                        if row.get("false_positive") == "True":
                            self.false_positive_images += 1
                        if row.get("missed_defect") == "True":
                            self.missed_defect_images += 1
                        # 将记录加载到历史列表中
                        record = {
                            "timestamp": row.get("timestamp"),
                            "image": row.get("image"),
                            "detected_count": int(row.get("detected_count")),
                            "reviewed": row.get("reviewed") == "True",
                            "correct": row.get("correct") == "True",
                            "false_positive": row.get("false_positive") == "True",
                            "missed_defect": row.get("missed_defect") == "True"
                        }
                        self.detections_history.append(record)
            except Exception as e:
                self.logger.error(f"Failed to load history file: {e}")
        
        # 确保 new_data_dir 路径正确

        if not os.path.exists(self.new_data_dir):
            self.new_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "new")

        #self.new_data_dir = os.path.join(os.getcwd(), "data", "new")
        os.makedirs(self.new_data_dir, exist_ok=True)
        self.logger.info(f"Monitoring directory: {self.new_data_dir}")
        
        # 只有当auto_detection为True时才启动自动检测
        if self.auto_detection:
            self.start_periodic_detection()
        
    def _init_database(self):
        """初始化SQLite数据库"""
        try:
            import sqlite3
            self.logger.info(f"正在初始化数据库，路径: {self.db_path}")
            # 创建数据库连接
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建检测记录表（添加日期和批次字段）
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image TEXT NOT NULL,
                image_name TEXT NOT NULL,
                detected_count INTEGER NOT NULL,
                reviewed INTEGER DEFAULT 0,    
                correct INTEGER DEFAULT 0,          
                false_positive INTEGER DEFAULT 0,   
                missed_defect INTEGER DEFAULT 0,    
                annotated_image TEXT,
                processed_path TEXT,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                batch_id TEXT
            )
            ''')
            
            # 创建缺陷详情表（保持原样）
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS defect_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                FOREIGN KEY (detection_id) REFERENCES detection_records (id)
            )
            ''')
            
            # 创建批次信息表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_info (
                batch_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_images INTEGER DEFAULT 0,
                total_defects INTEGER DEFAULT 0,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                description TEXT
            )
            ''')
            
            # 创建训练记录表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_path TEXT NOT NULL,
                map_value REAL,
                epochs INTEGER,
                batch_size INTEGER,
                duration_seconds INTEGER,
                improved INTEGER    
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("数据库初始化成功")
            
            # 从数据库加载历史记录
            self._load_history_from_db()
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            self.db_enabled = False
    def _load_history_from_db(self):
        """从数据库加载历史检测记录"""
        if not self.db_enabled:
            return
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
            cursor = conn.cursor()
            
            # 查询最近的记录（可以限制数量以提高性能）
            cursor.execute('''
            SELECT * FROM detection_records 
            ORDER BY timestamp DESC 
            LIMIT 1000
            ''')
            
            records = cursor.fetchall()
            self.detections_history = []
            
            # 确保统计计数已初始化
            if not hasattr(self, 'total_images'):
                self.total_images = 0
            if not hasattr(self, 'reviewed_images'):
                self.reviewed_images = 0
            if not hasattr(self, 'correct_detections'):
                self.correct_detections = 0
            if not hasattr(self, 'false_positive_images'):
                self.false_positive_images = 0
            if not hasattr(self, 'missed_defect_images'):
                self.missed_defect_images = 0
            
            # 重置统计计数
            self.total_images = 0
            self.reviewed_images = 0
            self.correct_detections = 0
            self.false_positive_images = 0
            self.missed_defect_images = 0
            
            for record in records:
                # 查询该记录的缺陷详情
                cursor.execute('''
                SELECT * FROM defect_details 
                WHERE detection_id = ?
                ''', (record['id'],))
                
                defects = cursor.fetchall()
                detected_defects = []
                
                for defect in defects:
                    detected_defects.append({
                        "class": defect['class_name'],
                        "bbox": [defect['x1'], defect['y1'], defect['x2'], defect['y2']],
                        "confidence": defect['confidence']
                    })
                
                # 构建记录对象 - 修复这里，使用索引访问而不是get方法
                batch_id = record['batch_id'] if 'batch_id' in record.keys() else ''
                
                detection_record = {
                    "db_id": record['id'],  # 保存数据库ID以便后续更新
                    "timestamp": record['timestamp'],
                    "image": record['image_name'],
                    "detected_count": record['detected_count'],
                    "defects": detected_defects,
                    "reviewed": bool(record['reviewed']),
                    "correct": bool(record['correct']),
                    "false_positive": bool(record['false_positive']),
                    "missed_defect": bool(record['missed_defect']),
                    "annotated_image": record['annotated_image'],
                    "processed_path": record['processed_path'],
                    "batch_id": batch_id  # 使用修复后的批次ID获取方式
                }
                
                self.detections_history.append(detection_record)
                
                # 更新统计计数
                self.total_images += 1
                if record['reviewed']:
                    self.reviewed_images += 1
                if record['correct']:
                    self.correct_detections += 1
                if record['false_positive']:
                    self.false_positive_images += 1
                if record['missed_defect']:
                    self.missed_defect_images += 1
            
            conn.close()
            self.logger.info(f"从数据库加载了 {len(self.detections_history)} 条历史记录")
            
        except Exception as e:
            self.logger.error(f"从数据库加载历史记录失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())



    def manual_review(self, index, false_positive=False, missed_defect=False):
        """
        记录人工复检结果
        
        参数:
            index (int): 检测记录的索引
            false_positive (bool): 是否存在误检
            missed_defect (bool): 是否存在漏检
        """
        try:
            if index < 0 or index >= len(self.detections_history):
                self.logger.error(f"无效的检测记录索引: {index}")
                return False
                
            # 获取记录
            record = self.detections_history[index]
            
            # 更新复检状态
            record["reviewed"] = True
            record["false_positive"] = false_positive
            record["missed_defect"] = missed_defect
            record["correct"] = not (false_positive or missed_defect)
            
            # 更新数据库记录
            if self.db_enabled and os.path.exists(self.db_path):
                try:
                    import sqlite3
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # 准备数据
                    image_name = record.get("image", "")
                    reviewed = 1  # 已复检
                    correct = 1 if record["correct"] else 0
                    false_pos = 1 if false_positive else 0
                    missed = 1 if missed_defect else 0
                    
                    # 更新数据库记录
                    cursor.execute("""
                        UPDATE detection_records 
                        SET reviewed = ?, correct = ?, false_positive = ?, missed_defect = ?
                        WHERE image_name = ?
                    """, (reviewed, correct, false_pos, missed, image_name))
                    
                    conn.commit()
                    conn.close()
                except Exception as e:
                    self.logger.error(f"更新数据库复检记录失败: {e}")
            
            self.logger.info(f"已完成图像 '{record.get('image', '')}' 的人工复检")
            return True
            
        except Exception as e:
            self.logger.error(f"人工复检失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


    def update_predict_model(self, model_path):
        """更新预测模型"""
        with self.model_lock:  # 确保线程安全
            try:
                self.predict_model = YOLO(model_path)
                self.logger.info(f"Predict model updated from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to update predict model: {e}")

    def start_periodic_detection(self):
        """启动定期检测新图片"""
        if self.detection_thread and self.detection_thread.is_alive():
            self.logger.info("Periodic detection is already running")
            return

        def detection_loop():
            while not self.stop_detection:
                try:
                    # 获取当前目录下的所有图片文件
                    current_files = set(
                        f for f in os.listdir(self.new_data_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
                    )
                    
                    # 找出新文件
                    new_files = current_files - self.known_files
                    if new_files:
                        self.logger.info(f"发现 {len(new_files)} 张新图片")
                        # 使用单图像模式检测
                        self.detect_new_data(single_image_mode=True)
                        # 只更新已处理的文件
                        processed_files = set(
                            f for f in os.listdir(self.processed_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
                        )
                        self.known_files.update(processed_files)
                    
                except Exception as e:
                    self.logger.error(f"检测新图片时出错: {e}")
                
                time.sleep(self.detection_interval)

        self.stop_detection = False
        self.detection_thread = threading.Thread(target=detection_loop, daemon=True)
        self.detection_thread.start()
        self.logger.info(f"定期检测，每 {self.detection_interval} 秒检测一次新图片")



    def stop_periodic_detection(self):
        """停止定期检测"""
        self.stop_detection = True
        if self.detection_thread:
            self.detection_thread.join(timeout=1)
        self.logger.info("已停止定期检测")


    def detect_new_data(self, single_image_mode=False):
        """
        检查 new_data_dir 目录，检测其中所有新PCB图像，将结果保存并记录。
        
        single_image_mode: 是否使用单图像模式（一次只检测一张图像）
        返回每张图像检测结果的简要摘要列表。
        """
        if self.detecting:
            self.logger.warning("检测任务已在进行中")
            return None
            
        self.detecting = True
        detection_results = []
        
        try:
            # 获取新图像文件
            image_files = [f for f in os.listdir(self.new_data_dir)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            
            if not image_files:
                self.logger.info("没有新图像需要检测")
                self.detecting = False
                return None
                
            # 创建新批次
            batch_id = None
            if not single_image_mode and len(image_files) > 1:
                # 生成批次ID: YYYYMMDD-序号
                now = datetime.now()
                date_str = now.strftime("%Y%m%d")
                
                if self.db_enabled:
                    import sqlite3
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # 查询当天最后一个批次
                    cursor.execute('''
                    SELECT batch_id FROM batch_info 
                    WHERE year=? AND month=? AND day=?
                    ORDER BY batch_id DESC LIMIT 1
                    ''', (now.year, now.month, now.day))
                    
                    last_batch = cursor.fetchone()
                    
                    if last_batch:
                        # 提取序号并加1
                        last_id = last_batch[0]
                        if '-' in last_id:
                            seq = int(last_id.split('-')[1]) + 1
                        else:
                            seq = 1
                    else:
                        seq = 1
                        
                    batch_id = f"{date_str}-{seq}"
                    
                    # 创建新批次记录
                    cursor.execute('''
                    INSERT INTO batch_info 
                    (batch_id, start_time, year, month, day, total_images, total_defects)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        batch_id, 
                        now.isoformat(), 
                        now.year, now.month, now.day,
                        0,  # 初始图像数
                        0   # 初始缺陷数
                    ))
                    
                    conn.commit()
                    conn.close()
            
            # 处理每个图像
            for image_file in image_files:
                # 如果是单图像模式且已处理过一张图像，则退出
                if single_image_mode and detection_results:
                    break
                    
                image_path = os.path.join(self.new_data_dir, image_file)
                
                # 检测图像
                result = self._detect_image(image_path)
                
                if result:
                    # 添加批次ID
                    if batch_id:
                        result["batch_id"] = batch_id
                    
                    # 确保缺陷信息完整
                    self.logger.info(f"检测到 {len(result.get('defects', []))} 个缺陷")
                    for i, defect in enumerate(result.get('defects', [])):
                        self.logger.info(f"缺陷 {i+1}: 类别={defect.get('class', '未知')}, 置信度={defect.get('confidence', 0)}")
                    
                    # 保存到数据库
                    if self.db_enabled:
                        db_id = self._save_detection_to_db(result)
                        if db_id:
                            result["db_id"] = db_id
                    
                    # 添加到检测历史
                    self.detections_history.append(result)
                    detection_results.append(result)
                    
                    # 更新统计数据
                    self.total_images += 1
                    
                    # 移动已处理的图像
                    try:
                        processed_path = os.path.join(self.processed_dir, image_file)
                        # 检查是否跨驱动器操作
                        if os.path.splitdrive(image_path)[0] != os.path.splitdrive(processed_path)[0]:
                            # 跨驱动器操作：先复制后删除
                            import shutil
                            shutil.copy2(image_path, processed_path)
                            os.remove(image_path)
                            self.logger.info(f"已复制并删除文件: {image_file}")
                        else:
                            # 同一驱动器：直接移动
                            os.replace(image_path, processed_path)
                            self.logger.info(f"已移动文件: {image_file}")
                            
                        result["processed_path"] = processed_path
                        
                        # 如果已经保存到数据库，更新处理路径
                        if "db_id" in result:
                            self._update_processed_path_in_db(result["db_id"], processed_path)
                    except Exception as e:
                        self.logger.warning(f"无法处理文件 {image_file}: {e}")
            
            # 如果有GUI界面，更新显示
            if hasattr(self, 'gui') and self.gui:
                # 获取当前日期范围
                if hasattr(self.gui, 'start_date') and hasattr(self.gui, 'end_date'):
                    start_date = self.gui.start_date
                    end_date = self.gui.end_date
                    # 更新趋势图
                    self.gui.update_trend_plot(start_date, end_date)
                
                # 更新历史记录列表
                if hasattr(self.gui, 'update_history_list'):
                    self.gui.update_history_list()
                
                # 更新统计信息
                if hasattr(self.gui, 'update_statistics'):
                    self.gui.update_statistics()
            
            # 更新批次信息
            if batch_id and self.db_enabled and len(detection_results) > 0:
                try:
                    import sqlite3
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # 计算总缺陷数
                    total_defects = sum(len(r.get("defects", [])) for r in detection_results)
                    
                    # 更新批次记录
                    cursor.execute('''
                    UPDATE batch_info 
                    SET end_time = ?, total_images = ?, total_defects = ?
                    WHERE batch_id = ?
                    ''', (
                        datetime.now().isoformat(),
                        len(detection_results),
                        total_defects,
                        batch_id
                    ))
                    
                    conn.commit()
                    conn.close()
                except Exception as e:
                    self.logger.error(f"更新批次信息失败: {e}")
            
            self.logger.info(f"检测完成，处理了 {len(detection_results)} 张图像")
            return detection_results
            
        except Exception as e:
            self.logger.error(f"检测过程出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
        finally:
            self.detecting = False





    def _detect_image(self, image_path):
        """检测单张图像"""
        try:
            # 确保图像存在
            if not os.path.exists(image_path):
                self.logger.warning(f"图像不存在: {image_path}")
                return None
                
            img_name = os.path.basename(image_path)
            self.logger.info(f"正在检测图像: {img_name}")
            
            # 使用预测模型进行检测
            results = self.predict_model.predict(
                source=image_path,
                save=True, 
                save_txt=True,
                project="runs/detect", 
                name=f"pcb_det_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                exist_ok=True
            )
            
            # 处理检测结果
            if not results or len(results) == 0:
                self.logger.warning(f"检测结果为空: {img_name}")
                return None
                
            result = results[0]
            
            # 确保结果在 CPU 上以便后处理
            try:
                result_cpu = result.cpu()
            except AttributeError:
                result_cpu = result
                
            boxes = result_cpu.boxes  # YOLO result 的检测框对象
            detected_defects = []
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # 提取每个检测框的信息（坐标、置信度、类别）
                    x1, y1, x2, y2 = boxes.xyxy[i]  # 检测框坐标
                    conf = float(boxes.conf[i]) if boxes.conf is not None else None
                    cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                    class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
                    
                    detected_defects.append({
                        "class": class_name,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf
                    })
            
            # 记录检测结果
            now = datetime.now()
            record = {
                "timestamp": now.isoformat(),
                "image": img_name,
                "detected_count": len(detected_defects),
                "defects": detected_defects,
                "reviewed": False,
                "correct": False,
                "false_positive": False,
                "missed_defect": False,
                "annotated_image": result.save_dir + "/" + img_name,  # YOLO 保存的标注图像路径
                "year": now.year,
                "month": now.month,
                "day": now.day
            }
            
            return record
            
        except Exception as e:
            self.logger.error(f"检测图像失败: {image_path}, 错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None




    def _handle_review_result(self, record, is_correct, window):
        """处理人工复检结果"""
        try:
            # 更新记录状态
            record_id = record.get("db_id")
            if record_id:
                # 更新数据库记录
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                UPDATE detection_records 
                SET reviewed = 1, correct = ?
                WHERE id = ?
                ''', (1 if is_correct else 0, record_id))
                
                conn.commit()
                conn.close()
            
            # 更新内存中的记录
            for i, r in enumerate(self.detections_history):
                if r.get("image") == record.get("image"):
                    self.detections_history[i]["reviewed"] = True
                    self.detections_history[i]["correct"] = is_correct
                    break
            
            # 关闭窗口
            window.destroy()
            
            # 显示确认消息
            messagebox.showinfo("复检完成", "复检结果已保存")
            
        except Exception as e:
            self.logger.error(f"保存复检结果失败: {e}")
            messagebox.showerror("错误", f"保存复检结果失败: {e}")

    def _save_detection_to_db(self, record):
        """将检测记录保存到数据库"""
        try:
            import sqlite3
            import json
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取表结构信息，检查列名
            cursor.execute("PRAGMA table_info(detection_records)")
            columns = [column[1] for column in cursor.fetchall()]
            self.logger.info(f"数据库表列名: {columns}")
            
            # 解析日期组件 - 修复ISO格式时间戳解析
            timestamp = record["timestamp"]
            try:
                # 尝试使用标准格式解析
                date_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # 尝试解析ISO格式时间戳
                    date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    # 如果还是失败，尝试其他常见格式
                    formats = [
                        "%Y-%m-%dT%H:%M:%S.%f",
                        "%Y-%m-%dT%H:%M:%S",
                        "%Y-%m-%d %H:%M:%S.%f"
                    ]
                    for fmt in formats:
                        try:
                            date_obj = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # 如果所有格式都失败，使用当前时间
                        self.logger.warning(f"无法解析时间戳: {timestamp}，使用当前时间")
                        date_obj = datetime.now()
            
            year = date_obj.year
            month = date_obj.month
            day = date_obj.day
            
            # 将缺陷列表转换为JSON字符串
            defects_json = json.dumps(record["defects"])
            self.logger.info(f"保存缺陷数据: {defects_json}")
            
            # 尝试使用动态SQL构建，根据实际存在的列
            fields = []
            values = []
            params = []
            
            # 映射字段
            field_mapping = {
                "timestamp": date_obj.strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": record["image"],
                "image": record["image"],
                "detected_count": record["detected_count"],
                "reviewed": int(record["reviewed"]),
                "correct": int(record["correct"]),
                "false_positive": int(record["false_positive"]),
                "missed_defect": int(record["missed_defect"]),
                "annotated_image": record.get("annotated_image", ""),
                "processed_path": record.get("processed_path", ""),
                "year": year,
                "month": month,
                "day": day,
                "defects": defects_json
            }
            
            # 只有当batch_id存在于记录中且数据库表中有该列时才添加
            if "batch_id" in record and "batch_id" in columns:
                field_mapping["batch_id"] = record.get("batch_id", "")
            
            # 根据实际存在的列构建SQL
            for col in columns:
                if col in field_mapping:
                    fields.append(col)
                    values.append("?")
                    params.append(field_mapping[col])
            
            # 构建并执行SQL
            sql = f"INSERT INTO detection_records ({', '.join(fields)}) VALUES ({', '.join(values)})"
            self.logger.info(f"执行SQL: {sql}")
            cursor.execute(sql, params)
            
            # 获取插入记录的ID
            detection_id = cursor.lastrowid
            
            # 插入缺陷详情
            for defect in record["defects"]:
                cursor.execute('''
                INSERT INTO defect_details 
                (detection_id, class_name, confidence, x1, y1, x2, y2)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_id,
                    defect["class"],
                    defect.get("confidence", 0),
                    defect["bbox"][0],
                    defect["bbox"][1],
                    defect["bbox"][2],
                    defect["bbox"][3]
                ))
            
            conn.commit()
            conn.close()
            
            # 添加成功保存的调试信息
            self.logger.info(f"成功保存检测记录到数据库，ID: {detection_id}，图像: {record['image']}，检测到 {record['detected_count']} 个缺陷")
            
            return detection_id
            
        except Exception as e:
            self.logger.error(f"保存检测记录到数据库失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

            

    def start_training(self):        
        """启动模型训练"""
        if self.training:
            self.logger.info("训练已在进行中")
            return False
            
        # 设置训练标志
        self.training = True
        self.training_paused = False
        self.logger.info("开始训练模型...")
        
        # 记录训练开始时间和原始模型信息
        self.training_start_time = datetime.now()
        self.original_model_path = self.best_model_path
        self.original_map = self.best_map
        
        # 启动训练线程（修复了嵌套函数的问题）
        self.training_thread = threading.Thread(target=self._train_thread_wrapper, daemon=True)
        self.training_thread.start()
        
        return True
    
    def _train_thread_wrapper(self):
        """训练线程包装器，处理异常并在完成后更新UI"""
        try:
            # 执行训练
            self._train_routine()
            
            # 训练完成后在主线程更新UI
            if hasattr(self, 'gui') and self.gui:
                self.gui.master.after(0, self.gui.on_training_complete)
        except Exception as e:
            self.logger.error(f"训练过程出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 在主线程重置UI
            if hasattr(self, 'gui') and self.gui:
                self.gui.master.after(0, self.gui.reset_training_ui)
    
    def _train_routine(self):
        """模型训练的实际实现"""
        try:
            self.logger.info("开始训练模型...")
            
            # 记录训练选项
            self.logger.info(f"训练选项:")
            self.logger.info(f"- 分阶段训练: {'启用' if self.use_staged_training else '禁用'}")
            self.logger.info(f"- 迁移学习: {'启用' if self.use_transfer_learning else '禁用'}")
            self.logger.info(f"- 半监督学习: {'启用' if self.use_semi_supervised else '禁用'}")
            self.logger.info(f"- 数据增强: {'启用' if self.augmentation_enabled else '禁用'}")
            
            # 准备基本训练参数（不包含高级选项）
            train_args = {
                "data": self.data_yaml,
                "epochs": self.epochs,
                "batch": self.batch_size,
                "imgsz": self.image_size,
                "patience": self.patience,
                "device": self.device,
                "workers": self.workers,
                "project": self.project_dir,
                "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "exist_ok": True,
                "optimizer": "SGD",  # 使用SGD优化器
                "lr0": 0.01,  # 初始学习率
                "lrf": 0.01,  # 最终学习率因子
                "momentum": 0.937,  # SGD动量
                "weight_decay": 0.0005,  # 权重衰减
                "warmup_epochs": 3.0,  # 预热轮数
                "warmup_momentum": 0.8,  # 预热动量
                "warmup_bias_lr": 0.1,  # 预热偏置学习率
                "box": 7.5,  # 边界框损失权重
                "cls": 0.5,  # 类别损失权重
                "dfl": 1.5,  # DFL损失权重
                "fl_gamma": 0.0,  # 焦点损失gamma
                "label_smoothing": 0.0,  # 标签平滑
                "nbs": 64,  # 标称批量大小
            }
            
            # 只有在启用迁移学习时才使用预训练权重
            if self.use_transfer_learning:
                train_args["pretrained"] = True
                self.logger.info("使用预训练权重进行迁移学习")
            else:
                train_args["pretrained"] = False
                self.logger.info("不使用预训练权重")
            
            # 只有在启用分阶段训练时才设置冻结层
            if self.use_staged_training:
                train_args["freeze"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 冻结前10层
                self.logger.info("启用分阶段训练，冻结前10层")
            else:
                train_args["freeze"] = []  # 不冻结任何层
                self.logger.info("不使用分阶段训练，不冻结任何层")
            
            # 只有在启用数据增强时才添加增强参数
            if self.augmentation_enabled:
                self.logger.info("启用数据增强")
                augmentation_params = {
                    "hsv_h": 0.015,  # 色调增强
                    "hsv_s": 0.7,    # 饱和度增强
                    "hsv_v": 0.4,    # 亮度增强
                    "degrees": 0.0,  # 旋转增强
                    "translate": 0.1, # 平移增强
                    "scale": 0.5,    # 缩放增强
                    "shear": 0.0,    # 剪切增强
                    "perspective": 0.0, # 透视增强
                    "flipud": 0.0,   # 上下翻转概率
                    "fliplr": 0.5,   # 左右翻转概率
                    "mosaic": 1.0,   # 马赛克增强
                    "mixup": 0.0,    # mixup增强
                    "copy_paste": 0.0, # 复制粘贴增强
                }
                train_args.update(augmentation_params)
            else:
                self.logger.info("禁用所有数据增强")
                # 明确禁用所有数据增强
                no_augmentation_params = {
                    "hsv_h": 0.0,
                    "hsv_s": 0.0,
                    "hsv_v": 0.0,
                    "degrees": 0.0,
                    "translate": 0.0,
                    "scale": 0.0,
                    "shear": 0.0,
                    "perspective": 0.0,
                    "flipud": 0.0,
                    "fliplr": 0.0,
                    "mosaic": 0.0,
                    "mixup": 0.0,
                    "copy_paste": 0.0,
                }
                train_args.update(no_augmentation_params)
            
            # 如果启用半监督学习，先处理未标注数据
            if self.use_semi_supervised and os.path.exists(self.unlabeled_data_dir):
                self.logger.info(f"使用半监督学习，处理未标注数据...")
                self.semi_supervised_learning(self.unlabeled_data_dir, self.pseudo_label_conf)
            else:
                self.logger.info("不使用半监督学习")
            
            # 保存当前训练参数
            self.hyperparams = train_args.copy()
            
            # 开始训练
            self.logger.info(f"开始训练模型，参数: {train_args}")
            
            # 使用模型锁确保线程安全
            with self.model_lock:
                # 使用当前最佳模型作为起点
                model = YOLO(self.best_model_path)
                
                # 开始训练
                results = model.train(**train_args)
                
                # 获取训练结果
                best_model_path = results.best
                
                # 如果训练成功并生成了新模型
                if best_model_path and os.path.exists(best_model_path):
                    # 获取新模型的性能指标
                    metrics = results.results_dict
                    best_map = metrics.get('metrics/mAP50-95(B)', 0)
                    
                    # 如果新模型性能更好，更新最佳模型
                    if best_map > self.best_map:
                        self.logger.info(f"新模型性能更好: mAP {best_map:.4f} > {self.best_map:.4f}")
                        self.best_model_path = best_model_path
                        self.best_map = best_map
                        
                        # 更新预测模型
                        self.predict_model = YOLO(best_model_path)
                        
                        # 保存模型信息
                        self._save_model_info()
                    else:
                        self.logger.info(f"新模型性能不如当前最佳模型: mAP {best_map:.4f} <= {self.best_map:.4f}")
                else:
                    self.logger.warning("训练未生成有效的新模型")
            
            self.logger.info("模型训练完成")
            
        except Exception as e:
            self.logger.error(f"训练过程出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        finally:
            # 重置训练标志
            self.training = False
            self.training_paused = False

    def on_training_complete(self):
        """训练完成后的处理"""
        self.reset_training_ui()
        self.log("模型训练完成")
        
        # 生成训练报告
        self.generate_training_report()
        
    def reset_training_ui(self):
        """重置训练相关UI元素"""
        self.train_button.config(state="normal")
        self.pause_button.config(state="disabled")
        self.pause_button.config(text="暂停训练")
        self.progress_var.set(0)
        
    def generate_training_report(self):
        """生成训练模型报告"""
        # 创建训练报告窗口
        report_window = Toplevel(self.master)
        report_window.title("模型训练报告")
        report_window.geometry("700x500")
        
        # 创建报告文本区域
        report_frame = Frame(report_window)
        report_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 添加滚动条
        report_scroll = Scrollbar(report_frame)
        report_scroll.pack(side="right", fill="y")
        
        # 使用Text控件显示报告
        report_text = Text(report_frame, wrap="word", yscrollcommand=report_scroll.set)
        report_text.pack(side="left", fill="both", expand=True)
        report_scroll.config(command=report_text.yview)
        
        # 计算训练时长
        training_duration = datetime.now() - self.training_start_time
        hours, remainder = divmod(training_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}小时{int(minutes)}分钟{int(seconds)}秒"
        
        # 生成报告内容
        report_lines = [
            "# 模型训练报告",
            f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"训练持续时间: {duration_str}",
            "",
            "## 模型信息",
            f"模型路径: {self.agent.best_model_path}",
            f"模型类型: YOLOv9",
            ""
        ]
        
        # 性能对比
        report_lines.append("## 性能对比")
        if hasattr(self.agent, 'best_map') and self.original_map is not None:
            map_change = self.agent.best_map - self.original_map
            change_percent = (map_change / self.original_map * 100) if self.original_map > 0 else float('inf')
            
            report_lines.extend([
                f"原始模型mAP: {self.original_map:.4f}",
                f"新模型mAP: {self.agent.best_map:.4f}",
                f"性能变化: {'+' if map_change >= 0 else ''}{map_change:.4f} ({'+' if change_percent >= 0 else ''}{change_percent:.2f}%)"
            ])
            
            if map_change > 0:
                report_lines.append("结论: 模型性能有所提升 ✓")
            elif map_change == 0:
                report_lines.append("结论: 模型性能无明显变化 ○")
            else:
                report_lines.append("结论: 模型性能有所下降 ✗")
        else:
            report_lines.append("无法获取完整的性能指标进行对比")
            
        report_lines.extend([
            "",
            "## 训练参数",
            f"批次大小: {self.agent.hyperparams.get('batch', 'N/A')}",
            f"训练轮次: {self.agent.hyperparams.get('epochs', 'N/A')}",
            f"学习率: {self.agent.hyperparams.get('lr0', 'N/A')}",
            f"图像尺寸: {self.agent.hyperparams.get('imgsz', 'N/A')}",
            f"训练策略: {'分阶段训练' if getattr(self.agent, 'use_staged_training', False) else '常规训练'}",
            "",
            "## 建议",
        ])
        
        # 根据训练结果给出建议
        if hasattr(self.agent, 'best_map') and self.original_map is not None:
            if map_change > 0.05:
                report_lines.append("- 模型性能显著提升，建议采用新模型进行生产部署")
            elif map_change > 0:
                report_lines.append("- 模型性能有小幅提升，可以考虑采用新模型")
            elif map_change >= -0.02:
                report_lines.append("- 模型性能变化不大，建议增加训练数据或调整训练参数后再次尝试")
            else:
                 report_lines.append("- 模型性能下降，建议检查训练数据质量或调整训练参数")
            
        # 将报告内容显示在文本框中
        report_text.insert("1.0", "\n".join(report_lines))
        report_text.config(state="disabled")  # 设为只读
        
        # 添加导出按钮
        export_frame = Frame(report_window)
        export_frame.pack(pady=10)
        export_button = Button(export_frame, text="导出报告", 
                              command=lambda: self.export_training_report(report_lines))
        export_button.pack(side="left", padx=10)
        close_button = Button(export_frame, text="关闭", 
                             command=report_window.destroy)
        close_button.pack(side="left", padx=10)
    




    def export_training_report(self, report_lines):
        """导出训练报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"d:/030923/agent/reports/training_report_{timestamp}.txt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            self.log(f"训练报告已导出到: {filename}")
            messagebox.showinfo("导出成功", f"训练报告已导出到:\n{filename}")
        except Exception as e:
            self.log(f"导出训练报告失败: {e}")
            messagebox.showerror("导出失败", f"导出训练报告时出错:\n{str(e)}")


    def _update_processed_path_in_db(self, detection_id, processed_path):
        """更新数据库中的图像处理路径"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE detection_records
            SET processed_path = ?
            WHERE id = ?
            ''', (processed_path, detection_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"更新数据库图像处理路径失败: {e}")

    def _update_review_in_db(self, record):
        """更新数据库中的复检结果"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE detection_records
            SET reviewed = ?, correct = ?, false_positive = ?, missed_defect = ?
            WHERE id = ?
            ''', (
                record["reviewed"],
                record["correct"],
                record["false_positive"],
                record["missed_defect"],
                record["db_id"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"更新数据库复检结果失败: {e}")






    def get_db_connection(self):
        """获取数据库连接"""
        if not self.db_enabled:
            self.logger.warning("数据库功能未启用")
            return None
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            self.logger.error(f"连接数据库失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None









    def _update_processed_path_in_db(self, detection_id, processed_path):
        """更新数据库中的图像处理路径"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            UPDATE detection_records
            SET processed_path = ?
            WHERE id = ?
            ''', (processed_path, detection_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"更新数据库图像处理路径失败: {e}")
            
    def get_detection_stats_by_period(self, period_type='month', start_date=None, end_date=None):
        """
        获取指定时间段内的检测统计数据
        
        period_type: 'day', 'month', 'year' 统计周期类型
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        
        返回: 统计数据字典列表
        """
        if not hasattr(self, 'db_path'):
            return {"error": "数据库未初始化"}
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
            cursor = conn.cursor()
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("timestamp >= ?")
                query_params.append(f"{start_date} 00:00:00")
                
            if end_date:
                query_conditions.append("timestamp <= ?")
                query_params.append(f"{end_date} 23:59:59")
                
            where_clause = " AND ".join(query_conditions) if query_conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 根据周期类型选择分组字段
            if period_type == 'day':
                group_by = "year, month, day"
                date_format = "printf('%04d-%02d-%02d', year, month, day)"
            elif period_type == 'month':
                group_by = "year, month"
                date_format = "printf('%04d-%02d', year, month)"
            else:  # year
                group_by = "year"
                date_format = "printf('%04d', year)"
            
            # 执行统计查询
            query = f'''
            SELECT 
                {date_format} AS period,
                COUNT(*) AS total_count,
                SUM(detected_count) AS defect_count,
                SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) AS reviewed_count,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN false_positive = 1 THEN 1 ELSE 0 END) AS false_positive_count,
                SUM(CASE WHEN missed_defect = 1 THEN 1 ELSE 0 END) AS missed_defect_count
            FROM detection_records
            {where_clause}
            GROUP BY {group_by}
            ORDER BY {group_by}
            '''
            
            cursor.execute(query, query_params)
            
            # 处理结果
            results = []
            for row in cursor.fetchall():
                results.append({
                    "period": row['period'],
                    "total_count": row['total_count'],
                    "defect_count": row['defect_count'] or 0,
                    "reviewed_count": row['reviewed_count'],
                    "correct_count": row['correct_count'],
                    "false_positive_count": row['false_positive_count'],
                    "missed_defect_count": row['missed_defect_count'],
                    "accuracy_rate": (row['correct_count'] / row['reviewed_count'] * 100) if row['reviewed_count'] > 0 else 0,
                    "false_positive_rate": (row['false_positive_count'] / row['reviewed_count'] * 100) if row['reviewed_count'] > 0 else 0,
                    "missed_defect_rate": (row['missed_defect_count'] / row['reviewed_count'] * 100) if row['reviewed_count'] > 0 else 0
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"获取检测统计数据失败: {e}")
            return {"error": str(e)}
    
    def get_defect_type_distribution(self, start_date=None, end_date=None):
        """
        获取缺陷类型分布统计
        
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        
        返回: 缺陷类型分布数据字典
        """
        if not hasattr(self, 'db_path'):
            return {"error": "数据库未初始化"}
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("r.timestamp >= ?")
                query_params.append(f"{start_date} 00:00:00")
                
            if end_date:
                query_conditions.append("r.timestamp <= ?")
                query_params.append(f"{end_date} 23:59:59")
                
            where_clause = " AND ".join(query_conditions) if query_conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 执行缺陷类型分布查询
            query = f'''
            SELECT 
                d.class_name,
                COUNT(*) AS defect_count,
                AVG(d.confidence) AS avg_confidence
            FROM defect_details d
            JOIN detection_records r ON d.detection_id = r.id
            {where_clause}
            GROUP BY d.class_name
            ORDER BY defect_count DESC
            '''
            
            cursor.execute(query, query_params)
            
            # 处理结果
            results = {}
            total_defects = 0
            
            for row in cursor.fetchall():
                class_name = row['class_name']
                defect_count = row['defect_count']
                total_defects += defect_count
                
                results[class_name] = {
                    "count": defect_count,
                    "avg_confidence": row['avg_confidence']
                }
            
            # 计算百分比
            if total_defects > 0:
                for class_name in results:
                    results[class_name]["percentage"] = (results[class_name]["count"] / total_defects) * 100
            
            conn.close()
            return {
                "total_defects": total_defects,
                "distribution": results
            }
            
        except Exception as e:
            self.logger.error(f"获取缺陷类型分布失败: {e}")
            return {"error": str(e)}
    
    def get_detection_performance_trend(self, period_type='month', limit=12):
        """
        获取检测性能趋势数据
        
        period_type: 'day', 'month', 'year' 统计周期类型
        limit: 返回的最近周期数量
        
        返回: 性能趋势数据列表
        """
        if not hasattr(self, 'db_path'):
            return {"error": "数据库未初始化"}
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 根据周期类型选择分组字段
            if period_type == 'day':
                group_by = "year, month, day"
                date_format = "printf('%04d-%02d-%02d', year, month, day)"
                order_by = "year DESC, month DESC, day DESC"
            elif period_type == 'month':
                group_by = "year, month"
                date_format = "printf('%04d-%02d', year, month)"
                order_by = "year DESC, month DESC"
            else:  # year
                group_by = "year"
                date_format = "printf('%04d', year)"
                order_by = "year DESC"
            
            # 执行趋势查询
            query = f'''
            SELECT 
                {date_format} AS period,
                COUNT(*) AS total_count,
                SUM(detected_count) AS defect_count,
                SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) AS reviewed_count,
                SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN false_positive = 1 THEN 1 ELSE 0 END) AS false_positive_count,
                SUM(CASE WHEN missed_defect = 1 THEN 1 ELSE 0 END) AS missed_defect_count
            FROM detection_records
            GROUP BY {group_by}
            ORDER BY {order_by}
            LIMIT ?
            '''
            
            cursor.execute(query, (limit,))
            
            # 处理结果
            results = []
            for row in cursor.fetchall():
                reviewed_count = row['reviewed_count'] or 0
                
                # 计算各种比率
                accuracy_rate = (row['correct_count'] / reviewed_count * 100) if reviewed_count > 0 else 0
                false_positive_rate = (row['false_positive_count'] / reviewed_count * 100) if reviewed_count > 0 else 0
                missed_defect_rate = (row['missed_defect_count'] / reviewed_count * 100) if reviewed_count > 0 else 0
                
                results.append({
                    "period": row['period'],
                    "total_count": row['total_count'],
                    "defect_count": row['defect_count'] or 0,
                    "reviewed_count": reviewed_count,
                    "accuracy_rate": accuracy_rate,
                    "false_positive_rate": false_positive_rate,
                    "missed_defect_rate": missed_defect_rate
                })
            
            # 反转结果以按时间顺序排列
            results.reverse()
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"获取检测性能趋势失败: {e}")
            return {"error": str(e)}
    
    def generate_annual_report(self, year=None):
        """
        生成年度检测报告
        
        year: 指定年份，默认为当前年份
        
        返回: 年度报告数据字典
        """
        if not hasattr(self, 'db_path'):
            return {"error": "数据库未初始化"}
            
        # 如果未指定年份，使用当前年份
        if year is None:
            year = datetime.now().year
            
        try:
            # 获取年度总体统计
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            # 获取月度统计数据
            monthly_stats = self.get_detection_stats_by_period(period_type='month', 
                                                              start_date=start_date, 
                                                              end_date=end_date)
            
            # 获取缺陷类型分布
            defect_distribution = self.get_defect_type_distribution(start_date=start_date, 
                                                                   end_date=end_date)
            
            # 计算年度总体指标
            total_images = sum(month["total_count"] for month in monthly_stats) if isinstance(monthly_stats, list) else 0
            total_defects = sum(month["defect_count"] for month in monthly_stats) if isinstance(monthly_stats, list) else 0
            total_reviewed = sum(month["reviewed_count"] for month in monthly_stats) if isinstance(monthly_stats, list) else 0
            total_correct = sum(month["correct_count"] for month in monthly_stats) if isinstance(monthly_stats, list) else 0
            total_false_positive = sum(month["false_positive_count"] for month in monthly_stats) if isinstance(monthly_stats, list) else 0
            total_missed = sum(month["missed_defect_count"] for month in monthly_stats) if isinstance(monthly_stats, list) else 0
            
            # 计算年度准确率指标
            accuracy_rate = (total_correct / total_reviewed * 100) if total_reviewed > 0 else 0
            false_positive_rate = (total_false_positive / total_reviewed * 100) if total_reviewed > 0 else 0
            missed_rate = (total_missed / total_reviewed * 100) if total_reviewed > 0 else 0
            
            # 查找检测量最高的月份
            peak_month = None
            peak_month_count = 0
            if isinstance(monthly_stats, list) and monthly_stats:
                for month in monthly_stats:
                    if month["total_count"] > peak_month_count:
                        peak_month_count = month["total_count"]
                        peak_month = month["period"]
            
            # 构建年度报告
            annual_report = {
                "year": year,
                "total_images": total_images,
                "total_defects": total_defects,
                "total_reviewed": total_reviewed,
                "accuracy_rate": accuracy_rate,
                "false_positive_rate": false_positive_rate,
                "missed_rate": missed_rate,
                "peak_month": peak_month,
                "peak_month_count": peak_month_count,
                "monthly_stats": monthly_stats,
                "defect_distribution": defect_distribution
            }
            
            return annual_report
            
        except Exception as e:
            self.logger.error(f"生成年度报告失败: {e}")
            return {"error": str(e)}
    

    def export_detection_data(self, start_date=None, end_date=None, export_path=None):
        """
        导出指定时间段内的检测数据到CSV文件
        
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        export_path: 导出文件路径，默认为当前目录下的'detection_export_YYYYMMDD.csv'
        """
        if not hasattr(self, 'db_path'):
            self.logger.error("数据库未初始化，无法导出数据")
            return None
            
        # 如果未指定导出路径，使用默认路径
        if export_path is None:
            timestamp = datetime.now().strftime('%Y%m%d')
            export_path = os.path.join(os.getcwd(), f"detection_export_{timestamp}.csv")
            
        try:
            import sqlite3
            import csv
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("r.timestamp >= ?")
                query_params.append(f"{start_date} 00:00:00")
                
            if end_date:
                query_conditions.append("r.timestamp <= ?")
                query_params.append(f"{end_date} 23:59:59")
                
            where_clause = " AND ".join(query_conditions) if query_conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 执行查询
            query = f'''
            SELECT 
                r.id, r.timestamp, r.image_name, r.detected_count, 
                r.reviewed, r.correct, r.false_positive, r.missed_defect,
                r.annotated_image, r.processed_path
            FROM detection_records r
            {where_clause}
            ORDER BY r.timestamp
            '''
            
            cursor.execute(query, query_params)
            records = cursor.fetchall()
            
            # 写入CSV文件
            with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'timestamp', 'image_name', 'detected_count', 
                             'reviewed', 'correct', 'false_positive', 'missed_defect',
                             'defect_details', 'annotated_image', 'processed_path']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in records:
                    # 获取该记录的缺陷详情
                    cursor.execute('''
                    SELECT class_name, confidence, x1, y1, x2, y2
                    FROM defect_details
                    WHERE detection_id = ?
                    ''', (record['id'],))
                    
                    defects = cursor.fetchall()
                    defect_details = []
                    
                    for defect in defects:
                        defect_details.append(f"{defect['class_name']}({defect['confidence']:.2f})")
                    
                    # 写入记录
                    writer.writerow({
                        'id': record['id'],
                        'timestamp': record['timestamp'],
                        'image_name': record['image_name'],
                        'detected_count': record['detected_count'],
                        'reviewed': record['reviewed'],
                        'correct': record['correct'],
                        'false_positive': record['false_positive'],
                        'missed_defect': record['missed_defect'],
                        'defect_details': '; '.join(defect_details),
                        'annotated_image': record['annotated_image'],
                        'processed_path': record['processed_path']
                    })
            
            conn.close()
            self.logger.info(f"检测数据已导出到: {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"导出检测数据失败: {e}")
            return None
    
    def generate_detection_report(self, start_date=None, end_date=None, output_dir=None):
        """
        生成指定时间段内的图文并茂的详细检测报告
        
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        output_dir: 报告输出目录，默认为'reports'子目录
        
        返回: 生成的报告文件路径
        """
        if not hasattr(self, 'db_path'):
            self.logger.error("数据库未初始化，无法生成报告")
            return None
            
        # 设置默认输出目录
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        period_str = ""
        if start_date and end_date:
            period_str = f"{start_date}_to_{end_date}_"
        elif start_date:
            period_str = f"from_{start_date}_"
        elif end_date:
            period_str = f"until_{end_date}_"
            
        report_filename = f"detection_report_{period_str}{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        try:
            # 获取统计数据
            period_stats = self.get_detection_stats_by_period(period_type='month', 
                                                             start_date=start_date, 
                                                             end_date=end_date)
            
            defect_distribution = self.get_defect_type_distribution(start_date=start_date, 
                                                                   end_date=end_date)
            
            performance_trend = self.get_detection_performance_trend(period_type='month', limit=12)
            
            # 获取样本图像（最多10张）
            sample_images = self._get_sample_detection_images(start_date, end_date, limit=10)
            
            # 计算总体指标
            total_images = sum(period["total_count"] for period in period_stats) if isinstance(period_stats, list) else 0
            total_defects = sum(period["defect_count"] for period in period_stats) if isinstance(period_stats, list) else 0
            total_reviewed = sum(period["reviewed_count"] for period in period_stats) if isinstance(period_stats, list) else 0
            
            if total_reviewed > 0:
                accuracy_rate = sum(period["correct_count"] for period in period_stats) / total_reviewed * 100
                false_positive_rate = sum(period["false_positive_count"] for period in period_stats) / total_reviewed * 100
                missed_rate = sum(period["missed_defect_count"] for period in period_stats) / total_reviewed * 100
            else:
                accuracy_rate = 0
                false_positive_rate = 0
                missed_rate = 0
            
            # 生成HTML报告
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import base64
            from io import BytesIO
            import numpy as np
            
            # 创建图表目录
            charts_dir = os.path.join(output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # 生成缺陷类型分布饼图
            plt.figure(figsize=(8, 6))
            if defect_distribution and "distribution" in defect_distribution:
                labels = list(defect_distribution["distribution"].keys())
                sizes = [defect_distribution["distribution"][label]["count"] for label in labels]
                if sum(sizes) > 0:  # 确保有数据
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title('缺陷类型分布')
                    
                    # 保存饼图
                    pie_chart_path = os.path.join(charts_dir, f"defect_distribution_{timestamp}.png")
                    plt.savefig(pie_chart_path)
                    plt.close()
                else:
                    pie_chart_path = None
            else:
                pie_chart_path = None
            
            # 生成检测性能趋势图
            plt.figure(figsize=(10, 6))
            if performance_trend and isinstance(performance_trend, list) and len(performance_trend) > 0:
                periods = [item["period"] for item in performance_trend]
                accuracy_rates = [item["accuracy_rate"] for item in performance_trend]
                false_positive_rates = [item["false_positive_rate"] for item in performance_trend]
                missed_rates = [item["missed_defect_rate"] for item in performance_trend]
                
                plt.plot(periods, accuracy_rates, 'g-', label='正确率')
                plt.plot(periods, false_positive_rates, 'r-', label='误检率')
                plt.plot(periods, missed_rates, 'b-', label='漏检率')
                
                plt.xlabel('时间周期')
                plt.ylabel('百分比 (%)')
                plt.title('检测性能趋势')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # 保存趋势图
                trend_chart_path = os.path.join(charts_dir, f"performance_trend_{timestamp}.png")
                plt.savefig(trend_chart_path)
                plt.close()
            else:
                trend_chart_path = None
            
            # 生成月度检测量柱状图
            plt.figure(figsize=(10, 6))
            if period_stats and isinstance(period_stats, list) and len(period_stats) > 0:
                periods = [item["period"] for item in period_stats]
                counts = [item["total_count"] for item in period_stats]
                defect_counts = [item["defect_count"] for item in period_stats]
                
                x = np.arange(len(periods))
                width = 0.35
                
                plt.bar(x - width/2, counts, width, label='总检测量')
                plt.bar(x + width/2, defect_counts, width, label='缺陷数量')
                
                plt.xlabel('时间周期')
                plt.ylabel('数量')
                plt.title('月度检测量统计')
                plt.xticks(x, periods, rotation=45)
                plt.legend()
                plt.grid(True, axis='y')
                plt.tight_layout()
                
                # 保存柱状图
                bar_chart_path = os.path.join(charts_dir, f"monthly_detection_{timestamp}.png")
                plt.savefig(bar_chart_path)
                plt.close()
            else:
                bar_chart_path = None
            # 构建HTML报告内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>模型对比报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .comparison {{ margin: 30px 0; }}
                    .comparison-images {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                    .comparison-image {{ width: 30%; margin-bottom: 15px; }}
                    .comparison-image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>PCB缺陷检测模型对比报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="summary">
                        <h2>模型信息</h2>
                        <table>
                            <tr>
                                <th>属性</th>
                                <th>{model1_name}</th>
                                <th>{model2_name}</th>
                            </tr>
                            <tr>
                                <td>文件路径</td>
                                <td>{model1_path}</td>
                                <td>{model2_path}</td>
                            </tr>
                            <tr>
                                <td>文件大小</td>
                                <td>{os.path.getsize(model1_path) / (1024 * 1024):.2f} MB</td>
                                <td>{os.path.getsize(model2_path) / (1024 * 1024):.2f} MB</td>
                            </tr>
                            <tr>
                                <td>最后修改时间</td>
                                <td>{datetime.fromtimestamp(os.path.getmtime(model1_path)).strftime('%Y-%m-%d %H:%M:%S')}</td>
                                <td>{datetime.fromtimestamp(os.path.getmtime(model2_path)).strftime('%Y-%m-%d %H:%M:%S')}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <h2>性能对比</h2>
                    
                    <!-- 性能对比图 -->
                    <div class="chart">
                        <img src="{os.path.relpath(comparison_chart_path, output_dir)}" alt="模型性能对比">
                    </div>
                    
                    <!-- 性能指标详情表格 -->
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>{model1_name}</th>
                            <th>{model2_name}</th>
                            <th>差异</th>
                        </tr>
                        <tr>
                            <td>precision</td>
                            <td>{model1_metrics.get('precision', 0):.4f}</td>
                            <td>{model2_metrics.get('precision', 0):.4f}</td>
                            <td>{(model2_metrics.get('precision', 0) - model1_metrics.get('precision', 0)):.4f}</td>
                        </tr>
                        <tr>
                            <td>recall</td>
                            <td>{model1_metrics.get('recall', 0):.4f}</td>
                            <td>{model2_metrics.get('recall', 0):.4f}</td>
                            <td>{(model2_metrics.get('recall', 0) - model1_metrics.get('recall', 0)):.4f}</td>
                        </tr>
                        <tr>
                            <td>mAP50</td>
                            <td>{model1_metrics.get('mAP50', 0):.4f}</td>
                            <td>{model2_metrics.get('mAP50', 0):.4f}</td>
                            <td>{(model2_metrics.get('mAP50', 0) - model1_metrics.get('mAP50', 0)):.4f}</td>
                        </tr>
                        <tr>
                            <td>mAP50-95</td>
                            <td>{model1_metrics.get('mAP50-95', 0):.4f}</td>
                            <td>{model2_metrics.get('mAP50-95', 0):.4f}</td>
                            <td>{(model2_metrics.get('mAP50-95', 0) - model1_metrics.get('mAP50-95', 0)):.4f}</td>
                        </tr>
                    </table>
                    
                    <h2>样本图像预测对比</h2>
                    
                    {self._generate_comparison_images_html(comparison_images, output_dir)}
                    
                    <div class="footer">
                        <p>PCB缺陷检测系统 - 模型对比报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 写入HTML文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"模型对比报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成模型对比报告失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _generate_comparison_images_html(self, comparison_images, output_dir):
        """生成对比图像HTML"""
        if not comparison_images:
            return '<p>无样本图像</p>'
            
        images_html = []
        for item in comparison_images:
            image_html = f"""
            <div class="comparison">
                <h3>{os.path.basename(item['original'])}</h3>
                <div class="comparison-images">
                    <div class="comparison-image">
                        <img src="{os.path.relpath(item['original'], output_dir)}" alt="原始图像">
                        <p>原始图像</p>
                    </div>
                    <div class="comparison-image">
                        <img src="{os.path.relpath(item['model1_pred'], output_dir)}" alt="{model1_name}预测">
                        <p>{model1_name}预测</p>
                    </div>
                    <div class="comparison-image">
                        <img src="{os.path.relpath(item['model2_pred'], output_dir)}" alt="{model2_name}预测">
                        <p>{model2_name}预测</p>
                    </div>
                </div>
            </div>
            """
            images_html.append(image_html)
        return ''.join(images_html)            







    def _generate_period_stats_rows(self, period_stats):
        """生成月度统计数据表格行HTML"""
        if not isinstance(period_stats, list) or not period_stats:
            return '<tr><td colspan="7">无统计数据</td></tr>'
            
        rows = []
        for period in period_stats:
            row = f"""
            <tr>
                <td>{period['period']}</td>
                <td>{period['total_count']}</td>
                <td>{period['defect_count']}</td>
                <td>{period['reviewed_count']}</td>
                <td>{period['accuracy_rate']:.1f}%</td>
                <td>{period['false_positive_rate']:.1f}%</td>
                <td>{period['missed_defect_rate']:.1f}%</td>
            </tr>
            """
            rows.append(row)
        return ''.join(rows)
    
    def _generate_defect_distribution_rows(self, defect_distribution):
        """生成缺陷类型详情表格行HTML"""
        if not defect_distribution or 'distribution' not in defect_distribution or not defect_distribution['distribution']:
            return '<tr><td colspan="4">无缺陷类型数据</td></tr>'
            
        rows = []
        for class_name in defect_distribution['distribution']:
            row = f"""
            <tr>
                <td>{class_name}</td>
                <td>{defect_distribution['distribution'][class_name]['count']}</td>
                <td>{defect_distribution['distribution'][class_name]['percentage']:.1f}%</td>
                <td>{defect_distribution['distribution'][class_name]['avg_confidence']:.2f}</td>
            </tr>
            """
            rows.append(row)
        return ''.join(rows)
    
    def _generate_sample_images_html(self, sample_images, output_dir):
        """生成样本图像HTML"""
        if not sample_images:
            return '<p>无样本图像</p>'
            
        images_html = []
        for img in sample_images:
            image_html = f"""
            <div class="sample-image">
                <img src="{os.path.relpath(img['annotated_image'], output_dir)}" alt="{img['image_name']}">
                <p>{img['image_name']} - 检测到 {img['detected_count']} 处缺陷</p>
                <p>{'✓ 正确' if img['correct'] else ''}{'⚠ 误检' if img['false_positive'] else ''}{'⚠ 漏检' if img['missed_defect'] else ''}</p>
            </div>
            """
            images_html.append(image_html)
        return ''.join(images_html)
    
    def _get_sample_detection_images(self, start_date=None, end_date=None, limit=10):
        """
        获取指定时间段内的样本检测图像
        
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        limit: 返回的样本数量上限
        
        返回: 样本图像信息列表
        """
        if not hasattr(self, 'db_path'):
            return []
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("timestamp >= ?")
                query_params.append(f"{start_date} 00:00:00")
                
            if end_date:
                query_conditions.append("timestamp <= ?")
                query_params.append(f"{end_date} 23:59:59")
                
            # 优先选择已复检的图像
            query_conditions.append("(reviewed = 1 OR 1=1)")
            
            where_clause = " AND ".join(query_conditions) if query_conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 执行查询，优先选择已复检的图像，然后是检测到缺陷的图像
            query = f'''
            SELECT 
                id, timestamp, image_name, detected_count, 
                reviewed, correct, false_positive, missed_defect,
                annotated_image
            FROM detection_records
            {where_clause}
            ORDER BY reviewed DESC, detected_count DESC, timestamp DESC
            LIMIT ?
            '''
            
            cursor.execute(query, query_params + [limit])
            
            # 处理结果
            results = []
            for row in cursor.fetchall():
                # 确保标注图像路径存在
                annotated_image = row['annotated_image']
                if annotated_image and os.path.exists(annotated_image):
                    results.append({
                        "id": row['id'],
                        "timestamp": row['timestamp'],
                        "image_name": row['image_name'],
                        "detected_count": row['detected_count'],
                        "reviewed": bool(row['reviewed']),
                        "correct": bool(row['correct']),
                        "false_positive": bool(row['false_positive']),
                        "missed_defect": bool(row['missed_defect']),
                        "annotated_image": annotated_image
                    })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"获取样本检测图像失败: {e}")
            return []
    


    def export_report_to_pdf(self, report_path, pdf_path=None):
        """
        将HTML报告导出为PDF格式
        
        report_path: HTML报告路径
        pdf_path: 输出PDF路径，默认为HTML报告同名但扩展名为.pdf
        
        返回: 生成的PDF文件路径
        """
        if not os.path.exists(report_path):
            self.logger.error(f"HTML报告文件不存在: {report_path}")
            return None
            
        # 如果未指定PDF路径，使用默认路径
        if pdf_path is None:
            pdf_path = os.path.splitext(report_path)[0] + ".pdf"
            
        try:
            # 尝试使用wkhtmltopdf转换HTML到PDF
            import subprocess
            
            # 检查wkhtmltopdf是否安装
            try:
                subprocess.run(["wkhtmltopdf", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                self.logger.warning("wkhtmltopdf未安装，尝试使用weasyprint")
                raise ImportError("wkhtmltopdf未安装")
                
            # 使用wkhtmltopdf转换
            cmd = ["wkhtmltopdf", "--enable-local-file-access", report_path, pdf_path]
            subprocess.run(cmd, check=True)
            
            self.logger.info(f"报告已导出为PDF: {pdf_path}")
            return pdf_path
            
        except (ImportError, subprocess.SubprocessError):
            # 如果wkhtmltopdf不可用，尝试使用weasyprint
            try:
                from weasyprint import HTML
                HTML(report_path).write_pdf(pdf_path)
                
                self.logger.info(f"报告已导出为PDF: {pdf_path}")
                return pdf_path
                
            except ImportError:
                self.logger.error("无法导出PDF，请安装wkhtmltopdf或weasyprint库")
                return None

    def generate_comparative_report(self, model1_path, model2_path, test_dataset_path, output_dir=None):
        """
        生成两个模型的对比报告
        
        model1_path: 第一个模型路径
        model2_path: 第二个模型路径
        test_dataset_path: 测试数据集路径
        output_dir: 报告输出目录
        
        返回: 生成的报告文件路径
        """
        if not os.path.exists(model1_path) or not os.path.exists(model2_path):
            self.logger.error("模型文件不存在")
            return None
            
        if not os.path.exists(test_dataset_path):
            self.logger.error(f"测试数据集不存在: {test_dataset_path}")
            return None
            
        # 设置默认输出目录
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"model_comparison_{timestamp}.html"
        report_path = os.path.join(output_dir, report_filename)
        
        try:
            # 加载两个模型
            model1_name = os.path.basename(model1_path).split('.')[0]
            model2_name = os.path.basename(model2_path).split('.')[0]
            
            model1 = YOLO(model1_path)
            model2 = YOLO(model2_path)
            
            # 在测试数据集上评估两个模型
            self.logger.info(f"在测试数据集上评估模型 {model1_name}...")
            model1_results = model1.val(data=test_dataset_path)
            
            self.logger.info(f"在测试数据集上评估模型 {model2_name}...")
            model2_results = model2.val(data=test_dataset_path)
            
            # 提取评估指标
            model1_metrics = {}
            model2_metrics = {}
            
            # 提取mAP和其他指标
            if hasattr(model1_results, 'metrics') and isinstance(model1_results.metrics, dict):
                model1_metrics = model1_results.metrics
            elif hasattr(model1_results, 'results'):
                # 旧版本可能返回一个列表形式 [precision, recall, mAP50, mAP50-95, ...]
                res_list = getattr(model1_results, 'results', [])
                if len(res_list) >= 4:
                    model1_metrics = {
                        'precision': res_list[0],
                        'recall': res_list[1],
                        'mAP50': res_list[2],
                        'mAP50-95': res_list[3]
                    }
            
            if hasattr(model2_results, 'metrics') and isinstance(model2_results.metrics, dict):
                model2_metrics = model2_results.metrics
            elif hasattr(model2_results, 'results'):
                res_list = getattr(model2_results, 'results', [])
                if len(res_list) >= 4:
                    model2_metrics = {
                        'precision': res_list[0],
                        'recall': res_list[1],
                        'mAP50': res_list[2],
                        'mAP50-95': res_list[3]
                    }
            
            # 获取测试数据集中的一些样本图像进行对比
            test_images = []
            if os.path.isdir(test_dataset_path):
                # 如果是目录，查找images子目录
                images_dir = os.path.join(test_dataset_path, "images")
                if os.path.exists(images_dir):
                    test_images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
                else:
                    # 直接在目录中查找图像
                    test_images = [os.path.join(test_dataset_path, f) for f in os.listdir(test_dataset_path)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            
            # 随机选择最多5张图像进行对比
            import random
            if len(test_images) > 5:
                sample_images = random.sample(test_images, 5)
            else:
                sample_images = test_images
            
            # 为每个样本图像生成两个模型的预测结果
            comparison_images = []
            for img_path in sample_images:
                img_name = os.path.basename(img_path)
                
                # 模型1预测
                model1_result = model1.predict(source=img_path, save=True, project=output_dir, name=f"model1_{timestamp}")
                model1_pred_path = os.path.join(output_dir, f"model1_{timestamp}", img_name)
                
                # 模型2预测
                model2_result = model2.predict(source=img_path, save=True, project=output_dir, name=f"model2_{timestamp}")
                model2_pred_path = os.path.join(output_dir, f"model2_{timestamp}", img_name)
                
                comparison_images.append({
                    "original": img_path,
                    "model1_pred": model1_pred_path,
                    "model2_pred": model2_pred_path
                })
            
            # 生成对比图表
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            
            # 创建图表目录
            charts_dir = os.path.join(output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # 生成mAP对比柱状图
            plt.figure(figsize=(10, 6))
            metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
            model1_values = [model1_metrics.get(m, 0) for m in metrics]
            model2_values = [model2_metrics.get(m, 0) for m in metrics]
            
            x = range(len(metrics))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], model1_values, width, label=model1_name)
            plt.bar([i + width/2 for i in x], model2_values, width, label=model2_name)
            
            plt.xlabel('评估指标')
            plt.ylabel('得分')
            plt.title('模型性能对比')
            plt.xticks(x, metrics)
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            # 保存对比图
            comparison_chart_path = os.path.join(charts_dir, f"model_comparison_{timestamp}.png")
            plt.savefig(comparison_chart_path)
            plt.close()
            
            # 构建HTML报告内容
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>模型对比报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .chart {{ margin: 20px 0; text-align: center; }}
                    .chart img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .comparison {{ margin: 30px 0; }}
                    .comparison-images {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                    .comparison-image {{ width: 30%; margin-bottom: 15px; }}
                    .comparison-image img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #7f8c8d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>PCB缺陷检测模型对比报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="summary">
                        <h2>模型信息</h2>
                        <table>
                            <tr>
                                <th>属性</th>
                                <th>{model1_name}</th>
                                <th>{model2_name}</th>
                            </tr>
                            <tr>
                                <td>文件路径</td>
                                <td>{model1_path}</td>
                                <td>{model2_path}</td>
                            </tr>
                            <tr>
                                <td>文件大小</td>
                                <td>{os.path.getsize(model1_path) / (1024 * 1024):.2f} MB</td>
                                <td>{os.path.getsize(model2_path) / (1024 * 1024):.2f} MB</td>
                            </tr>
                            <tr>
                                <td>最后修改时间</td>
                                <td>{datetime.fromtimestamp(os.path.getmtime(model1_path)).strftime('%Y-%m-%d %H:%M:%S')}</td>
                                <td>{datetime.fromtimestamp(os.path.getmtime(model2_path)).strftime('%Y-%m-%d %H:%M:%S')}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <h2>性能对比</h2>
                    
                    <!-- 性能对比图 -->
                    <div class="chart">
                        <img src="{os.path.relpath(comparison_chart_path, output_dir)}" alt="模型性能对比">
                    </div>
                    
                    <!-- 性能指标详情表格 -->
                    <table>
                        <tr>
                            <th>指标</th>
                            <th>{model1_name}</th>
                            <th>{model2_name}</th>
                            <th>差异</th>
                        </tr>
                        {self._generate_metrics_comparison_rows(model1_metrics, model2_metrics)}
                    </table>
                    
                    <h2>样本图像预测对比</h2>
                    
                    {self._generate_comparison_images_html(comparison_images, output_dir, model1_name, model2_name)}
                    
                    <div class="footer">
                        <p>PCB缺陷检测系统 - 模型对比报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 写入HTML文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"模型对比报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成模型对比报告失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _generate_metrics_comparison_rows(self, model1_metrics, model2_metrics):
        """生成性能指标对比表格行HTML"""
        metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
        rows = []
        
        for metric in metrics:
            model1_value = model1_metrics.get(metric, 0)
            model2_value = model2_metrics.get(metric, 0)
            diff = model2_value - model1_value
            
            row = f"""
            <tr>
                <td>{metric}</td>
                <td>{model1_value:.4f}</td>
                <td>{model2_value:.4f}</td>
                <td>{diff:.4f}</td>
            </tr>
            """
            rows.append(row)
            
        return ''.join(rows)
    
    def _generate_comparison_images_html(self, comparison_images, output_dir, model1_name, model2_name):
        """生成对比图像HTML"""
        if not comparison_images:
            return '<p>无样本图像</p>'
            
        images_html = []
        for item in comparison_images:
            image_html = f"""
            <div class="comparison">
                <h3>{os.path.basename(item['original'])}</h3>
                <div class="comparison-images">
                    <div class="comparison-image">
                        <img src="{os.path.relpath(item['original'], output_dir)}" alt="原始图像">
                        <p>原始图像</p>
                    </div>
                    <div class="comparison-image">
                        <img src="{os.path.relpath(item['model1_pred'], output_dir)}" alt="{model1_name}预测">
                        <p>{model1_name}预测</p>
                    </div>
                    <div class="comparison-image">
                        <img src="{os.path.relpath(item['model2_pred'], output_dir)}" alt="{model2_name}预测">
                        <p>{model2_name}预测</p>
                    </div>
                </div>
            </div>
            """
            images_html.append(image_html)
        return ''.join(images_html)




    def get_model_info(self, model_path=None):
        """
        获取模型信息
        
        model_path: 模型路径，默认为当前最佳模型
        
        返回: 模型信息字典
        """
        if model_path is None:
            model_path = self.best_model_path
            
        if not os.path.exists(model_path):
            return {"error": "模型文件不存在"}
            
        try:
            model = YOLO(model_path)
            
            # 获取模型基本信息
            model_info = {
                "path": model_path,
                "name": os.path.basename(model_path),
                "size": os.path.getsize(model_path) / (1024 * 1024),  # MB
                "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                "class_names": model.names,
                "num_classes": len(model.names)
            }
            
            # 获取模型架构信息
            if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
                model_info["architecture"] = model.model.yaml
                
            # 获取模型性能指标（如果有）
            try:
                val_results = model.val()
                if hasattr(val_results, 'metrics') and isinstance(val_results.metrics, dict):
                    model_info["metrics"] = val_results.metrics
                elif hasattr(val_results, 'results'):
                    res_list = getattr(val_results, 'results', [])
                    if len(res_list) >= 4:
                        model_info["metrics"] = {
                            'precision': res_list[0],
                            'recall': res_list[1],
                            'mAP50': res_list[2],
                            'mAP50-95': res_list[3]
                        }
            except Exception as e:
                self.logger.warning(f"获取模型性能指标失败: {e}")
                model_info["metrics"] = "未知"
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {"error": str(e)}
    
    def get_detection_stats_by_period(self, period_type='month', start_date=None, end_date=None, limit=12):
        """
        获取按时间周期统计的检测数据
        
        period_type: 时间周期类型，'day', 'month', 'year'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        limit: 返回的记录数量限制
        
        返回: 按时间周期统计的检测数据列表
        """
        if not hasattr(self, 'db_path'):
            return []
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("timestamp >= ?")
                query_params.append(f"{start_date} 00:00:00")
                
            if end_date:
                query_conditions.append("timestamp <= ?")
                query_params.append(f"{end_date} 23:59:59")
                
            where_clause = " AND ".join(query_conditions) if query_conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 根据周期类型构建分组字段
            if period_type == 'day':
                group_field = "year || '-' || printf('%02d', month) || '-' || printf('%02d', day)"
                order_field = "year, month, day"
            elif period_type == 'month':
                group_field = "year || '-' || printf('%02d', month)"
                order_field = "year, month"
            else:  # year
                group_field = "year"
                order_field = "year"
            
            # 执行查询
            query = f'''
            SELECT 
                {group_field} AS period,
                COUNT(*) AS total_count,
                SUM(detected_count) AS defect_count,
                SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) AS reviewed_count,
                SUM(CASE WHEN reviewed = 1 AND correct = 1 THEN 1 ELSE 0 END) AS correct_count,
                SUM(CASE WHEN reviewed = 1 AND false_positive = 1 THEN 1 ELSE 0 END) AS false_positive_count,
                SUM(CASE WHEN reviewed = 1 AND missed_defect = 1 THEN 1 ELSE 0 END) AS missed_defect_count
            FROM detection_records
            {where_clause}
            GROUP BY {group_field}
            ORDER BY {order_field} DESC
            LIMIT ?
            '''
            
            cursor.execute(query, query_params + [limit])
            
            # 处理结果
            results = []
            for row in cursor.fetchall():
                # 计算性能指标
                reviewed_count = row['reviewed_count']
                accuracy_rate = (row['correct_count'] / reviewed_count * 100) if reviewed_count > 0 else 0
                false_positive_rate = (row['false_positive_count'] / reviewed_count * 100) if reviewed_count > 0 else 0
                missed_defect_rate = (row['missed_defect_count'] / reviewed_count * 100) if reviewed_count > 0 else 0
                
                results.append({
                    "period": row['period'],
                    "total_count": row['total_count'],
                    "defect_count": row['defect_count'],
                    "reviewed_count": reviewed_count,
                    "correct_count": row['correct_count'],
                    "false_positive_count": row['false_positive_count'],
                    "missed_defect_count": row['missed_defect_count'],
                    "accuracy_rate": accuracy_rate,
                    "false_positive_rate": false_positive_rate,
                    "missed_defect_rate": missed_defect_rate
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"获取检测统计数据失败: {e}")
            return []
    
    def get_defect_type_distribution(self, start_date=None, end_date=None):
        """
        获取缺陷类型分布统计
        
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        
        返回: 缺陷类型分布统计字典
        """
        if not hasattr(self, 'db_path'):
            return {}
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            query_conditions = []
            query_params = []
            
            if start_date:
                query_conditions.append("r.timestamp >= ?")
                query_params.append(f"{start_date} 00:00:00")
                
            if end_date:
                query_conditions.append("r.timestamp <= ?")
                query_params.append(f"{end_date} 23:59:59")
                
            where_clause = " AND ".join(query_conditions) if query_conditions else ""
            if where_clause:
                where_clause = "WHERE " + where_clause
            
            # 执行查询
            query = f'''
            SELECT 
                d.class_name,
                COUNT(*) AS count,
                AVG(d.confidence) AS avg_confidence
            FROM defect_details d
            JOIN detection_records r ON d.detection_id = r.id
            {where_clause}
            GROUP BY d.class_name
            ORDER BY count DESC
            '''
            
            cursor.execute(query, query_params)
            
            # 处理结果
            results = {}
            total_count = 0
            distribution = {}
            
            for row in cursor.fetchall():
                class_name = row['class_name']
                count = row['count']
                total_count += count
                distribution[class_name] = {
                    "count": count,
                    "avg_confidence": row['avg_confidence']
                }
            
            # 计算百分比
            for class_name in distribution:
                distribution[class_name]["percentage"] = (distribution[class_name]["count"] / total_count * 100) if total_count > 0 else 0
            
            results = {
                "total_count": total_count,
                "distribution": distribution
            }
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"获取缺陷类型分布失败: {e}")
            return {}
    
    def get_detection_performance_trend(self, period_type='month', limit=12):
        """
        获取检测性能趋势数据
        
        period_type: 时间周期类型，'day', 'month', 'year'
        limit: 返回的记录数量限制
        
        返回: 检测性能趋势数据列表
        """
        return self.get_detection_stats_by_period(period_type=period_type, limit=limit)
    
    def create_gui(self):
        """创建图形用户界面"""
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox
            
            # 创建主窗口
            root = tk.Tk()
            root.title("PCB缺陷检测系统")
            root.geometry("1200x800")
            
            # 创建GUI实例
            gui = YOLOAgentGUI(root, self)
            
            # 启动主循环
            root.mainloop()
            
        except Exception as e:
            self.logger.error(f"创建GUI失败: {e}")
            return False
                        







    def pause_training(self):
        """暂停当前训练，保存检查点"""
        if not self.training or self.training_paused:
            self.logger.info("No active training to pause.")
            return False
            
        self.training_paused = True
        self.logger.info("Training paused. You can resume it later.")
        return True

    def resume_training(self):
        """从上次暂停的检查点恢复训练"""
        if not self.training_paused:
            self.logger.info("No paused training to resume.")
            return False
            
        checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "last_training_state.pt")
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"找不到训练检查点: {checkpoint_path}")
            return False
            
        self.logger.info(f"从检查点恢复训练: {checkpoint_path}")
        self.training_paused = False
        
        # 临时保存当前最佳模型路径
        original_best_model = self.best_model_path
        # 设置检查点为当前模型
        self.best_model_path = checkpoint_path
        
        # 在新线程中使用现有的训练方法恢复训练
        resume_thread = threading.Thread(target=self._train_routine, daemon=True)
        resume_thread.start()
        return True

            
    def _train_routine(self):
        """内部：执行模型训练并自动选择或回滚模型。"""
        self.logger.info(f"Starting training with hyperparameters: {self.hyperparams}")
        
        # 使用当前最佳模型作为初始权重继续训练新模型
        try:
            model = YOLO(self.best_model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model for training: {e}")
            self.training = False
            self.training_paused = False
            return
            
        # 获取超参数
        epochs = self.hyperparams.get("epochs", 50)
        batch = self.hyperparams.get("batch", 4)
        imgsz = self.hyperparams.get("imgsz", 640)
        lr0 = self.hyperparams.get("lr0", 0.01)
        
        # 创建训练检查点目录
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "last_training_state.pt")
        
        # 训练时间戳（用于命名输出目录）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 根据训练策略选择不同的训练方式
            if self.use_staged_training:
                # 分阶段训练：先冻结主干网络，再微调全网络
                self.logger.info("使用分阶段训练策略")
                
                # 第一阶段：冻结主干网络，只训练检测头
                first_stage_epochs = min(10, epochs // 3)  # 第一阶段使用总轮次的1/3，最多10轮
                self.logger.info(f"阶段1: 冻结主干网络，训练 {first_stage_epochs} 轮")
                # 确保输出目录存在
                #output_dir = "D:/030923/agent/yolo_output"
                output_dir = os.path.join(os.path.dirname(__file__), '..', 'yolo_output')
                os.makedirs(output_dir, exist_ok=True)
                results = model.train(
                    data=self.data_yaml, 
                    epochs=first_stage_epochs, 
                    batch=batch, 
                    imgsz=imgsz,
                    workers=0,
                    cache=True,
                    project=os.path.join(os.path.dirname(__file__), '..', 'yolo_output'),
                    #project="D:/030923/agent/yolo_output",
                    name=f"train_stage1_{timestamp}",
                    exist_ok=True,
                    freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # 冻结前10层（主干网络）
                    lr0=lr0,
                    device=self.device,
                )
                
                # 检查是否暂停或停止训练
                if self.training_paused or not self.training:
                    # 保存当前训练状态以便稍后恢复
                    try:
                        import shutil
                        stage1_weights = os.path.join(str(results.save_dir), "weights", "last.pt")
                        if os.path.exists(stage1_weights):
                            shutil.copy2(stage1_weights, checkpoint_path)
                            self.logger.info(f"训练暂停，保存检查点: {checkpoint_path}")
                    except Exception as e:
                        self.logger.error(f"保存训练检查点失败: {e}")
                    return
                
                # 第二阶段：解冻所有层，微调整个网络
                second_stage_epochs = epochs - first_stage_epochs
                self.logger.info(f"阶段2: 解冻所有层，微调整个网络，训练 {second_stage_epochs} 轮")
                
                # 从第一阶段的最佳权重继续训练
                stage1_weights = os.path.join(str(results.save_dir), "weights", "best.pt")
                model = YOLO(stage1_weights)
                
                results = model.train(
                    data=self.data_yaml, 
                    epochs=second_stage_epochs, 
                    batch=max(2, batch // 2),  # 减小batch size以避免过拟合
                    imgsz=imgsz,
                    workers=0,
                    cache=True,
                    project=os.path.join(os.path.dirname(__file__), '..', 'yolo_output'),                   
                    #project="D:/030923/agent/yolo_output",
                    name=f"train_stage2_{timestamp}",
                    exist_ok=True,
                    lr0=lr0 * 0.1,  # 使用较小的学习率进行微调
                    device=self.device
                )
            else:
                # 常规训练：一次性训练所有层
                self.logger.info("使用常规训练策略")
                
                # 设置训练参数
                train_args = {
                    "data": self.data_yaml, 
                    "epochs": epochs, 
                    "batch": batch, 
                    "imgsz": imgsz,
                    "workers": 0,
                    "cache": True,
                    "project": os.path.join(os.path.dirname(__file__), '..', 'yolo_output'),                    
                    #"project": "D:/030923/agent/yolo_output",
                    "name": f"train_{timestamp}",
                    "exist_ok": True,
                    "hsv_h": 0.02,
                    "translate": 0.2,
                    "flipud": 0.5,
                    "close_mosaic": 20,
                    "scale": 0.3,
                    "device": self.device,
                    "lr0": lr0
                }
                

                
                # 执行训练
                results = model.train(**train_args)
                
            # 推断训练输出路径以获取新模型权重
            new_weights_path = None
            if hasattr(results, 'save_dir'):
                # Ultralytics YOLOv8 会将训练结果保存在 save_dir 下
                new_weights_path = os.path.join(str(results.save_dir), "weights", "best.pt")
            else:
                # 如果结果对象没有 save_dir，按照约定路径查找
                default_dir = os.path.join("runs", "train")
                if os.path.exists(default_dir):
                    # 寻找最新的训练文件夹
                    subdirs = [os.path.join(default_dir, d) for d in os.listdir(default_dir)]
                    if subdirs:
                        latest_dir = max(subdirs, key=os.path.getmtime)
                        new_weights_path = os.path.join(latest_dir, "weights", "best.pt")
                        
            # 评估新模型的 mAP (使用验证集)
            new_map = None
            try:
                val_results = model.val()
                # 提取 mAP50-95 指标（Ultralytics 返回可能不同，尝试取 metrics 字典或属性）
                if hasattr(val_results, 'metrics') and isinstance(val_results.metrics, dict):
                    new_map = val_results.metrics.get('mAP50-95', None) or val_results.metrics.get('mAP50', None)
                elif hasattr(val_results, 'results'):
                    # 旧版本可能返回一个列表形式 [precision, recall, mAP50, mAP50-95, ...]
                    res_list = getattr(val_results, 'results', [])
                    if len(res_list) >= 4:
                        new_map = res_list[3]  # 假设索引3是 mAP50-95
            except Exception as e:
                self.logger.warning(f"Could not evaluate new model: {e}")
                
            # 决定是否更新最佳模型
            improved = False
            if new_map is not None:
                self.last_map = new_map
                if self.best_map is None or new_map >= self.best_map:
                    improved = True
                    self.logger.info(f"新模型性能提升: mAP {new_map:.4f} > {self.best_map:.4f if self.best_map else 'None'}")
                else:
                    self.logger.info(f"新模型性能未提升: mAP {new_map:.4f} <= {self.best_map:.4f}")
            else:
                self.logger.info("New model mAP unknown, assuming no improvement.")
                
            # 若有提升且存在新权重，则更新最佳模型
            if improved and new_weights_path and os.path.exists(new_weights_path):
                self.best_map = new_map if new_map is not None else self.best_map
                self.last_model_path = new_weights_path
                try:
                    # 同时更新预测模型和训练模型
                    self.predict_model = YOLO(new_weights_path)  # 更新预测模型
                    self.train_model = YOLO(new_weights_path)    # 更新训练模型
                    self.best_model_path = new_weights_path
                    self.class_names = self.train_model.names
                    self.logger.info(f"新模型被采用为最佳模型。权重路径: {new_weights_path}")
                except Exception as e:
                    self.logger.error(f"加载新的最佳模型权重失败: {e}")
            else:
                # 新模型效果不佳，记录最近模型但保持原模型
                if new_weights_path:
                    self.last_model_path = new_weights_path
                self.logger.info("新模型未能提升性能，保持使用现有模型。")
            # 自动调整超参数示例：如果没有提升，尝试增加 epochs 用于下次训练
            if not improved:
                self.hyperparams["epochs"] = int(self.hyperparams.get("epochs", 50) * 1.2)
                self.logger.info(f"Adjusting hyperparameters for next training: epochs -> {self.hyperparams['epochs']}")
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
        finally:
            # 标记训练结束
            self.training = False
            self.training_paused = False


    def load_model(self, model_path):
        """手动加载指定路径的模型权重为当前检测模型。"""
        try:
            self.model = YOLO(model_path)
            self.best_model_path = model_path
            self.class_names = self.model.names
            self.logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            return False


    def semi_supervised_learning(self, unlabeled_dir, confidence_threshold=0.7):
        """
        使用当前模型对未标注数据生成伪标签，然后将高置信度的伪标签加入训练集
        
        unlabeled_dir: 未标注图像目录
        confidence_threshold: 接受伪标签的置信度阈值
        """
        if not os.path.exists(unlabeled_dir):
            self.logger.error(f"未标注数据目录不存在: {unlabeled_dir}")
            return False
            
        self.logger.info(f"开始半监督学习，处理未标注数据: {unlabeled_dir}")
        
        # 获取未标注图像列表
        image_files = [os.path.join(unlabeled_dir, f) for f in os.listdir(unlabeled_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        if not image_files:
            self.logger.info("未找到未标注图像。")
            return False
            
        # 使用当前模型生成伪标签
        self.logger.info(f"使用当前模型为{len(image_files)}张图像生成伪标签...")
        
        pseudo_labeled_count = 0
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            # 预测并生成伪标签
            results = self.model.predict(source=img_path, save_txt=False, save=False)
            
            if not results:
                continue
                
            result = results[0]
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
                
            # 筛选高置信度的检测结果
            high_conf_boxes = []
            for i in range(len(boxes)):
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0
                if conf >= confidence_threshold:
                    cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                    # 获取归一化的中心点坐标和宽高
                    x1, y1, x2, y2 = boxes.xyxy[i]
                    img_h, img_w = result.orig_shape
                    
                    # 转换为YOLO格式：类别 中心x 中心y 宽 高（归一化）
                    x_center = (float(x1) + float(x2)) / 2 / img_w
                    y_center = (float(y1) + float(y2)) / 2 / img_h
                    width = (float(x2) - float(x1)) / img_w
                    height = (float(y2) - float(y1)) / img_h
                    
                    high_conf_boxes.append(f"{cls_id} {x_center} {y_center} {width} {height}")
            
            if high_conf_boxes:
                # 复制图像到训练目录
                dst_img_path = os.path.join(self.train_images_dir, img_name)
                try:
                    import shutil
                    shutil.copy2(img_path, dst_img_path)
                    
                    # 保存伪标签
                    label_path = os.path.join(self.train_labels_dir, f"{base_name}.txt")
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(high_conf_boxes))
                        
                    pseudo_labeled_count += 1
                except Exception as e:
                    self.logger.error(f"处理图像 {img_name} 时出错: {e}")
        
        self.logger.info(f"半监督学习完成，添加了 {pseudo_labeled_count} 张伪标签图像到训练集")
        return pseudo_labeled_count > 0


    def load_pretrained_model(self, pretrained_path, transfer_learning=True):
        """
        加载预训练模型并设置迁移学习参数
        
        pretrained_path: 预训练模型权重路径
        transfer_learning: 是否使用迁移学习（保留主干网络权重，重新初始化检测头）
        """
        try:
            if transfer_learning:
                # 加载预训练模型
                model = YOLO(pretrained_path)
                
                # 保存模型配置，但重新初始化检测头
                model_cfg = model.model.yaml
                if 'head' in model_cfg:
                    self.logger.info("使用迁移学习: 保留主干网络权重，重新初始化检测头")
                    # 在实际训练时，可以通过设置transfer=True参数实现
                
                self.model = model
                self.best_model_path = pretrained_path
                self.class_names = self.model.names
                self.logger.info(f"预训练模型已加载，准备进行迁移学习: {pretrained_path}")
            else:
                # 直接加载模型
                self.load_model(pretrained_path)
            return True
        except Exception as e:
            self.logger.error(f"加载预训练模型失败: {e}")
            return False





    def get_trend_analysis(self):
        """生成基于历史检测数据的趋势分析报告字符串。"""
        if self.reviewed_images == 0:
            return "尚无足够的数据生成趋势分析。"
        success_rate = (self.correct_detections / self.reviewed_images) * 100 if self.reviewed_images > 0 else 0.0
        false_rate = (self.false_positive_images / self.reviewed_images) * 100 if self.reviewed_images > 0 else 0.0
        missed_rate = (self.missed_defect_images / self.reviewed_images) * 100 if self.reviewed_images > 0 else 0.0
        report_lines = [
            f"已检测图像总数: {self.total_images}",
            f"已复检图像数: {self.reviewed_images}",
            f"检测正确率: {success_rate:.1f}%",
            f"误检率: {false_rate:.1f}%",
            f"漏检率: {missed_rate:.1f}%"
        ]
        # （可扩展：按时间序列分析近期趋势）
        return "\n".join(report_lines)

    def cleanup_old_data(self, start_date=None, end_date=None, clean_processed=True, clean_runs=True, clean_db=False):
        """
        清理指定日期范围内的检测数据和模型文件，释放存储空间。
        
        参数:
            start_date (datetime): 开始日期，默认为None表示不限制开始日期
            end_date (datetime): 结束日期，默认为None表示不限制结束日期
            clean_processed (bool): 是否清理已处理的图像目录，默认True
            clean_runs (bool): 是否清理runs目录中的训练和检测结果，默认True
            clean_db (bool): 是否清理数据库中的旧记录，默认False
            
        返回清理的文件数。
        """
        removed_files = 0
        
        # 如果没有提供日期，则使用默认的30天
        if start_date is None and end_date is None:
            now = time.time()
            cutoff = 30 * 24 * 3600  # 默认30天
            
            # 清理 processed_dir 中的旧图像文件
            if clean_processed:
                for fname in os.listdir(self.processed_dir):
                    fpath = os.path.join(self.processed_dir, fname)
                    if os.path.isfile(fpath):
                        mtime = os.path.getmtime(fpath)
                        if now - mtime > cutoff:
                            try:
                                os.remove(fpath)
                                removed_files += 1
                            except Exception as e:
                                self.logger.warning(f"删除文件 {fname} 失败: {e}")
            
            # 清理 runs/detect 和 runs/train 目录中过旧的结果
            if clean_runs:
                #base_dir = "D:\\030923\\agent"
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取agent目录
                for dirpath in ["runs/detect", "runs/train"]:
                    full_dirpath = os.path.join(base_dir, dirpath)
                    if os.path.exists(full_dirpath):
                        for sub in os.listdir(full_dirpath):
                            sub_path = os.path.join(full_dirpath, sub)
                            if os.path.isdir(sub_path):
                                mtime = os.path.getmtime(sub_path)
                                if now - mtime > cutoff:
                                    try:
                                        import shutil
                                        shutil.rmtree(sub_path)
                                        removed_files += 1
                                    except Exception as e:
                                        self.logger.warning(f"删除文件夹 {sub_path} 失败: {e}")
        else:
            # 使用指定的日期范围
            if start_date is None:
                start_date = datetime.min  # 如果没有指定开始日期，使用最小日期
            if end_date is None:
                end_date = datetime.now()  # 如果没有指定结束日期，使用当前日期
                
            # 确保end_date是当天的结束时间
            end_date = datetime.combine(end_date.date(), datetime.max.time())
            
            # 转换为时间戳
            start_timestamp = start_date.timestamp()
            end_timestamp = end_date.timestamp()
            
            self.logger.info(f"清理从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据")
            
            # 清理 processed_dir 中的图像文件
            if clean_processed:
                for fname in os.listdir(self.processed_dir):
                    fpath = os.path.join(self.processed_dir, fname)
                    if os.path.isfile(fpath):
                        mtime = os.path.getmtime(fpath)
                        if start_timestamp <= mtime <= end_timestamp:
                            try:
                                os.remove(fpath)
                                removed_files += 1
                            except Exception as e:
                                self.logger.warning(f"删除文件 {fname} 失败: {e}")
            
            # 清理 runs/detect 和 runs/train 目录中的结果
            if clean_runs:
                base_dir = "D:\\030923\\agent"
                for dirpath in ["runs/detect", "runs/train"]:
                    full_dirpath = os.path.join(base_dir, dirpath)
                    if os.path.exists(full_dirpath):
                        for sub in os.listdir(full_dirpath):
                            sub_path = os.path.join(full_dirpath, sub)
                            if os.path.isdir(sub_path):
                                mtime = os.path.getmtime(sub_path)
                                if start_timestamp <= mtime <= end_timestamp:
                                    try:
                                        import shutil
                                        shutil.rmtree(sub_path)
                                        removed_files += 1
                                    except Exception as e:
                                        self.logger.warning(f"删除文件夹 {sub_path} 失败: {e}")
        
        # 清理数据库中的记录
        if clean_db and self.db_enabled and os.path.exists(self.db_path):
            try:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 准备日期条件
                if start_date is None and end_date is None:
                    # 默认清理30天前的数据
                    cutoff_date = datetime.now() - timedelta(days=30)
                    cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 查询要删除的记录ID
                    cursor.execute("""
                        SELECT id FROM detection_records
                        WHERE timestamp < ?
                    """, (cutoff_str,))
                else:
                    # 使用指定的日期范围
                    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S") if start_date else "0001-01-01 00:00:00"
                    end_str = end_date.strftime("%Y-%m-%d %H:%M:%S") if end_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # 查询要删除的记录ID
                    cursor.execute("""
                        SELECT id FROM detection_records
                        WHERE timestamp BETWEEN ? AND ?
                    """, (start_str, end_str))
                
                record_ids = [row[0] for row in cursor.fetchall()]
                
                if record_ids:
                    # 删除相关的缺陷详情记录
                    placeholders = ','.join(['?'] * len(record_ids))
                    cursor.execute(f"""
                        DELETE FROM defect_details
                        WHERE detection_id IN ({placeholders})
                    """, record_ids)
                    
                    # 删除检测记录
                    cursor.execute(f"""
                        DELETE FROM detection_records
                        WHERE id IN ({placeholders})
                    """, record_ids)
                    
                    removed_files += len(record_ids)
                
                conn.commit()
                conn.close()
                self.logger.info(f"从数据库中删除了 {len(record_ids)} 条记录")
                
            except Exception as e:
                self.logger.error(f"清理数据库记录失败: {e}")
        
        self.logger.info(f"清理完成，共删除了 {removed_files} 个文件/记录。")
        return removed_files


# 图形界面类（基于 Tkinter）
class YOLOAgentGUI:

    def __init__(self, master, agent):
        self.master = master
        self.agent = agent
        self.logger = agent.logger

        # 添加数据库相关属性
        self.db_enabled = True  # 添加数据库启用标志
        self.db_path = agent.db_path  # 从agent获取数据库路径



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
        
        # 初始化时获取images_0中的图像列表并记录
        self.initial_images = self.get_images_list()
        self.logger.info(f"程序启动: 检测到 {len(self.initial_images)} 张原始图像")
        
        # 输出图像列表详情
        if self.initial_images:
            self.logger.info("原始图像列表:")
            for i, img_name in enumerate(sorted(self.initial_images)):
                if i < 10:  # 只显示前10张，避免列表过长
                    self.logger.info(f"  {i+1}. {img_name}")
                elif i == 10:
                    self.logger.info(f"  ... 以及其他 {len(self.initial_images)-10} 张图像")
                    break
        else:
            self.logger.info("原始图像目录为空")
            
        # 记录初始文件列表
        self.record_initial_files()
        
        # 先创建状态变量，然后再创建状态栏
        self.status_var = StringVar(master)
        self.status_var.set("就绪")
        # 创建主框架
        self.main_frame = tk.Frame(self.master)
        
        # 创建状态栏 - 确保它在底部
        self.status_frame = Frame(self.master)
        self.status_frame.pack(side="bottom", fill="x")
        self.status_label = Label(self.status_frame, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        self.status_label.pack(fill="x")

        # 主框架放在状态栏之上
        self.main_frame.pack(fill='both', expand=True)
        
        # 只在master是顶级窗口时设置标题和窗口尺寸
        if isinstance(master, (tk.Tk, tk.Toplevel)):
            master.title("YOLOv9 PCB缺陷检测 Agent")
            master.geometry("1200x800")  # 增加窗口尺寸以适应新布局

        # 创建两个独立的模型实例
        self.predict_model = YOLO(agent.model_path)  # 用于预测的模型
        self.train_model = YOLO(agent.model_path)    # 用于训练的模型

        # 定期检测状态标志
        self.periodic_detection_active = False
        self.current_image_path = None  # 当前显示的图像路径
        self.current_images = []  # 添加
        self.current_image_index = 0
        # 如果有历史检测记录，加载它们
        if hasattr(agent, 'detections_history') and agent.detections_history:
            self.current_images = agent.detections_history.copy()
            self.log(f"初始化时加载了 {len(self.current_images)} 个历史检测记录")

        # 主容器框架（使用已创建的self.main_frame）
        main_frame = self.main_frame
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 添加顶部控制面板
        self._create_top_control_panel()
        # 控制按钮区域
        control_frame = Frame(main_frame)
        control_frame.pack(side="top", fill="x", pady=5)
        
        # 一键检测按钮 - 修改为切换定期检测
        self.detect_button = Button(control_frame, text="开始检测", command=self.toggle_periodic_detection)
        self.detect_button.pack(side="left", padx=5)
        
        # 人工复检按钮
        self.review_button = Button(control_frame, text="人工复检", command=self.open_review_window)
        self.review_button.pack(side="left", padx=5)
        
        # 启动训练按钮
        self.train_button = Button(control_frame, text="开始训练模型", command=self.start_training)
        self.train_button.pack(side="left", padx=5)
        
        # 暂停/恢复训练按钮
        self.pause_button = Button(control_frame, text="暂停训练", command=self.toggle_training_pause, state="disabled")
        self.pause_button.pack(side="left", padx=5)
        
        # 模型选择下拉框和切换按钮
        model_options = ["当前最佳模型"]
        if agent.last_model_path:
            model_options.append("最近训练模型")
        self.model_var = StringVar(master)
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                        values=model_options, state="readonly", width=15)
        self.model_combo.set("当前最佳模型")
        self.model_combo.pack(side="left", padx=5)
        self.apply_model_button = Button(control_frame, text="切换模型", command=self.apply_model_selection)
        self.apply_model_button.pack(side="left", padx=5)
                
        # 添加加载模型权重文件按钮
        self.load_model_button = Button(control_frame, text="加载模型文件", command=self.load_model_file)
        self.load_model_button.pack(side="left", padx=5)
        
        # 清理数据按钮
        self.clean_button = Button(control_frame, text="清理旧数据", command=self.cleanup_data)
        self.clean_button.pack(side="left", padx=5)
        
        # 趋势分析按钮
        self.trend_button = Button(control_frame, text="生成检测报告", command=self.show_trend_analysis)
        self.trend_button.pack(side="left", padx=5)
        
        # 在控制按钮区域添加高级训练选项按钮
        self.advanced_train_button = Button(control_frame, text="高级训练选项", command=self.show_advanced_training)
        self.advanced_train_button.pack(side="left", padx=5)
    
        # 添加数据集扩增按钮
        self.augment_button = Button(control_frame, text="数据集扩增", command=self.augment_dataset)
        self.augment_button.pack(side="left", padx=5)

        # 添加删除数据按钮
        self.delete_data_button = Button(control_frame, text="删除图像与标注", command=self.delete_dataset)
        self.delete_data_button.pack(side="left", padx=5)


        # 确保按钮始终启用
        self.master.after(1000, lambda: self.pass_button.config(state="normal"))
        self.master.after(1000, lambda: self.fail_button.config(state="normal"))
        # 添加状态栏
        status_frame = Frame(self.master)
        status_frame.pack(side="bottom", fill="x")
        self.status_label = Label(status_frame, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        self.status_label.pack(fill="x")
        # 训练进度指示（Progressbar）
        self.progress_var = DoubleVar(master)
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="right", padx=5, fill="x", expand=True)
        
        # 创建主内容区域 - 分为左右两部分
        content_frame = Frame(main_frame)
        content_frame.pack(fill="both", expand=True, pady=5)
        
        # 左侧大画布 - 用于显示检测结果图
        self.left_frame = Frame(content_frame, width=600, height=600, bd=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.left_frame.pack_propagate(False)  # 防止子组件改变frame大小
        
        # 创建画布用于显示图像
        self.image_canvas = tk.Canvas(self.left_frame, bg="black")
        self.image_canvas.pack(fill="both", expand=True)
        self.current_image = None  # 保存当前显示的图像引用

        # 右侧工作区布局调整
        right_frame = Frame(content_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        # 右侧水平分割容器（左右两个工作区）
        right_paned = tk.PanedWindow(right_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        right_paned.pack(fill="both", expand=True)

        # 右侧第一工作区（图像列表和日志）- 左侧面板
        first_workzone = Frame(right_paned, bd=2, relief="groove")
        right_paned.add(first_workzone, minsize=400)  # 设置最小宽度
        
        # 右侧第二工作区（统计图表）- 右侧面板
        second_workzone = Frame(right_paned, bd=2, relief="groove") 
        right_paned.add(second_workzone, minsize=400)

        # 第一工作区内容 - 图像列表和日志
        # 图像列表（上部）
        img_list_frame = Frame(first_workzone, height=200)
        img_list_frame.pack(fill="both", expand=True, pady=(5, 5))
        Label(img_list_frame, text="检测图像列表:").pack(anchor="w", padx=5)
        image_list_container = Frame(img_list_frame)
        image_list_container.pack(fill="both", expand=True, padx=5)
        self.image_listbox = Listbox(image_list_container)
        self.image_listbox.pack(side="left", fill="both", expand=True)
        image_scrollbar = Scrollbar(image_list_container, command=self.image_listbox.yview)
        image_scrollbar.pack(side="right", fill="y")
        self.image_listbox.config(yscrollcommand=image_scrollbar.set)
        
        # 绑定列表选择事件
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 日志区域（下部）
        log_frame = Frame(first_workzone, height=200)
        log_frame.pack(fill="both", expand=True, pady=(5, 5))
        Label(log_frame, text="日志输出:").pack(anchor="w", padx=5)
        log_container = Frame(log_frame)
        log_container.pack(fill="both", expand=True, padx=5)
        self.log_text = Listbox(log_container)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = Scrollbar(log_container, command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scroll.set)

        # 第二工作区内容 - 统计图和趋势图
        stats_paned = tk.PanedWindow(second_workzone, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=5)
        stats_paned.pack(fill="both", expand=True)

        # 当前检测统计（上部）
        stats_frame = Frame(stats_paned)
        stats_paned.add(stats_frame, minsize=150)
        Label(stats_frame, text="当前检测缺陷统计").pack(anchor="w", padx=5)
        self.stats_canvas = tk.Canvas(stats_frame, bg="white")
        self.stats_canvas.pack(fill="both", expand=True, padx=5, pady=(0,5))
        
        # 缺陷检测趋势（下部）
        trend_frame = Frame(stats_paned)
        stats_paned.add(trend_frame, minsize=150)
        Label(trend_frame, text="缺陷检测趋势").pack(anchor="w", padx=5)
        self.trend_canvas = tk.Canvas(trend_frame, bg="white")
        self.trend_canvas.pack(fill="both", expand=True, padx=5, pady=(0,5))

        # 添加logger属性
        self.logger = logging.getLogger("YOLOAgentGUI")
        if not self.logger.handlers:
            # 配置日志处理器
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 设置自动检测间隔（毫秒）
        self.auto_detect_interval = 5000 


        # 初始化完成日志
        self.log("系统初始化完成。")
        
        # 加载历史检测记录到图像列表
        self.update_image_list()
        
        # 尝试导入matplotlib用于绘制统计图
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib.font_manager as fm
            
            # 设置中文字体支持
            # 尝试加载常见的中文字体
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong']
            font_found = False
            
            for font_name in chinese_fonts:
                try:
                    # 检查字体是否可用
                    font_path = fm.findfont(fm.FontProperties(family=font_name))
                    if os.path.exists(font_path) and font_name.lower() not in font_path.lower():
                        continue
                    
                    # 设置全局字体
                    plt_font = {'family': font_name}
                    matplotlib.rc('font', **plt_font)
                    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    
                    self.log(f"使用中文字体: {font_name}")
                    font_found = True
                    break
                except:
                    continue
            
            if not font_found:
                self.log("警告: 未找到合适的中文字体，图表中文可能显示为方块")
                
            self.matplotlib_available = True
            
            # 创建统计图Figure
            self.stats_figure = Figure(figsize=(5, 2), dpi=100)
            self.stats_subplot = self.stats_figure.add_subplot(111)
            self.stats_canvas_widget = FigureCanvasTkAgg(self.stats_figure, self.stats_canvas)
            self.stats_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            
            # 创建趋势图Figure
            self.trend_figure = Figure(figsize=(5, 2), dpi=100)
            self.trend_subplot = self.trend_figure.add_subplot(111)
            self.trend_canvas_widget = FigureCanvasTkAgg(self.trend_figure, self.trend_canvas)
            self.trend_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            
            # 初始绘制空图
            self.update_stats_plot()
            self.update_trend_plot()

            
        except ImportError:
            self.matplotlib_available = False
            self.log("警告: 未安装matplotlib，无法显示统计图表")
            Label(self.stats_canvas, text="未安装matplotlib，无法显示统计图表").pack(pady=20)
            Label(self.trend_canvas, text="未安装matplotlib，无法显示统计图表").pack(pady=20)




    def delete_dataset(self):
        """删除指定目录的图像数据与标注(json与txt)"""
        try:
            # 默认数据集基础目录 - 使用相对路径定位到D:\030923\data\train
            default_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'train'))
            self.log(f"数据集基础目录: {default_base_dir}")
            
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
                    if hasattr(self, 'initial_files') and dir_key and dir_key in self.initial_files:
                        initial_set = set(self.initial_files[dir_key])
                        current_set = set(current_files[name])
                        new_files = current_set - initial_set
                        new_files_count = len(new_files)
                    
                    existing_dirs[name] = (path, len(current_files[name]), new_files_count)
            
            if not existing_dirs:
                messagebox.showwarning("提示", f"在 {base_dir} 中没有找到任何数据目录")
                return
            
            # 创建选择对话框
            delete_window = tk.Toplevel(self.master)
            delete_window.title("选择要删除的数据")
            delete_window.geometry("500x500")  # 增加高度以容纳更多选项
            delete_window.transient(self.master)
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
                        if delete_mode.get() == "new" and hasattr(self, 'initial_files') and dir_key in self.initial_files:
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
                    self.log(f"已删除 {total_deleted} 个文件")
                    
                    # 如果删除了原始图像，更新initial_files
                    if hasattr(self, 'initial_files') and delete_mode.get() == "all":
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
                        
                        self.log("已清空程序记录的相应文件列表")
                    
                    messagebox.showinfo("成功", f"已成功删除 {total_deleted} 个文件")
                    delete_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("错误", f"删除文件时出错: {str(e)}")
                    import traceback
                    self.log(f"错误详情: {traceback.format_exc()}")
            
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
            self.log(f"错误详情: {traceback.format_exc()}")  


    def generate_train_val_split(self, train_ratio=0.8):
        """生成训练集和验证集的单测数据"""
        try:
            # 默认数据集基础目录 - 使用相对路径定位到D:\030923\data\train
            default_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'train'))
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
            progress_window = tk.Toplevel(self.master)
            progress_window.title("生成单测数据")
            progress_window.geometry("400x200")
            progress_window.transient(self.master)
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
                    import shutil
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
                        self.master.after(0, lambda p=progress: progress_var.set(p))
                        
                        if i % 10 == 0:
                            self.master.after(0, lambda i=i: info_label.config(
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
                        self.master.after(0, lambda p=progress: progress_var.set(p))
                        
                        if i % 10 == 0:
                            self.master.after(0, lambda i=i: info_label.config(
                                text=f"正在处理验证集... ({i+1}/{len(val_files)})"))
                            progress_window.update()
                    
                    # 完成后更新UI
                    self.master.after(0, lambda: info_label.config(text="单测数据生成完成!"))
                    self.master.after(0, lambda: progress_var.set(100))
                    self.master.after(2000, progress_window.destroy)
                    
                    # 添加到历史记录
                    self.log(f"单测数据生成完成，训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")
                    self.log(f"训练集图像保存在: {train_img_dir}")
                    self.log(f"训练集标注保存在: {train_label_dir}")
                    self.log(f"验证集图像保存在: {val_img_dir}")
                    self.log(f"验证集标注保存在: {val_label_dir}")
                    
                    # 显示成功消息
                    messagebox.showinfo("成功", 
                        f"单测数据生成完成!\n\n"
                        f"训练集: {len(train_files)} 张图像\n"
                        f"验证集: {len(val_files)} 张图像\n\n"
                        f"数据已保存到相应目录")
                    
                    # 更新initial_files，记录新生成的文件
                    if hasattr(self, 'record_initial_files'):
                        self.record_initial_files()
                    
                except Exception as e:
                    self.master.after(0, lambda: progress_window.destroy())
                    self.log(f"生成单测数据失败: {str(e)}")
                    messagebox.showerror("错误", f"生成单测数据失败: {str(e)}")
                    import traceback
                    self.log(f"错误详情: {traceback.format_exc()}")
            
            # 启动线程
            threading.Thread(target=run_split, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("错误", f"生成单测数据失败: {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")








    def augment_dataset(self):
        """调用img_en.py进行数据增强，只处理新增的图像"""
        try:
            # 默认数据集基础目录
            default_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'train'))
            self.log(f"数据集基础目录: {default_base_dir}")
            
            # 检查默认目录是否存在
            if not os.path.exists(default_base_dir):
                os.makedirs(default_base_dir, exist_ok=True)
                self.log(f"创建默认数据集目录: {default_base_dir}")
            
            # 检查images_0目录是否存在
            default_image_dir = os.path.join(default_base_dir, "images_0")
            if not os.path.exists(default_image_dir):
                os.makedirs(default_image_dir, exist_ok=True)
                self.log(f"创建原始图像目录: {default_image_dir}")
            
            # 检查json目录是否存在
            default_json_dir = os.path.join(default_base_dir, "json")
            if not os.path.exists(default_json_dir):
                os.makedirs(default_json_dir, exist_ok=True)
                self.log(f"创建JSON标注目录: {default_json_dir}")
            
            # 获取当前images_0目录中的图像列表（转换为小写进行比较）
            current_images = set()
            self.log(f"尝试读取目录: {default_image_dir}")
            if os.path.exists(default_image_dir):
                files = os.listdir(default_image_dir)
                self.log(f"目录中的所有文件: {files}")
                current_images = set([f.lower() for f in files 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                self.log(f"筛选后的图像文件: {current_images}")
            else:
                self.log(f"警告：目录不存在: {default_image_dir}")
            
            self.log(f"当前原始图像目录中有 {len(current_images)} 张图像")
            
            # 确保initial_images属性存在
            if not hasattr(self, 'initial_images'):
                self.initial_images = []
            
            # 将initial_images转换为小写的集合
            initial_images_set = set([f.lower() for f in self.initial_images])
            
            # 计算新增的图像（当前图像减去初始图像）
            new_images = current_images - initial_images_set
            new_images_list = list(new_images)
            
            # 调试输出
            self.log(f"程序启动时记录的图像数量: {len(initial_images_set)} 张")
            self.log(f"检测到的新增图像数量: {len(new_images)} 张")
            
            if new_images:
                self.log("新增图像列表:")
                for i, img in enumerate(sorted(new_images)):
                    if i < 10:  # 只显示前10张
                        self.log(f"  {i+1}. {img}")
                    elif i == 10:
                        self.log(f"  ... 以及其他 {len(new_images)-10} 张图像")
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
            progress_window = tk.Toplevel(self.master)
            progress_window.title("数据增强进度")
            progress_window.geometry("400x200")
            progress_window.transient(self.master)
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
                self.log(f"创建JSON目录: {json_dir}")
            
            if not os.path.exists(image_dir):
                os.makedirs(image_dir, exist_ok=True)
                self.log(f"创建原始图像目录: {image_dir}")
            
            # 更新进度信息
            count_label.config(text=f"发现 {len(current_images)} 张图像，其中 {len(new_images)} 张为新增")
            
            if len(new_images) == 0:
                info_label.config(text="没有发现新增图像，无需处理")
                progress_var.set(100)
                self.log("没有发现新增图像，无需进行数据增强")
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
                    import shutil
                    # 使用相对路径导入img_en模块
                    img_en_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', 'img_en.py'))
                    self.log(f"尝试导入模块: {img_en_path}")
                    
                    # 检查文件是否存在
                    if not os.path.exists(img_en_path):
                        raise FileNotFoundError(f"找不到文件: {img_en_path}")
                        
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
                    self.master.after(0, lambda: info_label.config(text="数据增强完成!"))
                    self.master.after(0, lambda: progress_var.set(100))
                    self.master.after(2000, progress_window.destroy)
                    
                    # 获取生成的图像数量
                    generated_images_count = 0
                    if os.path.exists(output_image_dir):
                        generated_images_count = len([f for f in os.listdir(output_image_dir) 
                                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    # 添加到历史记录
                    self.log(f"数据增强完成，共处理 {len(new_images)} 张新增图像")
                    self.log(f"生成了新的增强图像，保存在 {output_image_dir}")
                    self.log(f"对应的YOLO格式标注保存在 {output_label_dir}")
                    
                    # 显示成功消息
                    messagebox.showinfo("成功", 
                        f"数据增强完成!\n\n"
                        f"处理了 {len(new_images)} 张新增图像\n"
                        f"结果保存在:\n{output_image_dir}")
                    
                    # 更新initial_images，将新增的图像添加到记录中
                    self.initial_images.extend(new_images_list)
                    self.log(f"已更新程序记录的图像列表，现在包含 {len(self.initial_images)} 张图像")
                    
                except Exception as e:
                    self.master.after(0, lambda: progress_window.destroy())
                    self.log(f"数据增强失败: {str(e)}")
                    messagebox.showerror("错误", f"数据增强失败: {str(e)}")
                    import traceback
                    self.log(f"错误详情: {traceback.format_exc()}")
            
            # 启动线程
            threading.Thread(target=run_augmentation, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("错误", f"启动数据增强失败: {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")



    def get_images_list(self):
        """获取images_0目录中的图像列表"""
        try:
            # 获取并输出当前工作目录
            current_dir = os.getcwd()
            self.logger.info(f"当前工作目录: {current_dir}")
            
            # 使用相对路径
            # 从agent目录向上两级到达D:\030923，然后进入data\train
            base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'train'))
            self.logger.info(f"数据集目录: {base_dir}")
            input_dir = os.path.join(base_dir, "images_0")    
            
            


            self.logger.info(f"检查图像目录: {input_dir}")
            
            if not os.path.exists(input_dir):
                os.makedirs(input_dir, exist_ok=True)
                self.logger.info(f"创建图像目录: {input_dir}")
                return []
            
            # 获取所有图像文件
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 检查JSON目录
            json_dir = os.path.join(base_dir, "json")
            if os.path.exists(json_dir):
                json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
                self.logger.info(f"JSON标注目录: {json_dir}，包含 {len(json_files)} 个标注文件")
                
                # 检查图像和JSON文件的匹配情况
                matched_count = 0
                for img_file in image_files:
                    base_name = os.path.splitext(img_file)[0]
                    json_file = base_name + ".json"
                    if json_file.lower() in [f.lower() for f in json_files]:
                        matched_count += 1
                
                self.logger.info(f"图像与JSON匹配: {matched_count}/{len(image_files)} 张图像有对应的JSON标注")
            
            return image_files
        except Exception as e:
            self.logger.error(f"获取图像列表失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def record_initial_files(self):
        """记录程序启动时各目录下的文件列表"""
        try:
            # 使用相对路径
            # 从agent目录向上两级到达D:\030923，然后进入data\train
            base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data', 'train'))
            self.logger.info(f"数据集基础目录: {base_dir}")
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=True)
                self.logger.info(f"创建默认数据集目录: {base_dir}")
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
                    
                    self.logger.info(f"记录初始文件: {dir_key} 目录下有 {len(self.initial_files[dir_key])} 个文件")
        except Exception as e:
            self.logger.error(f"记录初始文件失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())


    def change_detection_path(self):
        """修改检测目录路径"""
        from tkinter import filedialog, messagebox
        
        new_path = filedialog.askdirectory(
            title="选择检测目录",
            initialdir=self.agent.new_data_dir
        )
        
        if new_path:
            self.agent.new_data_dir = new_path
            self.path_label.config(text=f"当前检测目录: {self.agent.new_data_dir}")
            self.log(f"已更新检测目录为: {self.agent.new_data_dir}")
            messagebox.showinfo("成功", f"检测目录已更新为:\n{self.agent.new_data_dir}")




    def load_model_file(self):
        """加载自定义模型权重文件"""
        try:
            # 打开文件选择对话框
            model_file = filedialog.askopenfilename(
                title="选择模型权重文件",
                filetypes=[("模型文件", "*.pt *.pth"), ("所有文件", "*.*")]
            )
            
            if not model_file:
                return  # 用户取消了选择
                
            # 更新状态
            self.status_var.set(f"正在加载模型: {os.path.basename(model_file)}...")
            self.master.update()
            
            # 加载模型
            with self.agent.model_lock:  # 确保线程安全
                try:
                    # 先尝试加载为预测模型
                    temp_model = YOLO(model_file)
                    
                    # 如果加载成功，更新预测模型
                    self.agent.predict_model = temp_model
                    self.agent.logger.info(f"已加载模型: {model_file}")
                    
                    # 更新模型路径
                    self.agent.model_path = model_file
                    
                    # 更新状态
                    self.status_var.set(f"模型已加载: {os.path.basename(model_file)}")
                    
                    # 显示成功消息
                    messagebox.showinfo("成功", f"模型已成功加载: {os.path.basename(model_file)}")
                    
                except Exception as e:
                    self.log(f"加载模型失败: {e}")
                    messagebox.showerror("错误", f"加载模型失败: {e}")
                    
        except Exception as e:
            self.log(f"选择模型文件时出错: {e}")
            messagebox.showerror("错误", f"选择模型文件时出错: {e}")

    def _create_top_control_panel(self):
        """创建顶部控制面板，包含定期检测控制和人工复检功能"""
        # 修改：将top_panel放在main_frame内而不是master上
        top_panel = Frame(self.main_frame)
        top_panel.pack(fill="x", pady=5)
        
        # 第一行：定期检测控制和日期范围选择
        detection_frame = Frame(top_panel)
        detection_frame.pack(fill="x", pady=5)
        # 添加置信度阈值输入
        Label(detection_frame, text="置信度阈值:").pack(side="left", padx=5)
        self.confidence_var = StringVar(value=str(self.agent.confidence_threshold))
        confidence_entry = Entry(detection_frame, textvariable=self.confidence_var, width=5)
        confidence_entry.pack(side="left", padx=5)
        
        # 添加应用置信度按钮
        apply_conf_button = Button(detection_frame, text="应用", command=self.apply_confidence_threshold)
        apply_conf_button.pack(side="left", padx=5)
        # 定期检测间隔
        Label(detection_frame, text="检测间隔(秒):").pack(side="left", padx=5)
        self.interval_var = StringVar(value=str(self.agent.detection_interval))
        interval_entry = Entry(detection_frame, textvariable=self.interval_var, width=5)
        interval_entry.pack(side="left", padx=5)
        
        # 添加日期范围选择
        Label(detection_frame, text="开始日期:").pack(side="left", padx=5)
        
        # 导入日期选择器
        try:
            from tkcalendar import DateEntry
            # 创建日期选择器
            today = datetime.now().date()
            one_month_ago = today - timedelta(days=30)
            
            self.start_date = DateEntry(detection_frame, width=10, 
                                      background='darkblue', foreground='white', 
                                      date_pattern='yyyy-mm-dd',
                                      year=one_month_ago.year, month=one_month_ago.month, day=one_month_ago.day)
            self.start_date.pack(side="left", padx=5)
            
            Label(detection_frame, text="结束日期:").pack(side="left", padx=5)
            self.end_date = DateEntry(detection_frame, width=10, 
                                    background='darkblue', foreground='white', 
                                    date_pattern='yyyy-mm-dd')
            self.end_date.pack(side="left", padx=5)
            
            # 添加日期筛选按钮
            self.filter_button = Button(detection_frame, text="筛选", command=self.filter_by_date)
            self.filter_button.pack(side="left", padx=5)
            
            self.date_filter_enabled = True
            
        except ImportError:
            self.log("警告: 未安装tkcalendar模块，日期筛选功能不可用")
            self.date_filter_enabled = False
            # 创建虚拟日期属性以避免AttributeError
            self.start_date = None
            self.end_date = None
        
        # 上一张/下一张按钮
        self.prev_button = Button(detection_frame, text="上一张", command=self._show_prev_image, state="normal")
        self.prev_button.pack(side="left", padx=10)
        
        self.next_button = Button(detection_frame, text="下一张", command=self._show_next_image, state="normal")
        self.next_button.pack(side="left", padx=10)
        
        # 定期检测开关
        self.auto_detect_var = IntVar(value=1 if self.agent.auto_detection else 0)
        auto_detect_check = ttk.Checkbutton(detection_frame, text="自动检测", 
                                          variable=self.auto_detect_var,
                                          command=self.toggle_auto_detection)
        auto_detect_check.pack(side="left", padx=10)
        # 添加路径控制按钮 - 修改为放在detection_frame中，与其他控件在同一行
        self.path_label = Label(detection_frame, text=f"当前检测目录: {self.agent.new_data_dir}")
        self.path_label.pack(side="left", padx=5)
        
        # 修改路径按钮
        self.change_path_btn = Button(
            detection_frame, 
            text="修改路径", 
            command=self.change_detection_path
        )
        self.change_path_btn.pack(side="left", padx=5)
        
        # 通过/不通过按钮也放在同一行
        self.pass_button = Button(detection_frame, text="通过", command=self.mark_as_pass, state="normal")
        self.pass_button.pack(side="left", padx=5)

        self.fail_button = Button(detection_frame, text="不通过", command=self.mark_as_fail, state="normal")
        self.fail_button.pack(side="left", padx=5)

    def apply_confidence_threshold(self):
        """应用用户设置的置信度阈值"""
        try:
            new_threshold = float(self.confidence_var.get())
            if 0.0 <= new_threshold <= 1.0:
                self.agent.confidence_threshold = new_threshold
                self.log(f"置信度阈值已更新为: {new_threshold}")
                self.status_var.set(f"置信度阈值已更新为: {new_threshold}")
            else:
                messagebox.showerror("错误", "置信度阈值必须在0.0到1.0之间")
                # 恢复原值
                self.confidence_var.set(str(self.agent.confidence_threshold))
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")
            # 恢复原值
            self.confidence_var.set(str(self.agent.confidence_threshold))

    def _create_chat_frame(self):
        """创建底部对话窗口"""
        # 创建一个带有边框的框架，使其更明显
        chat_frame = Frame(self.master, bd=2, relief="groove", bg="#f0f0f0")
        # 将对话框放在状态栏下面
        chat_frame.pack(fill="x", padx=10, pady=5, side="bottom", before=self.status_frame)
        
        # 添加标题标签
        Label(chat_frame, text="文本控制与分析", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor="w", padx=5, pady=2)
        
        # 创建响应显示区域
        response_frame = Frame(chat_frame, bg="#f0f0f0")
        response_frame.pack(fill="x", padx=5, pady=5)
        
        # 创建响应文本框
        self.response_text = Text(response_frame, height=4, width=80, font=("Arial", 10), wrap="word")
        self.response_text.pack(side="left", fill="x", expand=True)
        response_scroll = Scrollbar(response_frame, command=self.response_text.yview)
        response_scroll.pack(side="right", fill="y")
        self.response_text.config(yscrollcommand=response_scroll.set)
        
        # 创建输入区域框架
        input_frame = Frame(chat_frame, bg="#f0f0f0")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # 创建对话输入框
        self.chat_input = Entry(input_frame, width=80, font=("Arial", 10))
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", self.send_chat_message)
        
        # 创建图片上传按钮
        self.upload_img_button = Button(input_frame, text="上传图片", command=self.upload_image_for_chat)
        self.upload_img_button.pack(side="left", padx=5)
        
        # 创建发送按钮
        send_button = Button(input_frame, text="发送", command=self.send_chat_message, bg="#4CAF50", fg="white")
        send_button.pack(side="right", padx=5)
        
        # 创建清除历史按钮
        clear_button = Button(input_frame, text="清除对话历史", command=self.clear_chat_history, bg="#f44336", fg="white")
        clear_button.pack(side="right", padx=5)
        
        # 保存当前上传的图片路径
        self.chat_image_path = None
        
        # 添加提示文本
        self.chat_status = Label(chat_frame, text="准备就绪，可以开始对话", fg="gray", bg="#f0f0f0", anchor="w")
        self.chat_status.pack(fill="x", padx=5, pady=2)
        
        return chat_frame

    def clear_chat_history(self):
        """清除对话历史"""
        if hasattr(self, 'chat_history'):
            # 保留系统消息
            system_message = self.chat_history[0] if self.chat_history else None
            self.chat_history = [system_message] if system_message else []
            
        # 清空响应文本框
        if hasattr(self, 'response_text'):
            self.response_text.delete("1.0", "end")
            
        # 更新状态
        self.chat_status.config(text="对话历史已清除", fg="green")
        self.log("对话历史已清除")


    def upload_image_for_chat(self):
        """上传图片用于多模态对话"""
        try:
            from tkinter import filedialog
            
            # 打开文件选择对话框
            file_path = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
            )
            
            if file_path:
                self.chat_image_path = file_path
                filename = os.path.basename(file_path)
                self.chat_status.config(text=f"已选择图片: {filename}", fg="green")
                self.log(f"已选择图片用于对话: {filename}")
        except Exception as e:
            self.log(f"上传图片失败: {e}")
            self.chat_status.config(text=f"上传图片失败: {e}", fg="red")





    def send_chat_message(self, event=None):
        """发送对话消息到LLM并显示回复"""
        user_message = self.chat_input.get().strip()
        if not user_message and not self.chat_image_path:
            return
            
        # 清空输入框
        self.chat_input.delete(0, tk.END)
        
        # 记录用户消息到日志
        self.log(f"用户: {user_message}")
        
        # 获取当前上下文信息
        context = self._get_chat_context()
        
        # 更新状态
        self.chat_status.config(text="正在处理请求...", fg="blue")
        
        # 启动线程处理LLM请求，避免UI卡顿
        threading.Thread(target=self._process_chat_message, 
                         args=(user_message, context, self.chat_image_path), 
                         daemon=True).start()
        
        # 清除已使用的图片路径
        image_path = self.chat_image_path
        self.chat_image_path = None
        
        # 重置状态提示
        if not image_path:
            self.chat_status.config(text="准备就绪，可以开始对话", fg="gray")
        else:
            self.chat_status.config(text="图片已发送，可继续对话", fg="gray")







    def _get_chat_context(self):
        """获取当前上下文信息用于对话"""
        context = []
        
        # 1. 添加GUI代码信息
        context.append("## GUI功能信息")
        context.append("这是一个PCB缺陷检测系统，具有以下主要功能:")
        context.append("- 图像检测：检测PCB图像中的缺陷")
        context.append("- 历史记录：查看和管理历史检测记录")
        context.append("- 数据分析：生成趋势分析报告")
        context.append("- 模型训练：训练和优化检测模型")
        
        # 2. 添加当前图像信息
        if hasattr(self, 'current_image') and self.current_image:
            context.append("## 当前图像信息")
            if isinstance(self.current_image, dict):
                context.append(f"图像路径: {self.current_image.get('path', '未知')}")
                context.append(f"缩放比例: {self.current_image.get('scale', 1.0)}")
            else:
                context.append("当前有图像显示，但详细信息不可用")
        
        # 3. 添加检测信息
        if hasattr(self, 'agent') and hasattr(self.agent, 'detections_history') and self.agent.detections_history:
            latest_detection = self.agent.detections_history[0]
            context.append("## 最近检测信息")
            context.append(f"图像名称: {latest_detection.get('image', '未知')}")
            context.append(f"检测时间: {latest_detection.get('timestamp', '未知')}")
            context.append(f"检测到的缺陷数: {latest_detection.get('detected_count', 0)}")
            
            # 添加缺陷详情
            defects = latest_detection.get('defects', [])
            if defects:
                context.append("缺陷详情:")
                for i, defect in enumerate(defects[:5]):  # 最多显示5个缺陷
                    context.append(f"- 缺陷{i+1}: 类型={defect.get('class', '未知')}, 置信度={defect.get('confidence', 0):.2f}")
                if len(defects) > 5:
                    context.append(f"... 还有{len(defects)-5}个缺陷")
        
        # 4. 添加日志信息
        log_content = self.get_log_content()
        if log_content:
            context.append("## 最近日志信息")
            # 只取最后10行日志
            log_lines = log_content.strip().split('\n')[-10:]
            context.extend(log_lines)
        
        return "\n".join(context)
    

    def _process_chat_message(self, user_message, context, image_path=None):
        """处理对话消息并获取LLM回复，支持多模态输入和多轮对话"""
        try:
            import os
            from openai import OpenAI
            import base64
            
            # 构建提示词
            system_prompt = f"""
你是PCB缺陷检测系统的AI助手。请根据以下上下文信息，回答用户的问题。
如果用户询问的内容与上下文无关，请基于你对PCB缺陷检测的专业知识回答。

上下文信息:
{context}
"""
            
            try:
                # 创建OpenAI客户端
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                
                # 初始化或获取对话历史
                if not hasattr(self, 'chat_history'):
                    self.chat_history = []
                    # 添加系统消息
                    if image_path:
                        # 多模态系统消息
                        self.chat_history.append({
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}]
                        })
                    else:
                        # 纯文本系统消息
                        self.chat_history.append({
                            "role": "system", 
                            "content": system_prompt
                        })
                
                # 准备用户消息
                if image_path:
                    # 如果有图片，使用多模态格式
                    # 读取图片并转换为base64
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # 构建多模态消息
                    user_content = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                    
                    # 添加文本内容
                    if user_message:
                        user_content.append({"type": "text", "text": user_message})
                    
                    # 添加到对话历史
                    self.chat_history.append({
                        "role": "user",
                        "content": user_content
                    })
                else:
                    # 纯文本消息
                    self.chat_history.append({
                        "role": "user", 
                        "content": user_message
                    })
                
                # 选择合适的模型
                model_name = "qwen-vl-max-latest" 
                
                # 更新状态
                self.master.after(0, lambda: self.chat_status.config(text="正在生成回复...", fg="blue"))
                
                # 使用流式输出
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=self.chat_history,
                    stream=True
                )
                
                # 初始化完整回复内容
                full_response = ""
                
                # 处理流式输出
                for chunk in completion:
                    if chunk.choices[0].delta.content is None:
                        continue
                    
                    # 获取当前块的内容
                    content_chunk = chunk.choices[0].delta.content
                    full_response += content_chunk
                    
                    # 在UI线程中实时更新回复
                    self.master.after(0, lambda c=full_response: self._update_streaming_response(c))
                
                # 将助手回复添加到对话历史
                self.chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # 在UI线程中更新日志和状态
                self.master.after(0, lambda: self.log(f"AI助手: {full_response}"))
                self.master.after(0, lambda: self.chat_status.config(text="回复已完成", fg="green"))
                
            except Exception as e:
                error_msg = f"调用AI助手失败: {str(e)}"
                self.master.after(0, lambda: self.log(error_msg))
                self.master.after(0, lambda: self.chat_status.config(text=f"请求失败: {str(e)}", fg="red"))
                
        except Exception as e:
            error_msg = f"处理对话消息失败: {str(e)}"
            self.master.after(0, lambda: self.log(error_msg))
            self.master.after(0, lambda: self.chat_status.config(text=f"处理失败: {str(e)}", fg="red"))

    def _update_streaming_response(self, current_response):
        """更新流式响应到UI"""
        # 更新响应文本框
        if hasattr(self, 'response_text'):
            self.response_text.delete("1.0", "end")
            self.response_text.insert("1.0", current_response)
            self.response_text.see("end")  # 自动滚动到底部
            
        # 更新状态栏
        self.chat_status.config(text="正在生成回复...", fg="blue")

    def get_log_content(self):
        """获取日志内容"""
        if hasattr(self, 'log_text') and self.log_text:
            # 从Listbox获取所有项目
            items = []
            for i in range(self.log_text.size()):
                items.append(self.log_text.get(i))
            return "\n".join(items)
        return ""


    def display_image(self, image_path):
        """在左侧画布显示图像，支持缩放和拖曳"""
        try:
            # 清除当前画布内容
            self.image_canvas.delete("all")
            
            # 加载图像
            img = Image.open(image_path)
            
            # 获取画布尺寸
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # 如果画布尚未渲染，使用默认尺寸
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 400
            
            # 计算缩放比例
            img_width, img_height = img.size
            scale = min(canvas_width/img_width, canvas_height/img_height)
            
            # 缩放图像
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为PhotoImage
            self.current_image = ImageTk.PhotoImage(img)
            
            # 保存原始图像和缩放信息用于后续缩放操作
            self.original_image = Image.open(image_path)
            self.current_scale = scale
            self.image_x = canvas_width//2
            self.image_y = canvas_height//2
            
            # 更新画布显示
            self.image_canvas.delete("all")
            self.image_id = self.image_canvas.create_image(
                self.image_x, self.image_y,
                image=self.current_image,
                anchor="center",
                tags="image"
            )
            
            # 绑定鼠标事件用于拖曳
            self.image_canvas.tag_bind("image", "<ButtonPress-1>", self.start_drag)
            self.image_canvas.tag_bind("image", "<B1-Motion>", self.drag_image)
            
            # 绑定鼠标滚轮事件用于缩放
            self.image_canvas.bind("<MouseWheel>", self.zoom_image)
            
        except Exception as e:
            self.logger.error(f"显示图像失败: {e}")
            self.image_canvas.delete("all")
            self.image_canvas.create_text(300, 300, text=f"图像加载失败: {str(e)}", fill="white")
    

    def start_drag(self, event):
        """开始拖曳图像"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    
    def drag_image(self, event):
        """拖曳图像"""
        # 确保drag_start_x和drag_start_y已定义
        if not hasattr(self, 'drag_start_x') or not hasattr(self, 'drag_start_y'):
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            return
            
        # 计算移动距离
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # 更新图像位置
        self.image_canvas.move("image", dx, dy)
        
        # 更新拖曳起点
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # 更新当前图像位置
        if hasattr(self, 'image_x') and hasattr(self, 'image_y'):
            self.image_x += dx
            self.image_y += dy


        """拖曳图像"""
        if not hasattr(self, 'drag_start_x'):
            return
            
        # 计算移动距离
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # 更新图像位置
        self.image_canvas.move("image", dx, dy)
        
        # 更新拖曳起点
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        # 更新当前图像位置
        self.image_x += dx
        self.image_y += dy
        
    def zoom_image(self, event):
        """缩放图像"""
        if not hasattr(self, 'original_image') or not self.original_image:
            return
            
        # 确定缩放方向和比例
        scale_factor = 1.1 if event.delta > 0 else 0.9
        
        # 获取当前图像信息
        if not hasattr(self, 'current_scale'):
            self.current_scale = self.current_image.get("scale", 1.0) if isinstance(self.current_image, dict) else 1.0
        
        old_scale = self.current_scale
        new_scale = old_scale * scale_factor
        
        # 限制缩放范围
        if new_scale < 0.1 or new_scale > 5.0:
            return
            
        # 计算新尺寸
        img_width, img_height = self.original_image.size
        new_width = int(img_width * new_scale)
        new_height = int(img_height * new_scale)
        
        # 缩放图像
        resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)  # 使用 tk_image 而不是 current_image
        
        # 获取当前图像中心位置
        if isinstance(self.current_image, dict):
            self.image_x = self.current_image.get("x", self.image_canvas.winfo_width()//2)
            self.image_y = self.current_image.get("y", self.image_canvas.winfo_height()//2)
        else:
            self.image_x = self.image_canvas.winfo_width()//2
            self.image_y = self.image_canvas.winfo_height()//2
        
        # 更新画布上的图像
        self.image_canvas.delete("image")
        self.image_id = self.image_canvas.create_image(
            self.image_x, self.image_y, 
            image=self.tk_image, 
            anchor="center",
            tags="image"
        )
        
        # 更新当前比例
        self.current_scale = new_scale
        
        # 更新当前图像信息（保持字典结构）
        if isinstance(self.current_image, dict):
            self.current_image.update({
                "scale": new_scale
            })
        else:
            self.current_image = {
                "path": getattr(self, 'current_image_path', ''),
                "original": self.original_image,
                "scale": new_scale,
                "x": self.image_x,
                "y": self.image_y
            }




    def toggle_periodic_detection(self):
        """切换定期检测状态"""
        if not hasattr(self, 'periodic_detection_active') or not self.periodic_detection_active:
            # 开始定期检测
            try:
                interval = float(self.interval_var.get())
                if interval < 0:
                    messagebox.showerror("错误", "检测间隔必须大于等于0")
                    return
                    
                self.agent.detection_interval = interval
                self.periodic_detection_active = True
                
                # 使用auto_detect_var的值决定是否自动检测
                self.agent.auto_detection = bool(self.auto_detect_var.get())
                
                # 更新按钮文本
                self.detect_button.config(text="停止检测")
                
                # 启动检测线程
                self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
                self.detection_thread.start()
                
                # 启动UI更新线程
                self.start_ui_update_thread()
                
                self.log("已启动定期检测")
            except ValueError:
                messagebox.showerror("错误", "请输入有效的检测间隔")
        else:
            # 停止定期检测
            self.periodic_detection_active = False
            self.detect_button.config(text="开始检测")
            self.log("已停止定期检测")


    # 添加detection_loop方法
    def detection_loop(self):
        """检测线程的主循环"""
        try:
            while self.periodic_detection_active:
                # 执行一次检测
                self._perform_detection()
                
                # 如果不是自动检测模式，只检测一次
                if not self.agent.auto_detection:
                    break
                    
                # 等待指定的间隔时间
                time.sleep(self.agent.detection_interval)
                
        except Exception as e:
            self.log(f"检测线程出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
        finally:
            # 确保线程结束时更新UI
            self.master.after(0, lambda: self.detect_button.config(text="开始检测"))
            self.master.after(0, lambda: setattr(self, 'periodic_detection_active', False))
            self.log("检测线程已结束")



    def _start_detection_timer(self):
        """启动定期检测定时器"""
        if not self.periodic_detection_active or not self.agent.auto_detection:
            return
            
        interval_ms = int(self.agent.detection_interval * 1000)
        if interval_ms > 0:
            self.detection_timer = self.master.after(interval_ms, self._timed_detection)
    
    def _timed_detection(self):
        """定时执行的检测"""
        if self.periodic_detection_active and self.agent.auto_detection:
            self._perform_detection()
            self._start_detection_timer()


    def _perform_detection(self):
        """执行一次检测并更新UI"""
        try:
            # 更新状态栏
            self.status_var.set(f"正在检测新图像...")
            
            # 执行检测
            detected = self.agent.detect_new_data(single_image_mode=True)
            
            if detected:
                # 获取最新的检测记录
                latest_record = self.agent.detections_history[0]
                image_name = latest_record.get("image", "未知图像")
                
                # 更新状态栏
                self.status_var.set(f"当前图像: {image_name}")
                
                # 确保current_images包含当前记录 - 添加这段代码
                if not hasattr(self, 'current_images'):
                    self.current_images = []
                
                # 添加到current_images列表
                self.current_images.append(latest_record)
                self.current_image_index = len(self.current_images) - 1
                
                self.log(f"已添加图像到列表，当前索引: {self.current_image_index}, 总图像数: {len(self.current_images)}")
                


                # 更新图像列表
                self.update_image_list()
                
                # 自动选择最新的检测结果
                self.image_listbox.select_set(0)
                self.on_image_select(None)
                
                # 启用结果按钮
                self._enable_result_buttons()
                
                # 更新统计图表
                self.update_stats_plot()
                self.update_trend_plot()
            else:
                self.status_var.set("没有新图像需要检测")
                
        except Exception as e:
            self.log(f"检测过程出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self.status_var.set("检测失败")



    def _show_next_image(self):
        """显示下一张图像或检测新图像"""
        if self.current_image_index < len(self.current_images) - 1:
            # 如果当前不是最后一张，显示下一张已检测的图像
            self.current_image_index += 1
            self._show_current_image()
            self._enable_navigation_buttons()
            self._enable_result_buttons()
        else:
            # 如果是最后一张，检测新图像
            self._perform_detection()

    def _show_current_image(self):
        """显示当前索引的图像"""
        if not self.current_images or self.current_image_index >= len(self.current_images):
            return
            
        current = self.current_images[self.current_image_index]
        
        # 显示标注后的图像
        annotated_path = current.get("annotated_image", "")
        if annotated_path and os.path.exists(annotated_path):
            self.display_image(annotated_path)
            
            # 更新状态信息
            defect_count = current.get("detected_count", 0)
            status_text = f"图像 {self.current_image_index + 1}/{len(self.current_images)}: "
            status_text += f"检测到 {defect_count} 处缺陷"
            
            if current.get("reviewed", False):
                status_text += " (已复检)"
                
            self.status_var.set(status_text)
        else:
            self.log(f"无法显示图像: {annotated_path}")
    
    def _show_prev_image(self):
        """显示上一张图像"""
        self.log("调用_show_prev_image方法")
        
        # 检查当前图像状态并输出详细信息
        if not hasattr(self, 'current_images'):
            self.log("错误: current_images属性不存在")
            self.current_images = []
        
        if not self.current_images:
            self.log("错误: 当前图像列表为空")
            messagebox.showinfo("提示", "没有图像可供浏览")
            return
            
        if not hasattr(self, 'current_image_index'):
            self.log("错误: current_image_index属性不存在")
            self.current_image_index = 0
        
        # 确保索引在有效范围内
        if self.current_image_index <= 0:
            self.log("已经是第一张图像")
            messagebox.showinfo("提示", "已经是第一张图像")
            return
            
        # 更新索引并显示图像
        self.current_image_index -= 1
        self.log(f"切换到上一张图像，新索引: {self.current_image_index}")
        
        # 显示当前图像
        self._display_current_image()
        
    def _show_next_image(self):
        """显示下一张图像"""
        self.log("调用_show_next_image方法")
        
        # 检查当前图像状态并输出详细信息
        if not hasattr(self, 'current_images'):
            self.log("错误: current_images属性不存在")
            self.current_images = []
        
        if not self.current_images:
            self.log("错误: 当前图像列表为空")
            messagebox.showinfo("提示", "没有图像可供浏览")
            return
            
        if not hasattr(self, 'current_image_index'):
            self.log("错误: current_image_index属性不存在")
            self.current_image_index = 0
        
        # 确保索引在有效范围内
        if self.current_image_index >= len(self.current_images) - 1:
            self.log("已经是最后一张图像")
            messagebox.showinfo("提示", "已经是最后一张图像")
            return
            
        # 更新索引并显示图像
        self.current_image_index += 1
        self.log(f"切换到下一张图像，新索引: {self.current_image_index}")
        
        # 显示当前图像
        self._display_current_image()
    def display_detection_result(self, record):
        """显示检测结果"""
        try:
            # 显示标注后的图像
            annotated_path = record.get("annotated_image", "")
            if annotated_path and os.path.exists(annotated_path):
                self.display_image(annotated_path)
                
                # 更新检测信息显示
                self.update_detection_info(record)
                
                # 更新状态栏
                defect_count = record.get("detected_count", 0)
                self.status_var.set(f"当前图像: {record.get('image', '')} - 检测到 {defect_count} 处缺陷")
                
                # 启用导航按钮
                self._enable_navigation_buttons()
                
                # 启用结果按钮
                self._enable_result_buttons()
            else:
                self.log(f"无法显示检测结果: {annotated_path}")
                
        except Exception as e:
            self.log(f"显示检测结果失败: {e}")
            import traceback
            self.log(traceback.format_exc())
    
    # 修复_enable_navigation_buttons方法
    def _enable_navigation_buttons(self):
        """启用或禁用导航按钮"""
        if not hasattr(self, 'current_images') or not self.current_images:
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            return
        
        # 使用current_image_index而不是查找current_image_path
        # 启用/禁用上一张按钮
        if self.current_image_index > 0:
            self.prev_button.config(state="normal")
        else:
            self.prev_button.config(state="disabled")
            
        # 启用/禁用下一张按钮
        if self.current_image_index < len(self.current_images) - 1:
            self.next_button.config(state="normal")
        else:
            self.next_button.config(state="disabled")



    def _disable_navigation_buttons(self):
        """禁用导航按钮"""
        self.prev_button.config(state="disabled")
        self.next_button.config(state="disabled")
    
    def _enable_result_buttons(self):
        """启用结果按钮"""
        # 始终启用按钮，不再根据条件禁用
        self.pass_button.config(state="normal")
        self.fail_button.config(state="normal")
        
        # 添加调试日志
        self.log("已启用通过/不通过按钮")

    

    
    def update_confidence(self, value):
        """更新检测置信度阈值"""
        self.agent.confidence_threshold = float(value)
    
    def toggle_auto_detection(self):
        """切换自动/手动检测模式"""
        self.agent.auto_detection = bool(self.auto_detect_var.get())
        
        if self.periodic_detection_active:
            if self.agent.auto_detection:
                self._start_detection_timer()
            elif self.detection_timer:
                self.master.after_cancel(self.detection_timer)
                self.detection_timer = None



    def check_for_manual_review(self):
        """检查是否需要人工复检"""
        if not self.current_images:
            return
            
        current = self.current_images[self.current_image_index]
        
        # 检查是否有低于阈值的检测结果
        low_conf_defects = []
        for defect in current.get("defects", []):
            if defect.get("confidence", 1.0) < self.agent.confidence_threshold:
                low_conf_defects.append(defect)
        
        if low_conf_defects:
            # 询问是否进行人工复检
            defect_count = len(low_conf_defects)
            if messagebox.askyesno("人工复检", 
                                 f"检测到 {defect_count} 处置信度低于阈值的缺陷，是否进行人工复检？"):
                # 修改这里：传递图像名称而不是整个记录
                self.open_review_window(target_image=current.get("image", ""))

    def _mark_as_pass(self):
        """标记当前图像为检测合格"""
        if not self.current_images or self.current_image_index >= len(self.current_images):
            return
            
        current = self.current_images[self.current_image_index]
        
        # 更新检测记录
        current["reviewed"] = True
        current["correct"] = True
        current["false_positive"] = False
        current["missed_defect"] = False
        
        # 保存图像到训练目录
        image_path = self._save_image_for_labelme(current)
        
        # 如果成功保存图像，则转换为labelme格式
        if image_path:
            self._convert_to_labelme_json(current, image_path)
        
        # 更新数据库
        self._update_detection_record(current)
        
        # 更新UI
        self.log(f"已标记为检测合格: {current.get('image', '')}")
        self._enable_result_buttons()  # 更新按钮状态
        
        # 如果是自动模式，显示下一张图像
        if self.agent.auto_detection and self.current_image_index < len(self.current_images) - 1:
            self._show_next_image()
    
    def _mark_as_fail(self):
        """标记当前图像为检测不合格，启动人工标注"""
        if not self.current_images or self.current_image_index >= len(self.current_images):
            return
            
        current = self.current_images[self.current_image_index]
        
        # 更新检测记录
        current["reviewed"] = True
        current["correct"] = False
        
        # 询问具体问题
        result = messagebox.askyesnocancel("缺陷问题初筛", 
                                         "请指出检测问题类型:\n 是 - 存在误检\n 否 - 存在漏检\n 取消 - 两者都有")
        
        if result is None:  # 取消 - 两者都有
            current["false_positive"] = True
            current["missed_defect"] = True
            problem_type = "误检和漏检"
        elif result:  # 是 - 存在误检
            current["false_positive"] = True
            current["missed_defect"] = False
            problem_type = "误检"
        else:  # 否 - 存在漏检
            current["false_positive"] = False
            current["missed_defect"] = True
            problem_type = "漏检"
        
        # 保存图像并启动labelme
        image_path = self._save_image_for_labelme(current, auto_start_labelme=True)
        
        # 如果成功保存图像，则转换为labelme格式
        if image_path:
            self._convert_to_labelme_json(current, image_path)
            self.log(f"已保存图像到训练集: {image_path}")
        else:
            self.log("警告: 图像保存失败")
        
        # 更新数据库
        self._update_detection_record(current)
        
        # 更新UI
        self.log(f"已标记为检测不合格({problem_type}): {current.get('image', '')}")
        self._enable_result_buttons()  # 更新按钮状态



    def mark_as_pass(self):
        """通过按钮的回调函数"""
        self.log("调用mark_as_pass方法")
        
        # 检查当前图像状态并输出详细信息
        if not hasattr(self, 'current_images'):
            self.log("错误: current_images属性不存在")
            self.current_images = []
        
        if not self.current_images:
            self.log("错误: 当前图像列表为空")
            messagebox.showinfo("提示", "没有当前图像可用")
            return
            
        if not hasattr(self, 'current_image_index'):
            self.log("错误: current_image_index属性不存在")
            self.current_image_index = 0
        
        if self.current_image_index >= len(self.current_images):
            self.log(f"错误: 当前索引{self.current_image_index}超出图像列表范围{len(self.current_images)}")
            self.current_image_index = 0
            
        self.log(f"当前图像索引: {self.current_image_index}, 图像列表长度: {len(self.current_images)}")
        
        # 执行标记操作
        self._mark_as_pass()
    
    def mark_as_fail(self):
        """不通过按钮的回调函数"""
        self.log("调用mark_as_fail方法")
        
        # 检查当前图像状态并输出详细信息
        if not hasattr(self, 'current_images'):
            self.log("错误: current_images属性不存在")
            self.current_images = []
        
        if not self.current_images:
            self.log("错误: 当前图像列表为空")
            messagebox.showinfo("提示", "没有当前图像可用")
            return
            
        if not hasattr(self, 'current_image_index'):
            self.log("错误: current_image_index属性不存在")
            self.current_image_index = 0
        
        if self.current_image_index >= len(self.current_images):
            self.log(f"错误: 当前索引{self.current_image_index}超出图像列表范围{len(self.current_images)}")
            self.current_image_index = 0
            
        self.log(f"当前图像索引: {self.current_image_index}, 图像列表长度: {len(self.current_images)}")
        
        # 执行标记操作
        self._mark_as_fail()




    def open_review_window(self, target_image=None):
        """打开人工复检窗口，可以指定目标图像"""
        if not self.agent.detections_history:
            messagebox.showinfo("提示", "没有检测记录可供复检。")
            return
            
        review_win = Toplevel(self.master)
        review_win.title("人工复检")
        review_win.geometry("800x600")
        
        # 创建检测记录列表
        list_frame = Frame(review_win)
        list_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        Label(list_frame, text="检测记录:").pack(anchor="w")
        records_list = Listbox(list_frame, width=40)
        records_list.pack(fill="both", expand=True)
        
        # 填充检测记录
        for i, record in enumerate(self.agent.detections_history):
            status = "已复检" if record.get("reviewed", False) else "未复检"
            records_list.insert(END, f"{record['image']} - {record['detected_count']}个缺陷 - {status}")
        # 如果指定了目标图像，选中对应的记录
        if target_image:
            for i in range(records_list.size()):
                if target_image in records_list.get(i):
                    records_list.selection_clear(0, END)
                    records_list.selection_set(i)
                    records_list.see(i)  # 确保目标项可见
                    break
        
        # 创建图像预览和复检控制区域
        preview_frame = Frame(review_win)
        preview_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # 图像预览标签
        preview_label = Label(preview_frame, text="图像预览:")
        preview_label.pack(anchor="w")
        
        # 使用Canvas替代Label以支持缩放和拖曳
        canvas = tk.Canvas(preview_frame, bg="black")
        canvas.pack(fill="both", expand=True)
        
        # 复检控制按钮
        control_frame = Frame(preview_frame)
        control_frame.pack(side="bottom", fill="x", pady=10)
        
        # 复检结果变量
        false_positive_var = IntVar(value=0)
        missed_defect_var = IntVar(value=0)
        
        # 复检选项
        Checkbutton(control_frame, text="存在误检", variable=false_positive_var).pack(anchor="w")
        Checkbutton(control_frame, text="存在漏检", variable=missed_defect_var).pack(anchor="w")
        
        # 当前选中的记录索引
        current_index = [-1]  # 使用列表以便在内部函数中修改
        current_record = [None]  # 保存当前选中的记录
        
        # 添加人工标注按钮
        def start_manual_annotation():
            try:
                if current_record[0] is None:
                    messagebox.showwarning("提示", "请先选择一个图像")
                    return
                    
                # 保存原始图像到临时标注目录
                annotation_dir = os.path.join(os.getcwd(), "temp_annotation")
                os.makedirs(annotation_dir, exist_ok=True)
                
                # 获取原始图像路径
                original_image = current_record[0].get("processed_path")
                if not original_image or not os.path.exists(original_image):
                    messagebox.showerror("错误", "找不到原始图像")
                    return
                    
                # 复制原始图像到标注目录
                annotation_image = os.path.join(annotation_dir, os.path.basename(original_image))
                import shutil
                shutil.copy2(original_image, annotation_image)
                
                # 启动labelme
                import subprocess
                subprocess.Popen(f"labelme \"{annotation_image}\"", shell=True)
                
            except Exception as e:
                self.logger.error(f"启动标注失败: {e}")
                messagebox.showerror("错误", f"启动标注失败: {e}")
        
        # 添加人工标注按钮
        annotation_btn = Button(control_frame, text="人工标注", command=start_manual_annotation)
        annotation_btn.pack(side="left", padx=5, pady=5)
        
        # 选择记录时的回调函数
        def on_record_select(event):
            selection = records_list.curselection()
            if not selection:
                return
                
            index = selection[0]
            current_index[0] = index
            current_record[0] = self.agent.detections_history[index]
            record = current_record[0]
            
            # 更新复选框状态
            false_positive_var.set(1 if record.get("false_positive", False) else 0)
            missed_defect_var.set(1 if record.get("missed_defect", False) else 0)
            
            # 加载并显示图像
            annotated_path = record.get("annotated_image")
            if annotated_path and os.path.exists(annotated_path):
                try:
                    # 清除画布
                    canvas.delete("all")
                    
                    # 加载图像
                    img = Image.open(annotated_path)
                    canvas.original_image = img  # 保存原始图像
                    
                    # 调整图像大小以适应显示区域
                    canvas_width = canvas.winfo_width() or 400
                    canvas_height = canvas.winfo_height() or 400
                    
                    # 计算缩放比例
                    img_width, img_height = img.size
                    scale = min(canvas_width/img_width, canvas_height/img_height)
                    
                    # 缩放图像
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    resized_image = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 转换为PhotoImage
                    photo = ImageTk.PhotoImage(resized_image)
                    canvas.photo = photo  # 保持引用以防止垃圾回收
                    
                    # 在画布中央显示图像
                    canvas.image_id = canvas.create_image(
                        canvas_width//2, canvas_height//2, 
                        image=photo, anchor="center",
                        tags="image"
                    )
                    
                    # 保存图像信息用于缩放和拖曳
                    canvas.current_scale = scale
                    canvas.image_x = canvas_width//2
                    canvas.image_y = canvas_height//2
                    
                except Exception as e:
                    self.logger.error(f"无法加载图像: {e}")
                    canvas.create_text(canvas_width//2, canvas_height//2, 
                                      text=f"无法加载图像: {e}", fill="white")
            else:
                canvas.delete("all")
                canvas_width = canvas.winfo_width() or 400
                canvas_height = canvas.winfo_height() or 400
                canvas.create_text(canvas_width//2, canvas_height//2, 
                                  text="[图像不可用]", fill="white")
        
        # 图像拖曳功能
        def start_drag(event):
            canvas.drag_start_x = event.x
            canvas.drag_start_y = event.y
        
        def drag_image(event):
            if not hasattr(canvas, 'drag_start_x'):
                return
                
            # 计算移动距离
            dx = event.x - canvas.drag_start_x
            dy = event.y - canvas.drag_start_y
            
            # 更新图像位置
            canvas.move("image", dx, dy)
            
            # 更新拖曳起点
            canvas.drag_start_x = event.x
            canvas.drag_start_y = event.y
            
            # 更新图像位置信息
            if hasattr(canvas, 'image_x') and hasattr(canvas, 'image_y'):
                canvas.image_x += dx
                canvas.image_y += dy
        
        # 图像缩放功能
        def zoom_image(event):
            if not hasattr(canvas, 'original_image') or not canvas.original_image:
                return
                
            # 确定缩放方向和比例
            scale_factor = 1.1 if event.delta > 0 else 0.9
            
            # 获取当前图像信息
            img = canvas.original_image
            old_scale = canvas.current_scale
            new_scale = old_scale * scale_factor
            
            # 限制缩放范围
            if new_scale < 0.1 or new_scale > 5.0:
                return
                
            # 计算新尺寸
            img_width, img_height = img.size
            new_width = int(img_width * new_scale)
            new_height = int(img_height * new_scale)
            
            # 缩放图像
            resized_image = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            
            # 更新画布上的图像
            canvas.delete("image")
            canvas.photo = photo  # 更新引用
            canvas.image_id = canvas.create_image(
                canvas.image_x, canvas.image_y, 
                image=photo, anchor="center",
                tags="image"
            )
            
            # 更新当前比例
            canvas.current_scale = new_scale
        
        # 绑定鼠标事件
        canvas.tag_bind("image", "<ButtonPress-1>", start_drag)
        canvas.tag_bind("image", "<B1-Motion>", drag_image)
        canvas.bind("<MouseWheel>", zoom_image)
        
        # 提交复检结果
        def submit_review():
            selected = records_list.curselection()
            if not selected:
                messagebox.showinfo("提示", "请先选择一条检测记录。")
                return
                
            index = selected[0]
            false_positive = fp_var.get()
            missed_defect = missed_var.get()
            
            # 调用agent的manual_review方法
            if self.agent.manual_review(index, false_positive, missed_defect):
                messagebox.showinfo("成功", "复检结果已保存。")
                # 更新列表显示
                status = "已复检"
                record = self.agent.detections_history[index]
                records_list.delete(index)
                records_list.insert(index, f"{record['image']} - {record['detected_count']}个缺陷 - {status}")
                # 禁用复检按钮
                submit_btn.config(state="disabled")
                # 更新主界面
                self.update_image_list()
                self.update_stats_plot()
                self.update_trend_plot()
            else:
                messagebox.showerror("错误", "保存复检结果失败。")
        
        # 绑定列表选择事件
        records_list.bind('<<ListboxSelect>>', on_record_select)
        
        # 提交按钮
        submit_button = Button(control_frame, text="提交复检结果", command=submit_review)
        submit_button.pack(side="bottom", pady=10)
        
        # 如果有指定的目标图像，触发选择事件
        if target_image:
            records_list.event_generate("<<ListboxSelect>>")



    def update_detection_info(self, record):
        """更新检测信息显示"""
        try:
            # 清除之前的信息
            if hasattr(self, 'info_frame') and self.info_frame:
                self.info_frame.destroy()
                
            # 创建信息框架
            self.info_frame = Frame(self.left_frame, bg="white", bd=1, relief="solid")
            self.info_frame.place(relx=0.01, rely=0.01, relwidth=0.3, relheight=0.2)
            
            # 添加检测信息
            image_name = record.get("image", "未知")
            detected_count = record.get("detected_count", 0)
            timestamp = record.get("timestamp", "未知")
            
            # 格式化时间戳
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            # 添加标签显示信息
            Label(self.info_frame, text=f"图像: {image_name}", bg="white", anchor="w").pack(fill="x", padx=5, pady=2)
            Label(self.info_frame, text=f"检测时间: {timestamp}", bg="white", anchor="w").pack(fill="x", padx=5, pady=2)
            Label(self.info_frame, text=f"缺陷数量: {detected_count}", bg="white", anchor="w").pack(fill="x", padx=5, pady=2)
            
            # 添加缺陷类型统计
            defect_types = {}
            for defect in record.get("defects", []):
                defect_class = defect.get("class", "未知")
                defect_types[defect_class] = defect_types.get(defect_class, 0) + 1
            
            if defect_types:
                types_text = "缺陷类型: "
                for defect_class, count in defect_types.items():
                    types_text += f"{defect_class}({count}) "
                Label(self.info_frame, text=types_text, bg="white", anchor="w", wraplength=200).pack(fill="x", padx=5, pady=2)
            
            # 添加复检状态
            if record.get("reviewed", False):
                review_status = "已复检"
                if record.get("correct", True):
                    review_status += " - 检测正确"
                else:
                    if record.get("false_positive", False):
                        review_status += " - 存在误检"
                    if record.get("missed_defect", False):
                        review_status += " - 存在漏检"
                Label(self.info_frame, text=review_status, bg="white", fg="blue", anchor="w").pack(fill="x", padx=5, pady=2)
            
        except Exception as e:
            self.logger.error(f"更新检测信息失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())











    def _save_image_and_labels(self, detection_record, is_correct=True):
        """保存图像和YOLO格式标注，并进行数据增强"""
        try:
            # 获取原始图像路径
            image_path = detection_record.get("processed_path")
            if not image_path or not os.path.exists(image_path):
                self.log(f"找不到原始图像: {image_path}")
                return
            
            # 获取图像文件名（不含扩展名）
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 创建目标路径
            train_image_path = os.path.join(self.agent.train_images_dir, f"{image_name}.jpg")
            train_label_path = os.path.join(self.agent.train_labels_dir, f"{image_name}.txt")
            
            # 复制图像到训练目录
            shutil.copy2(image_path, train_image_path)
            
            # 创建YOLO格式标注文件
            with open(train_label_path, 'w') as f:
                for defect in detection_record.get("defects", []):
                    class_id = self.agent.class_names.index(defect["class"])
                    bbox = defect["bbox"]
                    
                    # 转换为YOLO格式（归一化坐标）
                    img = cv2.imread(image_path)
                    height, width = img.shape[:2]
                    
                    # 计算中心点和宽高
                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # 写入标注
                    f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")
            
            self.log(f"已保存图像和标注到训练集: {train_image_path}")
            
            # 如果检测正确且启用了数据增强，进行图像增强
            if is_correct and self.agent.augmentation_enabled:
                self._perform_data_augmentation(train_image_path, train_label_path)
        
        except Exception as e:
            self.log(f"保存图像和标注失败: {e}")
    
    def _perform_data_augmentation(self, image_path, label_path):
        """对图像进行数据增强并保存增强后的图像和标注"""
        try:
            # 读取原始图像和标注
            img = cv2.imread(image_path)
            if img is None:
                self.log(f"无法读取图像进行增强: {image_path}")
                return
                
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # 获取图像文件名（不含扩展名）
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 进行多种数据增强
            for i in range(self.agent.augmentation_count):
                # 创建增强后的文件名
                aug_image_path = os.path.join(self.agent.train_images_dir, f"{image_name}_aug{i+1}.jpg")
                aug_label_path = os.path.join(self.agent.train_labels_dir, f"{image_name}_aug{i+1}.txt")
                
                # 随机选择增强方法
                aug_type = np.random.choice(['flip', 'rotate', 'brightness', 'contrast'])
                aug_img = img.copy()
                aug_labels = labels.copy()
                
                height, width = img.shape[:2]
                
                if aug_type == 'flip':
                    # 水平翻转
                    aug_img = cv2.flip(aug_img, 1)
                    
                    # 更新标注
                    new_labels = []
                    for label in aug_labels:
                        parts = label.strip().split()
                        if len(parts) == 5:
                            class_id = parts[0]
                            x_center = 1.0 - float(parts[1])  # 水平翻转中心点
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            new_labels.append(f"{class_id} {x_center} {y_center} {w} {h}\n")
                    aug_labels = new_labels
                
                elif aug_type == 'rotate':
                    # 小角度旋转 (±15度)
                    angle = np.random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                    aug_img = cv2.warpAffine(aug_img, M, (width, height))
                    
                    # 旋转标注（简化处理，仅适用于小角度旋转）
                    # 对于大角度旋转，需要更复杂的坐标变换
                    # 这里简单保持标注不变，因为小角度旋转对目标位置影响较小
                
                elif aug_type == 'brightness':
                    # 亮度调整
                    factor = np.random.uniform(0.7, 1.3)
                    aug_img = cv2.convertScaleAbs(aug_img, alpha=factor, beta=0)
                    # 标注不变
                
                elif aug_type == 'contrast':
                    # 对比度调整
                    factor = np.random.uniform(0.7, 1.3)
                    mean = np.mean(aug_img)
                    aug_img = cv2.convertScaleAbs(aug_img, alpha=factor, beta=(1-factor)*mean)
                    # 标注不变
                
                # 保存增强后的图像和标注
                cv2.imwrite(aug_image_path, aug_img)
                with open(aug_label_path, 'w') as f:
                    f.writelines(aug_labels)
                
                self.log(f"已创建增强图像 ({aug_type}): {aug_image_path}")
            
        except Exception as e:
            self.log(f"数据增强失败: {e}")
    
    def _save_image_for_annotation(self, detection_record):
        """保存图像用于人工标注"""
        try:
            # 导入shutil模块
            import shutil
            
            # 获取原始图像路径
            image_path = detection_record.get("processed_path")
            if not image_path or not os.path.exists(image_path):
                self.log(f"找不到原始图像: {image_path}")
                return None
            
            # 创建临时标注目录
            annotation_dir = os.path.join(os.getcwd(), "temp_annotation")
            os.makedirs(annotation_dir, exist_ok=True)
            
            # 复制图像到标注目录
            annotation_image = os.path.join(annotation_dir, os.path.basename(image_path))
            shutil.copy2(image_path, annotation_image)
            
            # 保存路径到记录中
            detection_record["annotation_image"] = annotation_image
            
            return annotation_image
            
        except Exception as e:
            self.log(f"准备标注图像失败: {e}")
            import traceback
            self.log(traceback.format_exc())  # 添加详细错误信息
            return None

    def _save_image_for_labelme(self, detection_record, auto_start_labelme=False):
        """保存图像到训练集目录，可选择自动启动labelme进行标注"""
        try:
            # 获取原始图像路径
            image_path = detection_record.get("processed_path")
            self.log(f"原始图像路径: {image_path}")
            
            if not image_path:
                self.log("错误: 检测记录中没有processed_path")
                # 尝试使用其他可能的路径
                image_path = detection_record.get("annotated_image")
                self.log(f"尝试使用annotated_image路径: {image_path}")
                
                if not image_path:
                    self.log("错误: 检测记录中没有可用的图像路径")
                    return None
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.log(f"错误: 图像文件不存在: {image_path}")
                return None
                
            # 获取图像文件名（不含扩展名）
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.log(f"图像文件名: {image_name}")
            
            # 使用新变量名保存硬编码路径
            #labelme_images_dir = "D:/030923/data/train/images_0"
            labelme_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "train", "images_0")
            
            # 检查目录是否存在，如果不存在则创建
            if not os.path.exists(labelme_images_dir):
                self.log(f"创建标注图像目录: {labelme_images_dir}")
                os.makedirs(labelme_images_dir, exist_ok=True)
            
            # 创建目标路径
            train_image_path = os.path.join(labelme_images_dir, f"{image_name}.jpg")
            self.log(f"目标图像路径: {train_image_path}")
            
            # 复制图像到训练目录
            import shutil
            try:
                shutil.copy2(image_path, train_image_path)
                self.log(f"已复制图像: {image_path} -> {train_image_path}")
                
                # 验证文件是否成功复制
                if os.path.exists(train_image_path):
                    self.log(f"验证成功: 文件已存在于目标位置")
                    file_size = os.path.getsize(train_image_path)
                    self.log(f"文件大小: {file_size} 字节")
                else:
                    self.log(f"验证失败: 文件不存在于目标位置")
            except Exception as copy_error:
                self.log(f"复制文件时出错: {copy_error}")
                import traceback
                self.log(traceback.format_exc())
                return None
            
            # 如果需要自动启动labelme
            if auto_start_labelme:
                try:
                    # 检查labelme是否可用
                    import subprocess
                    self.log(f"尝试启动labelme进行标注: {train_image_path}")
                    
                    # 使用绝对路径启动labelme
                    labelme_cmd = self.agent.labelme_path if hasattr(self.agent, 'labelme_path') else "labelme"
                    self.log(f"使用labelme命令: {labelme_cmd}")
                    
                    # 启动labelme进程
                    subprocess.Popen(f"{labelme_cmd} \"{train_image_path}\"", shell=True)
                    self.log(f"已启动labelme进程")
                except Exception as e:
                    self.log(f"启动labelme失败: {e}")
                    import traceback
                    self.log(traceback.format_exc())
            
            return train_image_path
            
        except Exception as e:
            self.log(f"保存图像到训练集失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None

    def _convert_to_labelme_json(self, record, image_path):
        """将YOLO格式的检测结果转换为labelme的JSON格式"""
        try:
            if not image_path or not os.path.exists(image_path):
                self.log(f"找不到图像: {image_path}")
                return False
                
            # 读取图像获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                self.log(f"无法读取图像: {image_path}")
                return False
                
            height, width = img.shape[:2]
            
            # 创建labelme格式的JSON
            json_data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(image_path),
                "imageData": None,
                "imageHeight": height,
                "imageWidth": width
            }
            
            # 添加检测到的缺陷
            for defect in record.get("defects", []):
                class_name = defect.get("class", "unknown")
                bbox = defect.get("bbox", [0, 0, 0, 0])
                
                # 创建矩形标注
                shape = {
                    "label": class_name,
                    "points": [
                        [bbox[0], bbox[1]],  # 左上角
                        [bbox[2], bbox[3]]   # 右下角
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                
                json_data["shapes"].append(shape)
            
            # 修改这里：使用os.path.join正确构建路径
            json_dir = os.path.join("D:", os.sep, "030923", "data", "train", "json")
            
            # 确保目录存在
            os.makedirs(json_dir, exist_ok=True)
            
            # 构建完整的JSON文件路径
            json_path = os.path.join(json_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
            
            self.log(f"准备保存JSON文件到: {json_path}")
            
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
                
            # 验证文件是否成功保存
            if os.path.exists(json_path):
                file_size = os.path.getsize(json_path)
                self.log(f"已保存labelme格式标注: {json_path} (大小: {file_size} 字节)")
                return True
            else:
                self.log(f"警告: JSON文件似乎未成功保存")
                return False
            
        except Exception as e:
            self.log(f"转换为labelme格式失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
 
    
    def _start_labelme(self, image_path):
        """启动labelme进行标注"""
        try:
            import subprocess
            self.log(f"启动labelme标注图像: {image_path}")
            
            # 使用subprocess启动labelme
            subprocess.Popen(f"labelme \"{image_path}\"", shell=True)
            
        except Exception as e:
            self.log(f"启动labelme失败: {e}")
            import traceback
            self.log(traceback.format_exc())










    def _start_labelme_annotation(self, detection_record):
        """启动labelme进行标注"""
        try:
            # 获取图像路径
            image_path = detection_record.get("annotation_image")
            if not image_path or not os.path.exists(image_path):
                self.log("找不到标注图像路径")
                return
                
            # 确保标注目录存在
            annotation_dir = os.path.dirname(image_path)
            os.makedirs(annotation_dir, exist_ok=True)
            
            # 构建labelme命令
            labelme_cmd = f"labelme \"{image_path}\""
            
            # 在新进程中启动labelme
            import subprocess
            self.log(f"启动labelme标注工具: {labelme_cmd}")
            
            # 使用Popen而不是run，这样不会阻塞主程序
            subprocess.Popen(labelme_cmd, shell=True)
            
            # 更新检测记录
            detection_record["annotation_started"] = True
            self._update_detection_record(detection_record)
            
        except Exception as e:
            self.logger.error(f"加载历史记录失败: {e}")
            # 重置统计计数
            self.total_images = 0
            self.reviewed_images = 0
            self.correct_detections = 0
            self.false_positive_images = 0
            self.missed_defect_images = 0
            # 清空历史记录
            self.detections_history = []




    def _import_annotation(self, detection_record):
        """导入labelme标注并转换为YOLO格式"""
        try:
            # 获取标注图像路径
            annotation_image = detection_record.get("annotation_image")
            if not annotation_image:
                self.log("找不到标注图像路径")
                return
            
            # 查找JSON标注文件
            json_file = os.path.splitext(annotation_image)[0] + ".json"
            if not os.path.exists(json_file):
                messagebox.showwarning("导入失败", "找不到标注文件，请确保已完成标注并保存。")
                return
            
            # 读取JSON标注
            import json
            with open(json_file, 'r') as f:
                annotation_data = json.load(f)
            
            # 获取图像尺寸
            img_width = annotation_data.get("imageWidth", 0)
            img_height = annotation_data.get("imageHeight", 0)
            
            if img_width == 0 or img_height == 0:
                # 如果JSON中没有尺寸信息，从图像中读取
                img = cv2.imread(annotation_image)
                img_height, img_width = img.shape[:2]
            
            # 创建YOLO格式标注文件
            image_name = os.path.splitext(os.path.basename(annotation_image))[0]
            train_image_path = os.path.join(self.agent.train_images_dir, f"{image_name}.jpg")
            train_label_path = os.path.join(self.agent.train_labels_dir, f"{image_name}.txt")
            
            # 复制图像到训练目录
            shutil.copy2(annotation_image, train_image_path)
            
            # 转换标注为YOLO格式
            with open(train_label_path, 'w') as f:
                for shape in annotation_data.get("shapes", []):
                    label = shape.get("label", "")
                    points = shape.get("points", [])
                    
                    # 检查标签是否在类别列表中
                    if label not in self.agent.class_names:
                        self.log(f"警告: 标签 '{label}' 不在模型类别列表中，将被忽略")
                        continue
                    
                    # 获取类别索引
                    class_id = self.agent.class_names.index(label)
                    
                    # 处理不同形状的标注
                    shape_type = shape.get("shape_type", "")
                    
                    if shape_type == "rectangle":
                        # 矩形标注 - 两个点表示左上和右下角
                        if len(points) == 2:
                            x1, y1 = points[0]
                            x2, y2 = points[1]
                            
                            # 计算YOLO格式的中心点和宽高（归一化坐标）
                            x_center = (x1 + x2) / 2 / img_width
                            y_center = (y1 + y2) / 2 / img_height
                            width = abs(x2 - x1) / img_width
                            height = abs(y2 - y1) / img_height
                            
                            # 写入YOLO格式标注
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    
                    elif shape_type == "polygon":
                        # 多边形标注 - 转换为外接矩形
                        if len(points) >= 3:
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            
                            # 计算外接矩形
                            x1, y1 = min(x_coords), min(y_coords)
                            x2, y2 = max(x_coords), max(y_coords)
                            
                            # 计算YOLO格式的中心点和宽高（归一化坐标）
                            x_center = (x1 + x2) / 2 / img_width
                            y_center = (y1 + y2) / 2 / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height
                            
                            # 写入YOLO格式标注
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    
                    elif shape_type == "circle":
                        # 圆形标注 - 转换为外接正方形
                        if len(points) == 2:
                            # 第一个点是圆心，第二个点是圆上一点
                            center_x, center_y = points[0]
                            edge_x, edge_y = points[1]
                            
                            # 计算半径
                            import math
                            radius = math.sqrt((center_x - edge_x)**2 + (center_y - edge_y)**2)
                            
                            # 计算外接正方形
                            x1, y1 = center_x - radius, center_y - radius
                            x2, y2 = center_x + radius, center_y + radius
                            
                            # 计算YOLO格式的中心点和宽高（归一化坐标）
                            x_center = center_x / img_width
                            y_center = center_y / img_height
                            width = (2 * radius) / img_width
                            height = (2 * radius) / img_height
                            
                            # 写入YOLO格式标注
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    
                    else:
                        self.log(f"警告: 不支持的标注形状类型 '{shape_type}'")
            
            self.log(f"已导入标注并保存到训练集: {train_label_path}")
            
            # 禁用导入按钮
            if hasattr(self, 'import_button'):
                self.import_button.config(state="disabled")
            
            # 更新检测记录
            detection_record["manual_annotated"] = True
            self._update_detection_record(detection_record)

        except Exception as e:
            self.log(f"导入标注失败: {e}")
    
    def _update_detection_record(self, record):
        """更新检测记录到数据库"""
        try:
            # 获取数据库连接
            conn = self.agent.get_db_connection()
            cursor = conn.cursor()
            
            # 准备数据
            image_name = record.get("image", "")
            timestamp = record.get("timestamp", datetime.now().timestamp())
            
            # 如果timestamp是字符串格式，尝试转换为时间戳
            if isinstance(timestamp, str):
                try:
                    # 尝试解析ISO格式的时间字符串
                    if 'T' in timestamp:
                        # 处理可能包含微秒的ISO格式
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except ValueError:
                            # 如果标准fromisoformat失败，尝试手动解析
                            timestamp = timestamp.split('.')[0].replace('T', ' ')
                            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        timestamp = dt.timestamp()
                    else:
                        # 尝试解析标准格式的时间字符串
                        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        timestamp = dt.timestamp()
                except ValueError as e:
                    # 如果解析失败，使用当前时间
                    self.log(f"警告: 无法解析时间戳 '{timestamp}'，使用当前时间: {e}")
                    timestamp = datetime.now().timestamp()
            
            # 转换为datetime对象用于数据库存储
            dt = datetime.fromtimestamp(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            detected_count = record.get("detected_count", 0)
            reviewed = 1 if record.get("reviewed", False) else 0
            correct = 1 if record.get("correct", True) else 0
            false_positive = 1 if record.get("false_positive", False) else 0
            missed_defect = 1 if record.get("missed_defect", False) else 0
            
            # 检查记录是否已存在 - 修改表名为detection_records
            cursor.execute("SELECT id FROM detection_records WHERE image_name = ?", (image_name,))
            result = cursor.fetchone()
            
            if result:
                # 更新现有记录 - 修改表名为detection_records
                cursor.execute("""
                    UPDATE detection_records 
                    SET timestamp = ?, detected_count = ?, reviewed = ?, 
                        correct = ?, false_positive = ?, missed_defect = ?
                    WHERE image_name = ?
                """, (formatted_time, detected_count, reviewed, correct, 
                      false_positive, missed_defect, image_name))
            else:
                # 插入新记录 - 修改表名为detection_records
                cursor.execute("""
                    INSERT INTO detection_records 
                    (image_name, timestamp, detected_count, reviewed, correct, false_positive, missed_defect)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (image_name, formatted_time, detected_count, reviewed, 
                      correct, false_positive, missed_defect))
            
            # 提交事务
            conn.commit()
            
        except Exception as e:
            self.log(f"保存检测记录到数据库失败: {e}")
            import traceback
            self.log(traceback.format_exc())





    def start_ui_update_thread(self):
        """启动UI更新线程，定期检查新的检测结果并更新界面"""
        def update_loop():
            last_history_len = len(self.agent.detections_history)
            
            while self.periodic_detection_active:
                current_len = len(self.agent.detections_history)
                
                # 如果有新的检测结果
                if current_len > last_history_len:
                    # 在主线程中更新UI
                    self.master.after(0, self.update_image_list)
                    self.master.after(0, self.update_stats_plot)
                    self.master.after(0, self.update_trend_plot)
                    
                    # 自动选择最新的检测结果
                    self.master.after(0, lambda: self.image_listbox.select_set(0))
                    self.master.after(0, lambda: self.on_image_select(None))
                    
                    last_history_len = current_len
                
                time.sleep(1)  # 每秒检查一次
        
        threading.Thread(target=update_loop, daemon=True).start()


    def update_image_list(self):
        """更新图像列表显示"""
        try:
            # 清空列表
            self.image_listbox.delete(0, tk.END)
            
            # 获取日期范围
            start_date = None
            end_date = None
            
            if hasattr(self, 'date_filter_enabled') and self.date_filter_enabled:
                try:
                    if hasattr(self, 'start_date') and self.start_date:
                        start_date = self.start_date.get_date()
                    if hasattr(self, 'end_date') and self.end_date:
                        end_date = self.end_date.get_date()
                        # 将end_date调整为当天的23:59:59
                        end_date = datetime.combine(end_date, datetime.max.time())
                except Exception as e:
                    self.log(f"获取日期范围失败: {e}")
                    import traceback
                    self.log(traceback.format_exc())
            
            # 获取检测历史记录
            history = self.agent.detections_history
            
            # 按日期筛选（如果有日期范围）
            if start_date and end_date:
                filtered_history = []
                for record in history:
                    try:
                        # 解析记录的时间戳 - 增强时间戳解析能力
                        timestamp = record.get("timestamp", "")
                        if not timestamp:
                            continue
                            
                        # 尝试多种格式解析时间戳
                        record_time = None
                        try:
                            # 尝试ISO格式 (2025-03-21T09:50:57.452570)
                            if 'T' in timestamp:
                                record_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            # 尝试标准格式 (2025-03-21 09:50:57)
                            else:
                                record_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            # 如果上述格式都失败，尝试其他可能的格式
                            try:
                                # 尝试只有日期的格式 (2025-03-21)
                                record_time = datetime.strptime(timestamp.split()[0], "%Y-%m-%d")
                            except ValueError:
                                self.log(f"无法解析时间戳: {timestamp}")
                                continue
                        
                        if not record_time:
                            continue
                            
                        record_date = record_time.date()
                        # 检查是否在日期范围内
                        if start_date <= record_date <= end_date.date():
                            filtered_history.append(record)
                    except (ValueError, TypeError) as e:
                        # 如果时间戳格式不正确，跳过该记录
                        self.log(f"解析时间戳失败: {e}, 记录: {record.get('timestamp', '')}")
                        continue
                
                history = filtered_history
            
            # 按时间倒序排列
            history = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # 添加到列表
            for i, record in enumerate(history):
                # 获取图像名称和检测结果
                image_name = record.get("image", "未知图像")
                defect_count = record.get("detected_count", 0)
                
                # 获取时间戳
                timestamp = record.get("timestamp", "")
                time_str = ""
                if timestamp:
                    try:
                        # 统一时间戳显示格式
                        if 'T' in timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        time_str = dt.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        time_str = timestamp
                
                # 获取批次ID
                batch_id = record.get("batch_id", "")
                batch_str = f"[批次:{batch_id}] " if batch_id else ""
                
                # 构建显示文本
                display_text = f"{time_str} {batch_str}{image_name} - {defect_count}个缺陷"
                
                # 添加复检状态
                if record.get("reviewed", False):
                    display_text += " - 已复检"
                
                # 添加到列表
                self.image_listbox.insert(tk.END, display_text)
                
                # 根据检测结果设置颜色
                if record.get("false_positive", False):
                    self.image_listbox.itemconfig(i, {'bg': 'light yellow'})  # 误检为黄色
                elif record.get("missed_defect", False):
                    self.image_listbox.itemconfig(i, {'bg': 'light coral'})   # 漏检为红色
                elif record.get("correct", False):
                    self.image_listbox.itemconfig(i, {'bg': 'light green'})   # 正确为绿色
            
            self.log(f"图像列表已更新，共 {len(history)} 条记录")
            # 更新按钮状态
            self.enable_buttons()
            
        except Exception as e:
            self.log(f"更新图像列表失败: {e}")
            import traceback
            self.log(traceback.format_exc())

    def filter_by_date(self):
        """根据选择的日期范围筛选图像列表"""
        if not hasattr(self, 'date_filter_enabled') or not self.date_filter_enabled:
            self.log("日期筛选功能不可用")
            return
            
        try:
            # 更新图像列表
            self.update_image_list()
            
            # 更新统计图表
            if hasattr(self, 'matplotlib_available') and self.matplotlib_available:
                start_date = self.start_date.get_date()
                end_date = self.end_date.get_date()
                
                # 将日期转换为datetime对象
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.max.time())
                
                # 更新统计图表
                self.update_stats_plot(start_datetime, end_datetime)
                self.update_trend_plot(start_datetime, end_datetime)
                
            self.log(f"已筛选 {self.start_date.get_date()} 至 {self.end_date.get_date()} 的记录")
            
        except Exception as e:
            self.log(f"日期筛选失败: {e}")
            import traceback
            self.log(traceback.format_exc())


    def load_detection_history(self, start_date=None, end_date=None):
        """从数据库加载历史检测记录，可指定日期范围"""
        if not self.db_enabled:
            return
            
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 构建查询条件
            query = '''
            SELECT * FROM detection_records 
            '''
            
            params = []
            if start_date and end_date:
                query += "WHERE timestamp BETWEEN ? AND ? "
                params.extend([start_date.isoformat(), end_date.isoformat()])
                
            query += "ORDER BY timestamp DESC"
            
            # 执行查询
            cursor.execute(query, params)
            
            records = cursor.fetchall()
            self.detections_history = []
            
            # 重置统计计数
            self.total_images = 0
            self.reviewed_images = 0
            self.correct_detections = 0
            self.false_positive_images = 0
            self.missed_defect_images = 0
            
            for record in records:
                # 查询该记录的缺陷详情
                cursor.execute('''
                SELECT * FROM defect_details 
                WHERE detection_id = ?
                ''', (record['id'],))
                
                defects = cursor.fetchall()
                detected_defects = []
                
                for defect in defects:
                    detected_defects.append({
                        "class": defect['class_name'],
                        "bbox": [defect['x1'], defect['y1'], defect['x2'], defect['y2']],
                        "confidence": defect['confidence']
                    })
                
                # 构建记录对象
                detection_record = {
                    "db_id": record['id'],
                    "timestamp": record['timestamp'],
                    "image": record['image_name'],
                    "detected_count": record['detected_count'],
                    "defects": detected_defects,
                    "reviewed": bool(record['reviewed']),
                    "correct": bool(record['correct']),
                    "false_positive": bool(record['false_positive']),
                    "missed_defect": bool(record['missed_defect']),
                    "annotated_image": record['annotated_image'],
                    "processed_path": record['processed_path'],
                    "batch_id": record['batch_id']
                }
                
                self.detections_history.append(detection_record)
                
                # 更新统计计数
                self.total_images += 1
                if record['reviewed']:
                    self.reviewed_images += 1
                if record['correct']:
                    self.correct_detections += 1
                if record['false_positive']:
                    self.false_positive_images += 1
                if record['missed_defect']:
                    self.missed_defect_images += 1
            
            conn.close()
            self.logger.info(f"从数据库加载了 {len(self.detections_history)} 条历史记录")
            
        except Exception as e:
            self.logger.error(f"从数据库加载历史记录失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())



    def on_image_select(self, event):
        """图像列表选择事件处理"""
        try:
            # 获取选中的索引
            selection = self.image_listbox.curselection()
            if not selection:
                return
                
            index = selection[0]
            selected_text = self.image_listbox.get(index)
            
            # 从选中文本中提取图像名称
            # 格式: "2023-05-01 12:30 [批次:20230501-1] image.jpg - 3个缺陷 - 已复检"
            import re
            match = re.search(r'(\S+\.(?:jpg|png|jpeg|bmp))', selected_text)
            if not match:
                self.log(f"无法从选中项中提取图像名称: {selected_text}")
                return
                
            image_name = match.group(1)
            
            # 查找对应的记录
            record = None
            for r in self.agent.detections_history:
                if r.get("image") == image_name:
                    record = r
                    break
                    
            if not record:
                self.log(f"找不到图像的检测记录: {image_name}")
                return
            
            # 确保current_images存在
            if not hasattr(self, 'current_images'):
                self.current_images = []
                
            # 如果current_images为空，添加所有历史记录
            if not self.current_images:
                self.current_images = self.agent.detections_history.copy()
                self.log(f"从历史记录加载了 {len(self.current_images)} 个图像")
                
            # 设置当前索引
            if record in self.current_images:
                self.current_image_index = self.current_images.index(record)
            else:
                # 如果记录不在列表中，添加它
                self.current_images.append(record)
                self.current_image_index = len(self.current_images) - 1
                
            self.log(f"选择图像，设置当前索引: {self.current_image_index}, 总图像数: {len(self.current_images)}")
                
            # 显示选中的图像
            self.display_detection_result(record)
            
            # 更新导航按钮状态
            self.update_navigation_buttons()
            
        except Exception as e:
            self.log(f"选择图像时出错: {e}")
            import traceback
            self.log(traceback.format_exc())



    def load_image(self, image_path):
        """加载图像到左侧画布，支持缩放和拖曳"""
        try:
            # 清除当前画布内容
            self.image_canvas.delete("all")
            
            # 加载图像
            image = Image.open(image_path)
            self.original_image = image
            
            # 调整图像大小以适应画布
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # 如果画布尚未渲染，使用默认尺寸
            if canvas_width <= 1:
                canvas_width = 600
            if canvas_height <= 1:
                canvas_height = 600
                
            # 计算缩放比例
            img_width, img_height = image.size
            scale = min(canvas_width/img_width, canvas_height/img_height)
            
            # 缩放图像
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为PhotoImage
            self.tk_image = ImageTk.PhotoImage(resized_image)
            
            # 在画布中央显示图像
            self.image_id = self.image_canvas.create_image(
                canvas_width//2, canvas_height//2, 
                image=self.tk_image, anchor="center",
                tags="image"
            )
            
            # 保存图像信息
            self.current_image = {
                "path": image_path,
                "original": image,
                "scale": scale,
                "x": canvas_width//2,
                "y": canvas_height//2
            }
            
            # 绑定鼠标事件用于拖曳
            self.image_canvas.tag_bind("image", "<ButtonPress-1>", self.start_drag)
            self.image_canvas.tag_bind("image", "<B1-Motion>", self.drag_image)
            
            # 绑定鼠标滚轮事件用于缩放
            self.image_canvas.bind("<MouseWheel>", self.zoom_image)
            
            self.logger.info(f"已加载图像: {os.path.basename(image_path)}")
        except Exception as e:
            self.logger.error(f"加载图像失败: {e}")
    
    def start_drag(self, event):
        """开始拖曳图像"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
    




    def zoom_image(self, event):
        """缩放图像"""
        if not hasattr(self, 'original_image') or not self.original_image:
            return
            
        # 确定缩放方向和比例
        scale_factor = 1.1 if event.delta > 0 else 0.9
        
        # 获取当前图像信息
        old_scale = self.current_scale
        new_scale = old_scale * scale_factor
        
        # 限制缩放范围
        if new_scale < 0.1 or new_scale > 5.0:
            return
            
        # 计算新尺寸
        img_width, img_height = self.original_image.size
        new_width = int(img_width * new_scale)
        new_height = int(img_height * new_scale)
        
        # 缩放图像
        resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.current_image = ImageTk.PhotoImage(resized_image)
        
        # 更新画布上的图像
        self.image_canvas.delete("image")
        self.image_id = self.image_canvas.create_image(
            self.image_x, self.image_y, 
            image=self.current_image, 
            anchor="center",
            tags="image"
        )
        
        # 更新当前比例
        self.current_scale = new_scale
    def update_stats_plot(self, start_date=None, end_date=None):
        """更新统计图表，可选择指定日期范围"""
        try:
            if not hasattr(self, 'matplotlib_available') or not self.matplotlib_available:
                return
                
            # 清除现有图表
            self.stats_canvas.delete("all")
            
            # 移除旧的图表部件（如果存在）
            if hasattr(self, 'stats_canvas_widget') and self.stats_canvas_widget:
                try:
                    self.stats_canvas_widget.get_tk_widget().pack_forget()
                except AttributeError:
                    if hasattr(self.stats_canvas_widget, 'pack_forget'):
                        self.stats_canvas_widget.pack_forget()
                self.stats_canvas_widget = None
                
            # 导入必要的库
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # 创建新的Figure对象
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # 检查是否有足够的数据
            has_data = False
            defect_types = {}
            
            if hasattr(self.agent, 'detections_history') and self.agent.detections_history:
                # 如果指定了日期范围，筛选数据
                filtered_history = self.agent.detections_history
                if start_date and end_date:
                    filtered_history = []
                    for record in self.agent.detections_history:
                        try:
                            # 解析记录的时间戳
                            if isinstance(record.get("timestamp"), str):
                                if 'T' in record.get("timestamp", ""):
                                    try:
                                        record_time = datetime.fromisoformat(record.get("timestamp").replace('Z', '+00:00'))
                                    except ValueError:
                                        timestamp = record.get("timestamp").split('.')[0].replace('T', ' ')
                                        record_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                else:
                                    record_time = datetime.strptime(record.get("timestamp"), "%Y-%m-%d %H:%M:%S")
                            else:
                                record_time = datetime.fromtimestamp(record.get("timestamp"))
                                
                            # 检查是否在日期范围内
                            if start_date <= record_time <= end_date:
                                filtered_history.append(record)
                        except (ValueError, TypeError) as e:
                            self.log(f"解析时间戳失败: {e}, 记录: {record.get('timestamp', '')}")
                            continue
                
                # 统计缺陷类型
                for record in filtered_history:
                    for defect in record.get("defects", []):
                        defect_class = defect.get("class", "未知")
                        defect_types[defect_class] = defect_types.get(defect_class, 0) + 1
                has_data = bool(defect_types)
            
            if has_data:
                # 准备数据
                labels = list(defect_types.keys())
                sizes = list(defect_types.values())
                
                # 如果类型太多，合并小类别
                if len(labels) > 7:
                    # 按数量排序
                    sorted_data = sorted(zip(labels, sizes), key=lambda x: x[1], reverse=True)
                    top_labels = [item[0] for item in sorted_data[:6]]
                    top_sizes = [item[1] for item in sorted_data[:6]]
                    
                    # 合并其他类别
                    other_size = sum(item[1] for item in sorted_data[6:])
                    if other_size > 0:
                        top_labels.append("其他")
                        top_sizes.append(other_size)
                    
                    labels, sizes = top_labels, top_sizes
                
                # 生成颜色
                colors = plt.cm.tab10(range(len(labels)))
                
                # 绘制饼图
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels,  # 不在饼图上直接显示标签，使用图例代替
                    autopct='%1.1f%%',
                    startangle=90, 
                    colors=colors,
                    shadow=True,
                    explode=[0.05] * len(labels)  # 稍微分离每个扇区
                )
                
                # 设置字体大小
                plt.setp(autotexts, size=9, weight="bold")
                

                # 添加标题，包含日期范围
                title = "缺陷类型分布"
                if start_date and end_date:
                    date_format = "%Y-%m-%d"
                    title += f"\n{start_date.strftime(date_format)} 至 {end_date.strftime(date_format)}"
                ax.set_title(title, fontsize=12)
                
                # 确保饼图是圆的
                ax.axis('equal')
            else:
                # 没有数据时，绘制空白图表框架
                title = "缺陷类型分布"
                if start_date and end_date:
                    date_format = "%Y-%m-%d"
                    title += f"\n{start_date.strftime(date_format)} 至 {end_date.strftime(date_format)}"
                ax.set_title(title, fontsize=12)
                ax.set_aspect('equal')
                
                # 添加提示文本
                ax.text(0.5, 0.5, "暂无缺陷数据", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=12,
                       color='gray')
                
                # 隐藏坐标轴
                ax.set_axis_off()
            
            # 创建canvas
            canvas = FigureCanvasTkAgg(fig, master=self.stats_canvas)
            self.stats_canvas_widget = canvas.get_tk_widget()
            self.stats_canvas_widget.pack(fill='both', expand=True)
            
            # 绘制图表
            canvas.draw()
            
            if not has_data:
                self.log("暂无缺陷数据，已显示空白统计图表")
            
        except Exception as e:
            self.log(f"更新统计图表失败: {e}")
            import traceback
            self.log(traceback.format_exc())







    def update_trend_plot(self, start_date=None, end_date=None):
        """更新趋势图表，可选择指定日期范围"""
        try:
            if not hasattr(self, 'matplotlib_available') or not self.matplotlib_available:
                self.log("matplotlib不可用，无法绘制趋势图")
                return
                
            self.log("开始更新趋势图...")
            
            # 清除现有图表
            self.trend_canvas.delete("all")
            
            # 移除旧的图表部件（如果存在）
            if hasattr(self, 'trend_canvas_widget') and self.trend_canvas_widget:
                try:
                    self.trend_canvas_widget.get_tk_widget().pack_forget()
                except AttributeError:
                    if hasattr(self.trend_canvas_widget, 'pack_forget'):
                        self.trend_canvas_widget.pack_forget()
                self.trend_canvas_widget = None
                
            # 导入必要的库
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.dates as mdates
            
            # 创建新的Figure对象
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # 检查是否有足够的数据
            has_data = False
            dates = []
            defect_counts = []
            
            # 从数据库获取历史数据
            self.log("正在从数据库获取历史数据...")
            history_data = self._get_detection_history_from_db()
            
            self.log(f"获取到 {len(history_data)} 条历史记录")
            
            if history_data:
                # 如果指定了日期范围，筛选数据
                filtered_history = history_data
                if start_date and end_date:
                    filtered_history = []
                    for record in history_data:
                        try:
                            # 解析记录的时间戳
                            if isinstance(record.get("timestamp"), str):
                                if 'T' in record.get("timestamp", ""):
                                    try:
                                        record_time = datetime.fromisoformat(record.get("timestamp").replace('Z', '+00:00'))
                                    except ValueError:
                                        timestamp = record.get("timestamp").split('.')[0].replace('T', ' ')
                                        record_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                else:
                                    record_time = datetime.strptime(record.get("timestamp"), "%Y-%m-%d %H:%M:%S")
                            else:
                                record_time = datetime.fromtimestamp(record.get("timestamp"))
                                
                            # 检查是否在日期范围内
                            if start_date <= record_time <= end_date:
                                filtered_history.append(record)
                        except (ValueError, TypeError) as e:
                            self.log(f"解析时间戳失败: {e}, 记录: {record.get('timestamp', '')}")
                            continue
                
                self.log(f"筛选后有 {len(filtered_history)} 条记录")
                
                # 按日期分组统计缺陷数量
                date_defects = {}
                for i, record in enumerate(filtered_history):
                    try:
                        # 解析时间戳获取日期
                        if isinstance(record.get("timestamp"), str):
                            if 'T' in record.get("timestamp", ""):
                                try:
                                    record_time = datetime.fromisoformat(record.get("timestamp").replace('Z', '+00:00'))
                                except ValueError:
                                    timestamp = record.get("timestamp").split('.')[0].replace('T', ' ')
                                    record_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                            else:
                                record_time = datetime.strptime(record.get("timestamp"), "%Y-%m-%d %H:%M:%S")
                        else:
                            record_time = datetime.fromtimestamp(record.get("timestamp"))
                        
                        # 只取日期部分
                        date_only = record_time.date()
                        date_str = date_only.strftime("%Y-%m-%d")
                        
                        # 调试信息：输出记录中的缺陷信息
                        defects = record.get("defects", [])
                        defect_count = len(defects)
                        
                        # 每10条记录输出一次详细信息，避免日志过多
                        if i < 10 or i % 10 == 0:
                            self.log(f"记录 {i+1}: 日期={date_str}, 缺陷数={defect_count}, 缺陷类型={[d.get('class', '未知') for d in defects]}")
                        
                        # 统计该日期的缺陷数
                        if date_str in date_defects:
                            date_defects[date_str] += defect_count
                        else:
                            date_defects[date_str] = defect_count
                    except Exception as e:
                        self.log(f"处理记录时间戳失败: {e}")
                        import traceback
                        self.log(traceback.format_exc())
                        continue
                
                self.log(f"按日期统计后有 {len(date_defects)} 个不同日期")
                for date_str, count in date_defects.items():
                    self.log(f"日期: {date_str}, 缺陷总数: {count}")
                
                # 转换为列表并排序
                if date_defects:
                    sorted_dates = sorted(date_defects.keys())
                    dates = sorted_dates
                    defect_counts = [date_defects[date] for date in sorted_dates]
                    has_data = True
                    self.log(f"最终用于绘图的数据: 日期数={len(dates)}, 缺陷数={defect_counts}")
            
            # 其余代码保持不变
            if has_data:
                # 转换日期字符串为datetime对象用于绘图
                date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
                
                # 绘制趋势线
                line, = ax.plot(date_objects, defect_counts, 'o-', linewidth=2, markersize=6, color='#1f77b4', label='缺陷数量')
                
                # 设置x轴日期格式
                date_fmt = mdates.DateFormatter('%m-%d')
                ax.xaxis.set_major_formatter(date_fmt)
                
                # 根据数据点数量调整x轴刻度
                if len(dates) > 10:
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=len(dates)//10 + 1))
                else:
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                
                # 旋转日期标签，避免重叠
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # 添加网格线
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 添加标签
                ax.set_xlabel('日期')
                ax.set_ylabel('缺陷数量')
                
                # 添加图例
                ax.legend(loc='upper left')
                
                # 添加标题，包含日期范围
                title = "缺陷检测趋势"
                if start_date and end_date:
                    date_format = "%Y-%m-%d"
                    title += f"\n{start_date.strftime(date_format)} 至 {end_date.strftime(date_format)}"
                ax.set_title(title, fontsize=12)
                
                # 调整布局
                fig.tight_layout()
                self.log("成功绘制趋势图")
            else:
                # 没有数据时，绘制空白图表框架
                title = "缺陷检测趋势"
                if start_date and end_date:
                    date_format = "%Y-%m-%d"
                    title += f"\n{start_date.strftime(date_format)} 至 {end_date.strftime(date_format)}"
                ax.set_title(title, fontsize=12)
                
                # 添加提示文本
                ax.text(0.5, 0.5, "暂无趋势数据", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=12,
                       color='gray')
                
                # 设置坐标轴标签
                ax.set_xlabel('日期')
                ax.set_ylabel('缺陷数量')
                
                # 添加网格线
                ax.grid(True, linestyle='--', alpha=0.3)
                self.log("无数据可用，显示空白趋势图")
            
            # 创建canvas
            canvas = FigureCanvasTkAgg(fig, master=self.trend_canvas)
            self.trend_canvas_widget = canvas.get_tk_widget()
            self.trend_canvas_widget.pack(fill='both', expand=True)
            
            # 绘制图表
            canvas.draw()
            
            if not has_data:
                self.log("暂无趋势数据，已显示空白趋势图表")
            
        except Exception as e:
            self.log(f"更新趋势图表失败: {e}")
            import traceback
            self.log(traceback.format_exc())


    def _get_detection_history_from_db(self):
        """从数据库获取检测历史数据"""
        try:
            if not hasattr(self.agent, 'db_path') or not self.agent.db_path:
                self.log("数据库路径未设置，使用内存中的历史记录")
                return self.agent.detections_history
                
            self.log(f"正在从数据库 {self.agent.db_path} 获取历史数据...")
            
            import sqlite3
            import json
            conn = sqlite3.connect(self.agent.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection_records'")
            if not cursor.fetchone():
                self.log("数据库中不存在detection_records表")
                conn.close()
                return self.agent.detections_history
            
            # 检查defects列是否存在
            cursor.execute("PRAGMA table_info(detection_records)")
            columns = [column[1] for column in cursor.fetchall()]
            has_defects_column = 'defects' in columns
            
            if has_defects_column:
                self.log("数据库表包含defects列，将直接读取")
            else:
                self.log("数据库表不包含defects列，将从defect_details表读取")
            
            # 查询所有检测记录
            cursor.execute("""
                SELECT * FROM detection_records 
                ORDER BY timestamp DESC
            """)
            
            records = []
            rows = cursor.fetchall()
            self.log(f"从数据库获取到 {len(rows)} 条记录")
            
            for i, row in enumerate(rows):
                record = dict(row)
                
                # 转换布尔值字段
                for field in ['reviewed', 'correct', 'false_positive', 'missed_defect']:
                    record[field] = bool(record.get(field, 0))
                
                # 获取缺陷列表
                defects = []
                
                # 如果有defects列，尝试从中解析
                if has_defects_column and record.get('defects'):
                    try:
                        defects_json = record.get('defects', '[]')
                        defects = json.loads(defects_json)
                        self.log(f"记录 {i+1}: 从defects列解析到 {len(defects)} 个缺陷")
                    except Exception as json_error:
                        self.log(f"解析缺陷JSON失败: {json_error}, JSON: {defects_json}")
                        defects = []
                
                # 如果defects为空，尝试从defect_details表获取
                if not defects:
                    try:
                        # 查询缺陷详情
                        cursor.execute("""
                            SELECT * FROM defect_details 
                            WHERE detection_id = ?
                        """, (record.get('id'),))
                        
                        defect_details = cursor.fetchall()
                        if defect_details:
                            for defect in defect_details:
                                defects.append({
                                    "class": defect['class_name'],
                                    "bbox": [defect['x1'], defect['y1'], defect['x2'], defect['y2']],
                                    "confidence": defect['confidence']
                                })
                            self.log(f"记录 {i+1}: 从defect_details表获取到 {len(defects)} 个缺陷")
                    except Exception as e:
                        self.log(f"从defect_details获取缺陷失败: {e}")
                
                # 更新记录中的缺陷列表
                record['defects'] = defects
                
                # 确保detected_count与实际缺陷数量一致
                if record.get('detected_count', 0) != len(defects) and len(defects) > 0:
                    self.log(f"记录 {i+1}: 缺陷数量不一致，更新detected_count从 {record.get('detected_count', 0)} 到 {len(defects)}")
                    record['detected_count'] = len(defects)
                
                # 每10条记录输出一次详细信息
                if i < 10 or i % 10 == 0:
                    self.log(f"记录 {i+1}: ID={record.get('id')}, 图像={record.get('image_name')}, 缺陷数量={len(defects)}")
                    if defects:
                        self.log(f"缺陷类型: {[d.get('class', '未知') for d in defects]}")
                
                records.append(record)
            
            conn.close()
            self.log(f"成功从数据库获取并处理了 {len(records)} 条历史记录")
            return records
            
        except Exception as e:
            self.log(f"从数据库获取历史数据失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return self.agent.detections_history




    def start_periodic_detection(self):
        """启动定期检测新图片"""
        if self.detection_thread and self.detection_thread.is_alive():
            self.logger.info("定期检测已在运行中")
            return

        def detection_loop():
            while not self.stop_detection:
                try:
                    # 获取当前目录下的所有图片文件
                    current_files = set(
                        f for f in os.listdir(self.new_data_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
                    )
                    
                    # 找出新文件
                    new_files = current_files - self.known_files
                    if new_files:
                        self.logger.info(f"发现 {len(new_files)} 张新图片")
                        
                        # 根据自动检测模式决定检测方式
                        if hasattr(self, 'gui') and self.gui:
                            # 检查GUI中的自动检测设置
                            auto_detect = self.gui.auto_detect_var.get() if hasattr(self.gui, 'auto_detect_var') else self.auto_detection
                        else:
                            auto_detect = self.auto_detection
                            
                        self.logger.info(f"自动检测模式: {auto_detect}")
                        
                        if auto_detect:
                            # 自动模式：直接检测
                            self.detect_new_data(single_image_mode=True)
                            # 更新已知文件集合
                            processed_files = set(
                                f for f in os.listdir(self.processed_dir)
                                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))
                            )
                            self.known_files.update(processed_files)
                        else:
                            # 手动模式：通知GUI有新图像，但不自动检测
                            if hasattr(self, 'gui') and self.gui:
                                self.gui.master.after(0, lambda: self.gui.log(f"发现 {len(new_files)} 张新图像，请手动检测"))
                                # 可以添加一个提示或者闪烁按钮等
                                self.gui.master.after(0, self.gui.highlight_detect_button)
                    
                except Exception as e:
                    self.logger.error(f"检测新图片时出错: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                
                time.sleep(self.detection_interval)

        self.stop_detection = False
        self.detection_thread = threading.Thread(target=detection_loop, daemon=True)
        self.detection_thread.start()
        self.logger.info(f"定期检测已启动，每 {self.detection_interval} 秒检测一次新图片")

        
    def highlight_detect_button(self):
        """高亮显示检测按钮，提示用户有新图像需要检测"""
        if hasattr(self, 'detect_button'):
            # 保存原始背景色
            original_bg = self.detect_button.cget('background')
            
            # 闪烁效果
            def flash(count=0):
                if count >= 6:  # 闪烁3次
                    self.detect_button.config(background=original_bg)
                    return
                
                new_bg = "#ff6666" if count % 2 == 0 else original_bg
                self.detect_button.config(background=new_bg)
                self.master.after(500, lambda: flash(count + 1))
                
            flash()



    def run_detection_once(self):
        """执行一次检测并更新UI"""
        # 检查是否有新图像
        try:
            image_files = [f for f in os.listdir(self.agent.new_data_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
                        
            if image_files and not self.agent.detecting and not self.agent.training:
                self.log("定期检测：发现新图像，开始检测...")
                
                # 在后台线程执行检测
                def detect_task():
                    summary_list = self.agent.detect_new_data()
                    # 检测完成后在主线程更新UI
                    self.master.after(0, lambda: self.update_detection_results(summary_list))
                    self.master.after(0, self.update_image_list)
                    self.master.after(0, self.update_trend_plot)
                    
                    # 如果有检测结果，自动选择第一个
                    if summary_list:
                        self.master.after(0, lambda: self.image_listbox.select_set(0))
                        self.master.after(0, lambda: self.on_image_select(None))
                        
                threading.Thread(target=detect_task, daemon=True).start()
            else:
                if self.agent.detecting:
                    self.log("定期检测：检测任务正在进行中，跳过本次检测")
                elif self.agent.training:
                    self.log("定期检测：训练任务正在进行中，跳过本次检测")
                else:
                    self.log("定期检测：未发现新图像")
        except Exception as e:
            self.log(f"定期检测出错: {e}")
            









    def log(self, message):
        """在日志区域添加一条消息。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(END, f"[{timestamp}] {message}")
        self.log_text.yview_moveto(1)  # 滚动到底部


    def run_detection(self):
        """点击"一键启动检测"按钮后执行检测的逻辑。"""
        self.log("开始检测新PCB图像...")
        

        # 新建线程执行检测，以免阻塞 GUI
        def detect_task():
            summary_list = self.agent.detect_new_data()
            # 检测完成后在主线程更新UI
            self.master.after(0, lambda: self.update_detection_results(summary_list))
            self.master.after(0, self.update_image_list)
            self.master.after(0, self.update_stats_plot)
            self.master.after(0, self.update_trend_plot)
            
            # 启用按钮
            self.master.after(0, lambda: self.update_navigation_buttons())  # 使用新方法替代enable_buttons
            
            # 如果有检测结果，自动选择第一个
            if summary_list:
                self.master.after(0, lambda: self.image_listbox.select_set(0))
                self.master.after(0, lambda: self.on_image_select(None))
        
        threading.Thread(target=detect_task, daemon=True).start()



    def enable_buttons(self):
        """根据图像列表状态启用/禁用按钮"""
        # 检查列表是否包含图像
        has_images = self.image_listbox.size() > 0
        
        # 设置按钮状态
        button_state = "normal" if has_images else "disabled"
        for btn in ["pass_button", "fail_button", "prev_button", "next_button"]:
            if hasattr(self, btn):
                getattr(self, btn).config(state=button_state)
        
        # 确保在检测完成后也启用这些按钮
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """更新导航按钮状态"""
        # 检查是否有图像列表
        has_images = self.image_listbox.size() > 0
        
        if not has_images:
            # 如果没有图像，禁用所有按钮
            for btn in ["pass_button", "fail_button", "prev_button", "next_button"]:
                if hasattr(self, btn):
                    getattr(self, btn).config(state="disabled")
            return
            
        # 获取当前选中的索引
        selection = self.image_listbox.curselection()
        current_index = selection[0] if selection else -1
        
        # 启用/禁用上一张/下一张按钮
        if hasattr(self, 'prev_button'):
            self.prev_button.config(state="normal" if current_index > 0 else "disabled")
        
        if hasattr(self, 'next_button'):
            self.next_button.config(state="normal" if current_index < self.image_listbox.size() - 1 else "disabled")
        
        # 通过/不通过按钮始终启用（只要有图像）
        for btn in ["pass_button", "fail_button"]:
            if hasattr(self, btn):
                getattr(self, btn).config(state="normal")

    def start_training(self):
        """人工确认后启动模型训练"""
        confirm = messagebox.askyesno("确认", "确定开始训练模型吗？\n请确保已有足够的正确标注数据。")
        if not confirm:
            return
        self.log("模型训练开始...")
        # 禁用训练按钮，防止重复点击
        self.train_button.config(state="disabled")
        self.progress_var.set(0)  # 重置进度条
        # 在后台线程运行训练任务，并定期更新UI状态
        def train_task():
            started = self.agent.start_training()
            if not started:
                # 如果训练已在进行，恢复按钮状态并提示
                self.master.after(0, lambda: self.log("训练已在进行中。"))
                self.master.after(0, lambda: self.train_button.config(state="normal"))
                return
            # 模拟训练进度更新（这里简单每隔5秒记录日志，可根据实际情况更新进度条）
            while self.agent.training:
                self.master.after(0, lambda: self.log("训练进行中..."))
                time.sleep(5)
            # 训练结束，恢复UI状态并通知用户
            self.master.after(0, self.on_training_complete)
        threading.Thread(target=train_task, daemon=True).start()



    def toggle_training_pause(self):
        """暂停或恢复训练"""
        if not self.agent.training:
            return
            
        if self.agent.training_paused:
            # 恢复训练
            self.agent.resume_training()  # 修正：调用resume_training而不是start_training
            self.pause_button.config(text="暂停训练")
            self.log("训练已恢复...")
        else:
            # 暂停训练
            self.agent.pause_training()
            self.pause_button.config(text="恢复训练")
            self.log("训练已暂停，可以稍后恢复...")

    def update_detection_results(self, summary_list):
        """更新检测结果列表和日志。"""
        if not summary_list:
            self.log("无新图像或未检测到缺陷。")
        else:
            # 清空旧列表，插入新结果
            self.result_list.delete(0, END)
            for summary in summary_list:
                self.result_list.insert(END, summary)
            self.log(f"检测完成，共处理 {len(summary_list)} 张图像，请进行人工复检。")

    def run_semi_supervised(self):
        """运行半监督学习，生成伪标签"""
        try:
            # 确保Agent有半监督学习方法
            if not hasattr(self.agent, "semi_supervised_learning"):
                self.log("错误：Agent未实现半监督学习方法。")
                return
                
            # 运行半监督学习
            result = self.agent.semi_supervised_learning(
                self.agent.unlabeled_data_dir, 
                self.agent.pseudo_label_conf
            )
            
            if result:
                self.log("半监督学习完成，已生成伪标签。")
            else:
                self.log("半监督学习未生成任何伪标签，请检查未标注数据。")
        except Exception as e:
            self.log(f"半监督学习过程出错: {e}")

    def show_advanced_training(self):
        """显示高级训练选项对话框"""
        adv_window = Toplevel(self.master)
        adv_window.title("高级训练选项")
        adv_window.geometry("500x400")
        adv_window.resizable(False, False)
        
        # 创建选项框架
        options_frame = Frame(adv_window, padx=10, pady=10)
        options_frame.pack(fill="both", expand=True)
        
        # 分阶段训练选项 - 使用agent中的实际值初始化
        staged_var = IntVar(value=1 if self.agent.use_staged_training else 0)
        staged_check = Checkbutton(options_frame, text="启用分阶段训练 (先训练部分层，再训练全部层)", 
                                  variable=staged_var)
        staged_check.pack(anchor="w", pady=5)
        
        # 迁移学习选项 - 使用agent中的实际值初始化
        transfer_var = IntVar(value=1 if self.agent.use_transfer_learning else 0)
        transfer_check = Checkbutton(options_frame, text="启用迁移学习 (利用预训练模型知识)", 
                                    variable=transfer_var)
        transfer_check.pack(anchor="w", pady=5)
        
        # 半监督学习选项 - 使用agent中的实际值初始化
        semi_var = IntVar(value=1 if self.agent.use_semi_supervised else 0)
        semi_check = Checkbutton(options_frame, text="启用半监督学习 (利用未标注数据)", 
                               variable=semi_var)
        semi_check.pack(anchor="w", pady=5)
        
        # 未标注数据路径
        path_frame = Frame(options_frame)
        path_frame.pack(fill="x", pady=5)
        Label(path_frame, text="未标注数据路径:").pack(side="left")
        unlabeled_path = StringVar(value=self.agent.unlabeled_data_dir)
        path_entry = Entry(path_frame, textvariable=unlabeled_path, width=30)
        path_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # 修复：移除重复的路径输入框
        # 浏览按钮
        browse_button = Button(path_frame, text="浏览...", 
                              command=lambda: unlabeled_path.set(filedialog.askdirectory()))
        browse_button.pack(side="left")
        
        # 伪标签置信度阈值
        conf_frame = Frame(options_frame)
        conf_frame.pack(fill="x", pady=5)
        Label(conf_frame, text="伪标签置信度阈值:").pack(side="left")
        conf_var = DoubleVar(value=self.agent.pseudo_label_conf)
        conf_scale = Scale(conf_frame, from_=0.1, to=0.9, resolution=0.1, 
                          orient="horizontal", variable=conf_var)
        conf_scale.pack(side="left", fill="x", expand=True)
        
        # 数据增强选项
        aug_var = IntVar(value=0)  # 默认不勾选
        aug_check = Checkbutton(options_frame, text="启用数据增强 (生成更多训练样本)", 
                              variable=aug_var)
        aug_check.pack(anchor="w", pady=5)
        
        # 增强数量
        aug_frame = Frame(options_frame)
        aug_frame.pack(fill="x", pady=5)
        Label(aug_frame, text="每张图像增强数量:").pack(side="left")
        aug_count_var = IntVar(value=self.agent.augmentation_count)
        aug_count_scale = Scale(aug_frame, from_=1, to=10, resolution=1, 
                               orient="horizontal", variable=aug_count_var)
        aug_count_scale.pack(side="left", fill="x", expand=True)
        
        # 按钮框架
        button_frame = Frame(adv_window, pady=10)
        button_frame.pack(fill="x")
        
        
        # 保存按钮
        def save_options():
            # 更新agent中的训练选项
            self.agent.use_staged_training = bool(staged_var.get())
            self.agent.use_transfer_learning = bool(transfer_var.get())
            self.agent.use_semi_supervised = bool(semi_var.get())
            self.agent.unlabeled_data_dir = unlabeled_path.get()
            self.agent.pseudo_label_conf = conf_var.get()
            self.agent.augmentation_enabled = bool(aug_var.get())
            self.agent.augmentation_count = aug_count_var.get()
            
            # 确保目录存在
            os.makedirs(self.agent.unlabeled_data_dir, exist_ok=True)
            
            
            self.log(f"已保存高级训练选项")
            self.log(f"- 分阶段训练: {'启用' if self.agent.use_staged_training else '禁用'}")
            self.log(f"- 迁移学习: {'启用' if self.agent.use_transfer_learning else '禁用'}")
            self.log(f"- 半监督学习: {'启用' if self.agent.use_semi_supervised else '禁用'}")
            self.log(f"- 数据增强: {'启用' if self.agent.augmentation_enabled else '禁用'}")
            
            adv_window.destroy()
            
        save_button = Button(button_frame, text="保存设置", command=save_options)
        save_button.pack(side="right", padx=10)
        
        # 取消按钮
        cancel_button = Button(button_frame, text="取消", command=adv_window.destroy)
        cancel_button.pack(side="right", padx=10)



    # 在YOLOAgentGUI类中完善open_review_window方法
    def open_review_window(self, target_image=None):
        """打开人工复检窗口，可以指定目标图像"""
        if not self.agent.detections_history:
            messagebox.showinfo("提示", "没有检测记录可供复检。")
            return
            
        review_win = Toplevel(self.master)
        review_win.title("人工复检")
        review_win.geometry("800x600")
        
        # 创建检测记录列表
        list_frame = Frame(review_win)
        list_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        Label(list_frame, text="检测记录:").pack(anchor="w")
        records_list = Listbox(list_frame, width=40)
        records_list.pack(fill="both", expand=True)
        
        # 填充检测记录
        for i, record in enumerate(self.agent.detections_history):
            status = "已复检" if record["reviewed"] else "未复检"
            records_list.insert(END, f"{record['image']} - {record['detected_count']}个缺陷 - {status}")
        
        # 如果指定了目标图像，选中对应的记录
        if target_image:
            for i in range(records_list.size()):
                if target_image in records_list.get(i):
                    records_list.selection_clear(0, END)
                    records_list.selection_set(i)
                    records_list.see(i)  # 确保目标项可见
                    break
        
        # 创建图像预览和复检控制区域
        preview_frame = Frame(review_win)
        preview_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # 图像预览标签
        preview_label = Label(preview_frame, text="图像预览:")
        preview_label.pack(anchor="w")
        
        # 使用Canvas替代Label以支持缩放和拖曳
        canvas = tk.Canvas(preview_frame, bg="black")
        canvas.pack(fill="both", expand=True)
        
        # 创建缺陷信息框架
        defect_info_frame = Frame(preview_frame, relief="groove", bd=2)
        defect_info_frame.pack(side="bottom", fill="x", padx=10, pady=5, before=canvas)
        
        # 缺陷信息标签
        defect_info_label = Label(defect_info_frame, text="缺陷信息", font=("Arial", 10, "bold"))
        defect_info_label.pack(anchor="w", padx=5, pady=5)
        
        # 缺陷详情文本
        defect_details_label = Label(defect_info_frame, text="请选择一条记录查看详情", 
                                    justify="left", anchor="w", wraplength=350)
        defect_details_label.pack(fill="x", padx=5, pady=5)
        
        # 复检控制按钮
        control_frame = Frame(preview_frame)
        control_frame.pack(side="bottom", fill="x", pady=10)
        
        # 复检结果变量
        false_positive_var = IntVar(value=0)
        missed_defect_var = IntVar(value=0)
        
        # 复检选项
        options_frame = Frame(control_frame)
        options_frame.pack(fill="x", pady=5)
        
        Checkbutton(options_frame, text="存在误检", variable=false_positive_var).pack(side="left", padx=10)
        Checkbutton(options_frame, text="存在漏检", variable=missed_defect_var).pack(side="left", padx=10)
        
        # 当前选中的记录索引
        current_index = [-1]  # 使用列表以便在内部函数中修改
        current_record = [None]  # 保存当前选中的记录
        
        # 添加人工标注按钮
        def start_manual_annotation():
            try:
                if current_record[0] is None:
                    messagebox.showwarning("提示", "请先选择一个图像")
                    return
                    
                # 创建训练目录
                train_dir = os.path.join("D:", "030923", "data", "train")
                images_dir = os.path.join(train_dir, "images_0")
                json_dir = os.path.join(train_dir, "json")
                
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(json_dir, exist_ok=True)
                
                # 获取原始图像路径
                original_image = current_record[0].get("processed_path")
                if not original_image or not os.path.exists(original_image):
                    messagebox.showerror("错误", "找不到原始图像")
                    return
                    
                # 获取图像文件名（不含扩展名）
                image_name = os.path.splitext(os.path.basename(original_image))[0]
                
                # 目标图像路径
                target_image_path = os.path.join(images_dir, f"{image_name}.jpg")
                
                # 复制原始图像到训练目录
                import shutil
                shutil.copy2(original_image, target_image_path)
                
                self.log(f"已保存图像到训练集: {target_image_path}")
                
                # 启动labelme
                import subprocess
                subprocess.Popen(f"labelme \"{target_image_path}\"", shell=True)
                
            except Exception as e:
                self.logger.error(f"启动标注失败: {e}")
                messagebox.showerror("错误", f"启动标注失败: {e}")
        
        # 按钮框架 - 居中放置按钮
        buttons_frame = Frame(control_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        # 创建一个内部框架用于居中按钮
        center_buttons_frame = Frame(buttons_frame)
        center_buttons_frame.pack(side="top", fill="x")
        
        # 添加人工标注按钮
        annotation_btn = Button(center_buttons_frame, text="人工标注", command=start_manual_annotation, width=15)
        annotation_btn.pack(side="left", padx=10, pady=5)
        
        # 提交复检结果
        def submit_review():
            if current_index[0] < 0:
                messagebox.showinfo("提示", "请先选择一条记录。")
                return
                
            index = current_index[0]
            false_positive = bool(false_positive_var.get())
            missed_defect = bool(missed_defect_var.get())
            
            # 提交复检结果
            self.agent.manual_review(index, false_positive, missed_defect)
            
            # 更新列表显示
            record = self.agent.detections_history[index]
            status = "已复检" if record["reviewed"] else "未复检"
            records_list.delete(index)
            records_list.insert(index, f"{record['image']} - {record['detected_count']}个缺陷 - {status}")
            
            messagebox.showinfo("成功", "复检结果已提交。")
            
            # 更新统计图表
            self.update_stats_plot()
            self.update_trend_plot()
        
        # 提交按钮
        submit_button = Button(center_buttons_frame, text="提交复检结果", command=submit_review, width=15)
        submit_button.pack(side="left", padx=10, pady=5)
        
        # 选择记录时的回调函数
        def on_record_select(event):
            selection = records_list.curselection()
            if not selection:
                return
                
            index = selection[0]
            current_index[0] = index
            current_record[0] = self.agent.detections_history[index]
            record = current_record[0]
            
            # 更新复选框状态
            false_positive_var.set(1 if record.get("false_positive", False) else 0)
            missed_defect_var.set(1 if record.get("missed_defect", False) else 0)
            
            # 更新缺陷信息
            defects = record.get("defects", [])
            defect_info = f"检测到 {len(defects)} 个缺陷:\n"
            for i, defect in enumerate(defects):
                class_name = defect.get("class", "未知")
                confidence = defect.get("confidence", 0.0)
                defect_info += f"{i+1}. {class_name} (置信度: {confidence:.2f})\n"
            
            # 更新缺陷信息标签
            defect_details_label.config(text=defect_info)
            
            # 加载并显示图像
            annotated_path = record.get("annotated_image")
            if annotated_path and os.path.exists(annotated_path):
                try:
                    # 清除画布
                    canvas.delete("all")
                    
                    # 加载图像
                    img = Image.open(annotated_path)
                    canvas.original_image = img  # 保存原始图像
                    
                    # 调整图像大小以适应显示区域
                    canvas_width = canvas.winfo_width() or 400
                    canvas_height = canvas.winfo_height() or 400
                    
                    # 计算缩放比例
                    img_width, img_height = img.size
                    scale = min(canvas_width/img_width, canvas_height/img_height)
                    
                    # 缩放图像
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    resized_image = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # 转换为PhotoImage
                    photo = ImageTk.PhotoImage(resized_image)
                    canvas.photo = photo  # 保持引用以防止垃圾回收
                    
                    # 在画布中央显示图像
                    canvas.image_id = canvas.create_image(
                        canvas_width//2, canvas_height//2, 
                        image=photo, anchor="center",
                        tags="image"
                    )
                    
                    # 保存图像信息用于缩放和拖曳
                    canvas.current_scale = scale
                    canvas.image_x = canvas_width//2
                    canvas.image_y = canvas_height//2
                    
                except Exception as e:
                    self.log(f"无法加载图像: {e}")
                    canvas.create_text(canvas_width//2, canvas_height//2, 
                                      text=f"无法加载图像: {e}", fill="white")
            else:
                canvas.delete("all")
                canvas_width = canvas.winfo_width() or 400
                canvas_height = canvas.winfo_height() or 400
                canvas.create_text(canvas_width//2, canvas_height//2, 
                                  text="[图像不可用]", fill="white")
        
        # 图像拖曳功能
        def start_drag(event):
            canvas.drag_start_x = event.x
            canvas.drag_start_y = event.y
        
        def drag_image(event):
            if not hasattr(canvas, 'drag_start_x'):
                return
                
            # 计算移动距离
            dx = event.x - canvas.drag_start_x
            dy = event.y - canvas.drag_start_y
            
            # 更新图像位置
            canvas.move("image", dx, dy)
            
            # 更新拖曳起点
            canvas.drag_start_x = event.x
            canvas.drag_start_y = event.y
            
            # 更新图像位置信息
            if hasattr(canvas, 'image_x') and hasattr(canvas, 'image_y'):
                canvas.image_x += dx
                canvas.image_y += dy
        
        # 图像缩放功能
        def zoom_image(event):
            if not hasattr(canvas, 'original_image') or not canvas.original_image:
                return
                
            # 确定缩放方向和比例
            scale_factor = 1.1 if event.delta > 0 else 0.9
            
            # 获取当前图像信息
            img = canvas.original_image
            old_scale = canvas.current_scale
            new_scale = old_scale * scale_factor
            
            # 限制缩放范围
            if new_scale < 0.1 or new_scale > 5.0:
                return
                
            # 计算新尺寸
            img_width, img_height = img.size
            new_width = int(img_width * new_scale)
            new_height = int(img_height * new_scale)
            
            # 缩放图像
            resized_image = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            
            # 更新画布上的图像
            canvas.delete("image")
            canvas.photo = photo  # 更新引用
            canvas.image_id = canvas.create_image(
                canvas.image_x, canvas.image_y, 
                image=photo, anchor="center",
                tags="image"
            )
            
            # 更新当前比例
            canvas.current_scale = new_scale
        
        # 绑定鼠标事件
        canvas.tag_bind("image", "<ButtonPress-1>", start_drag)
        canvas.tag_bind("image", "<B1-Motion>", drag_image)
        canvas.bind("<MouseWheel>", zoom_image)
        
        # 绑定列表选择事件
        records_list.bind('<<ListboxSelect>>', on_record_select)




    def on_training_complete(self):
        """模型训练完成后的 UI 更新处理。"""
        self.log("模型训练完成。")
        self.train_button.config(state="normal")
        self.pause_button.config(state="disabled")  # 训练结束时禁用暂停按钮
        # 更新模型选择下拉列表
        options = ["当前最佳模型"]
        if self.agent.last_model_path:
            options.append("最近训练模型")
        self.model_combo.config(values=options)
        self.model_combo.set("当前最佳模型")
        # 弹出提示告知训练结果
        if self.agent.last_map is not None:
            if self.agent.best_map is not None and self.agent.last_map < self.agent.best_map:
                messagebox.showinfo("训练完成", "新模型性能未提升，已回滚至之前的最佳模型。")
            else:
                messagebox.showinfo("训练完成", f"新模型训练完成！\nmAP={self.agent.last_map:.4f}\n最佳模型已更新。")

    def apply_model_selection(self):
        """应用所选模型"""
        selection = self.model_var.get()
        if selection == "当前最佳模型":
            self.log("已选择当前最佳模型。")
            return
        elif selection == "最近训练模型" and self.agent.last_model_path:
            # 切换到最近训练的模型
            if self.agent.load_model(self.agent.last_model_path):
                self.log(f"已切换到最近训练模型: {os.path.basename(self.agent.last_model_path)}")
            else:
                self.log("切换模型失败。")
        else:
            self.log("无效的模型选择。")



    def cleanup_data(self):
        """清理指定日期范围内的数据"""
        try:
            # 创建一个对话框来获取清理选项
            cleanup_dialog = Toplevel(self.master)
            cleanup_dialog.title("清理数据")
            cleanup_dialog.geometry("400x300")
            cleanup_dialog.transient(self.master)
            cleanup_dialog.grab_set()
            
            # 日期范围选择
            date_frame = Frame(cleanup_dialog)
            date_frame.pack(fill="x", padx=10, pady=5)
            
            Label(date_frame, text="开始日期:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            Label(date_frame, text="结束日期:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            
            # 导入日期选择器
            try:
                from tkcalendar import DateEntry
                # 创建日期选择器
                today = datetime.now().date()
                one_month_ago = today - timedelta(days=30)
                
                start_date = DateEntry(date_frame, width=12, 
                                    background='darkblue', foreground='white', 
                                    date_pattern='yyyy-mm-dd',
                                    year=one_month_ago.year, month=one_month_ago.month, day=one_month_ago.day)
                start_date.grid(row=0, column=1, padx=5, pady=5)
                
                end_date = DateEntry(date_frame, width=12, 
                                    background='darkblue', foreground='white', 
                                    date_pattern='yyyy-mm-dd')
                end_date.grid(row=1, column=1, padx=5, pady=5)
                
            except ImportError:
                self.log("警告: 未安装tkcalendar模块，使用文本输入框")
                start_date_var = StringVar(value=one_month_ago.strftime("%Y-%m-%d"))
                end_date_var = StringVar(value=today.strftime("%Y-%m-%d"))
                
                start_date = Entry(date_frame, textvariable=start_date_var, width=12)
                start_date.grid(row=0, column=1, padx=5, pady=5)
                
                end_date = Entry(date_frame, textvariable=end_date_var, width=12)
                end_date.grid(row=1, column=1, padx=5, pady=5)
            
            # 数据类型选择
            type_frame = Frame(cleanup_dialog)
            type_frame.pack(fill="x", padx=10, pady=5)
            
            Label(type_frame, text="选择要清理的数据类型:").pack(anchor="w", padx=5, pady=5)
            
            clean_processed_var = IntVar(value=1)
            clean_runs_var = IntVar(value=1)
            clean_db_var = IntVar(value=0)
            
            Checkbutton(type_frame, text="已处理的图像", variable=clean_processed_var).pack(anchor="w", padx=20, pady=2)
            Checkbutton(type_frame, text="训练和检测结果", variable=clean_runs_var).pack(anchor="w", padx=20, pady=2)
            Checkbutton(type_frame, text="数据库记录", variable=clean_db_var).pack(anchor="w", padx=20, pady=2)
            
            # 按钮区域
            button_frame = Frame(cleanup_dialog)
            button_frame.pack(fill="x", padx=10, pady=10)
            
            def do_cleanup():
                # 获取日期
                try:
                    if hasattr(start_date, 'get_date'):
                        # DateEntry对象
                        start = start_date.get_date()
                        end = end_date.get_date()
                        # 转换为datetime对象
                        start_dt = datetime.combine(start, datetime.min.time())
                        end_dt = datetime.combine(end, datetime.max.time())
                    else:
                        # 文本输入框
                        start_str = start_date.get()
                        end_str = end_date.get()
                        # 解析日期字符串
                        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
                        end_dt = datetime.combine(end_dt.date(), datetime.max.time())
                    
                    # 获取清理选项
                    clean_processed = bool(clean_processed_var.get())
                    clean_runs = bool(clean_runs_var.get())
                    clean_db = bool(clean_db_var.get())
                    
                    # 更新状态
                    self.status_var.set(f"正在清理 {start_dt.strftime('%Y-%m-%d')} 到 {end_dt.strftime('%Y-%m-%d')} 的数据...")
                    self.master.update()
                    
                    # 调用agent的清理方法
                    removed = self.agent.cleanup_old_data(
                        start_date=start_dt,
                        end_date=end_dt,
                        clean_processed=clean_processed,
                        clean_runs=clean_runs,
                        clean_db=clean_db
                    )
                    
                    # 更新状态
                    self.status_var.set(f"清理完成，删除了 {removed} 个文件/记录")
                    
                    # 显示成功消息
                    messagebox.showinfo("清理完成", f"成功删除了 {removed} 个文件/记录")
                    
                    # 关闭对话框
                    cleanup_dialog.destroy()
                    
                except Exception as e:
                    self.log(f"清理数据失败: {e}")
                    messagebox.showerror("错误", f"清理数据失败: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # 确认和取消按钮
            Button(button_frame, text="确认清理", command=do_cleanup, bg="#4CAF50", fg="white").pack(side="right", padx=5)
            Button(button_frame, text="取消", command=cleanup_dialog.destroy).pack(side="right", padx=5)
            
        except Exception as e:
            self.log(f"打开清理对话框失败: {e}")
            messagebox.showerror("错误", f"打开清理对话框失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())



    def _get_data_type_name(self, data_type):
        """获取数据类型的中文名称"""
        type_names = {
            "database": "数据库记录",
            "processed_images": "已处理图像",
            "result_images": "检测结果图像",
            "original_images": "原始图像备份",
            "logs": "日志文件"
        }
        return type_names.get(data_type, data_type)





    
    def _perform_cleanup(self, start_date, end_date):
        """执行数据清理操作"""
        try:
            self.log(f"开始清理 {start_date} 到 {end_date} 之间的数据...")
            
            # 1. 清理数据库记录
            if self.db_enabled:
                import sqlite3
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 查询要删除的记录ID
                cursor.execute('''
                SELECT id, processed_path, annotated_image FROM detection_records
                WHERE timestamp BETWEEN ? AND ?
                ''', (f"{start_date} 00:00:00", f"{end_date} 23:59:59"))
                
                records = cursor.fetchall()
                if not records:
                    self.log("未找到符合条件的数据库记录")
                    conn.close()
                    messagebox.showinfo("清理完成", "未找到符合条件的数据库记录")
                    return
                
                # 收集要删除的文件路径
                files_to_delete = []
                for record in records:
                    record_id, processed_path, annotated_image = record
                    if processed_path and os.path.exists(processed_path):
                        files_to_delete.append(processed_path)
                    if annotated_image and os.path.exists(annotated_image):
                        files_to_delete.append(annotated_image)
                
                # 删除缺陷详情记录
                record_ids = [r[0] for r in records]
                placeholders = ','.join(['?'] * len(record_ids))
                cursor.execute(f'''
                DELETE FROM defect_details
                WHERE detection_id IN ({placeholders})
                ''', record_ids)
                
                # 删除检测记录
                cursor.execute(f'''
                DELETE FROM detection_records
                WHERE id IN ({placeholders})
                ''', record_ids)
                
                # 检查是否所有数据都被删除了
                cursor.execute("SELECT COUNT(*) FROM detection_records")
                remaining_count = cursor.fetchone()[0]
                
                # 如果所有数据都被删除，重置ID计数器
                if remaining_count == 0:
                    self.log("数据库已清空，重置ID计数器")
                    cursor.execute("DELETE FROM sqlite_sequence WHERE name='detection_records'")
                    cursor.execute("DELETE FROM sqlite_sequence WHERE name='defect_details'")
                else:
                    # 如果只删除了部分数据，重新整理ID
                    self.log("重新整理数据库记录ID，使编号连续")
                    
                    # 创建临时表存储detection_records
                    cursor.execute("CREATE TABLE temp_detection_records AS SELECT * FROM detection_records")
                    
                    # 创建临时表存储defect_details
                    cursor.execute("CREATE TABLE temp_defect_details AS SELECT * FROM defect_details")
                    
                    # 删除原表
                    cursor.execute("DROP TABLE detection_records")
                    cursor.execute("DROP TABLE defect_details")
                    
                    # 重新创建原表结构
                    cursor.execute('''
                    CREATE TABLE detection_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        image_name TEXT,
                        detected_count INTEGER,
                        reviewed INTEGER,
                        correct INTEGER,
                        false_positive INTEGER,
                        missed_defect INTEGER,
                        annotated_image TEXT,
                        processed_path TEXT,
                        year INTEGER,
                        month INTEGER,
                        day INTEGER
                    )
                    ''')
                    
                    cursor.execute('''
                    CREATE TABLE defect_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        detection_id INTEGER,
                        class_name TEXT,
                        confidence REAL,
                        x1 REAL,
                        y1 REAL,
                        x2 REAL,
                        y2 REAL,
                        FOREIGN KEY (detection_id) REFERENCES detection_records (id)
                    )
                    ''')
                    
                    # 从临时表复制数据回原表，ID会自动重新分配
                    cursor.execute('''
                    INSERT INTO detection_records 
                    (timestamp, image_name, detected_count, reviewed, correct, false_positive, 
                     missed_defect, annotated_image, processed_path, year, month, day)
                    SELECT timestamp, image_name, detected_count, reviewed, correct, false_positive, 
                           missed_defect, annotated_image, processed_path, year, month, day
                    FROM temp_detection_records
                    ''')
                    
                    # 获取新旧ID的映射关系
                    cursor.execute('''
                    SELECT temp.id, new.id FROM temp_detection_records temp
                    JOIN detection_records new ON 
                    temp.timestamp = new.timestamp AND 
                    temp.image_name = new.image_name
                    ''')
                    
                    id_mapping = {old_id: new_id for old_id, new_id in cursor.fetchall()}
                    
                    # 使用新的detection_id更新缺陷详情
                    for old_id, new_id in id_mapping.items():
                        cursor.execute('''
                        INSERT INTO defect_details 
                        (detection_id, class_name, confidence, x1, y1, x2, y2)
                        SELECT ?, class_name, confidence, x1, y1, x2, y2
                        FROM temp_defect_details
                        WHERE detection_id = ?
                        ''', (new_id, old_id))
                    
                    # 删除临时表
                    cursor.execute("DROP TABLE temp_detection_records")
                    cursor.execute("DROP TABLE temp_defect_details")
                    
                    self.log(f"数据库ID已重新整理，现在ID从1到{remaining_count}连续")
                
                conn.commit()
                conn.close()
                
                deleted_count = len(record_ids)
                self.log(f"已从数据库中删除 {deleted_count} 条记录")
                
                # 2. 删除相关文件
                file_delete_count = 0
                for file_path in files_to_delete:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            file_delete_count += 1
                    except Exception as e:
                        self.log(f"删除文件失败 {file_path}: {e}")
                
                self.log(f"已删除 {file_delete_count} 个相关文件")
                
                # 3. 清理CSV文件中的记录（如果有）
                # ... 保持原有CSV处理逻辑不变 ...
                
                # 更新UI
                self.update_image_list()
                self.update_stats_plot()
                self.update_trend_plot()
                
                # 显示结果
                messagebox.showinfo("清理完成", 
                                   f"已清理 {start_date} 到 {end_date} 之间的数据:\n"
                                   f"- 删除了 {deleted_count} 条数据库记录\n"
                                   f"- 删除了 {file_delete_count} 个相关文件\n"
                                   f"- 数据库ID已重新整理，保持连续")
            else:
                self.log("数据库未启用，无法清理数据")
                messagebox.showwarning("警告", "数据库未启用，无法清理数据")
                
        except Exception as e:
            self.log(f"清理数据失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("错误", f"清理数据失败: {e}")


    def show_trend_analysis(self):
        """显示趋势分析报告"""
        try:
            # 调用generate_trend_analysis_report生成HTML报告
            report_path = self.generate_trend_analysis_report()
            
            if report_path:
                # 打开生成的HTML报告
                import webbrowser
                webbrowser.open(f"file://{report_path}")
                self.log(f"已生成趋势分析报告: {report_path}")
            else:
                messagebox.showerror("错误", "生成趋势分析报告失败")
                
        except Exception as e:
            self.log(f"显示趋势分析报告失败: {e}")
            import traceback
            self.log(traceback.format_exc())            





    def generate_trend_analysis_report(self):
        """生成图文并茂的趋势分析HTML报告"""
        try:
            if not self.matplotlib_available:
                self.log("未安装matplotlib，无法生成图表报告")
                return None
                
            # 将数据量要求从5条降低到1条，这样即使只有少量数据也能生成报告
            if not hasattr(self.agent, 'detections_history') or len(self.agent.detections_history) < 1:
                self.log("历史数据不足，无法生成有意义的趋势分析")
                return None
            
            # 生成报告时间戳 - 修改为安全的文件名格式
            timestamp = datetime.now()
            safe_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")  # 使用下划线代替冒号
            
            # 确保报告目录存在
            report_dir = "D:\\030923\\reports"
            os.makedirs(report_dir, exist_ok=True)
            
            # 使用安全的文件名格式
            trend_chart_path = os.path.join(report_dir, f"trend_chart_{safe_timestamp}.png")
            type_chart_path = os.path.join(report_dir, f"type_chart_{safe_timestamp}.png")
            accuracy_chart_path = os.path.join(report_dir, f"accuracy_chart_{safe_timestamp}.png")
            
            # 生成HTML报告路径
            report_path = os.path.join(report_dir, f"trend_report_{safe_timestamp}.html")
            # 准备数据
            history = self.agent.detections_history
            
            # 1. 按日期统计缺陷数量
            date_stats = {}
            for record in history:
                timestamp = record.get("timestamp", "")
                try:
                    # 修改：增加对ISO格式时间戳的支持
                    if 'T' in timestamp:
                        # ISO格式时间戳 (如 2025-03-21T15:15:49.249317)
                        date_obj = datetime.fromisoformat(timestamp)
                    else:
                        # 标准格式时间戳 (如 2025-03-21 15:15:49)
                        date_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    date_key = date_obj.strftime("%Y-%m-%d")
                    
                    if date_key not in date_stats:
                        date_stats[date_key] = {
                            "total": 0,
                            "defect_count": 0,
                            "reviewed": 0,
                            "correct": 0,
                            "false_positive": 0,
                            "missed_defect": 0
                        }
                    
                    date_stats[date_key]["total"] += 1
                    date_stats[date_key]["defect_count"] += record.get("detected_count", 0)
                    
                    if record.get("reviewed", False):
                        date_stats[date_key]["reviewed"] += 1
                    if record.get("correct", False):
                        date_stats[date_key]["correct"] += 1
                    if record.get("false_positive", False):
                        date_stats[date_key]["false_positive"] += 1
                    if record.get("missed_defect", False):
                        date_stats[date_key]["missed_defect"] += 1
                        
                except Exception as e:
                    self.log(f"处理记录时间戳出错: {e}")
                    continue
            
            # 2. 统计缺陷类型分布
            defect_types = {}
            for record in history:
                for defect in record.get("defects", []):
                    defect_class = defect.get("class", "未知")
                    defect_types[defect_class] = defect_types.get(defect_class, 0) + 1
            
            # 3. 计算总体统计数据
            total_images = len(history)
            reviewed_images = sum(1 for r in history if r.get("reviewed", False))
            correct_detections = sum(1 for r in history if r.get("correct", False))
            false_positive_images = sum(1 for r in history if r.get("false_positive", False))
            missed_defect_images = sum(1 for r in history if r.get("missed_defect", False))
            
            success_rate = (correct_detections / reviewed_images * 100) if reviewed_images > 0 else 0
            false_rate = (false_positive_images / reviewed_images * 100) if reviewed_images > 0 else 0
            missed_rate = (missed_defect_images / reviewed_images * 100) if reviewed_images > 0 else 0
            
            # 4. 生成图表 - 修改文件名格式，避免使用冒号
            # 4.1 缺陷趋势图

            self._generate_trend_chart(date_stats, trend_chart_path)
            
            # 4.2 缺陷类型分布图

            self._generate_type_distribution_chart(defect_types, type_chart_path)
            
            # 4.3 检测准确率图

            self._generate_accuracy_chart(date_stats, accuracy_chart_path)
            # 5. 使用LLM生成报告文字部分
            llm_report = self._generate_llm_report(history, date_stats, defect_types)
            # 6. 生成HTML报告
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>PCB缺陷检测系统 - 趋势分析报告</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    .header {{
                        text-align: center;
                        margin-bottom: 30px;
                        padding-bottom: 20px;
                        border-bottom: 1px solid #eee;
                    }}
                    .summary-box {{
                        background-color: #f8f9fa;
                        border-radius: 5px;
                        padding: 20px;
                        margin-bottom: 30px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .summary-stats {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }}
                    .stat-item {{
                        flex: 1;
                        min-width: 200px;
                        margin: 10px;
                        padding: 15px;
                        background: white;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .stat-value {{
                        font-size: 24px;
                        font-weight: bold;
                        color: #3498db;
                    }}
                    .chart-container {{
                        margin: 30px 0;
                        text-align: center;
                    }}
                    .chart-container img {{
                        max-width: 100%;
                        height: auto;
                        border-radius: 5px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 12px 15px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    .footer {{
                        margin-top: 50px;
                        text-align: center;
                        font-size: 14px;
                        color: #7f8c8d;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>PCB缺陷检测系统 - 趋势分析报告</h1>
                    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary-box">
                    <h2>总体统计</h2>
                    <div class="summary-stats">
                        <div class="stat-item">
                            <h3>检测总数</h3>
                            <div class="stat-value">{total_images}</div>
                        </div>
                        <div class="stat-item">
                            <h3>已复检数</h3>
                            <div class="stat-value">{reviewed_images}</div>
                        </div>
                        <div class="stat-item">
                            <h3>检测正确率</h3>
                            <div class="stat-value">{success_rate:.1f}%</div>
                        </div>
                        <div class="stat-item">
                            <h3>误检率</h3>
                            <div class="stat-value">{false_rate:.1f}%</div>
                        </div>
                        <div class="stat-item">
                            <h3>漏检率</h3>
                            <div class="stat-value">{missed_rate:.1f}%</div>
                        </div>
                    </div>
                </div>
                
                <h2>缺陷检测趋势</h2>
                <div class="chart-container">
                    <img src="{os.path.basename(trend_chart_path)}" alt="缺陷检测趋势图">
                </div>
                
                <h2>缺陷类型分布</h2>
                <div class="chart-container">
                    <img src="{os.path.basename(type_chart_path)}" alt="缺陷类型分布图">
                </div>
                
                <h2>检测准确率趋势</h2>
                <div class="chart-container">
                    <img src="{os.path.basename(accuracy_chart_path)}" alt="检测准确率趋势图">
                </div>
                
                <h2>每日检测统计</h2>
                <table>
                    <tr>
                        <th>日期</th>
                        <th>检测总数</th>
                        <th>缺陷总数</th>
                        <th>已复检数</th>
                        <th>正确数</th>
                        <th>误检数</th>
                        <th>漏检数</th>
                        <th>正确率</th>
                    </tr>
            """
            
            # 添加每日统计数据行
            for date, stats in sorted(date_stats.items(), reverse=True):
                reviewed = stats["reviewed"]
                correct_rate = (stats["correct"] / reviewed * 100) if reviewed > 0 else 0
                
                html_content += f"""
                    <tr>
                        <td>{date}</td>
                        <td>{stats["total"]}</td>
                        <td>{stats["defect_count"]}</td>
                        <td>{stats["reviewed"]}</td>
                        <td>{stats["correct"]}</td>
                        <td>{stats["false_positive"]}</td>
                        <td>{stats["missed_defect"]}</td>
                        <td>{correct_rate:.1f}%</td>
                    </tr>
                """
            
            # 完成HTML
            html_content += """
                </table>
                
                <div class="footer">
                    <p>PCB缺陷检测系统 - 自动生成报告</p>
                </div>
            </body>
            </html>
            """
            
            # 写入HTML文件
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return report_path
            
        except Exception as e:
            self.log(f"生成趋势分析报告失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return None
    

    def _generate_llm_report(self, history, date_stats, defect_types):
        """使用LLM生成报告文字部分，如果LLM不可用则使用本地报告生成"""
        # 首先尝试使用本地报告生成功能
        try:
            # 尝试导入OpenAI模块，检查是否可用
            try:
                import os
                from openai import OpenAI
                openai_available = True
            except ImportError as e:
                self.log(f"OpenAI模块导入失败: {e}")
                return self._generate_local_report(history, date_stats, defect_types)
            
            # 检查API密钥是否设置
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                self.log("未设置API密钥，将使用本地报告生成")
                return self._generate_local_report(history, date_stats, defect_types)
            
            # 准备LLM的输入内容
            # 1. 收集日志内容
            log_content = self.get_log_content()
            
            # 2. 收集检测统计信息
            total_images = len(history)
            reviewed_images = sum(1 for r in history if r.get("reviewed", False))
            correct_detections = sum(1 for r in history if r.get("correct", False))
            false_positive_images = sum(1 for r in history if r.get("false_positive", False))
            missed_defect_images = sum(1 for r in history if r.get("missed_defect", False))
            
            success_rate = (correct_detections / reviewed_images * 100) if reviewed_images > 0 else 0
            false_rate = (false_positive_images / reviewed_images * 100) if reviewed_images > 0 else 0
            missed_rate = (missed_defect_images / reviewed_images * 100) if reviewed_images > 0 else 0
            
            # 3. 收集缺陷类型信息
            defect_type_info = "\n".join([f"- {defect_type}: {count}个" for defect_type, count in defect_types.items()])
            
            # 4. 收集每日统计数据
            daily_stats = []
            for date, stats in sorted(date_stats.items()):
                reviewed = stats["reviewed"]
                correct_rate = (stats["correct"] / reviewed * 100) if reviewed > 0 else 0
                daily_stats.append(f"日期: {date}, 检测总数: {stats['total']}, 缺陷总数: {stats['defect_count']}, " +
                                  f"已复检数: {stats['reviewed']}, 正确数: {stats['correct']}, " +
                                  f"误检数: {stats['false_positive']}, 漏检数: {stats['missed_defect']}, " +
                                  f"正确率: {correct_rate:.1f}%")
            
            daily_stats_text = "\n".join(daily_stats)
            
            # 构建提示词
            prompt = f"""
你是一个PCB缺陷检测系统的分析专家。请根据以下数据生成一份详细的分析报告。报告应包括对检测性能的评估、趋势分析、问题诊断和改进建议。

## 检测统计数据
- 检测总数: {total_images}
- 已复检数: {reviewed_images}
- 正确检测数: {correct_detections}
- 误检数: {false_positive_images}
- 漏检数: {missed_defect_images}
- 检测正确率: {success_rate:.1f}%
- 误检率: {false_rate:.1f}%
- 漏检率: {missed_rate:.1f}%

## 缺陷类型分布
{defect_type_info}

## 每日检测统计
{daily_stats_text}

## 系统日志摘要
{log_content[:2000] if log_content else "无可用日志"}

请提供以下内容：
1. 检测性能分析：评估当前系统的检测准确性和可靠性
2. 趋势分析：识别检测结果的时间趋势和模式
3. 问题诊断：分析系统可能存在的问题和挑战
4. 改进建议：提出具体的改进措施和优化方向

请用专业但易于理解的语言撰写，避免过于技术性的术语，并确保报告内容具有实用价值。
"""
            
            try:
                # 调用LLM API
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                
                completion = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "你是一个PCB缺陷检测系统的分析专家，擅长分析检测数据并提供专业见解。"},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # 获取LLM回复
                llm_response = completion.choices[0].message.content
                
                # 记录到日志
                self.log("已生成AI分析报告")
                
                return llm_response
                
            except Exception as e:
                self.log(f"调用LLM生成报告失败: {e}")
                import traceback
                self.log(traceback.format_exc())
                return self._generate_local_report(history, date_stats, defect_types)
                
        except Exception as e:
            self.log(f"准备LLM报告数据失败: {e}")
            return self._generate_local_report(history, date_stats, defect_types)
    
    def _generate_local_report(self, history, date_stats, defect_types):
        """生成本地分析报告（不依赖外部API）"""
        self.log("使用本地报告生成器")
        
        # 计算基本统计数据
        total_images = len(history)
        reviewed_images = sum(1 for r in history if r.get("reviewed", False))
        correct_detections = sum(1 for r in history if r.get("correct", False))
        false_positive_images = sum(1 for r in history if r.get("false_positive", False))
        missed_defect_images = sum(1 for r in history if r.get("missed_defect", False))
        
        success_rate = (correct_detections / reviewed_images * 100) if reviewed_images > 0 else 0
        false_rate = (false_positive_images / reviewed_images * 100) if reviewed_images > 0 else 0
        missed_rate = (missed_defect_images / reviewed_images * 100) if reviewed_images > 0 else 0
        
        # 生成HTML报告
        report = f"""
        <h2>PCB缺陷检测系统分析报告</h2>
        
        <h3>1. 检测性能分析</h3>
        <p>系统当前的检测准确率为<strong>{success_rate:.1f}%</strong>，误检率为{false_rate:.1f}%，漏检率为{missed_rate:.1f}%。
        总共检测了{total_images}张图像，其中{reviewed_images}张已经过人工复检。</p>
        
        <p>基于当前数据，系统的检测性能{'良好' if success_rate > 80 else '一般' if success_rate > 60 else '需要改进'}。
        {'误检率较高，建议优化模型减少误报。' if false_rate > 20 else ''}
        {'漏检率较高，建议增加训练样本提高检出率。' if missed_rate > 20 else ''}</p>
        
        <h3>2. 缺陷类型分布</h3>
        <p>系统检测到的主要缺陷类型包括：</p>
        <ul>
        """
        
        # 添加缺陷类型信息
        for defect_type, count in sorted(defect_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / sum(defect_types.values()) * 100) if defect_types else 0
            report += f"<li>{defect_type}: {count}个 ({percentage:.1f}%)</li>\n"
        
        report += """
        </ul>
        
        <h3>3. 时间趋势分析</h3>
        <p>根据历史数据分析，系统检测性能呈现以下趋势：</p>
        <ul>
        """
        
        # 添加时间趋势信息
        dates = sorted(date_stats.keys())
        if len(dates) >= 2:
            first_date = dates[0]
            last_date = dates[-1]
            first_stats = date_stats[first_date]
            last_stats = date_stats[last_date]
            
            # 计算第一天和最后一天的正确率
            first_correct_rate = (first_stats["correct"] / first_stats["reviewed"] * 100) if first_stats["reviewed"] > 0 else 0
            last_correct_rate = (last_stats["correct"] / last_stats["reviewed"] * 100) if last_stats["reviewed"] > 0 else 0
            
            # 判断趋势
            if last_correct_rate > first_correct_rate:
                report += f"<li>检测准确率呈上升趋势，从{first_date}的{first_correct_rate:.1f}%提高到{last_date}的{last_correct_rate:.1f}%</li>\n"
            elif last_correct_rate < first_correct_rate:
                report += f"<li>检测准确率呈下降趋势，从{first_date}的{first_correct_rate:.1f}%降低到{last_date}的{last_correct_rate:.1f}%</li>\n"
            else:
                report += f"<li>检测准确率保持稳定，维持在{last_correct_rate:.1f}%左右</li>\n"
                
            # 分析缺陷数量趋势
            first_defects = first_stats["defect_count"]
            last_defects = last_stats["defect_count"]
            if last_defects > first_defects:
                report += f"<li>缺陷检出数量呈上升趋势，从{first_date}的{first_defects}个增加到{last_date}的{last_defects}个</li>\n"
            elif last_defects < first_defects:
                report += f"<li>缺陷检出数量呈下降趋势，从{first_date}的{first_defects}个减少到{last_date}的{last_defects}个</li>\n"
            else:
                report += f"<li>缺陷检出数量保持稳定，维持在{last_defects}个左右</li>\n"
        else:
            report += "<li>数据量不足，无法分析时间趋势</li>\n"
        
        report += """
        </ul>
        
        <h3>4. 问题诊断</h3>
        <ul>
        """
        
        # 添加问题诊断
        if success_rate < 70:
            report += "<li>检测准确率低于70%，模型性能需要提升</li>\n"
        if false_rate > 20:
            report += "<li>误检率超过20%，模型可能对某些特征过度敏感</li>\n"
        if missed_rate > 20:
            report += "<li>漏检率超过20%，模型可能对某些缺陷类型不够敏感</li>\n"
        if reviewed_images / total_images < 0.3:
            report += "<li>人工复检比例低于30%，可能影响模型训练质量</li>\n"
        
        report += """
        </ul>
        
        <h3>5. 改进建议</h3>
        <ul>
        <li>定期更新训练数据，确保模型适应新的缺陷类型</li>
        <li>增加人工复检比例，提高模型训练质量</li>
        <li>对误检率较高的缺陷类型进行重点优化</li>
        <li>考虑增加数据增强方法，提高模型泛化能力</li>
        """
        
        # 添加针对性建议
        if success_rate < 70:
            report += "<li>考虑重新训练模型，增加训练轮次和样本数量</li>\n"
        if false_rate > 20:
            report += "<li>调整模型置信度阈值，减少误报</li>\n"
        if missed_rate > 20:
            report += "<li>增加难例样本，提高模型对难以检测缺陷的敏感度</li>\n"
        
        report += """
        </ul>
        
        <p><em>注：此报告由系统本地生成，未使用外部AI服务</em></p>
        """
        
        return report





    def _generate_trend_chart(self, date_stats, output_path):
        """生成缺陷趋势图表"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import matplotlib.dates as mdates
            
            # 准备数据
            dates = sorted(date_stats.keys())
            
            # 将日期字符串转换为datetime对象，以便matplotlib正确处理
            date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
            
            defect_counts = [date_stats[date]["defect_count"] for date in dates]
            total_counts = [date_stats[date]["total"] for date in dates]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 使用日期对象而不是字符串
            plt.plot(date_objects, defect_counts, 'o-', color='#3498db', linewidth=2, label='缺陷数量')
            plt.plot(date_objects, total_counts, 's-', color='#2ecc71', linewidth=2, label='检测总数')
            
            # 设置日期格式化器
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 旋转日期标签以避免重叠
            plt.gcf().autofmt_xdate()
            
            plt.title('PCB缺陷检测趋势', fontsize=14)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('数量', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(output_path, dpi=100)
            plt.close()
            
            return True
            
        except Exception as e:
            self.log(f"生成趋势图表失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def _generate_type_distribution_chart(self, defect_types, output_path):
        """生成缺陷类型分布图表"""
        try:
            import matplotlib.pyplot as plt
            
            # 如果没有缺陷数据，创建一个简单的空图表
            if not defect_types:
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, '暂无缺陷数据', horizontalalignment='center',
                        verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
                plt.savefig(output_path, dpi=100)
                plt.close()
                return True
            
            # 准备数据
            types = list(defect_types.keys())
            counts = list(defect_types.values())
            
            # 按数量排序
            sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
            types = [types[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 使用数字索引而不是字符串作为x轴
            indices = np.arange(len(types))
            bars = plt.bar(indices, counts, color='#3498db')
            
            # 添加数据标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            # 设置x轴标签
            plt.xticks(indices, types, rotation=45, ha='right')
            
            plt.title('PCB缺陷类型分布', fontsize=14)
            plt.xlabel('缺陷类型', fontsize=12)
            plt.ylabel('数量', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(output_path, dpi=100)
            plt.close()
            
            return True
            
        except Exception as e:
            self.log(f"生成缺陷类型分布图失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False
    
    def _generate_accuracy_chart(self, date_stats, output_path):
        """生成检测准确率趋势图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # 准备数据
            dates = sorted(date_stats.keys())
            
            # 将日期字符串转换为datetime对象
            date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in dates]
            
            # 计算各项指标
            correct_rates = []
            false_rates = []
            missed_rates = []
            
            for date in dates:
                stats = date_stats[date]
                reviewed = stats["reviewed"]
                
                if reviewed > 0:
                    correct_rate = (stats["correct"] / reviewed) * 100
                    false_rate = (stats["false_positive"] / reviewed) * 100
                    missed_rate = (stats["missed_defect"] / reviewed) * 100
                else:
                    correct_rate = false_rate = missed_rate = 0
                
                correct_rates.append(correct_rate)
                false_rates.append(false_rate)
                missed_rates.append(missed_rate)
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 使用日期对象而不是字符串
            plt.plot(date_objects, correct_rates, 'o-', color='#2ecc71', linewidth=2, label='正确率')
            plt.plot(date_objects, false_rates, 's-', color='#e74c3c', linewidth=2, label='误检率')
            plt.plot(date_objects, missed_rates, '^-', color='#f39c12', linewidth=2, label='漏检率')
            
            # 设置日期格式化器
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 旋转日期标签以避免重叠
            plt.gcf().autofmt_xdate()
            
            plt.title('PCB缺陷检测准确率趋势', fontsize=14)
            plt.xlabel('日期', fontsize=12)
            plt.ylabel('百分比 (%)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 设置y轴范围
            plt.ylim(0, 105)
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(output_path, dpi=100)
            plt.close()
            
            return True
            
        except Exception as e:
            self.log(f"生成准确率趋势图失败: {e}")
            import traceback
            self.log(traceback.format_exc())
            return False


    def generate_comprehensive_report(self):
        """生成全面的趋势分析报告"""
        history = self.agent.detections_history
        if not history:
            return "没有检测历史数据，无法生成报告。"
            
        # 报告标题和基本信息
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_lines = [
            f"# PCB缺陷检测趋势分析报告",
            f"生成时间: {now}",
            f"历史检测记录总数: {len(history)}",
            "",
            "## 1. 检测概况",
            f"首次检测时间: {history[0].get('timestamp', '未知')}",
            f"最近检测时间: {history[-1].get('timestamp', '未知')}",
            ""
        ]
        
        # 统计总体缺陷数量
        total_defects = sum(record.get("detected_count", 0) for record in history)
        total_images = len(history)
        avg_defects = total_defects / total_images if total_images > 0 else 0
        
        report_lines.extend([
            f"检测图像总数: {total_images}",
            f"检测到的缺陷总数: {total_defects}",
            f"平均每张图像缺陷数: {avg_defects:.2f}",
            ""
        ])
        
        # 按缺陷类型统计
        defect_types = {}
        for record in history:
            stats = record.get("defect_stats", {})
            for defect_type, count in stats.items():
                defect_types[defect_type] = defect_types.get(defect_type, 0) + count
        
        if defect_types:
            report_lines.append("## 2. 缺陷类型分布")
            for defect_type, count in sorted(defect_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_defects * 100) if total_defects > 0 else 0
                report_lines.append(f"{defect_type}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 时间趋势分析
        report_lines.append("## 3. 时间趋势分析")
        
        # 按天统计缺陷数量
        daily_stats = {}
        for record in history:
            timestamp = record.get("timestamp", "")
            try:
                date = timestamp.split(" ")[0]  # 提取日期部分
                defects = record.get("detected_count", 0)
                daily_stats[date] = daily_stats.get(date, 0) + defects
            except:
                continue
        
        if daily_stats:
            report_lines.append("每日缺陷检测数量:")
            for date, count in sorted(daily_stats.items()):
                report_lines.append(f"{date}: {count}")
            report_lines.append("")
        
        # 复检情况分析
        reviewed_count = sum(1 for record in history if record.get("reviewed", False))
        report_lines.extend([
            "## 4. 复检情况",
            f"已复检图像数: {reviewed_count}",
            f"复检率: {(reviewed_count/total_images*100):.1f}%" if total_images > 0 else "复检率: 0%",
            ""
        ])
        
        # 模型性能分析
        report_lines.append("## 5. 模型性能")
        if hasattr(self.agent, 'best_map') and self.agent.best_map is not None:
            report_lines.append(f"当前最佳模型mAP: {self.agent.best_map:.4f}")
        if hasattr(self.agent, 'last_map') and self.agent.last_map is not None:
            report_lines.append(f"最近训练模型mAP: {self.agent.last_map:.4f}")
        
        return "\n".join(report_lines)
        
    def export_report(self, report_content):
        """导出报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"d:/030923/agent/reports/trend_report_{timestamp}.txt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.log(f"报告已导出到: {filename}")
            messagebox.showinfo("导出成功", f"报告已导出到:\n{filename}")
        except Exception as e:
            self.log(f"导出报告失败: {e}")
            messagebox.showerror("导出失败", f"导出报告时出错:\n{str(e)}")





    def periodic_detection(self):
        """定期自动检测新图像"""
        # 检查是否有新图像
        image_files = [f for f in os.listdir(self.agent.new_data_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        if image_files and not self.agent.detecting and not self.agent.training:
            self.log("定时检测：发现新图像，开始自动检测...")
            self.run_detection()
        
        # 重新安排下一次检测
        self.master.after(self.auto_detect_interval, self.periodic_detection) # 安排下一次

# Flask API 设置
app = Flask(__name__)
global_agent = None  # 全局的 Agent 实例，用于在 API 中调用

@app.route("/detect", methods=["POST"])
def api_detect():
    """API 端点：上传单张PCB图像并返回检测结果"""
    if global_agent is None:
        return jsonify({"error": "Agent not initialized"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    # 保存上传文件到临时目录
    filename = secure_filename(file.filename)
    temp_dir = "temp_upload"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{int(time.time())}_{filename}")
    file.save(file_path)
    # 使用当前模型进行检测
    try:
        results = global_agent.model.predict(source=file_path)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500
    detections = []
    if results:
        res = results[0]  # 只处理单张图像结果
        try:
            res_cpu = res.cpu()
        except:
            res_cpu = res
        boxes = res_cpu.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i]
                conf = float(boxes.conf[i]) if boxes.conf is not None else None
                cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
                class_name = global_agent.class_names[cls_id] if cls_id < len(global_agent.class_names) else str(cls_id)
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })
    # 清理临时文件
    try:
        os.remove(file_path)
    except:
        pass
    return jsonify({"detections": detections})
@app.route("/train", methods=["POST"])
def api_train():
    """API 端点：启动模型训练（如果不在训练中）"""
    if global_agent is None:
        return jsonify({"error": "Agent not initialized"}), 500
    if global_agent.training:
        return jsonify({"status": "training_already_running"}), 200
    # 可接受JSON中的超参数设定（例如 {"epochs": 100}）
    data = request.get_json(silent=True) or {}
    epochs = data.get("epochs")
    if epochs:
        global_agent.hyperparams["epochs"] = int(epochs)
    # 启动训练
    started = global_agent.start_training()
    if not started:
        return jsonify({"status": "training_already_running"}), 200
    return jsonify({"status": "training_started"})
@app.route("/status", methods=["GET"])
def api_status():
    """API 端点：获取当前 Agent 状态和简单统计"""
    if global_agent is None:
        return jsonify({"error": "Agent not initialized"}), 500
    status = {
        "training": global_agent.training,
        "total_images": global_agent.total_images,
        "reviewed_images": global_agent.reviewed_images,
        "correct_detections": global_agent.correct_detections,
        "false_positive_images": global_agent.false_positive_images,
        "missed_defect_images": global_agent.missed_defect_images,
        "best_model_path": global_agent.best_model_path,
        "best_map": global_agent.best_map,
        "last_model_path": global_agent.last_model_path,
        "last_map": global_agent.last_map
    }
    return jsonify(status)
def run_gui(agent):
    """运行 Tkinter GUI 界面"""
    root = Tk()
    YOLOAgentGUI(root, agent)
    root.mainloop()
def run_api(agent):
    """运行 Flask API 服务"""
    global global_agent
    global_agent = agent
    app.run(host="0.0.0.0", port=5000)
if __name__ == "__main__":
    import sys
    # 请根据实际情况提供默认模型权重和数据配置文件路径
    default_model = "D:/030923/Model/best.pt"         # 预训练模型权重文件路径
    data_config = "D:/030923/agent/yolo_agent/dataset.yaml"#路径（包含类名和数据集划分）
    agent = YOLOAgent(model_path=default_model, data_yaml=data_config)
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        print("Starting YOLOv9 PCB Agent in API mode...")
        run_api(agent)
    else:
        print("Starting YOLOv9 PCB Agent in GUI mode...")
        run_gui(agent)
