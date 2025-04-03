import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog  # 添加 filedialog
import cv2
import time  # 添加 time 模块

class StatisticsAnalyzer:
    """PCB缺陷检测结果统计分析工具"""
    
    def __init__(self):
        self.defect_data = None
        self.batch_data = None
        
    def load_single_result(self, defect_info):
        """加载单次检测的缺陷信息"""
        self.defect_data = defect_info
        
    def load_batch_results(self, csv_path):
        """加载批量检测的CSV结果文件"""
        try:
            self.batch_data = pd.read_csv(csv_path)
            return True
        except Exception as e:
            print(f"加载CSV文件失败: {str(e)}")
            return False
    
    def show_analysis_window(self, parent):
        """显示统计分析窗口"""
        if self.defect_data is None and self.batch_data is None:
            messagebox.showwarning("提示", "没有可分析的数据")
            return
            
        # 创建分析窗口
        analysis_window = tk.Toplevel(parent)
        analysis_window.title("缺陷检测统计分析")
        analysis_window.geometry("900x700")
        
        # 创建选项卡
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 缺陷类型分布选项卡
        type_tab = ttk.Frame(notebook)
        notebook.add(type_tab, text="缺陷类型分布")
        self._create_type_distribution_tab(type_tab)
        
        # 置信度分析选项卡
        conf_tab = ttk.Frame(notebook)
        notebook.add(conf_tab, text="置信度分析")
        self._create_confidence_analysis_tab(conf_tab)
        
        # 空间分布选项卡
        spatial_tab = ttk.Frame(notebook)
        notebook.add(spatial_tab, text="空间分布")
        self._create_spatial_distribution_tab(spatial_tab)
        
        # 如果有批量数据，添加批量分析选项卡
        if self.batch_data is not None:
            batch_tab = ttk.Frame(notebook)
            notebook.add(batch_tab, text="批量分析")
            self._create_batch_analysis_tab(batch_tab)
        
        # 导出报告按钮
        btn_frame = ttk.Frame(analysis_window)
        btn_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(btn_frame, text="导出分析报告", 
                  command=lambda: self.export_analysis_report()).pack(side="right")
    
    def _create_type_distribution_tab(self, parent):
        """创建缺陷类型分布选项卡"""
        # 获取数据
        if self.defect_data is not None:
            data = self.defect_data
            types = [d['type'] for d in data]
        elif self.batch_data is not None:
            data = self.batch_data
            types = data['缺陷类型'].tolist()
        else:
            return
        
        # 计算各类型数量
        type_counts = {}
        for t in types:
            if t in type_counts:
                type_counts[t] += 1
            else:
                type_counts[t] = 1
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 饼图
        ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        ax1.set_title('缺陷类型分布')
        
        # 条形图
        ax2.bar(type_counts.keys(), type_counts.values())
        ax2.set_title('缺陷类型计数')
        ax2.set_ylabel('数量')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_confidence_analysis_tab(self, parent):
        """创建置信度分析选项卡"""
        # 获取数据
        if self.defect_data is not None:
            data = self.defect_data
            confidences = [d.get('confidence', 0) for d in data]
            types = [d['type'] for d in data]
        elif self.batch_data is not None:
            data = self.batch_data
            confidences = data['置信度'].tolist()
            types = data['缺陷类型'].tolist()
        else:
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # 直方图
        ax1.hist(confidences, bins=10)
        ax1.set_title('置信度分布')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('频率')
        
        # 按类型的箱线图
        type_conf = {}
        for t, c in zip(types, confidences):
            if t not in type_conf:
                type_conf[t] = []
            type_conf[t].append(c)
        
        box_data = [type_conf[t] for t in type_conf]
        ax2.boxplot(box_data, labels=type_conf.keys())
        ax2.set_title('各类型置信度分布')
        ax2.set_ylabel('置信度')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_spatial_distribution_tab(self, parent):
        """创建空间分布选项卡"""
        # 获取数据
        if self.defect_data is not None:
            data = self.defect_data
            x_coords = [d['x'] + d['w']/2 for d in data]
            y_coords = [d['y'] + d['h']/2 for d in data]
            types = [d['type'] for d in data]
        elif self.batch_data is not None:
            data = self.batch_data
            x_coords = [x + w/2 for x, w in zip(data['X'].tolist(), data['宽'].tolist())]
            y_coords = [y + h/2 for y, h in zip(data['Y'].tolist(), data['高'].tolist())]
            types = data['缺陷类型'].tolist()
        else:
            return
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 散点图，不同类型用不同颜色
        unique_types = list(set(types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        
        for i, t in enumerate(unique_types):
            indices = [j for j, x in enumerate(types) if x == t]
            ax.scatter([x_coords[j] for j in indices], 
                      [y_coords[j] for j in indices], 
                      color=colors[i], label=t)
        
        ax.set_title('缺陷空间分布')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_batch_analysis_tab(self, parent):
        """创建批量分析选项卡"""
        if self.batch_data is None:
            return
            
        # 按文件名分组统计
        file_stats = self.batch_data.groupby('文件名').agg({
            '缺陷类型': 'count',
            '置信度': 'mean'
        }).reset_index()
        file_stats.columns = ['文件名', '缺陷数量', '平均置信度']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # 每个文件的缺陷数量
        ax1.bar(file_stats['文件名'], file_stats['缺陷数量'])
        ax1.set_title('各文件缺陷数量')
        ax1.set_ylabel('缺陷数量')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 每个文件的平均置信度
        ax2.bar(file_stats['文件名'], file_stats['平均置信度'])
        ax2.set_title('各文件平均置信度')
        ax2.set_ylabel('平均置信度')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def export_analysis_report(self):
        """导出分析报告"""
        try:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                title="保存分析报告",
                defaultextension=".html",
                filetypes=[("HTML文件", "*.html"), ("PDF文件", "*.pdf"), ("所有文件", "*.*")]
            )
            
            if not file_path:
                return
                
            # 根据文件扩展名选择导出格式
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.html':
                self._export_html_report(file_path)
            elif ext == '.pdf':
                self._export_pdf_report(file_path)
            else:
                messagebox.showwarning("提示", "不支持的文件格式，将导出为HTML")
                self._export_html_report(file_path)
                
            messagebox.showinfo("成功", f"分析报告已导出到: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出报告失败: {str(e)}")
            
    def _export_html_report(self, file_path):
        """导出HTML格式的分析报告"""
        import base64
        from io import BytesIO
        
        # 创建HTML内容
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>PCB缺陷检测分析报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                .chart-container { margin: 20px 0; text-align: center; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>PCB缺陷检测分析报告</h1>
            <p>生成时间: {date}</p>
        """.format(date=time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # 添加缺陷类型分布
        html_content += "<h2>缺陷类型分布</h2>"
        
        # 生成饼图
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.defect_data is not None:
            data = self.defect_data
            types = [d['type'] for d in data]
        elif self.batch_data is not None:
            data = self.batch_data
            types = data['缺陷类型'].tolist()
        else:
            types = []
            
        # 计算各类型数量
        type_counts = {}
        for t in types:
            if t in type_counts:
                type_counts[t] += 1
            else:
                type_counts[t] = 1
        
        ax.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        ax.set_title('缺陷类型分布')
        
        # 将图表转换为base64编码
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        # 添加图表到HTML
        html_content += f'<div class="chart-container"><img src="data:image/png;base64,{img_str}" alt="缺陷类型分布"></div>'
        
        # 添加缺陷类型统计表格
        html_content += """
        <h2>缺陷类型统计</h2>
        <table>
            <tr>
                <th>缺陷类型</th>
                <th>数量</th>
                <th>百分比</th>
            </tr>
        """
        
        total = sum(type_counts.values())
        for t, count in type_counts.items():
            percentage = count / total * 100 if total > 0 else 0
            html_content += f"""
            <tr>
                <td>{t}</td>
                <td>{count}</td>
                <td>{percentage:.2f}%</td>
            </tr>
            """
            
        html_content += "</table>"
        
        # 添加置信度分析
        if self.defect_data is not None or self.batch_data is not None:
            html_content += "<h2>置信度分析</h2>"
            
            # 生成置信度直方图
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if self.defect_data is not None:
                confidences = [d.get('confidence', 0) for d in self.defect_data]
            else:
                confidences = self.batch_data['置信度'].tolist()
                
            ax.hist(confidences, bins=10)
            ax.set_title('置信度分布')
            ax.set_xlabel('置信度')
            ax.set_ylabel('频率')
            
            # 将图表转换为base64编码
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # 添加图表到HTML
            html_content += f'<div class="chart-container"><img src="data:image/png;base64,{img_str}" alt="置信度分布"></div>'
            
            # 添加置信度统计表格
            html_content += """
            <h2>置信度统计</h2>
            <table>
                <tr>
                    <th>统计指标</th>
                    <th>值</th>
                </tr>
            """
            
            # 计算统计指标
            mean_conf = np.mean(confidences) if confidences else 0
            median_conf = np.median(confidences) if confidences else 0
            min_conf = np.min(confidences) if confidences else 0
            max_conf = np.max(confidences) if confidences else 0
            std_conf = np.std(confidences) if confidences else 0
            
            html_content += f"""
            <tr><td>平均值</td><td>{mean_conf:.4f}</td></tr>
            <tr><td>中位数</td><td>{median_conf:.4f}</td></tr>
            <tr><td>最小值</td><td>{min_conf:.4f}</td></tr>
            <tr><td>最大值</td><td>{max_conf:.4f}</td></tr>
            <tr><td>标准差</td><td>{std_conf:.4f}</td></tr>
            """
            
            html_content += "</table>"
        
        # 添加空间分布分析
        if self.defect_data is not None or self.batch_data is not None:
            html_content += "<h2>缺陷空间分布</h2>"
            
            # 生成空间分布散点图
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if self.defect_data is not None:
                data = self.defect_data
                x_coords = [d['x'] + d['w']/2 for d in data]
                y_coords = [d['y'] + d['h']/2 for d in data]
                types = [d['type'] for d in data]
            else:
                data = self.batch_data
                x_coords = [x + w/2 for x, w in zip(data['X'].tolist(), data['宽'].tolist())]
                y_coords = [y + h/2 for y, h in zip(data['Y'].tolist(), data['高'].tolist())]
                types = data['缺陷类型'].tolist()
                
            # 散点图，不同类型用不同颜色
            unique_types = list(set(types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            
            for i, t in enumerate(unique_types):
                indices = [j for j, x in enumerate(types) if x == t]
                ax.scatter([x_coords[j] for j in indices], 
                          [y_coords[j] for j in indices], 
                          color=colors[i], label=t)
            
            ax.set_title('缺陷空间分布')
            ax.set_xlabel('X坐标')
            ax.set_ylabel('Y坐标')
            ax.legend()
            
            # 将图表转换为base64编码
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # 添加图表到HTML
            html_content += f'<div class="chart-container"><img src="data:image/png;base64,{img_str}" alt="缺陷空间分布"></div>'
        
        # 添加批量分析结果
        if self.batch_data is not None:
            html_content += "<h2>批量分析结果</h2>"
            
            # 按文件名分组统计
            file_stats = self.batch_data.groupby('文件名').agg({
                '缺陷类型': 'count',
                '置信度': 'mean'
            }).reset_index()
            file_stats.columns = ['文件名', '缺陷数量', '平均置信度']
            
            # 添加批量分析表格
            html_content += """
            <table>
                <tr>
                    <th>文件名</th>
                    <th>缺陷数量</th>
                    <th>平均置信度</th>
                </tr>
            """
            
            for _, row in file_stats.iterrows():
                html_content += f"""
                <tr>
                    <td>{row['文件名']}</td>
                    <td>{row['缺陷数量']}</td>
                    <td>{row['平均置信度']:.4f}</td>
                </tr>
                """
                
            html_content += "</table>"
            
            # 生成批量分析图表
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(file_stats['文件名'], file_stats['缺陷数量'])
            ax.set_title('各文件缺陷数量')
            ax.set_ylabel('缺陷数量')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 将图表转换为base64编码
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # 添加图表到HTML
            html_content += f'<div class="chart-container"><img src="data:image/png;base64,{img_str}" alt="各文件缺陷数量"></div>'
        
        # 结束HTML
        html_content += """
        </body>
        </html>
        """
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    def _export_pdf_report(self, file_path):
        """导出PDF格式的分析报告"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.font_manager as fm
            
            # 设置中文字体支持
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建PDF文件
            with PdfPages(file_path) as pdf:
                # 添加标题页
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle("PCB缺陷检测分析报告", fontsize=16)
                plt.figtext(0.5, 0.8, f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", 
                          ha="center", fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
                
                # 缺陷类型分布
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # 获取数据
                if self.defect_data is not None:
                    data = self.defect_data
                    types = [d['type'] for d in data]
                elif self.batch_data is not None:
                    data = self.batch_data
                    types = data['缺陷类型'].tolist()
                
                # 计算各类型数量
                type_counts = {}
                for t in types:
                    if t in type_counts:
                        type_counts[t] += 1
                    else:
                        type_counts[t] = 1
                
                # 饼图
                ax1.pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
                ax1.set_title('缺陷类型分布')
                
                # 条形图
                ax2.bar(type_counts.keys(), type_counts.values())
                ax2.set_title('缺陷类型计数')
                ax2.set_ylabel('数量')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
                # 其他图表...
                # 可以根据需要添加更多图表
                
                # 添加统计信息页
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle("缺陷统计信息", fontsize=16)
                
                # 添加统计文本
                stats_text = "缺陷总数: {}\n".format(len(types))
                for t, count in type_counts.items():
                    percentage = count / len(types) * 100
                    stats_text += "{}类型: {} ({}%)\n".format(t, count, round(percentage, 1))
                
                plt.figtext(0.1, 0.7, stats_text, fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
                
            return True
        except Exception as e:
            import traceback
            print(f"导出PDF报告失败: {str(e)}")
            print(traceback.format_exc())
            return False
    def save_analysis_data(self, file_path=None):
        """保存分析数据到JSON文件"""
        if file_path is None:
            file_path = filedialog.asksaveasfilename(
                title="保存分析数据",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
        if not file_path:
            return False
            
        try:
            data = {
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存单次检测数据
            if self.defect_data is not None:
                data["defect_data"] = self.defect_data
                
            # 保存批量检测数据
            if self.batch_data is not None:
                # 将DataFrame转换为字典
                data["batch_data"] = self.batch_data.to_dict(orient='records')
                
            # 写入JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            return True
            
        except Exception as e:
            messagebox.showerror("错误", f"保存分析数据失败: {str(e)}")
            return False
    
    def load_analysis_data(self, file_path=None):
        """从JSON文件加载分析数据"""
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="加载分析数据",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
        if not file_path or not os.path.exists(file_path):
            return False
            
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 加载单次检测数据
            if "defect_data" in data:
                self.defect_data = data["defect_data"]
                
            # 加载批量检测数据
            if "batch_data" in data:
                self.batch_data = pd.DataFrame(data["batch_data"])
                
            return True
            
        except Exception as e:
            messagebox.showerror("错误", f"加载分析数据失败: {str(e)}")
            return False
    
    def generate_summary_report(self):
        """生成摘要报告"""
        if self.defect_data is None and self.batch_data is None:
            return "没有可分析的数据"
            
        summary = []
        summary.append("=== PCB缺陷检测分析摘要 ===")
        summary.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # 分析单次检测数据
        if self.defect_data is not None:
            data = self.defect_data
            summary.append(f"单次检测结果分析:")
            summary.append(f"- 检测到缺陷总数: {len(data)}")
            
            # 缺陷类型统计
            types = [d['type'] for d in data]
            type_counts = {}
            for t in types:
                if t in type_counts:
                    type_counts[t] += 1
                else:
                    type_counts[t] = 1
                    
            summary.append("- 缺陷类型分布:")
            for t, count in type_counts.items():
                percentage = count / len(data) * 100
                summary.append(f"  * {t}: {count}个 ({percentage:.1f}%)")
                
            # 置信度统计
            confidences = [d.get('confidence', 0) for d in data]
            if confidences:
                summary.append(f"- 平均置信度: {np.mean(confidences):.4f}")
                summary.append(f"- 最高置信度: {np.max(confidences):.4f}")
                summary.append(f"- 最低置信度: {np.min(confidences):.4f}")
                
            summary.append("")
        
        # 分析批量检测数据
        if self.batch_data is not None:
            data = self.batch_data
            summary.append(f"批量检测结果分析:")
            summary.append(f"- 检测文件总数: {len(data['文件名'].unique())}")
            summary.append(f"- 检测到缺陷总数: {len(data)}")
            
            # 缺陷类型统计
            types = data['缺陷类型'].tolist()
            type_counts = {}
            for t in types:
                if t in type_counts:
                    type_counts[t] += 1
                else:
                    type_counts[t] = 1
                    
            summary.append("- 缺陷类型分布:")
            for t, count in type_counts.items():
                percentage = count / len(data) * 100
                summary.append(f"  * {t}: {count}个 ({percentage:.1f}%)")
                
            # 置信度统计
            confidences = data['置信度'].tolist()
            if confidences:
                summary.append(f"- 平均置信度: {np.mean(confidences):.4f}")
                summary.append(f"- 最高置信度: {np.max(confidences):.4f}")
                summary.append(f"- 最低置信度: {np.min(confidences):.4f}")
                
            # 每个文件的缺陷统计
            file_stats = data.groupby('文件名').agg({
                '缺陷类型': 'count',
                '置信度': 'mean'
            }).reset_index()
            file_stats.columns = ['文件名', '缺陷数量', '平均置信度']
            
            summary.append("- 各文件缺陷统计:")
            for _, row in file_stats.iterrows():
                summary.append(f"  * {row['文件名']}: {row['缺陷数量']}个缺陷, 平均置信度={row['平均置信度']:.4f}")
                
        return "\n".join(summary)
    
    def show_summary_dialog(self, parent):
        """显示摘要对话框"""
        summary = self.generate_summary_report()
        
        # 创建对话框
        dialog = tk.Toplevel(parent)
        dialog.title("分析摘要")
        dialog.geometry("600x400")
        dialog.transient(parent)
        dialog.grab_set()
        
        # 创建文本框
        text = tk.Text(dialog, wrap="word", padx=10, pady=10)
        text.pack(fill="both", expand=True)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(text, command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.config(yscrollcommand=scrollbar.set)
        
        # 插入摘要文本
        text.insert("1.0", summary)
        text.config(state="disabled")  # 设为只读
        
        # 添加关闭按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(btn_frame, text="导出摘要", 
                  command=lambda: self._export_summary(summary)).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="关闭", 
                  command=dialog.destroy).pack(side="right", padx=5)
    
    def _export_summary(self, summary):
        """导出摘要到文本文件"""
        file_path = filedialog.asksaveasfilename(
            title="导出摘要",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            messagebox.showinfo("成功", f"摘要已导出到: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出摘要失败: {str(e)}")
    
    def compare_results(self, other_analyzer):
        """比较两次分析结果"""
        if (self.defect_data is None and self.batch_data is None) or \
           (other_analyzer.defect_data is None and other_analyzer.batch_data is None):
            messagebox.showwarning("提示", "没有足够的数据进行比较")
            return
            
        # 创建比较窗口
        compare_window = tk.Toplevel()
        compare_window.title("检测结果比较")
        compare_window.geometry("900x700")
        
        # 创建选项卡
        notebook = ttk.Notebook(compare_window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 缺陷类型比较选项卡
        type_tab = ttk.Frame(notebook)
        notebook.add(type_tab, text="缺陷类型比较")
        self._create_type_comparison_tab(type_tab, other_analyzer)
        
        # 置信度比较选项卡
        conf_tab = ttk.Frame(notebook)
        notebook.add(conf_tab, text="置信度比较")
        self._create_confidence_comparison_tab(conf_tab, other_analyzer)
        
        # 如果有批量数据，添加批量比较选项卡
        if self.batch_data is not None and other_analyzer.batch_data is not None:
            batch_tab = ttk.Frame(notebook)
            notebook.add(batch_tab, text="批量比较")
            self._create_batch_comparison_tab(batch_tab, other_analyzer)
    
    def _create_type_comparison_tab(self, parent, other_analyzer):
        """创建缺陷类型比较选项卡"""
        # 获取数据
        if self.defect_data is not None:
            data1 = self.defect_data
            types1 = [d['type'] for d in data1]
        elif self.batch_data is not None:
            data1 = self.batch_data
            types1 = data1['缺陷类型'].tolist()
        else:
            return
            
        if other_analyzer.defect_data is not None:
            data2 = other_analyzer.defect_data
            types2 = [d['type'] for d in data2]
        elif other_analyzer.batch_data is not None:
            data2 = other_analyzer.batch_data
            types2 = data2['缺陷类型'].tolist()
        else:
            return
        
        # 计算各类型数量
        type_counts1 = {}
        for t in types1:
            if t in type_counts1:
                type_counts1[t] += 1
            else:
                type_counts1[t] = 1
                
        type_counts2 = {}
        for t in types2:
            if t in type_counts2:
                type_counts2[t] += 1
            else:
                type_counts2[t] = 1
        
        # 合并所有类型
        all_types = list(set(list(type_counts1.keys()) + list(type_counts2.keys())))
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置柱状图位置
        x = np.arange(len(all_types))
        width = 0.35
        
        # 绘制柱状图
        counts1 = [type_counts1.get(t, 0) for t in all_types]
        counts2 = [type_counts2.get(t, 0) for t in all_types]
        
        rects1 = ax.bar(x - width/2, counts1, width, label='结果1')
        rects2 = ax.bar(x + width/2, counts2, width, label='结果2')
        
        # 添加标签和图例
        ax.set_xlabel('缺陷类型')
        ax.set_ylabel('数量')
        ax.set_title('缺陷类型比较')
        ax.set_xticks(x)
        ax.set_xticklabels(all_types)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_confidence_comparison_tab(self, parent, other_analyzer):
        """创建置信度比较选项卡"""
        # 获取数据
        if self.defect_data is not None:
            data1 = self.defect_data
            confidences1 = [d.get('confidence', 0) for d in data1]
        elif self.batch_data is not None:
            data1 = self.batch_data
            confidences1 = data1['置信度'].tolist()
        else:
            return
            
        if other_analyzer.defect_data is not None:
            data2 = other_analyzer.defect_data
            confidences2 = [d.get('confidence', 0) for d in data2]
        elif other_analyzer.batch_data is not None:
            data2 = other_analyzer.batch_data
            confidences2 = data2['置信度'].tolist()
        else:
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 直方图比较
        ax1.hist(confidences1, bins=10, alpha=0.5, label='结果1')
        ax1.hist(confidences2, bins=10, alpha=0.5, label='结果2')
        ax1.set_title('置信度分布比较')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('频率')
        ax1.legend()
        
        # 箱线图比较
        box_data = [confidences1, confidences2]
        ax2.boxplot(box_data, labels=['结果1', '结果2'])
        ax2.set_title('置信度箱线图比较')
        ax2.set_ylabel('置信度')
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_batch_comparison_tab(self, parent, other_analyzer):
        """创建批量比较选项卡"""
        if self.batch_data is None or other_analyzer.batch_data is None:
            return
            
        # 按文件名分组统计
        file_stats1 = self.batch_data.groupby('文件名').agg({
            '缺陷类型': 'count',
            '置信度': 'mean'
        }).reset_index()
        file_stats1.columns = ['文件名', '缺陷数量1', '平均置信度1']
        
        file_stats2 = other_analyzer.batch_data.groupby('文件名').agg({
            '缺陷类型': 'count',
            '置信度': 'mean'
        }).reset_index()
        file_stats2.columns = ['文件名', '缺陷数量2', '平均置信度2']
        
        # 合并数据
        merged_stats = pd.merge(file_stats1, file_stats2, on='文件名', how='outer').fillna(0)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 缺陷数量比较
        x = np.arange(len(merged_stats))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, merged_stats['缺陷数量1'], width, label='结果1')
        rects2 = ax1.bar(x + width/2, merged_stats['缺陷数量2'], width, label='结果2')
        
        ax1.set_title('各文件缺陷数量比较')
        ax1.set_ylabel('缺陷数量')
        ax1.set_xticks(x)
        ax1.set_xticklabels(merged_stats['文件名'], rotation=45, ha='right')
        ax1.legend()
        
        # 平均置信度比较
        rects3 = ax2.bar(x - width/2, merged_stats['平均置信度1'], width, label='结果1')
        rects4 = ax2.bar(x + width/2, merged_stats['平均置信度2'], width, label='结果2')
        
        ax2.set_title('各文件平均置信度比较')
        ax2.set_ylabel('平均置信度')
        ax2.set_xticks(x)
        ax2.set_xticklabels(merged_stats['文件名'], rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加统计表格
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建表格
        tree = ttk.Treeview(frame, columns=("文件名", "缺陷数量1", "缺陷数量2", "差异", 
                                          "平均置信度1", "平均置信度2", "置信度差异"))
        
        # 设置列标题
        tree.heading("文件名", text="文件名")
        tree.heading("缺陷数量1", text="缺陷数量(结果1)")
        tree.heading("缺陷数量2", text="缺陷数量(结果2)")
        tree.heading("差异", text="数量差异")
        tree.heading("平均置信度1", text="平均置信度(结果1)")
        tree.heading("平均置信度2", text="平均置信度(结果2)")
        tree.heading("置信度差异", text="置信度差异")
        
        # 设置列宽度
        tree.column("#0", width=0, stretch=tk.NO)
        for col in tree["columns"]:
            tree.column(col, width=100)
        
        # 添加数据
        for _, row in merged_stats.iterrows():
            count_diff = row['缺陷数量1'] - row['缺陷数量2']
            conf_diff = row['平均置信度1'] - row['平均置信度2']
            
            tree.insert("", "end", values=(
                row['文件名'],
                f"{row['缺陷数量1']:.0f}",
                f"{row['缺陷数量2']:.0f}",
                f"{count_diff:+.0f}",
                f"{row['平均置信度1']:.4f}",
                f"{row['平均置信度2']:.4f}",
                f"{conf_diff:+.4f}"
            ))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(fill="both", expand=True)
        
        # 添加统计摘要
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill="x", padx=10, pady=10)
        
        # 计算总体统计
        total_defects1 = merged_stats['缺陷数量1'].sum()
        total_defects2 = merged_stats['缺陷数量2'].sum()
        avg_conf1 = merged_stats['平均置信度1'].mean()
        avg_conf2 = merged_stats['平均置信度2'].mean()
        
        # 创建统计标签
        ttk.Label(summary_frame, text=f"总缺陷数（结果1）: {total_defects1}").pack(side="left", padx=5)
        ttk.Label(summary_frame, text=f"总缺陷数（结果2）: {total_defects2}").pack(side="left", padx=5)
        ttk.Label(summary_frame, text=f"平均置信度（结果1）: {avg_conf1:.4f}").pack(side="left", padx=5)
        ttk.Label(summary_frame, text=f"平均置信度（结果2）: {avg_conf2:.4f}").pack(side="left", padx=5)
        
        # 添加导出按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        def export_comparison():
            # 创建导出文件路径
            file_path = filedialog.asksaveasfilename(
                title="导出比较结果",
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
            )
            
            if not file_path:
                return
                
            try:
                # 添加额外统计信息
                merged_stats['总缺陷数1'] = total_defects1
                merged_stats['总缺陷数2'] = total_defects2
                merged_stats['平均置信度1'] = avg_conf1
                merged_stats['平均置信度2'] = avg_conf2
                
                # 导出为CSV
                merged_stats.to_csv(file_path, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"比较结果已导出到: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {str(e)}")
        
        ttk.Button(btn_frame, text="导出比较结果", command=export_comparison).pack(side="right", padx=5)