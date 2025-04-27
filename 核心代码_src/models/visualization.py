# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型可视化模块
功能：创建模型相关的可视化图表，支持中文字体显示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体，防止乱码
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，可能会导致中文显示为方块")

# 定义紫色系渐变色彩映射
purple_colors = [(0.9, 0.8, 0.95), (0.7, 0.4, 0.9), (0.5, 0.0, 0.8), (0.3, 0.0, 0.5)]
purple_cmap = LinearSegmentedColormap.from_list('purple_gradient', purple_colors)

# 设置Seaborn风格
sns.set(style="whitegrid")

class ModelVisualization:
    """
    Prophet-GTWR耦合模型可视化类
    创建模型相关的可视化图表
    """
    
    def __init__(self, save_dir=None):
        """
        初始化可视化类
        
        Args:
            save_dir: 图片保存目录，默认为'项目产出_results/figures'
        """
        # 设置默认保存目录
        if save_dir is None:
            # 获取当前文件所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上两级目录，然后进入项目产出_results/figures
            self.save_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '项目产出_results', 'figures'))
        else:
            self.save_dir = save_dir
            
        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"图片将保存到: {self.save_dir}")
    
    def _save_figure(self, fig, filename):
        """
        保存图片
        
        Args:
            fig: matplotlib图形对象
            filename: 文件名
        """
        # 添加0422_前缀
        if not filename.startswith('0422_'):
            filename = f'0422_{filename}'
        
        # 确保文件扩展名为.png
        if not filename.endswith('.png'):
            filename += '.png'
        
        # 完整路径
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存图片
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"图片已保存: {filepath}")
        
        return filepath
    
    def plot_model_comparison(self, metrics_dict, filename='模型性能比较'):
        """
        绘制模型性能比较图
        
        Args:
            metrics_dict: 包含各模型评估指标的字典
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        # 提取RMSE和R²指标
        models = ['Prophet', 'GTWR', '耦合模型']
        rmse_values = [
            metrics_dict.get('prophet_rmse', 0),
            metrics_dict.get('gtwr_rmse', 0),
            metrics_dict.get('rmse', 0)
        ]
        r2_values = [
            metrics_dict.get('prophet_r2', 0),
            metrics_dict.get('gtwr_r2', 0),
            metrics_dict.get('r2', 0)
        ]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 绘制RMSE对比
        bars1 = ax1.bar(models, rmse_values, color=purple_colors[1:], alpha=0.8)
        ax1.set_title('模型RMSE对比（越低越好）', fontsize=14)
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 绘制R²对比
        bars2 = ax2.bar(models, r2_values, color=purple_colors[1:], alpha=0.8)
        ax2.set_title('模型R²对比（越高越好）', fontsize=14)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 设置图形标题和样式
        fig.suptitle('Prophet-GTWR耦合模型性能对比', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图片
        return self._save_figure(fig, filename)
    
    def plot_time_series_prediction(self, actual, prophet_pred, gtwr_pred, coupled_pred, 
                                  time_col='ds', filename='时序预测对比'):
        """
        绘制时间序列预测对比图
        
        Args:
            actual: 实际值Series或DataFrame
            prophet_pred: Prophet预测值
            gtwr_pred: GTWR预测值
            coupled_pred: 耦合模型预测值
            time_col: 时间列名
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制实际值
        if actual is not None:
            ax.scatter(actual[time_col], actual.iloc[:, 1], color='black', s=15, alpha=0.6, label='实际值')
        
        # 绘制Prophet预测
        if prophet_pred is not None:
            ax.plot(prophet_pred[time_col], prophet_pred['yhat'], color=purple_colors[0], 
                   linewidth=2, label='Prophet预测')
        
        # 绘制GTWR预测
        if gtwr_pred is not None:
            ax.plot(prophet_pred[time_col], gtwr_pred, color=purple_colors[1], 
                   linewidth=2, label='GTWR预测')
        
        # 绘制耦合预测
        if coupled_pred is not None:
            ax.plot(prophet_pred[time_col], coupled_pred, color=purple_colors[3], 
                   linewidth=2.5, label='耦合模型预测')
        
        # 设置图形属性
        ax.set_title('Prophet-GTWR耦合模型时序预测对比', fontsize=15)
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('充电需求', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # 美化图形
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        return self._save_figure(fig, filename)
    
    def plot_spatial_distribution(self, geo_data, value_col, coords_cols=['经度', '纬度'], 
                                title=None, filename='空间分布图'):
        """
        绘制空间分布图
        
        Args:
            geo_data: 包含地理坐标和值的DataFrame
            value_col: 值列名
            coords_cols: 坐标列名列表，默认['经度', '纬度']
            title: 图表标题
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制散点图
        scatter = ax.scatter(
            geo_data[coords_cols[0]], 
            geo_data[coords_cols[1]],
            c=geo_data[value_col], 
            cmap=purple_cmap,
            s=50, 
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(value_col, fontsize=12)
        
        # 设置图形属性
        if title is None:
            title = f'{value_col}空间分布'  
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(coords_cols[0], fontsize=12)
        ax.set_ylabel(coords_cols[1], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 美化图形
        plt.tight_layout()
        
        # 保存图片
        return self._save_figure(fig, filename)
    
    def plot_coupling_advantages(self, filename='耦合模型优势'):
        """
        绘制耦合模型优势图
        
        Args:
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        # 导入模型优势
        from .model_advantages import ModelAdvantages
        advantages = ModelAdvantages.get_coupling_advantages()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 设置标题
        ax.text(0.5, 0.98, 'Prophet-GTWR耦合模型优势', 
                fontsize=20, ha='center', va='top', weight='bold')
        
        # 绘制优势内容
        y_pos = 0.92
        for i, (category, items) in enumerate(advantages.items()):
            # 绘制类别标题
            ax.text(0.05, y_pos, f'{category}', 
                    fontsize=16, ha='left', va='top', weight='bold',
                    color=purple_colors[i % len(purple_colors)])
            
            y_pos -= 0.05
            
            # 绘制类别项目
            for j, item in enumerate(items, 1):
                ax.text(0.08, y_pos, f'{j}. ', 
                        fontsize=12, ha='left', va='top', weight='bold',
                        color=purple_colors[i % len(purple_colors)])
                ax.text(0.12, y_pos, f'{item}', 
                        fontsize=12, ha='left', va='top',
                        wrap=True)
                y_pos -= 0.05
            
            y_pos -= 0.03
        
        # 添加装饰边框
        border = plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
                              edgecolor=purple_colors[2], linewidth=2, alpha=0.7)
        ax.add_patch(border)
        
        # 添加角落装饰
        corner_size = 0.05
        corners = [
            plt.Rectangle((0.02, 0.02), corner_size, corner_size, fill=True, 
                         color=purple_colors[1], alpha=0.5),
            plt.Rectangle((0.02, 0.98-corner_size), corner_size, corner_size, fill=True, 
                         color=purple_colors[1], alpha=0.5),
            plt.Rectangle((0.98-corner_size, 0.02), corner_size, corner_size, fill=True, 
                         color=purple_colors[1], alpha=0.5),
            plt.Rectangle((0.98-corner_size, 0.98-corner_size), corner_size, corner_size, fill=True, 
                         color=purple_colors[1], alpha=0.5)
        ]
        for corner in corners:
            ax.add_patch(corner)
        
        # 保存图片
        return self._save_figure(fig, filename)
    
    def plot_feature_importance(self, feature_names, importance_values, 
                              title='特征重要性', filename='特征重要性'):
        """
        绘制特征重要性图
        
        Args:
            feature_names: 特征名称列表
            importance_values: 重要性值列表
            title: 图表标题
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        # 创建DataFrame并排序
        df = pd.DataFrame({'特征': feature_names, '重要性': importance_values})
        df = df.sort_values('重要性', ascending=True)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制水平条形图
        bars = ax.barh(df['特征'], df['重要性'], 
                      color=[purple_cmap(x/max(importance_values)) for x in df['重要性']], 
                      alpha=0.8, height=0.6)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', va='center', fontsize=10)
        
        # 设置图形属性
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_ylabel('特征', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 美化图形
        plt.tight_layout()
        
        # 保存图片
        return self._save_figure(fig, filename)
    
    def plot_coupling_workflow(self, filename='耦合模型工作流程'):
        """
        绘制耦合模型工作流程图
        
        Args:
            filename: 保存的文件名
            
        Returns:
            保存的文件路径
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 设置标题
        ax.text(0.5, 0.95, 'Prophet-GTWR耦合模型工作流程', 
                fontsize=18, ha='center', va='center', weight='bold')
        
        # 定义流程步骤和位置
        steps = [
            {'name': '数据预处理', 'x': 0.5, 'y': 0.85, 'width': 0.3, 'height': 0.08},
            {'name': 'Prophet时序模型', 'x': 0.3, 'y': 0.7, 'width': 0.25, 'height': 0.08},
            {'name': 'GTWR空间模型', 'x': 0.7, 'y': 0.7, 'width': 0.25, 'height': 0.08},
            {'name': '时序预测结果', 'x': 0.3, 'y': 0.55, 'width': 0.25, 'height': 0.08},
            {'name': '空间预测结果', 'x': 0.7, 'y': 0.55, 'width': 0.25, 'height': 0.08},
            {'name': '动态权重融合', 'x': 0.5, 'y': 0.4, 'width': 0.3, 'height': 0.08},
            {'name': '耦合模型预测结果', 'x': 0.5, 'y': 0.25, 'width': 0.3, 'height': 0.08},
            {'name': '模型评估与应用', 'x': 0.5, 'y': 0.1, 'width': 0.3, 'height': 0.08}
        ]
        
        # 绘制流程步骤
        for i, step in enumerate(steps):
            # 绘制步骤框
            rect = plt.Rectangle(
                (step['x'] - step['width']/2, step['y'] - step['height']/2),
                step['width'], step['height'],
                facecolor=purple_colors[i % len(purple_colors)],
                edgecolor='black',
                alpha=0.7,
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(rect)
            
            # 添加步骤名称
            ax.text(step['x'], step['y'], step['name'], 
                    ha='center', va='center', fontsize=12, weight='bold',
                    zorder=3)
        
        # 绘制连接箭头
        arrows = [
            {'start': 0, 'end': 1, 'style': '->', 'label': '时间序列数据'},
            {'start': 0, 'end': 2, 'style': '->', 'label': '空间属性数据'},
            {'start': 1, 'end': 3, 'style': '->', 'label': ''},
            {'start': 2, 'end': 4, 'style': '->', 'label': ''},
            {'start': 3, 'end': 5, 'style': '->', 'label': ''},
            {'start': 4, 'end': 5, 'style': '->', 'label': ''},
            {'start': 5, 'end': 6, 'style': '->', 'label': ''},
            {'start': 6, 'end': 7, 'style': '->', 'label': ''}
        ]
        
        # 添加双向反馈箭头
        feedback_arrows = [
            {'start': 3, 'end': 2, 'style': '->', 'label': '双向反馈', 'color': 'red', 'linestyle': '--'}
        ]
        
        # 绘制普通箭头
        for arrow in arrows:
            start = steps[arrow['start']]
            end = steps[arrow['end']]
            
            # 确定箭头起点和终点
            if start['y'] > end['y']:  # 垂直向下
                start_x, start_y = start['x'], start['y'] - start['height']/2
                end_x, end_y = end['x'], end['y'] + end['height']/2
            elif start['x'] < end['x']:  # 水平向右
                start_x, start_y = start['x'] + start['width']/2, start['y']
                end_x, end_y = end['x'] - end['width']/2, end['y']
            else:  # 水平向左
                start_x, start_y = start['x'] - start['width']/2, start['y']
                end_x, end_y = end['x'] + end['width']/2, end['y']
            
            # 绘制箭头
            ax.annotate(
                '', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle=arrow['style'], lw=1.5, 
                               color=purple_colors[2], connectionstyle='arc3,rad=0.1'),
                zorder=1
            )
            
            # 添加标签
            if arrow['label']:
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                ax.text(mid_x, mid_y, arrow['label'], 
                        ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # 绘制反馈箭头
        for arrow in feedback_arrows:
            start = steps[arrow['start']]
            end = steps[arrow['end']]
            
            # 确定箭头起点和终点
            start_x, start_y = start['x'] - start['width']/4, start['y']
            end_x, end_y = end['x'] - end['width']/4, end['y']
            
            # 绘制箭头
            ax.annotate(
                '', xy=(end_x, end_y), xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle=arrow['style'], lw=1.5, 
                               color=arrow['color'], linestyle=arrow['linestyle'],
                               connectionstyle='arc3,rad=0.3'),
                zorder=1
            )
            
            # 添加标签
            if arrow['label']:
                mid_x = (start_x + end_x) / 2 - 0.05
                mid_y = (start_y + end_y) / 2
                ax.text(mid_x, mid_y, arrow['label'], 
                        ha='center', va='center', fontsize=10, color='red',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # 保存图片
        return self._save_figure(fig, filename)


if __name__ == "__main__":
    # 创建可视化实例
    vis = ModelVisualization()
    
    # 测试绘制耦合模型优势图
    vis.plot_coupling_advantages()
    
    # 测试绘制耦合模型工作流程图
    vis.plot_coupling_workflow()
    
    print("可视化测试完成")