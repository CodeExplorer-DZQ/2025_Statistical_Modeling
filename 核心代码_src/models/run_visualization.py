# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型可视化运行脚本
功能：展示模型耦合优势并生成可视化图表
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from .model_advantages import ModelAdvantages
from .visualization import ModelVisualization

def run_visualization():
    """
    运行可视化脚本，生成模型优势和可视化图表
    """
    print("\n===== 开始生成Prophet-GTWR耦合模型可视化 =====\n")
    
    # 创建可视化实例
    vis = ModelVisualization()
    
    # 1. 打印并可视化模型优势
    print("\n1. 生成模型耦合优势说明和图表...")
    ModelAdvantages.print_advantages()
    vis.plot_coupling_advantages(filename='0422_耦合模型优势')
    
    # 2. 生成模型工作流程图
    print("\n2. 生成模型工作流程图...")
    vis.plot_coupling_workflow(filename='0422_耦合模型工作流程')
    
    # 3. 模拟数据：生成模型性能比较图
    print("\n3. 生成模型性能比较图...")
    metrics_dict = {
        'prophet_rmse': 0.1856,
        'prophet_r2': 0.7823,
        'gtwr_rmse': 0.1542,
        'gtwr_r2': 0.8245,
        'rmse': 0.1238,
        'r2': 0.8912
    }
    vis.plot_model_comparison(metrics_dict, filename='0422_模型性能比较')
    
    # 4. 模拟数据：生成特征重要性图
    print("\n4. 生成特征重要性图...")
    feature_names = [
        'Prophet预测值', '周边POI密度', '距离市中心', '人口密度', 
        '充电桩密度', '交通可达性', '用电负荷', '时间特征'
    ]
    importance_values = [0.32, 0.28, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04]
    vis.plot_feature_importance(
        feature_names, importance_values, 
        title='Prophet-GTWR耦合模型特征重要性',
        filename='0422_特征重要性'
    )
    
    # 5. 模拟数据：生成时序预测对比图
    print("\n5. 生成时序预测对比图...")
    # 创建模拟时间序列数据
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    n = len(dates)
    
    # 创建模拟实际值（带有趋势、季节性和噪声）
    trend = np.linspace(10, 20, n)  # 上升趋势
    seasonality = 5 * np.sin(np.linspace(0, 6*np.pi, n))  # 季节性
    noise = np.random.normal(0, 1, n)  # 随机噪声
    actual_values = trend + seasonality + noise
    
    # 创建模拟预测值
    prophet_values = trend + seasonality + np.random.normal(0, 1.5, n)
    gtwr_values = trend + seasonality + np.random.normal(0, 1.2, n)
    coupled_values = trend + seasonality + np.random.normal(0, 0.8, n)
    
    # 创建数据框
    actual_df = pd.DataFrame({'ds': dates, 'y': actual_values})
    prophet_df = pd.DataFrame({'ds': dates, 'yhat': prophet_values})
    
    # 绘制时序预测对比图
    vis.plot_time_series_prediction(
        actual_df, prophet_df, gtwr_values, coupled_values,
        time_col='ds', filename='0422_时序预测对比'
    )
    
    # 6. 模拟数据：生成空间分布图
    print("\n6. 生成空间分布图...")
    # 创建模拟空间数据（北京市范围内的随机点）
    n_points = 200
    np.random.seed(42)
    
    # 北京市大致经纬度范围
    lon_min, lon_max = 116.0, 116.8
    lat_min, lat_max = 39.6, 40.2
    
    # 生成随机坐标
    longitudes = np.random.uniform(lon_min, lon_max, n_points)
    latitudes = np.random.uniform(lat_min, lat_max, n_points)
    
    # 生成模拟预测值（与位置相关）
    # 中心点（天安门）
    center_lon, center_lat = 116.4, 39.9
    
    # 计算到中心的距离
    distances = np.sqrt((longitudes - center_lon)**2 + (latitudes - center_lat)**2)
    
    # 基于距离生成预测值（距离越近，值越高）
    predicted_values = 100 * (1 - distances / distances.max()) + np.random.normal(0, 10, n_points)
    
    # 创建数据框
    geo_data = pd.DataFrame({
        '经度': longitudes,
        '纬度': latitudes,
        '预测充电需求': predicted_values
    })
    
    # 绘制空间分布图
    vis.plot_spatial_distribution(
        geo_data, '预测充电需求',
        title='充电需求空间分布预测',
        filename='0422_充电需求空间分布'
    )
    
    print("\n===== Prophet-GTWR耦合模型可视化生成完成 =====\n")

if __name__ == "__main__":
    run_visualization()