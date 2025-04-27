# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型 - 示例运行脚本
功能：演示如何使用Prophet-GTWR耦合模型进行充电桩布局优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # 忽略Prophet的警告

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  

# 导入自定义模块
from 核心代码_src.models.prophet_gtwr_complete import ProphetGTWRComplete

def create_sample_data(n_samples=500, n_stations=20, start_date='2024-01-01', end_date='2024-04-01'):
    """
    创建示例数据
    
    Args:
        n_samples: 样本数量
        n_stations: 充电站数量
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        示例数据DataFrame
    """
    print("创建示例数据...")
    
    # 生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 生成充电站位置（深圳市范围内的经纬度）
    np.random.seed(42)
    stations = {
        'station_id': [f'SZ{i:03d}' for i in range(1, n_stations+1)],
        '经度': np.random.uniform(113.8, 114.5, n_stations),  # 深圳经度范围
        '纬度': np.random.uniform(22.4, 22.8, n_stations),    # 深圳纬度范围
        '充电桩数量': np.random.randint(5, 30, n_stations),
        '站点类型': np.random.choice(['公共', '专用', '快充', '慢充'], n_stations)
    }
    
    # 创建数据框
    data = []
    
    for station_id, lon, lat, num_chargers, station_type in zip(
        stations['station_id'], stations['经度'], stations['纬度'], 
        stations['充电桩数量'], stations['站点类型']):
        
        # 为每个充电站生成时间序列数据
        for date in date_range:
            # 基础功率
            base_power = np.random.uniform(30, 100)
            
            # 添加趋势（随时间缓慢增长）
            days_since_start = (date - pd.to_datetime(start_date)).days
            trend = days_since_start * 0.1
            
            # 添加周季节性（工作日比周末高）
            weekday = date.weekday()
            weekly_effect = -5 if weekday >= 5 else 5  # 周末降低，工作日增加
            
            # 添加时间特殊效应（春节期间低谷）
            spring_festival_2024 = pd.to_datetime('2024-02-10')
            days_to_festival = abs((date - spring_festival_2024).days)
            festival_effect = -15 if days_to_festival < 7 else 0
            
            # 添加空间效应（与市中心距离）
            # 深圳市中心坐标约为(114.06, 22.55)
            dist_to_center = np.sqrt((lon - 114.06)**2 + (lat - 22.55)**2)
            spatial_effect = -10 * dist_to_center  # 距离市中心越远，功率越低
            
            # 计算最终功率，添加随机噪声
            power = base_power + trend + weekly_effect + festival_effect + spatial_effect
            power = max(power, 10)  # 确保功率为正
            power += np.random.normal(0, 5)  # 添加随机噪声
            
            # 计算占用率（与功率相关但有随机性）
            occupancy = min(95, max(5, power * 0.8 + np.random.normal(0, 10)))
            
            # 添加环境特征
            temperature = 20 + 10 * np.sin(2 * np.pi * days_since_start / 365) + np.random.normal(0, 3)
            precipitation = max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0)
            
            # 添加交通特征
            traffic_flow = 1000 + 500 * np.sin(2 * np.pi * weekday / 7) + np.random.normal(0, 100)
            
            # 添加POI特征
            poi_commercial = int(np.random.poisson(10 * (1 - dist_to_center)))
            poi_residential = int(np.random.poisson(15 * (1 - dist_to_center)))
            
            # 添加到数据列表
            data.append({
                '站点ID': station_id,
                '记录时间': date,
                '经度': lon,
                '纬度': lat,
                '功率(kw)': power,
                '实时占用率': occupancy,
                '充电桩数量': num_chargers,
                '站点类型': station_type,
                '气温': temperature,
                '降水量': precipitation,
                '交通流量': traffic_flow,
                '商业POI数量': poi_commercial,
                '居住POI数量': poi_residential
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    print(f"示例数据创建完成，共{len(df)}条记录")
    return df

def main():
    """
    主函数：演示Prophet-GTWR耦合模型的使用流程
    """
    print("\n" + "=" * 80)
    print("Prophet-GTWR耦合模型 - 充电桩布局优化示例")
    print("=" * 80)
    
    # 1. 数据准备
    print("\n1. 数据准备")
    print("-" * 50)
    
    # 尝试加载真实数据，如果失败则创建示例数据
    try:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data = pd.read_csv(os.path.join(data_path, "数据_data/1_充电桩数据/processed/深圳高德最终数据_with_location_type.csv"))
        print("成功加载真实数据")
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        print("使用示例数据进行演示...")
        data = create_sample_data()
    
    # 显示数据基本信息
    print("\n数据基本信息:")
    print(f"样本数量: {len(data)}")
    print(f"特征数量: {len(data.columns)}")
    print("\n特征列表:")
    print(", ".join(data.columns.tolist()))
    
    # 2. 设置模型参数
    print("\n2. 设置模型参数")
    print("-" * 50)
    
    # Prophet模型参数
    prophet_params = {
        'seasonality_mode': 'multiplicative',  # 乘法季节性
        'yearly_seasonality': True,           # 年度季节性
        'weekly_seasonality': True,           # 周季节性
        'daily_seasonality': False,           # 不使用日季节性
        'changepoint_prior_scale': 0.05       # 趋势变化点先验尺度
    }
    
    # GTWR模型参数
    gtwr_params = {
        'spatial_bandwidth': None,            # 自适应空间带宽
        'temporal_bandwidth': None,           # 自适应时间带宽
        'kernel_function': 'gaussian',        # 高斯核函数
        'time_decay_lambda': 0.5,             # 时间衰减系数
        'optimization_method': 'cv'           # 交叉验证优化方法
    }
    
    # 耦合参数
    coupling_params = {
        'alpha': 0.5,                         # 融合权重参数
        'feedback_mode': 'two_way',           # 双向反馈模式
        'dynamic_weight': True                # 使用动态权重
    }
    
    print("Prophet参数:")
    for k, v in prophet_params.items():
        print(f"  {k}: {v}")
    
    print("\nGTWR参数:")
    for k, v in gtwr_params.items():
        print(f"  {k}: {v}")
    
    print("\n耦合参数:")
    for k, v in coupling_params.items():
        print(f"  {k}: {v}")
    
    # 3. 初始化并训练模型
    print("\n3. 初始化并训练模型")
    print("-" * 50)
    
    # 初始化模型
    model = ProphetGTWRComplete(
        prophet_params=prophet_params,
        gtwr_params=gtwr_params,
        coupling_params=coupling_params
    )
    
    # 准备特征列表
    features = ['气温', '降水量', '交通流量', '商业POI数量', '居住POI数量', '充电桩数量']
    
    # 训练耦合模型
    results = model.fit_coupling_model(
        data=data,
        features=features,
        target='功率(kw)',
        time_col='记录时间',
        coords_cols=['经度', '纬度'],
        test_size=0.2,
        random_state=42
    )
    
    # 4. 模型评估
    print("\n4. 模型评估")
    print("-" * 50)
    
    print(f"最终模型性能:")
    print(f"  均方误差(MSE): {results['mse']:.4f}")
    print(f"  平均绝对误差(MAE): {results['mae']:.4f}")
    print(f"  决定系数(R²): {results['r2']:.4f}")
    print(f"  相比Prophet改进: {results['mse_improvement']:.2f}%")
    
    # 5. 预测未来数据
    print("\n5. 预测未来数据")
    print("-" * 50)
    
    # 创建未来30天的预测数据框架
    last_date = pd.to_datetime(data['记录时间']).max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    
    # 为每个充电站创建未来数据点
    stations = data[['站点ID', '经度', '纬度', '充电桩数量', '站点类型']].drop_duplicates()
    future_data = []
    
    for _, station in stations.iterrows():
        for date in future_dates:
            # 复制最近的环境和交通特征，添加一些随机变化
            recent_data = data[data['站点ID'] == station['站点ID']].iloc[-1]
            
            future_data.append({
                '站点ID': station['站点ID'],
                '记录时间': date,
                '经度': station['经度'],
                '纬度': station['纬度'],
                '充电桩数量': station['充电桩数量'],
                '站点类型': station['站点类型'],
                '气温': recent_data['气温'] + np.random.normal(0, 2),
                '降水量': max(0, recent_data['降水量'] + np.random.normal(0, 1)),
                '交通流量': recent_data['交通流量'] + np.random.normal(0, 50),
                '商业POI数量': recent_data['商业POI数量'],
                '居住POI数量': recent_data['居住POI数量']
            })
    
    future_df = pd.DataFrame(future_data)
    
    # 使用模型预测未来数据
    print(f"预测未来{len(future_dates)}天的充电需求...")
    predictions = model.predict(future_df, features)
    
    print(f"预测完成，结果示例:")
    print(predictions[['站点ID', '记录时间', 'prophet_pred', 'gtwr_pred', 'final_pred']].head())
    
    # 6. 可视化结果
    print("\n6. 可视化结果")
    print("-" * 50)
    
    # 创建结果保存目录
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '项目产出_results', 'figures'))
    os.makedirs(results_dir, exist_ok=True)
    
    # 可视化模型结果
    print("生成可视化图表...")
    figures = model.visualize_results(save_path=results_dir)
    
    print(f"已生成{len(figures)}个图表并保存到: {results_dir}")
    
    # 7. 保存模型
    print("\n7. 保存模型")
    print("-" * 50)
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '项目产出_results', 'models', 'prophet_gtwr_model.json'))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    print("\n" + "=" * 80)
    print("Prophet-GTWR耦合模型演示完成")
    print("=" * 80)

if __name__ == "__main__":
    main()