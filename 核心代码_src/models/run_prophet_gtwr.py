# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型 - 主运行脚本
功能：演示如何使用Prophet-GTWR耦合模型进行充电桩布局优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入自定义模块
from 核心代码_src.models.data_interface import DataInterface
from 核心代码_src.models.prophet_model import ProphetModel
from 核心代码_src.models.gtwr_model import GTWRModel
from 核心代码_src.models.prophet_gtwr_coupling import ProphetGTWRCoupling

def main():
    """
    主函数：演示Prophet-GTWR耦合模型的使用流程
    """
    print("\n" + "=" * 80)
    print("Prophet-GTWR耦合模型 - 充电桩布局优化示例")
    print("=" * 80)
    
    # 1. 数据加载与预处理
    print("\n1. 数据加载与预处理")
    print("-" * 50)
    
    # 创建数据接口
    data_interface = DataInterface()
    
    # 注册数据源
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # 注册充电桩数据
    data_interface.register_data_source(
        name="charging_stations",
        source_type="csv",
        path=os.path.join(data_path, "数据_data/1_充电桩数据/processed/深圳高德最终数据_with_location_type.csv")
    )
    
    # 注册交通流量数据
    data_interface.register_data_source(
        name="traffic_flow",
        source_type="csv",
        path=os.path.join(data_path, "数据_data/2_时空动态数据/主要道路车流量数据.csv")
    )
    
    # 注册人口热力图数据
    data_interface.register_data_source(
        name="population_heatmap",
        source_type="csv",
        path=os.path.join(data_path, "数据_data/2_时空动态数据/深圳街道级人口热力图.csv")
    )
    
    # 加载数据
    try:
        data = data_interface.load_data()
        print("数据加载成功")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("使用示例数据进行演示...")
        # 创建示例数据
        create_sample_data(data_interface)
        data = data_interface.processed_data
    
    # 2. 数据融合与特征工程
    print("\n2. 数据融合与特征工程")
    print("-" * 50)
    
    # 时空对齐
    try:
        # 空间对齐
        grid_data = data_interface.spatial_alignment(
            data_list=["charging_stations", "traffic_flow", "population_heatmap"],
            grid_size=1000  # 1km网格
        )
        print(f"空间对齐完成，生成{len(grid_data)}个网格")
        
        # 时间对齐
        time_data = data_interface.temporal_alignment(
            data_list=["charging_stations", "traffic_flow"],
            target_freq='D'  # 日级粒度
        )
        print(f"时间对齐完成，生成{len(time_data)}个时间点")
        
        # 特征提取
        # 提取时间特征
        time_features = data_interface.extract_time_features(time_data)
        print(f"时间特征提取完成，生成{len(time_features.columns)}个特征")
        
        # 提取空间特征
        spatial_features = data_interface.extract_spatial_features(grid_data)
        print(f"空间特征提取完成，生成{len(spatial_features.columns)}个特征")
        
        # 特征归一化
        normalized_features = data_interface.normalize_features(spatial_features)
        print("特征归一化完成")
        
        # 准备模型数据
        target_col = "charging_stations_功率(kw)"
        X_train, X_test, y_train, y_test = data_interface.prepare_model_data(
            target_col=target_col,
            test_size=0.2
        )
        print(f"模型数据准备完成，训练集{len(X_train)}条，测试集{len(X_test)}条")
    except Exception as e:
        print(f"数据融合与特征工程失败: {e}")
        print("使用示例数据进行演示...")
        # 创建示例模型数据
        X_train, X_test, y_train, y_test, time_data, spatial_data = create_sample_model_data()
    
    # 3. 模型训练与评估
    print("\n3. 模型训练与评估")
    print("-" * 50)
    
    # 创建Prophet-GTWR耦合模型
    model = ProphetGTWRCoupling(
        prophet_params={
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        gtwr_params={
            'kernel_function': 'gaussian',
            'optimization_method': 'cv',
            'time_decay_lambda': 0.5
        },
        coupling_params={
            'alpha': 0.5,
            'feedback_mode': 'two_way',
            'dynamic_weight': True
        }
    )
    
    # 训练模型
    try:
        model.fit(
            time_series_data=time_data,
            spatial_data=spatial_data,
            coords_cols=['经度', '纬度'],
            time_col='记录时间',
            target_col='功率(kw)',
            feature_cols=['人口密度', '交通流量', 'POI密度']
        )
        
        # 模型评估
        print("\n模型评估结果:")
        for metric, value in model.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 绘制预测结果
        fig = model.plot_predictions()
        plt.savefig(os.path.join(data_path, "项目产出_results/figures/prophet_gtwr_predictions.png"))
        print("预测结果图已保存")
        
        # 绘制空间预测分布图
        fig = model.plot_spatial_prediction(
            prediction_data=spatial_data,
            coords_cols=['经度', '纬度'],
            value_col='功率(kw)'
        )
        plt.savefig(os.path.join(data_path, "项目产出_results/figures/spatial_prediction.png"))
        print("空间预测分布图已保存")
    except Exception as e:
        print(f"模型训练与评估失败: {e}")
    
    # 4. 充电桩布局优化
    print("\n4. 充电桩布局优化")
    print("-" * 50)
    
    try:
        # 优化充电桩布局
        layout_plan = model.optimize_layout(
            grid_data=spatial_data,
            objective='coverage',
            n_stations=20
        )
        
        print(f"布局优化完成，推荐在{len(layout_plan)}个位置布局充电桩")
        print("推荐布局位置的前5个:")
        print(layout_plan[['经度', '纬度', 'priority', 'gap']].head())
        
        # 绘制布局方案图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制所有网格
        spatial_data.plot(x='经度', y='纬度', kind='scatter', 
                         c='功率(kw)', cmap='viridis', alpha=0.5, ax=ax)
        
        # 高亮显示推荐布局位置
        layout_plan.plot(x='经度', y='纬度', kind='scatter', 
                        color='red', s=100, ax=ax, label='推荐布局位置')
        
        ax.set_title('充电桩布局优化方案')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.legend()
        
        plt.savefig(os.path.join(data_path, "项目产出_results/figures/layout_optimization.png"))
        print("布局优化方案图已保存")
    except Exception as e:
        print(f"充电桩布局优化失败: {e}")
    
    # 5. 模型保存
    print("\n5. 模型保存")
    print("-" * 50)
    
    try:
        # 创建模型保存目录
        model_dir = os.path.join(data_path, "项目产出_results/models/prophet_gtwr")
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存模型
        model.save_model(model_dir)
        print(f"模型已保存到: {model_dir}")
    except Exception as e:
        print(f"模型保存失败: {e}")
    
    print("\n" + "=" * 80)
    print("Prophet-GTWR耦合模型演示完成")
    print("=" * 80)

def create_sample_data(data_interface):
    """
    创建示例数据
    
    Args:
        data_interface: 数据接口对象
    """
    # 创建充电桩数据
    n_stations = 100
    np.random.seed(42)
    
    # 深圳市经纬度范围
    lon_range = [113.8, 114.5]
    lat_range = [22.4, 22.8]
    
    # 生成随机位置
    lons = np.random.uniform(lon_range[0], lon_range[1], n_stations)
    lats = np.random.uniform(lat_range[0], lat_range[1], n_stations)
    
    # 生成时间序列
    dates = pd.date_range(start='2023-01-01', end='2024-04-01', freq='D')
    
    # 生成充电桩数据
    charging_stations = []
    for i in range(n_stations):
        # 每个充电桩有多个时间点的记录
        n_records = np.random.randint(10, 100)
        record_dates = np.random.choice(dates, n_records, replace=False)
        
        for date in sorted(record_dates):
            # 生成功率数据（带有时间模式）
            base_power = np.random.uniform(30, 100)
            time_effect = 10 * np.sin(2 * np.pi * date.dayofweek / 7)  # 周模式
            seasonal_effect = 20 * np.sin(2 * np.pi * date.dayofyear / 365)  # 年模式
            random_effect = np.random.normal(0, 5)
            
            power = base_power + time_effect + seasonal_effect + random_effect
            
            charging_stations.append({
                'ID': f'CS{i:03d}',
                '名称': f'示例充电站{i}',
                '地址': f'深圳市示例地址{i}',
                '经度': lons[i],
                '纬度': lats[i],
                '记录时间': date,
                '功率(kw)': max(0, power),
                '充电类型': np.random.choice(['快充', '慢充']),
                '运营商': np.random.choice(['A公司', 'B公司', 'C公司']),
                '评分': np.random.uniform(3, 5),
                '位置类型': np.random.choice(['商场', '社区', '办公区', '高速服务区', '公共停车场'])
            })
    
    # 创建DataFrame
    charging_df = pd.DataFrame(charging_stations)
    
    # 生成交通流量数据
    n_traffic_points = 50
    traffic_lons = np.random.uniform(lon_range[0], lon_range[1], n_traffic_points)
    traffic_lats = np.random.uniform(lat_range[0], lat_range[1], n_traffic_points)
    
    traffic_data = []
    for i in range(n_traffic_points):
        # 每个点有多个时间点的记录
        n_records = np.random.randint(30, 100)
        record_dates = np.random.choice(dates, n_records, replace=False)
        
        for date in sorted(record_dates):
            # 生成交通流量数据（带有时间模式）
            base_flow = np.random.uniform(100, 1000)
            time_effect = 300 * np.sin(2 * np.pi * date.hour / 24)  # 日模式
            weekday_effect = 200 if date.dayofweek < 5 else -100  # 工作日vs周末
            random_effect = np.random.normal(0, 50)
            
            flow = base_flow + time_effect + weekday_effect + random_effect
            
            traffic_data.append({
                '道路ID': f'R{i:03d}',
                '道路名称': f'示例道路{i}',
                '经度': traffic_lons[i],
                '纬度': traffic_lats[i],
                '记录时间': date,
                '交通流量': max(0, flow),
                '平均车速': np.random.uniform(20, 80)
            })
    
    # 创建DataFrame
    traffic_df = pd.DataFrame(traffic_data)
    
    # 生成人口热力图数据
    n_population_points = 200
    pop_lons = np.random.uniform(lon_range[0], lon_range[1], n_population_points)
    pop_lats = np.random.uniform(lat_range[0], lat_range[1], n_population_points)
    
    population_data = []
    for i in range(n_population_points):
        # 生成人口密度数据
        base_density = np.random.uniform(1000, 10000)
        spatial_effect = 2000 * np.sin(np.pi * (pop_lons[i] - lon_range[0]) / (lon_range[1] - lon_range[0]))
        random_effect = np.random.normal(0, 500)
        
        density = base_density + spatial_effect + random_effect
        
        population_data.append({
            '网格ID': f'G{i:03d}',
            '街道名称': f'示例街道{i}',
            '经度': pop_lons[i],
            '纬度': pop_lats[i],
            '人口密度': max(0, density),
            '面积': np.random.uniform(0.5, 2.0)
        })
    
    # 创建DataFrame
    population_df = pd.DataFrame(population_data)
    
    # 将数据添加到数据接口
    data_interface.processed_data["charging_stations"] = charging_df
    data_interface.processed_data["traffic_flow"] = traffic_df
    data_interface.processed_data["population_heatmap"] = population_df
    
    print("示例数据创建完成:")
    print(f"充电桩数据: {len(charging_df)}条记录")
    print(f"交通流量数据: {len(traffic_df)}条记录")
    print(f"人口热力图数据: {len(population_df)}条记录")

def create_sample_model_data():
    """
    创建示例模型数据
    
    Returns:
        X_train, X_test, y_train, y_test, time_data, spatial_data
    """
    # 创建时间序列数据
    dates = pd.date_range(start='2023-01-01', end='2024-04-01', freq='D')
    values = np.sin(np.arange(len(dates)) * 0.1) * 10 + np.random.normal(0, 1, len(dates)) + 50
    
    time_data = pd.DataFrame({
        '记录时间': dates,
        '功率(kw)': values
    })
    
    # 创建空间数据
    n_samples = 100
    np.random.seed(42)
    
    # 空间坐标（经纬度）
    coords = np.random.uniform(low=[113.8, 22.4], high=[114.5, 22.8], size=(n_samples, 2))
    
    # 时间（随机选择时间序列中的时间点）
    times_idx = np.random.choice(len(dates), n_samples)
    times = dates[times_idx]
    
    # 特征
    population = np.random.normal(5000, 1000, n_samples)  # 人口密度
    traffic = np.random.normal(500, 100, n_samples)      # 交通流量
    poi = np.random.normal(50, 10, n_samples)           # POI密度
    
    # 目标变量（带空间效应）
    y = 30 + 0.001 * population + 0.01 * traffic + 0.1 * poi + \
        5 * np.sin(2 * np.pi * (coords[:, 0] - 113.8) / 0.7) + \
        3 * np.cos(2 * np.pi * (coords[:, 1] - 22.4) / 0.4) + \
        np.random.normal(0, 2, n_samples)
    
    # 创建空间数据DataFrame
    spatial_data = pd.DataFrame({
        '经度': coords[:, 0],
        '纬度': coords[:, 1],
        '记录时间': times,
        '人口密度': population,
        '交通流量': traffic,
        'POI密度': poi,
        '功率(kw)': y
    })
    
    # 分割训练集和测试集
    from sklearn.model_selection import train_test_split
    
    X = spatial_data[['人口密度', '交通流量', 'POI密度']]
    y = spatial_data['功率(kw)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("示例模型数据创建完成:")
    print(f"时间序列数据: {len(time_data)}条记录")
    print(f"空间数据: {len(spatial_data)}条记录")
    print(f"训练集: {len(X_train)}条记录")
    print(f"测试集: {len(X_test)}条记录")
    
    return X_train, X_test, y_train, y_test, time_data, spatial_data

if __name__ == "__main__":
    main()