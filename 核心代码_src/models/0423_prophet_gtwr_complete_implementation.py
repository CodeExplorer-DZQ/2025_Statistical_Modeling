# -*- coding: utf-8 -*-
"""
日期：2023年4月23日
功能：Prophet-GTWR耦合模型完整实现
包含：数据加载、模型训练、预测和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from .prophet_gtwr_coupling import ProphetGTWRCoupling
from .prophet_model import ProphetModel
from .gtwr_model import GTWRModel

# 设置中文字体，防止乱码
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，可能会导致中文显示为方块")

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上两级目录，然后进入数据目录
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '数据_data'))
# 向上两级目录，然后进入结果目录
result_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '项目产出_results'))
figures_dir = os.path.join(result_dir, 'figures')
models_dir = os.path.join(result_dir, 'models')

# 确保结果目录存在
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

def load_charging_station_data():
    """
    加载充电桩数据
    """
    # 充电桩数据路径
    charging_file = os.path.join(data_dir, '1_充电桩数据', 'charging_stations_data.csv')
    
    # 如果文件不存在，创建模拟数据
    if not os.path.exists(charging_file):
        print(f"警告: 充电桩数据文件 {charging_file} 不存在，使用模拟数据")
        
        # 创建日期范围
        date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # 创建模拟数据
        data = []
        for date in date_range:
            # 基础充电量
            base_charging = 1000
            # 添加趋势（逐月增长）
            trend = date.month * 50
            # 添加周内模式（周末充电量更高）
            weekday_effect = 200 if date.weekday() >= 5 else 0
            # 添加季节性（夏季充电量更高）
            month = date.month
            seasonal_effect = 300 if 6 <= month <= 8 else (150 if 3 <= month <= 5 or 9 <= month <= 11 else 0)
            # 添加随机噪声
            noise = np.random.normal(0, 50)
            
            # 计算总充电量
            charging_amount = base_charging + trend + weekday_effect + seasonal_effect + noise
            
            data.append({
                'date': date,
                'charging_amount': max(0, charging_amount),  # 确保充电量非负
                'ev_count': base_charging + trend * 0.5,  # 电动汽车数量
                'district': np.random.choice(['福田区', '罗湖区', '南山区', '宝安区', '龙岗区', '龙华区', '盐田区', '坪山区', '光明区', '大鹏新区', '深汕特别合作区']),
                'longitude': 114.0 + np.random.uniform(0, 0.5),  # 深圳经度范围
                'latitude': 22.5 + np.random.uniform(0, 0.3),   # 深圳纬度范围
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到文件
        os.makedirs(os.path.dirname(charging_file), exist_ok=True)
        df.to_csv(charging_file, index=False)
    else:
        # 读取数据
        df = pd.read_csv(charging_file)
        # 确保日期列是datetime类型
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
    
    return df

def load_traffic_data():
    """
    加载交通流量数据
    """
    # 交通流量数据路径
    traffic_file = os.path.join(data_dir, '2_时空动态数据', '0422_深圳区域交通流量数据.csv')
    
    # 读取数据
    if os.path.exists(traffic_file):
        df = pd.read_csv(traffic_file)
        # 确保日期列是datetime类型
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print(f"警告: 交通流量数据文件 {traffic_file} 不存在")
        return None

def load_energy_data():
    """
    加载能源消耗数据
    """
    # 能源消耗数据路径
    energy_file = os.path.join(data_dir, '3_环境与能源数据', '0423_深圳区域能源消耗数据.csv')
    
    # 读取数据
    if os.path.exists(energy_file):
        df = pd.read_csv(energy_file)
        # 创建日期列
        df['date'] = pd.to_datetime(df[['年份', '月份']].assign(day=1).apply(
            lambda x: f"{int(x['年份'])}-{int(x['月份'])}-{x['day']}", axis=1
        ))
        return df
    else:
        print(f"警告: 能源消耗数据文件 {energy_file} 不存在")
        return None

def load_population_data():
    """
    加载人口数据
    """
    # 人口数据路径
    population_file = os.path.join(data_dir, '2_时空动态数据', '0421_区级_深圳人口数据.csv')
    
    # 读取数据
    if os.path.exists(population_file):
        df = pd.read_csv(population_file)
        return df
    else:
        print(f"警告: 人口数据文件 {population_file} 不存在")
        return None

def prepare_model_data():
    """
    准备模型输入数据，融合多源异构数据
    """
    print("准备模型输入数据...")
    
    # 加载数据
    charging_data = load_charging_station_data()
    traffic_data = load_traffic_data()
    energy_data = load_energy_data()
    population_data = load_population_data()
    
    # 数据融合
    # 1. 准备时间序列数据（用于Prophet模型）
    time_series_data = charging_data.groupby('date')['charging_amount'].sum().reset_index()
    time_series_data.rename(columns={'charging_amount': 'charging_demand'}, inplace=True)
    
    # 添加电动汽车保有量
    if energy_data is not None:
        # 将月度数据转换为日度数据（线性插值）
        ev_count_monthly = energy_data.groupby('date')['电动汽车保有量(辆)'].mean().reset_index()
        ev_count_monthly.set_index('date', inplace=True)
        ev_count_daily = ev_count_monthly.resample('D').interpolate(method='linear')
        ev_count_daily.reset_index(inplace=True)
        
        # 合并到时间序列数据
        time_series_data = pd.merge(time_series_data, ev_count_daily, on='date', how='left')
    
    # 2. 准备空间数据（用于GTWR模型）
    spatial_data = charging_data.copy()
    
    # 添加交通流量数据
    if traffic_data is not None:
        # 假设交通数据有区域和日期信息
        if 'district' in traffic_data.columns and 'date' in traffic_data.columns:
            traffic_features = traffic_data.groupby(['district', 'date']).agg({
                '交通流量(辆/日)': 'mean',
                '拥堵指数': 'mean'
            }).reset_index()
            
            # 合并到空间数据
            spatial_data = pd.merge(
                spatial_data,
                traffic_features,
                on=['district', 'date'],
                how='left'
            )
    
    # 添加能源消耗数据
    if energy_data is not None:
        # 将月度数据转换为日度数据
        energy_features = energy_data.copy()
        
        # 获取每个区域每月的数据
        energy_monthly = energy_features.groupby(['区域', 'date']).agg({
            '总用电量(万千瓦时)': 'mean',
            '充电桩用电量(万千瓦时)': 'mean',
            '电动汽车保有量(辆)': 'mean'
        }).reset_index()
        
        # 重命名列
        energy_monthly.rename(columns={
            '区域': 'district',
            '总用电量(万千瓦时)': 'total_electricity',
            '充电桩用电量(万千瓦时)': 'charging_electricity',
            '电动汽车保有量(辆)': 'ev_count'
        }, inplace=True)
        
        # 为每个区域的每一天分配月度数据
        energy_daily = []
        for _, row in energy_monthly.iterrows():
            month_start = row['date']
            month_end = month_start + pd.offsets.MonthEnd(0)
            days = pd.date_range(month_start, month_end, freq='D')
            
            for day in days:
                energy_daily.append({
                    'district': row['district'],
                    'date': day,
                    'total_electricity': row['total_electricity'] / len(days),
                    'charging_electricity': row['charging_electricity'] / len(days),
                    'ev_count': row['ev_count']
                })
        
        energy_daily_df = pd.DataFrame(energy_daily)
        
        # 合并到空间数据
        spatial_data = pd.merge(
            spatial_data,
            energy_daily_df,
            on=['district', 'date'],
            how='left'
        )
    
    # 添加人口数据
    if population_data is not None:
        # 假设人口数据有区域信息
        if '区域' in population_data.columns and '人口数' in population_data.columns:
            population_features = population_data[['区域', '人口数']].copy()
            population_features.rename(columns={'区域': 'district'}, inplace=True)
            
            # 合并到空间数据
            spatial_data = pd.merge(
                spatial_data,
                population_features,
                on='district',
                how='left'
            )
    
    # 处理缺失值
    spatial_data.fillna({
        '交通流量(辆/日)': spatial_data['交通流量(辆/日)'].mean() if '交通流量(辆/日)' in spatial_data.columns else 0,
        '拥堵指数': spatial_data['拥堵指数'].mean() if '拥堵指数' in spatial_data.columns else 0,
        'total_electricity': spatial_data['total_electricity'].mean() if 'total_electricity' in spatial_data.columns else 0,
        'charging_electricity': spatial_data['charging_electricity'].mean() if 'charging_electricity' in spatial_data.columns else 0,
        'ev_count_x': spatial_data['ev_count_x'].mean() if 'ev_count_x' in spatial_data.columns else 0,
        'ev_count_y': spatial_data['ev_count_y'].mean() if 'ev_count_y' in spatial_data.columns else 0,
        '人口数': spatial_data['人口数'].mean() if '人口数' in spatial_data.columns else 0
    }, inplace=True)
    
    # 重命名可能重复的列
    if 'ev_count_x' in spatial_data.columns and 'ev_count_y' in spatial_data.columns:
        spatial_data['ev_count'] = spatial_data['ev_count_x']  # 使用充电桩数据中的电动汽车数量
        spatial_data.drop(['ev_count_x', 'ev_count_y'], axis=1, inplace=True)
    
    print("数据准备完成")
    print(f"时间序列数据形状: {time_series_data.shape}")
    print(f"空间数据形状: {spatial_data.shape}")
    
    return time_series_data, spatial_data

def train_prophet_gtwr_model():
    """
    训练Prophet-GTWR耦合模型
    """
    print("开始训练Prophet-GTWR耦合模型...")
    
    # 准备数据
    time_series_data, spatial_data = prepare_model_data()
    
    # 设置模型参数
    prophet_params = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'seasonality_mode': 'multiplicative',
        'changepoint_prior_scale': 0.05
    }
    
    gtwr_params = {
        'kernel_function': 'gaussian',
        'optimization_method': 'cv',
        'time_decay_lambda': 0.5
    }
    
    coupling_params = {
        'alpha': 0.5,  # 初始融合权重
        'feedback_mode': 'two_way',  # 双向反馈
        'dynamic_weight': True  # 使用动态权重
    }
    
    # 初始化耦合模型
    model = ProphetGTWRCoupling(
        prophet_params=prophet_params,
        gtwr_params=gtwr_params,
        coupling_params=coupling_params
    )
    
    # 训练模型
    coords_cols = ['longitude', 'latitude']
    time_col = 'date'
    target_col = 'charging_amount'
    
    # 确定特征列
    feature_cols = [
        'ev_count',
        '交通流量(辆/日)' if '交通流量(辆/日)' in spatial_data.columns else None,
        '拥堵指数' if '拥堵指数' in spatial_data.columns else None,
        'total_electricity' if 'total_electricity' in spatial_data.columns else None,
        'charging_electricity' if 'charging_electricity' in spatial_data.columns else None,
        '人口数' if '人口数' in spatial_data.columns else None
    ]
    feature_cols = [col for col in feature_cols if col is not None]
    
    # 训练模型
    model.fit(
        time_series_data=time_series_data,
        spatial_data=spatial_data,
        coords_cols=coords_cols,
        time_col=time_col,
        target_col=target_col,
        feature_cols=feature_cols
    )
    
    # 保存模型
    model_path = os.path.join(models_dir, 'prophet_gtwr_model.pkl')
    try:
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {model_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")
    
    return model

def predict_and_visualize(model, periods=30):
    """
    使用训练好的模型进行预测并可视化结果
    """
    print("生成预测结果并可视化...")
    
    # 生成预测
    forecast = model.predict(periods=periods)
    
    # 可视化时间序列预测
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['ds'], forecast['yhat'], label='预测值', color='blue')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='blue', alpha=0.2, label='95%置信区间')
    if 'y' in forecast.columns:
        plt.plot(forecast['ds'], forecast['y'], 'k.', label='实际值')
    plt.title('充电需求预测')
    plt.xlabel('日期')
    plt.ylabel('充电量')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    forecast_path = os.path.join(figures_dir, 'charging_demand_forecast.png')
    plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测图表已保存到: {forecast_path}")
    
    # 可视化空间预测（热力图）
    if hasattr(model, 'spatial_forecast') and model.spatial_forecast is not None:
        spatial_forecast = model.spatial_forecast
        
        # 创建地图可视化
        try:
            import folium
            from folium.plugins import HeatMap
            
            # 创建地图，中心点设为深圳
            m = folium.Map(location=[22.5431, 114.0579], zoom_start=11)
            
            # 准备热力图数据
            heat_data = []
            for _, row in spatial_forecast.iterrows():
                heat_data.append([row['latitude'], row['longitude'], row['predicted_demand']])
            
            # 添加热力图层
            HeatMap(heat_data).add_to(m)
            
            # 保存地图
            map_path = os.path.join(figures_dir, 'charging_demand_heatmap.html')
            m.save(map_path)
            print(f"空间预测热力图已保存到: {map_path}")
        except ImportError:
            print("警告: 无法导入folium库，跳过地图可视化")
    
    # 可视化模型组件
    if hasattr(model, 'prophet_model') and model.prophet_model.components is not None:
        components = model.prophet_model.components
        
        plt.figure(figsize=(15, 10))
        
        # 趋势组件
        plt.subplot(3, 1, 1)
        plt.plot(components['ds'], components['trend'])
        plt.title('趋势组件')
        plt.grid(True)
        
        # 年度季节性
        if 'yearly' in components.columns:
            plt.subplot(3, 1, 2)
            yearly_df = components[['ds', 'yearly']].copy()
            yearly_df['month'] = yearly_df['ds'].dt.month
            yearly_avg = yearly_df.groupby('month')['yearly'].mean().reset_index()
            plt.plot(yearly_avg['month'], yearly_avg['yearly'])
            plt.title('年度季节性')
            plt.xticks(range(1, 13), ['一月', '二月', '三月', '四月', '五月', '六月', 
                                      '七月', '八月', '九月', '十月', '十一月', '十二月'])
            plt.grid(True)
        
        # 周度季节性
        if 'weekly' in components.columns:
            plt.subplot(3, 1, 3)
            weekly_df = components[['ds', 'weekly']].copy()
            weekly_df['weekday'] = weekly_df['ds'].dt.weekday
            weekly_avg = weekly_df.groupby('weekday')['weekly'].mean().reset_index()
            plt.plot(weekly_avg['weekday'], weekly_avg['weekly'])
            plt.title('周度季节性')
            plt.xticks(range(7), ['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
            plt.grid(True)
        
        plt.tight_layout()
        
        # 保存组件图表
        components_path = os.path.join(figures_dir, 'model_components.png')
        plt.savefig(components_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"模型组件图表已保存到: {components_path}")
    
    # 可视化GTWR系数空间分布
    if hasattr(model, 'gtwr_model') and hasattr(model.gtwr_model, 'local_coefficients'):
        try:
            local_coefs = model.gtwr_model.local_coefficients()
            
            # 为每个特征创建系数空间分布图
            for feature in local_coefs.columns:
                if feature in ['intercept', 'coords_x', 'coords_y', 'time']:
                    continue
                
                plt.figure(figsize=(10, 8))
                
                # 创建散点图，颜色表示系数大小
                scatter = plt.scatter(
                    local_coefs['coords_x'], 
                    local_coefs['coords_y'],
                    c=local_coefs[feature],
                    cmap='viridis',
                    alpha=0.8,
                    s=50
                )
                
                plt.colorbar(scatter, label=f'{feature}系数')
                plt.title(f'{feature}的空间异质性')
                plt.xlabel('经度')
                plt.ylabel('纬度')
                plt.grid(True)
                
                # 保存图表
                coef_path = os.path.join(figures_dir, f'gtwr_coefficient_{feature}.png')
                plt.savefig(coef_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"{feature}系数空间分布图已保存到: {coef_path}")
        except Exception as e:
            print(f"可视化GTWR系数时出错: {e}")
    
    return forecast



def main():
    """
    主函数
    """
    print("启动Prophet-GTWR耦合模型完整实现流程...")
    
    # 训练模型
    model = train_prophet_gtwr_model()
    
    # 预测和可视化
    forecast = predict_and_visualize(model, periods=30)
    
    print("Prophet-GTWR耦合模型完整实现流程完成")
    
    return model, forecast

if __name__ == "__main__":
    main()