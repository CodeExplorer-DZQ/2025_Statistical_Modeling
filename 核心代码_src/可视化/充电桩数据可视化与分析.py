# -*- coding: utf-8 -*-
"""
充电桩数据可视化与分析

此脚本用于对UrbanEV_data数据集进行可视化分析，包括：
1. 充电桩使用量时间序列分析
2. 充电桩占用率热力图
3. 充电桩使用与天气关系分析
4. 基于Prophet-GTWR耦合模型的预测结果可视化

日期：2025年4月23日
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from tqdm import tqdm

# 设置中文字体
try:
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    print("警告：未找到中文字体，图表中文可能显示为方块")

# 数据路径
DATA_PATH = r'd:\DZQ_Projects_项目合集\Competitions\2025_Statistical_Modeling_competition\EV_item_for_Trae\数据_data\1_充电桩数据\processed\UrbanEV_data'
RESULT_PATH = r'd:\DZQ_Projects_项目合集\Competitions\2025_Statistical_Modeling_competition\EV_item_for_Trae\项目产出_results\figures'

# 确保结果目录存在
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# 创建Prophet-GTWR可视化子目录
prophet_gtwr_dir = os.path.join(RESULT_PATH, 'Prophet_GTWR_可视化')
if not os.path.exists(prophet_gtwr_dir):
    os.makedirs(prophet_gtwr_dir)


def load_data():
    """
    加载所有相关数据文件
    """
    print("正在加载数据...")
    data = {}
    
    # 加载充电量数据
    data['volume'] = pd.read_csv(os.path.join(DATA_PATH, 'volume.csv'))
    data['volume']['time'] = pd.to_datetime(data['volume']['time'])
    
    # 加载占用率数据
    data['occupancy'] = pd.read_csv(os.path.join(DATA_PATH, 'occupancy.csv'))
    data['occupancy']['time'] = pd.to_datetime(data['occupancy']['time'])
    
    # 加载天气数据
    data['weather'] = pd.read_csv(os.path.join(DATA_PATH, 'weather_airport.csv'))
    data['weather']['time'] = pd.to_datetime(data['weather']['time'])
    
    # 加载POI数据
    try:
        data['poi'] = pd.read_csv(os.path.join(DATA_PATH, 'poi.csv'))
    except:
        print("警告：未找到POI数据文件")
    
    print("数据加载完成！")
    return data


def plot_charging_volume_time_series(data, station_ids=None, save_path=None):
    """
    绘制充电桩使用量的时间序列图
    
    参数:
        data: 包含volume数据的字典
        station_ids: 要绘制的充电站ID列表，如果为None则选择使用量最高的5个站点
        save_path: 保存路径，如果为None则显示图表
    """
    volume_data = data['volume']
    
    if station_ids is None:
        # 计算每个站点的总使用量
        station_totals = volume_data.iloc[:, 1:].sum().sort_values(ascending=False)
        station_ids = station_totals.index[:5].tolist()
    
    plt.figure(figsize=(15, 8))
    
    for station_id in station_ids:
        plt.plot(volume_data['time'], volume_data[station_id], label=f'充电站 {station_id}')
    
    plt.title('充电站使用量时间序列分析', fontproperties=font, fontsize=16)
    plt.xlabel('时间', fontproperties=font, fontsize=14)
    plt.ylabel('充电量 (kWh)', fontproperties=font, fontsize=14)
    plt.legend(prop=font)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_occupancy_heatmap(data, time_period=None, save_path=None):
    """
    绘制充电桩占用率热力图
    
    参数:
        data: 包含occupancy数据的字典
        time_period: 时间段，格式为(start_date, end_date)，如果为None则使用全部数据
        save_path: 保存路径，如果为None则显示图表
    """
    occupancy_data = data['occupancy']
    
    if time_period:
        start_date, end_date = time_period
        occupancy_data = occupancy_data[(occupancy_data['time'] >= start_date) & 
                                       (occupancy_data['time'] <= end_date)]
    
    # 计算每小时的平均占用率
    occupancy_data['hour'] = occupancy_data['time'].dt.hour
    occupancy_data['day_of_week'] = occupancy_data['time'].dt.dayofweek
    
    # 选择使用量最高的20个站点
    station_totals = occupancy_data.iloc[:, 1:-2].sum().sort_values(ascending=False)
    top_stations = station_totals.index[:20].tolist()
    
    # 创建每小时每天的平均占用率数据框
    weekly_hourly_avg = pd.DataFrame()
    
    for station in top_stations:
        station_data = occupancy_data.pivot_table(
            values=station, 
            index='hour', 
            columns='day_of_week', 
            aggfunc='mean'
        )
        weekly_hourly_avg = pd.concat([weekly_hourly_avg, station_data])
    
    # 绘制热力图
    plt.figure(figsize=(12, 10))
    
    # 自定义颜色映射
    colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    cmap = LinearSegmentedColormap.from_list('custom_blue', colors)
    
    ax = sns.heatmap(
        weekly_hourly_avg, 
        cmap=cmap,
        linewidths=0.5, 
        linecolor='white',
        cbar_kws={'label': '平均占用率'}
    )
    
    # 设置坐标轴标签
    day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    ax.set_xticklabels(day_names, fontproperties=font, fontsize=12)
    ax.set_yticklabels(range(24), fontsize=12)
    
    plt.title('充电桩每周每小时平均占用率热力图', fontproperties=font, fontsize=16)
    plt.xlabel('星期', fontproperties=font, fontsize=14)
    plt.ylabel('小时', fontproperties=font, fontsize=14)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('平均占用率', fontproperties=font, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_weather_charging_correlation(data, save_path=None):
    """
    分析天气与充电量的关系
    
    参数:
        data: 包含volume和weather数据的字典
        save_path: 保存路径，如果为None则显示图表
    """
    volume_data = data['volume']
    weather_data = data['weather']
    
    # 合并数据
    merged_data = pd.merge(volume_data, weather_data, on='time')
    
    # 计算每天的总充电量
    merged_data['date'] = merged_data['time'].dt.date
    merged_data['total_volume'] = merged_data.iloc[:, 1:len(volume_data.columns)].sum(axis=1)
    
    # 按日期分组计算平均值
    daily_data = merged_data.groupby('date').agg({
        'total_volume': 'mean',
        'T': 'mean',  # 温度
        'P': 'mean',  # 气压
        'U': 'mean',  # 湿度
        'nRAIN': 'sum'  # 降雨量
    }).reset_index()
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 温度与充电量关系
    sns.scatterplot(x='T', y='total_volume', data=daily_data, ax=axes[0, 0], alpha=0.7)
    axes[0, 0].set_title('温度与充电量关系', fontproperties=font, fontsize=14)
    axes[0, 0].set_xlabel('温度 (°C)', fontproperties=font, fontsize=12)
    axes[0, 0].set_ylabel('平均充电量', fontproperties=font, fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 湿度与充电量关系
    sns.scatterplot(x='U', y='total_volume', data=daily_data, ax=axes[0, 1], alpha=0.7)
    axes[0, 1].set_title('湿度与充电量关系', fontproperties=font, fontsize=14)
    axes[0, 1].set_xlabel('湿度 (%)', fontproperties=font, fontsize=12)
    axes[0, 1].set_ylabel('平均充电量', fontproperties=font, fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # 气压与充电量关系
    sns.scatterplot(x='P', y='total_volume', data=daily_data, ax=axes[1, 0], alpha=0.7)
    axes[1, 0].set_title('气压与充电量关系', fontproperties=font, fontsize=14)
    axes[1, 0].set_xlabel('气压 (hPa)', fontproperties=font, fontsize=12)
    axes[1, 0].set_ylabel('平均充电量', fontproperties=font, fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # 降雨量与充电量关系
    rain_data = daily_data.copy()
    rain_data['has_rain'] = rain_data['nRAIN'] > 0
    sns.boxplot(x='has_rain', y='total_volume', data=rain_data, ax=axes[1, 1])
    axes[1, 1].set_title('降雨与充电量关系', fontproperties=font, fontsize=14)
    axes[1, 1].set_xlabel('是否有雨', fontproperties=font, fontsize=12)
    axes[1, 1].set_xticklabels(['无雨', '有雨'], fontproperties=font)
    axes[1, 1].set_ylabel('平均充电量', fontproperties=font, fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('天气因素与充电量关系分析', fontproperties=font, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_charging_patterns(data, save_path=None):
    """
    分析充电模式（工作日vs周末，高峰vs低谷）
    
    参数:
        data: 包含volume数据的字典
        save_path: 保存路径，如果为None则显示图表
    """
    volume_data = data['volume']
    
    # 添加时间特征
    volume_data['hour'] = volume_data['time'].dt.hour
    volume_data['day_of_week'] = volume_data['time'].dt.dayofweek
    volume_data['is_weekend'] = volume_data['day_of_week'] >= 5
    
    # 计算总充电量
    volume_data['total_volume'] = volume_data.iloc[:, 1:len(volume_data.columns)-3].sum(axis=1)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 工作日vs周末的小时充电模式
    hourly_pattern = volume_data.groupby(['hour', 'is_weekend'])['total_volume'].mean().unstack()
    hourly_pattern.columns = ['工作日', '周末']
    
    hourly_pattern.plot(ax=ax1, marker='o')
    ax1.set_title('工作日vs周末充电模式', fontproperties=font, fontsize=16)
    ax1.set_xlabel('小时', fontproperties=font, fontsize=14)
    ax1.set_ylabel('平均充电量', fontproperties=font, fontsize=14)
    ax1.set_xticks(range(24))
    ax1.legend(prop=font)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 每周每天的充电模式
    daily_pattern = volume_data.groupby('day_of_week')['total_volume'].mean()
    days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    
    ax2.bar(days, daily_pattern.values, color=sns.color_palette('Blues_d', 7))
    ax2.set_title('每周每天充电模式', fontproperties=font, fontsize=16)
    ax2.set_xlabel('星期', fontproperties=font, fontsize=14)
    ax2.set_ylabel('平均充电量', fontproperties=font, fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.suptitle('充电模式分析', fontproperties=font, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    主函数，运行所有可视化分析
    """
    # 加载数据
    data = load_data()
    
    # 创建可视化
    print("\n1. 生成充电站使用量时间序列分析图...")
    plot_charging_volume_time_series(
        data, 
        save_path=os.path.join(prophet_gtwr_dir, '充电站使用量时间序列分析.png')
    )
    
    print("\n2. 生成充电桩占用率热力图...")
    plot_occupancy_heatmap(
        data, 
        save_path=os.path.join(prophet_gtwr_dir, '充电桩占用率热力图.png')
    )
    
    print("\n3. 生成天气与充电量关系分析图...")
    plot_weather_charging_correlation(
        data, 
        save_path=os.path.join(prophet_gtwr_dir, '天气与充电量关系分析.png')
    )
    
    print("\n4. 生成充电模式分析图...")
    plot_charging_patterns(
        data, 
        save_path=os.path.join(prophet_gtwr_dir, '充电模式分析.png')
    )
    
    print("\n所有可视化已完成！")


if __name__ == "__main__":
    main()