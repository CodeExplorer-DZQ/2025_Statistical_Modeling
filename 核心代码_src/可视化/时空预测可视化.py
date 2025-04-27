# -*- coding: utf-8 -*-
"""
时空预测可视化

此脚本用于可视化Prophet-GTWR耦合模型的时空预测结果，包括：
1. 时间序列预测结果可视化
2. 空间异质性分析可视化
3. 预测误差空间分布可视化
4. 时空耦合效应可视化

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
import matplotlib.dates as mdates

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
    
    print("数据加载完成！")
    return data


def simulate_prophet_predictions(data, station_id, periods=168):
    """
    模拟Prophet模型的预测结果
    在实际应用中，这里应该调用真实的Prophet模型预测结果
    
    参数:
        data: 包含volume数据的字典
        station_id: 充电站ID
        periods: 预测未来的小时数
    
    返回:
        包含预测结果的DataFrame
    """
    volume_data = data['volume']
    
    # 提取指定充电站的数据
    station_data = volume_data[['time', station_id]].copy()
    station_data.columns = ['ds', 'y']
    
    # 获取最后一个时间点
    last_date = station_data['ds'].max()
    
    # 创建未来时间点
    future_dates = [last_date + timedelta(hours=i) for i in range(1, periods+1)]
    future_df = pd.DataFrame({'ds': future_dates})
    
    # 模拟预测结果（实际应用中应替换为真实预测）
    # 这里使用简单的时间序列模型模拟预测结果
    y_values = station_data['y'].values
    n = len(y_values)
    
    # 使用过去一周的数据模式作为预测基础
    week_pattern = y_values[max(0, n-168):n]
    if len(week_pattern) < 168:
        week_pattern = np.tile(week_pattern, 168 // len(week_pattern) + 1)[:168]
    
    # 添加一些随机波动
    np.random.seed(42)  # 设置随机种子以确保可重复性
    noise = np.random.normal(0, week_pattern.std() * 0.1, periods)
    
    # 生成预测值
    forecast_values = np.array([week_pattern[i % len(week_pattern)] for i in range(periods)]) + noise
    
    # 创建预测结果DataFrame
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': forecast_values * 0.9,
        'yhat_upper': forecast_values * 1.1
    })
    
    # 合并历史数据和预测结果
    result = pd.concat([station_data, forecast])
    
    return result


def plot_prophet_forecast(data, station_id, save_path=None):
    """
    绘制Prophet模型的预测结果
    
    参数:
        data: 包含volume数据的字典
        station_id: 充电站ID
        save_path: 保存路径，如果为None则显示图表
    """
    # 获取预测结果
    forecast = simulate_prophet_predictions(data, station_id)
    
    # 分割历史数据和预测数据
    history = forecast[forecast['yhat'].isna()]
    prediction = forecast[~forecast['yhat'].isna()]
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 绘制历史数据
    plt.plot(history['ds'], history['y'], 'b-', label='历史数据')
    
    # 绘制预测数据
    plt.plot(prediction['ds'], prediction['yhat'], 'r-', label='预测值')
    
    # 绘制预测区间
    plt.fill_between(prediction['ds'], prediction['yhat_lower'], prediction['yhat_upper'], 
                     color='r', alpha=0.2, label='预测区间')
    
    # 设置图表属性
    plt.title(f'充电站 {station_id} 使用量预测 (Prophet模型)', fontproperties=font, fontsize=16)
    plt.xlabel('时间', fontproperties=font, fontsize=14)
    plt.ylabel('充电量 (kWh)', fontproperties=font, fontsize=14)
    plt.legend(prop=font)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置x轴日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def simulate_gtwr_coefficients():
    """
    模拟GTWR模型的空间系数
    在实际应用中，这里应该使用真实的GTWR模型系数
    
    返回:
        包含模拟GTWR系数的DataFrame
    """
    # 模拟100个充电站的位置和GTWR系数
    np.random.seed(42)  # 设置随机种子以确保可重复性
    n_stations = 100
    
    # 模拟深圳市的经纬度范围
    lon_min, lon_max = 113.8, 114.5
    lat_min, lat_max = 22.4, 22.8
    
    # 生成随机位置
    longitudes = np.random.uniform(lon_min, lon_max, n_stations)
    latitudes = np.random.uniform(lat_min, lat_max, n_stations)
    
    # 生成模拟的GTWR系数
    # 假设有三个主要影响因素：交通流量、人口密度和商业活动
    traffic_coef = np.random.normal(0.5, 0.2, n_stations)  # 交通流量系数
    population_coef = np.random.normal(0.3, 0.15, n_stations)  # 人口密度系数
    commercial_coef = np.random.normal(0.4, 0.25, n_stations)  # 商业活动系数
    
    # 创建DataFrame
    gtwr_coef = pd.DataFrame({
        'station_id': [f'Station_{i}' for i in range(n_stations)],
        'longitude': longitudes,
        'latitude': latitudes,
        'traffic_coef': traffic_coef,
        'population_coef': population_coef,
        'commercial_coef': commercial_coef,
        'intercept': np.random.normal(10, 2, n_stations)  # 截距项
    })
    
    return gtwr_coef


def plot_gtwr_spatial_coefficients(save_path=None):
    """
    绘制GTWR模型的空间系数分布图
    
    参数:
        save_path: 保存路径，如果为None则显示图表
    """
    # 获取GTWR系数
    gtwr_coef = simulate_gtwr_coefficients()
    
    # 创建地图
    m = folium.Map(location=[22.6, 114.1], zoom_start=11, tiles='CartoDB positron')
    
    # 添加交通流量系数图层
    traffic_layer = folium.FeatureGroup(name='交通流量系数')
    for _, row in gtwr_coef.iterrows():
        color = 'blue' if row['traffic_coef'] > 0.5 else 'green' if row['traffic_coef'] > 0 else 'red'
        radius = abs(row['traffic_coef']) * 500  # 根据系数大小调整圆圈半径
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"站点: {row['station_id']}<br>交通流量系数: {row['traffic_coef']:.3f}"
        ).add_to(traffic_layer)
    traffic_layer.add_to(m)
    
    # 添加人口密度系数图层
    population_layer = folium.FeatureGroup(name='人口密度系数')
    for _, row in gtwr_coef.iterrows():
        color = 'purple' if row['population_coef'] > 0.3 else 'orange' if row['population_coef'] > 0 else 'red'
        radius = abs(row['population_coef']) * 500  # 根据系数大小调整圆圈半径
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"站点: {row['station_id']}<br>人口密度系数: {row['population_coef']:.3f}"
        ).add_to(population_layer)
    
    # 添加商业活动系数图层
    commercial_layer = folium.FeatureGroup(name='商业活动系数')
    for _, row in gtwr_coef.iterrows():
        color = 'darkgreen' if row['commercial_coef'] > 0.4 else 'lightgreen' if row['commercial_coef'] > 0 else 'red'
        radius = abs(row['commercial_coef']) * 500  # 根据系数大小调整圆圈半径
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"站点: {row['station_id']}<br>商业活动系数: {row['commercial_coef']:.3f}"
        ).add_to(commercial_layer)
    
    # 添加图层控制
    folium.LayerControl().add_to(m)
    
    # 保存地图
    if save_path:
        m.save(save_path)
        print(f"地图已保存至: {save_path}")
    else:
        return m


def plot_spatiotemporal_coupling_effect(data, save_path=None):
    """
    绘制时空耦合效应图
    
    参数:
        data: 包含volume数据的字典
        save_path: 保存路径，如果为None则显示图表
    """
    volume_data = data['volume']
    
    # 计算每个站点的总充电量
    station_totals = volume_data.iloc[:, 1:].sum().sort_values(ascending=False)
    top_stations = station_totals.index[:5].tolist()
    
    # 添加时间特征
    volume_data['hour'] = volume_data['time'].dt.hour
    volume_data['day_of_week'] = volume_data['time'].dt.dayofweek
    volume_data['is_weekend'] = volume_data['day_of_week'] >= 5
    
    # 创建图表
    fig, axes = plt.subplots(len(top_stations), 2, figsize=(18, 5*len(top_stations)))
    
    for i, station_id in enumerate(top_stations):
        # 工作日vs周末的小时充电模式
        hourly_pattern = volume_data.groupby(['hour', 'is_weekend'])[station_id].mean().unstack()
        hourly_pattern.columns = ['工作日', '周末']
        
        hourly_pattern.plot(ax=axes[i, 0], marker='o')
        axes[i, 0].set_title(f'充电站 {station_id} 工作日vs周末充电模式', fontproperties=font, fontsize=14)
        axes[i, 0].set_xlabel('小时', fontproperties=font, fontsize=12)
        axes[i, 0].set_ylabel('平均充电量', fontproperties=font, fontsize=12)
        axes[i, 0].set_xticks(range(24))
        axes[i, 0].legend(prop=font)
        axes[i, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 每周每天的充电模式
        daily_pattern = volume_data.groupby('day_of_week')[station_id].mean()
        days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        
        axes[i, 1].bar(days, daily_pattern.values, color=sns.color_palette('Blues_d', 7))
        axes[i, 1].set_title(f'充电站 {station_id} 每周每天充电模式', fontproperties=font, fontsize=14)
        axes[i, 1].set_xlabel('星期', fontproperties=font, fontsize=12)
        axes[i, 1].set_ylabel('平均充电量', fontproperties=font, fontsize=12)
        axes[i, 1].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.suptitle('充电站时空耦合效应分析', fontproperties=font, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_error_distribution(data, save_path=None):
    """
    绘制预测误差的空间分布图
    
    参数:
        data: 包含volume数据的字典
        save_path: 保存路径，如果为None则显示图表
    """
    # 模拟预测误差数据
    np.random.seed(42)  # 设置随机种子以确保可重复性
    n_stations = 100
    
    # 模拟深圳市的经纬度范围
    lon_min, lon_max = 113.8, 114.5
    lat_min, lat_max = 22.4, 22.8
    
    # 生成随机位置
    longitudes = np.random.uniform(lon_min, lon_max, n_stations)
    latitudes = np.random.uniform(lat_min, lat_max, n_stations)
    
    # 生成模拟的预测误差
    # 假设误差与位置有一定关系
    errors = np.zeros(n_stations)
    for i in range(n_stations):
        # 模拟城市中心区域误差较小，边缘区域误差较大的情况
        dist_to_center = np.sqrt((longitudes[i] - 114.1)**2 + (latitudes[i] - 22.6)**2)
        errors[i] = dist_to_center * 20 + np.random.normal(0, 5)
    
    # 创建DataFrame
    error_data = pd.DataFrame({
        'station_id': [f'Station_{i}' for i in range(n_stations)],
        'longitude': longitudes,
        'latitude': latitudes,
        'error': errors,
        'error_pct': errors / (100 + np.random.uniform(50, 150, n_stations)) * 100  # 百分比误差
    })
    
    # 创建地图
    m = folium.Map(location=[22.6, 114.1], zoom_start=11, tiles='CartoDB positron')
    
    # 添加误差图层
    for _, row in error_data.iterrows():
        # 根据误差大小确定颜色
        if row['error_pct'] < 5:
            color = 'green'
        elif row['error_pct'] < 10:
            color = 'blue'
        elif row['error_pct'] < 15:
            color = 'orange'
        else:
            color = 'red'
        
        # 根据误差大小调整圆圈半径
        radius = min(10 + row['error_pct'], 30)
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=f"站点: {row['station_id']}<br>预测误差: {row['error']:.2f}<br>误差百分比: {row['error_pct']:.2f}%"
        ).add_to(m)
    
    # 添加图例
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
    padding: 10px; border: 2px solid grey; border-radius: 5px">
    <p><b>预测误差百分比</b></p>
    <p><i class="fa fa-circle" style="color: green"></i> < 5%</p>
    <p><i class="fa fa-circle" style="color: blue"></i> 5% - 10%</p>
    <p><i class="fa fa-circle" style="color: orange"></i> 10% - 15%</p>
    <p><i class="fa fa-circle" style="color: red"></i> > 15%</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 保存地图
    if save_path:
        m.save(save_path)
        print(f"地图已保存至: {save_path}")
    else:
        return m


def main():
    """
    主函数，运行所有可视化分析
    """
    # 加载数据
    data = load_data()
    
    # 创建可视化
    print("\n1. 生成Prophet模型预测结果可视化...")
    # 选择使用量最高的充电站
    volume_data = data['volume']
    station_totals = volume_data.iloc[:, 1:].sum().sort_values(ascending=False)
    top_station = station_totals.index[0]
    
    plot_prophet_forecast(
        data, 
        top_station, 
        save_path=os.path.join(prophet_gtwr_dir, 'Prophet模型预测结果.png')
    )
    
    print("\n2. 生成GTWR模型空间系数分布图...")
    plot_gtwr_spatial_coefficients(
        save_path=os.path.join(prophet_gtwr_dir, 'GTWR模型空间系数分布.html')
    )
    
    print("\n3. 生成时空耦合效应分析图...")
    plot_spatiotemporal_coupling_effect(
        data, 
        save_path=os.path.join(prophet_gtwr_dir, '时空耦合效应分析.png')
    )
    
    print("\n4. 生成预测误差空间分布图...")
    plot_prediction_error_distribution(
        data, 
        save_path=os.path.join(prophet_gtwr_dir, '预测误差空间分布.html')
    )
    
    print("\n所有可视化已完成！")


if __name__ == "__main__":
    main()