# -*- coding: utf-8 -*-
"""
多源异构数据融合处理器
功能：实现高德充电桩数据和UrbanEV数据集的融合，按照接口规范进行处理
日期：2025年4月23日
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import sys
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入数据接口模块
from 核心代码_src.models.data_interface import DataInterface

class DataFusionProcessor:
    """
    数据融合处理器
    实现高德充电桩数据和UrbanEV数据集的融合，按照接口规范进行处理
    """
    
    def __init__(self, config=None):
        """
        初始化数据融合处理器
        
        Args:
            config: 配置参数字典，包含数据路径等信息
        """
        self.config = config or {}
        self.data_interface = DataInterface(config)
        self.gaode_data = None
        self.urbanev_data = {}
        self.fused_data = None
        self.prophet_input = None
        self.gtwr_input = None
        
        # 设置数据路径
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.data_path = os.path.join(self.base_path, '数据_data')
        self.output_path = os.path.join(self.base_path, '数据_data/7_跨域融合数据')
        
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
    
    def register_data_sources(self):
        """
        注册数据源
        """
        # 注册高德充电桩数据
        gaode_path = os.path.join(self.data_path, '1_充电桩数据/processed/深圳高德最终数据_cleaned_0417-0419.csv')
        self.data_interface.register_data_source(
            name='gaode_charging',
            source_type='csv',
            path=gaode_path,
            params={'encoding': 'utf-8'}
        )
        
        # 注册UrbanEV数据集
        urbanev_base_path = os.path.join(self.data_path, '1_充电桩数据/processed/UrbanEV_data')
        
        # 注册UrbanEV基础信息数据
        self.data_interface.register_data_source(
            name='urbanev_inf',
            source_type='csv',
            path=os.path.join(urbanev_base_path, 'inf.csv')
        )
        
        # 注册UrbanEV充电量数据
        self.data_interface.register_data_source(
            name='urbanev_volume',
            source_type='csv',
            path=os.path.join(urbanev_base_path, 'volume.csv')
        )
        
        # 注册UrbanEV占用率数据
        self.data_interface.register_data_source(
            name='urbanev_occupancy',
            source_type='csv',
            path=os.path.join(urbanev_base_path, 'occupancy.csv')
        )
        
        # 注册UrbanEV天气数据
        self.data_interface.register_data_source(
            name='urbanev_weather',
            source_type='csv',
            path=os.path.join(urbanev_base_path, 'weather_central.csv')
        )
        
        # 注册人口数据
        self.data_interface.register_data_source(
            name='population',
            source_type='csv',
            path=os.path.join(self.data_path, '2_时空动态数据/0421_区级_深圳人口数据.csv'),
            params={'encoding': 'utf-8'}
        )
        
        # 注册经济指标数据
        self.data_interface.register_data_source(
            name='economic',
            source_type='csv',
            path=os.path.join(self.data_path, '4_社会经济数据/0422_深圳区域经济指标数据.csv'),
            params={'encoding': 'utf-8'}
        )
    
    def load_data(self):
        """
        加载数据
        """
        # 加载高德充电桩数据
        self.gaode_data = self.data_interface.load_data('gaode_charging')
        
        # 加载UrbanEV数据集
        self.urbanev_data['inf'] = self.data_interface.load_data('urbanev_inf')
        self.urbanev_data['volume'] = self.data_interface.load_data('urbanev_volume')
        self.urbanev_data['occupancy'] = self.data_interface.load_data('urbanev_occupancy')
        self.urbanev_data['weather'] = self.data_interface.load_data('urbanev_weather')
        
        # 加载人口数据
        self.population_data = self.data_interface.load_data('population')
        
        # 加载经济指标数据
        self.economic_data = self.data_interface.load_data('economic')
        
        print("数据加载完成")
    
    def preprocess_gaode_data(self):
        """
        预处理高德充电桩数据
        """
        if self.gaode_data is None:
            raise ValueError("高德充电桩数据未加载")
        
        # 转换时间戳为日期格式
        self.gaode_data['date'] = pd.to_datetime(self.gaode_data['timestamp']).dt.date
        
        # 确保经纬度为数值型
        self.gaode_data['longitude'] = self.gaode_data['lng'].astype(float)
        self.gaode_data['latitude'] = self.gaode_data['lat'].astype(float)
        
        # 标准化区域名称
        district_mapping = {
            '南山区': '南山区',
            '福田区': '福田区',
            '罗湖区': '罗湖区',
            '盐田区': '盐田区',
            '宝安区': '宝安区',
            '龙岗区': '龙岗区',
            '龙华区': '龙华区',
            '坪山区': '坪山区',
            '光明区': '光明区',
            '大鹏新区': '大鹏新区',
            '深汕特别合作区': '深汕特别合作区'
        }
        
        # 应用区域映射
        self.gaode_data['district_std'] = self.gaode_data['district'].map(district_mapping)
        
        # 填充缺失值
        self.gaode_data['district_std'].fillna('未知区域', inplace=True)
        
        print("高德充电桩数据预处理完成")
        return self.gaode_data
    
    def preprocess_urbanev_data(self):
        """
        预处理UrbanEV数据集
        """
        if not self.urbanev_data:
            raise ValueError("UrbanEV数据集未加载")
        
        # 处理基础信息数据
        inf_data = self.urbanev_data['inf']
        
        # 处理充电量数据 - 假设volume.csv的列是日期，行是充电站ID
        volume_data = self.urbanev_data['volume']
        
        # 处理占用率数据 - 假设occupancy.csv的列是日期，行是充电站ID
        occupancy_data = self.urbanev_data['occupancy']
        
        # 处理天气数据
        weather_data = self.urbanev_data['weather']
        
        # 将充电量数据从宽格式转换为长格式
        volume_long = self._wide_to_long(volume_data, 'TAZID', 'date', 'charging_amount')
        
        # 将占用率数据从宽格式转换为长格式
        occupancy_long = self._wide_to_long(occupancy_data, 'TAZID', 'date', 'occupancy_rate')
        
        # 合并数据
        merged_data = inf_data.merge(volume_long, on='TAZID', how='left')
        merged_data = merged_data.merge(occupancy_long, on=['TAZID', 'date'], how='left')
        
        # 存储处理后的UrbanEV数据
        self.urbanev_processed = merged_data
        
        print("UrbanEV数据集预处理完成")
        return self.urbanev_processed
    
    def _wide_to_long(self, df, id_col, date_col, value_col):
        """
        将宽格式数据转换为长格式
        
        Args:
            df: 宽格式DataFrame
            id_col: ID列名
            date_col: 日期列名
            value_col: 值列名
            
        Returns:
            长格式DataFrame
        """
        # 复制数据框并设置索引
        df_copy = df.copy()
        
        # 获取ID列
        id_values = df_copy[id_col].values
        
        # 删除ID列，其余列假设为日期
        df_copy = df_copy.drop(columns=[id_col])
        
        # 转换为长格式
        long_df = pd.melt(
            df_copy.reset_index(), 
            id_vars=['index'], 
            var_name=date_col, 
            value_name=value_col
        )
        
        # 添加ID列
        long_df[id_col] = long_df['index'].map(lambda i: id_values[i])
        
        # 删除索引列
        long_df = long_df.drop(columns=['index'])
        
        # 转换日期格式
        long_df[date_col] = pd.to_datetime(long_df[date_col])
        
        return long_df
    
    def align_spatial_data(self):
        """
        空间维度对齐
        按照接口规范3.1.2进行空间维度对齐
        """
        # 确保数据已预处理
        if self.gaode_data is None or not hasattr(self, 'urbanev_processed'):
            raise ValueError("请先预处理数据")
        
        # 创建GeoDataFrame
        gaode_gdf = gpd.GeoDataFrame(
            self.gaode_data, 
            geometry=gpd.points_from_xy(self.gaode_data.longitude, self.gaode_data.latitude),
            crs="EPSG:4326"
        )
        
        urbanev_gdf = gpd.GeoDataFrame(
            self.urbanev_processed, 
            geometry=gpd.points_from_xy(self.urbanev_processed.longitude, self.urbanev_processed.latitude),
            crs="EPSG:4326"
        )
        
        # 空间连接 - 为每个高德充电桩找到最近的UrbanEV点位
        # 使用最近邻插值方法
        from sklearn.neighbors import BallTree
        
        # 提取坐标
        urbanev_coords = np.radians(urbanev_gdf[['latitude', 'longitude']].values)
        gaode_coords = np.radians(gaode_gdf[['latitude', 'longitude']].values)
        
        # 构建BallTree
        tree = BallTree(urbanev_coords, leaf_size=15, metric='haversine')
        
        # 查找最近邻
        distances, indices = tree.query(gaode_coords, k=1)
        
        # 地球半径（公里）
        earth_radius = 6371.0
        
        # 转换距离为公里
        distances_km = distances.flatten() * earth_radius
        
        # 为高德数据添加最近的UrbanEV点位信息
        gaode_gdf['nearest_urbanev_id'] = urbanev_gdf.iloc[indices.flatten()]['TAZID'].values
        gaode_gdf['distance_to_urbanev_km'] = distances_km
        
        # 设置距离阈值（例如5公里）
        distance_threshold = 5.0
        
        # 仅保留距离在阈值内的匹配
        gaode_gdf['valid_match'] = gaode_gdf['distance_to_urbanev_km'] <= distance_threshold
        
        # 存储空间对齐结果
        self.spatial_aligned_data = gaode_gdf
        
        print("空间维度对齐完成")
        return self.spatial_aligned_data
    
    def align_temporal_data(self):
        """
        时间维度对齐
        按照接口规范3.1.1进行时间维度对齐
        """
        # 确保数据已空间对齐
        if not hasattr(self, 'spatial_aligned_data'):
            raise ValueError("请先进行空间维度对齐")
        
        # 获取所有唯一日期
        all_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # 为每个充电桩创建完整的日期序列
        charging_stations = self.spatial_aligned_data['id'].unique()
        
        # 创建笛卡尔积 - 所有充电桩与所有日期的组合
        from itertools import product
        station_date_pairs = list(product(charging_stations, all_dates))
        
        # 创建完整的时间序列数据框
        time_series_df = pd.DataFrame(station_date_pairs, columns=['id', 'date'])
        
        # 将日期转换为datetime64格式
        time_series_df['date'] = pd.to_datetime(time_series_df['date'])
        
        # 合并空间对齐数据
        # 先选择需要的列
        spatial_data_subset = self.spatial_aligned_data[[
            'id', 'name', 'longitude', 'latitude', 'district_std', 
            'nearest_urbanev_id', 'distance_to_urbanev_km', 'valid_match'
        ]].drop_duplicates(subset=['id'])
        
        # 合并
        time_space_df = time_series_df.merge(spatial_data_subset, on='id', how='left')
        
        # 存储时空对齐结果
        self.time_space_aligned_data = time_space_df
        
        print("时间维度对齐完成")
        return self.time_space_aligned_data
    
    def merge_features(self):
        """
        合并特征数据
        将UrbanEV数据的特征合并到时空对齐的数据中
        """
        # 确保数据已时空对齐
        if not hasattr(self, 'time_space_aligned_data'):
            raise ValueError("请先进行时空维度对齐")
        
        # 获取有效匹配的数据
        valid_data = self.time_space_aligned_data[self.time_space_aligned_data['valid_match'] == True].copy()
        
        # 准备UrbanEV特征数据
        urbanev_features = self.urbanev_processed[[
            'TAZID', 'date', 'charging_amount', 'occupancy_rate', 'charge_count', 'area'
        ]].copy()
        
        # 确保日期格式一致
        urbanev_features['date'] = pd.to_datetime(urbanev_features['date']).dt.date
        valid_data['date'] = pd.to_datetime(valid_data['date']).dt.date
        
        # 合并特征
        merged_data = valid_data.merge(
            urbanev_features,
            left_on=['nearest_urbanev_id', 'date'],
            right_on=['TAZID', 'date'],
            how='left'
        )
        
        # 合并人口数据（假设按区域匹配）
        if hasattr(self, 'population_data'):
            # 假设人口数据有district和population列
            merged_data = merged_data.merge(
                self.population_data,
                left_on='district_std',
                right_on='district',
                how='left'
            )
        
        # 合并经济指标数据（假设按区域匹配）
        if hasattr(self, 'economic_data'):
            # 假设经济数据有district和相关经济指标列
            merged_data = merged_data.merge(
                self.economic_data,
                left_on='district_std',
                right_on='district',
                how='left'
            )
        
        # 存储合并后的特征数据
        self.feature_merged_data = merged_data
        
        print("特征数据合并完成")
        return self.feature_merged_data
    
    def prepare_prophet_input(self):
        """
        准备Prophet模型输入数据
        按照接口规范2.1进行处理
        """
        # 确保特征数据已合并
        if not hasattr(self, 'feature_merged_data'):
            raise ValueError("请先合并特征数据")
        
        # 按日期聚合充电需求
        prophet_data = self.feature_merged_data.groupby('date').agg({
            'charging_amount': 'sum',  # 充电需求量
            'charge_count': 'sum'      # 电动汽车数量（作为代理变量）
        }).reset_index()
        
        # 重命名列以符合Prophet要求
        prophet_data.rename(columns={
            'date': 'ds',
            'charging_amount': 'y',
            'charge_count': 'ev_count'
        }, inplace=True)
        
        # 确保日期格式为datetime64
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # 处理缺失值 - 使用线性插值
        prophet_data = prophet_data.sort_values('ds')
        prophet_data['y'] = prophet_data['y'].interpolate(method='linear')
        prophet_data['ev_count'] = prophet_data['ev_count'].interpolate(method='linear')
        
        # 处理异常值 - 使用3σ原则
        for col in ['y', 'ev_count']:
            mean = prophet_data[col].mean()
            std = prophet_data[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            prophet_data[col] = prophet_data[col].clip(lower_bound, upper_bound)
        
        # 存储Prophet输入数据
        self.prophet_input = prophet_data
        
        print("Prophet模型输入数据准备完成")
        return self.prophet_input
    
    def prepare_gtwr_input(self):
        """
        准备GTWR模型输入数据
        按照接口规范2.2进行处理
        """
        # 确保特征数据已合并
        if not hasattr(self, 'feature_merged_data'):
            raise ValueError("请先合并特征数据")
        
        # 选择GTWR所需字段
        gtwr_data = self.feature_merged_data[[
            'longitude', 'latitude', 'date', 'district_std', 'charging_amount',
            'charge_count', 'area', 'distance_to_urbanev_km'
        ]].copy()
        
        # 重命名列以符合GTWR要求
        gtwr_data.rename(columns={
            'district_std': 'district',
            'charge_count': 'ev_count',
            'distance_to_urbanev_km': '拥堵指数'  # 使用距离作为拥堵指数的代理变量
        }, inplace=True)
        
        # 添加其他必要字段（如果有）
        # 这里可以添加交通流量、总用电量等字段
        
        # 确保日期格式为datetime64
        gtwr_data['date'] = pd.to_datetime(gtwr_data['date'])
        
        # 处理缺失值
        gtwr_data = gtwr_data.fillna({
            'charging_amount': gtwr_data['charging_amount'].median(),
            'ev_count': gtwr_data['ev_count'].median(),
            '拥堵指数': gtwr_data['拥堵指数'].median()
        })
        
        # 存储GTWR输入数据
        self.gtwr_input = gtwr_data
        
        print("GTWR模型输入数据准备完成")
        return self.gtwr_input
    
    def save_processed_data(self):
        """
        保存处理后的数据
        """
        # 保存Prophet输入数据
        if hasattr(self, 'prophet_input'):
            prophet_output_path = os.path.join(self.output_path, 'prophet_input_data.csv')
            self.prophet_input.to_csv(prophet_output_path, index=False)
            print(f"Prophet输入数据已保存至: {prophet_output_path}")
        
        # 保存GTWR输入数据
        if hasattr(self, 'gtwr_input'):
            gtwr_output_path = os.path.join(self.output_path, 'gtwr_input_data.csv')
            self.gtwr_input.to_csv(gtwr_output_path, index=False)
            print(f"GTWR输入数据已保存至: {gtwr_output_path}")
        
        # 保存合并后的特征数据
        if hasattr(self, 'feature_merged_data'):
            merged_output_path = os.path.join(self.output_path, 'fused_feature_data.csv')
            self.feature_merged_data.to_csv(merged_output_path, index=False)
            print(f"合并后的特征数据已保存至: {merged_output_path}")
    
    def run_fusion_pipeline(self):
        """
        运行完整的数据融合流程
        """
        print("开始数据融合流程...")
        
        # 注册数据源
        self.register_data_sources()
        
        # 加载数据
        self.load_data()
        
        # 预处理数据
        self.preprocess_gaode_data()
        self.preprocess_urbanev_data()
        
        # 空间维度对齐
        self.align_spatial_data()
        
        # 时间维度对齐
        self.align_temporal_data()
        
        # 合并特征
        self.merge_features()
        
        # 准备模型输入
        self.prepare_prophet_input()
        self.prepare_gtwr_input()
        
        # 保存处理后的数据
        self.save_processed_data()
        
        print("数据融合流程完成！")


if __name__ == "__main__":
    # 创建数据融合处理器实例
    processor = DataFusionProcessor()
    
    # 运行数据融合流程
    processor.run_fusion_pipeline()