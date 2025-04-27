# -*- coding: utf-8 -*-
"""
Prophet-GTWR耦合模型 - 数据接口模块
功能：处理多源异构数据，提供统一的数据接口
"""

import pandas as pd
import numpy as np
import os
import json
import geopandas as gpd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata

class DataInterface:
    """
    数据接口类
    处理多源异构数据，提供统一的数据接口
    支持七类数据源的加载、预处理和融合
    """
    
    def __init__(self, config=None):
        """
        初始化数据接口
        
        Args:
            config: 配置参数字典，包含数据路径等信息
        """
        self.config = config or {}
        self.data_sources = {}
        self.processed_data = {}
        self.grid_data = None
        self.time_aligned_data = None
        self.feature_data = None
        
    def register_data_source(self, name, source_type, path, params=None):
        """
        注册数据源
        
        Args:
            name: 数据源名称
            source_type: 数据源类型 (csv, json, api, etc.)
            path: 数据源路径或URL
            params: 额外参数
        """
        self.data_sources[name] = {
            'type': source_type,
            'path': path,
            'params': params or {}
        }
        print(f"已注册数据源: {name}, 类型: {source_type}, 路径: {path}")
    
    def load_data(self, source_name=None):
        """
        加载指定数据源的数据
        
        Args:
            source_name: 数据源名称，如果为None则加载所有已注册数据源
            
        Returns:
            加载的数据对象
        """
        if source_name is None:
            # 加载所有数据源
            for name in self.data_sources:
                self._load_single_source(name)
            return self.processed_data
        else:
            # 加载单个数据源
            return self._load_single_source(source_name)
    
    def _load_single_source(self, source_name):
        """
        加载单个数据源
        
        Args:
            source_name: 数据源名称
            
        Returns:
            加载的数据对象
        """
        if source_name not in self.data_sources:
            raise ValueError(f"未注册的数据源: {source_name}")
        
        source = self.data_sources[source_name]
        source_type = source['type']
        path = source['path']
        params = source['params']
        
        print(f"正在加载数据源: {source_name}")
        
        # 根据数据源类型加载数据
        if source_type == 'csv':
            data = pd.read_csv(path, **params)
        elif source_type == 'excel':
            data = pd.read_excel(path, **params)
        elif source_type == 'json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 转换为DataFrame
            if isinstance(data, list):
                data = pd.DataFrame(data)
            else:
                data = pd.DataFrame([data])
        elif source_type == 'geojson':
            data = gpd.read_file(path)
        elif source_type == 'api':
            # 需要实现API调用逻辑
            data = self._call_api(path, params)
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
        
        # 存储加载的数据
        self.processed_data[source_name] = data
        
        print(f"数据源 {source_name} 加载完成，共 {len(data)} 条记录")
        return data
    
    def _call_api(self, url, params):
        """
        调用API获取数据
        
        Args:
            url: API URL
            params: API参数
            
        Returns:
            API返回的数据
        """
        # 需要实现具体的API调用逻辑
        import requests
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code}")
    
    def preprocess_data(self, data_name, processors=None):
        """
        数据预处理
        
        Args:
            data_name: 数据名称
            processors: 预处理器列表，每个元素是一个函数
            
        Returns:
            处理后的数据
        """
        if data_name not in self.processed_data:
            raise ValueError(f"未找到数据: {data_name}")
        
        data = self.processed_data[data_name].copy()
        
        if processors:
            for processor in processors:
                data = processor(data)
        
        # 更新处理后的数据
        self.processed_data[data_name] = data
        
        return data
    
    def spatial_alignment(self, data_list, target_crs='EPSG:4326', grid_size=1000):
        """
        空间数据对齐
        将不同空间参考系的数据对齐到统一的网格
        
        Args:
            data_list: 数据名称列表
            target_crs: 目标坐标系
            grid_size: 网格大小(米)
            
        Returns:
            对齐后的网格数据
        """
        # 检查数据是否存在
        for data_name in data_list:
            if data_name not in self.processed_data:
                raise ValueError(f"未找到数据: {data_name}")
        
        # 创建空的GeoDataFrame作为网格数据
        from shapely.geometry import box
        
        # 获取所有数据的空间范围
        bounds = None
        for data_name in data_list:
            data = self.processed_data[data_name]
            if 'geometry' in data.columns:
                # GeoDataFrame
                data_bounds = data.total_bounds
            elif all(col in data.columns for col in ['经度', '纬度']):
                # 普通DataFrame带经纬度
                data_bounds = [data['经度'].min(), data['纬度'].min(), 
                              data['经度'].max(), data['纬度'].max()]
            else:
                continue
            
            if bounds is None:
                bounds = data_bounds
            else:
                bounds = [min(bounds[0], data_bounds[0]), min(bounds[1], data_bounds[1]),
                          max(bounds[2], data_bounds[2]), max(bounds[3], data_bounds[3])]
        
        if bounds is None:
            raise ValueError("没有找到空间数据")
        
        # 创建网格
        from shapely.geometry import box
        grid_cells = []
        grid_ids = []
        
        # 计算网格数量
        x_min, y_min, x_max, y_max = bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # 转换为米的近似值（简化处理）
        x_range_m = x_range * 111320 * np.cos(np.radians((y_min + y_max) / 2))
        y_range_m = y_range * 111320
        
        # 计算网格数量
        nx = int(np.ceil(x_range_m / grid_size))
        ny = int(np.ceil(y_range_m / grid_size))
        
        # 创建网格单元
        for i in range(nx):
            for j in range(ny):
                x0 = x_min + i * (x_range / nx)
                y0 = y_min + j * (y_range / ny)
                x1 = x_min + (i + 1) * (x_range / nx)
                y1 = y_min + (j + 1) * (y_range / ny)
                grid_cells.append(box(x0, y0, x1, y1))
                grid_ids.append(f"grid_{i}_{j}")
        
        # 创建网格GeoDataFrame
        grid_gdf = gpd.GeoDataFrame({'grid_id': grid_ids, 'geometry': grid_cells}, crs=target_crs)
        
        # 将各数据源数据映射到网格
        for data_name in data_list:
            data = self.processed_data[data_name]
            
            # 检查数据类型并进行相应处理
            if isinstance(data, gpd.GeoDataFrame):
                # 空间连接
                if data.crs != target_crs:
                    data = data.to_crs(target_crs)
                
                # 空间连接，将数据属性聚合到网格
                joined = gpd.sjoin(grid_gdf, data, how='left', op='intersects')
                
                # 聚合数据到网格
                for col in data.columns:
                    if col != 'geometry' and pd.api.types.is_numeric_dtype(data[col]):
                        grid_gdf[f"{data_name}_{col}"] = joined.groupby('grid_id')[col].mean()
            
            elif all(col in data.columns for col in ['经度', '纬度']):
                # 点数据，使用空间连接
                points_gdf = gpd.GeoDataFrame(
                    data,
                    geometry=gpd.points_from_xy(data['经度'], data['纬度']),
                    crs=target_crs
                )
                
                # 空间连接
                joined = gpd.sjoin(grid_gdf, points_gdf, how='left', op='contains')
                
                # 聚合数据到网格
                for col in data.columns:
                    if col not in ['经度', '纬度', 'geometry'] and pd.api.types.is_numeric_dtype(data[col]):
                        grid_gdf[f"{data_name}_{col}"] = joined.groupby('grid_id')[col].mean()
        
        # 保存网格数据
        self.grid_data = grid_gdf
        
        return grid_gdf
    
    def temporal_alignment(self, data_list, target_freq='H', timezone='Asia/Shanghai'):
        """
        时间数据对齐
        将不同时间粒度的数据对齐到统一的时间频率
        
        Args:
            data_list: 数据名称列表
            target_freq: 目标时间频率，如'H'(小时),'D'(天),'W'(周),'M'(月)
            timezone: 目标时区
            
        Returns:
            对齐后的时间序列数据
        """
        # 检查数据是否存在
        for data_name in data_list:
            if data_name not in self.processed_data:
                raise ValueError(f"未找到数据: {data_name}")
        
        # 确定时间范围
        min_time = None
        max_time = None
        
        for data_name in data_list:
            data = self.processed_data[data_name]
            
            # 查找时间列
            time_cols = [col for col in data.columns if '时间' in col or 'date' in col.lower() or 'time' in col.lower()]
            
            if not time_cols:
                continue
            
            # 使用第一个时间列
            time_col = time_cols[0]
            
            # 确保时间列是datetime类型
            if not pd.api.types.is_datetime64_dtype(data[time_col]):
                data[time_col] = pd.to_datetime(data[time_col])
            
            # 更新时间范围
            data_min = data[time_col].min()
            data_max = data[time_col].max()
            
            if min_time is None or data_min < min_time:
                min_time = data_min
            
            if max_time is None or data_max > max_time:
                max_time = data_max
        
        if min_time is None or max_time is None:
            raise ValueError("没有找到时间数据")
        
        # 创建统一的时间索引
        time_index = pd.date_range(start=min_time, end=max_time, freq=target_freq, tz=timezone)
        
        # 创建对齐后的数据框
        aligned_data = pd.DataFrame(index=time_index)
        
        # 对齐各数据源
        for data_name in data_list:
            data = self.processed_data[data_name]
            
            # 查找时间列
            time_cols = [col for col in data.columns if '时间' in col or 'date' in col.lower() or 'time' in col.lower()]
            
            if not time_cols:
                continue
            
            # 使用第一个时间列
            time_col = time_cols[0]
            
            # 确保时间列是datetime类型
            if not pd.api.types.is_datetime64_dtype(data[time_col]):
                data[time_col] = pd.to_datetime(data[time_col])
            
            # 设置时间索引
            data_indexed = data.set_index(time_col)
            
            # 对数值列进行重采样
            for col in data.columns:
                if col != time_col and pd.api.types.is_numeric_dtype(data[col]):
                    # 重采样并插值
                    resampled = data_indexed[col].resample(target_freq).mean()
                    # 线性插值填充缺失值
                    resampled = resampled.interpolate(method='linear')
                    # 添加到对齐数据中
                    aligned_data[f"{data_name}_{col}"] = resampled
        
        # 保存对齐后的时间序列数据
        self.time_aligned_data = aligned_data
        
        return aligned_data
    
    def extract_features(self, data=None, feature_extractors=None):
        """
        特征提取
        从原始数据中提取有用的特征
        
        Args:
            data: 数据对象，如果为None则使用对齐后的数据
            feature_extractors: 特征提取器列表，每个元素是一个函数
            
        Returns:
            提取的特征数据
        """
        if data is None:
            # 使用时空对齐后的数据
            if self.time_aligned_data is not None and self.grid_data is not None:
                # 需要实现时空数据的特征提取逻辑
                pass
            elif self.time_aligned_data is not None:
                data = self.time_aligned_data
            elif self.grid_data is not None:
                data = self.grid_data
            else:
                raise ValueError("没有可用的对齐数据")
        
        # 复制数据，避免修改原始数据
        features = data.copy()
        
        # 应用特征提取器
        if feature_extractors:
            for extractor in feature_extractors:
                features = extractor(features)
        
        # 保存特征数据
        self.feature_data = features
        
        return features
    
    def extract_time_features(self, time_series):
        """
        提取时间特征
        从时间戳中提取有用的时间特征
        
        Args:
            time_series: 时间序列数据
            
        Returns:
            增加了时间特征的数据
        """
        features = time_series.copy()
        
        # 提取时间特征
        if isinstance(features.index, pd.DatetimeIndex):
            features['hour'] = features.index.hour
            features['dayofweek'] = features.index.dayofweek
            features['month'] = features.index.month
            features['is_weekend'] = features.index.dayofweek >= 5
            
            # 添加周期性特征
            features['hour_sin'] = np.sin(2 * np.pi * features.index.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features.index.hour / 24)
            features['day_sin'] = np.sin(2 * np.pi * features.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * features.index.dayofweek / 7)
        
        return features
    
    def extract_spatial_features(self, spatial_data, poi_data=None, road_network=None):
        """
        提取空间特征
        从空间数据中提取有用的空间特征
        
        Args:
            spatial_data: 空间数据
            poi_data: POI数据
            road_network: 道路网络数据
            
        Returns:
            增加了空间特征的数据
        """
        features = spatial_data.copy()
        
        # 如果有POI数据，计算POI密度
        if poi_data is not None and isinstance(poi_data, gpd.GeoDataFrame):
            # 空间连接
            joined = gpd.sjoin(features, poi_data, how='left', op='intersects')
            # 计算每个网格内的POI数量
            poi_counts = joined.groupby(joined.index).size()
            features['poi_density'] = poi_counts
            # 填充缺失值
            features['poi_density'] = features['poi_density'].fillna(0)
        
        # 如果有道路网络数据，计算道路密度
        if road_network is not None and isinstance(road_network, gpd.GeoDataFrame):
            # 计算每个网格内的道路长度总和
            def calculate_road_length(grid_geom):
                # 选择与网格相交的道路
                intersecting_roads = road_network[road_network.intersects(grid_geom)]
                # 计算相交部分的长度总和
                if len(intersecting_roads) > 0:
                    return sum(road.intersection(grid_geom).length for road in intersecting_roads.geometry)
                return 0
            
            features['road_density'] = features.geometry.apply(calculate_road_length)
        
        return features
    
    def normalize_features(self, features=None, columns=None):
        """
        特征归一化
        将特征归一化到[0,1]区间
        
        Args:
            features: 特征数据，如果为None则使用已提取的特征
            columns: 要归一化的列，如果为None则归一化所有数值列
            
        Returns:
            归一化后的特征数据
        """
        if features is None:
            if self.feature_data is None:
                raise ValueError("没有可用的特征数据")
            features = self.feature_data.copy()
        else:
            features = features.copy()
        
        # 确定要归一化的列
        if columns is None:
            columns = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
        
        # 归一化
        scaler = MinMaxScaler()
        features[columns] = scaler.fit_transform(features[columns])
        
        return features
    
    def prepare_model_data(self, target_col, feature_cols=None, test_size=0.2, random_state=42):
        """
        准备模型数据
        将特征数据分割为训练集和测试集
        
        Args:
            target_col: 目标变量列名
            feature_cols: 特征列名列表，如果为None则使用所有数值列
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if self.feature_data is None:
            raise ValueError("没有可用的特征数据")
        
        data = self.feature_data.copy()
        
        # 确定特征列
        if feature_cols is None:
            feature_cols = [col for col in data.columns 
                           if col != target_col and pd.api.types.is_numeric_dtype(data[col])]
        
        # 分割数据
        from sklearn.model_selection import train_test_split
        
        X = data[feature_cols]
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

# 示例用法
if __name__ == "__main__":
    # 创建数据接口
    data_interface = DataInterface()
    
    # 注册数据源
    data_interface.register_data_source(
        name="charging_stations",
        source_type="csv",
        path="../数据_data/1_充电桩数据/processed/深圳高德最终数据_with_location_type.csv"
    )
    
    data_interface.register_data_source(
        name="traffic_flow",
        source_type="csv",
        path="../数据_data/2_时空动态数据/主要道路车流量数据.csv"
    )
    
    # 加载数据
    data = data_interface.load_data()
    
    print("数据加载完成")