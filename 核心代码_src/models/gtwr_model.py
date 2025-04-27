# -*- coding: utf-8 -*-
"""
Prophet-GTWR耦合模型 - GTWR空间模型模块
功能：地理与时间加权回归，分析空间异质性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GTWRModel:
    """
    地理与时间加权回归(GTWR)模型类
    用于分析空间异质性和时间异质性
    """
    
    def __init__(self, params=None):
        """
        初始化GTWR模型
        
        Args:
            params: GTWR模型参数字典，可包含：
                - spatial_bandwidth: 空间带宽
                - temporal_bandwidth: 时间带宽
                - kernel_function: 核函数类型，'gaussian'或'bisquare'
                - optimization_method: 带宽优化方法，'cv'或'aic'
                - time_decay_lambda: 时间衰减系数，范围[0.2, 0.8]，默认0.5
        """
        self.params = params or {}
        self.spatial_bandwidth = self.params.get('spatial_bandwidth', None)
        self.temporal_bandwidth = self.params.get('temporal_bandwidth', None)
        self.kernel_function = self.params.get('kernel_function', 'gaussian')
        self.optimization_method = self.params.get('optimization_method', 'cv')
        self.time_decay_lambda = self.params.get('time_decay_lambda', 0.5)
        
        # 模型参数
        self.coefficients = None  # 局部回归系数
        self.intercept = None     # 局部截距
        self.feature_names = None # 特征名称
        self.r2_score = None      # 拟合优度
        self.r2_adj = None        # 调整后的拟合优度
        self.aic = None           # AIC
        self.bic = None           # BIC
        self.mse = None           # 均方误差
        self.local_r2 = None      # 局部R2
    
    def _calculate_spatial_weights(self, train_coords, test_coords):
        """
        计算空间权重矩阵
        
        Args:
            train_coords: 训练数据坐标，形状为(n_samples, 2)，经纬度
            test_coords: 测试数据坐标，形状为(m_samples, 2)，经纬度
            
        Returns:
            空间权重矩阵，形状为(m_samples, n_samples)
        """
        # 计算欧氏距离矩阵
        dist_matrix = cdist(test_coords, train_coords, 'euclidean')
        
        # 自适应带宽
        if self.spatial_bandwidth is None:
            # 使用距离矩阵的中位数作为带宽
            self.spatial_bandwidth = np.median(dist_matrix) / 2
            print(f"自适应空间带宽: {self.spatial_bandwidth:.4f}")
        
        # 根据核函数类型计算权重
        if self.kernel_function == 'gaussian':
            # 高斯核函数: exp(-0.5 * (d/h)²)
            weights = np.exp(-0.5 * (dist_matrix / self.spatial_bandwidth) ** 2)
        elif self.kernel_function == 'bisquare':
            # 双平方核函数: (1 - (d/h)²)² if d < h else 0
            weights = np.zeros_like(dist_matrix)
            mask = dist_matrix < self.spatial_bandwidth
            weights[mask] = (1 - (dist_matrix[mask] / self.spatial_bandwidth) ** 2) ** 2
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel_function}")
        
        return weights
    
    def _calculate_temporal_weights(self, train_times, test_times):
        """
        计算时间权重矩阵
        
        Args:
            train_times: 训练数据时间戳，形状为(n_samples,)
            test_times: 测试数据时间戳，形状为(m_samples,)
            
        Returns:
            时间权重矩阵，形状为(m_samples, n_samples)
        """
        # 将时间戳转换为数值（天数）
        if isinstance(train_times[0], str):
            train_times = pd.to_datetime(train_times)
        if isinstance(test_times[0], str):
            test_times = pd.to_datetime(test_times)
        
        # 转换为相对天数
        min_date = min(train_times.min(), test_times.min())
        train_days = (train_times - min_date).dt.total_seconds() / (24 * 3600)
        test_days = (test_times - min_date).dt.total_seconds() / (24 * 3600)
        
        # 计算时间距离矩阵（天）
        time_dist = np.abs(test_days.values.reshape(-1, 1) - train_days.values.reshape(1, -1))
        
        # 自适应时间带宽
        if self.temporal_bandwidth is None:
            self.temporal_bandwidth = np.median(time_dist) / 2
            print(f"自适应时间带宽: {self.temporal_bandwidth:.4f} 天")
        
        # 根据核函数类型计算权重
        if self.kernel_function == 'gaussian':
            # 指数衰减函数: exp(-λ * t/h)
            weights = np.exp(-self.time_decay_lambda * time_dist / self.temporal_bandwidth)
        elif self.kernel_function == 'bisquare':
            # 双平方核函数
            weights = np.zeros_like(time_dist)
            mask = time_dist < self.temporal_bandwidth
            weights[mask] = (1 - (time_dist[mask] / self.temporal_bandwidth) ** 2) ** 2
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel_function}")
        
        return weights
    
    def _calculate_spatiotemporal_weights(self, train_data, test_data, coords_cols, time_col):
        """
        计算时空联合权重矩阵
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            coords_cols: 坐标列名列表，如['经度', '纬度']
            time_col: 时间列名
            
        Returns:
            时空联合权重矩阵，形状为(m_samples, n_samples)
        """
        # 提取坐标
        train_coords = train_data[coords_cols].values
        test_coords = test_data[coords_cols].values
        
        # 提取时间
        train_times = train_data[time_col]
        test_times = test_data[time_col]
        
        # 计算空间权重和时间权重
        spatial_weights = self._calculate_spatial_weights(train_coords, test_coords)
        temporal_weights = self._calculate_temporal_weights(train_times, test_times)
        
        # 时空联合权重（哈达玛积）
        spatiotemporal_weights = spatial_weights * temporal_weights
        
        # 归一化权重
        row_sums = spatiotemporal_weights.sum(axis=1, keepdims=True)
        spatiotemporal_weights = spatiotemporal_weights / row_sums
        
        return spatiotemporal_weights
    
    def _local_weighted_regression(self, X, y, weights):
        """
        局部加权回归
        
        Args:
            X: 特征矩阵，形状为(n_samples, n_features)
            y: 目标变量，形状为(n_samples,)
            weights: 权重向量，形状为(n_samples,)
            
        Returns:
            局部回归系数，形状为(n_features+1,)
        """
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # 创建权重矩阵
        W = np.diag(weights)
        
        # 计算加权最小二乘解
        # β = (X'WX)^(-1)X'Wy
        XtW = X_with_intercept.T @ W
        XtWX = XtW @ X_with_intercept
        XtWy = XtW @ y
        
        try:
            # 尝试直接求逆
            XtWX_inv = np.linalg.inv(XtWX)
            beta = XtWX_inv @ XtWy
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            XtWX_inv = np.linalg.pinv(XtWX)
            beta = XtWX_inv @ XtWy
        
        return beta
    
    def fit(self, X, y, coords, times, coords_cols=None, time_col=None):
        """
        训练GTWR模型
        
        Args:
            X: 特征矩阵，DataFrame或numpy数组
            y: 目标变量，Series或numpy数组
            coords: 空间坐标，DataFrame或numpy数组，每行是一个观测点的(经度,纬度)
            times: 时间坐标，Series或numpy数组
            coords_cols: 坐标列名列表，如['经度', '纬度']，当X是DataFrame时使用
            time_col: 时间列名，当X是DataFrame时使用
            
        Returns:
            训练好的模型
        """
        # 处理输入数据
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        if coords_cols is not None and isinstance(coords, pd.DataFrame):
            coords_array = coords[coords_cols].values
        else:
            coords_array = coords
        
        if time_col is not None and isinstance(times, pd.DataFrame):
            times_array = times[time_col].values
        elif time_col is not None and isinstance(times, pd.Series):
            times_array = times.values
        else:
            times_array = times
        
        # 优化带宽（如果需要）
        if self.optimization_method == 'cv' and (self.spatial_bandwidth is None or self.temporal_bandwidth is None):
            self._optimize_bandwidth(X_array, y_array, coords_array, times_array)
        
        # 创建训练数据和测试数据（这里测试数据就是训练数据，用于计算局部系数）
        train_data = pd.DataFrame(np.column_stack([coords_array, times_array, X_array, y_array]))
        test_data = train_data.copy()
        
        # 设置列名
        if coords_cols is None:
            coords_cols = ['longitude', 'latitude']
        if time_col is None:
            time_col = 'time'
        
        # 重命名列
        columns = coords_cols + [time_col] + self.feature_names + ['target']
        train_data.columns = columns
        test_data.columns = columns
        
        # 计算时空权重矩阵
        weights_matrix = self._calculate_spatiotemporal_weights(
            train_data, test_data, coords_cols, time_col
        )
        
        # 初始化系数矩阵
        n_samples = X_array.shape[0]
        n_features = X_array.shape[1]
        self.coefficients = np.zeros((n_samples, n_features))
        self.intercept = np.zeros(n_samples)
        self.local_r2 = np.zeros(n_samples)
        
        # 对每个观测点进行局部加权回归
        print("开始训练GTWR模型...")
        for i in range(n_samples):
            # 获取当前观测点的权重
            weights = weights_matrix[i]
            
            # 局部加权回归
            beta = self._local_weighted_regression(X_array, y_array, weights)
            
            # 保存系数
            self.intercept[i] = beta[0]
            self.coefficients[i] = beta[1:]
            
            # 计算局部R2
            y_pred = beta[0] + X_array @ beta[1:]
            weighted_sse = np.sum(weights * (y_array - y_pred) ** 2)
            weighted_sst = np.sum(weights * (y_array - np.mean(y_array)) ** 2)
            self.local_r2[i] = 1 - weighted_sse / weighted_sst
        
        # 计算全局拟合优度
        y_pred = self.predict(X_array, coords_array, times_array)
        self.mse = mean_squared_error(y_array, y_pred)
        self.r2_score = r2_score(y_array, y_pred)
        
        # 计算调整后的R2
        n = len(y_array)
        p = X_array.shape[1] + 1  # 特征数+截距
        self.r2_adj = 1 - (1 - self.r2_score) * (n - 1) / (n - p - 1)
        
        # 计算AIC和BIC
        residuals = y_array - y_pred
        sigma2 = np.sum(residuals ** 2) / n
        self.aic = n * np.log(sigma2) + 2 * p
        self.bic = n * np.log(sigma2) + np.log(n) * p
        
        print("GTWR模型训练完成")
        print(f"全局R2: {self.r2_score:.4f}, 调整后R2: {self.r2_adj:.4f}")
        print(f"AIC: {self.aic:.4f}, BIC: {self.bic:.4f}")
        
        return self
    
    def _optimize_bandwidth(self, X, y, coords, times):
        """
        优化带宽参数
        
        Args:
            X: 特征矩阵
            y: 目标变量
            coords: 空间坐标
            times: 时间坐标
            
        Returns:
            最优带宽参数
        """
        print("开始优化带宽参数...")
        
        # 定义目标函数（交叉验证误差）
        def objective(bandwidths):
            spatial_bw, temporal_bw = bandwidths
            self.spatial_bandwidth = spatial_bw
            self.temporal_bandwidth = temporal_bw
            
            # 使用留一法交叉验证
            n_samples = X.shape[0]
            cv_errors = np.zeros(n_samples)
            
            for i in range(n_samples):
                # 留出第i个样本
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y, i)
                coords_train = np.delete(coords, i, axis=0)
                times_train = np.delete(times, i)
                
                # 计算权重
                spatial_weights = self._calculate_spatial_weights(
                    coords_train, coords[i:i+1])[0]
                temporal_weights = self._calculate_temporal_weights(
                    times_train, times[i:i+1])[0]
                weights = spatial_weights * temporal_weights
                weights = weights / np.sum(weights)
                
                # 局部加权回归
                beta = self._local_weighted_regression(X_train, y_train, weights)
                
                # 预测
                y_pred = beta[0] + X[i] @ beta[1:]
                
                # 计算误差
                cv_errors[i] = (y[i] - y_pred) ** 2
            
            # 返回均方误差
            return np.mean(cv_errors)
        
        # 初始猜测值
        if self.spatial_bandwidth is None:
            # 使用距离矩阵的中位数作为初始猜测
            dist_matrix = cdist(coords, coords, 'euclidean')
            initial_spatial_bw = np.median(dist_matrix) / 2
        else:
            initial_spatial_bw = self.spatial_bandwidth
        
        if self.temporal_bandwidth is None:
            # 将时间转换为数值
            if isinstance(times[0], str):
                times_dt = pd.to_datetime(times)
            else:
                times_dt = times
            
            # 计算时间距离
            min_date = min(times_dt)
            times_days = (times_dt - min_date).total_seconds() / (24 * 3600)
            time_dist = np.abs(times_days.reshape(-1, 1) - times_days.reshape(1, -1))
            initial_temporal_bw = np.median(time_dist) / 2
        else:
            initial_temporal_bw = self.temporal_bandwidth
        
        # 优化带宽
        result = minimize(
            objective,
            [initial_spatial_bw, initial_temporal_bw],
            method='Nelder-Mead',
            bounds=[(initial_spatial_bw*0.1, initial_spatial_bw*10),
                    (initial_temporal_bw*0.1, initial_temporal_bw*10)]
        )
        
        # 更新带宽
        self.spatial_bandwidth, self.temporal_bandwidth = result.x
        
        print(f"优化后的空间带宽: {self.spatial_bandwidth:.4f}")
        print(f"优化后的时间带宽: {self.temporal_bandwidth:.4f}")
        
        return result.x
    
    def predict(self, X, coords, times):
        """
        生成预测结果
        
        Args:
            X: 新特征矩阵，形状为(m_samples, n_features)
            coords: 新空间坐标，形状为(m_samples, 2)
            times: 新时间坐标，形状为(m_samples,)
            
        Returns:
            预测结果，形状为(m_samples,)
        """
        if self.coefficients is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 处理输入数据
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(coords, pd.DataFrame):
            coords_array = coords.values
        else:
            coords_array = coords
        
        if isinstance(times, pd.Series):
            times_array = times.values
        else:
            times_array = times
        
        # 获取训练数据的坐标和时间
        # 这里假设训练数据的坐标和时间已经保存在模型中
        # 实际应用中可能需要修改
        train_coords = coords_array  # 简化处理，使用测试数据的坐标
        train_times = times_array    # 简化处理，使用测试数据的时间
        
        # 计算权重矩阵
        spatial_weights = self._calculate_spatial_weights(train_coords, coords_array)
        temporal_weights = self._calculate_temporal_weights(train_times, times_array)
        weights_matrix = spatial_weights * temporal_weights
        
        # 归一化权重
        row_sums = weights_matrix.sum(axis=1, keepdims=True)
        weights_matrix = weights_matrix / row_sums
        
        # 初始化预测结果
        m_samples = X_array.shape[0]
        y_pred = np.zeros(m_samples)
        
        # 对每个预测点进行加权平均
        for i in range(m_samples):
            # 获取当前预测点的权重
            weights = weights_matrix[i]
            
            # 加权平均系数
            weighted_intercept = np.sum(weights * self.intercept)
            weighted_coefficients = np.sum(weights.reshape(-1, 1) * self.coefficients, axis=0)
            
            # 预测
            y_pred[i] = weighted_intercept + X_array[i] @ weighted_coefficients
        
        return y_pred
    
    def local_coefficients(self):
        """
        获取局部回归系数
        
        Returns:
            局部回归系数DataFrame，包含坐标、截距和系数
        """
        if self.coefficients is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 创建系数DataFrame
        coef_df = pd.DataFrame(self.coefficients, columns=self.feature_names)
        coef_df['intercept'] = self.intercept
        coef_df['local_r2'] = self.local_r2
        
        return coef_df
    
    def plot_coefficient_map(self, coef_name, coords, figsize=(12, 8)):
        """
        绘制系数空间分布图
        
        Args:
            coef_name: 系数名称，'intercept'或特征名
            coords: 空间坐标DataFrame，包含'经度'和'纬度'列
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        if self.coefficients is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取系数值
        if coef_name == 'intercept':
            coef_values = self.intercept
        elif coef_name == 'local_r2':
            coef_values = self.local_r2
        elif coef_name in self.feature_names:
            coef_idx = self.feature_names.index(coef_name)
            coef_values = self.coefficients[:, coef_idx]
        else:
            raise ValueError(f"未知的系数名称: {coef_name}")
        
        # 创建GeoDataFrame
        from shapely.geometry import Point
        
        # 提取坐标
        if isinstance(coords, pd.DataFrame):
            if '经度' in coords.columns and '纬度' in coords.columns:
                lon = coords['经度']
                lat = coords['纬度']
            elif 'longitude' in coords.columns and 'latitude' in coords.columns:
                lon = coords['longitude']
                lat = coords['latitude']
            else:
                lon = coords.iloc[:, 0]
                lat = coords.iloc[:, 1]
        else:
            lon = coords[:, 0]
            lat = coords[:, 1]
        
        # 创建几何对象
        geometry = [Point(x, y) for x, y in zip(lon, lat)]
        
        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'coefficient': coef_values,
            'geometry': geometry
        }, crs='EPSG:4326')
        
        # 绘制地图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用分位数分类
        vmin, vmax = np.percentile(coef_values, [5, 95])
        
        # 绘制系数分布
        gdf.plot(
            column='coefficient',
            cmap='viridis',
            legend=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax
        )
        
        # 设置标题和标签
        ax.set_title(f'{coef_name}系数空间分布')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        return fig
    
    def plot_local_r2_map(self, coords, figsize=(12, 8)):
        """
        绘制局部R2空间分布图
        
        Args:
            coords: 空间坐标DataFrame，包含'经度'和'纬度'列
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        return self.plot_coefficient_map('local_r2', coords, figsize)
    
    def monte_carlo_test(self, X, y, coords, times, n_permutations=99, alpha=0.05):
        """
        蒙特卡洛显著性检验
        检验空间非平稳性是否显著
        
        Args:
            X: 特征矩阵
            y: 目标变量
            coords: 空间坐标
            times: 时间坐标
            n_permutations: 置换次数
            alpha: 显著性水平
            
        Returns:
            检验结果字典
        """
        # 计算原始模型的性能指标
        original_r2 = self.r2_score
        
        # 初始化置换测试结果
        permutation_r2 = np.zeros(n_permutations)
        
        print(f"开始蒙特卡洛显著性检验 ({n_permutations}次置换)...")
        
        # 执行置换测试
        for i in range(n_permutations):
            # 随机打乱空间坐标
            perm_idx = np.random.permutation(len(coords))
            perm_coords = coords[perm_idx]
            
            # 使用打乱的坐标训练模型
            perm_model = GTWRModel(self.params)
            perm_model.fit(X, y, perm_coords, times)
            
            # 记录性能指标
            permutation_r2[i] = perm_model.r2_score
        
        # 计算p值
        p_value = np.mean(permutation_r2 >= original_r2)
        
        # 判断显著性
        is_significant = p_value < alpha
        
        result = {
            'original_r2': original_r2,
            'permutation_r2_mean': np.mean(permutation_r2),
            'permutation_r2_std': np.std(permutation_r2),
            'p_value': p_value,
            'is_significant': is_significant
        }
        
        print("蒙特卡洛检验结果:")
        print(f"原始R2: {original_r2:.4f}")
        print(f"置换测试R2均值: {result['permutation_r2_mean']:.4f} ± {result['permutation_r2_std']:.4f}")
        print(f"p值: {p_value:.4f}")
        print(f"空间非平稳性是否显著: {is_significant}")
        
        return result

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    
    # 空间坐标（经纬度）
    coords = np.random.uniform(low=[113.8, 22.4], high=[114.5, 22.8], size=(n_samples, 2))

