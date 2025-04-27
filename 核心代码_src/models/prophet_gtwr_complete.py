# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型 - 完整实现
功能：结合Facebook Prophet时序预测和GTWR地理时空加权回归模型的完整实现
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from datetime import datetime, timedelta
import geopandas as gpd
import os
import json
import warnings
warnings.filterwarnings('ignore')  # 忽略Prophet的警告

class ProphetGTWRComplete:
    """
    Prophet-GTWR耦合模型完整实现
    
    结合Facebook Prophet时序预测和地理时空加权回归(GTWR)的耦合模型
    支持多源异构数据融合、双向反馈机制和动态权重自适应
    """
    
    def __init__(self, prophet_params=None, gtwr_params=None, coupling_params=None):
        """
        初始化模型
        
        Args:
            prophet_params: Prophet模型参数字典
            gtwr_params: GTWR模型参数字典
            coupling_params: 耦合参数字典
        """
        # 设置默认参数
        self.prophet_params = prophet_params or {
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': True,
            'changepoint_prior_scale': 0.05
        }
        
        self.gtwr_params = gtwr_params or {
            'spatial_bandwidth': None,  # 自适应带宽
            'temporal_bandwidth': None,  # 自适应带宽
            'kernel_function': 'gaussian',
            'time_decay_lambda': 0.5,
            'optimization_method': 'cv'
        }
        
        self.coupling_params = coupling_params or {
            'alpha': 0.5,  # 融合权重参数
            'feedback_mode': 'two_way',  # 反馈模式：'one_way'或'two_way'
            'dynamic_weight': True  # 是否使用动态权重
        }
        
        # 初始化模型组件
        self.prophet_model = None
        self.spatial_bandwidth = self.gtwr_params.get('spatial_bandwidth')
        self.temporal_bandwidth = self.gtwr_params.get('temporal_bandwidth')
        self.kernel_function = self.gtwr_params.get('kernel_function')
        self.time_decay_lambda = self.gtwr_params.get('time_decay_lambda')
        
        # 模型状态和结果
        self.coefficients = None  # GTWR局部回归系数
        self.intercept = None     # GTWR局部截距
        self.feature_names = None # 特征名称
        self.r2_score = None      # 拟合优度
        self.r2_adj = None        # 调整后的拟合优度
        self.mse = None           # 均方误差
        self.mae = None           # 平均绝对误差
        self.prophet_forecast = None  # Prophet预测结果
        self.prophet_components = None # Prophet分解组件
        self.final_prediction = None   # 最终融合预测结果
        self.local_r2 = None      # 局部R2
        
        # 耦合参数
        self.alpha = self.coupling_params.get('alpha')  # 融合权重
        self.feedback_mode = self.coupling_params.get('feedback_mode')  # 反馈模式
        self.dynamic_weight = self.coupling_params.get('dynamic_weight')  # 动态权重
    
    def _calculate_spatial_weights(self, train_coords, test_coords):
        """
        计算空间权重矩阵
        
        使用高斯核函数或双平方核函数计算空间权重
        
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
    
    def _calculate_temporal_weights(self, train_times, test_times, train_data=None, test_data=None):
        """
        计算时间权重矩阵
        
        使用指数衰减函数计算时间权重，并根据占用率动态调整衰减系数
        
        Args:
            train_times: 训练数据时间戳，形状为(n_samples,)
            test_times: 测试数据时间戳，形状为(m_samples,)
            train_data: 训练数据完整DataFrame，用于获取占用率
            test_data: 测试数据完整DataFrame，用于获取占用率
            
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
        
        # 动态调整时间衰减系数λ
        lambda_adjusted = np.full((len(test_times), len(train_times)), self.time_decay_lambda)
        
        # 当测试数据中的占用率>80%时，提高λ值0.1（增加近期数据权重）
        if self.dynamic_weight and test_data is not None and train_data is not None and '实时占用率' in test_data.columns:
            high_occupancy_mask = (test_data['实时占用率'].values > 80).reshape(-1, 1)
            lambda_adjusted[high_occupancy_mask] += 0.1
            print("检测到高占用率(>80%)样本，已动态调整时间衰减系数")
        
        # 指数衰减函数: exp(-λ * t/h)
        # λ是时间衰减系数，t是时间距离，h是时间带宽
        weights = np.exp(-lambda_adjusted * time_dist / self.temporal_bandwidth)
        
        return weights
    
    def _calculate_spatiotemporal_weights(self, train_data, test_data):
        """
        计算时空联合权重矩阵
        
        结合空间权重和时间权重
        
        Args:
            train_data: 训练数据，包含坐标和时间
            test_data: 测试数据，包含坐标和时间
            
        Returns:
            时空联合权重矩阵，形状为(m_samples, n_samples)
        """
        # 提取坐标
        train_coords = train_data[['经度', '纬度']].values
        test_coords = test_data[['经度', '纬度']].values
        
        # 提取时间
        train_times = train_data['记录时间']
        test_times = test_data['记录时间']
        
        # 计算空间权重和时间权重（传入完整数据用于占用率判断）
        spatial_weights = self._calculate_spatial_weights(train_coords, test_coords)
        temporal_weights = self._calculate_temporal_weights(train_times, test_times, train_data, test_data)
        
        # 时空联合权重（哈达玛积）
        spatiotemporal_weights = spatial_weights * temporal_weights
        
        # 归一化权重
        row_sums = spatiotemporal_weights.sum(axis=1, keepdims=True)
        spatiotemporal_weights = spatiotemporal_weights / row_sums
        
        return spatiotemporal_weights
    
    def fit_prophet(self, data, time_col='记录时间', target_col='功率(kw)'):
        """
        拟合Prophet时序模型
        
        Args:
            data: 包含时间和目标变量的DataFrame
            time_col: 时间列名
            target_col: 目标变量列名
            
        Returns:
            Prophet预测结果
        """
        print("开始拟合Prophet时序模型...")
        
        # 准备Prophet所需的数据格式
        prophet_data = data[[time_col, target_col]].copy()
        prophet_data.columns = ['ds', 'y']  # Prophet要求的列名
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(prophet_data['ds']):
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        # 初始化并拟合Prophet模型
        self.prophet_model = Prophet(
            seasonality_mode=self.prophet_params.get('seasonality_mode', 'multiplicative'),
            yearly_seasonality=self.prophet_params.get('yearly_seasonality', True),
            weekly_seasonality=self.prophet_params.get('weekly_seasonality', True),
            daily_seasonality=self.prophet_params.get('daily_seasonality', True),
            changepoint_prior_scale=self.prophet_params.get('changepoint_prior_scale', 0.05)
        )
        
        # 添加中国节假日效应
        self.prophet_model.add_country_holidays(country_name='CN')
        
        # 添加自定义节假日
        holidays = pd.DataFrame([
            # 2024年春节
            {'holiday': 'chinese_new_year', 'ds': pd.to_datetime('2024-02-10'), 'lower_window': 0, 'upper_window': 7},
            # 2024年清明节
            {'holiday': 'qingming', 'ds': pd.to_datetime('2024-04-04'), 'lower_window': 0, 'upper_window': 2},
            # 2024年劳动节
            {'holiday': 'labor_day', 'ds': pd.to_datetime('2024-05-01'), 'lower_window': 0, 'upper_window': 4},
            # 2024年端午节
            {'holiday': 'dragon_boat', 'ds': pd.to_datetime('2024-06-10'), 'lower_window': 0, 'upper_window': 2},
            # 2024年中秋节
            {'holiday': 'mid_autumn', 'ds': pd.to_datetime('2024-09-17'), 'lower_window': 0, 'upper_window': 2},
            # 2024年国庆节
            {'holiday': 'national_day', 'ds': pd.to_datetime('2024-10-01'), 'lower_window': 0, 'upper_window': 6},
            # 2025年春节
            {'holiday': 'chinese_new_year', 'ds': pd.to_datetime('2025-01-29'), 'lower_window': 0, 'upper_window': 7},
        ])
        self.prophet_model.add_holidays(holidays)
        
        # 拟合模型
        self.prophet_model.fit(prophet_data)
        
        # 生成预测
        future = self.prophet_model.make_future_dataframe(periods=30)  # 预测未来30天
        forecast = self.prophet_model.predict(future)
        
        # 保存预测结果和组件
        self.prophet_forecast = forecast
        self.prophet_components = self.prophet_model.plot_components(forecast)
        
        print("Prophet时序模型拟合完成")
        return forecast
    
    def fit_gtwr(self, data, features, target='功率(kw)', time_col='记录时间', coords_cols=['经度', '纬度'], test_size=0.2, random_state=42):
        """
        拟合GTWR模型
        
        Args:
            data: 包含特征、目标变量和时空信息的DataFrame
            features: 特征列表
            target: 目标变量列名
            time_col: 时间列名
            coords_cols: 坐标列名列表，如['经度', '纬度']
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            GTWR模型评估指标
        """
        print("\n开始拟合GTWR模型...")
        
        # 保存特征名称
        self.feature_names = features
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # 划分训练集和测试集
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集样本数: {len(train_data)}, 测试集样本数: {len(test_data)}")
        
        # 提取特征和目标变量
        X_train = train_data[features].values
        y_train = train_data[target].values
        X_test = test_data[features].values
        y_test = test_data[target].values
        
        # 计算时空权重矩阵
        weights = self._calculate_spatiotemporal_weights(train_data, test_data)
        
        # 为每个测试样本拟合局部加权回归模型
        n_test = len(test_data)
        n_features = len(features)
        self.coefficients = np.zeros((n_test, n_features))
        self.intercept = np.zeros(n_test)
        y_pred = np.zeros(n_test)
        self.local_r2 = np.zeros(n_test)
        
        print("正在拟合局部加权回归模型...")
        for i in range(n_test):
            # 获取当前测试样本的权重
            sample_weights = weights[i]
            
            # 加权最小二乘回归
            # 添加截距项
            X_train_with_intercept = np.column_stack([np.ones(len(X_train)), X_train])
            
            # 计算加权最小二乘解
            # (X^T W X)^(-1) X^T W y
            W = np.diag(sample_weights)
            XTW = X_train_with_intercept.T @ W
            XTWX = XTW @ X_train_with_intercept
            XTWy = XTW @ y_train
            try:
                beta = np.linalg.solve(XTWX, XTWy)
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                beta = np.linalg.pinv(XTWX) @ XTWy
            
            # 保存系数
            self.intercept[i] = beta[0]
            self.coefficients[i] = beta[1:]
            
            # 预测
            X_test_with_intercept = np.concatenate([[1], X_test[i]])
            y_pred[i] = X_test_with_intercept @ beta
            
            # 计算局部R2
            y_weighted = y_train * sample_weights
            y_mean_weighted = np.sum(y_weighted) / np.sum(sample_weights)
            tss = np.sum(sample_weights * (y_train - y_mean_weighted) ** 2)
            rss = np.sum(sample_weights * (y_train - X_train_with_intercept @ beta) ** 2)
            self.local_r2[i] = 1 - rss / tss if tss > 0 else 0
            
            # 进度显示
            if (i + 1) % 10 == 0 or i == n_test - 1:
                print(f"已完成: {i+1}/{n_test} ({(i+1)/n_test*100:.1f}%)")
        
        # 计算评估指标
        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2_score = r2_score(y_test, y_pred)
        
        # 计算调整后的R2
        n = len(y_test)
        p = len(features)
        self.r2_adj = 1 - (1 - self.r2_score) * (n - 1) / (n - p - 1)
        
        print(f"\nGTWR模型评估指标:")
        print(f"均方误差(MSE): {self.mse:.4f}")
        print(f"平均绝对误差(MAE): {self.mae:.4f}")
        print(f"决定系数(R²): {self.r2_score:.4f}")
        print(f"调整后的决定系数(Adjusted R²): {self.r2_adj:.4f}")
        print(f"局部R²均值: {np.mean(self.local_r2):.4f}")
        
        # 保存测试集预测结果
        test_data['gtwr_pred'] = y_pred
        
        return {
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2_score,
            'r2_adj': self.r2_adj,
            'local_r2_mean': np.mean(self.local_r2),
            'test_data': test_data
        }
    
    def fit_coupling_model(self, data, features, target='功率(kw)', time_col='记录时间', 
                          coords_cols=['经度', '纬度'], test_size=0.2, random_state=42):
        """
        拟合耦合模型
        
        Args:
            data: 包含特征、目标变量和时空信息的DataFrame
            features: 特征列表
            target: 目标变量列名
            time_col: 时间列名
            coords_cols: 坐标列名列表，如['经度', '纬度']
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            耦合模型评估指标
        """
        print("\n开始拟合Prophet-GTWR耦合模型...")
        print("="*50)
        
        # 1. 拟合Prophet模型
        prophet_forecast = self.fit_prophet(data, time_col=time_col, target_col=target)
        
        # 2. 将Prophet预测结果添加到数据中
        # 创建时间索引的映射
        prophet_forecast_dict = dict(zip(prophet_forecast['ds'], prophet_forecast['yhat']))
        
        # 添加Prophet预测作为GTWR的特征
        data['prophet_pred'] = data[time_col].map(prophet_forecast_dict)
        
        # 确保没有缺失值
        data = data.dropna(subset=['prophet_pred'])
        
        # 添加Prophet预测到特征列表
        features_with_prophet = features + ['prophet_pred']
        
        # 3. 拟合GTWR模型
        gtwr_results = self.fit_gtwr(
            data=data,
            features=features_with_prophet,
            target=target,
            time_col=time_col,
            coords_cols=coords_cols,
            test_size=test_size,
            random_state=random_state
        )
        
        test_data = gtwr_results['test_data']
        
        # 4. 实现双向反馈机制
        if self.feedback_mode == 'two_way':
            print("\n实施双向反馈机制...")
            
            # 获取GTWR系数中Prophet预测的权重
            prophet_coef_idx = features_with_prophet.index('prophet_pred')
            prophet_weights = self.coefficients[:, prophet_coef_idx]
            
            # 计算Prophet预测的调整系数
            adjustment_factors = np.clip(prophet_weights, 0.5, 1.5)
            
            # 调整Prophet预测
            test_data['prophet_adjusted'] = test_data['prophet_pred'] * adjustment_factors
            
            print(f"Prophet预测调整系数范围: [{adjustment_factors.min():.2f}, {adjustment_factors.max():.2f}]")
            print(f"平均调整系数: {adjustment_factors.mean():.2f}")
        else:
            # 单向模式下不调整Prophet预测
            test_data['prophet_adjusted'] = test_data['prophet_pred']
        
        # 5. 融合预测结果
        if self.dynamic_weight:
            print("\n使用动态权重融合预测结果...")
            
            # 计算每个样本的Prophet和GTWR预测误差
            prophet_errors = np.abs(test_data[target] - test_data['prophet_adjusted'])
            gtwr_errors = np.abs(test_data[target] - test_data['gtwr_pred'])
            
            # 基于误差计算动态权重
            # 当Prophet误差较小时，增加其权重；当GTWR误差较小时，增加其权重
            total_errors = prophet_errors + gtwr_errors
            prophet_weights = 1 - (prophet_errors / total_errors)
            gtwr_weights = 1 - (gtwr_errors / total_errors)
            
            # 归一化权重
            sum_weights = prophet_weights + gtwr_weights
            prophet_weights = prophet_weights / sum_weights
            gtwr_weights = gtwr_weights / sum_weights
            
            # 使用动态权重融合预测
            test_data['final_pred'] = prophet_weights * test_data['prophet_adjusted'] + \
                                     gtwr_weights * test_data['gtwr_pred']
            
            print(f"Prophet平均权重: {prophet_weights.mean():.4f}")
            print(f"GTWR平均权重: {gtwr_weights.mean():.4f}")
        else:
            print(f"\n使用固定权重融合预测结果 (alpha={self.alpha})...")
            
            # 使用固定权重融合预测
            test_data['final_pred'] = self.alpha * test_data['prophet_adjusted'] + \
                                     (1 - self.alpha) * test_data['gtwr_pred']
        
        # 6. 计算最终融合模型的评估指标
        final_mse = mean_squared_error(test_data[target], test_data['final_pred'])
        final_mae = mean_absolute_error(test_data[target], test_data['final_pred'])
        final_r2 = r2_score(test_data[target], test_data['final_pred'])
        
        # 计算相对于单一模型的改进
        prophet_mse = mean_squared_error(test_data[target], test_data['prophet_adjusted'])
        prophet_r2 = r2_score(test_data[target], test_data['prophet_adjusted'])
        
        mse_improvement = (prophet_mse - final_mse) / prophet_mse * 100
        r2_improvement = (final_r2 - prophet_r2) / (1 - prophet_r2) * 100 if prophet_r2 < 1 else 0
        
        print("\n耦合模型最终评估指标:")
        print(f"均方误差(MSE): {final_mse:.4f} (相比Prophet改进 {mse_improvement:.2f}%)")
        print(f"平均绝对误差(MAE): {final_mae:.4f}")
        print(f"决定系数(R²): {final_r2:.4f} (相比Prophet改进 {r2_improvement:.2f}%)")
        
        # 保存最终预测结果
        self.final_prediction = test_data
        
        return {
            'mse': final_mse,
            'mae': final_mae,
            'r2': final_r2,
            'mse_improvement': mse_improvement,
            'r2_improvement': r2_improvement,
            'test_data': test_data
        }
    
    def predict(self, new_data, features, time_col='记录时间', coords_cols=['经度', '纬度'], reference_data=None):
        """
        使用训练好的耦合模型进行预测
        
        Args:
            new_data: 新数据，包含特征和时空信息
            features: 特征列表
            time_col: 时间列名
            coords_cols: 坐标列名列表
            reference_data: 参考数据，用于计算时空权重，如果为None则使用训练数据
            
        Returns:
            包含预测结果的DataFrame
        """
        if self.prophet_model is None or self.coefficients is None:
            raise ValueError("模型尚未训练，请先调用fit_coupling_model方法")
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(new_data[time_col]):
            new_data[time_col] = pd.to_datetime(new_data[time_col])
        
        # 1. 使用Prophet模型预测
        prophet_data = pd.DataFrame({
            'ds': new_data[time_col]
        })
        prophet_pred = self.prophet_model.predict(prophet_data)
        
        # 将Prophet预测结果添加到新数据中
        prophet_pred_dict = dict(zip(prophet_pred['ds'], prophet_pred['yhat']))
        new_data['prophet_pred'] = new_data[time_col].map(prophet_pred_dict)
        
        # 2. 使用GTWR模型预测
        # 如果没有提供参考数据，使用最终预测结果作为参考
        if reference_data is None and self.final_prediction is not None:
            reference_data = self.final_prediction
        elif reference_data is None:
            raise ValueError("未提供参考数据且模型未训练完成，无法进行GTWR预测")
        
        # 计算时空权重
        weights = self._calculate_spatiotemporal_weights(reference_data, new_data)
        
        # 为每个新样本预测
        n_new = len(new_data)
        y_pred_gtwr = np.zeros(n_new)
        
        # 提取参考数据的特征和目标变量
        features_with_prophet = features + ['prophet_pred']
        X_ref = reference_data[features_with_prophet].values
        y_ref = reference_data['final_pred'].values if 'final_pred' in reference_data.columns else reference_data[features[0]].values
        
        print("正在进行GTWR预测...")
        for i in range(n_new):
            # 获取当前样本的权重
            sample_weights = weights[i]
            
            # 加权最小二乘回归
            X_ref_with_intercept = np.column_stack([np.ones(len(X_ref)), X_ref])
            
            # 计算加权最小二乘解
            W = np.diag(sample_weights)
            XTW = X_ref_with_intercept.T @ W
            XTWX = XTW @ X_ref_with_intercept
            XTWy = XTW @ y_ref
            
            try:
                beta = np.linalg.solve(XTWX, XTWy)
            except np.linalg.LinAlgError:
                # 如果矩阵奇异，使用伪逆
                beta = np.linalg.pinv(XTWX) @ XTWy
            
            # 预测
            X_new = new_data[features_with_prophet].values[i]
            X_new_with_intercept = np.concatenate([[1], X_new])
            y_pred_gtwr[i] = X_new_with_intercept @ beta
            
            # 进度显示
            if (i + 1) % 10 == 0 or i == n_new - 1:
                print(f"已完成: {i+1}/{n_new} ({(i+1)/n_new*100:.1f}%)")
        
        # 保存GTWR预测结果
        new_data['gtwr_pred'] = y_pred_gtwr
        
        # 3. 应用双向反馈机制和融合
        if self.feedback_mode == 'two_way':
            # 使用平均调整系数
            if hasattr(self, 'final_prediction') and self.final_prediction is not None:
                adjustment_factors = self.final_prediction['prophet_adjusted'] / self.final_prediction['prophet_pred']
                avg_adjustment = np.mean(adjustment_factors)
                new_data['prophet_adjusted'] = new_data['prophet_pred'] * avg_adjustment
            else:
                new_data['prophet_adjusted'] = new_data['prophet_pred']
        else:
            new_data['prophet_adjusted'] = new_data['prophet_pred']
        
        # 融合预测结果
        if self.dynamic_weight:
            # 使用平均权重
            if hasattr(self, 'prophet_weight_mean') and hasattr(self, 'gtwr_weight_mean'):
                prophet_weight = self.prophet_weight_mean
                gtwr_weight = self.gtwr_weight_mean
            else:
                prophet_weight = self.alpha
                gtwr_weight = 1 - self.alpha
            
            new_data['final_pred'] = prophet_weight * new_data['prophet_adjusted'] + \
                                    gtwr_weight * new_data['gtwr_pred']
        else:
            new_data['final_pred'] = self.alpha * new_data['prophet_adjusted'] + \
                                    (1 - self.alpha) * new_data['gtwr_pred']
        
        print("预测完成")
        return new_data
    
    def visualize_results(self, save_path=None):
        """
        可视化模型结果
        
        Args:
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            图表对象列表
        """
        if self.final_prediction is None:
            raise ValueError("模型尚未训练，请先调用fit_coupling_model方法")
        
        figures = []
        
        # 1. 预测结果对比图
        plt.figure(figsize=(12, 6))
        plt.plot(self.final_prediction['记录时间'], self.final_prediction['功率(kw)'], 'o-', label='实际值', alpha=0.7)
        plt.plot(self.final_prediction['记录时间'], self.final_prediction['prophet_pred'], 's--', label='Prophet预测', alpha=0.7)
        plt.plot(self.final_prediction['记录时间'], self.final_prediction['gtwr_pred'], '^--', label='GTWR预测', alpha=0.7)
        plt.plot(self.final_prediction['记录时间'], self.final_prediction['final_pred'], 'D-', label='耦合模型预测', linewidth=2)
        plt.xlabel('时间')
        plt.ylabel('功率(kW)')
        plt.title('Prophet-GTWR耦合模型预测结果对比')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'prediction_comparison.png'), dpi=300, bbox_inches='tight')
        
        figures.append(plt.gcf())
        
        # 2. 预测误差分布图
        plt.figure(figsize=(12, 6))
        prophet_error = self.final_prediction['功率(kw)'] - self.final_prediction['prophet_pred']
        gtwr_error = self.final_prediction['功率(kw)'] - self.final_prediction['gtwr_pred']
        coupling_error = self.final_prediction['功率(kw)'] - self.final_prediction['final_pred']
        
        plt.hist(prophet_error, bins=20, alpha=0.5, label='Prophet误差')
        plt.hist(gtwr_error, bins=20, alpha=0.5, label='GTWR误差')
        plt.hist(coupling_error, bins=20, alpha=0.5, label='耦合模型误差')
        plt.xlabel('预测误差')
        plt.ylabel('频数')
        plt.title('预测误差分布对比')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        
        figures.append(plt.gcf())
        
        # 3. 空间异质性可视化
        if '经度' in self.final_prediction.columns and '纬度' in self.final_prediction.columns:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                self.final_prediction['经度'], 
                self.final_prediction['纬度'],
                c=self.local_r2,
                cmap='viridis',
                s=50,
                alpha=0.8
            )
            plt.colorbar(scatter, label='局部R²')
            plt.xlabel('经度')
            plt.ylabel('纬度')
            plt.title('GTWR模型局部R²的空间分布')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(os.path.join(save_path, 'spatial_heterogeneity.png'), dpi=300, bbox_inches='tight')
            
            figures.append(plt.gcf())
        
        # 4. 系数空间变异性可视化
        if self.feature_names and len(self.feature_names) > 0:
            plt.figure(figsize=(12, 8))
            
            # 绘制箱线图
            plt.boxplot([self.coefficients[:, i] for i in range(len(self.feature_names))], 
                        labels=self.feature_names)
            
            plt.xlabel('特征')
            plt.ylabel('系数值')
            plt.title('GTWR模型系数的空间变异性')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(os.path.join(save_path, 'coefficient_variation.png'), dpi=300, bbox_inches='tight')
            
            figures.append(plt.gcf())
        
        # 5. Prophet组件分解图
        if self.prophet_components:
            figures.append(self.prophet_components)
            
            if save_path:
                self.prophet_components.savefig(os.path.join(save_path, 'prophet_components.png'), 
                                              dpi=300, bbox_inches='tight')
        
        return figures
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 准备要保存的数据
        model_data = {
            'prophet_params': self.prophet_params,
            'gtwr_params': self.gtwr_params,
            'coupling_params': self.coupling_params,
            'spatial_bandwidth': self.spatial_bandwidth,
            'temporal_bandwidth': self.temporal_bandwidth,
            'coefficients': self.coefficients.tolist() if self.coefficients is not None else None,
            'intercept': self.intercept.tolist() if self.intercept is not None else None,
            'feature_names': self.feature_names,
            'r2_score': self.r2_score,
            'r2_adj': self.r2_adj,
            'mse': self.mse,
            'mae': self.mae
        }
        
        # 保存模型数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=4)
        
        # 如果有Prophet模型，单独保存
        if self.prophet_model:
            prophet_path = os.path.splitext(filepath)[0] + '_prophet.json'
            with open(prophet_path, 'w') as f:
                self.prophet_model.serialize_posterior(f)
        
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型对象
        """
        # 加载模型数据
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # 创建模型实例
        model = cls(
            prophet_params=model_data.get('prophet_params'),
            gtwr_params=model_data.get('gtwr_params'),
            coupling_params=model_data.get('coupling_params')
        )
        
        # 恢复模型参数
        model.spatial_bandwidth = model_data.get('spatial_bandwidth')
        model.temporal_bandwidth = model_data.get('temporal_bandwidth')
        model.coefficients = np.array(model_data.get('coefficients')) if model_data.get('coefficients') else None
        model.intercept = np.array(model_data.get('intercept')) if model_data.get('intercept') else None
        model.feature_names = model_data.get('feature_names')
        model.r2_score = model_data.get('r2_score')
        model.r2_adj = model_data.get('r2_adj')
        model.mse = model_data.get('mse')
        model.mae = model_data.get('mae')
        
        # 尝试加载Prophet模型
        prophet_path = os.path.splitext(filepath)[0] + '_prophet.json'
        if os.path.exists(prophet_path):
            with open(prophet_path, 'r') as f:
                model.prophet_model = Prophet.deserialize(json.load(f))
        
        print(f"模型已从{filepath}加载")
        return model