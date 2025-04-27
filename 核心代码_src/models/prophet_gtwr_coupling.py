# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型 - 耦合模型模块
功能：实现Prophet时序模型和GTWR空间模型的耦合
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from .prophet_model import ProphetModel
from .gtwr_model import GTWRModel

class ProphetGTWRCoupling:
    """
    Prophet-GTWR耦合模型类
    结合Facebook Prophet时序预测和地理与时间加权回归(GTWR)的时空耦合模型
    """
    
    def __init__(self, prophet_params=None, gtwr_params=None, coupling_params=None):
        """
        初始化Prophet-GTWR耦合模型
        
        Args:
            prophet_params: Prophet模型参数
            gtwr_params: GTWR模型参数
            coupling_params: 耦合参数，可包含：
                - alpha: 融合权重参数，范围[0,1]，默认0.5
                - feedback_mode: 反馈模式，'one_way'或'two_way'，默认'two_way'
                - dynamic_weight: 是否使用动态权重，默认True
        """
        self.prophet_model = ProphetModel(prophet_params)
        self.gtwr_model = GTWRModel(gtwr_params)
        self.coupling_params = coupling_params or {}
        
        # 设置耦合参数
        self.alpha = self.coupling_params.get('alpha', 0.5)  # 融合权重参数
        self.feedback_mode = self.coupling_params.get('feedback_mode', 'two_way')  # 反馈模式
        self.dynamic_weight = self.coupling_params.get('dynamic_weight', True)  # 是否使用动态权重
        
        # 模型状态
        self.is_fitted = False
        self.prophet_forecast = None
        self.gtwr_prediction = None
        self.final_prediction = None
        self.metrics = None
    
    def fit(self, time_series_data, spatial_data, coords_cols, time_col, target_col, feature_cols=None):
        """
        训练耦合模型
        
        Args:
            time_series_data: 时间序列数据，DataFrame格式
            spatial_data: 空间数据，DataFrame格式
            coords_cols: 坐标列名列表，如['经度', '纬度']
            time_col: 时间列名
            target_col: 目标变量列名
            feature_cols: 特征列名列表，如果为None则使用所有数值列
            
        Returns:
            训练好的耦合模型
        """
        print("开始训练Prophet-GTWR耦合模型...")
        
        # 1. 训练Prophet时序模型
        print("\n第一阶段: 训练Prophet时序模型")
        prophet_data = time_series_data[[time_col, target_col]].copy()
        self.prophet_model.fit(prophet_data, time_col=time_col, target_col=target_col)
        
        # 生成Prophet预测
        prophet_forecast = self.prophet_model.predict(include_history=True)
        self.prophet_forecast = prophet_forecast
        
        # 2. 准备GTWR模型的输入数据
        print("\n第二阶段: 准备GTWR模型输入")
        
        # 合并空间数据和时间序列数据
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(spatial_data[time_col]):
            spatial_data[time_col] = pd.to_datetime(spatial_data[time_col])
        
        # 将Prophet预测结果添加到空间数据中
        # 创建时间索引的映射
        prophet_forecast_dict = dict(zip(prophet_forecast['ds'], prophet_forecast['yhat']))
        
        # 添加Prophet预测作为GTWR的特征
        spatial_data['prophet_pred'] = spatial_data[time_col].map(prophet_forecast_dict)
        
        # 确定特征列
        if feature_cols is None:
            # 使用所有数值列作为特征，排除目标变量、时间和坐标列
            exclude_cols = [target_col, time_col] + coords_cols
            feature_cols = [col for col in spatial_data.columns 
                           if col not in exclude_cols and pd.api.types.is_numeric_dtype(spatial_data[col])]
        
        # 添加Prophet预测到特征列
        if 'prophet_pred' not in feature_cols:
            feature_cols.append('prophet_pred')
        
        # 3. 训练GTWR模型
        print("\n第三阶段: 训练GTWR模型")
        X = spatial_data[feature_cols]
        y = spatial_data[target_col]
        coords = spatial_data[coords_cols]
        times = spatial_data[time_col]
        
        self.gtwr_model.fit(X, y, coords, times, coords_cols=coords_cols, time_col=time_col)
        
        # 4. 如果是双向反馈模式，将GTWR的结果反馈给Prophet
        if self.feedback_mode == 'two_way':
            print("\n第四阶段: 实施双向反馈")
            # 获取GTWR的局部系数
            local_coefs = self.gtwr_model.local_coefficients()
            
            # 提取Prophet预测特征的系数
            if 'prophet_pred' in feature_cols:
                prophet_coef = local_coefs['prophet_pred'].values
                
                # 根据GTWR系数调整Prophet预测
                # 这里使用一个简单的加权方法，实际应用中可能需要更复杂的方法
                adjusted_forecast = prophet_forecast.copy()
                adjusted_forecast['yhat'] = adjusted_forecast['yhat'] * np.mean(prophet_coef)
                
                # 重新训练Prophet模型，使用调整后的预测作为先验
                # 这里简化处理，实际应用中可能需要更复杂的方法
                self.prophet_forecast = adjusted_forecast
        
        # 5. 生成最终预测
        print("\n第五阶段: 生成耦合预测")
        self.gtwr_prediction = self.gtwr_model.predict(X, coords, times)
        
        # 根据权重参数融合两个模型的预测
        if self.dynamic_weight:
            # 动态权重：基于两个模型的预测误差
            prophet_error = mean_squared_error(y, spatial_data['prophet_pred'])
            gtwr_error = mean_squared_error(y, self.gtwr_prediction)
            
            # 误差越小，权重越大
            total_error = prophet_error + gtwr_error
            if total_error > 0:
                self.alpha = gtwr_error / total_error  # Prophet的权重
            else:
                self.alpha = 0.5  # 默认权重
            
            print(f"动态权重: Prophet权重={self.alpha:.4f}, GTWR权重={1-self.alpha:.4f}")
        
        # 融合预测
        self.final_prediction = self.alpha * spatial_data['prophet_pred'] + (1 - self.alpha) * self.gtwr_prediction
        
        # 计算评估指标
        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y, self.final_prediction)),
            'r2': r2_score(y, self.final_prediction),
            'prophet_rmse': np.sqrt(mean_squared_error(y, spatial_data['prophet_pred'])),
            'prophet_r2': r2_score(y, spatial_data['prophet_pred']),
            'gtwr_rmse': np.sqrt(mean_squared_error(y, self.gtwr_prediction)),
            'gtwr_r2': r2_score(y, self.gtwr_prediction)
        }
        
        print("\nProphet-GTWR耦合模型训练完成")
        print("评估指标:")
        print(f"耦合模型 - RMSE: {self.metrics['rmse']:.4f}, R²: {self.metrics['r2']:.4f}")
        print(f"Prophet模型 - RMSE: {self.metrics['prophet_rmse']:.4f}, R²: {self.metrics['prophet_r2']:.4f}")
        print(f"GTWR模型 - RMSE: {self.metrics['gtwr_rmse']:.4f}, R²: {self.metrics['gtwr_r2']:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, future_time_series=None, future_spatial_data=None, periods=30, freq='D'):
        """
        生成预测结果
        
        Args:
            future_time_series: 未来时间序列数据，如果为None则自动生成
            future_spatial_data: 未来空间数据，如果为None则使用训练数据的空间信息
            periods: 预测未来的时间段数，当future_time_series为None时使用
            freq: 时间频率，如'D'(天),'H'(小时)等，当future_time_series为None时使用
            
        Returns:
            预测结果DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        print("生成Prophet-GTWR耦合模型预测...")
        
        # 1. 生成Prophet预测
        if future_time_series is None:
            # 自动生成未来时间序列
            prophet_future = self.prophet_model.predict(periods=periods, freq=freq)
        else:
            # 使用提供的未来时间序列
            prophet_future = self.prophet_model.predict(future_df=future_time_series)
        
        # 2. 准备GTWR预测的输入数据
        if future_spatial_data is None:
            # 使用训练数据的空间信息，但更新时间和Prophet预测
            # 这里简化处理，实际应用中可能需要更复杂的方法
            print("警告: 未提供未来空间数据，使用训练数据的空间信息进行预测")
            return prophet_future
        
        # 将Prophet预测添加到未来空间数据中
        prophet_future_dict = dict(zip(prophet_future['ds'], prophet_future['yhat']))
        future_spatial_data['prophet_pred'] = future_spatial_data[self.time_col].map(prophet_future_dict)
        
        # 3. 生成GTWR预测
        X_future = future_spatial_data[self.feature_cols]
        coords_future = future_spatial_data[self.coords_cols]
        times_future = future_spatial_data[self.time_col]
        
        gtwr_future = self.gtwr_model.predict(X_future, coords_future, times_future)
        
        # 4. 融合预测
        final_future = self.alpha * future_spatial_data['prophet_pred'] + (1 - self.alpha) * gtwr_future
        
        # 创建结果DataFrame
        result = future_spatial_data.copy()
        result['prophet_pred'] = future_spatial_data['prophet_pred']
        result['gtwr_pred'] = gtwr_future
        result['coupled_pred'] = final_future
        
        print("预测完成")
        return result
    
    def plot_predictions(self, actual=None, figsize=(12, 6)):
        """
        绘制预测结果对比图
        
        Args:
            actual: 实际值，如果为None则使用训练数据
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制实际值
        if actual is not None:
            ax.plot(actual.index, actual.values, 'k.', label='实际值')
        
        # 绘制Prophet预测
        if self.prophet_forecast is not None:
            ax.plot(self.prophet_forecast['ds'], self.prophet_forecast['yhat'], 'b-', label='Prophet预测')
        
        # 绘制GTWR预测
        if self.gtwr_prediction is not None:
            # 这里简化处理，实际应用中需要考虑时间索引
            ax.plot(self.prophet_forecast['ds'], self.gtwr_prediction, 'g-', label='GTWR预测')
        
        # 绘制耦合预测
        if self.final_prediction is not None:
            ax.plot(self.prophet_forecast['ds'], self.final_prediction, 'r-', label='耦合预测')
        
        # 设置图形属性
        ax.set_xlabel('时间')
        ax.set_ylabel('值')
        ax.set_title('Prophet-GTWR耦合模型预测结果')
        ax.legend()
        
        return fig
    
    def plot_spatial_prediction(self, prediction_data, coords_cols, value_col, 
                              figsize=(12, 8), cmap='viridis'):
        """
        绘制空间预测分布图
        
        Args:
            prediction_data: 预测结果数据，包含坐标和预测值
            coords_cols: 坐标列名列表，如['经度', '纬度']
            value_col: 预测值列名
            figsize: 图形大小
            cmap: 颜色映射
            
        Returns:
            matplotlib图形对象
        """
        import geopandas as gpd
        from shapely.geometry import Point
        
        # 创建GeoDataFrame
        geometry = [Point(xy) for xy in zip(prediction_data[coords_cols[0]], prediction_data[coords_cols[1]])]
        gdf = gpd.GeoDataFrame(prediction_data, geometry=geometry, crs='EPSG:4326')
        
        # 绘制地图
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用分位数分类
        vmin, vmax = np.percentile(prediction_data[value_col], [5, 95])
        
        # 绘制预测分布
        gdf.plot(
            column=value_col,
            cmap=cmap,
            legend=True,
            ax=ax,
            vmin=vmin,
            vmax=vmax
        )
        
        # 设置标题和标签
        ax.set_title(f'{value_col}空间分布')
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        
        return fig
    
    def optimize_layout(self, grid_data, constraints=None, objective='coverage', n_stations=None):
        """
        优化充电桩布局
        
        Args:
            grid_data: 网格数据，包含空间信息和预测需求
            constraints: 布局约束条件
            objective: 优化目标，'coverage'(覆盖率)或'efficiency'(效率)
            n_stations: 要布局的充电桩数量，如果为None则自动确定
            
        Returns:
            优化后的布局方案
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        print("开始优化充电桩布局...")
        
        # 1. 识别供需缺口
        # 这里简化处理，实际应用中需要更复杂的方法
        if 'demand' in grid_data.columns and 'supply' in grid_data.columns:
            grid_data['gap'] = grid_data['demand'] - grid_data['supply']
        else:
            # 使用预测值作为需求
            grid_data['gap'] = grid_data['coupled_pred']
        
        # 2. 根据优化目标确定布局策略
        if objective == 'coverage':
            # 覆盖率优化：优先考虑高需求区域
            priority = grid_data['gap']
        elif objective == 'efficiency':
            # 效率优化：考虑需求和成本的比率
            if 'cost' in grid_data.columns:
                priority = grid_data['gap'] / grid_data['cost']
            else:
                priority = grid_data['gap']
        else:
            raise ValueError(f"不支持的优化目标: {objective}")
        
        # 3. 应用约束条件
        if constraints is not None:
            # 应用各种约束条件
            # 这里简化处理，实际应用中需要更复杂的方法
            pass
        
        # 4. 确定布局方案
        # 按优先级排序
        grid_data['priority'] = priority
        sorted_grids = grid_data.sort_values('priority', ascending=False)
        
        # 确定布局数量
        if n_stations is None:
            # 自动确定数量：选择所有正缺口的网格
            layout_grids = sorted_grids[sorted_grids['gap'] > 0]
        else:
            # 使用指定数量
            layout_grids = sorted_grids.head(n_stations)
        
        print(f"布局优化完成，建议在{len(layout_grids)}个网格中布局充电桩")
        
        return layout_grids
    
    def save_model(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 模型保存目录
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 确保目录存在
        os.makedirs(filepath, exist_ok=True)
        
        # 保存Prophet模型
        prophet_path = os.path.join(filepath, 'prophet_model.json')
        self.prophet_model.save_model(prophet_path)
        
        # 保存GTWR模型参数
        # 由于GTWR模型较复杂，这里简化处理，只保存参数
        gtwr_params = {
            'spatial_bandwidth': self.gtwr_model.spatial_bandwidth,
            'temporal_bandwidth': self.gtwr_model.temporal_bandwidth,
            'kernel_function': self.gtwr_model.kernel_function,
            'time_decay_lambda': self.gtwr_model.time_decay_lambda
        }
        
        gtwr_path = os.path.join(filepath, 'gtwr_params.json')
        with open(gtwr_path, 'w') as f:
            json.dump(gtwr_params, f)
        
        # 保存耦合参数
        coupling_path = os.path.join(filepath, 'coupling_params.json')
        with open(coupling_path, 'w') as f:
            json.dump({
                'alpha': self.alpha,
                'feedback_mode': self.feedback_mode,
                'dynamic_weight': self.dynamic_weight,
                'metrics': self.metrics
            }, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件目录
            
        Returns:
            加载的模型
        """
        # 加载Prophet模型
        prophet_path = os.path.join(filepath, 'prophet_model.json')
        self.prophet_model.load_model(prophet_path)
        
        # 加载GTWR模型参数
        gtwr_path = os.path.join(filepath, 'gtwr_params.json')
        with open(gtwr_path, 'r') as f:
            gtwr_params = json.load(f)
        
        # 更新GTWR模型参数
        self.gtwr_model.spatial_bandwidth = gtwr_params['spatial_bandwidth']
        self.gtwr_model.temporal_bandwidth = gtwr_params['temporal_bandwidth']
        self.gtwr_model.kernel_function = gtwr_params['kernel_function']
        self.gtwr_model.time_decay_lambda = gtwr_params['time_decay_lambda']
        
        # 加载耦合参数
        coupling_path = os.path.join(filepath, 'coupling_params.json')
        with open(coupling_path, 'r') as f:
            coupling_params = json.load(f)
        
        # 更新耦合参数
        self.alpha = coupling_params['alpha']
        self.feedback_mode = coupling_params['feedback_mode']
        self.dynamic_weight = coupling_params['dynamic_weight']
        self.metrics = coupling_params['metrics']
        
        self.is_fitted = True
        
        print(f"模型已从{filepath}加载")
        return self

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    # 时间序列数据
    dates = pd.date_range(start='2023-01-01', end='2024-04-01', freq='D')
    values = np.sin(np.arange(len(dates)) * 0.1) * 10 + np.random.normal(0, 1, len(dates)) + 50
    
    time_series_data = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    # 空间数据
    n_samples = 100
    np.random.seed(42)
    
    # 空间坐标（经纬度）
    coords = np.random.uniform(low=[113.8, 22.4], high=[114.5, 22.8], size=(n_samples, 2))
    
    # 时间（随机选择时间序列中的时间点）
    times_idx = np.random.choice(len(dates), n_samples)
    times = dates[times_idx]
    
    # 特征
    X1 = np.random.normal(0, 1, n_samples)  # 随机特征1
    X2 = np.random.normal(0, 1, n_samples)  # 随机特征2
    
    # 目标变量（带空间效应）
    y = 2 + 0.5 * X1 + 0.3 * X2 + 0.1 * coords[:, 0] + 0.2 * coords[:, 1] + np.random.normal(0, 0.5, n_samples)
    
    # 创建空间数据DataFrame
    spatial_data = pd.DataFrame({
        '经度': coords[:, 0],
        '纬度': coords[:, 1],
        '记录时间': times,
        'X1': X1,
        'X2': X2,
        'y': y
    })
    
    # 创建耦合模型
    model = ProphetGTWRCoupling(
        prophet_params={
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        gtwr_params={
            'kernel_function': 'gaussian',
            'optimization_method': 'cv'
        },
        coupling_params={
            'alpha': 0.5,
            'feedback_mode': 'two_way',
            'dynamic_weight': True
        }
    )
    
    # 训练模型
    model.fit(
        time_series_data=time_series_data,
        spatial_data=spatial_data,
        coords_cols=['经度', '纬度'],
        time_col='记录时间',
        target_col='y',
        feature_cols=['X1', 'X2']
    )
    
    # 预测
    future_dates = pd.date_range(start='2024-04-02', end='2024-05-01', freq='D')
    future_time_series = pd.DataFrame({'ds': future_dates})
    
    # 创建未来空间数据（简化示例）
    future_spatial_data = spatial_data.copy()
    future_spatial_data['记录时间'] = np.random.choice(future_dates, n_samples)
    
    # 生成预测
    predictions = model.predict(future_time_series=future_time_series, future_spatial_data=future_spatial_data)
    
    # 绘制预测结果
    fig = model.plot_predictions()
    plt.show()