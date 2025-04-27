# -*- coding: utf-8 -*-
"""
日期：2025/4/22 10:47
Prophet-GTWR耦合模型 - Prophet时序模型模块
功能：时间序列预测，识别趋势和季节性模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # 忽略Prophet的警告

class ProphetModel:
    """
    Prophet时序模型类
    用于时间序列预测，识别趋势和季节性模式
    """
    
    def __init__(self, params=None):
        """
        初始化Prophet模型
        
        Args:
            params: Prophet模型参数字典
        """
        self.params = params or {}
        self.model = None
        self.forecast = None
        self.components = None
        self.metrics = None
        self.cv_results = None
        
    def _add_chinese_holidays(self):
        """
        添加中国主要节假日到Prophet模型
        """
        # 使用Prophet内置的中国节假日
        self.model.add_country_holidays(country_name='CN')
        
        # 定义2024-2025年中国主要节假日（额外补充）
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
        
        # 添加自定义节假日
        self.model.add_holidays(holidays)
    
    def fit(self, time_series_data, time_col='ds', target_col='y', add_holidays=True):
        """
        训练Prophet模型
        
        Args:
            time_series_data: 时间序列数据，DataFrame格式
            time_col: 时间列名
            target_col: 目标变量列名
            add_holidays: 是否添加节假日效应
            
        Returns:
            训练好的模型
        """
        # 准备Prophet所需的数据格式
        df = time_series_data.copy()
        
        # 重命名列以符合Prophet要求
        if time_col != 'ds':
            df = df.rename(columns={time_col: 'ds'})
        if target_col != 'y':
            df = df.rename(columns={target_col: 'y'})
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # 初始化Prophet模型
        self.model = Prophet(**self.params)
        
        # 添加节假日效应
        if add_holidays:
            self._add_chinese_holidays()
        
        # 添加额外的季节性（如果在参数中指定）
        if 'add_seasonalities' in self.params:
            for name, seasonality in self.params['add_seasonalities'].items():
                self.model.add_seasonality(
                    name=name,
                    period=seasonality['period'],
                    fourier_order=seasonality.get('fourier_order', 5)
                )
        
        # 添加额外的回归变量（如果在数据中存在）
        if 'add_regressors' in self.params:
            for regressor in self.params['add_regressors']:
                if regressor in df.columns:
                    self.model.add_regressor(regressor)
        
        # 拟合模型
        print("开始训练Prophet模型...")
        self.model.fit(df)
        print("Prophet模型训练完成")
        
        return self.model
    
    def predict(self, periods=30, freq='D', include_history=True, future_df=None):
        """
        生成预测结果
        
        Args:
            periods: 预测未来的时间段数
            freq: 时间频率，如'D'(天),'H'(小时)等
            include_history: 是否包含历史数据的预测
            future_df: 自定义的未来数据框，如果为None则自动生成
            
        Returns:
            预测结果DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 生成未来数据框
        if future_df is None:
            future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
        else:
            future = future_df.copy()
            # 确保future_df包含'ds'列
            if 'ds' not in future.columns:
                raise ValueError("future_df必须包含'ds'列")
        
        # 生成预测
        print(f"预测未来{periods}个{freq}时间单位...")
        forecast = self.model.predict(future)
        
        # 保存预测结果
        self.forecast = forecast
        
        # 生成组件图
        self.components = self.model.plot_components(forecast)
        
        return forecast
    
    def plot_forecast(self, uncertainty=True, figsize=(12, 6)):
        """
        绘制预测结果图
        
        Args:
            uncertainty: 是否显示不确定性区间
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        if self.forecast is None:
            raise ValueError("尚未生成预测结果，请先调用predict方法")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        
        # 绘制实际值和预测值
        ax.plot(self.forecast['ds'], self.forecast['y'], 'k.', label='实际值')
        ax.plot(self.forecast['ds'], self.forecast['yhat'], 'b-', label='预测值')
        
        # 绘制不确定性区间
        if uncertainty:
            ax.fill_between(self.forecast['ds'], self.forecast['yhat_lower'], 
                           self.forecast['yhat_upper'], color='b', alpha=0.2, label='95%置信区间')
        
        # 设置图形属性
        ax.set_xlabel('日期')
        ax.set_ylabel('值')
        ax.set_title('Prophet时序预测结果')
        ax.legend()
        
        return fig
    
    def cross_validate(self, initial='730 days', period='180 days', horizon='365 days', parallel=None):
        """
        交叉验证
        
        Args:
            initial: 初始训练期
            period: 每次验证的间隔
            horizon: 预测范围
            parallel: 并行处理的方式，如'processes'或None
            
        Returns:
            交叉验证结果DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        print("开始交叉验证...")
        # 执行交叉验证
        df_cv = cross_validation(
            model=self.model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=parallel
        )
        
        # 计算性能指标
        df_p = performance_metrics(df_cv)
        
        # 保存结果
        self.cv_results = df_cv
        self.metrics = df_p
        
        print("交叉验证完成")
        print("性能指标:")
        print(df_p[['horizon', 'rmse', 'mae', 'mape']].tail())
        
        return df_cv, df_p
    
    def plot_cv_metrics(self, metric='rmse', figsize=(10, 6)):
        """
        绘制交叉验证指标图
        
        Args:
            metric: 要绘制的指标，如'rmse', 'mae', 'mape'
            figsize: 图形大小
            
        Returns:
            matplotlib图形对象
        """
        if self.cv_results is None:
            raise ValueError("尚未执行交叉验证，请先调用cross_validate方法")
        
        fig = plot_cross_validation_metric(self.cv_results, metric=metric, figsize=figsize)
        return fig
    
    def evaluate(self, test_data, time_col='ds', target_col='y'):
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            time_col: 时间列名
            target_col: 目标变量列名
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 准备测试数据
        df_test = test_data.copy()
        
        # 重命名列以符合Prophet要求
        if time_col != 'ds':
            df_test = df_test.rename(columns={time_col: 'ds'})
        if target_col != 'y':
            df_test = df_test.rename(columns={target_col: 'y'})
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(df_test['ds']):
            df_test['ds'] = pd.to_datetime(df_test['ds'])
        
        # 生成预测
        forecast = self.model.predict(df_test)
        
        # 合并实际值和预测值
        evaluation = pd.merge(df_test, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
        
        # 计算评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_true = evaluation['y']
        y_pred = evaluation['yhat']
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 计算预测区间覆盖率
        coverage = np.mean((y_true >= evaluation['yhat_lower']) & 
                          (y_true <= evaluation['yhat_upper'])) * 100
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'coverage': coverage
        }
        
        print("模型评估指标:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        保存模型
        
        Args:
            filepath: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        with open(filepath, 'w') as f:
            json.dump(model_to_json(self.model), f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型
        """
        # 加载模型
        with open(filepath, 'r') as f:
            self.model = model_from_json(json.load(f))
        
        print(f"模型已从{filepath}加载")
        return self.model
    
    def get_seasonality_components(self):
        """
        获取季节性组件
        
        Returns:
            季节性组件字典
        """
        if self.forecast is None:
            raise ValueError("尚未生成预测结果，请先调用predict方法")
        
        # 提取季节性组件
        components = {}
        for col in self.forecast.columns:
            if col.startswith('trend') or col.startswith('seasonal') or col.startswith('holidays'):
                components[col] = self.forecast[col].values
        
        return components

# 辅助函数：将Prophet模型转换为JSON
def model_to_json(model):
    """
    将Prophet模型转换为JSON格式
    
    Args:
        model: Prophet模型
        
    Returns:
        JSON格式的模型
    """
    return model.to_json()

# 辅助函数：从JSON加载Prophet模型
def model_from_json(json_str):
    """
    从JSON加载Prophet模型
    
    Args:
        json_str: JSON格式的模型
        
    Returns:
        Prophet模型
    """
    model = Prophet.from_json(json_str)
    return model

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range(start='2023-01-01', end='2024-04-01', freq='D')
    values = np.sin(np.arange(len(dates)) * 0.1) * 10 + np.random.normal(0, 1, len(dates)) + 50
    
    # 创建DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    # 创建Prophet模型
    model = ProphetModel({
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False
    })
    
    # 训练模型
    model.fit(df)
    
    # 预测
    forecast = model.predict(periods=90)
    
    # 绘制预测结果
    fig = model.plot_forecast()
    plt.show()