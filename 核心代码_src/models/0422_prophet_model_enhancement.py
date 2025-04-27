# -*- coding: utf-8 -*-
"""
日期：2023年4月22日
功能：Prophet模型增强实现，增加对深圳统计年鉴数据的支持，并实现模型鲁棒性测试
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import os
import json
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')  # 忽略Prophet的警告

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
result_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '项目产出_results', 'figures'))

# 确保结果目录存在
os.makedirs(result_dir, exist_ok=True)

class EnhancedProphetModel:
    """
    增强版Prophet模型类
    增加对深圳统计年鉴数据的支持，并实现模型鲁棒性测试
    """
    
    def __init__(self, params=None):
        """
        初始化增强版Prophet模型
        
        Args:
            params: Prophet模型参数字典
        """
        self.params = params or {}
        self.model = None
        self.forecast = None
        self.components = None
        self.metrics = None
        self.cv_results = None
        self.robustness_results = None
        
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
    
    def _add_economic_regressors(self, time_series_data, economic_data):
        """
        添加经济指标作为外部回归变量
        
        Args:
            time_series_data: 时间序列数据
            economic_data: 经济指标数据
        
        Returns:
            添加了经济指标的时间序列数据
        """
        # 创建经济指标的时间序列
        # 这里假设经济指标是年度数据，需要转换为日度数据
        
        # 获取时间范围
        min_date = time_series_data['ds'].min()
        max_date = time_series_data['ds'].max()
        
        # 创建日期范围
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # 创建经济指标的日度数据框
        economic_ts = pd.DataFrame({'ds': date_range})
        
        # 添加GDP增长率（模拟数据，实际应从统计年鉴中提取）
        # 这里假设GDP增长率在一年内是恒定的
        gdp_growth = 0.065  # 6.5%的GDP增长率
        economic_ts['gdp_growth'] = gdp_growth
        
        # 添加居民消费价格指数CPI（模拟数据，实际应从统计年鉴中提取）
        # 这里假设CPI在一年内有季节性波动
        economic_ts['cpi'] = 102.5 + 1.5 * np.sin(2 * np.pi * (economic_ts['ds'].dt.dayofyear / 365))
        
        # 添加电力消费增长率（模拟数据，实际应从电网负荷报告中提取）
        # 这里假设电力消费增长率在一年内有季节性波动
        economic_ts['power_consumption_growth'] = 0.08 + 0.03 * np.sin(2 * np.pi * (economic_ts['ds'].dt.dayofyear / 365))
        
        # 合并到原始时间序列数据
        enhanced_data = pd.merge(time_series_data, economic_ts, on='ds', how='left')
        
        return enhanced_data
    
    def fit(self, time_series_data, economic_data=None, time_col='ds', target_col='y', add_holidays=True, add_regressors=True):
        """
        训练增强版Prophet模型
        
        Args:
            time_series_data: 时间序列数据，DataFrame格式
            economic_data: 经济指标数据，DataFrame格式
            time_col: 时间列名
            target_col: 目标变量列名
            add_holidays: 是否添加节假日效应
            add_regressors: 是否添加外部回归变量
            
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
        
        # 添加经济指标作为外部回归变量
        if add_regressors and economic_data is not None:
            df = self._add_economic_regressors(df, economic_data)
        
        # 初始化Prophet模型
        self.model = Prophet(**self.params)
        
        # 添加外部回归变量
        if add_regressors and economic_data is not None:
            self.model.add_regressor('gdp_growth')
            self.model.add_regressor('cpi')
            self.model.add_regressor('power_consumption_growth')
        
        # 添加节假日效应
        if add_holidays:
            self._add_chinese_holidays()
        
        # 添加自定义季节性
        # 添加周末效应（周六和周日的充电模式可能不同）
        self.model.add_seasonality(name='weekend', period=7, fourier_order=3, condition_name='is_weekend')
        
        # 添加工作日/周末条件
        df['is_weekend'] = df['ds'].dt.dayofweek >= 5
        
        # 训练模型
        self.model.fit(df)
        
        return self.model
    
    def predict(self, periods=30, freq='D', include_history=False, future_df=None):
        """
        生成预测结果
        
        Args:
            periods: 预测期数
            freq: 预测频率，默认为天
            include_history: 是否包含历史数据
            future_df: 自定义未来数据框
            
        Returns:
            预测结果DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 生成未来数据框
        if future_df is None:
            future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
            
            # 添加工作日/周末条件
            future['is_weekend'] = future['ds'].dt.dayofweek >= 5
        else:
            future = future_df
        
        # 生成预测
        self.forecast = self.model.predict(future)
        
        # 提取组件
        self.components = self.model.plot_components(self.forecast)
        
        return self.forecast
    
    def cross_validate(self, initial='730 days', period='180 days', horizon='365 days', parallel=None):
        """
        执行交叉验证
        
        Args:
            initial: 初始训练期
            period: 每次验证的间隔
            horizon: 预测期
            parallel: 并行处理的方式
            
        Returns:
            交叉验证结果DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 执行交叉验证
        self.cv_results = cross_validation(
            model=self.model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=parallel
        )
        
        # 计算性能指标
        self.metrics = performance_metrics(self.cv_results)
        
        return self.cv_results, self.metrics
    
    def plot_cv_metrics(self, metric='mae', save_path=None):
        """
        绘制交叉验证指标
        
        Args:
            metric: 要绘制的指标，如'mae', 'rmse', 'mape'
            save_path: 图片保存路径
            
        Returns:
            图形对象
        """
        if self.cv_results is None:
            raise ValueError("尚未执行交叉验证，请先调用cross_validate方法")
        
        # 绘制交叉验证指标
        fig = plot_cross_validation_metric(self.cv_results, metric=metric)
        
        # 设置标题
        plt.title(f'Prophet模型交叉验证 - {metric.upper()}', fontsize=14)
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"交叉验证图已保存至: {save_path}")
        
        return fig
    
    def test_robustness(self, time_series_data, time_col='ds', target_col='y', noise_levels=[0.05, 0.1, 0.2]):
        """
        测试模型鲁棒性
        
        Args:
            time_series_data: 时间序列数据
            time_col: 时间列名
            target_col: 目标变量列名
            noise_levels: 噪声水平列表
            
        Returns:
            鲁棒性测试结果字典
        """
        # 准备数据
        df = time_series_data.copy()
        
        # 重命名列以符合Prophet要求
        if time_col != 'ds':
            df = df.rename(columns={time_col: 'ds'})
        if target_col != 'y':
            df = df.rename(columns={target_col: 'y'})
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # 存储结果
        robustness_results = {}
        
        # 原始数据的性能（基准）
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        # 训练基准模型
        base_model = Prophet(**self.params)
        base_model.fit(train_df)
        
        # 预测测试集
        future = base_model.make_future_dataframe(periods=len(test_df), include_history=True)
        forecast = base_model.predict(future)
        
        # 提取测试期间的预测值
        test_forecast = forecast.iloc[-len(test_df):]
        
        # 计算基准性能指标
        base_mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
        base_rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
        base_r2 = r2_score(test_df['y'], test_forecast['yhat'])
        
        robustness_results['baseline'] = {
            'mae': base_mae,
            'rmse': base_rmse,
            'r2': base_r2
        }
        
        # 对不同噪声水平进行测试
        for noise_level in noise_levels:
            # 添加噪声
            noisy_train_df = train_df.copy()
            noise = np.random.normal(0, noise_level * noisy_train_df['y'].std(), len(noisy_train_df))
            noisy_train_df['y'] = noisy_train_df['y'] + noise
            
            # 训练模型
            noisy_model = Prophet(**self.params)
            noisy_model.fit(noisy_train_df)
            
            # 预测测试集
            future = noisy_model.make_future_dataframe(periods=len(test_df), include_history=True)
            forecast = noisy_model.predict(future)
            
            # 提取测试期间的预测值
            test_forecast = forecast.iloc[-len(test_df):]
            
            # 计算性能指标
            mae = mean_absolute_error(test_df['y'], test_forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(test_df['y'], test_forecast['yhat']))
            r2 = r2_score(test_df['y'], test_forecast['yhat'])
            
            robustness_results[f'noise_{noise_level}'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        
        self.robustness_results = robustness_results
        
        return robustness_results
    
    def plot_robustness_results(self, save_path=None):
        """
        绘制鲁棒性测试结果
        
        Args:
            save_path: 图片保存路径
            
        Returns:
            图形对象
        """
        if self.robustness_results is None:
            raise ValueError("尚未执行鲁棒性测试，请先调用test_robustness方法")
        
        # 提取结果
        noise_levels = [k for k in self.robustness_results.keys() if k != 'baseline']
        noise_values = [float(level.split('_')[1]) for level in noise_levels]
        
        mae_values = [self.robustness_results[level]['mae'] for level in noise_levels]
        rmse_values = [self.robustness_results[level]['rmse'] for level in noise_levels]
        r2_values = [self.robustness_results[level]['r2'] for level in noise_levels]
        
        # 添加基准值
        noise_values = [0] + noise_values
        mae_values = [self.robustness_results['baseline']['mae']] + mae_values
        rmse_values = [self.robustness_results['baseline']['rmse']] + rmse_values
        r2_values = [self.robustness_results['baseline']['r2']] + r2_values
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制MAE和RMSE
        ax1.plot(noise_values, mae_values, 'o-', label='MAE')
        ax1.plot(noise_values, rmse_values, 's-', label='RMSE')
        ax1.set_xlabel('噪声水平 (标准差的比例)', fontsize=12)
        ax1.set_ylabel('误差', fontsize=12)
        ax1.set_title('噪声水平对预测误差的影响', fontsize=14)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制R²
        ax2.plot(noise_values, r2_values, 'o-', color='green')
        ax2.set_xlabel('噪声水平 (标准差的比例)', fontsize=12)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_title('噪声水平对R²的影响', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"鲁棒性测试图已保存至: {save_path}")
        
        return fig

def generate_synthetic_charging_data(start_date='2023-01-01', end_date='2023-12-31', num_stations=5):
    """
    生成合成充电站数据用于测试
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        num_stations: 充电站数量
        
    Returns:
        合成数据DataFrame
    """
    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 创建基础时间序列
    df = pd.DataFrame({'ds': date_range})
    
    # 基础充电需求（每天）
    base_demand = 100
    
    # 添加趋势（逐渐增长）
    trend = np.linspace(0, 50, len(date_range))
    
    # 添加周季节性（工作日vs周末）
    weekly = 20 * (df['ds'].dt.dayofweek < 5).astype(int) - 10 * (df['ds'].dt.dayofweek >= 5).astype(int)
    
    # 添加年季节性（夏季用电高峰）
    yearly = 30 * np.sin(2 * np.pi * (df['ds'].dt.dayofyear / 365 - 0.5))
    
    # 添加节假日效应
    holidays = {
        '2023-01-01': 30,  # 元旦
        '2023-01-22': 50,  # 春节
        '2023-04-05': 20,  # 清明节
        '2023-05-01': 25,  # 劳动节
        '2023-06-22': 15,  # 端午节
        '2023-09-29': 20,  # 中秋节
        '2023-10-01': 40,  # 国庆节
    }
    
    holiday_effect = df['ds'].dt.strftime('%Y-%m-%d').map(holidays).fillna(0)
    
    # 组合所有成分
    df['y'] = base_demand + trend + weekly + yearly + holiday_effect
    
    # 添加随机噪声
    np.random.seed(42)
    df['y'] = df['y'] + np.random.normal(0, 10, len(df))
    
    # 确保值为正
    df['y'] = np.maximum(df['y'], 0)
    
    return df

def main():
    """
    主函数
    """
    print("开始Prophet模型增强与鲁棒性测试...")
    
    # 生成合成数据用于测试
    print("生成合成充电站数据...")
    charging_data = generate_synthetic_charging_data()
    
    # 初始化增强版Prophet模型
    print("初始化增强版Prophet模型...")
    model_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'holidays_prior_scale': 10,
        'seasonality_mode': 'multiplicative',
        'weekly_seasonality': True,
        'yearly_seasonality': True,
        'daily_seasonality': False
    }
    
    prophet_model = EnhancedProphetModel(params=model_params)
    
    # 训练模型
    print("训练模型...")
    prophet_model.fit(charging_data)
    
    # 生成预测
    print("生成预测...")
    forecast = prophet_model.predict(periods=90)
    
    # 绘制预测结果
    fig, ax = plt.subplots(figsize=(12, 6))
    prophet_model.model.plot(forecast, ax=ax)
    ax.set_title('充电需求预测', fontsize=14)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('充电需求', fontsize=12)
    plt.tight_layout()
    forecast_path = os.path.join(result_dir, '0422_充电需求预测.png')
    plt.savefig(forecast_path, dpi=300)
    plt.close()
    print(f"预测图已保存至: {forecast_path}")
    
    # 绘制组件图
    components_path = os.path.join(result_dir, '0422_预测组件分解.png')
    fig = prophet_model.model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(components_path, dpi=300)
    plt.close()
    print(f"组件分解图已保存至: {components_path}")
    
    # 执行交叉验证
    print("执行交叉验证...")
    cv_results, metrics = prophet_model.cross_validate(initial='180 days', period='30 days', horizon='60 days')
    
    # 绘制交叉验证结果
    cv_path = os.path.join(result_dir, '0422_交叉验证结果.png')
    prophet_model.plot_cv_metrics(metric='mae', save_path=cv_path)
    
    # 执行鲁棒性测试
    print("执行鲁棒性测试...")
    robustness_results = prophet_model.test_robustness(charging_data)
    
    # 绘制鲁棒性测试结果
    robustness_path = os.path.join(result_dir, '0422_模型鲁棒性测试.png')
    prophet_model.plot_robustness_results(save_path=robustness_path)
    
    print("Prophet模型增强与鲁棒性测试完成！")

if __name__ == "__main__":
    main()