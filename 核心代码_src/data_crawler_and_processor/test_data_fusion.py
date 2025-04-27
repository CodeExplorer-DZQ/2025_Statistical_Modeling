# -*- coding: utf-8 -*-
"""
多源异构数据融合处理器测试脚本
功能：测试数据融合处理器的功能，验证高德充电桩数据和UrbanEV数据集的融合结果
日期：2025年4月23日
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 导入数据融合处理器
from 核心代码_src.data_crawler_and_processor.data_fusion_processor import DataFusionProcessor

def test_data_fusion():
    """
    测试数据融合处理器的功能
    """
    print("开始测试数据融合处理器...")
    
    # 创建数据融合处理器实例
    processor = DataFusionProcessor()
    
    # 运行数据融合流程
    processor.run_fusion_pipeline()
    
    # 验证输出文件是否存在
    output_path = os.path.join(processor.output_path)
    expected_files = [
        'prophet_input_data.csv',
        'gtwr_input_data.csv',
        'fused_feature_data.csv'
    ]
    
    for file in expected_files:
        file_path = os.path.join(output_path, file)
        if os.path.exists(file_path):
            print(f"✓ 文件 {file} 已成功生成")
            # 读取文件并显示基本信息
            df = pd.read_csv(file_path)
            print(f"  - 记录数: {len(df)}")
            print(f"  - 列数: {len(df.columns)}")
            print(f"  - 列名: {', '.join(df.columns)}")
            print(f"  - 前5行数据预览:")
            print(df.head())
            print("\n")
        else:
            print(f"✗ 文件 {file} 未生成")
    
    # 如果Prophet输入数据存在，绘制时间序列图
    prophet_file = os.path.join(output_path, 'prophet_input_data.csv')
    if os.path.exists(prophet_file):
        try:
            prophet_data = pd.read_csv(prophet_file)
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(prophet_data['ds'], prophet_data['y'], 'b-', label='充电需求量')
            plt.title('充电需求量时间序列')
            plt.xlabel('日期')
            plt.ylabel('充电需求量')
            plt.legend()
            plt.grid(True)
            
            # 保存图表
            fig_path = os.path.join(output_path, 'prophet_input_visualization.png')
            plt.savefig(fig_path)
            plt.close()
            print(f"✓ Prophet输入数据可视化已保存至: {fig_path}")
        except Exception as e:
            print(f"绘制Prophet输入数据图表时出错: {e}")
    
    # 如果GTWR输入数据存在，绘制空间分布图
    gtwr_file = os.path.join(output_path, 'gtwr_input_data.csv')
    if os.path.exists(gtwr_file):
        try:
            gtwr_data = pd.read_csv(gtwr_file)
            
            # 按区域统计充电量
            district_stats = gtwr_data.groupby('district')['charging_amount'].sum().reset_index()
            district_stats = district_stats.sort_values('charging_amount', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='district', y='charging_amount', data=district_stats)
            plt.title('各区域充电量统计')
            plt.xlabel('区域')
            plt.ylabel('充电量总和')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 保存图表
            fig_path = os.path.join(output_path, 'gtwr_input_visualization.png')
            plt.savefig(fig_path)
            plt.close()
            print(f"✓ GTWR输入数据可视化已保存至: {fig_path}")
        except Exception as e:
            print(f"绘制GTWR输入数据图表时出错: {e}")
    
    print("数据融合处理器测试完成！")

def validate_data_format():
    """
    验证数据格式是否符合接口规范
    """
    print("\n开始验证数据格式...")
    
    output_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')), 
                              '数据_data/7_跨域融合数据')
    
    # 验证Prophet输入数据格式
    prophet_file = os.path.join(output_path, 'prophet_input_data.csv')
    if os.path.exists(prophet_file):
        prophet_data = pd.read_csv(prophet_file)
        
        # 检查必要字段
        required_fields = ['ds', 'y', 'ev_count']
        missing_fields = [field for field in required_fields if field not in prophet_data.columns]
        
        if not missing_fields:
            print("✓ Prophet输入数据格式符合规范")
            # 检查日期格式
            try:
                prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                print("  - 日期格式正确")
            except:
                print("  - 日期格式不正确")
        else:
            print(f"✗ Prophet输入数据缺少必要字段: {', '.join(missing_fields)}")
    
    # 验证GTWR输入数据格式
    gtwr_file = os.path.join(output_path, 'gtwr_input_data.csv')
    if os.path.exists(gtwr_file):
        gtwr_data = pd.read_csv(gtwr_file)
        
        # 检查必要字段
        required_fields = ['longitude', 'latitude', 'date', 'district', 'charging_amount']
        missing_fields = [field for field in required_fields if field not in gtwr_data.columns]
        
        if not missing_fields:
            print("✓ GTWR输入数据格式符合规范")
            # 检查日期格式
            try:
                gtwr_data['date'] = pd.to_datetime(gtwr_data['date'])
                print("  - 日期格式正确")
            except:
                print("  - 日期格式不正确")
            
            # 检查坐标系
            lon_range = (113.5, 115.0)  # 深圳经度范围
            lat_range = (22.0, 23.0)    # 深圳纬度范围
            
            lon_valid = gtwr_data['longitude'].between(*lon_range).all()
            lat_valid = gtwr_data['latitude'].between(*lat_range).all()
            
            if lon_valid and lat_valid:
                print("  - 坐标范围符合深圳地区")
            else:
                print("  - 坐标范围异常，可能不符合深圳地区")
        else:
            print(f"✗ GTWR输入数据缺少必要字段: {', '.join(missing_fields)}")
    
    print("数据格式验证完成！")

if __name__ == "__main__":
    # 测试数据融合处理器
    test_data_fusion()
    
    # 验证数据格式
    validate_data_format()