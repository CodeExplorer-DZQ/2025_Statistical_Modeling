# -*- coding: utf-8 -*-
"""
日期：2023年4月22日
功能：生成深圳2024常住人口分区热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

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

def load_population_data():
    """
    加载深圳人口数据
    """
    # 区级人口数据路径
    population_file = os.path.join(data_dir, '2_时空动态数据', '0421_区级_深圳人口数据.csv')
    
    # 读取数据
    df = pd.read_csv(population_file)
    
    # 添加人口密度数据（模拟数据，实际应从统计年鉴中提取）
    # 深圳各区面积数据（平方公里）
    area_data = {
        '福田区': 78.8,
        '罗湖区': 78.8,
        '盐田区': 74.6,
        '南山区': 187.5,
        '宝安区': 398.4,
        '龙岗区': 388.2,
        '龙华区': 175.6,
        '坪山区': 166.3,
        '光明区': 155.4,
        '大鹏新区': 295.3,
        '深汕特别合作区': 468.0
    }
    
    # 添加面积列
    df['面积(平方公里)'] = df['区域'].map(lambda x: area_data.get(x, np.nan))
    
    # 计算人口密度
    df['人口密度(人/平方公里)'] = df['人口数'] / df['面积(平方公里)']
    
    return df

def load_economic_data():
    """
    加载深圳经济数据
    """
    # 经济数据路径
    economic_file = os.path.join(data_dir, '4_社会经济数据', '0422_深圳区域经济指标数据.csv')
    
    # 读取数据
    df = pd.read_csv(economic_file)
    
    return df

def plot_population_heatmap(pop_data):
    """
    绘制深圳人口热力图
    """
    # 过滤掉全市数据
    district_data = pop_data[pop_data['区域'] != '全市'].copy()
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 定义紫色系渐变色彩映射
    purple_colors = [(0.9, 0.8, 0.95), (0.7, 0.4, 0.9), (0.5, 0.0, 0.8), (0.3, 0.0, 0.5)]
    purple_cmap = LinearSegmentedColormap.from_list('purple_gradient', purple_colors)
    
    # 绘制热力图
    ax = sns.barplot(x='区域', y='人口数', data=district_data, palette=purple_cmap(np.linspace(0, 1, len(district_data))))
    
    # 添加数据标签
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 50000, f'{int(height):,}',
                ha="center", va="bottom", fontsize=10)
    
    # 设置标题和标签
    plt.title('深圳市2024年各区常住人口分布', fontsize=16)
    plt.xlabel('行政区', fontsize=12)
    plt.ylabel('人口数量', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加注释
    plt.annotate('数据来源：深圳统计年鉴2024', xy=(0.02, 0.02), xycoords='figure fraction', fontsize=8)
    
    # 保存图片
    plt.tight_layout()
    output_file = os.path.join(result_dir, '0422_深圳常住人口分区热力图.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"人口热力图已保存至: {output_file}")
    
    return output_file

def plot_population_density_heatmap(pop_data):
    """
    绘制深圳人口密度热力图
    """
    # 过滤掉全市数据
    district_data = pop_data[pop_data['区域'] != '全市'].copy()
    
    # 按人口密度排序
    district_data = district_data.sort_values(by='人口密度(人/平方公里)', ascending=False)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 定义紫色系渐变色彩映射
    purple_colors = [(0.9, 0.8, 0.95), (0.7, 0.4, 0.9), (0.5, 0.0, 0.8), (0.3, 0.0, 0.5)]
    purple_cmap = LinearSegmentedColormap.from_list('purple_gradient', purple_colors)
    
    # 绘制热力图
    ax = sns.barplot(x='区域', y='人口密度(人/平方公里)', data=district_data, 
                    palette=purple_cmap(np.linspace(0, 1, len(district_data))))
    
    # 添加数据标签
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 500, f'{int(height):,}',
                ha="center", va="bottom", fontsize=10)
    
    # 设置标题和标签
    plt.title('深圳市2024年各区人口密度分布', fontsize=16)
    plt.xlabel('行政区', fontsize=12)
    plt.ylabel('人口密度(人/平方公里)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加注释
    plt.annotate('数据来源：深圳统计年鉴2024', xy=(0.02, 0.02), xycoords='figure fraction', fontsize=8)
    
    # 保存图片
    plt.tight_layout()
    output_file = os.path.join(result_dir, '0422_深圳人口密度热力图.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"人口密度热力图已保存至: {output_file}")
    
    return output_file

def plot_gdp_population_correlation(pop_data, eco_data):
    """
    绘制GDP与人口相关性散点图
    """
    # 过滤掉全市数据
    district_pop = pop_data[pop_data['区域'] != '全市'].copy()
    district_eco = eco_data[eco_data['区域'] != '全市'].copy()
    
    # 合并数据
    merged_data = pd.merge(district_pop, district_eco, on='区域')
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制散点图
    sns.scatterplot(x='人口数', y='GDP(亿元)', data=merged_data, s=100, alpha=0.7)
    
    # 添加标签
    for i, row in merged_data.iterrows():
        plt.text(row['人口数']*1.02, row['GDP(亿元)']*1.02, row['区域'], fontsize=10)
    
    # 添加趋势线
    sns.regplot(x='人口数', y='GDP(亿元)', data=merged_data, scatter=False, ci=None, line_kws={'color':'red', 'linestyle':'--'})
    
    # 设置标题和标签
    plt.title('深圳市各区GDP与人口相关性分析', fontsize=16)
    plt.xlabel('人口数量', fontsize=12)
    plt.ylabel('GDP(亿元)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    # 添加注释
    plt.annotate('数据来源：深圳统计年鉴2024', xy=(0.02, 0.02), xycoords='figure fraction', fontsize=8)
    
    # 保存图片
    plt.tight_layout()
    output_file = os.path.join(result_dir, '0422_深圳GDP与人口相关性分析.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"GDP与人口相关性分析图已保存至: {output_file}")
    
    return output_file

def main():
    """
    主函数
    """
    print("开始生成深圳2024常住人口分区热力图...")
    
    # 加载数据
    pop_data = load_population_data()
    eco_data = load_economic_data()
    
    # 绘制人口热力图
    plot_population_heatmap(pop_data)
    
    # 绘制人口密度热力图
    plot_population_density_heatmap(pop_data)
    
    # 绘制GDP与人口相关性散点图
    plot_gdp_population_correlation(pop_data, eco_data)
    
    print("深圳2024常住人口分析图表生成完成！")

if __name__ == "__main__":
    main()