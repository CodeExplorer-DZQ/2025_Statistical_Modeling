# -*- coding: utf-8 -*-
"""
日期：2023年4月22日
功能：运行人口热力图生成和Prophet模型增强功能，生成可视化结果
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到系统路径
sys.path.append(current_dir)

# 导入自定义模块
import importlib
# 使用importlib动态导入模块
population_heatmap = importlib.import_module('0422_population_heatmap')

# 获取main函数
run_population_heatmap = population_heatmap.main

# 尝试导入prophet_model_enhancement模块
try:
    prophet_model_enhancement = importlib.import_module('0422_prophet_model_enhancement')
    run_prophet_enhancement = prophet_model_enhancement.main
except (ImportError, AttributeError):
    print("警告: prophet_model_enhancement模块不存在或没有main函数")
    run_prophet_enhancement = lambda: print("Prophet模型增强功能未实现")

def main():
    """
    主函数：运行人口热力图生成和Prophet模型增强功能
    """
    print("="*80)
    print("开始运行Prophet-GTWR耦合模型可视化与增强功能...")
    print("="*80)
    
    # 第一步：生成深圳人口热力图
    print("\n第一步：生成深圳2024常住人口分区热力图...")
    try:
        run_population_heatmap()
        print("深圳人口热力图生成成功！")
    except Exception as e:
        print(f"生成深圳人口热力图时出错: {e}")
    
    # 第二步：运行Prophet模型增强功能
    print("\n第二步：运行Prophet模型增强与鲁棒性测试...")
    try:
        run_prophet_enhancement()
        print("Prophet模型增强与鲁棒性测试完成！")
    except Exception as e:
        print(f"运行Prophet模型增强时出错: {e}")
    
    print("\n所有任务已完成！请查看项目产出_results/figures目录下的可视化结果。")

if __name__ == "__main__":
    main()