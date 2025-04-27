# -*- coding: utf-8 -*-
"""
运行所有可视化

此脚本是Prophet-GTWR耦合模型可视化系统的主入口，用于一键运行所有可视化功能。
包括：
1. 充电桩数据可视化与分析
2. 时空预测可视化
3. 模型评估与比较

日期：2025年4月23日
"""

import os
import sys
import importlib
import subprocess
from datetime import datetime

# 获取当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 结果保存路径
RESULT_PATH = r'd:\DZQ_Projects_项目合集\Competitions\2025_Statistical_Modeling_competition\EV_item_for_Trae\项目产出_results\figures'
PROPHET_GTWR_DIR = os.path.join(RESULT_PATH, 'Prophet_GTWR_可视化')

# 确保结果目录存在
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
if not os.path.exists(PROPHET_GTWR_DIR):
    os.makedirs(PROPHET_GTWR_DIR)


def run_script(script_name):
    """
    运行指定的Python脚本
    
    参数:
        script_name: 脚本文件名（不含路径）
    
    返回:
        成功返回True，失败返回False
    """
    script_path = os.path.join(CURRENT_DIR, script_name)
    print(f"\n正在运行: {script_name}")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"\n{script_name} 运行成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n运行 {script_name} 时出错: {e}")
        return False
    except Exception as e:
        print(f"\n运行 {script_name} 时发生未知错误: {e}")
        return False


def main():
    """
    主函数，按顺序运行所有可视化脚本
    """
    print("\n" + "=" * 60)
    print("     Prophet-GTWR耦合模型可视化系统 - 主运行脚本")
    print("=" * 60)
    
    start_time = datetime.now()
    print(f"\n开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果将保存在: {PROPHET_GTWR_DIR}")
    
    # 要运行的脚本列表
    scripts = [
        '充电桩数据可视化与分析.py',
        '时空预测可视化.py',
        '模型评估与运行脚本.py'
    ]
    
    # 运行所有脚本
    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    # 打印运行结果摘要
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"运行完成! 成功: {success_count}/{len(scripts)}")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {duration}")
    print("=" * 60)
    
    # 如果所有脚本都成功运行，打开结果目录
    if success_count == len(scripts):
        print(f"\n所有可视化已成功生成，结果保存在: {PROPHET_GTWR_DIR}")
        try:
            os.startfile(PROPHET_GTWR_DIR)  # 在Windows上打开结果文件夹
        except:
            print(f"请手动打开结果目录查看生成的可视化: {PROPHET_GTWR_DIR}")
    else:
        print(f"\n部分脚本运行失败，请检查错误信息并重试。")


if __name__ == "__main__":
    main()