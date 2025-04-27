# -*- coding: utf-8 -*-
"""
安装脚本 - 充电桩布局优化项目
功能：创建项目结构、安装依赖、准备环境
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime

def print_header(message):
    """打印带格式的标题"""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

def create_directories():
    """创建项目目录结构"""
    print_header("创建项目目录结构")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义需要创建的目录
    directories = [
        os.path.join(current_dir, "processed_data"),
        os.path.join(current_dir, "models"),
        os.path.join(current_dir, "reports"),
        os.path.join(current_dir, "reports", "figures"),
        os.path.join(current_dir, "reports", "shap_analysis")
    ]
    
    # 创建目录
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ 创建目录: {directory}")
        else:
            print(f"✓ 目录已存在: {directory}")
    
    return True

def install_dependencies():
    """安装项目依赖包"""
    print_header("安装项目依赖包")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    # 检查requirements.txt是否存在
    if not os.path.exists(requirements_path):
        print("❌ 错误: requirements.txt 文件不存在")
        return False
    
    # 安装依赖
    try:
        print("开始安装依赖包，这可能需要几分钟时间...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("✓ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装依赖包时出错: {str(e)}")
        return False

def check_data_files():
    """检查数据文件是否存在"""
    print_header("检查数据文件")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "scrapy_gaode_01", "spiders", "大湾区充电桩基础数据.csv")
    
    # 检查数据文件是否存在
    if os.path.exists(data_path):
        print(f"✓ 数据文件已存在: {data_path}")
        # 获取文件大小和修改时间
        file_size = os.path.getsize(data_path) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
        print(f"  - 文件大小: {file_size:.2f} KB")
        print(f"  - 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    else:
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        return False

def create_backup():
    """创建项目备份"""
    print_header("创建项目备份")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建备份目录
    backup_dir = os.path.join(current_dir, "backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 创建带时间戳的备份文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
    
    # 要备份的文件列表
    files_to_backup = [
        "data_processor.py",
        "prophet_gtwr_model.py",
        "main.py",
        "requirements.txt",
        "README.md"
    ]
    
    # 创建备份文件夹
    os.makedirs(backup_path)
    
    # 复制文件
    for file in files_to_backup:
        src = os.path.join(current_dir, file)
        if os.path.exists(src):
            dst = os.path.join(backup_path, file)
            shutil.copy2(src, dst)
            print(f"✓ 已备份: {file}")
    
    print(f"✓ 备份完成: {backup_path}")
    return True

def main():
    """主函数"""
    print_header("充电桩布局优化项目 - 环境设置")
    
    # 创建目录结构
    create_directories()
    
    # 检查数据文件
    data_exists = check_data_files()
    
    # 安装依赖
    deps_installed = install_dependencies()
    
    # 创建备份
    create_backup()
    
    # 总结
    print_header("设置完成")
    if data_exists and deps_installed:
        print("✓ 环境设置成功！")
        print("\n运行以下命令启动项目:")
        print("  python main.py")
    else:
        print("⚠️ 环境设置完成，但存在一些问题需要解决")
        if not data_exists:
            print("  - 请确保数据文件位于正确位置")
        if not deps_installed:
            print("  - 请手动安装依赖包: pip install -r requirements.txt")

if __name__ == "__main__":
    main()