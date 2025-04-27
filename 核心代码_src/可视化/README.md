# Prophet-GTWR耦合模型可视化系统

## 系统概述

本可视化系统基于UrbanEV_data数据集，结合Prophet-GTWR耦合建模方法，生成一系列可视化文档，用于分析电动汽车充电桩使用情况、预测充电需求，并评估模型性能。

## 文件结构

- `充电桩数据可视化与分析.py`: 对充电桩使用数据进行可视化分析，包括时间序列分析、占用率热力图、天气关系分析等
- `时空预测可视化.py`: 可视化Prophet-GTWR耦合模型的时空预测结果，包括时间序列预测、空间异质性分析等
- `模型评估与运行脚本.py`: 评估模型性能，包括模型比较、参数敏感性分析、特征重要性分析等
- `运行所有可视化.py`: 主运行脚本，一键运行所有可视化功能

## 使用方法

### 环境要求

- Python 3.6+
- 依赖包：pandas, numpy, matplotlib, seaborn, folium, tqdm

### 安装依赖

```bash
pip install pandas numpy matplotlib seaborn folium tqdm
```

### 运行方式

1. **一键运行所有可视化**

   ```bash
   python 运行所有可视化.py
   ```

2. **单独运行各模块**

   ```bash
   # 充电桩数据可视化与分析
   python 充电桩数据可视化与分析.py
   
   # 时空预测可视化
   python 时空预测可视化.py
   
   # 模型评估与比较
   python 模型评估与运行脚本.py
   ```

## 输出结果

所有可视化结果将保存在以下目录：

```
项目产出_results/figures/Prophet_GTWR_可视化/
```

主要输出文件包括：

1. **充电桩数据分析**
   - 充电站使用量时间序列分析.png
   - 充电桩占用率热力图.png
   - 天气与充电量关系分析.png
   - 充电模式分析.png

2. **时空预测结果**
   - Prophet模型预测结果.png
   - GTWR模型空间系数分布.html
   - 时空耦合效应分析.png
   - 预测误差空间分布.html

3. **模型评估**
   - 模型比较分析.png
   - 参数敏感性分析.png
   - 特征重要性分析.png

## 与Prophet-GTWR耦合模型的集成

本可视化系统设计为与Prophet-GTWR耦合模型无缝集成。在实际应用中，可以通过以下方式将模型与可视化系统连接：

1. 将模型的预测结果保存为标准格式的CSV文件
2. 修改可视化脚本中的数据加载部分，指向实际的模型输出文件
3. 运行可视化脚本生成结果

## 注意事项

- 当前版本使用模拟数据展示可视化效果，实际应用时需替换为真实模型输出
- 确保数据路径正确设置，默认路径为：`d:\DZQ_Projects_项目合集\Competitions\2025_Statistical_Modeling_competition\EV_item_for_Trae\数据_data\1_充电桩数据\processed\UrbanEV_data`
- 如遇到中文显示问题，请确保系统安装了中文字体（如SimHei）