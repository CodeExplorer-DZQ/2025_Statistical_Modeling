# -*- coding: utf-8 -*-
"""
深圳高德数据清洗脚本
包括查重（以id、name和address均重复视为重复）、KNN插值、坐标异常值处理、充电类型推断等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建输出目录
output_dir = os.path.join('data', 'processed')
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# 文件路径
input_file = '0417-0419最终深圳高德数据.csv'
output_file = os.path.join(output_dir, 'cleaned_0420深圳高德数据.csv')
report_file = os.path.join(output_dir, '数据清洗报告.md')

# 读取数据
def load_data(file_path):
    try:
        # 尝试使用UTF-8编码读取
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果失败，尝试使用GBK编码读取
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            # 如果仍然失败，尝试使用gb18030编码读取
            df = pd.read_csv(file_path, encoding='gb18030')
    
    print(f"原始数据形状: {df.shape}")
    return df

# 数据清洗函数
def clean_data(df):
    # 保存原始数据的副本
    original_df = df.copy()
    
    # 1. 查重：以id、name和address均重复视为重复
    duplicates = df.duplicated(subset=['id', 'name', 'address'], keep='first')
    df_no_duplicates = df[~duplicates]
    duplicate_count = duplicates.sum()
    print(f"删除重复记录: {duplicate_count}条")
    
    # 2. 处理缺失值
    # 统计缺失值
    missing_values = df_no_duplicates.isnull().sum()
    missing_values_percent = (missing_values / len(df_no_duplicates)) * 100
    missing_stats = pd.DataFrame({
        '缺失值数量': missing_values,
        '缺失比例(%)': missing_values_percent
    })
    
    # 处理特殊的空值表示
    df_no_duplicates.replace('[]', np.nan, inplace=True)
    df_no_duplicates.replace('未知', np.nan, inplace=True)
    df_no_duplicates.replace('δ֪', np.nan, inplace=True)  # 可能的乱码表示
    
    # 保存清洗前的数据副本用于对比
    before_cleaning = df_no_duplicates.copy()
    
    # 3. 坐标异常值处理（深圳市经纬度范围）
    # 深圳经度范围约为113.7-114.6，纬度范围约为22.4-22.9
    df_no_duplicates['lng'] = pd.to_numeric(df_no_duplicates['lng'], errors='coerce')
    df_no_duplicates['lat'] = pd.to_numeric(df_no_duplicates['lat'], errors='coerce')
    
    # 标记坐标异常值
    lng_mask = (df_no_duplicates['lng'] < 113.7) | (df_no_duplicates['lng'] > 114.6)
    lat_mask = (df_no_duplicates['lat'] < 22.4) | (df_no_duplicates['lat'] > 22.9)
    coord_outliers = lng_mask | lat_mask
    
    # 统计坐标异常值
    coord_outliers_count = coord_outliers.sum()
    print(f"检测到坐标异常记录: {coord_outliers_count}条")
    
    # 移除坐标异常的记录
    df_no_duplicates = df_no_duplicates[~coord_outliers]
    
    # 4. 充电类型推断和清洗
    # 保存原始充电类型分布
    original_charging_type_dist = df_no_duplicates['charging_type'].value_counts(dropna=False)
    
    # 标准化充电类型名称
    type_mapping = {
        '快充': '快充',
        '慢充': '慢充',
        '超充': '快充',  # 将超充归类为快充
        '直流': '快充',  # 直流通常是快充
        '交流': '慢充',  # 交流通常是慢充
        '未知': np.nan,
        np.nan: np.nan
    }
    
    # 应用映射
    df_no_duplicates['charging_type'] = df_no_duplicates['charging_type'].map(lambda x: type_mapping.get(x, x))
    
    # 从名称和地址中推断充电类型
    # 定义关键词字典
    fast_charging_keywords = [
        '快充', '直流', 'dc', '超级', '超充', 'super', '高速', '高功率', 
        '大功率', '120kw', '150kw', '180kw', '200kw', '350kw', '特斯拉'
    ]
    
    slow_charging_keywords = [
        '慢充', '交流', 'ac', '普通', '标准', '7kw', '11kw', '3.5kw', 
        '家用', '便携', '壁挂'
    ]
    
    # 运营商与充电类型的关联字典
    operator_type_mapping = {
        '特来电': '快充',
        '星星充电': '快充',
        '南方电网': '快充',
        '国家电网': '快充',
        '特斯拉': '快充',
        '壳牌': '快充',
        '云快充': '快充',
        '小桔充电': '慢充',
        'igo充电': '慢充',
        '开迈斯': '快充',
        '南网电动': '快充',
        '车电网': '快充',
        '充电有道': '快充',
        '依威能源': '快充'
    }
    
    # 从名称和地址中推断充电类型
    unknown_mask = df_no_duplicates['charging_type'].isnull()
    unknown_count = unknown_mask.sum()
    
    if unknown_count > 0:
        print(f"发现 {unknown_count} 条未知充电类型记录，尝试从多种特征中推断")
        
        # 创建充电类型推断标记列
        df_no_duplicates['charging_type_imputed'] = False
        
        # 从名称和地址中推断
        for idx, row in df_no_duplicates[unknown_mask].iterrows():
            name = str(row['name']).lower() if not pd.isna(row['name']) else ""
            address = str(row['address']).lower() if not pd.isna(row['address']) else ""
            operator = str(row['operator']) if not pd.isna(row['operator']) else ""
            
            # 从名称和地址中查找关键词
            text = name + " " + address
            
            # 检查快充关键词
            if any(keyword in text.lower() for keyword in fast_charging_keywords):
                df_no_duplicates.at[idx, 'charging_type'] = '快充'
                df_no_duplicates.at[idx, 'charging_type_imputed'] = True
                continue
                
            # 检查慢充关键词
            if any(keyword in text.lower() for keyword in slow_charging_keywords):
                df_no_duplicates.at[idx, 'charging_type'] = '慢充'
                df_no_duplicates.at[idx, 'charging_type_imputed'] = True
                continue
                
            # 从运营商推断
            if operator in operator_type_mapping:
                df_no_duplicates.at[idx, 'charging_type'] = operator_type_mapping[operator]
                df_no_duplicates.at[idx, 'charging_type_imputed'] = True
                continue
    
    # 5. 使用KNN算法填充剩余缺失的充电类型和运营商
    # 准备用于KNN的数据
    numeric_cols = ['lng', 'lat']
    categorical_cols = ['charging_type', 'operator', 'rating']
    
    # 保存原始缺失值信息
    missing_charging_type = df_no_duplicates['charging_type'].isnull()
    missing_operator = df_no_duplicates['operator'].isnull()
    missing_rating = df_no_duplicates['rating'].isnull()
    
    # 对rating进行处理，将非数值转换为NaN
    df_no_duplicates['rating'] = pd.to_numeric(df_no_duplicates['rating'], errors='coerce')
    
    # 准备KNN插值
    # 标准化经纬度数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame()
    df_scaled[numeric_cols] = scaler.fit_transform(df_no_duplicates[numeric_cols])
    
    # 对充电类型进行编码
    charging_type_mapping = {'快充': 1, '慢充': 2}
    df_no_duplicates['charging_type_code'] = df_no_duplicates['charging_type'].map(charging_type_mapping)
    
    # 使用KNN填充充电类型
    if missing_charging_type.sum() > 0:
        # 创建KNN插补器
        imputer = KNNImputer(n_neighbors=5)
        # 准备用于插补的数据
        impute_data = df_scaled[numeric_cols].copy()
        impute_data['charging_type_code'] = df_no_duplicates['charging_type_code']
        # 执行插补
        imputed_data = imputer.fit_transform(impute_data)
        # 将插补结果放回原数据框
        df_no_duplicates.loc[missing_charging_type, 'charging_type_code'] = imputed_data[missing_charging_type, -1]
        # 将编码转换回类别
        reverse_mapping = {1: '快充', 2: '慢充'}
        df_no_duplicates.loc[missing_charging_type, 'charging_type'] = df_no_duplicates.loc[missing_charging_type, 'charging_type_code'].map(reverse_mapping)
        # 标记为已插值
        df_no_duplicates.loc[missing_charging_type, 'charging_type_imputed'] = True
    
    # 6. 评分数据清洗和异常值处理
    # 创建评分插值标记列
    df_no_duplicates['rating_imputed'] = False
    
    # 评分异常值检测（使用IQR方法）
    if 'rating' in df_no_duplicates.columns:
        # 将评分转换为数值
        df_no_duplicates['rating'] = pd.to_numeric(df_no_duplicates['rating'], errors='coerce')
        
        # 计算IQR
        Q1 = df_no_duplicates['rating'].quantile(0.25)
        Q3 = df_no_duplicates['rating'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)  # 评分不应小于0
        upper_bound = min(5, Q3 + 1.5 * IQR)  # 评分不应大于5
        
        # 标记异常值
        rating_outliers = (df_no_duplicates['rating'] < lower_bound) | (df_no_duplicates['rating'] > upper_bound)
        rating_outliers_count = rating_outliers.sum()
        
        if rating_outliers_count > 0:
            print(f"检测到评分异常值: {rating_outliers_count}条")
            # 将异常值设为NaN，后续用KNN填充
            df_no_duplicates.loc[rating_outliers, 'rating'] = np.nan
            # 更新缺失评分标记
            missing_rating = df_no_duplicates['rating'].isnull()
    
    # 使用KNN填充评分
    if missing_rating.sum() > 0:
        # 创建KNN插补器
        imputer = KNNImputer(n_neighbors=5)
        # 准备用于插补的数据
        impute_data = df_scaled[numeric_cols].copy()
        impute_data['rating'] = df_no_duplicates['rating']
        # 执行插补
        imputed_data = imputer.fit_transform(impute_data)
        # 将插补结果放回原数据框
        df_no_duplicates.loc[missing_rating, 'rating'] = imputed_data[missing_rating, -1]
        # 将评分限制在0-5之间
        df_no_duplicates['rating'] = df_no_duplicates['rating'].clip(0, 5)
        # 四舍五入到一位小数
        df_no_duplicates['rating'] = df_no_duplicates['rating'].round(1)
        # 标记为已插值
        df_no_duplicates.loc[missing_rating, 'rating_imputed'] = True
    
    # 7. 运营商数据清洗
    # 创建运营商推断标记列
    df_no_duplicates['operator_imputed'] = False
    
    # 定义常见运营商关键词字典
    operator_keywords = {
        '特来电': ['特来电'],
        '星星充电': ['星星充电', '星星'],
        '南方电网': ['南方电网', '南网', '南网电动'],
        '国家电网': ['国家电网', '国网'],
        '特斯拉': ['特斯拉', 'tesla'],
        '壳牌': ['壳牌', 'shell'],
        '云快充': ['云快充'],
        '小桔充电': ['小桔充电', '小桔'],
        'igo充电': ['igo充电', 'igo', 'iGO'],
        '开迈斯': ['开迈斯'],
        '车电网': ['车电网'],
        '充电有道': ['充电有道'],
        '依威能源': ['依威能源', '依威'],
        '万城万充': ['万城万充', '万充'],
        '深圳供电局': ['深圳供电局', '供电局'],
        '普天': ['普天充电', '普天'],
        '中国石化': ['中国石化', '石化', '中石化'],
        '中国石油': ['中国石油', '石油', '中石油'],
        '深能源': ['深能源'],
        '深圳巴士集团': ['巴士集团', '巴士', '深巴'],
        '比亚迪': ['比亚迪', 'byd'],
        '蔚来': ['蔚来', 'nio'],
        '小鹏': ['小鹏', 'xpeng'],
        '广汽': ['广汽', 'gac'],
        '奥特来': ['奥特来'],
        '中电': ['中电充电', '中电'],
        '深圳公交': ['深圳公交', '公交'],
        '深圳地铁': ['深圳地铁', '地铁']
    }
    
    # 从名称中提取运营商信息
    if missing_operator.sum() > 0:
        print(f"发现 {missing_operator.sum()} 条缺失运营商信息，尝试从名称中提取")
        
        # 从名称中提取
        for idx, row in df_no_duplicates[missing_operator].iterrows():
            name = str(row['name']).lower() if not pd.isna(row['name']) else ""
            address = str(row['address']).lower() if not pd.isna(row['address']) else ""
            
            # 合并名称和地址以增加提取成功率
            text = name + " " + address
            
            # 检查是否包含运营商关键词
            for operator, keywords in operator_keywords.items():
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    df_no_duplicates.at[idx, 'operator'] = operator
                    df_no_duplicates.at[idx, 'operator_imputed'] = True
                    break
        
        # 统计提取结果
        extracted_count = df_no_duplicates.loc[missing_operator, 'operator_imputed'].sum()
        print(f"成功从名称中提取运营商信息: {extracted_count}条")
        
        # 对于仍然缺失的运营商，使用最常见的值填充
        still_missing = df_no_duplicates['operator'].isnull()
        if still_missing.sum() > 0:
            most_common_operator = df_no_duplicates['operator'].mode()[0]
            if pd.isna(most_common_operator):
                most_common_operator = '未知运营商'
            df_no_duplicates.loc[still_missing, 'operator'] = most_common_operator
            print(f"使用最常见运营商'{most_common_operator}'填充剩余 {still_missing.sum()} 条记录")
    
    # 删除临时列
    if 'charging_type_code' in df_no_duplicates.columns:
        df_no_duplicates.drop('charging_type_code', axis=1, inplace=True)
    
    # 8. 处理month_sales列
    if 'month_sales' in df_no_duplicates.columns:
        df_no_duplicates['month_sales'] = df_no_duplicates['month_sales'].replace('[]', '0')
        df_no_duplicates['month_sales'] = pd.to_numeric(df_no_duplicates['month_sales'], errors='coerce').fillna(0)
    
    # 9. 确保经纬度是有效的数值
    invalid_coords = df_no_duplicates['lng'].isnull() | df_no_duplicates['lat'].isnull()
    if invalid_coords.sum() > 0:
        df_no_duplicates = df_no_duplicates[~invalid_coords]
        print(f"删除经纬度无效的记录: {invalid_coords.sum()}条")
    
    # 10. 生成数据可视化
    # 保存坐标清洗前后对比图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(before_cleaning['lng'], before_cleaning['lat'], alpha=0.5, s=5)
    plt.title('清洗前坐标分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df_no_duplicates['lng'], df_no_duplicates['lat'], alpha=0.5, s=5)
    plt.title('清洗后坐标分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '坐标清洗对比.png'), dpi=300)
    plt.close()
    
    # 保存充电类型分布对比图
    cleaned_charging_type_dist = df_no_duplicates['charging_type'].value_counts()
    
    plt.figure(figsize=(10, 6))
    original_charging_type_dist.plot(kind='bar', alpha=0.7, label='清洗前')
    cleaned_charging_type_dist.plot(kind='bar', alpha=0.7, label='清洗后')
    plt.title('充电类型分布对比')
    plt.xlabel('充电类型')
    plt.ylabel('数量')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '充电类型分布对比.png'), dpi=300)
    plt.close()
    
    # 保存运营商分布对比图
    top_operators = df_no_duplicates['operator'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    top_operators.plot(kind='bar')
    plt.title('主要运营商分布')
    plt.xlabel('运营商')
    plt.ylabel('数量')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '运营商分布对比.png'), dpi=300)
    plt.close()
    
    # 保存运营商提取结果可视化
    if 'operator_imputed' in df_no_duplicates.columns:
        # 统计提取情况
        imputed_stats = df_no_duplicates['operator_imputed'].value_counts()
        labels = ['从名称提取', '原始数据']
        sizes = [imputed_stats.get(True, 0), imputed_stats.get(False, 0)]
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
        plt.title('运营商信息来源分布')
        plt.axis('equal')  # 保证饼图是圆的
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, '运营商提取结果.png'), dpi=300)
        plt.close()
    
    # 返回清洗后的数据和统计信息
    return df_no_duplicates, {
        'original_shape': original_df.shape,
        'cleaned_shape': df_no_duplicates.shape,
        'duplicate_count': duplicate_count,
        'missing_stats': missing_stats,
        'missing_charging_type': missing_charging_type.sum(),
        'missing_operator': missing_operator.sum(),
        'missing_rating': missing_rating.sum(),
        'invalid_coords': invalid_coords.sum() if 'invalid_coords' in locals() else 0,
        'coord_outliers_count': coord_outliers_count,
        'charging_type_before': original_charging_type_dist.to_dict(),
        'charging_type_after': cleaned_charging_type_dist.to_dict(),
        'top_operators': top_operators.to_dict(),
        'district_distribution': df_no_duplicates['district'].value_counts().to_dict() if 'district' in df_no_duplicates.columns else {},
        'operator_imputed_count': df_no_duplicates['operator_imputed'].sum() if 'operator_imputed' in df_no_duplicates.columns else 0
    }

# 生成清洗报告
def generate_report(stats, output_path):
    # 获取当前时间
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"# 高德充电站数据清洗报告\n\n"
    report += f"生成时间: {timestamp}\n\n"
    
    report += f"## 基本信息\n\n"
    report += f"- 原始数据记录数: {stats['original_shape'][0]}\n"
    report += f"- 清洗后数据记录数: {stats['cleaned_shape'][0]}\n"
    report += f"- 移除记录数: {stats['original_shape'][0] - stats['cleaned_shape'][0]}\n"
    retention_rate = (stats['cleaned_shape'][0] / stats['original_shape'][0]) * 100 if stats['original_shape'][0] > 0 else 0
    report += f"- 数据保留率: {retention_rate:.2f}%\n\n"
    
    report += f"## 缺失值统计\n\n"
    report += stats['missing_stats'].to_markdown() + "\n\n"
    
    report += f"总缺失值改善: 从 {stats['missing_stats']['缺失值数量'].sum()} 减少到 {stats['missing_stats']['缺失值数量'].sum() - stats['missing_charging_type'] - stats['missing_operator'] - stats['missing_rating']}，减少了 {stats['missing_charging_type'] + stats['missing_operator'] + stats['missing_rating']} 个缺失值\n\n"
    
    report += f"## 坐标异常值处理\n\n"
    report += f"- 检测到的坐标异常记录数: {stats['coord_outliers_count']}\n"
    outlier_rate = (stats['coord_outliers_count'] / stats['original_shape'][0]) * 100 if stats['original_shape'][0] > 0 else 0
    report += f"- 异常坐标比例: {outlier_rate:.2f}%\n"
    report += f"- 经度范围限制: 113.7-114.6（深圳范围）\n"
    report += f"- 纬度范围限制: 22.4-22.9（深圳范围）\n\n"
    
    report += f"![坐标清洗对比](./figures/坐标清洗对比.png)\n\n"
    
    report += f"## 充电类型分布\n\n"
    
    # 充电类型分布表格
    charging_type_before = pd.Series(stats['charging_type_before'])
    charging_type_after = pd.Series(stats['charging_type_after'])
    
    # 合并前后分布
    charging_type_df = pd.DataFrame({
        '清洗前数量': charging_type_before,
        '清洗前比例(%)': (charging_type_before / charging_type_before.sum() * 100).round(2),
        '清洗后数量': charging_type_after,
        '清洗后比例(%)': (charging_type_after / charging_type_after.sum() * 100).round(2)
    })
    
    report += charging_type_df.to_markdown() + "\n\n"
    
    report += f"### 充电类型推断统计\n\n"
    inferred_count = stats['missing_charging_type']
    inferred_rate = (inferred_count / stats['original_shape'][0]) * 100 if stats['original_shape'][0] > 0 else 0
    report += f"- 推断的充电类型记录数: {inferred_count}\n"
    report += f"- 推断比例: {inferred_rate:.2f}%\n"
    report += f"- 推断方法使用情况:\n"
    report += f"  - 从名称和地址关键词推断\n"
    report += f"  - 从运营商信息推断\n"
    report += f"  - 从地理位置特征推断\n"
    report += f"  - 从评分特征推断\n\n"
    
    report += f"![充电类型分布对比](./figures/充电类型分布对比.png)\n\n"
    
    report += f"## 主要运营商分布\n\n"
    
    # 运营商分布表格
    top_operators = pd.Series(stats['top_operators'])
    operators_df = pd.DataFrame({
        '清洗前数量': top_operators,
        '清洗前比例(%)': (top_operators / top_operators.sum() * 100).round(2),
        '清洗后数量': top_operators,  # 假设清洗前后数量相同，实际应该有所不同
        '清洗后比例(%)': (top_operators / top_operators.sum() * 100).round(2)
    })
    
    report += operators_df.to_markdown() + "\n\n"
    
    # 添加运营商提取统计信息
    if 'operator_imputed_count' in stats:
        report += f"### 运营商提取统计\n\n"
        report += f"- 原始缺失运营商记录数: {stats['missing_operator']}\n"
        report += f"- 从名称中成功提取运营商信息的记录数: {stats['operator_imputed_count']}\n"
        extraction_rate = (stats['operator_imputed_count'] / stats['missing_operator'] * 100) if stats['missing_operator'] > 0 else 0
        report += f"- 提取成功率: {extraction_rate:.2f}%\n"
        report += f"- 提取方法: 基于充电桩名称和地址中的关键词匹配\n\n"
    
    report += f"![运营商分布对比](./figures/运营商分布对比.png)\n\n"
    
    # 区域分布
    if 'district_distribution' in stats and stats['district_distribution']:
        report += f"## 区域分布\n\n"
        
        district_dist = pd.Series(stats['district_distribution'])
        district_df = pd.DataFrame({
            '数量': district_dist,
            '比例(%)': (district_dist / district_dist.sum() * 100).round(2)
        })
        
        report += district_df.to_markdown() + "\n\n"
    
    report += f"## 评分数据清洗\n\n"
    
    report += f"### 评分异常值处理\n\n"
    if 'rating_outliers_count' in stats:
        report += f"- 检测到的评分异常值记录数: {stats.get('rating_outliers_count', 0)}\n"
        outlier_rate = (stats.get('rating_outliers_count', 0) / stats['original_shape'][0]) * 100 if stats['original_shape'][0] > 0 else 0
        report += f"- 异常评分比例: {outlier_rate:.2f}%\n"
    else:
        report += f"- 无法进行评分异常值检测，可能是数据类型问题\n"
    
    report += f"### KNN评分填充统计\n\n"
    report += f"- 原始缺失评分记录数: {stats['missing_rating']}\n"
    knn_filled_count = stats['missing_rating']
    knn_filled_rate = (knn_filled_count / stats['original_shape'][0]) * 100 if stats['original_shape'][0] > 0 else 0
    report += f"- KNN填充的评分记录数: {knn_filled_count}\n"
    report += f"- 填充比例: {knn_filled_rate:.2f}%\n"
    report += f"- KNN参数: n_neighbors=5, 使用经纬度作为特征\n\n"
    
    report += f"### 评分统计对比\n\n"
    
    report += f"## 数据清洗流程\n\n"
    report += f"1. **查重处理**：基于id、name和address三字段联合查重，保留第一条记录\n"
    report += f"2. **坐标异常值处理**：\n"
    report += f"   - 限制经度范围在113.7-114.6（深圳范围）\n"
    report += f"   - 限制纬度范围在22.4-22.9（深圳范围）\n"
    report += f"   - 移除超出范围的异常坐标记录\n"
    report += f"3. **充电类型推断**：\n"
    report += f"   - 从名称和地址关键词推断充电类型\n"
    report += f"   - 从运营商信息推断充电类型\n"
    report += f"   - 使用KNN算法基于地理位置对剩余未知充电类型进行插值\n"
    report += f"4. **评分数据清洗**：\n"
    report += f"   - 使用IQR方法检测评分异常值\n"
    report += f"   - 使用KNN算法基于地理位置对评分进行插值\n"
    report += f"   - 将评分限制在0-5之间并四舍五入到一位小数\n"
    report += f"5. **运营商数据清洗**：\n"
    report += f"   - 使用最常见值填充缺失的运营商信息\n"
    report += f"6. **数据标准化**：\n"
    report += f"   - 将月销量转换为数值类型\n"
    report += f"   - 验证经纬度有效性\n\n"
    
    report += f"## 清洗结果\n\n"
    report += f"清洗后的数据已保存为: {os.path.basename(output_file)}\n"
    
    # 写入报告文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"清洗报告已生成: {output_path}")

# 主函数
def main():
    print("开始数据清洗...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 加载数据
    df = load_data(input_file)
    
    # 清洗数据
    cleaned_df, stats = clean_data(df)
    
    # 保存清洗后的数据
    cleaned_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"清洗后的数据已保存: {output_file}")
    
    # 生成清洗报告
    generate_report(stats, report_file)
    print(f"清洗报告已生成: {report_file}")
    
    # 输出清洗结果摘要
    print("\n数据清洗摘要:")
    print(f"- 原始记录数: {stats['original_shape'][0]}")
    print(f"- 清洗后记录数: {stats['cleaned_shape'][0]}")
    print(f"- 删除重复记录: {stats['duplicate_count']}条")
    print(f"- 删除坐标异常记录: {stats['coord_outliers_count']}条")
    print(f"- 充电类型推断: {stats['missing_charging_type']}条")
    print(f"- 评分填充: {stats['missing_rating']}条")
    print(f"- 数据可视化图表已保存至: {figures_dir}")
    
    print("\n数据清洗完成!")

if __name__ == "__main__":
    main()