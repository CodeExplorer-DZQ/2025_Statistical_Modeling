import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（修改为实际路径）
df = pd.read_csv(r'd:\DZQ_Projects_项目合集\Competitions\2025_Statistical_Modeling_competition\EV_item_for_Trae\数据_data\1_充电桩数据\processed\深圳高德最终数据_cleaned_0417-0419.csv', usecols=['lng','lat','charging_type','operator','district','rating'], encoding='GBK')

# 配置紫色系主题
plt.style.use('ggplot')
sns.set_palette("Purples_r")
colors = ['#4B0082','#9370DB','#D8BFD8']  # 深紫/中紫/浅紫

# 图表1：充电桩地理分布（紫色密度图）
plt.figure(figsize=(12,8))
sns.kdeplot(x=df['lng'], y=df['lat'], cmap='Purples', fill=True, thresh=0)
plt.title('深圳充电桩空间密度分布', fontsize=16, fontproperties='SimHei')
plt.xlabel('经度', fontproperties='SimHei')
plt.ylabel('纬度', fontproperties='SimHei')

# 图表2：行政区充电桩数量TOP10（横向柱状图）
plt.figure(figsize=(12,6))
district_counts = df['district'].value_counts().head(10)
sns.barplot(x=district_counts.values, y=district_counts.index, palette=colors)
plt.title('充电桩数量TOP10行政区', fontsize=14, fontproperties='SimHei')
plt.xlabel('数量', fontproperties='SimHei')

# 图表3：充电类型占比（环形图）
plt.figure(figsize=(8,8))
type_counts = df['charging_type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, 
        colors=colors, autopct='%1.1f%%',
        wedgeprops=dict(width=0.3))
plt.title('快充/慢充比例', fontsize=14, fontproperties='SimHei')

# 图表4：运营商市场份额TOP10（点阵图）
plt.figure(figsize=(12,6))
operator_top10 = df['operator'].value_counts().head(10).index
sns.stripplot(x='operator', y='rating', data=df[df['operator'].isin(operator_top10)],
              palette='Purples', jitter=0.25)
plt.xticks(rotation=45, fontproperties='SimHei')
plt.title('TOP10运营商服务质量分布', fontsize=14, fontproperties='SimHei')
plt.xlabel('运营商', fontproperties='SimHei')
plt.ylabel('用户评分', fontproperties='SimHei')

plt.tight_layout()
plt.show()