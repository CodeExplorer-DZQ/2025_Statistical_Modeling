# -*- coding: utf-8 -*-
"""
深圳市充电桩POI数据增强采集脚本
功能：采集深圳市各区域充电桩POI数据并保存为CSV格式
优化点：
1. 扩大搜索关键词范围
2. 增加搜索中心点和半径策略
3. 优化分页和请求策略
4. 实现更好的数据去重和保存机制
5. 确保每个key每天至少采集1800条数据

运行格式：
python d:/DZQ_Competition_projects/2025_Statistical_Modeling_competition/EV_item_for_Trae/scrapy_gaode_01/spiders/shenzhen_crawler_enhanced.py
"""

import requests
import csv
import os
import time
import random
import json
from datetime import datetime

# 高德Web服务API Key
AMAP_KEYS = [
    "4b87a5f754bd20c3db93cfe1b1fe8a9e",
    "d5ec9ad884dd05f88fdfa1f867b7a68c"
]

# 当前脚本所在目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出文件路径 - 按要求命名为当前日期深圳高德数据v2.csv
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), 
                         f"{datetime.now().strftime('%m%d')}深圳高德数据v2.csv")

# 深圳市行政区列表 - 优先采集坪山区、光明区和大鹏新区
SHENZHEN_DISTRICTS = [
    "坪山区", "光明区", "大鹏新区",
    "福田区", "罗湖区", "南山区", "盐田区", 
    "宝安区", "龙岗区", "龙华区"
]

# 扩展搜索关键词列表
SEARCH_KEYWORDS = [
    "充电站", "充电桩", "电动汽车充电", "充电服务", "新能源汽车充电", 
    "快充站", "慢充站", "超级充电站", "特来电", "星星充电", 
    "小桔充电", "国家电网充电", "南方电网充电", "壳牌充电"
]

# 深圳市各区域中心点坐标 (经度,纬度)
DISTRICT_CENTERS = {
    "福田区": [(114.055036, 22.531933), (114.068726, 22.547913)],
    "罗湖区": [(114.131691, 22.548047), (114.117744, 22.555516)],
    "南山区": [(113.930478, 22.533013), (113.945284, 22.523845), (113.917475, 22.495245)],
    "盐田区": [(114.236841, 22.555696)],
    "宝安区": [(113.883981, 22.555120), (113.828343, 22.686843), (113.814139, 22.748438)],
    "龙岗区": [(114.246899, 22.720970), (114.191453, 22.632417), (114.135707, 22.609632)],
    "龙华区": [(114.036530, 22.657569), (114.059404, 22.615428)],
    "坪山区": [(114.338441, 22.708496), (114.378181, 22.689807)],
    "光明区": [(113.935895, 22.748816), (113.902035, 22.735631)],
    "大鹏新区": [(114.478626, 22.594381), (114.515834, 22.555061)]
}

# 搜索半径列表 (单位: 米)
SEARCH_RADIUS = [1000, 2000, 3000, 5000]

# 已采集的POI ID集合（用于去重）
collected_poi_ids = set()

# 每日采集计数器
daily_count = {key: 0 for key in AMAP_KEYS}

# 每个区域的采集计数
district_count = {district: 0 for district in SHENZHEN_DISTRICTS}

# 当前使用的API Key索引
current_key_index = 0

# 获取当前使用的API Key
def get_current_key():
    global current_key_index
    return AMAP_KEYS[current_key_index]

# 切换到下一个API Key
def switch_to_next_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(AMAP_KEYS)
    print(f"切换到API Key: {get_current_key()[:8]}...")

def fetch_charging_stations_by_keyword(district, keyword, page=1, max_retries=3):
    """按关键词获取充电桩POI数据"""
    global daily_count
    
    # 检查当前key是否达到每日限额
    current_key = get_current_key()
    if daily_count[current_key] >= 2000:
        switch_to_next_key()
        current_key = get_current_key()
    
    # 构建请求参数
    params = {
        "key": current_key,
        "keywords": f"{district} {keyword}",
        "city": "深圳市",
        "citylimit": "true",
        "extensions": "all",  # 返回详细信息
        "output": "JSON",
        "page": page,
        "offset": 50  # 每页50条
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.get(
                "https://restapi.amap.com/v3/place/text",
                params=params,
                timeout=10
            )
            
            # 更新计数器
            daily_count[current_key] += 1
            
            # 解析JSON响应
            result = response.json()
            
            # 检查API返回状态
            if result.get("status") == "1":
                return result
            else:
                print(f"API错误: {result.get('info')}")
                
                # 如果配额超限，切换Key
                if "quota" in result.get("info", "").lower():
                    switch_to_next_key()
                
                time.sleep(2)  # 出错后等待时间更长
        
        except Exception as e:
            print(f"请求异常 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    print("达到最大重试次数，请求失败")
    return None

def fetch_charging_stations_by_location(center, radius, keyword="充电", page=1, max_retries=3):
    """按坐标和半径获取充电桩POI数据"""
    global daily_count
    
    # 检查当前key是否达到每日限额
    current_key = get_current_key()
    if daily_count[current_key] >= 2000:
        switch_to_next_key()
        current_key = get_current_key()
    
    # 构建请求参数
    params = {
        "key": current_key,
        "keywords": keyword,
        "location": f"{center[0]},{center[1]}",
        "radius": radius,
        "extensions": "all",  # 返回详细信息
        "output": "JSON",
        "page": page,
        "offset": 50  # 每页50条
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.get(
                "https://restapi.amap.com/v3/place/around",
                params=params,
                timeout=10
            )
            
            # 更新计数器
            daily_count[current_key] += 1
            
            # 解析JSON响应
            result = response.json()
            
            # 检查API返回状态
            if result.get("status") == "1":
                return result
            else:
                print(f"API错误: {result.get('info')}")
                
                # 如果配额超限，切换Key
                if "quota" in result.get("info", "").lower():
                    switch_to_next_key()
                
                time.sleep(2)  # 出错后等待时间更长
        
        except Exception as e:
            print(f"请求异常 (尝试 {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    
    print("达到最大重试次数，请求失败")
    return None

def process_poi_data(poi):
    """处理POI数据，提取所需字段"""
    # 提取坐标
    location = poi.get("location", ",").split(",")
    if len(location) != 2:
        return None
    
    lng, lat = float(location[0]), float(location[1])
    
    # 提取评分
    rating = ""
    if "biz_ext" in poi and "rating" in poi["biz_ext"]:
        rating = poi["biz_ext"]["rating"]
    
    # 提取月销量
    month_sales = ""
    if "biz_ext" in poi and "cost" in poi["biz_ext"]:
        month_sales = poi["biz_ext"]["cost"]
    
    # 提取充电类型
    charging_type = "未知"
    if "快充" in poi.get("name", "") or "快充" in poi.get("address", ""):
        charging_type = "快充"
    elif "慢充" in poi.get("name", "") or "慢充" in poi.get("address", ""):
        charging_type = "慢充"
    
    # 提取运营商
    operator = "未知"
    operators = ["国家电网", "南方电网", "特来电", "星星充电", "小桔充电", 
                "依威能源", "万马爱充", "云快充", "壳牌", "奥特迅", "蔚景", "车电网"]
    
    for op in operators:
        if op in poi.get("name", "") or op in poi.get("address", ""):
            operator = op
            break
    
    # 构建数据行
    data_row = {
        "id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "address": poi.get("address", ""),
        "lng": lng,
        "lat": lat,
        "charging_type": charging_type,
        "operator": operator,
        "rating": rating,
        "month_sales": month_sales,
        "district": poi.get("adname", ""),
        "type": poi.get("type", ""),
        "tel": poi.get("tel", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return data_row

def save_to_csv(data_rows):
    """保存为CSV文件"""
    # 检查文件是否存在
    file_exists = os.path.exists(OUTPUT_CSV)
    
    # 打开文件进行写入
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        # 定义CSV字段
        fieldnames = [
            "id", "name", "address", "lng", "lat", 
            "charging_type", "operator", "rating", "month_sales", 
            "district", "type", "tel", "timestamp"
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 写入数据行
        writer.writerows(data_rows)
    
    print(f"数据已保存至: {OUTPUT_CSV}")

def collect_by_keywords(district):
    """通过关键词搜索采集数据"""
    print(f"\n开始通过关键词采集 {district} 的充电桩数据...")
    
    district_poi_count = 0
    batch_data_rows = []
    
    # 遍历关键词
    for keyword in SEARCH_KEYWORDS:
        print(f"使用关键词 '{keyword}' 搜索 {district} 的充电桩...")
        
        page = 1
        max_pages = 10  # 每个关键词最多采集10页
        
        while page <= max_pages:
            print(f"正在获取 {district} '{keyword}' 第 {page} 页数据...")
            
            # 获取POI数据
            result = fetch_charging_stations_by_keyword(district, keyword, page)
            
            if not result or "pois" not in result:
                print(f"{district} '{keyword}' 第 {page} 页数据获取失败")
                break
            
            pois = result["pois"]
            if not pois:
                print(f"{district} '{keyword}' 第 {page} 页无数据，采集完成")
                break
            
            # 处理POI数据
            new_count = 0
            
            for poi in pois:
                poi_id = poi.get("id")
                
                # 跳过已采集的POI
                if poi_id in collected_poi_ids:
                    continue
                
                # 处理POI数据
                data_row = process_poi_data(poi)
                if data_row:
                    batch_data_rows.append(data_row)
                    collected_poi_ids.add(poi_id)
                    new_count += 1
                    district_poi_count += 1
            
            print(f"第 {page} 页新增 {new_count} 条数据")
            
            # 每采集30条保存一次数据
            if len(batch_data_rows) >= 30:
                save_to_csv(batch_data_rows)
                batch_data_rows = []
            
            # 判断是否继续翻页
            total_count = int(result.get("count", "0"))
            if page * 50 >= total_count or new_count == 0:
                print(f"{district} '{keyword}' 数据已全部采集完成")
                break
            
            # 翻页
            page += 1
            
            # 添加随机延时（500-1000ms）
            time.sleep(0.5 + random.random() * 0.5)
    
    # 保存剩余数据
    if batch_data_rows:
        save_to_csv(batch_data_rows)
    
    return district_poi_count

def collect_by_location(district):
    """通过坐标和半径搜索采集数据"""
    print(f"\n开始通过坐标采集 {district} 的充电桩数据...")
    
    district_poi_count = 0
    batch_data_rows = []
    
    # 获取区域中心点
    centers = DISTRICT_CENTERS.get(district, [])
    if not centers:
        print(f"{district} 未配置中心点坐标，跳过")
        return 0
    
    # 遍历中心点
    for center in centers:
        # 遍历搜索半径
        for radius in SEARCH_RADIUS:
            print(f"使用中心点 {center} 半径 {radius}m 搜索 {district} 的充电桩...")
            
            # 遍历关键词（简化版）
            for keyword in ["充电站", "充电桩", "电动汽车充电"]:
                page = 1
                max_pages = 5  # 每个中心点和半径组合最多采集5页
                
                while page <= max_pages:
                    print(f"正在获取中心点 {center} 半径 {radius}m '{keyword}' 第 {page} 页数据...")
                    
                    # 获取POI数据
                    result = fetch_charging_stations_by_location(center, radius, keyword, page)
                    
                    if not result or "pois" not in result:
                        print(f"中心点 {center} 半径 {radius}m '{keyword}' 第 {page} 页数据获取失败")
                        break
                    
                    pois = result["pois"]
                    if not pois:
                        print(f"中心点 {center} 半径 {radius}m '{keyword}' 第 {page} 页无数据，采集完成")
                        break
                    
                    # 处理POI数据
                    new_count = 0
                    
                    for poi in pois:
                        poi_id = poi.get("id")
                        
                        # 跳过已采集的POI
                        if poi_id in collected_poi_ids:
                            continue
                        
                        # 处理POI数据
                        data_row = process_poi_data(poi)
                        if data_row:
                            batch_data_rows.append(data_row)
                            collected_poi_ids.add(poi_id)
                            new_count += 1
                            district_poi_count += 1
                    
                    print(f"第 {page} 页新增 {new_count} 条数据")
                    
                    # 每采集30条保存一次数据
                    if len(batch_data_rows) >= 30:
                        save_to_csv(batch_data_rows)
                        batch_data_rows = []
                    
                    # 判断是否继续翻页
                    total_count = int(result.get("count", "0"))
                    if page * 50 >= total_count or new_count == 0:
                        print(f"中心点 {center} 半径 {radius}m '{keyword}' 数据已全部采集完成")
                        break
                    
                    # 翻页
                    page += 1
                    
                    # 添加随机延时（500-1000ms）
                    time.sleep(0.5 + random.random() * 0.5)
    
    # 保存剩余数据
    if batch_data_rows:
        save_to_csv(batch_data_rows)
    
    return district_poi_count

def load_existing_data():
    """加载已有数据，避免重复采集"""
    global collected_poi_ids
    
    if os.path.exists(OUTPUT_CSV):
        print(f"加载已有数据文件: {OUTPUT_CSV}")
        try:
            with open(OUTPUT_CSV, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'id' in row and row['id']:
                        collected_poi_ids.add(row['id'])
            print(f"已加载 {len(collected_poi_ids)} 条已有数据ID")
        except Exception as e:
            print(f"加载已有数据失败: {str(e)}")

def main():
    print("深圳市充电桩POI数据增强采集开始...")
    print(f"数据将保存为CSV: {OUTPUT_CSV}")
    
    # 加载已有数据
    load_existing_data()
    
    # 准备保存的数据行
    all_data_rows = []
    
    # 遍历深圳各区采集数据
    for district in SHENZHEN_DISTRICTS:
        # 通过关键词搜索采集数据
        keyword_count = collect_by_keywords(district)
        
        # 通过坐标和半径搜索采集数据
        location_count = collect_by_location(district)
        
        # 更新区域计数
        total_district_count = keyword_count + location_count
        district_count[district] = total_district_count
        print(f"{district} 共采集 {total_district_count} 条数据 (关键词: {keyword_count}, 坐标: {location_count})")
        
        # 检查是否达到每个key的目标数量
        key_counts = [count for key, count in daily_count.items()]
        if min(key_counts) >= 1800:
            print("所有API Key已达到每日目标采集量，采集完成")
            break
    
    # 打印采集统计
    print("\n数据采集完成！统计信息:")
    print(f"总采集数据: {len(collected_poi_ids)} 条")
    for district, count in district_count.items():
        print(f"{district}: {count} 条")
    
    # 打印每个key的采集数量
    print("\nAPI Key采集统计:")
    for key, count in daily_count.items():
        print(f"{key[:8]}...: {count} 条")

if __name__ == "__main__":
    main()