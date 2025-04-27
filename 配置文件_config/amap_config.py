# -*- coding: utf-8 -*-
"""
高德地图API配置文件
用于存储API密钥和相关配置参数
"""

# 高德Web服务API Key列表
AMAP_KEYS = [
    "4b87a5f754bd20c3db93cfe1b1fe8a9e",
    "d5ec9ad884dd05f88fdfa1f867b7a68c"
]

# API请求配置
API_REQUEST_CONFIG = {
    "max_retries": 3,        # 最大重试次数
    "timeout": 10,          # 请求超时时间（秒）
    "sleep_time": 0.5,      # 请求间隔时间（秒）
    "batch_size": 100,      # 批处理大小
    "max_daily_requests": 2000  # 每个key每日最大请求次数
}

# 搜索配置
SEARCH_CONFIG = {
    "radius": 500,           # 默认搜索半径（米）
    "max_pois": 50          # 每次搜索返回的最大POI数量
}