import pandas as pd
from tqdm import tqdm
import json
import requests
import os
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
from pykrige.ok import OrdinaryKriging
from shapely.geometry import LineString
from shapely.ops import unary_union
from geopy.distance import distance
from geopy.distance import geodesic
from shapely.geometry import Point
from shapely.geometry import Polygon
from skimage import measure

lon_field = 'lon'  # 结点经度
lat_field = 'lat'  # 结点纬度
level_col = 'tim_level'


def get_pt_cost(grid_gdf=None, sample_rate=None, origin_loc=None, key=None, city=None, walk_speed=1.5):
    """

    :param grid_gdf: gpd.GeoDataFrame
    :param sample_rate: float, 0.05
    :param origin_loc: tuple, (lng, lat)
    :param key:
    :param city:
    :param walk_speed:
    :return:
    """

    # 从栅格数据中随机抽样, 用作起点
    data_routing = grid_gdf.sample(int(len(grid_gdf) * sample_rate),
                                   random_state=1,
                                   ignore_index=True)

    data_routing['lat'] = (data_routing['top'] + data_routing['bottom']) / 2
    data_routing['lon'] = (data_routing['left'] + data_routing['right']) / 2

    # 起点坐标
    origin_location = str(origin_loc[0]) + ',' + str(origin_loc[1])

    # 计算指定起点到其他所有节点的最短路
    trip_data = route_planning(from_location=origin_location, od_data=data_routing,
                               key=key, city=city, walk_speed=walk_speed)

    return trip_data


# 指定一个起终点坐标
def route_planning(from_location=None, od_data=None, key=None, city=None, walk_speed=1.5):
    """
    指定一个起点坐标和含有 'lng' 列, 'lat' 列的df, 进行高德的请求
    :param from_location: str, (121.091, 21.001)
    :param od_data: pd.DataFrame
    :param key: str, 高德KEY
    :param walk_speed: float, 步行速度, m/s
    :param city: str, 城市, '烟台'
    :return:
    """
    route_list = []
    for k in tqdm(range(0, len(od_data))):
        to_location = str(od_data.loc[k, 'lon']) + ',' + str(od_data.loc[k, 'lat'])
        parameters = {
            'key': key,
            'origin': str(from_location),
            'destination': str(to_location),
            'extensions': 'all',
            'output': 'json',
            'city': city,
        }

        pt_url = "https://restapi.amap.com/v3/direction/transit/integrated"  # 公交
        pt_response = request_url_get(url=pt_url, parameters=parameters)

        route_res = pd.DataFrame()
        # 如果请求失败
        if pt_response is None:
            print('请求超时!')
        else:
            pt_txt = json.loads(pt_response)  # 把数据变成字典格式
            if pt_txt['status'] == '0':
                print('传参错误!')

            else:
                t = []
                # 没有公交方案, 直接步行
                if not pt_txt['route']['transits']:
                    # 返回数据为空
                    if not pt_txt['route']['distance']:
                        print('无有效数据!')

                    else:
                        # 距离
                        trip_distance = int(pt_txt['route']['distance'])

                        # 步行距离
                        walking_distance = int(pt_txt['route']['distance'])

                        # 步行时间
                        walk_time = walking_distance / walk_speed

                        # 此换乘方案预期时间
                        duration = walk_time

                        t.append([trip_distance, walking_distance, duration])

                else:

                    for a in pt_txt['route']['transits']:
                        trip_distance = int(a['distance'])

                        walking_distance = int(a['walking_distance'])

                        duration = int(a['duration'])

                        t.append([trip_distance, walking_distance, duration])

                route_res = pd.DataFrame(t, columns=['transit_distance', 'walking_distance', 'transit_time'])

        route_res['ori_FID'] = od_data.loc[0, 'FID_11']
        route_res['des_FID'] = od_data.loc[k, 'FID_11']
        route_res['lon'] = od_data.loc[k, 'lon']
        route_res['lat'] = od_data.loc[k, 'lat']
        route_shortest = route_res[0:1]
        route_list.append(route_shortest)
    trip_data = pd.concat(route_list)
    trip_df = trip_data[['lon', 'lat', 'transit_distance', 'transit_time']]
    trip_df['lon'] = trip_df['lon'].astype(float)
    trip_df['lat'] = trip_df['lat'].astype(float)
    trip_df = trip_df.append(pd.DataFrame({'lon': [float(from_location[0])],
                                           'lat': [float(from_location[1])],
                                           'transit_distance': [0],
                                           'transit_time': [0]}))

    return trip_df


def request_url_get(url=None, parameters=None):
    try:
        r = requests.get(url=url, params=parameters, timeout=30)
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        print('请求url返回错误异常')
        return None


if __name__ == '__main__':
    pass

