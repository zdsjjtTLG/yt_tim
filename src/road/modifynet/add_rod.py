# -- coding: utf-8 --
# @Time    : 2021/7/1 0001 16:46
# @Author  : TangKai
# @Team    : SuperModel
# @File    : connect_centroids.py

"""生成小区形心连杆, 需要提供路网数据和形心点层数据"""

import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
import time
import pandas as pd

# 线层数据、点层数据必需字段
length_field = 'length'  # 线层的长度, m
from_node_id_field = 'from_node'  # 线层的拓扑起始结点
to_node_id_field = 'to_node'  # 线层的拓扑终到结点
node_id_field = 'node_id'  # 点层的id
geometry_field = 'geometry'  # 几何属性字段
total_time_field = 'total_time'


def modify_net(origin_lon=None, origin_lat=None,
               link_df=None, node_gdf=None,
               buffer=1000, maximum_number=3, rod_speed=15):
    """

    :param origin_lon: float, 起点的经度, EPSG:4326
    :param origin_lat: float, 起点的纬度, EPSG:4326
    :param link_df: pd.DataFrame
    :param node_gdf: gpd.GeoDataFrame
    :param buffer: int or float, 搜索半径
    :param maximum_number: int, 最大连杆数目
    :param rod_speed: int, 连杆的步行速度, km/h
    :return:
    """

    link_df = link_df.copy()

    # 坐标转换
    node_gdf = node_gdf.to_crs('EPSG:32650')

    # 起点转化为gdf
    origin_point_gdf = gpd.GeoDataFrame({geometry_field: [Point((origin_lon, origin_lat))]},
                                        geometry=geometry_field, crs='EPSG:4326')
    origin_point_gdf = origin_point_gdf.to_crs('EPSG:32650')

    link_df, node_gdf, origin_node = connect_centroids_to_node(link_df=link_df, node_gdf=node_gdf,
                                                               centroids_gdf=origin_point_gdf,
                                                               maximum_distance=buffer, maximum_number=maximum_number,
                                                               rod_speed=rod_speed)
    return link_df, node_gdf, origin_node


# 连杆功能主函数, 形心连接到node
def connect_centroids_to_node(link_df=None, node_gdf=None, centroids_gdf=None,
                              maximum_distance=10.0, maximum_number=1,
                              rod_speed=15):
    """
    将形心点连接到邻近的结点上, 使用 'EPSG:32650'
    :param link_df: pd.DataFrame, 线层数据(无几何列)
    :param node_gdf: gpd.GeoDataFrame, 点层数据
    :param centroids_gdf: gpd.GeoDataFrame, 形心点数据
    :param maximum_distance: float, 搜索半径, m
    :param maximum_number: int, 一个形心点生成连杆的最大数目
    :param rod_speed: float, 连杆上的速度, km/h
    :return: 点层、线层的gpd.GeoDataFrame、新节点的ID

    """

    # 规避重名问题
    rename_dict = avoid_duplicate_cols(built_in_col_list=['index_left', 'index_right', '__geo__', '_centroids'],
                                       df=node_gdf)

    # 记录结点的几何信息, 防止sjoin后几何信息丢失
    node_gdf['__geo__'] = node_gdf[geometry_field]

    # 对每一个形心点做buffer
    centroids_buffer_gdf = gpd.GeoDataFrame([], geometry=centroids_gdf.buffer(maximum_distance))

    # 使用gdf_buffer和所有的node做sjoin
    join_data = gpd.sjoin(left_df=centroids_buffer_gdf, right_df=node_gdf, op='intersects', how='left')
    join_data.dropna(subset=['__geo__'], inplace=True, axis=0)

    if join_data.empty:
        raise ValueError(f'指定起点附近{maximum_distance}m内无道路节点!')
    else:
        # 计算距离在形心点buffer范围内的结点到其的距离
        join_data['__dis__'] = join_data[[geometry_field, '__geo__']]. \
            apply(lambda x: x[1].distance(Point(x[0].centroid)), axis=1)

        # 得到连杆信息, 更新字段后将其插入link_gdf中 #

        # 分组排序函数
        def filter_group(x):
            # 按照点到点的距离排序
            sorted_x = x.sort_values(by='__dis__', ascending=True)

            if len(sorted_x) < maximum_number:
                return sorted_x
            else:
                return sorted_x[:maximum_number]

        # 按照join_data的索引(形心数据的索引)分组, 取出每组排名前maximum_number的数据
        connecting_rod_df = join_data[['index_right', '__dis__']].groupby(level=0).apply(filter_group)

        # connecting_rod_data的索引是双重索引
        connecting_rod_df.reset_index(inplace=True, drop=False)
        connecting_rod_df.rename(columns={'level_1': 'index_left'}, inplace=True)
        connecting_rod_df[geometry_field] = connecting_rod_df[['index_left', 'index_right']]. \
            apply(lambda x: LineString([centroids_gdf.at[x[0], geometry_field], node_gdf.at[x[1], geometry_field]]),
                  axis=1)

        connecting_rod_gdf = gpd.GeoDataFrame(connecting_rod_df, geometry=geometry_field)

        # 修改形心点的id, 从原节点表的最大id开始编号
        max_node_id = max(node_gdf[node_id_field].to_list())
        centroids_gdf['__new_id__'] = [x + max_node_id for x in range(1, len(centroids_gdf) + 1)]

        # 更新连杆表的基本字段, link_id, from_node, to_node, dir, length
        connecting_rod_gdf[from_node_id_field] = \
            connecting_rod_gdf['index_left'].apply(lambda x: centroids_gdf.at[x, '__new_id__'])

        connecting_rod_gdf[to_node_id_field] = \
            connecting_rod_gdf['index_right'].apply(lambda x: node_gdf.at[x, node_id_field])

        connecting_rod_gdf[length_field] = connecting_rod_gdf[geometry_field].apply(lambda x: x.length)
        connecting_rod_gdf[total_time_field] = connecting_rod_gdf[length_field].apply(lambda x: 3.6 * x / rod_speed)

        # 删除无关字段
        connecting_rod_gdf.drop(columns=['index_left', 'index_right', 'level_0', '__dis__', geometry_field],
                                axis=1, inplace=True)
        link_df = link_df.append(connecting_rod_gdf)
        link_df.reset_index(inplace=True, drop=True)

        # 将形心数据插入node_gdf中 #
        new_node_gdf = centroids_gdf[['__new_id__', geometry_field]].copy()
        new_node_gdf.rename(columns={'__new_id__': node_id_field}, inplace=True)
        new_node_id = new_node_gdf.at[0, node_id_field]

        node_gdf = node_gdf.append(new_node_gdf)

        # 删除__geo__字段
        node_gdf.drop(columns=['__geo__'], axis=1, inplace=True)

        # 恢复重名字段
        if rename_dict:
            reverse_rename_dict = dict((v, k) for k, v in rename_dict.items() if k != '_centroids')
            node_gdf.rename(columns=reverse_rename_dict, inplace=True)

    node_gdf = node_gdf.to_crs('EPSG:4326')
    node_gdf.reset_index(inplace=True, drop=True)

    return link_df, node_gdf, new_node_id


# 逻辑子模块
def avoid_duplicate_cols(built_in_col_list=None, df=None):
    """
    重命名数据表中和内置名称冲突的字段
    :param built_in_col_list: list, 要使用的内置名称字段列表
    :param df: pd.DataFrame, 数据表
    :return: dict
    """

    rename_dict = dict()

    # 数据表的所有列名称
    df_cols_list = list(df.columns)

    # 遍历每一个在函数内部需要使用的内置字段, 检查其是否已经存在数据表字段中
    for built_in_col in built_in_col_list:
        if built_in_col in df_cols_list:
            num = 1
            while '_'.join([built_in_col, str(num)]) in df_cols_list:
                num += 1
            rename_col = '_'.join([built_in_col, str(num)])
            rename_dict[built_in_col] = rename_col
        else:
            pass
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    return rename_dict


if __name__ == '__main__':
    pass


