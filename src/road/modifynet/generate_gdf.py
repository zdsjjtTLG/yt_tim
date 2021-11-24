# -- coding: utf-8 --
# @Time    : 2021/11/23 0023 16:12
# @Author  : TangKai
# @Team    : SuperModel
# @File    : generate_gdf.py
from shapely import wkt
import geopandas as gpd
# import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import pandas as pd


geometry_field = 'geometry'
node_id_field = 'node_id'

# 路段几何表中link_id字段
link_id_field = 'fid'
direction_field = 'dir_type'
block_id_field = 'block_fid'
ffs_field = 'speed'
length_field = 'length'

from_node_id_field = 'from_node'
to_node_id_field = 'to_node'


# 主函数
def generate_gdf(link_df=None, wkt_cols='geom', from_node_field=None, to_node_field=None):
    link_df[wkt_cols] = link_df.apply(lambda x: x[wkt_cols].replace('MULTILINESTRING((', 'LINESTRING('), axis=1)
    link_df[wkt_cols] = link_df.apply(lambda x: x[wkt_cols].replace('))', ')'), axis=1)

    # 双向表达的路网
    link_df[geometry_field] = gpd.GeoSeries.from_wkt(link_df[wkt_cols])
    link_gdf = gpd.GeoDataFrame(link_df, geometry=geometry_field, crs='EPSG:4326')
    link_gdf.drop(columns=[wkt_cols], inplace=True, axis=1)

    node_gdf = generate_node_from_link(link_gdf=link_gdf, from_node_field=from_node_field, to_node_field=to_node_field)

    # 目前是双向表达的路网
    link_gdf = link_gdf[[link_id_field, direction_field,
                         from_node_id_field, to_node_id_field, length_field,
                         "ab_" + ffs_field, "ba_" + ffs_field, block_id_field, geometry_field]].copy()
    link_gdf.rename(columns={"ab_" + ffs_field: ffs_field + "_ab",
                             "ba_" + ffs_field: ffs_field + "_ba"}, inplace=True)

    # 路网转化为单向表达
    link_df = get_single_net(net_data=link_gdf, dir_field_name=direction_field,
                             from_node_name=from_node_id_field,
                             to_node_name=to_node_id_field, geo_bool=False)

    return link_df, node_gdf


def generate_node_from_link(link_gdf=None, from_node_field=None, to_node_field=None):
    used_link_gdf = link_gdf.copy()

    used_link_gdf['from_node_point'] = used_link_gdf.apply(lambda x: Point(list(x[geometry_field].coords)[0]), axis=1)
    used_link_gdf['to_node_point'] = used_link_gdf.apply(lambda x: Point(list(x[geometry_field].coords)[-1]), axis=1)

    from_point_gdf = used_link_gdf[[from_node_field, 'from_node_point']].copy()
    to_point_gdf = used_link_gdf[[to_node_field, 'to_node_point']].copy()

    from_point_gdf.rename(columns={from_node_field: node_id_field, 'from_node_point': geometry_field}, inplace=True)
    to_point_gdf.rename(columns={to_node_field: node_id_field, 'to_node_point': geometry_field}, inplace=True)
    node_gdf = pd.concat([from_point_gdf, to_point_gdf], axis=0)
    node_gdf.drop_duplicates(subset=[node_id_field], inplace=True)
    node_gdf.reset_index(inplace=True, drop=True)
    node_gdf = gpd.GeoDataFrame(node_gdf, geometry=geometry_field, crs='EPSG:4326')
    return node_gdf


def get_single_net(net_data=None, cols_field_name_list=None, dir_field_name=None,
                   from_node_name=None, to_node_name=None, geo_bool=True):
    """将具有方向字段的路网格式转化为单向的路网格式(没有方向字段, 仅靠from_node, to_node即可判别方向)
    :param net_data: pd.DataFrame, 原路网数据
    :param cols_field_name_list: list, 列名称列表
    :param dir_field_name: str, 原路网数据代表方向的字段名称
    :param from_node_name: str, 原路网数据代表拓扑起始结点的字段名称
    :param to_node_name: str, 原路网数据代表拓扑终端结点的字段名称
    :param geo_bool: bool, 路网数据是否带几何列
    :return: gpd.DatFrame or pd.DatFrame
    """

    if cols_field_name_list is None:
        cols_field_name_list = list(net_data.columns)

    # 找出双向字段, 双向字段都应该以_ab或者_ba结尾
    two_way_field_list = list()
    for cols_name in cols_field_name_list:
        if cols_name.endswith('_ab') or cols_name.endswith('_ba'):
            two_way_field_list.append(cols_name[:-3])
    two_way_field_list = list(set(two_way_field_list))
    ab_field_del = [x + '_ab' for x in two_way_field_list]
    ba_field_del = [x + '_ba' for x in two_way_field_list]
    ab_rename_dict = {x: y for x, y in zip(ab_field_del, two_way_field_list)}
    ba_rename_dict = {x: y for x, y in zip(ba_field_del, two_way_field_list)}

    # 方向为拓扑反向的
    net_negs = net_data[net_data[dir_field_name] == -1].copy()
    net_negs.drop(ab_field_del, axis=1, inplace=True)
    net_negs.rename(columns=ba_rename_dict, inplace=True)
    net_negs['temp'] = net_negs[from_node_name]
    net_negs[from_node_name] = net_negs[to_node_name]
    net_negs[to_node_name] = net_negs['temp']
    if geo_bool:
        net_negs[geometry_field] = net_negs[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))
    net_negs.drop(['temp', dir_field_name], inplace=True, axis=1)

    # 方向为拓扑正向的
    net_poss = net_data[net_data[dir_field_name] == 1].copy()
    net_poss.drop(ba_field_del, axis=1, inplace=True)
    net_poss.rename(columns=ab_rename_dict, inplace=True)
    net_poss.drop([dir_field_name], inplace=True, axis=1)

    # 方向为拓扑双向的, 改为拓扑正向
    net_zero_poss = net_data[net_data[dir_field_name] == 0].copy()
    net_zero_poss[dir_field_name] = 1
    net_zero_poss.drop(ba_field_del, axis=1, inplace=True)
    net_zero_poss.rename(columns=ab_rename_dict, inplace=True)
    net_zero_poss.drop([dir_field_name], inplace=True, axis=1)

    # 方向为拓扑双向的, 改为拓扑反向
    net_zero_negs = net_data[net_data[dir_field_name] == 0].copy()
    net_zero_negs.drop(ab_field_del, axis=1, inplace=True)
    net_zero_negs.rename(columns=ba_rename_dict, inplace=True)
    net_zero_negs['temp'] = net_zero_negs[from_node_name]
    net_zero_negs[from_node_name] = net_zero_negs[to_node_name]
    net_zero_negs[to_node_name] = net_zero_negs['temp']
    if geo_bool:
        net_zero_negs[geometry_field] = net_zero_negs[geometry_field].apply(lambda x: LineString(list(x.coords)[::-1]))
    net_zero_negs.drop(['temp', dir_field_name], inplace=True, axis=1)

    net = net_poss.append(net_zero_poss, ignore_index=True)
    net = net.append(net_negs, ignore_index=True)
    net = net.append(net_zero_negs, ignore_index=True)

    net.reset_index(inplace=True, drop=True)

    return net

