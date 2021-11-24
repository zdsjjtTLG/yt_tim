# -- coding: utf-8 --
# @Time    : 2021/11/24 0024 19:01
# @Author  : TangKai
# @Team    : SuperModel
# @File    : road_search.py
import pandas as pd
from src.road.modifynet.generate_gdf import generate_gdf
from src.conn.get_df import get_link_geo


# 从数据库中获取单向表示的路网
def get_road_net(db_type=None,
                 drivertype=None,
                 username=None,
                 password=None,
                 host=None,
                 port=None,
                 databasename=None, db=True):
    """

    :param db_type:
    :param drivertype:
    :param username:
    :param password:
    :param host:
    :param port:
    :param databasename:
    :param db:
    :return:
    单向表示的路网
    """
    # 读取道路表, 一天内道路表不重复读取
    # 双向形式表达的路网
    if db:
        double_link_df = get_link_geo(db_type=db_type,
                                      drivertype=drivertype,
                                      username=username,
                                      password=password,
                                      host=host,
                                      port=port,
                                      databasename=databasename)
    else:
        double_link_df = pd.read_csv(r'E:\work\yantai_tim\road_acess\data\input\link_df.csv',
                                     encoding='utf_8_sig')

    # 处理单向表达的link和带几何信息的node
    single_link_df, node_gdf = generate_gdf(link_df=double_link_df,
                                            wkt_cols='geom',
                                            from_node_field='from_node',
                                            to_node_field='to_node')
    return single_link_df, node_gdf


def road_cost(link_df=None, node_gdf=None, weight_field=None, export_field=None, sample_rate=None,
              day=None, time_period=None, db=True):
    """
    输入一个单向带权重的路网(未匹配实时路况), 指定搜路字段, 指定其他skim属性字段, 指定起点坐标, 指定抽样率, 输出此起点到所有其他抽样点的cost
    :param link_df:
    :param node_gdf:
    :return:
    """
    # 首先进行路况抓取
    if db:
        link_time = get_day_time_period_link_time(db_name=clc_db_name, user_name=clc_user_name,
                                                  password=clc_password,
                                                  host=clc_host, port=clc_port, day=day,
                                                  time_slice=time_period_dict[time_period])
    else:
        # test, 读取一天的某个时段的数据
        link_time = pd.read_csv(r'E:/work/yantai_tim/road_acess/data/input/link_time.csv', encoding='utf_8_sig')

    # 将道路路况时间匹配到路网上, 没有路况的路段, 参考同街道其他路段进行系数计算
    # from_node_id_field, to_node_id_field, total_time_field, length_field
    used_link_df = process_link_time(slice_link_time_df=link_time,
                                     link_df=origin_single_link_df)


    # 添加连杆, 不会改变used_link_df
    poi_link_df, node_gdf, origin_node = modify_net(origin_lon=float(poi_item[0]),
                                                    origin_lat=float(poi_item[1]),
                                                    link_df=used_link_df,
                                                    node_gdf=node_gdf,
                                                    buffer=1000,
                                                    maximum_number=3,
                                                    rod_speed=15)
