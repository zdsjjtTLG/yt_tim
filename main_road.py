import os
import configparser
import pandas as pd
import geopandas as gpd
from src.tim.road_access import main_road_atos
from src.road.modifynet.generate_gdf import generate_gdf
from src.road.link_time.process import process_link_time
from src.road.modifynet.add_rod import modify_net
from src.transit.get_pt_time import get_pt_cost
from src.conn.get_df import get_link_geo, get_day_time_period_link_time

if __name__ == '__main__':

    # 读取参数 #
    config = configparser.ConfigParser()
    config.read(os.path.abspath('./config/config.ini'), encoding='utf-8')
    input_fldr = config.get('calculate_config', 'input_fldr')
    output_fldr = config.get('calculate_config', 'output_fldr')
    origin_lon = float(config.get('calculate_config', 'origin_lon'))
    origin_lat = float(config.get('calculate_config', 'origin_lat'))
    sample_rate = float(config.get('calculate_config', 'sample_rate'))
    cost_weight = str(config.get('calculate_config', 'cost_weight'))
    time_interval = float(config.get('calculate_config', 'time_interval'))

    # clickHouse数据库
    clc_db_name = 'tpi_sdyt'
    clc_user_name = 'default'
    clc_password = 'sutpc1234'
    clc_host = '10.10.5.109'
    clc_port = '9000'

    # pg数据库
    pg_db_type = 'postgresql'
    pg_driver_type = 'psycopg2'
    pg_username = 'postgres'
    pg_password = '123456'
    pg_host = '10.10.5.108'
    pg_port = '5432'
    pg_databasename = 'tpi_yt'

    # 获取热点图层相关信息(静态数据)
    poi_gdf = gpd.read_file(r'data/input/tpi_poi.shp')
    poi_gdf['loc'] = poi_gdf.apply(lambda x: (x['id'], x['poi_lng'], x['poi_lat']), axis=1)
    poi_loc_list = poi_gdf['loc'].to_list()

    # 栅格图层信息
    grid_gdf = gpd.read_file(os.path.join(input_fldr, 'yt500.shp'), crs='EPSG:4326')

    day_list = []
    time_period_dict = {'平峰': [108, 210],
                        '早高峰': [84, 108],
                        '晚高峰': [210, 234],
                        '全天': [1, 288]}
    time_period_id_dict = {'平峰': 1,
                           '早高峰': 2,
                           '晚高峰': 3,
                           '全天': 4,
                           '早或晚': 5}

    # # 读取道路表, 一天内道路表不重复读取
    # # 双向形式表达的路网
    # # link_df = get_link_geo(db_type=pg_db_type,
    # #                        drivertype=pg_driver_type,
    # #                        username=pg_username,
    # #                        password=pg_password,
    # #                        host=pg_host,
    # #                        port=pg_port,
    # #                        databasename=pg_databasename)
    # double_link_df = pd.read_csv(r'data/input/link_df.csv', encoding='utf_8_sig')
    #
    # # 处理单向表达的link和带几何信息的node
    # origin_single_link_df, node_gdf = \
    #     generate_gdf(link_df=double_link_df,
    #                  wkt_cols='geom',
    #                  from_node_field='from_node',
    #                  to_node_field='to_node')

    # 分天分时段
    for day in day_list:
        for time_period in time_period_dict.keys():


            for poi_item in poi_loc_list:



                # 时间属性的单位是秒, 计算道路可达性
                tim_level_gdf = main_road_atos(link_df=poi_link_df,
                                               node_gdf=node_gdf,
                                               grid_gdf=grid_gdf,
                                               origin_node_id=origin_node,
                                               sample_rate=0.05,
                                               cost_weight='total_time',
                                               fig_label='road_tim',
                                               time_interval=5)

    # # 获得公交的路径规划数据
    # pt_od_cost = get_pt_cost(grid_gdf=grid_gdf,
    #                          sample_rate=0.015,
    #                          origin_loc=None,
    #                          key=None,
    #                          city=None,
    #                          walk_speed=1.5)
    #
    # if isinstance(tim_level_gdf, int):
    #     print('拟合失败!请更改抽样率!')
    # else:
    #     cols_list = list(tim_level_gdf.columns)
    #     cols_list.remove('geometry')
    #     cols_list.append('geometry')
    #     tim_level_gdf[cols_list].to_file(os.path.join(output_fldr, 'res.geojson'), driver='GeoJSON')
    #     tim_level_gdf[cols_list].to_file(os.path.join(output_fldr, 'res.shp'), encoding='gbk')


