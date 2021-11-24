# -- coding: utf-8 --
# @Time    : 2021/11/23 0023 12:53
# @Author  : TangKai
# @Team    : SuperModel
# @File    : get_df.py
import pandas as pd
from src.conn.pg_connect import OperatePgSql
from src.conn.clickhouse_parser import ClickhouseParserCls


def get_df_from_pg(db_type='postgresql',
                   drivertype='psycopg2',
                   username='postgres',
                   password='123456',
                   host='10.10.5.108',
                   port='5432',
                   databasename='tpi_yt',
                   sql_str=None):

    pgsql = OperatePgSql(db_type, drivertype, username, password, host, port, databasename)
    engine = pgsql.get_con()

    df = pd.read_sql(sql_str, engine)

    return df


def get_df_from_clc(db_name='test', user_name='default', password='default', host='10.10.201.15',
                    sql_str=None, port='9000', cols_list=None):

    pg_cls = ClickhouseParserCls(host, port, db_name, user_name, password)

    data = pg_cls.executeSql(sql_str)
    return pd.DataFrame(data, columns=cols_list)


# 从数据库获取道路link
def get_link_geo(db_type=None,
                 drivertype=None,
                 username=None,
                 password=None,
                 host=None,
                 port=None,
                 databasename=None):

    # 从pg数据库读取小路段几何表
    select_link_geo_list = ['fid', 'geom', 'dir_type', 'length', 'road_type_fid', 'speed',
                            'ab_speed', 'ba_speed', 'type_name', 'from_node', 'to_node', 'district_fid',
                            'block_fid']

    select_link_geo_cols = ', '.join(select_link_geo_list)
    select_link_geo_cols = select_link_geo_cols.replace('geom', 'ST_AsText(geom) as geom')

    link_df = get_df_from_pg(db_type=db_type,
                             drivertype=drivertype,
                             username=username,
                             password=password,
                             host=host,
                             port=port,
                             databasename=databasename,
                             sql_str=f'select {select_link_geo_cols} from t_base_link_geom')
    return link_df


def get_day_time_period_link_time(db_name=None, user_name=None, password=None,
                                  host=None, port=None, day=None, time_slice=None):
    """

    :param db_name:
    :param user_name:
    :param password:
    :param host:
    :param port:
    :param day:
    :param time_slice: List[int], [起始时间片, 终点时间片]
    :return:
    """
    # 从clc数据库读取实时流量表
    # 'fdate', 'period', 'link_fid', 'from_node', 'to_node', 'speed', 'total_length', 'total_time'
    select_time_list = ['fdate', 'period', 'link_fid', 'from_node', 'to_node', 'total_time']
    select_time_cols = ', '.join(select_time_list)
    link_time = get_df_from_clc(db_name=db_name,
                                user_name=user_name,
                                password=password,
                                host=host,
                                port=port,
                                cols_list=select_time_list,
                                sql_str=f'select {select_time_cols} from t_tpi_link_speed where fdate = {int(day)} and (period between {time_slice[0]} and {time_slice[1]})')
    return link_time

if __name__ == '__main__':
    pass
    # db_type = 'postgresql'
    # drivertype = 'psycopg2'
    # username = 'postgres'
    # password = '123456'
    # host = '10.10.5.108'
    # port = '5432'
    # databasename = 'tpi_yt'

    # # 从数据库读取路段表
    # link_df = get_df_from_pg(db_type=db_type,
    #                          drivertype=drivertype,
    #                          username=username,
    #                          password=password,
    #                          host=host,
    #                          port=port,
    #                          databasename=databasename,
    #                          sql_str_list=["select * from t_base_link_geom"])
    # clc_db_name = 'tpi_sdyt'
    # clc_user_name = 'default'
    # clc_password = 'sutpc1234'
    # clc_host = '10.10.5.109'
    # clc_port = '9000'
    # col_list = ['fdate', 'period', 'link_fid', 'from_node', 'to_node', 'total_time']
    # col = ','.join(col_list)
    # print(get_df_from_clc(db_name=clc_db_name,
    #                       user_name=clc_user_name,
    #                       password=clc_password,
    #                       host=clc_host,
    #                       sql_str=f'select {col} from t_tpi_link_speed where fdate = 20210816 and (period between 1 and 2)',
    #                       port='9000',
    #                       cols_list=col_list))

