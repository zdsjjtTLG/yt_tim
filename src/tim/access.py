import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import math
from pykrige.ok import OrdinaryKriging
from shapely.geometry import LineString
from shapely.ops import unary_union
from geopy.distance import distance
from shapely.geometry import Polygon


# 线层数据、点层数据必需字段
length_field = 'length'  # 线层的长度, km
direction_field = 'dir'  # 线层的方向, 0, 1, -1
link_id_field = 'link_id'  # 线层的id
from_node_id_field = 'from_node'  # 线层的拓扑起始结点
to_node_id_field = 'to_node'  # 线层的拓扑终到结点
node_id_field = 'node_id'  # 点层的id
geometry_field = 'geometry'  # 几何属性字段
lon_field = 'lon'  # 结点经度
lat_field = 'lat'  # 结点纬度
required_link_filed_list = \
    [link_id_field, from_node_id_field, to_node_id_field, length_field, direction_field, geometry_field]
required_node_filed_list = [node_id_field, geometry_field]


def road_access_connectivity(node=None, link=None, grid_gdf=None, origin_node=None, sample_rate=None, pre_model=None,
                             cost_weight=None, fig_label=None, time_interval=5, export_field_list=None):
    """
    根据预测得到的数据，绘制等时线图
    :param node: gpd.GeoDataFrame, 点层文件
    :param link: gpd.GeoDataFrame, 线层文件
    :param grid_gdf: gpd.GeoDataFrame, 网格文件
    :param origin_node: int, 起始点的id
    :param sample_rate: float, 预测抽样率
    :param pre_model: str, 预测模型
    :param cost_weight:  str, 开销权重字段
    :param fig_label:  str, 标注的名称
    :param time_interval:  int, 时间间隔(分钟)
    :param export_field_list:
    :return:
    """

    # 创建用于搜路的网络, 将fft字段作为权重字段
    search_net = create_net(link_gdf=link, node_gdf=node, link_weight_col_list=[cost_weight] + export_field_list)

    # 计算指定起点到其他所有节点的最短路
    multi_od_df = get_multi_od_cost(net=search_net, origin_node_id=origin_node, weight_name=cost_weight,
                                    export_field_list=export_field_list)
    multi_od_df[cost_weight] = multi_od_df[cost_weight]

    # 指定预测模型函数，预测权重
    predict_res = pyk_predict(data_train=multi_od_df, data_test=grid_gdf,
                              predict_model=pre_model, cost_field_name=cost_weight, sample_rate=sample_rate)

    tim_level_gdf = plot_time_map(od_cost_df=predict_res, time_interval=time_interval, time_unit='m',
                                  cost_field_name=cost_weight)
    test_level_df = plot_time_map(od_cost_df=multi_od_df, time_interval=time_interval, time_unit='m',
                                  cost_field_name=cost_weight)
    return tim_level_gdf, test_level_df

    # # 转为矢量图层
    # res_shp = res_to_shp(level_data=tim_level_gdf, tim_level_col='tim_level')
    # res_shp.plot(column='tim_level', legend=True,
    #              legend_kwds={'label': fig_label, 'orientation': "horizontal"},
    #              cmap='rainbow')
    # plt.show()


# 建立路网功能主函数
def create_net(link_gdf=None, node_gdf=None, link_weight_col_list=None):
    """
    根据路网线层数据和点层数据创建路网
    :param link_gdf: gpd.GeoDatFrame, 线层数据
    :param node_gdf: gpd.GeoDatFrame, 点层数据
    :param link_weight_col_list: list, 选取哪些字段作为权重指标
    :return: pd.DataFrame
    """

    # 转化为单向路网后创建边列表
    net_df = link_gdf

    export_field_list = [from_node_id_field, to_node_id_field] + link_weight_col_list
    net_df = net_df[export_field_list]
    edge_list = get_edge_list(df=net_df, from_node_field=from_node_id_field, to_node_field=to_node_id_field,
                              weight_field_list=link_weight_col_list)

    # 创建节点列表
    node_gdf[lon_field] = node_gdf[geometry_field].x
    node_gdf[lat_field] = node_gdf[geometry_field].y
    node_list = get_node_list(df=node_gdf, node_id_col=node_id_field, attr_field_list=[lon_field, lat_field])

    # 创建网络
    di_graph = nx.DiGraph()
    di_graph.add_nodes_from(node_list)
    di_graph.add_edges_from(edge_list)

    return di_graph


# 逻辑子模块: 生成边列表用于创建图
def get_edge_list(df=None, from_node_field=None, to_node_field=None, weight_field_list=None):
    """
    生成边列表用于创建图
    :param df: pd.DataFrame, 路网数据
    :param from_node_field: str, 起始节点字段名称
    :param to_node_field: str, 终到节点字段名称
    :param weight_field_list: list, 代表边权重的字段列表名称
    :return: edge_list
    """

    # 起终点
    from_list = [from_node for from_node in df[from_node_field].to_list()]
    to_list = [to_node for to_node in df[to_node_field].to_list()]

    if weight_field_list is not None:
        # 这一步非常重要, 保证迭代的顺序是按照用户传入的列顺序
        weight_data = df[weight_field_list].copy()

        # 获取权重字典
        weight_list = [list(item) for item in weight_data.itertuples(index=False)]

        # 边列表
        edge_list = [(from_node, to_node, dict(zip(weight_field_list, data)))
                     for from_node, to_node, data in zip(from_list, to_list, weight_list)]
    else:
        # 边列表
        edge_list = [(from_node, to_node) for from_node, to_node in zip(from_list, to_list)]

    return edge_list


# 逻辑子模块: 生成边节点用于创建图
def get_node_list(df=None, node_id_col=None, attr_field_list=None):
    """
    生成节点列表用于创建图
    :param df: pd.DataFrame, 路网节点数据
    :param node_id_col: str, 节点id字段名称
    :param attr_field_list: list, 代表边权重的字段列表名称
    :return: node_list
    """

    # 节点ID
    node_id_list = [node_id for node_id in df[node_id_col].to_list()]

    if attr_field_list is not None:
        # 这一步非常重要, 保证迭代的顺序是按照用户传入的列顺序
        attr_data = df[attr_field_list].copy()

        # 获取权重字典
        attr_list = [list(item) for item in attr_data.itertuples(index=False)]

        # 边列表
        node_list = [(node_id, dict(zip(attr_field_list, data)))
                     for node_id, data in zip(node_id_list, attr_list)]
    else:
        # 节点列表
        node_list = [(node_id) for node_id in node_id_list]

    return node_list


# 获取多od对的最短路开销主函数
def get_multi_od_cost(net=None, weight_name=None, origin_node_id=None, export_field_list=None):
    """指定起点结点, 计算其到其他所有结点的最短路开销, 使用epsg:4326
    :param net: nx.network, 路网
    :param weight_name: str, 使用哪个权重指标进行搜录
    :param origin_node_id: int, 指定起点的结点id
    :param export_field_list: list, 同时输出哪些属性字段
    :return: pd.DataFrame
    """

    # 以指定起点结点进行搜路
    cost_dict, path_dict = nx.multi_source_dijkstra(net, {origin_node_id}, weight=weight_name)
    # 获得路径开销, 'node_id', {weight_name}

    cost_df = pd.DataFrame(cost_dict, index=[weight_name]).T
    cost_df.reset_index(inplace=True, drop=False)
    cost_df.rename(columns={'index': node_id_field}, inplace=True)
    print(cost_df)

    # 从节点中取出属性
    node_df = pd.DataFrame(net.nodes.data())
    node_df[lon_field] = node_df[1].apply(lambda x: x[lon_field])
    node_df[lat_field] = node_df[1].apply(lambda x: x[lat_field])
    node_df.drop(columns=1, inplace=True, axis=1)
    node_df.rename(columns={0: node_id_field}, inplace=True)

    res_df = pd.merge(cost_df, node_df, on=node_id_field, how='left')
    res_df.sort_values(by='total_time', ascending=True, inplace=True)

    # 找出在以weight_name为搜路权重下其他属性的累加值是多少
    df_list = []
    for edge in list(net.edges):
        df_list.append([edge] + [net[edge[0]][edge[1]][export_field] for export_field in export_field_list])
    edge_other_cost_df = pd.DataFrame(df_list, columns=['edge'] + export_field_list)
    print(edge_other_cost_df)

    path_df = pd.DataFrame([to_node, path_dict[to_node]] for to_node in path_dict.keys())
    path_df.columns = [node_id_field, 'path']
    path_df['path'] = path_df['path'].apply(lambda x:
                                            [(x[i], x[i + 1]) for i in range(0, len(x) - 1)]
                                            if len(x) > 1
                                            else [(-99)])
    path_df = path_df.explode(column=['path'], ignore_index=True)

    other_cost_df = pd.merge(path_df, edge_other_cost_df, left_on='path', right_on='edge', how='left')
    other_cost_df = other_cost_df.groupby(node_id_field).agg({col: 'sum' for col in export_field_list}).reset_index()

    res_df = pd.merge(res_df, other_cost_df, on=node_id_field, how='left')
    print(res_df)

    return res_df


# 获取栅格预测点
def get_grid_data(polygon_gdf=None, meter_step=None):
    """
    切分面域，得到面域上结点的经纬度坐标
    :param polygon_gdf: gdf.GeoDataFrame, 面域数据
    :param meter_step: int, 栅格区域大小, m
    :return: pd.Dataframe
    """

    geo_list = polygon_gdf[geometry_field].to_list()
    polygon_obj = unary_union(geo_list)

    # 根据栅格区域大小对面域进行栅格划分
    grid_gdf = generate_mesh(polygon_obj=polygon_obj, meter_step=meter_step)

    # 获取每个栅格中心点坐标
    grid_gdf[lon_field] = grid_gdf[geometry_field].apply(lambda x: x.centroid.x)
    grid_gdf[lat_field] = grid_gdf[geometry_field].apply(lambda x: x.centroid.y)

    return grid_gdf[['dx', 'dy', lon_field, lat_field]]


# 逻辑子模块：生成栅格用于获取预测点
def generate_mesh(polygon_obj=None, meter_step=100):
    """
    生成栅格用于获取预测点
    :param polygon_obj: gdf.GeoDataFrame, 面域数据
    :param meter_step: int, 栅格大小
    :return: gdf.GeoDataFrame
    """

    (min_x, min_y, max_x, max_y) = polygon_obj.bounds

    cen_x = polygon_obj.centroid.x
    cen_y = polygon_obj.centroid.y

    # 计算区域的长宽
    _width = max_y - min_y
    _length = max_x - min_x

    # 根据区域的中心点确定经纬度步长
    lon_step = get_geo_step(lon=cen_x, lat=cen_y, direction=1, step=meter_step)
    lat_step = get_geo_step(lon=cen_x, lat=cen_y, direction=0, step=meter_step)

    # 计算长宽多少个格子
    width_n = math.ceil(_width / lat_step)
    length_n = math.ceil(_length / lon_step)
    all_grid_list = []

    for n in range(width_n):
        point_list = [(min_x + k * lon_step, max_y - n * lat_step) for k in range(length_n)]

        def generate(xy):
            return Polygon([(xy[0], xy[1]), (xy[0] + lon_step, xy[1]),
                            (xy[0] + lon_step, xy[1] - lat_step), (xy[0], xy[1] - lat_step)])

        grid_list = list(map(generate, point_list))
        all_grid_list += grid_list

    index_list = [[i, j] for i in range(width_n) for j in range(length_n)]

    grid_gdf = gpd.GeoDataFrame({'mat_index': index_list}, geometry=all_grid_list, crs='EPSG:4326')

    # dx代表行索引, dy代表列索引
    grid_gdf['dx'] = grid_gdf['mat_index'].apply(lambda x: x[0])
    grid_gdf['dy'] = grid_gdf['mat_index'].apply(lambda x: x[1])
    grid_gdf.drop(columns='mat_index', axis=1, inplace=True)

    grid_gdf['bool'] = grid_gdf[geometry_field].apply(lambda x: x.intersects(polygon_obj))
    grid_gdf.drop(grid_gdf[grid_gdf['bool'] == False].index, axis=0, inplace=True)
    grid_gdf.reset_index(inplace=True, drop=True)
    grid_gdf.drop(columns='bool', inplace=True, axis=1)
    # res_grid_gdf = gpd.overlay(df1=polygon_gdf, df2=grid_gdf, how='intersection', keep_geom_type=True)
    return grid_gdf


# 逻辑子模块：确定经纬度步长
def get_geo_step(lon=None, lat=None, direction=1, step=100):
    """
    根据区域中心点确定经纬度步长
    :param lon: float, 经度
    :param lat: float, 纬度
    :param direction: int, 方向
    :param step: int, 步长
    :return:
    """

    if direction == 1:
        new_lon = lon + 0.1
        dis = distance((lat, lon), (lat, new_lon)).m
        return 0.1 / (dis / step)
    else:
        new_lat = lat + 0.1
        dis = distance((lat, lon), (new_lat, lon)).m
        return 0.1 / (dis / step)


# 使用克里金插值预测行程开销
def pyk_predict(data_train=None, cost_field_name=None, data_test=None, predict_model='linear', nlags=6, sample_rate=0.2):
    """
    根据已知点的坐标及行程开销，对预测点的行程开销进行预测
    :param data_train: pd.DataFrame, 已知点的坐标及行程开销
    :param cost_field_name: 训练数据中的值字段名称
    :param data_test: pd.DataFrame, 预测点的坐标
    :param predict_model: str, 选用的预测模型函数
    :param nlags: int,
    :param sample_rate: float
    :return: pd.DataFrame
    """
    print(f'抽样数量为{int(len(data_train) * sample_rate)}, 抽样率为{sample_rate * 100}%, 使用 {predict_model} 模型')

    # 抽样
    data_train = data_train.sample(int(len(data_train) * sample_rate), random_state=1, axis=0)

    data_test[lon_field] = data_test[geometry_field].apply(lambda x: x.centroid.x)
    data_test[lat_field] = data_test[geometry_field].apply(lambda x: x.centroid.y)

    # 获取已知点坐标和行程开销所在列、预测点坐标所在列
    lon_col = data_train.columns.get_loc(lon_field)
    lat_col = data_train.columns.get_loc(lat_field)
    cost_col = data_train.columns.get_loc(cost_field_name)

    pre_lon_col = data_test.columns.get_loc(lon_field)
    pre_lat_col = data_test.columns.get_loc(lat_field)
    train_data = np.array(data_train)
    test_data = np.array(data_test)

    # 根据已知点的数据进行训练
    OK = OrdinaryKriging(
        train_data[:, lon_col],
        train_data[:, lat_col],
        train_data[:, cost_col],
        variogram_model=predict_model,
        nlags=nlags,
        verbose=False,
        enable_plotting=False,
        coordinates_type="geographic",
    )

    # 计算kriged网格和相关的方差，得到预测结果
    z1, ss1 = OK.execute("points",
                         test_data[:, pre_lon_col].astype(np.float64),
                         test_data[:, pre_lat_col].astype(np.float64)
                         )
    predict_weight = pd.DataFrame({cost_field_name: z1})
    predict_res = pd.concat([data_test, predict_weight], axis=1)
    return predict_res


# 计算行程开销水平
def plot_time_map(od_cost_df=None, time_interval=15, time_unit='s', cost_field_name=None):
    """
    根据预测得到的数据, 计算行程开销水平
    :param od_cost_df: pd.DataFrame, od行程开销数据(时间列需要为秒)
    :param time_interval, int, 时间间隔
    :param time_unit: str, 时间间隔的单位, 秒's', 时'h', 分钟'm'
    :param cost_field_name, 开销字段的名称
    :return: gpd.GeoDataFrame
    """

    # 将间隔单位换算为秒
    if time_unit == 's':
        timing = time_interval
    elif time_unit == 'h':
        timing = time_interval * 3600
    elif time_unit == 'm':
        timing = time_interval * 60
    else:
        raise ValueError(f'no such a value: \'{time_unit}\' for parameter \'time_unit\'!')

    # 计算行程开销水平
    od_cost_df['tim_level'] = od_cost_df[cost_field_name].apply(lambda x: math.ceil(x / timing))

    return od_cost_df


# 逻辑子模块: 将具有方向字段的路网格式转化为单向的路网格式
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

    return net


# # 获取轮廓
# def res_to_shp(level_data=None, tim_level_col=None):
#     """
#     获取不同行程开销水平的轮廓
#     :param level_data: gpd.GeoDataFrame, od行程开销水平数据
#     :param tim_level_col, str, 行程开销水平字段
#     :return: gdf.GeoDataFrame
#     """
#
#     lon_data = level_data[['lon', 'dy']].sort_values(by="lon", ascending=True)  # lon列按升序排序
#     lat_data = level_data[['lat', 'dx']].sort_values(by="lat", ascending=False)  # lat列按降序排序
#
#     lon_length = lon_data[lon_data.groupby(['dy'])['lon'].rank(method="first", ascending=False) == 1]
#     lat_length = lat_data[lat_data.groupby(['dx'])['lat'].rank(method="first", ascending=False) == 1]
#
#     # 计算经纬度坐标区间大小diff及原点对应的经纬度坐标origin
#     lon_diff = round(abs(lon_length['lon'].diff().mean()), 6)  # 所有格跨越经纬度的均值
#     lat_diff = round(abs(lat_length['lat'].diff().mean()), 6)
#     origin_lon = round(lon_data['lon'].min(), 3)
#     origin_lat = round(lat_data['lat'].max(), 3)
#
#     # 构建xy矩阵
#     x_max = level_data['dx'].astype(int).max()
#     y_max = level_data['dy'].astype(int).max()
#     matrix = np.zeros([x_max + 3, y_max + 3])
#
#     contour_gdf = gpd.GeoDataFrame()
#     level_min = level_data[tim_level_col].min()
#     level_max = level_data[tim_level_col].max()
#
#     for level in range(level_min, level_max + 1):
#         level_data['__level_' + str(level)] = level_data[tim_level_col].apply(lambda x: True if x == level else False)
#
#         # 构建矩阵
#         x = np.array(level_data['dx'].astype(int))
#         y = np.array(level_data['dy'].astype(int))
#         z = np.array(level_data['__level_' + str(level)])
#         matrix[x, y] = z
#
#         # 检测图形的轮廓,获取边界点
#         contours = measure.find_contours(matrix, 0.05)
#         coord = pd.DataFrame()
#
#         # 同一level中可能有多个边界
#         for n, contour in enumerate(contours):
#             con = pd.DataFrame(contour)
#             # 计算轮廓点对应的经纬度坐标
#             con['lon'] = con[1].apply(lambda x: origin_lon + x * lon_diff)
#             con['lat'] = con[0].apply(lambda x: origin_lat - x * lat_diff)
#
#             con = con[['lon', 'lat']]
#             con['coord'] = con.apply(tuple, axis=1)
#             coord_list = con['coord'].tolist()
#             df_coord = gpd.GeoDataFrame([Polygon(coord_list)])
#
#             # 判断图形是否有包含关系
#             if n > 0:
#                 for polygon in range(0, len(coord)):
#                     poly1 = df_coord.iloc[0, 0]
#                     poly2 = coord.iloc[polygon, 0]
#                     poly = poly1.symmetric_difference(poly2)
#                     coord.iloc[polygon, 0] = poly
#             else:
#                 coord = coord.append(df_coord, ignore_index=True)
#         coord[tim_level_col] = level
#
#         contour_gdf = contour_gdf.append(coord, ignore_index=True)
#     contour_gdf = contour_gdf.rename(columns={0: 'geometry'})
#     contour_gdf.set_geometry(col='geometry', inplace=True)
#     return contour_gdf


if __name__ == '__main__':
    pass

    # # 读取线层文件, 点层文件, 面域文件
    # link_path = r'E:\tim\sz\shp\link.shp'
    # node_path = r'E:\tim\sz\shp\node.shp'
    # region_path = r'E:\tim\sz\shp\DISTRICT.shp'
    #
    # node_file = gpd.read_file(node_path, crs='EPSG:4326')
    # link_file = gpd.read_file(link_path, crs='EPSG:4326')
    # region_file = gpd.read_file(region_path, crs='EPSG:4326')
    #
    # # 指定起点id、预测模型的抽样率和模型、权重字段, 计算可达性并进行可视化
    # access_connectivity(node=node_file, link=link_file, city_gdf=region_file,
    #                     origin_node=1765, sample_rate=0.20, pre_model='linear',
    #                     cost_weight='fft', fig_label='transport connectivity in SZ')

    # link_gdf = gpd.read_file(r'D:\acess\data\input\link.shp', encoding='gbk')
    # print(link_gdf)
    # used_link = link_gdf.copy()
    #
    # used_link['from_p'] = used_link[geometry_field].apply(lambda x: Point(list(x.coords)[0]))
    # used_link['to_p'] = used_link[geometry_field].apply(lambda x: Point(list(x.coords)[-1]))
    #
    # f_node = used_link[[from_node_id_field, 'from_p']].copy()
    # t_node = used_link[[to_node_id_field, 'to_p']].copy()
    #
    # f_node.rename(columns={from_node_id_field: node_id_field, 'from_p': geometry_field}, inplace=True)
    # t_node.rename(columns={to_node_id_field: node_id_field, 'to_p': geometry_field}, inplace=True)
    #
    # all_node = pd.concat([f_node, t_node], axis=0)
    # all_node.reset_index(drop=True, inplace=True)
    # print(all_node)
    # all_node.drop_duplicates(subset=['node_id'], keep='first', inplace=True)
    # print(all_node)
    # all_node.reset_index(drop=True, inplace=True)
    # node_gdf = gpd.GeoDataFrame(all_node, geometry=geometry_field)
    # node_gdf.to_file(r'D:\acess\data\input\node.shp', encoding='gbk')





