from src.tim.access import road_access_connectivity
import geopandas as gpd
import matplotlib.pyplot as plt

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


def main_road_atos(link_df=None, node_gdf=None, grid_gdf=None,
                   sample_rate=0.05, origin_node_id=None,
                   cost_weight='fft', fig_label='road_tim', time_interval=5):

    pre_model_list = ['linear', 'power', 'exponential', 'gaussian']

    # 关键字段确保是int
    link_df[from_node_id_field] = link_df[from_node_id_field].astype(int)
    link_df[to_node_id_field] = link_df[to_node_id_field].astype(int)
    node_gdf[node_id_field] = node_gdf[node_id_field].astype(int)

    # 计算栅格的时间等级, 依次使用不同的模型
    for pre_model in pre_model_list:
        tim_level_gdf, test_level_df = road_access_connectivity(node=node_gdf, link=link_df, grid_gdf=grid_gdf,
                                                                origin_node=origin_node_id, sample_rate=sample_rate,
                                                                pre_model=pre_model,
                                                                cost_weight='total_time', fig_label=fig_label,
                                                                export_field_list=['length'],
                                                                time_interval=time_interval)

        # 如果训练数据的tim等级数目大于预测数据的tim等级数目, 很可能出现了过拟合
        if len(tim_level_gdf['tim_level'].unique()) < 0.5 * len(test_level_df['tim_level'].unique()):
            tim_level_gdf.plot(column='tim_level', legend=True,
                               legend_kwds={'label': fig_label, 'orientation': "horizontal"},
                               cmap='rainbow')
            # plt.show()
            print(f'使用了{pre_model}拟合失败!')
        else:
            tim_level_gdf.plot(column='tim_level', legend=True,
                               legend_kwds={'label': fig_label, 'orientation': "horizontal"},
                               cmap='rainbow')
            plt.show()

            print(f'使用了{pre_model}拟合成功!')
            return tim_level_gdf

    return -1

