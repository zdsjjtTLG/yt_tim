# -- coding: utf-8 --
# @Time    : 2021/11/23 0023 16:41
# @Author  : TangKai
# @Team    : SuperModel
# @File    : process.py

import pandas as pd
import numpy as np

# 路况表中的link_id字段
time_link_id_field = 'link_fid'
total_time_field = 'total_time'

# 路段几何表中link_id字段
link_id_field = 'fid'
direction_field = 'dir_type'
block_id_field = 'block_fid'
ffs_field = 'speed'
fft_field = 'fft'
length_field = 'length'

from_node_id_field = 'from_node'
to_node_id_field = 'to_node'


def process_link_time(slice_link_time_df=None,
                      link_df=None):
    """
    将某一天的某时段(多个时间片)的路段实时流量数据按照时间片进行平均后匹配到link_gdf, 空值则使用
    同一街道下有值的 (实时速度 / 自由流速)比值的平均值 * 自由流速度
    :param slice_link_time_df: 某一天的某些时间片的路段实时行驶时间(单向)
    :param link_df: 单向路网(不带几何列)
    :return:
    """
    # 不改变原路网
    link_df = link_df.copy()

    # 单向表达的实时时间
    period_link_time_df = \
        slice_link_time_df.groupby([time_link_id_field,
                                    from_node_id_field,
                                    to_node_id_field])[total_time_field].mean().reset_index()

    # 单向表达的路网进行匹配
    link_df = pd.merge(link_df,
                       period_link_time_df[[from_node_id_field, to_node_id_field, total_time_field]],
                       on=[from_node_id_field, to_node_id_field], how='left')

    # 计算所有路段的自由流时间, 单位是秒
    link_df[fft_field] = 3.6 * link_df[length_field] / link_df[ffs_field]

    # 按街道统计 实际速度 / 自由流速度的比值(街道平均值)
    link_df['para'] = link_df.apply(lambda x:
                                    round(x[total_time_field] / x[fft_field], 2)
                                    if ~np.isnan(x[total_time_field])
                                    else np.nan, axis=1)
    link_df['avg_para'] = link_df.groupby(block_id_field)['para'].transform('mean')
    link_df['avg_para'] = link_df['avg_para'].fillna(1)


    # 计算
    link_df[total_time_field] = link_df.apply(lambda x:
                                              x[total_time_field]
                                              if ~np.isnan(x[total_time_field])
                                              else x['avg_para'] * x[fft_field], axis=1)

    return link_df[[from_node_id_field, to_node_id_field, total_time_field, length_field]]

