'''
Author: chuzeyu 3343447088@qq.com
Date: 2024-07-01 15:23:46
LastEditors: chuzeyu 3343447088@qq.com
LastEditTime: 2024-07-15 13:41:25
FilePath: \RoadFindWeb\Route\calculate\routeFitting copy.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from .baseMethod import *

def Spline_fit_byXY(x, y):
    """
    初次计算拟合斜率曲线
    
    参数:
    - file_path: LAS文件路径
    - method: 插值方法，可选值为 'cubic', 'univariate', 'pchip'，默认为 'cubic'
    
    返回:
    - 无，直接绘制插值曲线
    """
    # 归一化 x 和 y 坐标
    x_scaled, y_scaled = normal_method(x, y)
    # 根据选择的方法进行样条插值
    f = interpolate(x_scaled, y_scaled)
    y_prime, y_second, y_third = compute_derivatives(f, x_scaled)
    zero_segments, non_zero_segments = find_zero_second_derivatives(x_scaled, y_second, 1e-1)

    initial_params = []  # 初始参数值
    param_names = []  # 参数名称
    for i, segment in enumerate(non_zero_segments):
        initial_params.append(100)  # 根据实际情况设置初始值
        param_names.append(f'第{i+1}段圆曲线拟合点数')

    zero_segments_data, non_zero_segments_data, fits_data, centers = return_spline_fit_segments(
        zero_segments, 
        non_zero_segments, 
        f, 
        x_scaled)
    return zero_segments_data, non_zero_segments_data, initial_params, fits_data, centers

def Re_spline_fit_byXY(x, y, new_params2, new_params1):
    """
    重新计算拟合斜率曲线
    
    参数:
    - file_path: LAS文件路径
    - method: 插值方法，可选值为 'cubic', 'univariate', 'pchip'，默认为 'cubic'
    
    返回:
    - 无，直接绘制插值曲线
    """
    # 归一化 x 和 y 坐标
    x_scaled, y_scaled = normal_method(x, y)
    
    # 根据选择的方法进行样条插值
    f = interpolate(x_scaled, y_scaled)
    y_prime, y_second, y_third = compute_derivatives(f, x_scaled)
    zero_segments, non_zero_segments = find_zero_second_derivatives(x_scaled, y_second, 1e-1)

    initial_params = []  # 初始参数值
    param_names = []  # 参数名称
    for i, segment in enumerate(non_zero_segments):
        initial_params.append(100)  # 根据实际情况设置初始值
        param_names.append(f'第{i+1}段圆曲线拟合点数')
    print("传回数据",new_params2, new_params1)
    zero_segments_data, non_zero_segments_data, fits_data, centers = recal_spline_fit_segments(zero_segments, non_zero_segments, f, x_scaled, new_params2, new_params1)
    return zero_segments_data, non_zero_segments_data, initial_params, fits_data, centers

def return_spline_fit_segments(zero_segments, non_zero_segments, f, x_scaled):
    """
    绘制样条拟合的各段和拟合圆
    
    参数:
    - zero_segments: 零二阶导数的段的列表
    - non_zero_segments: 非零二阶导数的段的列表
    - f: 插值函数
    - x_scaled: 归一化的 x 坐标
    - y_scaled: 归一化的 y 坐标
    
    返回:
    - zero_segments_data: 红色段的 x 和 y 数据
    - non_zero_segments_data: 蓝色段的 x 和 y 数据
    - fits_data: 拟合曲线的 x 和 y 数据
    """
    # 计算零二阶导数段的数据
    x_segments, y_segments = zero_segments_calculator(x_scaled, f, zero_segments)
    zero_segments_data = {'x': [x.tolist() for x in x_segments], 'y': [y.tolist() for y in y_segments]}
    
    # 计算非零二阶导数段的数据
    non_zero_segments_data = {'x': [], 'y': []}
    for segment in non_zero_segments:
        start_index, end_index = segment
        x_segment = x_scaled[start_index:end_index+1]
        y_segment = f(x_segment)
        non_zero_segments_data['x'].append(x_segment.tolist())
        non_zero_segments_data['y'].append(y_segment.tolist())
    initial_params = [100] * len(non_zero_segments)
    # 计算拟合曲线的数据
    x_fits, y_fits, errors, centers = non_zero_segments_calculator(x_scaled, f, non_zero_segments, initial_params, None)
    fits_data = {'x': [x.tolist() for x in x_fits], 'y': [y.tolist() for y in y_fits]}

    return zero_segments_data, non_zero_segments_data, fits_data, centers

def recal_spline_fit_segments(zero_segments, non_zero_segments, f, x_scaled, new_params2, new_params1):
    """
    绘制样条拟合的各段和拟合圆
    
    参数:
    - zero_segments: 零二阶导数的段的列表
    - non_zero_segments: 非零二阶导数的段的列表
    - f: 插值函数
    - x_scaled: 归一化的 x 坐标
    - new_params2: 圆曲线长度参数
    - new_params1: 圆曲线位置参数
    
    返回:
    - zero_segments_data: 红色段的 x 和 y 数据
    - non_zero_segments_data: 蓝色段的 x 和 y 数据
    - fits_data: 拟合曲线的 x 和 y 数据
    """
    # 计算零二阶导数段的数据
    x_segments, y_segments = zero_segments_calculator(x_scaled, f, zero_segments)
    zero_segments_data = {'x': [x.tolist() for x in x_segments], 'y': [y.tolist() for y in y_segments]}
    
    # 计算非零二阶导数段的数据
    non_zero_segments_data = {'x': [], 'y': []}
    for segment in non_zero_segments:
        start_index, end_index = segment
        x_segment = x_scaled[start_index:end_index+1]
        y_segment = f(x_segment)
        non_zero_segments_data['x'].append(x_segment.tolist())
        non_zero_segments_data['y'].append(y_segment.tolist())

    x_fits, y_fits, errors, centers = non_zero_segments_calculator(x_scaled, f, non_zero_segments, new_params2, new_params1)

    fits_data = {'x': [x.tolist() for x in x_fits], 'y': [y.tolist() for y in y_fits]}

    return zero_segments_data, non_zero_segments_data, fits_data, centers
