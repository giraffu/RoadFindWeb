from .baseMethod import *
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fit_line(x, y):
    """拟合直线并计算误差"""
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    y_fit = m * x + c
    error = np.sum((y - y_fit) ** 2)
    return m, c, error

def fit_circle(x, y):
    """拟合圆弧并计算误差"""
    def calc_R(xc, yc):
        return np.sqrt((x - xc)**2 + (y - yc)**2)
    
    def func(center):
        xc, yc = center
        Ri = calc_R(xc, yc)
        return Ri - Ri.mean()

    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = (x_m, y_m)
    center, _ = curve_fit(func, center_estimate, np.zeros(len(x)))
    xc, yc = center
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    error = np.sum((Ri - R) ** 2)
    return xc, yc, R, error

def calculate_error(x, y):
    """计算误差矩阵"""
    n = len(x)
    line_err = np.full((n, n), float('inf'))
    circle_err = np.full((n, n), float('inf'))
    for i in range(n):
        for j in range(i+2, n):
            m, c, e_line = fit_line(x[i:j+1], y[i:j+1])
            line_err[i][j] = e_line
            xc, yc, R, e_circle = fit_circle(x[i:j+1], y[i:j+1])
            circle_err[i][j] = e_circle
    return line_err, circle_err

def find_optimal_segments(x, y, k):
    """动态规划寻找最优分段点，确保直线和圆弧交替出现"""
    n = len(x)
    line_err, circle_err = calculate_error(x, y)
    
    dp = np.full((k+1, n), float('inf'))
    segments = np.zeros((k+1, n), dtype=int)
    shapes = np.zeros((k+1, n), dtype=int) # 0表示直线，1表示圆弧
    
    dp[0][0] = 0
    
    for i in range(1, k+1):
        for j in range(1, n):
            for m in range(j):
                if i % 2 == 0:  # 偶数段为直线
                    if dp[i-1][m] + line_err[m+1][j] < dp[i][j]:
                        dp[i][j] = dp[i-1][m] + line_err[m+1][j]
                        segments[i][j] = m
                        shapes[i][j] = 0
                else:  # 奇数段为圆弧
                    if dp[i-1][m] + circle_err[m+1][j] < dp[i][j]:
                        dp[i][j] = dp[i-1][m] + circle_err[m+1][j]
                        segments[i][j] = m
                        shapes[i][j] = 1
    
    optimal_points = [n-1]
    shape_types = [shapes[k][n-1]]
    for i in range(k, 0, -1):
        optimal_points.append(segments[i][optimal_points[-1]])
        shape_types.append(shapes[i][optimal_points[-1]])
    
    optimal_points.reverse()
    shape_types.reverse()
    return optimal_points, shape_types

def Spline_fit_byXY_lon(x, y):
    """
    初次计算拟合斜率曲线
    
    参数:
    - file_path: LAS文件路径
    - method: 插值方法，可选值为 'cubic', 'univariate', 'pchip'，默认为 'cubic'
    
    返回:
    - 无，直接绘制插值曲线
    """
    # 归一化 x 和 y 坐标
    x_scaled, y_scaled, x_min, x_max, y_min, y_max = normalize(x, y)

    # 根据选择的方法进行样条插值
    f = interpolate(x_scaled, y_scaled)
    y_prime, y_second, y_third = compute_derivatives(f, x_scaled)
    # 设置打印选项以显示完整数组
    np.set_printoptions(threshold=np.inf)
    # 计算一阶导数之间的差值
    y_prime_diff = np.diff(y_prime)
    print("数据", y_prime_diff)
    
    zero_segments, non_zero_segments = find_zero_derivatives(x_scaled, y_scaled, y_prime, y_second, y_prime_diff, 1.1)
    

    initial_params = []  # 初始参数值
    param_names = []  # 参数名称
    for i, segment in enumerate(non_zero_segments):
        initial_params.append(100)  # 根据实际情况设置初始值
        param_names.append(f'第{i+1}段圆曲线拟合点数')
    
    zero_segments_data, non_zero_segments_data, fits_data, centers, consistent_curvature_indices = return_spline_fit_segments_lon(
        zero_segments, 
        non_zero_segments, 
        f, 
        x_scaled, x_min, x_max, y_min, y_max)
    
    # 对 zero_segments_data 进行反归一化
    zero_segments_data['x'] = [denormalize(np.array(x), np.array([0] * len(x)), x_min, x_max, y_min, y_max)[0].tolist() for x in zero_segments_data['x']]
    zero_segments_data['y'] = [denormalize(np.array([0] * len(y)), np.array(y), x_min, x_max, y_min, y_max)[1].tolist() for y in zero_segments_data['y']]
    
    # 对 non_zero_segments_data 进行反归一化
    non_zero_segments_data['x'] = [denormalize(np.array(x), np.array([0] * len(x)), x_min, x_max, y_min, y_max)[0].tolist() for x in non_zero_segments_data['x']]
    non_zero_segments_data['y'] = [denormalize(np.array([0] * len(y)), np.array(y), x_min, x_max, y_min, y_max)[1].tolist() for y in non_zero_segments_data['y']]

    # 分段数量
    k = 9

    # 找到最优分段点
    optimal_points, shape_types = find_optimal_segments(x, y, k)

    # 打印分段点
    print("Optimal Segments:", optimal_points)
    print("Shape Types (0: Line, 1: Circle):", shape_types)

    # 可视化结果
    plt.plot(x, y, "o", label="Data")
    for i in range(len(optimal_points) - 1):
        start = optimal_points[i]
        end = optimal_points[i+1]
        if shape_types[i] == 0:
            m, c, _ = fit_line(x[start:end+1], y[start:end+1])
            plt.plot(x[start:end+1], m * x[start:end+1] + c, label=f"Line Segment {i+1}")
        else:
            xc, yc, R, _ = fit_circle(x[start:end+1], y[start:end+1])
            theta = np.linspace(0, 2*np.pi, 100)
            x_circle = xc + R * np.cos(theta)
            y_circle = yc + R * np.sin(theta)
            plt.plot(x_circle, y_circle, label=f"Circle Segment {i+1}")
    plt.legend()
    plt.show()

    return zero_segments_data, non_zero_segments_data, initial_params, fits_data, centers, consistent_curvature_indices

def return_spline_fit_segments_lon(zero_segments, non_zero_segments, f, x_scaled, x_min, x_max, y_min, y_max):
    """
    初次计算样条拟合的各段和拟合圆
    
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

    # 合并两个列表并添加标记
    combined_segments = [(segment, 'zero') for segment in zero_segments] + \
                        [(segment, 'non_zero') for segment in non_zero_segments]
    # 按照段的开始索引排序
    combined_segments.sort(key=lambda x: x[0][0])

    # 计算拟合曲线的数据
    x_fits, y_fits, tolerances, centers = non_zero_segments_calculator(x_scaled, f, combined_segments, None, None, x_min, x_max, y_min, y_max, type="基本对称", type1="无缓和曲线")
    fits_data = {'x': [x.tolist() for x in x_fits], 'y': [y.tolist() for y in y_fits]}

    return zero_segments_data, non_zero_segments_data, fits_data, centers, tolerances