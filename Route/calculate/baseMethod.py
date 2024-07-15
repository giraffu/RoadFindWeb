# 导入必要的库
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline, PchipInterpolator
import re
from scipy.optimize import leastsq
from openpyxl import load_workbook
import os
# 插值函数参数
file_path ='data//施工图-3Y-路线逐桩坐标表.xlsx'
method = 'cubic'
Univariate_s = 0.05
isDrawRadius = False

def interpolate(x, y):
    """
    对给定的 x 和 y 数据进行插值。

    参数:
    x (list or np.ndarray): 自变量数据。
    y (list or np.ndarray): 因变量数据。

    返回:
    function: 返回插值函数，可以用于计算任意 x 值对应的 y 值。
    """
    # 将 x 和 y 转换为 numpy 数组
    x = np.array(x)
    y = np.array(y)

    # 检查 x 是否是严格递减的，如果是则翻转 x 和 y
    if np.all(np.diff(x) < 0):
        x = x[::-1]
        y = y[::-1]

    # 对 x 和 y 进行排序
    sorted_indices = np.argsort(x)  # 获取排序索引
    x_sorted = x[sorted_indices]    # 根据索引对 x 进行排序
    y_sorted = y[sorted_indices]    # 根据索引对 y 进行排序

    # 根据选择的插值方法进行插值
    if method == 'cubic':
        # 使用 Cubic Spline 方法进行插值
        f = CubicSpline(x_sorted, y_sorted)
    elif method == 'univariate':
        # 使用 Univariate Spline 方法进行插值
        f = UnivariateSpline(x_sorted, y_sorted, s=Univariate_s)
    elif method == 'pchip':
        # 使用 PCHIP 方法进行插值
        f = PchipInterpolator(x_sorted, y_sorted)
    else:
        # 如果指定的方法无效，抛出错误
        raise ValueError("Invalid method specified")
    
    # 返回插值函数
    return f

def calc_R(xc, yc, x, y):
    """
    计算每个点到圆心 (xc, yc) 的距离。

    参数:
    xc (float): 圆心的 x 坐标。
    yc (float): 圆心的 y 坐标。
    x (numpy.ndarray): 所有点的 x 坐标。
    y (numpy.ndarray): 所有点的 y 坐标。

    返回:
    numpy.ndarray: 包含每个点到圆心距离的数组。
    """
    return np.sqrt((x - xc)**2 + (y - yc)**2)

def f_2(c, x, y):
    """
    计算每个点到圆心距离的偏差。

    参数:
    c (tuple): 包含圆心的坐标 (xc, yc)。
    x (numpy.ndarray): 所有点的 x 坐标。
    y (numpy.ndarray): 所有点的 y 坐标。

    返回:
    numpy.ndarray: 每个点到圆心距离的偏差。
    """
    Ri = calc_R(c[0], c[1], x, y)
    return Ri - Ri.mean()

def normal_method(x, y):
    """
    对输入的点集进行归一化处理，将 x 和 y 值缩放到 [0, 1] 的范围内。

    参数:
    x (numpy.ndarray): 所有点的 x 坐标。
    y (numpy.ndarray): 所有点的 y 坐标。

    返回:
    numpy.ndarray, numpy.ndarray: 缩放后的 x 坐标数组和 y 坐标数组。
    """
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    
    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)
    
    return x_scaled, y_scaled

def zero_segments_calculator(x, f, zero_segments, A=10):
    """
    计算零导数段的样条曲线段并返回。

    参数：
    x (numpy.ndarray): 自变量数组。
    f (callable): 函数或插值对象，接受 x 数组作为参数并返回对应的因变量数组。
    zero_segments (list): 包含元组 (start_index, end_index) 的列表，表示零导数段在 x 中的起始和结束索引。
    A (float, optional): 控制样条曲线绘制的参数，默认为 10。

    返回：
    tuple: 包含两个列表 x_segments 和 y_segments，分别表示每个零导数段的 x 和 y 值数组。
    """
    x_segments = []
    y_segments = []

    for segment in zero_segments:
        start_index, end_index = segment
        x_segment = x[start_index:end_index+1]
        y_segment = f(x_segment)
        x_segments.append(x_segment)
        y_segments.append(y_segment)

    return x_segments, y_segments

def non_zero_segments_calculator(x, f, non_zero_segments, A, B):
    """
    计算非零导数段的样条曲线，并进行圆拟合。

    参数：
    x (numpy.ndarray): 自变量数组。
    f (callable): 函数或插值对象，接受 x 数组作为参数并返回对应的因变量数组。
    non_zero_segments (list): 包含元组 (start_index, end_index) 的列表，表示非零导数段在 x 中的起始和结束索引。
    A (list): 控制每个段样条曲线的参数列表。
    B (list or None): 可选参数，用于指定每个段的圆拟合中心索引，如果为 None，则自动选择最大曲率处作为圆心。

    返回：
    tuple: 包含四个列表 x_fits, y_fits, errors, centers，分别表示每个非零导数段的圆拟合 x 值数组、y 值数组、拟合误差和圆心索引。
    """
    x_fits = []
    y_fits = []
    errors = []
    centers = []

    for i, segment in enumerate(non_zero_segments):
        start_index, end_index = segment
        x_segment = x[start_index:end_index+1]
        y_segment = f(x_segment)

        # 计算曲率半径
        curvature = calculate_curvature(f, x_segment)

        # 找到每个段内的最大曲率及其对应的索引
        if B is None:
            center = np.argmax(curvature)
        else:
            center = B[i]

        # 进行圆拟合
        x_fit_circle, y_fit_circle, circle_error = circle_fit(center, x_segment, y_segment, A[i])

        if x_fit_circle is not None and y_fit_circle is not None:
            x_fits.append(x_fit_circle)
            y_fits.append(y_fit_circle)
            errors.append(circle_error)
            centers.append(center)

    return x_fits, y_fits, errors, centers

def circle_fit(curvature_center, x, y, A):
    """
    对给定的数据点进行圆拟合。

    参数：
    curvature_center (int): 曲率中心的索引。
    x (numpy.ndarray): x 坐标数组。
    y (numpy.ndarray): y 坐标数组。
    A (int): 拟合范围半径。

    返回：
    tuple: 包含 x_fit_circle, y_fit_circle, fitting_error 的元组，分别表示拟合的圆的 x 坐标数组、y 坐标数组和拟合误差。

    注意：
    - 如果拟合点数量不足，将打印警告信息并返回 None。
    """
    # 提取拟合范围内的数据点
    fit_start = max(0, curvature_center - A)
    fit_end = min(len(x), curvature_center + A)
    x_fit = x[fit_start:fit_end]
    y_fit = y[fit_start:fit_end]

    # 检查拟合点的数量
    if len(x_fit) < 3:  # 至少需要三个点来拟合圆
        print('拟合点数量不足')
        return None, None, None

    # 初步估计圆心位置
    center_estimate = np.mean(x_fit), np.mean(y_fit)
    
    # 使用最小二乘法进行圆拟合
    center, ier = leastsq(f_2, center_estimate, args=(x_fit, y_fit))
    xc, yc = center
    
    # 计算拟合半径
    Ri = calc_R(xc, yc, x_fit, y_fit)
    R = Ri.mean()
    
    # 计算拟合误差
    fitting_error = np.sum((Ri - R)**2)
    
    # 根据拟合结果生成圆的坐标点
    theta_start = np.arctan2(y_fit[0] - yc, x_fit[0] - xc)
    theta_end = np.arctan2(y_fit[-1] - yc, x_fit[-1] - xc)
    theta_fit = np.linspace(theta_start, theta_end, 100)
    x_fit_circle = xc + R * np.cos(theta_fit)
    y_fit_circle = yc + R * np.sin(theta_fit)
    
    return x_fit_circle, y_fit_circle, fitting_error

def calculate_curvature(f, x):
    """
    计算给定样条曲线在指定 x 值处的曲率。

    参数：
    f (callable): 函数或插值对象，接受 x 数组作为参数并返回对应的因变量数组。
    x (numpy.ndarray): 自变量数组。

    返回：
    numpy.ndarray: 对应于 x 数组的曲率数组。

    注意：
    - 计算曲率公式使用一阶和二阶导数。
    """
    # 计算一阶和二阶导数
    f_prime = f.derivative()
    f_second = f_prime.derivative()
    y_prime_segment = f_prime(x)
    y_second_segment = f_second(x)
    
    # 计算曲率
    curvature = np.abs(y_second_segment) / (1 + y_prime_segment**2)**(3/2)
    
    return curvature

# find_zero_second_derivatives函数用于在给定的数据点 (x, y_dedrivative) 中查找二阶导数为零的段落和二阶导数不为零的段落。

def find_zero_second_derivatives(x, y_dedrivative, tolerance):
    """
    在数据点 (x, y_dedrivative) 中查找二阶导数为零的段落和二阶导数不为零的段落。

    Parameters:
    x : numpy.ndarray
        输入数据的 x 值数组。
    y_dedrivative : numpy.ndarray
        输入数据的二阶导数值数组。
    tolerance : float
        允许的二阶导数接近零的容差。

    Returns:
    tuple
        包含二阶导数为零段落和二阶导数不为零段落的元组。
        每个段落表示为 (start_index, end_index) 形式的元组。
    """
    # 找到二阶导数为零的段落
    zero_crossings = np.where(np.abs(y_dedrivative) < tolerance)[0]  # 找到二阶导数绝对值小于指定容差的索引
    zero_segments = np.split(zero_crossings, np.where(np.diff(zero_crossings) != 1)[0] + 1)  # 将连续的索引分割成段落
    zero_segments = [(segment[0], segment[-1]) for segment in zero_segments if len(segment) > 0]  # 记录每个段落的起始和结束索引

    # 找到二阶导数不为零的段落
    nonzero_segments = []
    start_index = 0
    for segment in zero_segments:
        end_index = segment[0]
        if start_index < end_index:
            nonzero_segments.append((start_index, end_index))  # 添加二阶导数不为零的段落的起始和结束索引
        start_index = segment[-1]
    if start_index < len(x):
        nonzero_segments.append((start_index, len(x)))  # 添加最后一个段落的起始和结束索引

    segments_to_remove_index = []  # 记录需要删除的段落索引

    # 处理重叠段落
    for i, nonzero_segment in enumerate(nonzero_segments):
        if nonzero_segment[1] - nonzero_segment[0] > 2:
            continue
        # 找到尾数等于nonzero_segment[0]的段落索引，将其尾数改为nonzero_segment[1]
        segment_index = next((i for i, segment in enumerate(zero_segments) if segment[1] == nonzero_segment[0]), None)
        if segment_index is not None:
            zero_segments[segment_index] = (zero_segments[segment_index][0], nonzero_segment[1])  # 更新重叠段落的结束索引
            segments_to_remove_index.append(i)  # 记录需要删除的段落索引

    # 删除需要删除的段落
    for index in sorted(segments_to_remove_index, reverse=True):
        del nonzero_segments[index]  # 删除二阶导数不为零的段落中的重叠部分

    return zero_segments, nonzero_segments  # 返回二阶导数为零和二阶导数不为零的段落列表

def compute_derivatives(f, x):
    """
    计算函数 f 在给定点 x 处的一阶、二阶和三阶导数值。

    Parameters:
    f : callable
        可调用的函数或插值对象，可以计算导数。
    x : numpy.ndarray
        输入数据的 x 值数组。

    Returns:
    tuple
        包含一阶、二阶和三阶导数值的元组。
    """
    f_prime = f.derivative()  # 计算一阶导数
    f_second = f_prime.derivative()  # 计算二阶导数
    f_third = f_second.derivative()  # 计算三阶导数
    y_prime = f_prime(x)  # 计算一阶导数值
    y_second = f_second(x)  # 计算二阶导数值
    y_third = f_third(x)  # 计算三阶导数值  
    return y_prime, y_second, y_third  # 返回一阶、二阶和三阶导数值的元组

def extract_data_from_excel(file_path):
    """
    从指定的 Excel 文件中提取数据。

    Parameters:
    file_path : str
        Excel 文件的路径。

    Returns:
    numpy.ndarray
        包含从 Excel 中提取的数据的 NumPy 数组，每行包括"桩号"、"桩号数字"、"X 值"和"Y 值"。
    """
    # 将相对路径转换为绝对路径
    abs_file_path = os.path.abspath(file_path)
    
    # 加载 Excel 文件
    wb = load_workbook(abs_file_path)
    ws = wb.active
    # 用于存储结果的列表
    data = []

    # 遍历工作表中的每一行
    for row in ws.iter_rows(min_row=1, max_col=ws.max_column, max_row=ws.max_row):
        for cell in row:
            # 检查单元格的值是否包含"k"
            if isinstance(cell.value, str) and "k" in cell.value.lower():
                # 提取当前单元格及其右侧两个单元格的值
                stake = cell.value
                stake_number = extract_number(stake)  # 匹配整个单词
    
                x_value = ws.cell(row=cell.row, column=cell.column + 1).value
                y_value = ws.cell(row=cell.row, column=cell.column + 2).value
                if x_value is not None and y_value is not None:
                    # 将数据添加到结果列表中
                    data.append([stake, stake_number, float(x_value), float(y_value)])
    
    # 关闭 Excel 文件
    wb.close()

    # 根据 stake_number 进行排序
    sorted_data = sorted(data, key=lambda x: x[1])
    
    return np.asarray(sorted_data)

def extract_number(text):
    """
    从给定的文本中提取并组合数字。

    参数:
    text (str): 包含数字的字符串。

    返回:
    float: 提取并组合后的浮点数。如果未找到匹配的数字，则返回 None。
    """
    # 使用正则表达式匹配两个部分的数字（例如 "123+45.67"）
    number_match = re.search(r'(\d+)\+(\d+\.?\d*)', text)
    
    if number_match:
        # 如果找到匹配的数字，提取第一部分和第二部分
        number_part1 = number_match.group(1)
        number_part2 = number_match.group(2)
        
        # 将两个部分组合成一个字符串
        combined_number = number_part1 + number_part2
        
        # 将组合后的字符串转换为浮点数并返回
        return float(combined_number)
    else:
        # 如果未找到匹配的数字，返回 None
        return None
    