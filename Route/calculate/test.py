import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

# 读取CSV数据
file_path = 'E:/RoadFindWeb/Route/uploaded_files/K线中桩高程.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 处理数据，只取前200行
df.columns = ['Point', 'Y']
df['X'] = df['Point'].apply(lambda x: int(x.replace('K', '').replace('+', '')))  # 提取x坐标并去掉K和+号
df['Y'] = df['Y'].astype(float)  # 将y坐标转换为浮点数

x = df['X'].values
y = df['Y'].values

# 计算二阶商差
def second_order_difference(y):
    return np.array([y[i+2] - 2*y[i+1] + y[i] for i in range(len(y)-2)])

def apply_filter(y, filter_type='moving_average', **kwargs):
    if filter_type == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        return moving_average(y, window_size)
    elif filter_type == 'gaussian':
        sigma = kwargs.get('sigma', 2)
        return gaussian_filter1d(y, sigma)
    elif filter_type == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 2)
        return savgol_filter(y, window_length, polyorder)
    else:
        raise ValueError("Invalid filter_type. Choose 'moving_average', 'gaussian', or 'savgol'.")

def moving_average(y, window_size):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

# 设置滤波方式
filter_type = 'savgol'  # 可以设置为 'moving_average', 'gaussian' 或 'savgol'

# 应用滤波
if filter_type == 'moving_average':
    window_size = 5
    y_smooth = apply_filter(y, filter_type, window_size=window_size)
    x_smooth = x[(window_size-1)//2:-(window_size-1)//2]
elif filter_type == 'gaussian':
    sigma = 2
    y_smooth = apply_filter(y, filter_type, sigma=sigma)
    x_smooth = x
elif filter_type == 'savgol':
    window_length = 11
    polyorder = 2
    y_smooth = apply_filter(y, filter_type, window_length=window_length, polyorder=polyorder)
    x_smooth = x

# 计算二阶商差
second_order_diff = second_order_difference(y_smooth)
x_diff = x_smooth[1:-1]

# 配置 Matplotlib 以支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 选择一个支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# 创建图形和子图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制原始数据点图
color = 'tab:gray'
ax1.scatter(x, y, color=color, label='原始数据点')
#ax1.plot(x_smooth, y_smooth, color='tab:green', label='拟合数据')
ax1.set_xlabel('X')
ax1.set_ylabel('Y', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('数据点和滤波数据图')
ax1.grid(True)

# 创建第二个 y 轴
ax2 = ax1.twinx()
color = 'tab:red'
ax2.plot(x_diff, second_order_diff, color=color, marker='o', linestyle='-', label='二阶商差')

# 标记二阶商差值在 -0.01 到 0.01 之间的点
mask = (second_order_diff >= -0.008) & (second_order_diff <= 0.008)
ax2.plot(x_diff[mask], second_order_diff[mask], 'ko', label='直线段对应的二阶商差')  # 'ko' 表示黑色圆点

# 提取直线段
def extract_segments(x, mask):
    segments = []
    start = None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i-1))
                start = None
    if start is not None:
        segments.append((start, len(mask)-1))
    return segments

segments = extract_segments(x_diff, mask)

# 拟合每一个直线段的原始数据点
for start, end in segments:
    if start != end:  # 确保不是单个数据点
        segment_indices = np.arange(start, end+1) + 1  # +1 是因为二阶商差缩短了数据的长度
        segment_x = x[segment_indices]
        segment_y = y[segment_indices]
        
        # 线性拟合
        reg = LinearRegression().fit(segment_x.reshape(-1, 1), segment_y)
        fit_line = reg.predict(segment_x.reshape(-1, 1))
        
        # 绘制拟合直线
        ax1.plot(segment_x, fit_line, linestyle='--', label=f'拟合直线段 {start}-{end}')

# 添加图例，放置在图框外侧
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.8))

# 调整布局
fig.tight_layout(rect=[0, 0, 0.8, 1])  # 留出右侧空间以显示图例
plt.show()
