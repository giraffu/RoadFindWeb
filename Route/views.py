'''
Author: chuzeyu 3343447088@qq.com
Date: 2024-07-02 10:16:30
LastEditors: chuzeyu 3343447088@qq.com
LastEditTime: 2024-07-15 11:32:49
FilePath: \RoadFindWeb\Route\views.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from .calculate.routeFitting import extract_data_from_excel, Spline_fit_byXY, Re_spline_fit_byXY
import json

global global_x_values, global_y_values
global_x_values = None
global_y_values = None

@csrf_exempt  # 禁用 CSRF 保护，仅用于开发环境，请在生产环境中小心使用
def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(base_dir, 'uploaded_files')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
        return JsonResponse({'file_path': file_path})
    else:
        return JsonResponse({'error': 'No file uploaded or invalid request method'}, status=400)

@csrf_exempt  # 同样禁用 CSRF 保护
def calculate_data(request):
    if request.method == 'POST':
        file_path = request.POST.get('file_path')
        if file_path:
            data = extract_data_from_excel(file_path)
            global global_x_values, global_y_values
            # 全局变量用于存储计算的长度和坐标数据
            global_x_values = data[:, 2].astype(float)
            global_y_values = data[:, 3].astype(float)
            zero_segments, non_zero_segments, initial_params, fits_data, centers = Spline_fit_byXY(
                global_x_values, 
                global_y_values)
            # 打印数据以调试
            print("zero_segments:", zero_segments)
            print("non_zero_segments:", non_zero_segments)
            print("initial_params:", initial_params)
            print("centers:", centers)
            centers = [int(center) for center in centers]
            return JsonResponse({
                'zero_segments': zero_segments,
                'non_zero_segments': non_zero_segments,
                'initial_params': initial_params,
                'fits': fits_data,
                'centers': centers
            })
        else:
            return JsonResponse({'error': 'Missing file_path'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt  # 禁用 CSRF 保护
def recalculate_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            centers = data.get('centers')
            lengths = data.get('lengths')
            file_path = data.get('file_path')

            print("Received centers:", centers)  # 调试信息
            print("Received lengths:", lengths)  # 调试信息
            print("global_x_values:", global_x_values)  # 调试信息
            zero_segments_data, non_zero_segments_data, initial_params, fits_data, centers = Re_spline_fit_byXY(
                global_x_values, 
                global_y_values, 
                new_params2=lengths,
                new_params1=centers)

            return JsonResponse({
                'zero_segments': zero_segments_data,
                'non_zero_segments': non_zero_segments_data,
                'initial_params': initial_params,
                'fits': fits_data,
                'centers': centers
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)