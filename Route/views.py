from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileUploadSerializer, FilePathSerializer, DataSerializer
from .calculate.routeFitting import extract_data_from_excel, Spline_fit_byXY, Re_spline_fit_byXY, extract_data_from_excel_lon
import os
import json
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.parsers import MultiPartParser, FormParser
class UploadFile(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        operation_description="上传逐桩坐标表文件(用于平面拟合计算)",
        manual_parameters=[
            openapi.Parameter(
                'file',
                openapi.IN_FORM,
                description="上传的文件",
                type=openapi.TYPE_FILE,
                required=True
            ),
        ],
        tags=['平面计算'],
        responses={200: 'Success', 400: 'Bad Request'}
    )
    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            base_dir = os.path.dirname(os.path.abspath(__file__))
            upload_dir = os.path.join(base_dir, 'uploaded_files')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            return Response({'file_path': file_path})
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CalculateData(APIView):
    @swagger_auto_schema(
        operation_description="由逐桩坐标表拟合平面线元参数",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'file_path': openapi.Schema(type=openapi.TYPE_STRING, description='后端服务器上的excel文件的路径'),
            }
        ),
        tags=['平面计算'],
        responses={200: 'Success', 400: 'Bad Request'}
    )
    def post(self, request):
        serializer = FilePathSerializer(data=request.data)
        if serializer.is_valid():
            file_path = serializer.validated_data['file_path']
            data = extract_data_from_excel(file_path)
            global global_x_values, global_y_values
            global_x_values = data[:, 2].astype(float)
            global_y_values = data[:, 3].astype(float)
            zero_segments, non_zero_segments, initial_params, fits_data, centers, tolerances = Spline_fit_byXY(
                global_x_values, 
                global_y_values)
            centers = [int(center) for center in centers]
            return Response({
                'zero_segments': zero_segments,
                'non_zero_segments': non_zero_segments,
                'initial_params': initial_params,
                'fits': fits_data,
                'centers': centers,
                'tolerances': tolerances,
            })
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RecalculateData(APIView):
    def post(self, request):
        serializer = DataSerializer(data=json.loads(request.body))
        if serializer.is_valid():
            centers = serializer.validated_data.get('centers')
            lengths = serializer.validated_data.get('lengths')
            file_path = serializer.validated_data.get('file_path')

            zero_segments_data, non_zero_segments_data, initial_params, fits_data, centers, tolerances = Re_spline_fit_byXY(
                global_x_values, 
                global_y_values, 
                new_params2=lengths,
                new_params1=centers)
            tolerances = [int(t) for t in tolerances]
            centers = [int(c) for c in centers]
            initial_params = [int(p) for p in initial_params]

            return Response({
                'zero_segments': zero_segments_data,
                'non_zero_segments': non_zero_segments_data,
                'fits': fits_data,
                'centers': centers,
                'tolerances': tolerances
            })
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UploadFileLon(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @swagger_auto_schema(
        operation_description="上传中桩高程表文件(用于纵断面拟合)",
        manual_parameters=[
            openapi.Parameter(
                'file',
                openapi.IN_FORM,
                description="上传的文件",
                type=openapi.TYPE_FILE,
                required=True
            ),
        ],
        tags=['纵面计算'],
        responses={200: 'Success', 400: 'Bad Request'}
    )
    def post(self, request):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            base_dir = os.path.dirname(os.path.abspath(__file__))
            upload_dir = os.path.join(base_dir, 'uploaded_files_lon')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            return Response({'file_path': file_path})
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CalculateDataLon(APIView):
    @swagger_auto_schema(
        operation_description="由中桩高程表拟合纵面线元参数",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'file_path': openapi.Schema(type=openapi.TYPE_STRING, description='后端服务器上的excel文件的路径'),
            }
        ),
        tags=['纵面计算'],
        responses={200: 'Success', 400: 'Bad Request'}
    )
    def post(self, request):
        serializer = FilePathSerializer(data=request.data)
        if serializer.is_valid():
            file_path = serializer.validated_data['file_path']
            data = extract_data_from_excel_lon(file_path)
            global global_x_values, global_y_values
            z_values = data[:, 2].astype(float)
            pile_values = data[:, 1].astype(float)
            zero_segments, non_zero_segments, initial_params, fits_data, centers, tolerances = Spline_fit_byXY(
                pile_values, 
                z_values)
            centers = [int(center) for center in centers]
            return Response({
                'zero_segments': zero_segments,
                'non_zero_segments': non_zero_segments,
                'initial_params': initial_params,
                'fits': fits_data,
                'centers': centers,
                'tolerances': tolerances,
            })
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RecalculateDataLon(APIView):
    def post(self, request):
        serializer = DataSerializer(data=json.loads(request.body))
        if serializer.is_valid():
            centers = serializer.validated_data.get('centers')
            lengths = serializer.validated_data.get('lengths')
            file_path = serializer.validated_data.get('file_path')

            zero_segments_data, non_zero_segments_data, initial_params, fits_data, centers, tolerances = Re_spline_fit_byXY(
                global_x_values, 
                global_y_values, 
                new_params2=lengths,
                new_params1=centers)
            tolerances = [int(t) for t in tolerances]
            centers = [int(c) for c in centers]
            initial_params = [int(p) for p in initial_params]

            return Response({
                'zero_segments': zero_segments_data,
                'non_zero_segments': non_zero_segments_data,
                'fits': fits_data,
                'centers': centers,
                'tolerances': tolerances
            })
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
