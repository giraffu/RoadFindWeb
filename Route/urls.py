from django.urls import path
from .views import UploadFile, CalculateData, RecalculateData, UploadFileLon, CalculateDataLon, RecalculateDataLon

urlpatterns = [
    path('upload/', UploadFile.as_view(), name='upload_file'),
    path('calculate/', CalculateData.as_view(), name='calculate_data'),
    path('recalculate/', RecalculateData.as_view(), name='recalculate_data'),
    path('upload_lon/', UploadFileLon.as_view(), name='upload_file_lon'),
    path('calculate_lon/', CalculateDataLon.as_view(), name='calculate_data_lon'),
    path('recalculate_lon/', RecalculateDataLon.as_view(), name='recalculate_data_lon'),
]
