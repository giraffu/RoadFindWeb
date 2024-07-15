# Route/urls.py

from django.urls import path
from .views import upload_file, calculate_data,recalculate_data

urlpatterns = [
    path('upload/', upload_file, name='upload_file'),
    path('calculate/', calculate_data, name='calculate_data'),
    path('recalculate/', recalculate_data, name='recalculate_data'),
]
