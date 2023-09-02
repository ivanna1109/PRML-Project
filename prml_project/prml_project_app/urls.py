from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('get_results/', views.get_results, name='get_results'),
]