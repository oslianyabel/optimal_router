from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/route", views.api_route, name="api_route"),
    path("api/save", views.api_save_route, name="api_save_route"),
    path("api/saved", views.api_list_saved, name="api_list_saved"),
    path("api/saved/<int:route_id>", views.api_get_saved, name="api_get_saved"),
]
