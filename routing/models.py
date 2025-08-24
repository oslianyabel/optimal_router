from __future__ import annotations
from django.db import models


class SavedRoute(models.Model):
    name = models.CharField(max_length=200, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Inputs
    start = models.JSONField()  # [lat, lon]
    destinations = models.JSONField()  # list of [lat, lon]
    user_stations = models.JSONField(blank=True, null=True)  # list of [lat, lon]
    return_to_start = models.BooleanField(default=False)

    # Parameters
    initial_fuel = models.FloatField()
    tank_capacity = models.FloatField()
    consumption_km_per_unit = models.FloatField()

    # Results
    order_indices = models.JSONField(blank=True, null=True)  # list of 1-based indices
    itinerary = models.JSONField(blank=True, null=True)
    path = models.JSONField(blank=True, null=True)  # GeoJSON

    total_distance_km = models.FloatField(blank=True, null=True)
    total_duration_s = models.FloatField(blank=True, null=True)
    fuel_units_total = models.FloatField(blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name} ({self.created_at:%Y-%m-%d %H:%M})"
