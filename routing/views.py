from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple, Optional
import itertools

import requests
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import SavedRoute


# Simple constants for demo purposes
CONSUMPTION_KM_PER_UNIT = 10.0  # km per unit of fuel
TANK_CAPACITY_UNITS = 40.0      # total tank capacity in fuel units (default)
OSRM_BASE = "https://router.project-osrm.org"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "routing/index.html")


def _osrm_base() -> str:
    # Defensive: always return the canonical public OSRM host
    return "https://router.project-osrm.org"


def osrm_table(coords: List[Tuple[float, float]]) -> Dict[str, Any]:
    # coords as [(lat, lon), ...]
    locs = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"{_osrm_base()}/table/v1/driving/{locs}?annotations=distance,duration"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def osrm_route(line_coords: List[Tuple[float, float]]) -> Dict[str, Any]:
    locs = ";".join([f"{lon},{lat}" for lat, lon in line_coords])
    url = f"{_osrm_base()}/route/v1/driving/{locs}?overview=full&geometries=geojson&steps=false"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def query_fuel_stations(bbox: Tuple[float, float, float, float], limit: int = 100) -> List[Dict[str, Any]]:
    # bbox = (min_lat, min_lon, max_lat, max_lon)
    q = f"""
    [out:json][timeout:25];
    node["amenity"="fuel"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    out center {limit};
    """
    r = requests.post(OVERPASS_URL, data={"data": q}, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    for el in data.get("elements", []):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        results.append({
            "lat": lat,
            "lon": lon,
            "name": (el.get("tags") or {}).get("name", "Gas Station"),
        })
    return results


def expand_bbox(points: List[Tuple[float, float]], pad: float = 0.1) -> Tuple[float, float, float, float]:
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return (
        min(lats) - pad,
        min(lons) - pad,
        max(lats) + pad,
        max(lons) + pad,
    )


def nearest_neighbor_order(start_idx: int, dest_indices: List[int], dist_matrix: List[List[float]]) -> List[int]:
    order = []
    current = start_idx
    remaining = set(dest_indices)
    while remaining:
        nxt = min(remaining, key=lambda i: dist_matrix[current][i])
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order


def tsp_optimal_order_from_start(dist_km: List[List[float]]) -> List[int]:
    """
    Compute optimal order of visiting all destinations using Held–Karp.
    Nodes: 0 is start, 1..n-1 are destinations. Returns a list of node indices (subset of 1..n-1) in optimal order.
    This solves shortest Hamiltonian path starting at 0 and ending anywhere among destinations.
    Complexity: O(n^2 * 2^(n-1)). Suitable for <= 12-14 destinations.
    """
    n = len(dist_km)
    if n <= 2:
        return [1] if n == 2 else []

    m = n - 1  # number of destinations
    # Map destination local index j in [0..m-1] to global node index g = j+1
    from math import inf
    size = 1 << m
    dp = [[inf] * m for _ in range(size)]
    parent: List[List[int]] = [[-1] * m for _ in range(size)]

    # Initialize: subsets with single destination j
    for j in range(m):
        dp[1 << j][j] = dist_km[0][j + 1]
        parent[1 << j][j] = -1

    # Iterate over subsets
    for mask in range(size):
        if mask == 0:
            continue
        for j in range(m):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            if prev_mask == 0:
                continue
            best_cost = dp[mask][j]
            best_k = parent[mask][j]
            # try previous destination k
            k_mask = prev_mask
            k = 0
            while k_mask:
                lsb = k_mask & -k_mask
                k = (lsb.bit_length() - 1)
                cand = dp[prev_mask][k] + dist_km[k + 1][j + 1]
                if cand < best_cost:
                    best_cost = cand
                    best_k = k
                k_mask ^= lsb
            dp[mask][j] = best_cost
            parent[mask][j] = best_k

    # Termination: path can end at any destination j
    full = size - 1
    end_j = min(range(m), key=lambda j: dp[full][j])

    # Reconstruct path (dest local indices), then map to global node indices
    order_local: List[int] = []
    mask = full
    j = end_j
    while j != -1:
        order_local.append(j)
        pj = parent[mask][j]
        if pj == -1:
            break
        mask ^= (1 << j)
        j = pj
    order_local.reverse()
    order_global = [idx + 1 for idx in order_local]
    return order_global


def tsp_optimal_cycle_from_start(dist_km: List[List[float]]) -> List[int]:
    """Optimal tour (cycle) starting and ending at node 0; returns order of destinations (1..n-1)."""
    n = len(dist_km)
    if n <= 2:
        return [1] if n == 2 else []
    m = n - 1
    from math import inf
    size = 1 << m
    dp = [[inf] * m for _ in range(size)]
    parent: List[List[int]] = [[-1] * m for _ in range(size)]
    for j in range(m):
        dp[1 << j][j] = dist_km[0][j + 1]
        parent[1 << j][j] = -1
    for mask in range(size):
        if mask == 0:
            continue
        for j in range(m):
            if not (mask & (1 << j)):
                continue
            prev_mask = mask ^ (1 << j)
            if prev_mask == 0:
                continue
            best_cost = dp[mask][j]
            best_k = parent[mask][j]
            k_mask = prev_mask
            while k_mask:
                lsb = k_mask & -k_mask
                k = (lsb.bit_length() - 1)
                cand = dp[prev_mask][k] + dist_km[k + 1][j + 1]
                if cand < best_cost:
                    best_cost = cand
                    best_k = k
                k_mask ^= lsb
            dp[mask][j] = best_cost
            parent[mask][j] = best_k
    full = size - 1
    # Choose end that minimizes cycle cost back to start
    end_j = min(range(m), key=lambda j: dp[full][j] + dist_km[j + 1][0])
    # Reconstruct
    order_local: List[int] = []
    mask = full
    j = end_j
    while j != -1:
        order_local.append(j)
        pj = parent[mask][j]
        if pj == -1:
            break
        mask ^= (1 << j)
        j = pj
    order_local.reverse()
    return [idx + 1 for idx in order_local]


def plan_with_refuel(start: Tuple[float, float], destinations: List[Tuple[float, float]], initial_fuel: float,
                     tank_capacity: float = TANK_CAPACITY_UNITS, consumption_km_per_unit: float = CONSUMPTION_KM_PER_UNIT,
                     user_station_points: Optional[List[Tuple[float, float]]] = None,
                     return_to_start: bool = False,
                     dist_km: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    # Build full list of points for matrix: start + destinations
    points = [start] + destinations
    if dist_km is None:
        try:
            table = osrm_table(points)
            distances = table.get("distances")  # in meters
            if not distances:
                raise ValueError("OSRM table returned no distances")
        except Exception as e:
            # Fallback: cannot get table; return warning and naive geometry using given order
            warning = f"Servicio de enrutamiento no disponible (tabla): {str(e)}"
            path_nodes = points[:1]  # start only; no route computed
            geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
            return {
                "order_indices": [],
                "itinerary": [],
                "path": geometry,
                "total_distance_m": None,
                "total_distance_km": None,
                "total_duration_s": None,
                "warning": warning,
                "break_point": points[0],
            }
        # Convert to km
        dist_km = [[(d or 0.0)/1000.0 for d in row] for row in distances]

    # Optimal order using Held–Karp
    # Fallback to nearest neighbor if too many points for performance.
    if len(points) - 1 <= 12:
        visit_order = tsp_optimal_cycle_from_start(dist_km) if return_to_start else tsp_optimal_order_from_start(dist_km)
    else:
        dest_indices = list(range(1, len(points)))
        visit_order = nearest_neighbor_order(0, dest_indices, dist_km)

    # Fuel stations: use user-provided if available, otherwise query Overpass in the area
    if user_station_points:
        station_points = list(user_station_points)
    else:
        bbox = expand_bbox(points, pad=0.2)
        stations = query_fuel_stations(bbox)
        station_points = [(s["lat"], s["lon"]) for s in stations]

    # Greedy simulation with refuel when needed: choose nearest station along the way
    fuel = min(initial_fuel, tank_capacity)
    path_nodes: List[Tuple[float, float]] = [start]
    itinerary: List[Dict[str, Any]] = []
    current_idx = 0

    for nxt_idx in visit_order:
        leg_distance_km = dist_km[current_idx][nxt_idx]
        required_units = leg_distance_km / consumption_km_per_unit
        if fuel + 1e-6 >= required_units:
            # can reach
            fuel -= required_units
            itinerary.append({"type": "visit", "point": points[nxt_idx], "distance_km": leg_distance_km})
            current_idx = nxt_idx
        else:
            # Need to refuel.
            if not station_points:
                warning = "No hay gasolineras disponibles para continuar desde este punto."
                # Build partial geometry
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": points[current_idx],
                }
            # distances from current point to stations
            tmp_points = [points[current_idx]] + station_points
            try:
                tmp_table = osrm_table(tmp_points)
            except Exception as e:
                warning = f"Servicio de enrutamiento no disponible (tabla estaciones): {str(e)}"
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": points[current_idx],
                }
            tmp_d_km = [((d or 0.0)/1000.0) for d in tmp_table.get("distances", [[]])[0]]
            # find reachable stations
            reachable = []
            for i, dkm in enumerate(tmp_d_km[1:], start=1):
                need_units = dkm / consumption_km_per_unit
                reachable.append((need_units <= fuel + 1e-6, dkm, i))
            reachable_true = [t for t in reachable if t[0]]
            if not reachable_true:
                # No reachable stations; return partial result and warn
                warning = "No hay gasolineras alcanzables desde este punto."
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": points[current_idx],
                }
            candidates = sorted(reachable_true, key=lambda t: t[1])
            chosen = candidates[0]
            station_idx = chosen[2] - 1
            station_point = station_points[station_idx]
            # move to station
            leg_to_station_km = chosen[1]
            units_to_station = leg_to_station_km / consumption_km_per_unit
            fuel = max(0.0, fuel - units_to_station)
            itinerary.append({"type": "refuel", "point": station_point, "distance_km": leg_to_station_km})
            fuel = tank_capacity
            path_nodes.append(station_point)
            # After refuel, go to nxt_idx
            tmp_points2 = [station_point, points[nxt_idx]]
            try:
                tmp_table2 = osrm_table(tmp_points2)
            except Exception as e:
                warning = f"Servicio de enrutamiento no disponible (tabla destino tras estación): {str(e)}"
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": station_point,
                }
            dkm2 = (tmp_table2.get("distances", [[0, 0], [0, 0]])[0][1] or 0.0)/1000.0
            need2 = dkm2 / consumption_km_per_unit
            if fuel + 1e-6 < need2:
                warning = "No alcanza el combustible para el punto siguiente. No se encontraron estaciones de combistibles intermedias"
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": station_point,
                }
            fuel -= need2
            itinerary.append({"type": "visit", "point": points[nxt_idx], "distance_km": dkm2})
            current_idx = nxt_idx
            path_nodes.append(points[nxt_idx])
            continue
        path_nodes.append(points[nxt_idx])

    # If required, return to start with refueling if needed
    if return_to_start and current_idx != 0:
        leg_distance_km = dist_km[current_idx][0]
        required_units = leg_distance_km / consumption_km_per_unit
        if fuel + 1e-6 >= required_units:
            fuel -= required_units
            itinerary.append({"type": "return", "point": points[0], "distance_km": leg_distance_km})
            current_idx = 0
            path_nodes.append(points[0])
        else:
            if not station_points:
                warning = "No hay gasolineras disponibles para regresar al inicio."
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": points[current_idx],
                }
            tmp_points = [points[current_idx]] + station_points
            tmp_table = osrm_table(tmp_points)
            tmp_d_km = [((d or 0.0)/1000.0) for d in tmp_table.get("distances", [[]])[0]]
            reachable = []
            for i, dkm in enumerate(tmp_d_km[1:], start=1):
                need_units = dkm / consumption_km_per_unit
                reachable.append((need_units <= fuel + 1e-6, dkm, i))
            reachable_true = [t for t in reachable if t[0]]
            if not reachable_true:
                warning = "No hay gasolineras alcanzables para regresar al inicio."
                try:
                    route_geo = osrm_route(path_nodes)
                    geometry = route_geo["routes"][0]["geometry"]
                    distance_m = route_geo["routes"][0]["distance"]
                    duration_s = route_geo["routes"][0]["duration"]
                except Exception:
                    geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
                    distance_m = None
                    duration_s = None
                return {
                    "order_indices": visit_order,
                    "itinerary": itinerary,
                    "path": geometry,
                    "total_distance_m": distance_m,
                    "total_distance_km": (distance_m/1000.0) if distance_m else None,
                    "total_duration_s": duration_s,
                    "warning": warning,
                    "break_point": points[current_idx],
                }
            candidates = sorted(reachable_true, key=lambda t: t[1])
            chosen = candidates[0]
            station_idx = chosen[2] - 1
            station_point = station_points[station_idx]
            # move to station
            leg_to_station_km = chosen[1]
            units_to_station = leg_to_station_km / consumption_km_per_unit
            fuel = max(0.0, fuel - units_to_station)
            itinerary.append({"type": "refuel", "point": station_point, "distance_km": leg_to_station_km})
            fuel = tank_capacity
            path_nodes.append(station_point)
            # station -> start
            tmp_points2 = [station_point, points[0]]
            tmp_table2 = osrm_table(tmp_points2)
            dkm2 = (tmp_table2.get("distances", [[0, 0], [0, 0]])[0][1] or 0.0)/1000.0
            need2 = dkm2 / consumption_km_per_unit
            if fuel + 1e-6 < need2:
                raise ValueError("Even after refuel, cannot return to start due to data issues.")
            fuel -= need2
            itinerary.append({"type": "return", "point": points[0], "distance_km": dkm2})
            current_idx = 0
            path_nodes.append(points[0])

    # Build display route polyline via OSRM route on the sequence path_nodes
    try:
        route_geo = osrm_route(path_nodes)
        geometry = route_geo["routes"][0]["geometry"]
        distance_m = route_geo["routes"][0]["distance"]
        duration_s = route_geo["routes"][0]["duration"]
    except Exception:
        geometry = {"type": "LineString", "coordinates": [(p[1], p[0]) for p in path_nodes]}
        distance_m = sum([dist_km[i][j] for i, j in zip([0]+visit_order, visit_order)]) * 1000.0
        duration_s = None

    return {
        "order_indices": visit_order,
        "itinerary": itinerary,
        "path": geometry,
        "total_distance_m": distance_m,
        "total_distance_km": (distance_m / 1000.0) if distance_m is not None else None,
        "total_duration_s": duration_s,
    }


def _comparison_all_orders(start: Tuple[float, float], destinations: List[Tuple[float, float]],
                           return_to_start: bool,
                           consumption_km_per_unit: float,
                           dist_m: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
    """
    Compute distance and fuel usage for all permutations of destination orders.
    Caps computation to m <= 8 destinations. Returns a list sorted by fuel usage.
    order values are 1-based destination indices referencing the original destinations list.
    """
    points = [start] + destinations
    if dist_m is None:
        try:
            table = osrm_table(points)
            distances = table.get("distances")
            if not distances:
                return []
        except Exception:
            return []
        # keep raw meters to preserve None; convert on demand
        dist_m = distances
    m = len(destinations)
    if m == 0:
        return []
    if m > 8:
        # too many permutations for real-time UI
        return []
    results: List[Dict[str, Any]] = []
    if consumption_km_per_unit <= 0:
        return []
    for perm in itertools.permutations(range(1, m + 1)):
        legs_m: List[float] = []
        # start -> first
        first = dist_m[0][perm[0]]
        if first is None:
            continue
        legs_m.append(first)
        total_m = first
        # internal legs
        valid = True
        for a, b in zip(perm, perm[1:]):
            seg = dist_m[a][b]
            if seg is None:
                valid = False
                break
            legs_m.append(seg)
            total_m += seg
        if not valid:
            continue
        if return_to_start:
            back = dist_m[perm[-1]][0]
            if back is None:
                continue
            legs_m.append(back)
            total_m += back
        legs_km = [(lm or 0.0) / 1000.0 for lm in legs_m]
        total_km = (total_m or 0.0) / 1000.0
        fuel = total_km / consumption_km_per_unit
        results.append({
            "order": list(perm),
            "legs_km": legs_km,
            "distance_km": total_km,
            "fuel_units": fuel,
        })
    results.sort(key=lambda r: (r["fuel_units"], r["distance_km"]))
    return results


@csrf_exempt
def api_route(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
        start = tuple(data.get("start"))  # [lat, lon]
        destinations = [tuple(p) for p in data.get("destinations", [])]
        initial_fuel = float(data.get("initial_fuel", TANK_CAPACITY_UNITS))
        tank_capacity = float(data.get("tank_capacity", TANK_CAPACITY_UNITS))
        consumption = float(data.get("consumption_km_per_unit", CONSUMPTION_KM_PER_UNIT))
        user_stations_raw = data.get("user_stations", []) or []
        user_stations = [tuple(p) for p in user_stations_raw if isinstance(p, (list, tuple)) and len(p) == 2]
        return_to_start = bool(data.get("return_to_start", False))
        compare_all = bool(data.get("compare_all_orders", False))
        if not start or not destinations:
            return JsonResponse({"error": "Provide 'start' and at least one 'destinations'"}, status=400)
        # Precompute OSRM distance matrix once, if possible
        points = [start] + destinations
        dist_km_pref: Optional[List[List[float]]] = None
        dist_m_pref: Optional[List[List[float]]] = None
        try:
            table_pref = osrm_table(points)
            distances_pref = table_pref.get("distances")
            if distances_pref:
                dist_m_pref = distances_pref
                dist_km_pref = [[(d or 0.0)/1000.0 for d in row] for row in distances_pref]
        except Exception:
            # Fallback to per-function fetching
            pass

        plan = plan_with_refuel(start, destinations, initial_fuel, tank_capacity, consumption,
                                user_station_points=user_stations, return_to_start=return_to_start,
                                dist_km=dist_km_pref)
        if compare_all:
            try:
                comparison = _comparison_all_orders(start, destinations, return_to_start, consumption,
                                                    dist_m=dist_m_pref)
            except Exception:
                comparison = []
            plan["comparison"] = comparison
        # Add total fuel consumption if distance available
        try:
            td_km = plan.get("total_distance_km")
            plan["fuel_units_total"] = (td_km / consumption) if (td_km is not None and consumption > 0) else None
        except Exception:
            plan["fuel_units_total"] = None
        return JsonResponse(plan)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
def api_save_route(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        data = json.loads(request.body.decode("utf-8"))
        name = (data.get("name") or "").strip()
        if not name:
            return JsonResponse({"error": "Falta el nombre de identificación"}, status=400)

        # Required computed plan fields
        plan = data.get("plan") or {}
        # Required inputs to reproduce
        start = tuple((data.get("start") or []))
        destinations = [tuple(p) for p in (data.get("destinations") or [])]
        user_stations = [tuple(p) for p in (data.get("user_stations") or [])]
        return_to_start = bool(data.get("return_to_start", False))
        initial_fuel = float(data.get("initial_fuel"))
        tank_capacity = float(data.get("tank_capacity"))
        consumption = float(data.get("consumption_km_per_unit"))

        if not start or not destinations:
            return JsonResponse({"error": "Faltan puntos (inicio/destinos)"}, status=400)

        obj = SavedRoute.objects.create(
            name=name,
            start=list(start),
            destinations=[list(p) for p in destinations],
            user_stations=[list(p) for p in user_stations] if user_stations else None,
            return_to_start=return_to_start,
            initial_fuel=initial_fuel,
            tank_capacity=tank_capacity,
            consumption_km_per_unit=consumption,
            order_indices=plan.get("order_indices"),
            itinerary=plan.get("itinerary"),
            path=plan.get("path"),
            total_distance_km=plan.get("total_distance_km"),
            total_duration_s=plan.get("total_duration_s"),
            fuel_units_total=plan.get("fuel_units_total"),
        )
        return JsonResponse({"ok": True, "id": obj.id, "name": obj.name})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


def api_list_saved(request: HttpRequest) -> JsonResponse:
    """List saved routes, optionally filtered by ?q= substring on name."""
    q = (request.GET.get("q") or "").strip().lower()
    qs = SavedRoute.objects.all()
    if q:
        qs = qs.filter(name__icontains=q)
    items = [
        {
            "id": r.id,
            "name": r.name,
            "created_at": r.created_at.isoformat(timespec="seconds"),
            "total_distance_km": r.total_distance_km,
        }
        for r in qs[:50]
    ]
    return JsonResponse({"results": items})


def api_get_saved(request: HttpRequest, route_id: int) -> JsonResponse:
    """Return a saved route by id with all fields needed to load into the UI."""
    try:
        r = SavedRoute.objects.get(id=route_id)
    except SavedRoute.DoesNotExist:
        return JsonResponse({"error": "No encontrado"}, status=404)
    data = {
        "id": r.id,
        "name": r.name,
        "created_at": r.created_at.isoformat(timespec="seconds"),
        "start": r.start,
        "destinations": r.destinations,
        "user_stations": r.user_stations or [],
        "return_to_start": r.return_to_start,
        "initial_fuel": r.initial_fuel,
        "tank_capacity": r.tank_capacity,
        "consumption_km_per_unit": r.consumption_km_per_unit,
        # Plan-like fields expected by frontend
        "plan": {
            "order_indices": r.order_indices or [],
            "itinerary": r.itinerary or [],
            "path": r.path or None,
            "total_distance_km": r.total_distance_km,
            "total_duration_s": r.total_duration_s,
            "fuel_units_total": r.fuel_units_total,
        },
    }
    return JsonResponse(data)
