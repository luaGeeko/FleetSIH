from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict

@dataclass
class EnvironmentState:
    time_step: int = 0
    traffic_zones: Dict[str, float] = field(default_factory=dict)
    active_alerts: List[str] = field(default_factory=list)

@dataclass
class Shipment:
    id: str
    pickup: Tuple[float, float]
    destination: Tuple[float, float]
    weight: int = 1
    priority: int = 1
    status: str = "pending"  # pending, assigned, picked_up, delivered
    assigned_vehicle_id: Optional[str] = None
    created_tick: int = 0
    deadline: int = 150
    # Telemetry
    assigned_tick: Optional[int] = None  # Changed from -1 to None for safe logic checks
    pickup_tick: Optional[int] = None    # Changed from -1 to None
    delivery_tick: Optional[int] = None  # Changed from -1 to None

@dataclass
class RouteStop:
    location: Tuple[float, float]
    shipment_id: str
    action: str  # "pickup" or "delivery"

@dataclass
class Vehicle:
    id: str
    position: Tuple[float, float]
    capacity: int = 5
    current_payload: int = 0
    assigned_shipments: List[str] = field(default_factory=list)
    route: List[RouteStop] = field(default_factory=list)
    status: str = "idle"  # idle, en_route, broken_down
    speed: float = 2.0
    # Telemetry
    total_distance: float = 0.0
    time_active: int = 0