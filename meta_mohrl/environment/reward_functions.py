"""
Custom multi-objective reward functions for SUMO-RL traffic control.

Three objectives at three hierarchy levels:
1. Average speed reward (high-level strategic objective)
2. Waiting time reward (mid-level tactical objective)
3. Queue length reward (low-level operational objective)
"""

import numpy as np


def speed_reward(traffic_signal) -> float:
    """Maximize average speed across the intersection.

    High-level objective: network-wide throughput.
    Normalized to roughly [-1, 1] range.
    """
    speeds = []
    for lane in traffic_signal.lanes:
        vehicles = traffic_signal.sumo.lane.getLastStepVehicleIDs(lane)
        for veh in vehicles:
            speeds.append(traffic_signal.sumo.vehicle.getSpeed(veh))

    if len(speeds) == 0:
        return 0.0

    avg_speed = np.mean(speeds)
    max_speed = 13.89  # ~50 km/h
    return avg_speed / max_speed  # normalized [0, 1]


def waiting_time_reward(traffic_signal) -> float:
    """Minimize total waiting time at the intersection.

    Mid-level objective: reduce delays.
    Returns negative waiting time (higher = better).
    """
    total_wait = 0.0
    for lane in traffic_signal.lanes:
        total_wait += traffic_signal.sumo.lane.getWaitingTime(lane)

    # Normalize: typical max waiting time per lane ~100s
    max_wait = 100.0 * len(traffic_signal.lanes)
    return -total_wait / max(max_wait, 1.0)  # [-1, 0]


def queue_length_reward(traffic_signal) -> float:
    """Minimize queue length at the intersection.

    Low-level objective: reduce congestion.
    Returns negative queue count (higher = better).
    """
    total_queue = 0
    for lane in traffic_signal.lanes:
        total_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    # Normalize: typical max queue per lane ~20 vehicles
    max_queue = 20.0 * len(traffic_signal.lanes)
    return -total_queue / max(max_queue, 1.0)  # [-1, 0]


def multi_objective_reward(traffic_signal) -> np.ndarray:
    """Combined multi-objective reward vector.

    Returns: [speed_reward, waiting_time_reward, queue_length_reward]
    """
    return np.array([
        speed_reward(traffic_signal),
        waiting_time_reward(traffic_signal),
        queue_length_reward(traffic_signal)
    ], dtype=np.float32)


def pressure_reward(traffic_signal) -> float:
    """Intersection pressure (difference between incoming and outgoing vehicles).

    Lower pressure indicates better flow balance.
    """
    incoming = 0
    outgoing = 0
    for lane in traffic_signal.lanes:
        incoming += traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)

    for lane in traffic_signal.out_lanes:
        outgoing += traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)

    pressure = abs(incoming - outgoing)
    max_pressure = 20.0 * len(traffic_signal.lanes)
    return -pressure / max(max_pressure, 1.0)
