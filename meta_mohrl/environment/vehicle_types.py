"""
Heterogeneous vehicle type definitions for SUMO simulation.

Defines car, truck, and bus types with realistic parameters.
Generates SUMO-compatible additional files (.add.xml) for vehicle types
and route files (.rou.xml) with mixed traffic flows.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Dict


# Vehicle type specifications
VEHICLE_TYPES = {
    'car': {
        'id': 'car',
        'accel': '2.6',
        'decel': '4.5',
        'sigma': '0.5',
        'length': '5.0',
        'minGap': '2.5',
        'maxSpeed': '13.89',       # ~50 km/h
        'color': '1,0.5,0.5',      # light red
        'guiShape': 'passenger',
        'emissionClass': 'HBEFA3/PC_G_EU4',
    },
    'truck': {
        'id': 'truck',
        'accel': '1.3',
        'decel': '3.0',
        'sigma': '0.5',
        'length': '12.0',
        'minGap': '3.0',
        'maxSpeed': '11.11',       # ~40 km/h
        'color': '0.5,0.5,1',      # light blue
        'guiShape': 'truck',
        'emissionClass': 'HBEFA3/HDV_D_EU4',
    },
    'bus': {
        'id': 'bus',
        'accel': '1.5',
        'decel': '3.5',
        'sigma': '0.5',
        'length': '15.0',
        'minGap': '3.5',
        'maxSpeed': '12.50',       # ~45 km/h
        'color': '0.5,1,0.5',      # light green
        'guiShape': 'bus',
        'emissionClass': 'HBEFA3/Bus',
    }
}


def generate_vehicle_types_xml(output_path: str):
    """Generate SUMO additional file with vehicle type definitions."""
    root = ET.Element('additional')

    for vtype_id, params in VEHICLE_TYPES.items():
        vtype = ET.SubElement(root, 'vType')
        for key, value in params.items():
            vtype.set(key, value)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    print(f"Vehicle types written to {output_path}")


def generate_routes_xml(
    output_path: str,
    edges: List[str],
    num_vehicles: int = 1000,
    sim_duration: int = 3600,
    vehicle_ratios: Dict[str, float] = None
):
    """Generate route file with heterogeneous vehicle flows.

    Args:
        output_path: path to output .rou.xml file
        edges: list of edge IDs for routes
        num_vehicles: total number of vehicles
        sim_duration: simulation duration in seconds
        vehicle_ratios: dict of vehicle_type → ratio (must sum to 1)
    """
    if vehicle_ratios is None:
        vehicle_ratios = {'car': 0.7, 'truck': 0.2, 'bus': 0.1}

    root = ET.Element('routes')

    # Add vehicle types
    for vtype_id, params in VEHICLE_TYPES.items():
        vtype = ET.SubElement(root, 'vType')
        for key, value in params.items():
            vtype.set(key, value)

    # Generate flows for each route and vehicle type
    flow_id = 0
    for i, edge_from in enumerate(edges):
        for j, edge_to in enumerate(edges):
            if i == j:
                continue

            route_id = f"route_{edge_from}_to_{edge_to}"
            route = ET.SubElement(root, 'route')
            route.set('id', route_id)
            route.set('edges', f"{edge_from} {edge_to}")

            for vtype_id, ratio in vehicle_ratios.items():
                flow = ET.SubElement(root, 'flow')
                flow.set('id', f"flow_{flow_id}")
                flow.set('type', vtype_id)
                flow.set('route', route_id)
                flow.set('begin', '0')
                flow.set('end', str(sim_duration))
                vehicles_of_type = max(1, int(num_vehicles * ratio / (len(edges) * (len(edges) - 1))))
                flow.set('vehsPerHour', str(vehicles_of_type))
                flow.set('departSpeed', 'max')
                flow.set('departLane', 'best')
                flow_id += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding='unicode', xml_declaration=True)
    print(f"Routes written to {output_path}")
