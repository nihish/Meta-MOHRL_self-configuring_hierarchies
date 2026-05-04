"""
Generate SUMO network and route configuration files for experiments.

Creates:
1. Single intersection (4-way) for validation
2. 2x2 grid network for multi-agent experiments
Both with heterogeneous vehicle types (car, truck, bus).
"""

import os
import subprocess
import sys


def generate_single_intersection(output_dir: str):
    """Generate a simple 4-way single intersection network."""
    os.makedirs(output_dir, exist_ok=True)

    # Node file
    node_file = os.path.join(output_dir, "single.nod.xml")
    with open(node_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<nodes>
    <node id="C" x="0.0" y="0.0" type="traffic_light"/>
    <node id="N" x="0.0" y="200.0" type="priority"/>
    <node id="S" x="0.0" y="-200.0" type="priority"/>
    <node id="E" x="200.0" y="0.0" type="priority"/>
    <node id="W" x="-200.0" y="0.0" type="priority"/>
</nodes>
""")

    # Edge file
    edge_file = os.path.join(output_dir, "single.edg.xml")
    with open(edge_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<edges>
    <edge id="NC" from="N" to="C" numLanes="2" speed="13.89"/>
    <edge id="CN" from="C" to="N" numLanes="2" speed="13.89"/>
    <edge id="SC" from="S" to="C" numLanes="2" speed="13.89"/>
    <edge id="CS" from="C" to="S" numLanes="2" speed="13.89"/>
    <edge id="EC" from="E" to="C" numLanes="2" speed="13.89"/>
    <edge id="CE" from="C" to="E" numLanes="2" speed="13.89"/>
    <edge id="WC" from="W" to="C" numLanes="2" speed="13.89"/>
    <edge id="CW" from="C" to="W" numLanes="2" speed="13.89"/>
</edges>
""")

    # Build network using netconvert
    net_file = os.path.join(output_dir, "single_intersection.net.xml")
    sumo_home = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
    netconvert = os.path.join(sumo_home, 'bin', 'netconvert')

    try:
        subprocess.run([
            netconvert,
            '--node-files', node_file,
            '--edge-files', edge_file,
            '--output-file', net_file,
            '--no-turnarounds'
        ], check=True, capture_output=True)
        print(f"Network generated: {net_file}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"netconvert failed ({e}), writing fallback net.xml")
        _write_fallback_single_net(net_file)

    # Route file with heterogeneous vehicles
    route_file = os.path.join(output_dir, "single_intersection.rou.xml")
    with open(route_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0"
           minGap="2.5" maxSpeed="13.89" color="1,0.5,0.5" guiShape="passenger"/>
    <vType id="truck" accel="1.3" decel="3.0" sigma="0.5" length="12.0"
           minGap="3.0" maxSpeed="11.11" color="0.5,0.5,1" guiShape="truck"/>
    <vType id="bus" accel="1.5" decel="3.5" sigma="0.5" length="15.0"
           minGap="3.5" maxSpeed="12.50" color="0.5,1,0.5" guiShape="bus"/>

    <!-- North-South flows -->
    <flow id="ns_car" type="car" from="NC" to="CS" begin="0" end="3600"
          vehsPerHour="400" departSpeed="max" departLane="best"/>
    <flow id="ns_truck" type="truck" from="NC" to="CS" begin="0" end="3600"
          vehsPerHour="100" departSpeed="max" departLane="best"/>
    <flow id="ns_bus" type="bus" from="NC" to="CS" begin="0" end="3600"
          vehsPerHour="50" departSpeed="max" departLane="best"/>

    <!-- South-North flows -->
    <flow id="sn_car" type="car" from="SC" to="CN" begin="0" end="3600"
          vehsPerHour="350" departSpeed="max" departLane="best"/>
    <flow id="sn_truck" type="truck" from="SC" to="CN" begin="0" end="3600"
          vehsPerHour="80" departSpeed="max" departLane="best"/>

    <!-- East-West flows -->
    <flow id="ew_car" type="car" from="EC" to="CW" begin="0" end="3600"
          vehsPerHour="300" departSpeed="max" departLane="best"/>
    <flow id="ew_truck" type="truck" from="EC" to="CW" begin="0" end="3600"
          vehsPerHour="60" departSpeed="max" departLane="best"/>
    <flow id="ew_bus" type="bus" from="EC" to="CW" begin="0" end="3600"
          vehsPerHour="30" departSpeed="max" departLane="best"/>

    <!-- West-East flows -->
    <flow id="we_car" type="car" from="WC" to="CE" begin="0" end="3600"
          vehsPerHour="320" departSpeed="max" departLane="best"/>
    <flow id="we_truck" type="truck" from="WC" to="CE" begin="0" end="3600"
          vehsPerHour="70" departSpeed="max" departLane="best"/>
</routes>
""")
    print(f"Routes generated: {route_file}")


def generate_grid_2x2(output_dir: str):
    """Generate a 2x2 grid network for multi-agent experiments."""
    os.makedirs(output_dir, exist_ok=True)

    sumo_home = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
    netgenerate = os.path.join(sumo_home, 'bin', 'netgenerate')
    net_file = os.path.join(output_dir, "grid2x2.net.xml")

    try:
        subprocess.run([
            netgenerate,
            '--grid',
            '--grid.x-number', '2',
            '--grid.y-number', '2',
            '--grid.x-length', '200',
            '--grid.y-length', '200',
            '--grid.attach-length', '200',
            '--default.lanenumber', '2',
            '--default.speed', '13.89',
            '--tls.guess', 'true',
            '--output-file', net_file,
            '--no-turnarounds'
        ], check=True, capture_output=True)
        print(f"Grid network generated: {net_file}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"netgenerate failed ({e}), writing fallback grid net.xml")
        _write_fallback_grid_net(net_file)

    # Generate routes for 2x2 grid
    route_file = os.path.join(output_dir, "grid2x2.rou.xml")
    _generate_grid_routes(route_file)
    print(f"Grid routes generated: {route_file}")


def _generate_grid_routes(route_file: str):
    """Generate route file for 2x2 grid with heterogeneous vehicles."""
    with open(route_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0"
           minGap="2.5" maxSpeed="13.89" color="1,0.5,0.5" guiShape="passenger"/>
    <vType id="truck" accel="1.3" decel="3.0" sigma="0.5" length="12.0"
           minGap="3.0" maxSpeed="11.11" color="0.5,0.5,1" guiShape="truck"/>
    <vType id="bus" accel="1.5" decel="3.5" sigma="0.5" length="15.0"
           minGap="3.5" maxSpeed="12.50" color="0.5,1,0.5" guiShape="bus"/>

    <!-- Horizontal flows (left to right) -->
    <flow id="h1_car" type="car" from="left0A0" to="B1right1"
          begin="0" end="3600" vehsPerHour="300" departSpeed="max" departLane="best"/>
    <flow id="h1_truck" type="truck" from="left0A0" to="B1right1"
          begin="0" end="3600" vehsPerHour="80" departSpeed="max" departLane="best"/>
    <flow id="h2_car" type="car" from="left1A1" to="B0right0"
          begin="0" end="3600" vehsPerHour="280" departSpeed="max" departLane="best"/>
    <flow id="h2_bus" type="bus" from="left1A1" to="B0right0"
          begin="0" end="3600" vehsPerHour="40" departSpeed="max" departLane="best"/>

    <!-- Vertical flows (bottom to top) -->
    <flow id="v1_car" type="car" from="bottom0A0" to="A1top0"
          begin="0" end="3600" vehsPerHour="250" departSpeed="max" departLane="best"/>
    <flow id="v1_truck" type="truck" from="bottom0A0" to="A1top0"
          begin="0" end="3600" vehsPerHour="60" departSpeed="max" departLane="best"/>
    <flow id="v2_car" type="car" from="bottom1B0" to="B1top1"
          begin="0" end="3600" vehsPerHour="270" departSpeed="max" departLane="best"/>
    <flow id="v2_bus" type="bus" from="bottom1B0" to="B1top1"
          begin="0" end="3600" vehsPerHour="35" departSpeed="max" departLane="best"/>

    <!-- Reverse flows -->
    <flow id="rh1_car" type="car" from="right1B1" to="A0left0"
          begin="0" end="3600" vehsPerHour="260" departSpeed="max" departLane="best"/>
    <flow id="rv1_car" type="car" from="top0A1" to="A0bottom0"
          begin="0" end="3600" vehsPerHour="220" departSpeed="max" departLane="best"/>
</routes>
""")


def _write_fallback_single_net(net_file: str):
    """Write a minimal valid net.xml if netconvert unavailable."""
    with open(net_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <location netOffset="200.00,200.00" convBoundary="0.00,0.00,400.00,400.00"
              origBoundary="-200.00,-200.00,200.00,200.00"/>
    <edge id="NC" from="N" to="C" priority="1">
        <lane id="NC_0" index="0" speed="13.89" length="200.00"
              shape="200.00,400.00 200.00,200.00"/>
    </edge>
    <junction id="C" type="traffic_light" x="200.00" y="200.00"
              incLanes="NC_0 SC_0 EC_0 WC_0"/>
    <tlLogic id="C" type="static" programID="0" offset="0">
        <phase duration="31" state="GGrrGGrr"/>
        <phase duration="6"  state="yyrryyrr"/>
        <phase duration="31" state="rrGGrrGG"/>
        <phase duration="6"  state="rryyrryy"/>
    </tlLogic>
</net>
""")


def _write_fallback_grid_net(net_file: str):
    """Write a minimal valid 2x2 grid net.xml."""
    with open(net_file, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.16">
    <location netOffset="200.00,200.00" convBoundary="0.00,0.00,600.00,600.00"
              origBoundary="-200.00,-200.00,400.00,400.00"/>

    <junction id="A0" type="traffic_light" x="200.00" y="200.00" incLanes=""/>
    <junction id="A1" type="traffic_light" x="200.00" y="400.00" incLanes=""/>
    <junction id="B0" type="traffic_light" x="400.00" y="200.00" incLanes=""/>
    <junction id="B1" type="traffic_light" x="400.00" y="400.00" incLanes=""/>

    <tlLogic id="A0" type="static" programID="0" offset="0">
        <phase duration="31" state="GGrrGGrr"/>
        <phase duration="6"  state="yyrryyrr"/>
        <phase duration="31" state="rrGGrrGG"/>
        <phase duration="6"  state="rryyrryy"/>
    </tlLogic>
    <tlLogic id="A1" type="static" programID="0" offset="0">
        <phase duration="31" state="GGrrGGrr"/>
        <phase duration="6"  state="yyrryyrr"/>
        <phase duration="31" state="rrGGrrGG"/>
        <phase duration="6"  state="rryyrryy"/>
    </tlLogic>
    <tlLogic id="B0" type="static" programID="0" offset="0">
        <phase duration="31" state="GGrrGGrr"/>
        <phase duration="6"  state="yyrryyrr"/>
        <phase duration="31" state="rrGGrrGG"/>
        <phase duration="6"  state="rryyrryy"/>
    </tlLogic>
    <tlLogic id="B1" type="static" programID="0" offset="0">
        <phase duration="31" state="GGrrGGrr"/>
        <phase duration="6"  state="yyrryyrr"/>
        <phase duration="31" state="rrGGrrGG"/>
        <phase duration="6"  state="rryyrryy"/>
    </tlLogic>
</net>
""")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, 'sumo_configs')
    generate_single_intersection(config_dir)
    generate_grid_2x2(config_dir)
    print("All SUMO configs generated successfully.")
