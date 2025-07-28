# Fog-Edge Computing Simulation

## Overview
This project implements a simulation framework for fog and edge computing security based on the paper "An Overview of Fog Computing and Edge Computing Security and Privacy Issues" (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226). The simulation models IoT devices, edge nodes, fog nodes, and cloud datacenters, with a focus on security attacks and countermeasures in a fog-edge computing environment.

## Features
- **Device Modeling**: Simulates various device types (IoT devices, edge nodes, fog nodes, cloud datacenters) with configurable parameters
- **Network Modeling**: Simulates network connections, latency, bandwidth, and congestion
- **Task Management**: Generates, assigns, and tracks tasks across the network
- **Security Simulation**: Models various attack types and countermeasures
- **Comprehensive Metrics**: Collects and analyzes performance, network, security, and energy metrics
- **Configurable Parameters**: All simulation parameters can be adjusted via a properties file

## Project Structure
```
fog-edge-sim/
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── nci/
│                   └── fogedge/
│                       ├── SimulationDemo.java       # Main demo class
│                       ├── FogEdgeSimulation.java    # Core simulation engine
│                       ├── model/                    # Data models
│                       │   ├── SimulationConfig.java
│                       │   ├── SimulationResults.java
│                       │   ├── Device.java
│                       │   ├── DeviceType.java
│                       │   ├── IoTDevice.java
│                       │   ├── EdgeNode.java
│                       │   ├── FogNode.java
│                       │   ├── CloudDatacenter.java
│                       │   ├── Task.java
│                       │   ├── NetworkLink.java
│                       │   └── NetworkCondition.java
│                       ├── security/                 # Security models
│                       │   ├── SecurityManager.java
│                       │   ├── AttackType.java
│                       │   ├── CountermeasureType.java
│                       │   ├── AttackSimulation.java
│                       │   └── SecurityMeasure.java
│                       ├── topology/                 # Network topology
│                       │   ├── TopologyManager.java
│                       │   └── DeviceManager.java
│                       └── util/                     # Utilities
│                           ├── LogManager.java
│                           ├── NetworkModel.java
│                           └── TaskManager.java
├── simulation.properties                # Configuration file
├── build_run.bat                        # Windows build and run script
├── build_run.sh                         # Linux build and run script
└── README.md                            # This file
```

## Requirements
- Java Development Kit (JDK) 11 or higher
- No external dependencies required

## Building and Running
### Windows
```
build_run.bat
```

### Linux/macOS
```
chmod +x build_run.sh
./build_run.sh
```

## Configuration
The simulation parameters can be configured in the `simulation.properties` file. Key parameters include:

- Simulation duration
- Number and types of devices
- Network parameters
- Task generation settings
- Security attack and countermeasure settings
- Logging preferences

## Output
The simulation produces the following outputs:
- Console logs showing simulation progress
- Detailed simulation results in the console
- CSV file with metrics in the `results` directory
- Log files in the `logs` directory

## Academic Context
This project was developed for the National College of Ireland (NCI) H9FEC: Fog and Edge Computing module. It implements a proof-of-concept based on recent peer-reviewed research in fog and edge computing security.

## License
This project is for academic purposes only and is not licensed for commercial use.

## Author
[Your Name]
National College of Ireland
Student ID: [Your Student ID]
