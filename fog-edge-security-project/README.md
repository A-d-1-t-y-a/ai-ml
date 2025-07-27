# Fog and Edge Computing Security Simulation

## Overview
This project implements a simulation framework for fog and edge computing security based on the paper:
**"An Overview of Fog Computing and Edge Computing Security and Privacy Issues"** (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226).

The simulation models IoT devices, edge nodes, and fog nodes in a hierarchical network topology, and simulates various security attacks and countermeasures as described in the paper. It collects and analyzes performance, security, and energy consumption metrics to evaluate the trade-offs between security and performance in fog and edge computing environments.

## Project Structure
```
fog-edge-security-project/
├── src/main/java/org/nci/fogedge/
│   ├── model/                  # Simulation configuration and results classes
│   ├── security/               # Security-related classes and enums
│   ├── topology/               # IoT, edge, fog node classes and network topology
│   ├── util/                   # Utility classes for configuration, logging, etc.
│   ├── FogEdgeSecuritySimulation.java  # Main simulation class
│   └── SimulationDemo.java     # Standalone demo with interactive mode
├── src/main/resources/         # Configuration files
├── build_run.bat               # Windows build and run script
├── build_run.sh                # Linux/Mac build and run script
└── pom.xml                     # Maven project file
```

## Features
- Modular Java implementation of fog and edge computing architecture
- Simulation of IoT devices with various wireless technologies
- Edge and fog node processing with data reduction
- Security attack simulation targeting different layers:
  - IoT layer (physical tampering, malware injection, battery draining)
  - Edge layer (DoS, man-in-the-middle, authentication bypass)
  - Fog layer (data theft, privilege escalation, VM escape)
  - Network layer (eavesdropping, traffic analysis, routing attacks)
- Security countermeasures with configurable security levels
- Comprehensive metrics collection and analysis:
  - Performance metrics (processing time, data reduction)
  - Security metrics (attack detection and prevention rates)
  - Energy consumption metrics (including security overhead)

## Requirements
- Java 8 or higher
- Maven (optional, for dependency management)

## Dependencies
- Log4j 2 (logging)
- BouncyCastle (cryptography)
- org.json (JSON processing)
- JUnit 4 (testing)

## Building and Running
### Using the provided scripts
#### Windows:
```
build_run.bat
```

#### Linux/Mac:
```
chmod +x build_run.sh
./build_run.sh
```

### Using Maven:
```
mvn clean compile
mvn exec:java -Dexec.mainClass="org.nci.fogedge.SimulationDemo"
```

### Interactive Mode
The simulation can be run in interactive mode to customize parameters:
```
java -cp "target/classes;lib/*" org.nci.fogedge.SimulationDemo --interactive
```
or
```
build_run.bat --interactive
```

## Configuration
The simulation can be configured through the `simulation.properties` file in the resources directory. Key configuration options include:
- Number of IoT devices, edge nodes, and fog nodes
- Security level (LOW, MEDIUM, HIGH, VERY_HIGH)
- Attack types to simulate
- Security enablement at different layers

## Results
The simulation generates comprehensive results including:
- Performance metrics (data generation, processing, reduction)
- Security metrics (attack detection and prevention rates)
- Energy consumption metrics
- Analysis of security-performance trade-offs

Results can be saved to a file in interactive mode.

## Paper Implementation
This simulation implements the key concepts from the Sensors 2021 paper, focusing on:
1. The hierarchical architecture of IoT-Edge-Fog computing
2. Security threats at different layers as identified in the paper
3. Security countermeasures and their impact on performance and energy consumption
4. Analysis of trade-offs between security, performance, and energy efficiency

## License
This project is for educational purposes as part of the National College of Ireland (NCI) H9FEC: Fog and Edge Computing module.

## Author
Student project for NCI H9FEC: Fog and Edge Computing, 2023-2024.
