# Fog and Edge Computing Project

## Smart Campus Simulation using PureEdgeSim

This project implements a proof-of-concept simulation of a smart campus environment using the PureEdgeSim framework, based on the paper:

> "PureEdgeSim: A Simulation Framework for Performance Evaluation of Cloud, Edge and Mist Computing Environments" by Charafeddine Mechalikh, Hajer Taktak, and Faouzi Moussa

## Project Overview

This implementation demonstrates a multi-tier architecture for task orchestration in a smart campus environment, utilizing Cloud, Edge (Fog), and Mist computing paradigms. The project focuses on the Fuzzy Decision Tree orchestration algorithm proposed in the paper, which makes intelligent decisions about where to offload computational tasks based on multiple criteria.

### Key Features

- **Multi-tier Architecture**: Implementation of Cloud, Edge (Fog), and Mist computing layers
- **Fuzzy Decision Tree Orchestrator**: Two-stage decision-making process for task offloading
- **Smart Campus Scenario**: Simulation of a university campus with heterogeneous devices
- **Performance Evaluation**: Analysis of latency, energy consumption, resource utilization, and task success rate

## System Architecture

The system architecture consists of four main layers:

1. **IoT Sensors Layer**: The source of data generation
2. **Smart Edge Devices Layer (Mist Computing)**: Includes laptops, smartphones, and other edge devices
3. **Fog Layer**: Edge data centers distributed across the campus
4. **Cloud Layer**: Remote cloud data centers for computationally intensive tasks

The Fuzzy Decision Tree orchestrator makes decisions about where to offload tasks based on:
- Task latency sensitivity
- Fog resources utilization
- Device mobility
- WAN bandwidth
- Energy source (battery-powered vs. wall-powered)

## Setup Instructions

### Prerequisites

- Java Development Kit (JDK) 11 or higher
- Maven 3.6 or higher
- Git (optional)

### Installation

1. Clone the repository (or download and extract the ZIP file):
   ```
   git clone https://github.com/yourusername/fog-edge-computing-project.git
   cd fog-edge-computing-project
   ```

2. Build the project using Maven:
   ```
   mvn clean package
   ```

3. Run the simulation:
   ```
   java -jar target/fog-edge-computing-project-1.0-SNAPSHOT-jar-with-dependencies.jar
   ```

## Configuration

The simulation can be configured by modifying the following files in the `src/main/resources` directory:

- `simulation_parameters.properties`: General simulation parameters
- `applications.xml`: Application types and their characteristics
- `edge_devices.xml`: Edge device types and their specifications
- `edge_datacenters.xml`: Edge data center configurations
- `cloud.xml`: Cloud data center configurations

## Evaluation

The simulation generates the following output files in the `simulation_results` directory:

- `task_results.csv`: Detailed results for each task
- `energy_consumption.csv`: Energy consumption by device
- `resource_utilization.csv`: Resource utilization by device
- `network_usage.csv`: Network usage statistics
- `performance_metrics.csv`: Overall performance metrics

## Project Structure

```
fog-edge-computing-project/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── org/fog/edge/computing/
│   │   │       ├── Main.java
│   │   │       ├── orchestration/
│   │   │       │   ├── CustomOrchestrator.java
│   │   │       │   └── FuzzyDecisionTreeOrchestrator.java
│   │   │       ├── simulation/
│   │   │       │   ├── SimulationManager.java
│   │   │       │   └── SimulationScenario.java
│   │   │       └── utils/
│   │   │           ├── SimulationParameters.java
│   │   │           └── SimulationResults.java
│   │   └── resources/
│   │       ├── applications.xml
│   │       ├── cloud.xml
│   │       ├── edge_datacenters.xml
│   │       ├── edge_devices.xml
│   │       └── simulation_parameters.properties
│   └── test/
│       └── java/
├── pom.xml
└── README.md
```

## References

1. Mechalikh, C., Taktak, H., & Moussa, F. (2020). PureEdgeSim: A Simulation Framework for Performance Evaluation of Cloud, Edge and Mist Computing Environments.

## Author

Student Name
National College of Ireland
MSc in Cloud Computing
H9FEC: Fog and Edge Computing
