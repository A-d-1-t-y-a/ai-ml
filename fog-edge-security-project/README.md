# Secure Fog Computing Framework

This project implements a lightweight security framework for IoT-Fog-Cloud architecture based on a 2021 research paper. The implementation uses iFogSim to simulate a secure fog computing environment with IoT devices, edge nodes, and fog nodes.

## Project Overview

The secure fog computing framework simulates a hierarchical architecture with:
- IoT devices that generate data with different wireless technologies
- Edge nodes that process data from IoT devices
- Fog nodes that perform advanced analytics on aggregated data
- Security mechanisms including encryption, authentication, and intrusion detection

## Features

- **Hierarchical Topology**: IoT devices connect to edge nodes, which connect to fog nodes
- **Wireless Technology Simulation**: WiFi, BLE, LoRaWAN with different data rates and energy profiles
- **Security Implementation**: AES encryption, authentication, and intrusion detection
- **Energy Consumption Modeling**: Energy usage for data transmission and security operations
- **Performance Metrics**: Processing time, energy consumption, security overhead
- **Big Data Analytics**: Simulated data processing and analytics at fog level
- **Configurable Parameters**: Adjustable simulation parameters via configuration file

## Project Structure

```
fog-edge-security-project/
├── src/
│   └── main/
│       └── java/
│           └── org/
│               └── nci/
│                   └── fogedge/
│                       ├── SecureFogSimulation.java
│                       ├── topology/
│                       │   ├── IoTDevice.java
│                       │   ├── EdgeNode.java
│                       │   └── FogNode.java
│                       ├── security/
│                       │   ├── SecurityManager.java
│                       │   └── SecurityLevel.java
│                       ├── model/
│                       │   └── SimulationResults.java
│                       └── util/
│                           ├── ConfigurationManager.java
│                           ├── DataProcessor.java
│                           └── LoggingUtil.java
├── pom.xml
└── README.md
```

## Dependencies

- **iFogSim**: Fog computing simulation framework
- **CloudSim**: Cloud computing simulation framework
- **BouncyCastle**: Cryptographic operations
- **Log4j**: Logging framework
- **JSON**: Data handling

## Getting Started

### Prerequisites

- Java JDK 8 or higher
- Maven

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/fog-edge-security-project.git
cd fog-edge-security-project
```

2. Install dependencies using Maven:
```
mvn clean install
```

### Running the Simulation

Run the main simulation class:
```
mvn exec:java -Dexec.mainClass="org.nci.fogedge.SecureFogSimulation"
```

## Configuration

The simulation can be configured through the `config.properties` file that is automatically generated on first run. Key configuration parameters include:

- **Simulation duration**
- **Security settings** (enabled/disabled, encryption algorithm, key size)
- **Topology parameters** (number of IoT devices, edge nodes, fog nodes)
- **IoT device settings** (data generation rate, energy capacity)
- **Edge and fog node capabilities** (processing capacity, storage, bandwidth)

## Results

Simulation results are logged to the console and to CSV files for further analysis. The results include:

- Total data generated and processed
- Energy consumption across all nodes
- Processing time and security overhead
- Number of detected attacks
- Energy efficiency metrics
- Data reduction percentage

## Implementation Details

### IoT Devices
IoT devices generate data based on their wireless technology type (WiFi, BLE, LoRaWAN) and can encrypt data before transmission to edge nodes. Energy consumption is modeled for both data transmission and security operations.

### Edge Nodes
Edge nodes receive data from IoT devices, process it (filtering, basic aggregation), and forward relevant data to fog nodes. They implement security measures including decryption and re-encryption.

### Fog Nodes
Fog nodes perform advanced analytics on aggregated data from edge nodes. They have higher processing capabilities and implement the same security measures as edge nodes.

### Security Manager
The Security Manager handles encryption, decryption, authentication, and intrusion detection across the architecture. It uses AES encryption with configurable key sizes and implements a lightweight authentication mechanism.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the paper "A Lightweight Security Framework for IoT-Fog-Cloud Architecture" (2021)
- Uses the iFogSim simulation framework
- Developed for the National College of Ireland (NCI) module H9FEC: Fog and Edge Computing
