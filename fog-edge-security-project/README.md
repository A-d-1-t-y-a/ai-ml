# Secure Fog Computing Prototype

## Overview
This project implements a proof-of-concept prototype for a secure fog computing architecture based on the research paper "An Overview of Fog Computing and Edge Computing Security and Privacy Issues" (MDPI, 2021). The implementation demonstrates a multi-layer fog computing environment with security countermeasures at each layer.

## Project Structure
The project is organized as a Java Maven project with the following structure:

```
fog-edge-security-project/
├── src/
│   └── main/
│       └── java/
│           └── org/
│               └── nci/
│                   └── fogedge/
│                       ├── SecureFogSimulation.java (Main class)
│                       ├── model/
│                       │   ├── SimulationParameters.java
│                       │   └── SimulationResults.java
│                       ├── security/
│                       │   ├── SecurityManager.java
│                       │   └── SecurityIncident.java
│                       ├── simulation/
│                       │   └── SimulationEngine.java
│                       ├── topology/
│                       │   ├── CloudDatacenter.java
│                       │   ├── EdgeNode.java
│                       │   ├── FogNode.java
│                       │   ├── IoTDevice.java
│                       │   ├── Location.java
│                       │   ├── NetworkTopology.java
│                       │   └── NetworkTopologyBuilder.java
│                       └── utils/
│                           └── ReportGenerator.java
├── pom.xml
└── README.md
```

## Architecture
The prototype implements a four-layer fog computing architecture:

1. **IoT Layer**: IoT devices generate data and apply basic security measures (encryption, authentication).
2. **Edge Layer**: Edge nodes receive data from IoT devices, perform initial processing, and implement security measures like intrusion detection.
3. **Fog Layer**: Fog nodes receive data from edge nodes, perform advanced processing, and implement advanced security measures like blockchain and decoy techniques.
4. **Cloud Layer**: Cloud datacenter receives data from fog nodes for final processing and storage.

## Security Features
The prototype implements the following security countermeasures as discussed in the research paper:

1. **Encryption**: Data is encrypted using AES-256 at the IoT layer.
2. **Authentication**: Multi-tier authentication is implemented at all layers.
3. **Intrusion Detection**: Edge and fog nodes implement intrusion detection to identify security incidents.
4. **Blockchain**: Fog nodes implement blockchain-based security for data integrity.
5. **Decoy Techniques**: Fog nodes implement decoy techniques to mislead attackers.

## Requirements
- Java 11 or higher
- Maven 3.6 or higher

## Building the Project
To build the project, run the following command:

```bash
mvn clean package
```

This will create a JAR file in the `target` directory.

## Running the Simulation
To run the simulation, execute the following command:

```bash
java -jar target/fog-edge-security-project-1.0-SNAPSHOT-jar-with-dependencies.jar
```

## Simulation Output
The simulation generates the following reports in the `reports` directory:

1. **Performance Report**: Contains packet statistics, latency statistics, bandwidth statistics, and energy statistics.
2. **Security Report**: Contains security incident statistics, security incidents by type, security countermeasures effectiveness, and security response times.
3. **Network Report**: Contains network topology information, network connectivity, data transmission statistics, and network efficiency metrics.
4. **Summary Report**: Contains a summary of the simulation results.

## Implementation Details

### Network Topology
The network topology consists of:
- 20 IoT devices
- 5 edge nodes
- 2 fog nodes
- 1 cloud datacenter

The devices and nodes are placed in a 1000m x 1000m area, with connection ranges of 200m for IoT-to-edge and 500m for edge-to-fog.

### Simulation Parameters
The simulation runs for 10 seconds of simulated time, with IoT devices generating data every 0.5 seconds. The forwarding probability is 70% from edge to fog and 30% from fog to cloud.

### Security Parameters
- Encryption: AES-256
- Authentication: Multi-tier authentication
- Intrusion detection: Enabled at edge and fog layers
- Blockchain: Enabled at fog layer
- Decoy techniques: Enabled at fog layer

## Evaluation
The prototype is evaluated based on the following metrics:
1. **Performance**: End-to-end latency, processing time at each layer, bandwidth saved by edge and fog processing.
2. **Security**: Security incidents detected, mitigation success rate, security response time.
3. **Energy Efficiency**: Energy consumed by each layer, total energy consumed.

## References
1. "An Overview of Fog Computing and Edge Computing Security and Privacy Issues" (MDPI, 2021)
2. CloudSim: A Framework for Modeling and Simulation of Cloud Computing Infrastructures and Services

## Author
NCI H9FEC Student

## License
This project is licensed under the MIT License - see the LICENSE file for details.
