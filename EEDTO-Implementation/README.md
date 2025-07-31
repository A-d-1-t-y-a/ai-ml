# EEDTO: Energy-Efficient Dynamic Task Offloading

This project implements a proof-of-concept simulation for the EEDTO (Energy-Efficient Dynamic Task Offloading) algorithm for blockchain-enabled IoT-Edge-Cloud orchestrated computing, based on the IEEE IoT Journal paper from 2021.

## Overview

The EEDTO algorithm provides an energy-efficient approach to dynamically offload computational tasks from IoT devices to edge servers or cloud servers, considering factors such as:

- Energy consumption
- Response time/latency
- Security requirements
- Resource availability
- Task deadlines

The implementation includes a blockchain component to secure task offloading transactions and ensure transparency in the offloading process.

## Architecture

The system follows a three-tier architecture:

1. **IoT Layer**: Resource-constrained devices that generate computational tasks
2. **Edge Layer**: Medium-capacity servers located closer to IoT devices with lower latency
3. **Cloud Layer**: High-capacity servers with higher latency but greater processing power

## Features

- Dynamic task offloading based on energy efficiency, latency, and security
- Blockchain integration for secure task offloading transactions
- Simulation of IoT devices, edge servers, and cloud servers
- Energy consumption modeling
- Response time calculation
- Comprehensive logging
- Visualization of simulation results

## Requirements

- Java 11 or higher
- Maven 3.6 or higher

## Dependencies

- CloudSim Plus 7.3.0 (Simulation framework)
- Web3j 4.9.5 (Blockchain functionality)
- Log4j 2.17.2 (Logging)
- JFreeChart 1.5.3 (Visualization)
- Apache Commons Lang 3.12.0 (Utilities)
- JUnit 4.13.2 (Testing)

## Project Structure

```
EEDTO-Implementation/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── fog/
│   │   │           └── eedto/
│   │   │               ├── algorithm/
│   │   │               │   └── EEDTOAlgorithm.java
│   │   │               ├── blockchain/
│   │   │               │   ├── Block.java
│   │   │               │   ├── BlockchainService.java
│   │   │               │   └── TaskTransaction.java
│   │   │               ├── model/
│   │   │               │   ├── CloudServer.java
│   │   │               │   ├── Device.java
│   │   │               │   ├── EdgeServer.java
│   │   │               │   ├── IoTDevice.java
│   │   │               │   └── Task.java
│   │   │               ├── simulation/
│   │   │               │   ├── Simulation.java
│   │   │               │   └── SimulationResults.java
│   │   │               └── Main.java
│   │   └── resources/
│   │       └── log4j2.xml
│   └── test/
│       └── java/
│           └── com/
│               └── fog/
│                   └── eedto/
├── logs/
├── output/
├── pom.xml
├── build.bat
├── build.sh
├── run.bat
├── run.sh
├── .gitignore
└── README.md
```

## Building and Running

### On Windows

1. Build the project:
   ```
   build.bat
   ```

2. Run the simulation:
   ```
   run.bat
   ```

### On Linux

1. Build the project:
   ```
   ./build.sh
   ```

2. Run the simulation:
   ```
   ./run.sh
   ```

## Simulation Parameters

The simulation runs with the following configurations:

1. **Baseline**: Equal weights for energy, latency, and security
2. **Energy-Focused**: Higher weight for energy efficiency
3. **Latency-Focused**: Higher weight for latency/response time
4. **Security-Focused**: Higher weight for security

## Output

The simulation generates the following outputs:

1. **Logs**: Detailed logs of the simulation process in the `logs` directory
2. **Visualizations**: Charts and graphs in the `output` directory, including:
   - Task distribution (local execution vs. edge/cloud offloading)
   - Energy consumption
   - Response time
   - Comparative analysis of different configurations

## References

This implementation is based on the paper:

"EEDTO: An Energy-Efficient Dynamic Task Offloading Algorithm for Blockchain-Enabled IoT-Edge-Cloud Orchestrated Computing" (IEEE IoT Journal, 2021)

## License

This project is for educational purposes only.
