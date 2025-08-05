# JCORA-MEC Implementation

A Java-based implementation of the paper "A DRL Agent for Jointly Optimizing Computation Offloading and Resource Allocation in Mobile Edge Computing" (IEEE, 2021).

## Overview

This project implements a proof-of-concept prototype for a Deep Reinforcement Learning (DRL) based approach to jointly optimize computation offloading and resource allocation in Mobile Edge Computing (MEC) environments. The implementation focuses on the following key aspects:

- Task modeling with data size, computational requirements, and deadlines
- IoT device modeling with processing power, energy consumption, and battery capacity
- Edge server modeling with processing power, bandwidth allocation, and energy consumption
- DRL agent implementation using Deep Q-Network (DQN) for decision making
- Simulation environment for evaluating the performance of the DRL agent
- Comprehensive logging and visualization of simulation results

## Requirements

- Java 11 or higher
- Maven 3.6 or higher

## Project Structure

```
JCORA-MEC-Implementation/
├── config/                     # Configuration files
│   └── simulation.properties   # Default simulation configuration
├── src/                        # Source code
│   └── main/
│       └── java/
│           └── org/jcora/mec/
│               ├── config/     # Configuration loading utilities
│               ├── drl/        # DRL agent implementation
│               ├── model/      # Core model classes
│               ├── simulation/ # Simulation environment
│               └── util/       # Logging and visualization utilities
├── output/                     # Generated logs and graphs
├── build.bat                   # Windows build script
├── build.sh                    # Linux build script
├── run.bat                     # Windows run script
├── run.sh                      # Linux run script
└── README.md                   # This file
```

## Building the Project

### Windows

```
build.bat
```

### Linux

```
./build.sh
```

## Running the Simulation

### Windows

```
run.bat [config_file]
```

### Linux

```
./run.sh [config_file]
```

If no configuration file is specified, the default `config/simulation.properties` will be used.

## Configuration

The simulation can be configured by modifying the `config/simulation.properties` file. The following parameters can be adjusted:

- Simulation parameters (duration, time step, task generation probability)
- IoT device parameters (processing power, energy consumption, battery capacity)
- Edge server parameters (processing power, bandwidth, energy consumption)
- DRL agent parameters (state size, action size, gamma, epsilon, etc.)

## Output

The simulation generates the following outputs in the `output` directory:

- CSV files with simulation metrics over time
- CSV files with device and server statistics
- Summary report of the simulation results
- Charts visualizing energy consumption, response time, deadline miss rate, and task completion rate

## Implementation Details

### Task Model

Tasks are modeled with the following attributes:
- Data size (KB)
- Computational requirement (MI)
- Deadline (seconds)
- Status (pending, processing, completed, failed)
- Timing information (arrival time, start time, completion time)

### IoT Device Model

IoT devices are modeled with the following attributes:
- Processing power (MIPS)
- Energy consumption (J/MI)
- Battery capacity (J)
- Task queue

### Edge Server Model

Edge servers are modeled with the following attributes:
- Processing power (MIPS)
- Energy consumption (J/MI)
- Maximum bandwidth (Mbps)
- Maximum connections
- Task queue

### DRL Agent

The DRL agent uses a Deep Q-Network (DQN) to make decisions on:
- Whether to offload a task or process it locally
- Which edge server to offload to
- How much bandwidth to allocate for the task

The agent is trained using experience replay and a target network to stabilize learning.

## References

This implementation is based on the paper:
"A DRL Agent for Jointly Optimizing Computation Offloading and Resource Allocation in Mobile Edge Computing" (IEEE, 2021)

## License

This project is for academic purposes only.
