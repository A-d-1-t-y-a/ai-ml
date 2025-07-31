# TODG: Distributed Task Offloading With Delay Guarantees for Edge Computing

This project is an implementation of the TODG (Task Offloading with Delay Guarantees) algorithm for edge computing environments, based on the paper published in IEEE Transactions on Parallel and Distributed Systems (2021).

## Project Overview

The TODG algorithm addresses the challenge of task offloading in IoT-Edge computing environments with delay guarantees. It provides a distributed online algorithm for making offloading decisions that consider:

- Task deadlines and computational requirements
- Network conditions and channel dynamics
- Edge server processing capabilities and current load
- Energy consumption of IoT devices

This implementation simulates a network of IoT devices generating computational tasks and edge servers processing these tasks, with stochastic communication channels between them.

## Features

- Simulation of IoT devices generating tasks with varying computational requirements and deadlines
- Modeling of edge servers with different processing capabilities
- Stochastic communication channels with dynamic bandwidth and interference
- Implementation of the TODG algorithm for distributed task offloading decisions
- Comprehensive metrics collection and visualization
- Cross-platform execution (Windows and Linux)

## Requirements

- Java 11 or higher
- Apache Maven 3.6 or higher
- At least 2GB of RAM for simulation execution

## Project Structure

```
TODG-Implementation/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── org/todg/simulation/
│   │   │       ├── algorithm/
│   │   │       │   └── TODGAlgorithm.java
│   │   │       ├── metrics/
│   │   │       │   └── MetricsCollector.java
│   │   │       ├── model/
│   │   │       │   ├── Channel.java
│   │   │       │   ├── EdgeServer.java
│   │   │       │   ├── IoTDevice.java
│   │   │       │   └── Task.java
│   │   │       ├── util/
│   │   │       │   └── SimulationConfig.java
│   │   │       ├── Main.java
│   │   │       └── TODGSimulator.java
│   │   └── resources/
│   │       └── simulation.properties
│   └── test/
│       └── java/
│           └── org/todg/simulation/
├── output/
├── todg.bat
├── todg.sh
├── pom.xml
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/TODG-Implementation.git
   cd TODG-Implementation
   ```

2. Build and run the project:
   - On Windows:
     ```
     todg.bat [build|run|both]
     ```
   - On Linux:
     ```
     chmod +x todg.sh
     ./todg.sh [build|run|both]
     ```

   Options:
   - `build`: Only build the project
   - `run`: Only run the simulation (requires previous build)
   - `both`: Build and run (default if no option specified)

2. To use a custom configuration, modify the `src/main/resources/simulation.properties` file.

## Configuration

The simulation can be configured by editing the `simulation.properties` file. Key parameters include:

- `simulation.duration`: Total simulation time in seconds
- `device.count`: Number of IoT devices in the simulation
- `server.count`: Number of edge servers
- `algorithm.alpha`, `algorithm.beta`, `algorithm.gamma`: Weights for delay, energy, and load balancing in the utility function
- Various parameters for IoT devices, edge servers, channels, and tasks

## Output

The simulation generates the following outputs in the `output` directory:

1. Charts (PNG format):
   - Task generation, offloading, completion, and failure over time
   - Energy consumption over time
   - Average delay and completion rate
   - Server utilization
   - Summary charts comparing offloaded vs. local processing

2. Data files:
   - `simulation_metrics.csv`: Raw metrics data for further analysis
   - `simulation_summary.txt`: Summary of simulation results

## References

This implementation is based on the paper:
"TODG: Distributed Task Offloading With Delay Guarantees for Edge Computing" published in IEEE Transactions on Parallel and Distributed Systems (2021).

## License

This project is for educational purposes only.
