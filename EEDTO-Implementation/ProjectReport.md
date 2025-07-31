# EEDTO: Energy-Efficient Dynamic Task Offloading Algorithm
## Project Report

### 1. Introduction

This report presents the implementation of a proof-of-concept simulation for the EEDTO (Energy-Efficient Dynamic Task Offloading) algorithm for blockchain-enabled IoT-Edge-Cloud orchestrated computing, based on the IEEE IoT Journal paper from 2021. The implementation demonstrates how IoT devices can efficiently offload computational tasks to edge or cloud servers based on energy efficiency, latency, and security considerations, with blockchain integration for secure task offloading transactions.

### 2. Paper Summary

The paper "EEDTO: An Energy-Efficient Dynamic Task Offloading Algorithm for Blockchain-Enabled IoT-Edge-Cloud Orchestrated Computing" addresses the challenge of optimizing task offloading decisions in IoT environments. The key contributions of the paper include:

1. A three-tier architecture (IoT-Edge-Cloud) for efficient task processing
2. An energy-efficient dynamic task offloading algorithm that considers multiple factors
3. Integration of blockchain technology for secure and transparent task offloading
4. A comprehensive evaluation framework for assessing the performance of the algorithm

The EEDTO algorithm makes offloading decisions based on a weighted combination of energy efficiency, latency, and security factors, adapting to the specific requirements of different applications and environments.

### 3. System Architecture

Our implementation follows the three-tier architecture proposed in the paper:

#### 3.1 IoT Layer
- Resource-constrained devices that generate computational tasks
- Limited processing power, memory, storage, and battery capacity
- Need to offload computationally intensive tasks to conserve energy

#### 3.2 Edge Layer
- Medium-capacity servers located closer to IoT devices
- Lower latency compared to cloud servers
- Suitable for time-sensitive applications

#### 3.3 Cloud Layer
- High-capacity servers with greater processing power
- Higher latency due to geographical distance
- Suitable for computationally intensive tasks with less strict timing requirements

#### 3.4 Blockchain Component
- Ensures secure and transparent task offloading
- Records all offloading decisions in an immutable ledger
- Provides a trust mechanism between devices and servers

### 4. Implementation Details

#### 4.1 Core Components

1. **Device Models**:
   - Abstract `Device` class with common properties and methods
   - `IoTDevice` class for resource-constrained devices
   - `EdgeServer` class for medium-capacity servers
   - `CloudServer` class for high-capacity servers

2. **Task Model**:
   - Represents computational tasks with properties like length, input/output size, deadline
   - Includes methods for calculating execution time, response time, and deadline checking

3. **Blockchain Component**:
   - `BlockchainService` for managing the blockchain
   - `Block` class for representing blocks in the blockchain
   - `TaskTransaction` class for recording task offloading transactions

4. **EEDTO Algorithm**:
   - Implements the decision-making logic for task offloading
   - Considers energy efficiency, latency, and security factors
   - Adapts to different weights for different application requirements

5. **Simulation Framework**:
   - `Simulation` class for orchestrating the simulation
   - `SimulationResults` class for storing and analyzing results
   - Visualization capabilities for generating charts and graphs

#### 4.2 Key Algorithms

1. **Task Offloading Decision Algorithm**:
   ```
   function makeOffloadingDecision(task, sourceDevice, edgeServers, cloudServers):
       if canExecuteLocally(task, sourceDevice) and task is lightweight:
           return sourceDevice
       
       Calculate scores for all available devices:
           - Local execution score
           - Edge server scores
           - Cloud server scores
       
       Select device with highest score
       
       Record offloading transaction in blockchain
       
       return selectedDevice
   ```

2. **Score Calculation**:
   ```
   score = (energyWeight * energyScore) + (latencyWeight * latencyScore) + (securityWeight * securityScore)
   ```

3. **Blockchain Mining**:
   ```
   function mineBlock(difficulty):
       Set target = string of 'difficulty' zeros
       
       while hash does not start with target:
           increment nonce
           recalculate hash
       
       Add block to blockchain
   ```

### 5. Experimental Setup

The simulation was run with the following configurations:

1. **Baseline**: Equal weights for energy, latency, and security
   - Energy weight: 0.33
   - Latency weight: 0.33
   - Security weight: 0.33

2. **Energy-Focused**: Higher weight for energy efficiency
   - Energy weight: 0.6
   - Latency weight: 0.2
   - Security weight: 0.2

3. **Latency-Focused**: Higher weight for latency/response time
   - Energy weight: 0.2
   - Latency weight: 0.6
   - Security weight: 0.2

4. **Security-Focused**: Higher weight for security
   - Energy weight: 0.2
   - Latency weight: 0.2
   - Security weight: 0.6

Each configuration was run with the following parameters:
- 10 IoT devices
- 3 edge servers
- 1 cloud server
- 300 seconds simulation time
- 0.1 tasks per second per IoT device
- Energy threshold: 20%
- Latency threshold: 5 seconds
- Security level: 3
- Blockchain difficulty: 2

### 6. Results and Analysis

#### 6.1 Task Distribution

The simulation results show how tasks are distributed among local execution, edge offloading, and cloud offloading for different configurations:

- **Baseline**: Balanced distribution with slight preference for edge offloading
- **Energy-Focused**: More local executions for lightweight tasks, edge offloading for medium tasks
- **Latency-Focused**: More edge offloading to minimize response time
- **Security-Focused**: More cloud offloading due to higher security level

#### 6.2 Energy Consumption

Energy consumption varies across different configurations:

- **Energy-Focused**: Lowest average energy consumption per task
- **Security-Focused**: Highest energy consumption due to more cloud offloading
- **Baseline** and **Latency-Focused**: Moderate energy consumption

#### 6.3 Response Time

Response time also varies across configurations:

- **Latency-Focused**: Lowest average response time
- **Security-Focused**: Highest response time due to more cloud offloading
- **Baseline** and **Energy-Focused**: Moderate response time

#### 6.4 Blockchain Integration

The blockchain component successfully recorded all task offloading transactions, ensuring transparency and security in the offloading process. The blockchain size grew proportionally to the number of offloaded tasks.

### 7. Conclusion

The implementation successfully demonstrates the EEDTO algorithm's ability to make intelligent task offloading decisions based on energy efficiency, latency, and security considerations. The blockchain integration provides an additional layer of security and transparency.

The simulation results show that the algorithm can adapt to different application requirements by adjusting the weights of the decision factors. For energy-constrained applications, the energy-focused configuration provides the best results in terms of energy consumption. For time-sensitive applications, the latency-focused configuration minimizes response time.

### 8. Future Work

Several areas for future improvement have been identified:

1. **Dynamic Weight Adjustment**: Implement a mechanism to dynamically adjust the weights based on the current state of the system
2. **Machine Learning Integration**: Use machine learning techniques to predict task characteristics and optimize offloading decisions
3. **More Realistic Network Model**: Implement a more realistic network model with variable latency and bandwidth
4. **Smart Contract Integration**: Extend the blockchain component with smart contracts for automated service level agreements
5. **Distributed Consensus**: Implement a distributed consensus mechanism for the blockchain component

### 9. References

1. "EEDTO: An Energy-Efficient Dynamic Task Offloading Algorithm for Blockchain-Enabled IoT-Edge-Cloud Orchestrated Computing" (IEEE IoT Journal, 2021)
2. CloudSim Plus: A modern Java 8 framework for modeling and simulation of cloud computing infrastructures and services
3. Web3j: A lightweight Java library for integration with Ethereum blockchains
4. JFreeChart: A Java chart library for generating various types of charts
