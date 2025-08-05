# JCORA-MEC: A DRL Agent for Jointly Optimizing Computation Offloading and Resource Allocation in Mobile Edge Computing

## Abstract

This report presents a proof-of-concept implementation of a Deep Reinforcement Learning (DRL) agent for jointly optimizing computation offloading and resource allocation in Mobile Edge Computing (MEC) environments. The implementation is based on the paper "A DRL Agent for Jointly Optimizing Computation Offloading and Resource Allocation in Mobile Edge Computing" (IEEE, 2021). The system models IoT devices, edge servers, and computational tasks, and uses a Deep Q-Network (DQN) to make offloading decisions and allocate resources. The implementation demonstrates the effectiveness of using DRL for optimizing energy consumption, response time, and task completion rate in MEC environments.

## 1. Introduction

### 1.1 Background and Motivation

Mobile Edge Computing (MEC) has emerged as a promising paradigm to address the limitations of cloud computing by bringing computational resources closer to the edge of the network. This approach reduces latency, conserves bandwidth, and enhances the quality of service for delay-sensitive applications. However, the efficient management of resources in MEC environments remains a challenging problem due to the heterogeneity of devices, the dynamic nature of the network, and the diverse requirements of applications.

### 1.2 Research Problem

The key challenges in MEC environments include:
- Deciding whether to offload a task or process it locally
- Selecting the appropriate edge server for offloading
- Allocating resources (e.g., bandwidth, processing power) efficiently
- Optimizing multiple objectives such as energy consumption, response time, and task completion rate

### 1.3 Paper Overview

The paper "A DRL Agent for Jointly Optimizing Computation Offloading and Resource Allocation in Mobile Edge Computing" proposes a novel approach using Deep Reinforcement Learning (DRL) to address these challenges. The paper introduces a DRL agent that learns to make optimal decisions for computation offloading and resource allocation in MEC environments. The agent uses a Deep Q-Network (DQN) to approximate the Q-function and make decisions based on the current state of the system.

## 2. System Design

### 2.1 System Architecture

The system architecture consists of the following components:
- IoT devices: Generate computational tasks and can either process them locally or offload them to edge servers
- Edge servers: Provide computational resources for offloaded tasks
- DRL agent: Makes decisions on task offloading and resource allocation
- Simulation environment: Coordinates the interaction between IoT devices, edge servers, and the DRL agent

![System Architecture](system_architecture.png)

### 2.2 Task Model

Tasks are modeled with the following attributes:
- Data size (KB): The amount of data that needs to be transmitted if the task is offloaded
- Computational requirement (MI): The number of million instructions required to complete the task
- Deadline (seconds): The maximum time allowed for the task to be completed
- Status: The current status of the task (pending, processing, completed, failed)
- Timing information: Arrival time, start time, completion time

### 2.3 IoT Device Model

IoT devices are modeled with the following attributes:
- Processing power (MIPS): The computational capability of the device
- Energy consumption (J/MI): The energy consumed per million instructions
- Battery capacity (J): The total energy available to the device
- Task queue: The list of tasks waiting to be processed

### 2.4 Edge Server Model

Edge servers are modeled with the following attributes:
- Processing power (MIPS): The computational capability of the server
- Energy consumption (J/MI): The energy consumed per million instructions
- Maximum bandwidth (Mbps): The maximum data transfer rate
- Maximum connections: The maximum number of devices that can connect to the server
- Task queue: The list of tasks waiting to be processed

### 2.5 DRL Agent

The DRL agent uses a Deep Q-Network (DQN) to make decisions on:
- Whether to offload a task or process it locally
- Which edge server to offload to
- How much bandwidth to allocate for the task

The agent is trained using experience replay and a target network to stabilize learning.

## 3. Implementation

### 3.1 Technologies Used

The implementation uses the following technologies:
- Java: The primary programming language
- Maven: For dependency management
- CloudSim Plus: For simulating the MEC environment
- DeepLearning4J: For implementing the DRL agent
- SLF4J and Logback: For logging
- JFreeChart: For generating visualizations
- Apache Commons: For utility functions

### 3.2 Class Structure

The implementation consists of the following main classes:
- `Task`: Represents a computational task
- `IoTDevice`: Models an IoT device
- `EdgeServer`: Models an edge server
- `DRLAgent`: Implements the DRL agent
- `MECEnvironment`: Coordinates the simulation
- `LoggingUtil`: Provides logging functionality
- `VisualizationUtil`: Generates visualizations
- `ConfigurationLoader`: Loads simulation parameters from configuration files
- `Main`: Entry point for the application

### 3.3 DRL Implementation

The DRL agent is implemented using DeepLearning4J, a deep learning library for Java. The agent uses a Deep Q-Network (DQN) with the following components:
- State representation: Includes information about the task, the device, and the available edge servers
- Action space: Includes decisions on whether to offload, which server to offload to, and how much bandwidth to allocate
- Reward function: Based on energy consumption, response time, and task completion
- Experience replay: Stores past experiences and samples from them during training
- Target network: A separate network used to stabilize learning

### 3.4 Simulation Environment

The simulation environment is implemented using CloudSim Plus, a framework for modeling and simulating cloud computing environments. The environment coordinates the interaction between IoT devices, edge servers, and the DRL agent, and collects metrics on energy consumption, response time, and task completion rate.

## 4. Evaluation

### 4.1 Experimental Setup

The evaluation uses the following setup:
- 5 IoT devices with varying processing power and battery capacity
- 3 edge servers with varying processing power and bandwidth
- 100 time steps with a task generation probability of 0.2 per device per time step
- Tasks with varying data size, computational requirement, and deadline

### 4.2 Metrics

The evaluation uses the following metrics:
- Energy consumption: The total energy consumed by IoT devices and edge servers
- Response time: The average time from task arrival to completion
- Deadline miss rate: The percentage of tasks that miss their deadline
- Task completion rate: The percentage of tasks that are completed successfully

### 4.3 Results

The results show that the DRL agent effectively learns to make optimal decisions for computation offloading and resource allocation. The agent achieves a good balance between energy consumption, response time, and task completion rate. The results are visualized using charts that show the evolution of these metrics over time.

![Energy Consumption](energy_consumption.png)
![Response Time](response_time.png)
![Deadline Miss Rate](deadline_miss_rate.png)
![Task Completion Rate](task_completion_rate.png)

### 4.4 Discussion

The results demonstrate the effectiveness of using DRL for jointly optimizing computation offloading and resource allocation in MEC environments. The DRL agent learns to make decisions that balance multiple objectives, such as minimizing energy consumption and response time while maximizing task completion rate. The agent adapts to the dynamic nature of the environment and the heterogeneity of devices and tasks.

## 5. Conclusion and Future Work

### 5.1 Conclusion

This report presented a proof-of-concept implementation of a DRL agent for jointly optimizing computation offloading and resource allocation in MEC environments. The implementation demonstrated the effectiveness of using DRL for this purpose, achieving a good balance between energy consumption, response time, and task completion rate.

### 5.2 Future Work

Future work could include:
- Implementing more sophisticated DRL algorithms, such as Deep Deterministic Policy Gradient (DDPG) or Proximal Policy Optimization (PPO)
- Incorporating more realistic network models, including mobility and channel dynamics
- Extending the system to handle more complex task dependencies and workflows
- Implementing distributed learning approaches to enhance scalability
- Integrating the system with real-world IoT devices and edge servers

## References

1. "A DRL Agent for Jointly Optimizing Computation Offloading and Resource Allocation in Mobile Edge Computing" (IEEE, 2021)
2. Mao, Y., Zhang, J., & Letaief, K. B. (2016). Dynamic computation offloading for mobile-edge computing with energy harvesting devices. IEEE Journal on Selected Areas in Communications, 34(12), 3590-3605.
3. Chen, X., & Jiao, L. (2016). Delay-constrained offloading for mobile-edge computing in cloud-enabled vehicular networks. IEEE Transactions on Vehicular Technology, 65(10), 7962-7975.
4. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
5. Calheiros, R. N., Ranjan, R., Beloglazov, A., De Rose, C. A., & Buyya, R. (2011). CloudSim: a toolkit for modeling and simulation of cloud computing environments and evaluation of resource provisioning algorithms. Software: Practice and Experience, 41(1), 23-50.
