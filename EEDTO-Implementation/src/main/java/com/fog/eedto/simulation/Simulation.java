package com.fog.eedto.simulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import java.util.logging.Logger;
import java.util.logging.Level;

import com.fog.eedto.algorithm.EEDTOAlgorithm;
import com.fog.eedto.blockchain.BlockchainService;
import com.fog.eedto.model.CloudServer;
import com.fog.eedto.model.Device;
import com.fog.eedto.model.EdgeServer;
import com.fog.eedto.model.IoTDevice;
import com.fog.eedto.model.Task;
import com.fog.eedto.util.ConfigurationManager;

/**
 * Simulation class for the EEDTO system.
 * This class orchestrates the simulation of IoT devices, edge servers, and cloud servers,
 * and runs the EEDTO algorithm for task offloading decisions.
 */
public class Simulation {
    private static final Logger logger = Logger.getLogger(Simulation.class.getName());
    
    private final List<IoTDevice> iotDevices;
    private final List<EdgeServer> edgeServers;
    private final List<CloudServer> cloudServers;
    private final BlockchainService blockchainService;
    private final EEDTOAlgorithm eedtoAlgorithm;
    private final Random random;
    private final AtomicInteger taskIdCounter;
    
    private double currentTime;
    private double simulationEndTime;
    private double taskGenerationRate; // Tasks per second per IoT device
    
    // Simulation statistics
    private int totalTasksGenerated;
    private int totalTasksCompleted;
    private int totalTasksRejected;
    private double totalEnergyConsumed;
    private double totalResponseTime;
    private double totalExecutionCost;
    
    /**
     * Constructor for the Simulation class
     * 
     * @param numIoTDevices Number of IoT devices in the simulation
     * @param numEdgeServers Number of edge servers in the simulation
     * @param numCloudServers Number of cloud servers in the simulation
     * @param simulationEndTime Simulation end time in seconds
     * @param taskGenerationRate Task generation rate per second per IoT device
     * @param energyWeight Weight factor for energy efficiency in decision-making
     * @param latencyWeight Weight factor for latency in decision-making
     * @param securityWeight Weight factor for security in decision-making
     * @param energyThreshold Minimum battery level for IoT devices (percentage)
     * @param latencyThreshold Maximum acceptable latency in seconds
     * @param securityLevel Required security level (1-5)
     * @param blockchainDifficulty Mining difficulty for the blockchain
     */
    public Simulation(int numIoTDevices, int numEdgeServers, int numCloudServers,
                     double simulationEndTime, double taskGenerationRate,
                     double energyWeight, double latencyWeight, double securityWeight,
                     double energyThreshold, double latencyThreshold, int securityLevel,
                     int blockchainDifficulty) {
        this.iotDevices = new ArrayList<>();
        this.edgeServers = new ArrayList<>();
        this.cloudServers = new ArrayList<>();
        this.random = new Random();
        this.taskIdCounter = new AtomicInteger(0);
        
        this.currentTime = 0;
        this.simulationEndTime = simulationEndTime;
        this.taskGenerationRate = taskGenerationRate;
        
        this.totalTasksGenerated = 0;
        this.totalTasksCompleted = 0;
        this.totalTasksRejected = 0;
        this.totalEnergyConsumed = 0;
        this.totalResponseTime = 0;
        this.totalExecutionCost = 0;
        
        // Create blockchain service
        this.blockchainService = new BlockchainService(blockchainDifficulty);
        
        // Create EEDTO algorithm
        this.eedtoAlgorithm = new EEDTOAlgorithm(
            energyWeight, latencyWeight, securityWeight,
            energyThreshold, latencyThreshold, securityLevel,
            blockchainService
        );
        
        // Initialize devices
        initializeDevices(numIoTDevices, numEdgeServers, numCloudServers);
        
        logger.info(String.format("Simulation initialized with %d IoT devices, %d edge servers, and %d cloud servers",
                   numIoTDevices, numEdgeServers, numCloudServers));
    }
    
    /**
     * Initialize devices for the simulation
     * 
     * @param numIoTDevices Number of IoT devices
     * @param numEdgeServers Number of edge servers
     * @param numCloudServers Number of cloud servers
     */
    private void initializeDevices(int numIoTDevices, int numEdgeServers, int numCloudServers) {
        // Create IoT devices
        for (int i = 0; i < numIoTDevices; i++) {
            // Get device parameters from configuration
            double mipsMin = ConfigurationManager.getDouble("iotDevice.mips.min", 500);
            double mipsMax = ConfigurationManager.getDouble("iotDevice.mips.max", 1000);
            double batteryCapacityMin = ConfigurationManager.getDouble("iotDevice.batteryCapacity.min", 5000);
            double batteryCapacityMax = ConfigurationManager.getDouble("iotDevice.batteryCapacity.max", 10000);
            double transmissionRangeMin = ConfigurationManager.getDouble("iotDevice.transmissionRange.min", 100);
            double transmissionRangeMax = ConfigurationManager.getDouble("iotDevice.transmissionRange.max", 200);
            double bandwidthMin = ConfigurationManager.getDouble("iotDevice.bandwidth.min", 1);
            double bandwidthMax = ConfigurationManager.getDouble("iotDevice.bandwidth.max", 10);
            
            // Generate random values within configured ranges
            double mips = mipsMin + random.nextDouble() * (mipsMax - mipsMin);
            double batteryCapacity = batteryCapacityMin + random.nextDouble() * (batteryCapacityMax - batteryCapacityMin);
            double transmissionRange = transmissionRangeMin + random.nextDouble() * (transmissionRangeMax - transmissionRangeMin);
            double bandwidth = bandwidthMin + random.nextDouble() * (bandwidthMax - bandwidthMin);
            
            IoTDevice iotDevice = new IoTDevice(
                i, "IoT Device " + i, mips, batteryCapacity, transmissionRange, bandwidth
            );
            iotDevices.add(iotDevice);
            logger.fine(String.format("Created IoT device %d: %s", i, iotDevice));
        }
        
        // Create edge servers
        for (int i = 0; i < numEdgeServers; i++) {
            // Get edge server parameters from configuration
            double mipsMin = ConfigurationManager.getDouble("edgeServer.mips.min", 5000);
            double mipsMax = ConfigurationManager.getDouble("edgeServer.mips.max", 10000);
            double transmissionRangeMin = ConfigurationManager.getDouble("edgeServer.transmissionRange.min", 500);
            double transmissionRangeMax = ConfigurationManager.getDouble("edgeServer.transmissionRange.max", 1000);
            double bandwidthMin = ConfigurationManager.getDouble("edgeServer.bandwidth.min", 50);
            double bandwidthMax = ConfigurationManager.getDouble("edgeServer.bandwidth.max", 100);
            double costPerTaskMin = ConfigurationManager.getDouble("edgeServer.costPerTask.min", 0.05);
            double costPerTaskMax = ConfigurationManager.getDouble("edgeServer.costPerTask.max", 0.10);
            
            // Generate random values within configured ranges
            double mips = mipsMin + random.nextDouble() * (mipsMax - mipsMin);
            double transmissionRange = transmissionRangeMin + random.nextDouble() * (transmissionRangeMax - transmissionRangeMin);
            double bandwidth = bandwidthMin + random.nextDouble() * (bandwidthMax - bandwidthMin);
            double costPerTask = costPerTaskMin + random.nextDouble() * (costPerTaskMax - costPerTaskMin);
            
            EdgeServer edgeServer = new EdgeServer(
                i, "Edge Server " + i, mips, transmissionRange, bandwidth, costPerTask
            );
            edgeServers.add(edgeServer);
            logger.fine(String.format("Created edge server %d: %s", i, edgeServer));
        }
        
        // Create cloud servers
        for (int i = 0; i < numCloudServers; i++) {
            // Get cloud server parameters from configuration
            double mipsMin = ConfigurationManager.getDouble("cloudServer.mips.min", 20000);
            double mipsMax = ConfigurationManager.getDouble("cloudServer.mips.max", 50000);
            double bandwidthMin = ConfigurationManager.getDouble("cloudServer.bandwidth.min", 100);
            double bandwidthMax = ConfigurationManager.getDouble("cloudServer.bandwidth.max", 1000);
            double costPerTaskMin = ConfigurationManager.getDouble("cloudServer.costPerTask.min", 0.20);
            double costPerTaskMax = ConfigurationManager.getDouble("cloudServer.costPerTask.max", 0.50);
            
            // Generate random values within configured ranges
            double mips = mipsMin + random.nextDouble() * (mipsMax - mipsMin);
            double bandwidth = bandwidthMin + random.nextDouble() * (bandwidthMax - bandwidthMin);
            double costPerTask = costPerTaskMin + random.nextDouble() * (costPerTaskMax - costPerTaskMin);
            
            CloudServer cloudServer = new CloudServer(
                i, "Cloud Server " + i, mips, bandwidth, costPerTask
            );
            cloudServers.add(cloudServer);
            logger.fine(String.format("Created cloud server %d: %s", i, cloudServer));
        }
    }
    
    /**
     * Run the simulation
     */
    public void run() {
        logger.info(String.format("Starting simulation for %.2f seconds", simulationEndTime));
        
        // Event-driven simulation
        while (currentTime < simulationEndTime) {
            // Generate tasks for IoT devices
            generateTasks();
            
            // Process tasks
            processTasks();
            
            // Mine blockchain blocks periodically
            if ((int) currentTime % 10 == 0 && blockchainService.getPendingTransactionsCount() > 0) {
                blockchainService.minePendingTransactions();
            }
            
            // Advance simulation time
            currentTime += 1;
            
            // Log progress
            if ((int) currentTime % 10 == 0) {
                logger.info(String.format("Simulation time: %d / %d seconds", (int) currentTime, (int) simulationEndTime));
                logger.info(String.format("Tasks generated: %d, completed: %d, rejected: %d", 
                           totalTasksGenerated, totalTasksCompleted, totalTasksRejected));
            }
        }
        
        // Log final statistics
        logStatistics();
    }
    
    /**
     * Generate tasks for IoT devices
     */
    private void generateTasks() {
        // Generate tasks for each IoT device based on task generation rate
        for (IoTDevice iotDevice : iotDevices) {
            // Calculate number of tasks to generate for this time step
            double expectedTasks = taskGenerationRate;
            int numTasks = (int) Math.floor(expectedTasks);
            if (random.nextDouble() < (expectedTasks - numTasks)) {
                numTasks++;
            }
            
            // Generate tasks
            for (int i = 0; i < numTasks; i++) {
                Task task = generateRandomTask(iotDevice);
                iotDevice.addTask(task);
                totalTasksGenerated++;
                logger.fine(String.format("Generated task %d for device %d: %s", task.getId(), iotDevice.getId(), task));
            }
        }
    }
    
    /**
     * Generate a random task for an IoT device
     * 
     * @param sourceDevice Source IoT device
     * @return Generated task
     */
    private Task generateRandomTask(IoTDevice sourceDevice) {
        int taskId = taskIdCounter.incrementAndGet();
        
        // Get task parameters from configuration
        double lightweightProbability = ConfigurationManager.getDouble("task.lightweightProbability", 0.6);
        double mediumProbability = ConfigurationManager.getDouble("task.mediumProbability", 0.3);
        
        // Determine task type based on probabilities
        Task.TaskType taskType;
        double rand = random.nextDouble();
        if (rand < lightweightProbability) {
            taskType = Task.TaskType.LIGHTWEIGHT;
        } else if (rand < lightweightProbability + mediumProbability) {
            taskType = Task.TaskType.MEDIUM;
        } else {
            taskType = Task.TaskType.HEAVYWEIGHT;
        }
        
        // Get task size ranges from configuration based on type
        double minSize, maxSize;
        double minMI, maxMI;
        double minDeadline, maxDeadline;
        
        switch (taskType) {
            case LIGHTWEIGHT:
                minSize = ConfigurationManager.getDouble("task.lightweight.size.min", 10);
                maxSize = ConfigurationManager.getDouble("task.lightweight.size.max", 100);
                minMI = ConfigurationManager.getDouble("task.lightweight.mi.min", 100);
                maxMI = ConfigurationManager.getDouble("task.lightweight.mi.max", 1000);
                minDeadline = ConfigurationManager.getDouble("task.lightweight.deadline.min", 1);
                maxDeadline = ConfigurationManager.getDouble("task.lightweight.deadline.max", 5);
                break;
            case MEDIUM:
                minSize = ConfigurationManager.getDouble("task.medium.size.min", 100);
                maxSize = ConfigurationManager.getDouble("task.medium.size.max", 1000);
                minMI = ConfigurationManager.getDouble("task.medium.mi.min", 1000);
                maxMI = ConfigurationManager.getDouble("task.medium.mi.max", 10000);
                minDeadline = ConfigurationManager.getDouble("task.medium.deadline.min", 5);
                maxDeadline = ConfigurationManager.getDouble("task.medium.deadline.max", 20);
                break;
            case HEAVYWEIGHT:
            default:
                minSize = ConfigurationManager.getDouble("task.heavyweight.size.min", 1000);
                maxSize = ConfigurationManager.getDouble("task.heavyweight.size.max", 10000);
                minMI = ConfigurationManager.getDouble("task.heavyweight.mi.min", 10000);
                maxMI = ConfigurationManager.getDouble("task.heavyweight.mi.max", 100000);
                minDeadline = ConfigurationManager.getDouble("task.heavyweight.deadline.min", 20);
                maxDeadline = ConfigurationManager.getDouble("task.heavyweight.deadline.max", 60);
                break;
        }
        
        // Generate random values within configured ranges
        double size = minSize + random.nextDouble() * (maxSize - minSize); // KB
        double mi = minMI + random.nextDouble() * (maxMI - minMI); // Million Instructions
        double deadline = currentTime + minDeadline + random.nextDouble() * (maxDeadline - minDeadline); // seconds
        
        return new Task(taskId, taskType, size, mi, currentTime, deadline, sourceDevice.getId());
    }
    
    /**
     * Process tasks for all devices
     */
    private void processTasks() {
        // Process tasks for IoT devices
        for (IoTDevice iotDevice : iotDevices) {
            List<Task> taskQueue = new ArrayList<>(iotDevice.getTaskQueue());
            
            for (Task task : taskQueue) {
                if (task.getStatus() == Task.TaskStatus.CREATED) {
                    // Make offloading decision
                    Device targetDevice = eedtoAlgorithm.makeOffloadingDecision(
                        task, iotDevice, edgeServers, cloudServers, currentTime
                    );
                    
                    if (targetDevice == null) {
                        // No suitable device found, reject the task
                        task.setStatus(Task.TaskStatus.REJECTED);
                        iotDevice.removeTask(task);
                        totalTasksRejected++;
                        
                        logger.fine(String.format("Task %d rejected, no suitable device found", task.getId()));
                    } else if (targetDevice == iotDevice) {
                        // Execute locally
                        double finishTime = iotDevice.executeTask(task, currentTime);
                        
                        // Update statistics
                        totalTasksCompleted++;
                        totalEnergyConsumed += task.getEnergyConsumed();
                        totalResponseTime += (finishTime - task.getArrivalTime());
                        
                        // Remove task from queue
                        iotDevice.removeTask(task);
                        
                        logger.fine(String.format("Task %d executed locally on device %d, finish time: %.2f", 
                                    task.getId(), iotDevice.getId(), finishTime));
                    } else {
                        // Offload to target device
                        task.setStatus(Task.TaskStatus.OFFLOADED);
                        
                        // Calculate transmission time
                        double transmissionTime = iotDevice.calculateTransmissionTime(task, targetDevice);
                        
                        // Execute on target device
                        double finishTime;
                        double cost = 0;
                        
                        if (targetDevice instanceof EdgeServer) {
                            EdgeServer edgeServer = (EdgeServer) targetDevice;
                            finishTime = edgeServer.executeTask(task, currentTime + transmissionTime);
                            cost = edgeServer.calculateCost(task);
                        } else if (targetDevice instanceof CloudServer) {
                            CloudServer cloudServer = (CloudServer) targetDevice;
                            finishTime = cloudServer.executeTask(task, currentTime + transmissionTime);
                            cost = cloudServer.calculateCost(task);
                        } else {
                            finishTime = currentTime + transmissionTime + task.calculateExecutionTime(targetDevice.getMips());
                        }
                        
                        // Update statistics
                        totalTasksCompleted++;
                        totalEnergyConsumed += task.getEnergyConsumed();
                        totalResponseTime += (finishTime - task.getArrivalTime());
                        totalExecutionCost += cost;
                        
                        // Remove task from queue
                        iotDevice.removeTask(task);
                        
                        logger.fine(String.format("Task %d offloaded from device %d to %s, finish time: %.2f", 
                                    task.getId(), iotDevice.getId(), targetDevice.getName(), finishTime));
                    }
                }
            }
            
            // Update battery level for idle time
            iotDevice.updateBatteryLevel(1.0, true);
        }
    }
    
    /**
     * Log simulation statistics
     */
    private void logStatistics() {
        logger.info("Simulation completed");
        logger.info(String.format("Total tasks generated: %d", totalTasksGenerated));
        logger.info(String.format("Total tasks completed: %d", totalTasksCompleted));
        logger.info(String.format("Total tasks rejected: %d", totalTasksRejected));
        logger.info(String.format("Task completion rate: %.2f%%", 
                   totalTasksGenerated > 0 ? (double) totalTasksCompleted / totalTasksGenerated * 100 : 0));
        logger.info(String.format("Average energy consumed per task: %.2f J", 
                   totalTasksCompleted > 0 ? totalEnergyConsumed / totalTasksCompleted : 0));
        logger.info(String.format("Average response time per task: %.2f s", 
                   totalTasksCompleted > 0 ? totalResponseTime / totalTasksCompleted : 0));
        logger.info(String.format("Average execution cost per task: $%.2f",
                   totalTasksCompleted > 0 ? totalExecutionCost / totalTasksCompleted : 0));
        logger.info(String.format("Blockchain size: %d blocks", blockchainService.getBlockchainSize()));
        logger.info(String.format("Blockchain valid: %b", blockchainService.isChainValid()));
        
        // Log EEDTO algorithm statistics
        logger.info(String.format("EEDTO algorithm statistics: %s", eedtoAlgorithm));
        
        // Log device statistics
        for (IoTDevice iotDevice : iotDevices) {
            logger.info(String.format("IoT device %d statistics: energy consumed: %.2f J, remaining battery: %.2f%%",
                       iotDevice.getId(), iotDevice.getEnergyConsumed(),
                       iotDevice.getRemainingBattery() / iotDevice.getBatteryCapacity() * 100));
        }
        
        for (EdgeServer edgeServer : edgeServers) {
            logger.info(String.format("Edge server %d statistics: energy consumed: %.2f J",
                       edgeServer.getId(), edgeServer.getEnergyConsumed()));
        }
        
        for (CloudServer cloudServer : cloudServers) {
            logger.info(String.format("Cloud server %d statistics: energy consumed: %.2f J",
                       cloudServer.getId(), cloudServer.getEnergyConsumed()));
        }
    }
    
    /**
     * Get the simulation results
     * 
     * @return SimulationResults object containing the simulation results
     */
    public SimulationResults getResults() {
        return new SimulationResults(
            totalTasksGenerated,
            totalTasksCompleted,
            totalTasksRejected,
            totalEnergyConsumed,
            totalResponseTime,
            totalExecutionCost,
            eedtoAlgorithm.getLocalExecutions(),
            eedtoAlgorithm.getEdgeOffloads(),
            eedtoAlgorithm.getCloudOffloads(),
            eedtoAlgorithm.getFailedOffloads(),
            blockchainService.getBlockchainSize()
        );
    }
}
