package com.fog.eedto.simulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.fog.eedto.algorithm.EEDTOAlgorithm;
import com.fog.eedto.blockchain.BlockchainService;
import com.fog.eedto.model.CloudServer;
import com.fog.eedto.model.Device;
import com.fog.eedto.model.EdgeServer;
import com.fog.eedto.model.IoTDevice;
import com.fog.eedto.model.Task;

/**
 * Simulation class for the EEDTO system.
 * This class orchestrates the simulation of IoT devices, edge servers, and cloud servers,
 * and runs the EEDTO algorithm for task offloading decisions.
 */
public class Simulation {
    private static final Logger logger = LogManager.getLogger(Simulation.class);
    
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
        
        logger.info("Simulation initialized with {} IoT devices, {} edge servers, and {} cloud servers",
                   numIoTDevices, numEdgeServers, numCloudServers);
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
            // IoT devices have limited resources
            double mips = 500 + random.nextDouble() * 500; // 500-1000 MIPS
            int ram = 256 + random.nextInt(768); // 256-1024 MB
            long storage = 1024 + random.nextInt(3072); // 1-4 GB
            double bandwidth = 10 + random.nextDouble() * 40; // 10-50 Mbps
            double energyEfficiency = 100 + random.nextDouble() * 200; // 100-300 MIPS/W
            double batteryCapacity = 10000 + random.nextDouble() * 10000; // 10000-20000 J
            double idlePower = 0.1 + random.nextDouble() * 0.4; // 0.1-0.5 W
            
            IoTDevice iotDevice = new IoTDevice(
                i, "IoT-" + i, mips, ram, storage, bandwidth, energyEfficiency,
                batteryCapacity, idlePower
            );
            
            iotDevices.add(iotDevice);
            logger.debug("Created IoT device: {}", iotDevice);
        }
        
        // Create edge servers
        for (int i = 0; i < numEdgeServers; i++) {
            // Edge servers have moderate resources
            double mips = 5000 + random.nextDouble() * 5000; // 5000-10000 MIPS
            int ram = 4096 + random.nextInt(4096); // 4-8 GB
            long storage = 102400 + random.nextInt(102400); // 100-200 GB
            double bandwidth = 100 + random.nextDouble() * 400; // 100-500 Mbps
            double energyEfficiency = 300 + random.nextDouble() * 200; // 300-500 MIPS/W
            double powerConsumption = 10 + random.nextDouble() * 10; // 10-20 W
            double latency = 5 + random.nextDouble() * 15; // 5-20 ms
            int maxConcurrentTasks = 10 + random.nextInt(10); // 10-20 tasks
            double costPerMI = 0.00001 + random.nextDouble() * 0.00001; // $0.00001-0.00002 per MI
            
            EdgeServer edgeServer = new EdgeServer(
                i, "Edge-" + i, mips, ram, storage, bandwidth, energyEfficiency,
                powerConsumption, latency, maxConcurrentTasks, costPerMI
            );
            
            edgeServers.add(edgeServer);
            logger.debug("Created edge server: {}", edgeServer);
        }
        
        // Create cloud servers
        for (int i = 0; i < numCloudServers; i++) {
            // Cloud servers have high resources
            double mips = 20000 + random.nextDouble() * 30000; // 20000-50000 MIPS
            int ram = 16384 + random.nextInt(16384); // 16-32 GB
            long storage = 1048576 + random.nextInt(1048576); // 1-2 TB
            double bandwidth = 500 + random.nextDouble() * 500; // 500-1000 Mbps
            double energyEfficiency = 500 + random.nextDouble() * 300; // 500-800 MIPS/W
            double powerConsumption = 50 + random.nextDouble() * 50; // 50-100 W
            double latency = 50 + random.nextDouble() * 50; // 50-100 ms
            int maxConcurrentTasks = 100 + random.nextInt(100); // 100-200 tasks
            double costPerMI = 0.000005 + random.nextDouble() * 0.000005; // $0.000005-0.00001 per MI
            double scalingFactor = 1.5 + random.nextDouble(); // 1.5-2.5
            
            CloudServer cloudServer = new CloudServer(
                i, "Cloud-" + i, mips, ram, storage, bandwidth, energyEfficiency,
                powerConsumption, latency, maxConcurrentTasks, costPerMI, scalingFactor
            );
            
            cloudServers.add(cloudServer);
            logger.debug("Created cloud server: {}", cloudServer);
        }
    }
    
    /**
     * Run the simulation
     */
    public void run() {
        logger.info("Starting simulation for {} seconds", simulationEndTime);
        
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
                logger.info("Simulation time: {} / {} seconds", (int) currentTime, (int) simulationEndTime);
                logger.info("Tasks generated: {}, completed: {}, rejected: {}", 
                           totalTasksGenerated, totalTasksCompleted, totalTasksRejected);
            }
        }
        
        // Log final statistics
        logStatistics();
    }
    
    /**
     * Generate tasks for IoT devices
     */
    private void generateTasks() {
        for (IoTDevice iotDevice : iotDevices) {
            // Generate tasks based on task generation rate
            double taskProbability = taskGenerationRate; // Tasks per second
            
            if (random.nextDouble() < taskProbability) {
                Task task = iotDevice.generateTask(taskIdCounter.incrementAndGet(), currentTime);
                iotDevice.addTask(task);
                totalTasksGenerated++;
                
                logger.debug("Generated task {} on device {}", task.getId(), iotDevice.getId());
            }
        }
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
                        
                        logger.debug("Task {} rejected, no suitable device found", task.getId());
                    } else if (targetDevice == iotDevice) {
                        // Execute locally
                        double finishTime = iotDevice.executeTask(task, currentTime);
                        
                        // Update statistics
                        totalTasksCompleted++;
                        totalEnergyConsumed += task.getEnergyConsumed();
                        totalResponseTime += (finishTime - task.getArrivalTime());
                        
                        // Remove task from queue
                        iotDevice.removeTask(task);
                        
                        logger.debug("Task {} executed locally on device {}, finish time: {}", 
                                    task.getId(), iotDevice.getId(), finishTime);
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
                        
                        logger.debug("Task {} offloaded from device {} to {}, finish time: {}", 
                                    task.getId(), iotDevice.getId(), targetDevice.getName(), finishTime);
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
        logger.info("Total tasks generated: {}", totalTasksGenerated);
        logger.info("Total tasks completed: {}", totalTasksCompleted);
        logger.info("Total tasks rejected: {}", totalTasksRejected);
        logger.info("Task completion rate: {}%", 
                   totalTasksGenerated > 0 ? (double) totalTasksCompleted / totalTasksGenerated * 100 : 0);
        logger.info("Average energy consumed per task: {} J", 
                   totalTasksCompleted > 0 ? totalEnergyConsumed / totalTasksCompleted : 0);
        logger.info("Average response time per task: {} s", 
                   totalTasksCompleted > 0 ? totalResponseTime / totalTasksCompleted : 0);
        logger.info("Average execution cost per task: ${}",
                   totalTasksCompleted > 0 ? totalExecutionCost / totalTasksCompleted : 0);
        logger.info("Blockchain size: {} blocks", blockchainService.getBlockchainSize());
        logger.info("Blockchain valid: {}", blockchainService.isChainValid());
        
        // Log EEDTO algorithm statistics
        logger.info("EEDTO algorithm statistics: {}", eedtoAlgorithm);
        
        // Log device statistics
        for (IoTDevice iotDevice : iotDevices) {
            logger.info("IoT device {} statistics: energy consumed: {} J, remaining battery: {}%",
                       iotDevice.getId(), iotDevice.getEnergyConsumed(),
                       iotDevice.getRemainingBattery() / iotDevice.getBatteryCapacity() * 100);
        }
        
        for (EdgeServer edgeServer : edgeServers) {
            logger.info("Edge server {} statistics: energy consumed: {} J",
                       edgeServer.getId(), edgeServer.getEnergyConsumed());
        }
        
        for (CloudServer cloudServer : cloudServers) {
            logger.info("Cloud server {} statistics: energy consumed: {} J",
                       cloudServer.getId(), cloudServer.getEnergyConsumed());
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
