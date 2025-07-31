package com.fog.eedto.algorithm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fog.eedto.blockchain.BlockchainService;
import com.fog.eedto.model.CloudServer;
import com.fog.eedto.model.Device;
import com.fog.eedto.model.EdgeServer;
import com.fog.eedto.model.IoTDevice;
import com.fog.eedto.model.Task;

/**
 * Implementation of the Energy-Efficient Dynamic Task Offloading (EEDTO) algorithm.
 * This algorithm makes offloading decisions based on energy efficiency, latency, and security
 * considerations in an IoT-Edge-Cloud orchestrated computing environment.
 */
public class EEDTOAlgorithm {
    // Weight factors for the decision-making process
    private final double energyWeight;
    private final double latencyWeight;
    private final double securityWeight;
    
    // Threshold values
    private final double energyThreshold; // Minimum battery level for IoT devices (percentage)
    private final double latencyThreshold; // Maximum acceptable latency in seconds
    private final int securityLevel; // Required security level (1-5)
    
    private final BlockchainService blockchainService;
    
    // Counters for statistics
    private int totalTasks;
    private int localExecutions;
    private int edgeOffloads;
    private int cloudOffloads;
    private int failedOffloads;
    
    /**
     * Constructor for the EEDTOAlgorithm class
     * 
     * @param energyWeight Weight factor for energy efficiency in decision-making
     * @param latencyWeight Weight factor for latency in decision-making
     * @param securityWeight Weight factor for security in decision-making
     * @param energyThreshold Minimum battery level for IoT devices (percentage)
     * @param latencyThreshold Maximum acceptable latency in seconds
     * @param securityLevel Required security level (1-5)
     * @param blockchainService Blockchain service for secure task offloading
     */
    public EEDTOAlgorithm(double energyWeight, double latencyWeight, double securityWeight,
                         double energyThreshold, double latencyThreshold, int securityLevel,
                         BlockchainService blockchainService) {
        // Normalize weights
        double sum = energyWeight + latencyWeight + securityWeight;
        this.energyWeight = energyWeight / sum;
        this.latencyWeight = latencyWeight / sum;
        this.securityWeight = securityWeight / sum;
        
        this.energyThreshold = energyThreshold;
        this.latencyThreshold = latencyThreshold;
        this.securityLevel = securityLevel;
        
        this.blockchainService = blockchainService;
        
        this.totalTasks = 0;
        this.localExecutions = 0;
        this.edgeOffloads = 0;
        this.cloudOffloads = 0;
        this.failedOffloads = 0;
    }
    
    /**
     * Make an offloading decision for a task
     * 
     * @param task Task to be offloaded
     * @param sourceDevice Source device (IoT device)
     * @param edgeServers List of available edge servers
     * @param cloudServers List of available cloud servers
     * @param currentTime Current simulation time
     * @return The selected target device or null if no suitable device is found
     */
    public Device makeOffloadingDecision(Task task, IoTDevice sourceDevice,
                                        List<EdgeServer> edgeServers, List<CloudServer> cloudServers,
                                        double currentTime) {
        totalTasks++;
        
        // Check if the task can be executed locally
        boolean canExecuteLocally = sourceDevice.canExecuteTask(task);
        
        // If the IoT device battery is below threshold, always offload
        boolean forcedOffload = sourceDevice.getRemainingBattery() / sourceDevice.getBatteryCapacity() < energyThreshold;
        
        // If the task is lightweight and can be executed locally and no forced offload, execute locally
        if (canExecuteLocally && !forcedOffload && task.getTaskType() == Task.TaskType.LIGHTWEIGHT) {
            localExecutions++;
            return sourceDevice;
        }
        
        // Calculate scores for all available devices
        Map<Device, Double> deviceScores = new HashMap<>();
        
        // Calculate score for local execution if possible
        if (canExecuteLocally && !forcedOffload) {
            double energyScore = calculateEnergyScore(task, sourceDevice, null);
            double latencyScore = calculateLatencyScore(task, sourceDevice, null);
            double securityScore = calculateSecurityScore(Task.DeviceType.IOT_DEVICE);
            
            double totalScore = (energyWeight * energyScore) + 
                               (latencyWeight * latencyScore) + 
                               (securityWeight * securityScore);
            
            deviceScores.put(sourceDevice, totalScore);
        }
        
        // Calculate scores for edge servers
        for (EdgeServer edgeServer : edgeServers) {
            if (edgeServer.canExecuteTask(task)) {
                double energyScore = calculateEnergyScore(task, sourceDevice, edgeServer);
                double latencyScore = calculateLatencyScore(task, sourceDevice, edgeServer);
                double securityScore = calculateSecurityScore(Task.DeviceType.EDGE_SERVER);
                
                double totalScore = (energyWeight * energyScore) + 
                                   (latencyWeight * latencyScore) + 
                                   (securityWeight * securityScore);
                
                deviceScores.put(edgeServer, totalScore);
            }
        }
        
        // Calculate scores for cloud servers
        for (CloudServer cloudServer : cloudServers) {
            if (cloudServer.canExecuteTask(task)) {
                double energyScore = calculateEnergyScore(task, sourceDevice, cloudServer);
                double latencyScore = calculateLatencyScore(task, sourceDevice, cloudServer);
                double securityScore = calculateSecurityScore(Task.DeviceType.CLOUD_SERVER);
                
                double totalScore = (energyWeight * energyScore) + 
                                   (latencyWeight * latencyScore) + 
                                   (securityWeight * securityScore);
                
                deviceScores.put(cloudServer, totalScore);
            }
        }
        
        // Find the device with the highest score
        Device selectedDevice = null;
        double highestScore = -1;
        
        for (Map.Entry<Device, Double> entry : deviceScores.entrySet()) {
            if (entry.getValue() > highestScore) {
                highestScore = entry.getValue();
                selectedDevice = entry.getKey();
            }
        }
        
        // Update statistics
        if (selectedDevice == null) {
            failedOffloads++;
        } else if (selectedDevice instanceof IoTDevice) {
            localExecutions++;
        } else if (selectedDevice instanceof EdgeServer) {
            edgeOffloads++;
            
            // Record offloading transaction in blockchain
            blockchainService.addTaskOffloadingTransaction(task, sourceDevice, selectedDevice);
        } else if (selectedDevice instanceof CloudServer) {
            cloudOffloads++;
            
            // Record offloading transaction in blockchain
            blockchainService.addTaskOffloadingTransaction(task, sourceDevice, selectedDevice);
        }
        
        return selectedDevice;
    }
    
    /**
     * Calculate the energy score for a task offloading decision
     * 
     * @param task Task to be offloaded
     * @param sourceDevice Source device (IoT device)
     * @param targetDevice Target device (null if executed locally)
     * @return Energy score (0-1, higher is better)
     */
    private double calculateEnergyScore(Task task, IoTDevice sourceDevice, Device targetDevice) {
        // If executed locally
        if (targetDevice == null) {
            double energyConsumption = sourceDevice.calculateEnergyConsumption(task);
            double remainingBatteryPercentage = sourceDevice.getRemainingBattery() / sourceDevice.getBatteryCapacity();
            
            // Lower energy consumption and higher remaining battery is better
            return (1 - (energyConsumption / sourceDevice.getRemainingBattery())) * remainingBatteryPercentage;
        }
        
        // If offloaded, consider transmission energy and target device efficiency
        double transmissionEnergy = calculateTransmissionEnergy(task, sourceDevice, targetDevice);
        double targetEfficiency = targetDevice.getEnergyEfficiency();
        
        // Normalize transmission energy (lower is better)
        double normalizedTransmissionEnergy = Math.min(1.0, transmissionEnergy / sourceDevice.getRemainingBattery());
        
        // Normalize target efficiency (higher is better)
        double maxEfficiency = 1000.0; // Assuming a reasonable maximum efficiency
        double normalizedEfficiency = Math.min(1.0, targetEfficiency / maxEfficiency);
        
        // Combined score: lower transmission energy and higher target efficiency is better
        return (1 - normalizedTransmissionEnergy) * 0.5 + normalizedEfficiency * 0.5;
    }
    
    /**
     * Calculate the latency score for a task offloading decision
     * 
     * @param task Task to be offloaded
     * @param sourceDevice Source device (IoT device)
     * @param targetDevice Target device (null if executed locally)
     * @return Latency score (0-1, higher is better)
     */
    private double calculateLatencyScore(Task task, IoTDevice sourceDevice, Device targetDevice) {
        double responseTime;
        
        // Calculate response time based on execution location
        if (targetDevice == null) {
            // Local execution
            responseTime = task.calculateExecutionTime(sourceDevice.getMips());
        } else {
            // Offloaded execution
            double transmissionTime = sourceDevice.calculateTransmissionTime(task, targetDevice);
            double executionTime;
            
            if (targetDevice instanceof EdgeServer) {
                executionTime = ((EdgeServer) targetDevice).calculateTotalLatency(task);
            } else if (targetDevice instanceof CloudServer) {
                executionTime = ((CloudServer) targetDevice).calculateTotalLatency(task);
            } else {
                executionTime = task.calculateExecutionTime(targetDevice.getMips());
            }
            
            responseTime = transmissionTime + executionTime;
        }
        
        // Check if response time meets the deadline
        boolean meetsDeadline = responseTime <= (task.getDeadline() - task.getArrivalTime());
        
        // Normalize response time (lower is better)
        double normalizedResponseTime = Math.min(1.0, responseTime / latencyThreshold);
        
        // Final score: meets deadline and lower normalized response time is better
        return (meetsDeadline ? 1.0 : 0.5) * (1 - normalizedResponseTime);
    }
    
    /**
     * Calculate the security score for a task offloading decision
     * 
     * @param deviceType Type of device where the task will be executed
     * @return Security score (0-1, higher is better)
     */
    private double calculateSecurityScore(Task.DeviceType deviceType) {
        // Define security levels for different device types
        int deviceSecurityLevel;
        
        switch (deviceType) {
            case IOT_DEVICE:
                deviceSecurityLevel = 2; // IoT devices have lower security
                break;
            case EDGE_SERVER:
                deviceSecurityLevel = 4; // Edge servers have medium security
                break;
            case CLOUD_SERVER:
                deviceSecurityLevel = 5; // Cloud servers have high security
                break;
            default:
                deviceSecurityLevel = 1;
        }
        
        // Calculate security score based on how well the device meets the required security level
        return Math.min(1.0, (double) deviceSecurityLevel / securityLevel);
    }
    
    /**
     * Calculate the energy required for transmitting a task to another device
     * 
     * @param task Task to be transmitted
     * @param sourceDevice Source device
     * @param targetDevice Target device
     * @return Energy consumption in Joules
     */
    private double calculateTransmissionEnergy(Task task, Device sourceDevice, Device targetDevice) {
        // Energy model for data transmission: E = P * T
        // where P is transmission power and T is transmission time
        
        double transmissionPower = 0.1; // Watts (simplified model)
        double transmissionTime = sourceDevice.calculateTransmissionTime(task, targetDevice);
        
        return transmissionPower * transmissionTime;
    }
    
    // Getters for statistics
    public int getTotalTasks() {
        return totalTasks;
    }
    
    public int getLocalExecutions() {
        return localExecutions;
    }
    
    public int getEdgeOffloads() {
        return edgeOffloads;
    }
    
    public int getCloudOffloads() {
        return cloudOffloads;
    }
    
    public int getFailedOffloads() {
        return failedOffloads;
    }
    
    public double getLocalExecutionPercentage() {
        return totalTasks > 0 ? (double) localExecutions / totalTasks * 100 : 0;
    }
    
    public double getEdgeOffloadPercentage() {
        return totalTasks > 0 ? (double) edgeOffloads / totalTasks * 100 : 0;
    }
    
    public double getCloudOffloadPercentage() {
        return totalTasks > 0 ? (double) cloudOffloads / totalTasks * 100 : 0;
    }
    
    public double getFailedOffloadPercentage() {
        return totalTasks > 0 ? (double) failedOffloads / totalTasks * 100 : 0;
    }
    
    @Override
    public String toString() {
        return "EEDTOAlgorithm{" +
                "energyWeight=" + energyWeight +
                ", latencyWeight=" + latencyWeight +
                ", securityWeight=" + securityWeight +
                ", energyThreshold=" + energyThreshold +
                ", latencyThreshold=" + latencyThreshold +
                ", securityLevel=" + securityLevel +
                ", totalTasks=" + totalTasks +
                ", localExecutions=" + localExecutions + " (" + getLocalExecutionPercentage() + "%)" +
                ", edgeOffloads=" + edgeOffloads + " (" + getEdgeOffloadPercentage() + "%)" +
                ", cloudOffloads=" + cloudOffloads + " (" + getCloudOffloadPercentage() + "%)" +
                ", failedOffloads=" + failedOffloads + " (" + getFailedOffloadPercentage() + "%)" +
                '}';
    }
}
