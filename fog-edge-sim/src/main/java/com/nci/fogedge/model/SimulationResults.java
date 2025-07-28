package com.nci.fogedge.model;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class for tracking and calculating simulation results.
 * Collects metrics during simulation and provides methods for analysis.
 */
public class SimulationResults {
    // Task metrics
    private int totalTasksGenerated;
    private int completedTasks;
    private int failedTasks;
    private List<Double> taskExecutionTimes;
    private List<Double> taskWaitingTimes;
    private List<Double> taskResponseTimes;
    
    // Resource utilization metrics
    private List<Double> cpuUtilization;
    private List<Double> ramUtilization;
    private List<Double> storageUtilization;
    private List<Double> batteryConsumption;
    
    // Network metrics
    private List<Double> networkLatency;
    private List<Double> networkBandwidth;
    private int totalPacketsSent;
    private int totalPacketsLost;
    
    // Security metrics
    private int totalSecurityAttacks;
    private int detectedAttacks;
    private int mitigatedAttacks;
    private Map<String, Integer> attacksByType;
    private int totalSecurityMeasures;
    private Map<String, Integer> measuresByType;
    
    // Energy metrics
    private double totalEnergyConsumption;
    private double averageEnergyPerTask;
    
    // Device metrics
    private int totalDevices;
    private int activeDevices;
    private int compromisedDevices;
    
    /**
     * Creates a new SimulationResults object.
     */
    public SimulationResults() {
        // Initialize task metrics
        totalTasksGenerated = 0;
        completedTasks = 0;
        failedTasks = 0;
        taskExecutionTimes = new ArrayList<>();
        taskWaitingTimes = new ArrayList<>();
        taskResponseTimes = new ArrayList<>();
        
        // Initialize resource utilization metrics
        cpuUtilization = new ArrayList<>();
        ramUtilization = new ArrayList<>();
        storageUtilization = new ArrayList<>();
        batteryConsumption = new ArrayList<>();
        
        // Initialize network metrics
        networkLatency = new ArrayList<>();
        networkBandwidth = new ArrayList<>();
        totalPacketsSent = 0;
        totalPacketsLost = 0;
        
        // Initialize security metrics
        totalSecurityAttacks = 0;
        detectedAttacks = 0;
        mitigatedAttacks = 0;
        attacksByType = new HashMap<>();
        totalSecurityMeasures = 0;
        measuresByType = new HashMap<>();
        
        // Initialize energy metrics
        totalEnergyConsumption = 0.0;
        averageEnergyPerTask = 0.0;
        
        // Initialize device metrics
        totalDevices = 0;
        activeDevices = 0;
        compromisedDevices = 0;
    }
    
    /**
     * Increments the total number of tasks generated.
     */
    public void incrementTotalTasksGenerated() {
        totalTasksGenerated++;
    }
    
    /**
     * Increments the number of completed tasks.
     */
    public void incrementCompletedTasks() {
        completedTasks++;
    }
    
    /**
     * Increments the number of failed tasks.
     */
    public void incrementFailedTasks() {
        failedTasks++;
    }
    
    /**
     * Adds a task execution time measurement.
     * 
     * @param time Task execution time in milliseconds
     */
    public void addTaskExecutionTime(double time) {
        taskExecutionTimes.add(time);
    }
    
    /**
     * Adds a task waiting time measurement.
     * 
     * @param time Task waiting time in milliseconds
     */
    public void addTaskWaitingTime(double time) {
        taskWaitingTimes.add(time);
    }
    
    /**
     * Adds a task response time measurement.
     * 
     * @param time Task response time in milliseconds
     */
    public void addTaskResponseTime(double time) {
        taskResponseTimes.add(time);
    }
    
    /**
     * Adds a CPU utilization measurement.
     * 
     * @param utilization CPU utilization percentage (0-100)
     */
    public void addCpuUtilization(double utilization) {
        cpuUtilization.add(utilization);
    }
    
    /**
     * Adds a RAM utilization measurement.
     * 
     * @param utilization RAM utilization percentage (0-100)
     */
    public void addRamUtilization(double utilization) {
        ramUtilization.add(utilization);
    }
    
    /**
     * Adds a storage utilization measurement.
     * 
     * @param utilization Storage utilization percentage (0-100)
     */
    public void addStorageUtilization(double utilization) {
        storageUtilization.add(utilization);
    }
    
    /**
     * Adds a battery consumption measurement.
     * 
     * @param consumption Battery consumption in mAh
     */
    public void addBatteryConsumption(double consumption) {
        batteryConsumption.add(consumption);
        totalEnergyConsumption += consumption;
    }
    
    /**
     * Adds a network latency measurement.
     * 
     * @param latency Network latency in milliseconds
     */
    public void addNetworkLatency(double latency) {
        networkLatency.add(latency);
    }
    
    /**
     * Adds a network bandwidth measurement.
     * 
     * @param bandwidth Network bandwidth in Mbps
     */
    public void addNetworkBandwidth(double bandwidth) {
        networkBandwidth.add(bandwidth);
    }
    
    /**
     * Sets the average network latency.
     * 
     * @param latency Average network latency in milliseconds
     */
    public void setAverageNetworkLatency(double latency) {
        if (networkLatency.isEmpty()) {
            networkLatency.add(latency);
        } else {
            networkLatency.set(0, latency);
        }
    }
    
    /**
     * Sets the average network bandwidth.
     * 
     * @param bandwidth Average network bandwidth in Mbps
     */
    public void setAverageNetworkBandwidth(double bandwidth) {
        if (networkBandwidth.isEmpty()) {
            networkBandwidth.add(bandwidth);
        } else {
            networkBandwidth.set(0, bandwidth);
        }
    }
    
    /**
     * Increments the total number of packets sent.
     */
    public void incrementTotalPacketsSent() {
        totalPacketsSent++;
    }
    
    /**
     * Increments the total number of packets lost.
     */
    public void incrementTotalPacketsLost() {
        totalPacketsLost++;
    }
    
    /**
     * Increments the total number of security attacks.
     */
    public void incrementTotalSecurityAttacks() {
        totalSecurityAttacks++;
    }
    
    /**
     * Increments the number of detected attacks.
     */
    public void incrementDetectedAttacks() {
        detectedAttacks++;
    }
    
    /**
     * Increments the number of mitigated attacks.
     */
    public void incrementMitigatedAttacks() {
        mitigatedAttacks++;
    }
    
    /**
     * Increments the count for a specific attack type.
     * 
     * @param attackType Type of attack
     */
    public void incrementAttacksByType(String attackType) {
        attacksByType.put(attackType, attacksByType.getOrDefault(attackType, 0) + 1);
    }
    
    /**
     * Increments the total number of security measures applied.
     */
    public void incrementTotalSecurityMeasures() {
        totalSecurityMeasures++;
    }
    
    /**
     * Increments the count for a specific security measure type.
     * 
     * @param measureType Type of security measure
     */
    public void incrementMeasuresByType(String measureType) {
        measuresByType.put(measureType, measuresByType.getOrDefault(measureType, 0) + 1);
    }
    
    /**
     * Updates the total number of devices in the simulation.
     * 
     * @param count Total number of devices
     */
    public void setTotalDevices(int count) {
        totalDevices = count;
    }
    
    /**
     * Updates the number of active devices in the simulation.
     * 
     * @param count Number of active devices
     */
    public void setActiveDevices(int count) {
        activeDevices = count;
    }
    
    /**
     * Updates the number of compromised devices in the simulation.
     * 
     * @param count Number of compromised devices
     */
    public void setCompromisedDevices(int count) {
        compromisedDevices = count;
    }
    
    /**
     * Calculates the average task execution time.
     * 
     * @return Average task execution time in milliseconds
     */
    public double getAverageTaskExecutionTime() {
        if (taskExecutionTimes.isEmpty()) {
            return 0.0;
        }
        return taskExecutionTimes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average task waiting time.
     * 
     * @return Average task waiting time in milliseconds
     */
    public double getAverageTaskWaitingTime() {
        if (taskWaitingTimes.isEmpty()) {
            return 0.0;
        }
        return taskWaitingTimes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average task response time.
     * 
     * @return Average task response time in milliseconds
     */
    public double getAverageTaskResponseTime() {
        if (taskResponseTimes.isEmpty()) {
            return 0.0;
        }
        return taskResponseTimes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average CPU utilization.
     * 
     * @return Average CPU utilization percentage
     */
    public double getAverageCpuUtilization() {
        if (cpuUtilization.isEmpty()) {
            return 0.0;
        }
        return cpuUtilization.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average RAM utilization.
     * 
     * @return Average RAM utilization percentage
     */
    public double getAverageRamUtilization() {
        if (ramUtilization.isEmpty()) {
            return 0.0;
        }
        return ramUtilization.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average storage utilization.
     * 
     * @return Average storage utilization percentage
     */
    public double getAverageStorageUtilization() {
        if (storageUtilization.isEmpty()) {
            return 0.0;
        }
        return storageUtilization.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average network latency.
     * 
     * @return Average network latency in milliseconds
     */
    public double getAverageNetworkLatency() {
        if (networkLatency.isEmpty()) {
            return 0.0;
        }
        return networkLatency.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the average network bandwidth.
     * 
     * @return Average network bandwidth in Mbps
     */
    public double getAverageNetworkBandwidth() {
        if (networkBandwidth.isEmpty()) {
            return 0.0;
        }
        return networkBandwidth.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
    
    /**
     * Calculates the packet loss rate.
     * 
     * @return Packet loss rate as a percentage
     */
    public double getPacketLossRate() {
        if (totalPacketsSent == 0) {
            return 0.0;
        }
        return (double) totalPacketsLost / totalPacketsSent * 100.0;
    }
    
    /**
     * Calculates the attack detection rate.
     * 
     * @return Attack detection rate as a percentage
     */
    public double getAttackDetectionRate() {
        if (totalSecurityAttacks == 0) {
            return 0.0;
        }
        return (double) detectedAttacks / totalSecurityAttacks * 100.0;
    }
    
    /**
     * Calculates the attack mitigation rate.
     * 
     * @return Attack mitigation rate as a percentage
     */
    public double getAttackMitigationRate() {
        if (detectedAttacks == 0) {
            return 0.0;
        }
        return (double) mitigatedAttacks / detectedAttacks * 100.0;
    }
    
    /**
     * Calculates the task success rate.
     * 
     * @return Task success rate as a percentage
     */
    public double getTaskSuccessRate() {
        if (totalTasksGenerated == 0) {
            return 0.0;
        }
        return (double) completedTasks / totalTasksGenerated * 100.0;
    }
    
    /**
     * Calculates the task failure rate.
     * 
     * @return Task failure rate as a percentage
     */
    public double getTaskFailureRate() {
        if (totalTasksGenerated == 0) {
            return 0.0;
        }
        return (double) failedTasks / totalTasksGenerated * 100.0;
    }
    
    /**
     * Calculates the average energy consumption per task.
     * 
     * @return Average energy consumption per task in mAh
     */
    public double getAverageEnergyPerTask() {
        if (completedTasks == 0) {
            return 0.0;
        }
        return totalEnergyConsumption / completedTasks;
    }
    
    /**
     * Calculates the device compromise rate.
     * 
     * @return Device compromise rate as a percentage
     */
    public double getDeviceCompromiseRate() {
        if (totalDevices == 0) {
            return 0.0;
        }
        return (double) compromisedDevices / totalDevices * 100.0;
    }
    
    /**
     * Calculates the device activity rate.
     * 
     * @return Device activity rate as a percentage
     */
    public double getDeviceActivityRate() {
        if (totalDevices == 0) {
            return 0.0;
        }
        return (double) activeDevices / totalDevices * 100.0;
    }
    
    /**
     * Exports the simulation results to a CSV file.
     * 
     * @param filePath Path to the output CSV file
     * @return True if the export was successful, false otherwise
     */
    public boolean exportToCSV(String filePath) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            // Write header
            writer.println("Metric,Value");
            
            // Write task metrics
            writer.println("Total Tasks Generated," + totalTasksGenerated);
            writer.println("Completed Tasks," + completedTasks);
            writer.println("Failed Tasks," + failedTasks);
            writer.println("Task Success Rate (%)," + String.format("%.2f", getTaskSuccessRate()));
            writer.println("Task Failure Rate (%)," + String.format("%.2f", getTaskFailureRate()));
            writer.println("Average Task Execution Time (ms)," + String.format("%.2f", getAverageTaskExecutionTime()));
            writer.println("Average Task Waiting Time (ms)," + String.format("%.2f", getAverageTaskWaitingTime()));
            writer.println("Average Task Response Time (ms)," + String.format("%.2f", getAverageTaskResponseTime()));
            
            // Write resource utilization metrics
            writer.println("Average CPU Utilization (%)," + String.format("%.2f", getAverageCpuUtilization()));
            writer.println("Average RAM Utilization (%)," + String.format("%.2f", getAverageRamUtilization()));
            writer.println("Average Storage Utilization (%)," + String.format("%.2f", getAverageStorageUtilization()));
            writer.println("Total Energy Consumption (mAh)," + String.format("%.2f", totalEnergyConsumption));
            writer.println("Average Energy Per Task (mAh)," + String.format("%.2f", getAverageEnergyPerTask()));
            
            // Write network metrics
            writer.println("Average Network Latency (ms)," + String.format("%.2f", getAverageNetworkLatency()));
            writer.println("Average Network Bandwidth (Mbps)," + String.format("%.2f", getAverageNetworkBandwidth()));
            writer.println("Total Packets Sent," + totalPacketsSent);
            writer.println("Total Packets Lost," + totalPacketsLost);
            writer.println("Packet Loss Rate (%)," + String.format("%.2f", getPacketLossRate()));
            
            // Write security metrics
            writer.println("Total Security Attacks," + totalSecurityAttacks);
            writer.println("Detected Attacks," + detectedAttacks);
            writer.println("Mitigated Attacks," + mitigatedAttacks);
            writer.println("Attack Detection Rate (%)," + String.format("%.2f", getAttackDetectionRate()));
            writer.println("Attack Mitigation Rate (%)," + String.format("%.2f", getAttackMitigationRate()));
            writer.println("Total Security Measures," + totalSecurityMeasures);
            
            // Write device metrics
            writer.println("Total Devices," + totalDevices);
            writer.println("Active Devices," + activeDevices);
            writer.println("Compromised Devices," + compromisedDevices);
            writer.println("Device Activity Rate (%)," + String.format("%.2f", getDeviceActivityRate()));
            writer.println("Device Compromise Rate (%)," + String.format("%.2f", getDeviceCompromiseRate()));
            
            return true;
        } catch (IOException e) {
            System.err.println("Error exporting simulation results to CSV: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Returns a string representation of the simulation results.
     * 
     * @return String representation of the simulation results
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Simulation Results ===\n\n");
        
        // Task metrics
        sb.append("--- Task Metrics ---\n");
        sb.append("Total Tasks Generated: ").append(totalTasksGenerated).append("\n");
        sb.append("Completed Tasks: ").append(completedTasks).append("\n");
        sb.append("Failed Tasks: ").append(failedTasks).append("\n");
        sb.append("Task Success Rate: ").append(String.format("%.2f%%", getTaskSuccessRate())).append("\n");
        sb.append("Task Failure Rate: ").append(String.format("%.2f%%", getTaskFailureRate())).append("\n");
        sb.append("Average Task Execution Time: ").append(String.format("%.2f ms", getAverageTaskExecutionTime())).append("\n");
        sb.append("Average Task Waiting Time: ").append(String.format("%.2f ms", getAverageTaskWaitingTime())).append("\n");
        sb.append("Average Task Response Time: ").append(String.format("%.2f ms", getAverageTaskResponseTime())).append("\n\n");
        
        // Resource utilization metrics
        sb.append("--- Resource Utilization Metrics ---\n");
        sb.append("Average CPU Utilization: ").append(String.format("%.2f%%", getAverageCpuUtilization())).append("\n");
        sb.append("Average RAM Utilization: ").append(String.format("%.2f%%", getAverageRamUtilization())).append("\n");
        sb.append("Average Storage Utilization: ").append(String.format("%.2f%%", getAverageStorageUtilization())).append("\n");
        sb.append("Total Energy Consumption: ").append(String.format("%.2f mAh", totalEnergyConsumption)).append("\n");
        sb.append("Average Energy Per Task: ").append(String.format("%.2f mAh", getAverageEnergyPerTask())).append("\n\n");
        
        // Network metrics
        sb.append("--- Network Metrics ---\n");
        sb.append("Average Network Latency: ").append(String.format("%.2f ms", getAverageNetworkLatency())).append("\n");
        sb.append("Average Network Bandwidth: ").append(String.format("%.2f Mbps", getAverageNetworkBandwidth())).append("\n");
        sb.append("Total Packets Sent: ").append(totalPacketsSent).append("\n");
        sb.append("Total Packets Lost: ").append(totalPacketsLost).append("\n");
        sb.append("Packet Loss Rate: ").append(String.format("%.2f%%", getPacketLossRate())).append("\n\n");
        
        // Security metrics
        sb.append("--- Security Metrics ---\n");
        sb.append("Total Security Attacks: ").append(totalSecurityAttacks).append("\n");
        sb.append("Detected Attacks: ").append(detectedAttacks).append("\n");
        sb.append("Mitigated Attacks: ").append(mitigatedAttacks).append("\n");
        sb.append("Attack Detection Rate: ").append(String.format("%.2f%%", getAttackDetectionRate())).append("\n");
        sb.append("Attack Mitigation Rate: ").append(String.format("%.2f%%", getAttackMitigationRate())).append("\n");
        sb.append("Total Security Measures: ").append(totalSecurityMeasures).append("\n");
        
        // Attack types breakdown
        sb.append("Attack Types Breakdown:\n");
        for (Map.Entry<String, Integer> entry : attacksByType.entrySet()) {
            sb.append("  - ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        
        // Security measures breakdown
        sb.append("Security Measures Breakdown:\n");
        for (Map.Entry<String, Integer> entry : measuresByType.entrySet()) {
            sb.append("  - ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }
        sb.append("\n");
        
        // Device metrics
        sb.append("--- Device Metrics ---\n");
        sb.append("Total Devices: ").append(totalDevices).append("\n");
        sb.append("Active Devices: ").append(activeDevices).append("\n");
        sb.append("Compromised Devices: ").append(compromisedDevices).append("\n");
        sb.append("Device Activity Rate: ").append(String.format("%.2f%%", getDeviceActivityRate())).append("\n");
        sb.append("Device Compromise Rate: ").append(String.format("%.2f%%", getDeviceCompromiseRate())).append("\n");
        
        return sb.toString();
    }
    
    // Getters for direct access to metrics
    
    public int getTotalTasksGenerated() {
        return totalTasksGenerated;
    }
    
    public int getCompletedTasks() {
        return completedTasks;
    }
    
    public int getFailedTasks() {
        return failedTasks;
    }
    
    public List<Double> getTaskExecutionTimes() {
        return new ArrayList<>(taskExecutionTimes);
    }
    
    public List<Double> getTaskWaitingTimes() {
        return new ArrayList<>(taskWaitingTimes);
    }
    
    public List<Double> getTaskResponseTimes() {
        return new ArrayList<>(taskResponseTimes);
    }
    
    public int getTotalSecurityAttacks() {
        return totalSecurityAttacks;
    }
    
    public int getDetectedAttacks() {
        return detectedAttacks;
    }
    
    public int getMitigatedAttacks() {
        return mitigatedAttacks;
    }
    
    public Map<String, Integer> getAttacksByType() {
        return new HashMap<>(attacksByType);
    }
    
    public int getTotalSecurityMeasures() {
        return totalSecurityMeasures;
    }
    
    public Map<String, Integer> getMeasuresByType() {
        return new HashMap<>(measuresByType);
    }
    
    public double getTotalEnergyConsumption() {
        return totalEnergyConsumption;
    }
    
    public int getTotalDevices() {
        return totalDevices;
    }
    
    public int getActiveDevices() {
        return activeDevices;
    }
    
    public int getCompromisedDevices() {
        return compromisedDevices;
    }
}
