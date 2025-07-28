package com.nci.fogedge.model;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class for storing and analyzing simulation results.
 * Collects metrics throughout the simulation and provides methods for analysis.
 */
public class SimulationResults {
    // Simulation metrics
    private long totalSimulationTime;
    
    // Task metrics
    private int totalTasksGenerated;
    private int totalTasksCompleted;
    private int totalTasksFailed;
    private int tasksExecutedOnIoT;
    private int tasksExecutedOnEdge;
    private int tasksExecutedOnFog;
    private int tasksExecutedOnCloud;
    
    // Network metrics
    private double totalDataTransferred;
    private List<Double> networkLatencies;
    private double averageNetworkLatency;
    private int networkCongestionEvents;
    private int packetLossEvents;
    
    // Security metrics
    private int totalAttackAttempts;
    private int successfulAttacks;
    private int detectedAttacks;
    private int mitigatedAttacks;
    private Map<String, Integer> attacksByType;
    
    // Energy metrics
    private double totalEnergyConsumed;
    private double ioTEnergyConsumed;
    private double edgeEnergyConsumed;
    private double fogEnergyConsumed;
    private double cloudEnergyConsumed;
    
    /**
     * Constructor initializes all metrics
     */
    public SimulationResults() {
        // Initialize task metrics
        totalTasksGenerated = 0;
        totalTasksCompleted = 0;
        totalTasksFailed = 0;
        tasksExecutedOnIoT = 0;
        tasksExecutedOnEdge = 0;
        tasksExecutedOnFog = 0;
        tasksExecutedOnCloud = 0;
        
        // Initialize network metrics
        totalDataTransferred = 0.0;
        networkLatencies = new ArrayList<>();
        averageNetworkLatency = 0.0;
        networkCongestionEvents = 0;
        packetLossEvents = 0;
        
        // Initialize security metrics
        totalAttackAttempts = 0;
        successfulAttacks = 0;
        detectedAttacks = 0;
        mitigatedAttacks = 0;
        attacksByType = new HashMap<>();
        
        // Initialize energy metrics
        totalEnergyConsumed = 0.0;
        ioTEnergyConsumed = 0.0;
        edgeEnergyConsumed = 0.0;
        fogEnergyConsumed = 0.0;
        cloudEnergyConsumed = 0.0;
    }
    
    /**
     * Sets the total simulation time
     * @param totalSimulationTime Total simulation time in milliseconds
     */
    public void setTotalSimulationTime(long totalSimulationTime) {
        this.totalSimulationTime = totalSimulationTime;
    }
    
    /**
     * Increments the total tasks generated count
     */
    public void incrementTasksGenerated() {
        totalTasksGenerated++;
    }
    
    /**
     * Increments the total tasks generated count
     * This method is kept for backward compatibility
     */
    public void incrementTotalTasksGenerated() {
        incrementTasksGenerated();
    }
    
    /**
     * Increments the total tasks completed count
     */
    public void incrementTasksCompleted() {
        totalTasksCompleted++;
    }
    
    /**
     * Increments the total tasks completed count
     * This method is kept for backward compatibility
     */
    public void incrementCompletedTasks() {
        incrementTasksCompleted();
    }
    
    /**
     * Increments the total tasks failed count
     */
    public void incrementTasksFailed() {
        totalTasksFailed++;
    }
    
    /**
     * Increments the total tasks failed count
     * This method is kept for backward compatibility
     */
    public void incrementFailedTasks() {
        incrementTasksFailed();
    }
    
    /**
     * Increments the tasks executed on IoT count
     */
    public void incrementTasksExecutedOnIoT() {
        tasksExecutedOnIoT++;
    }
    
    /**
     * Increments the tasks executed on Edge count
     */
    public void incrementTasksExecutedOnEdge() {
        tasksExecutedOnEdge++;
    }
    
    /**
     * Increments the tasks executed on Fog count
     */
    public void incrementTasksExecutedOnFog() {
        tasksExecutedOnFog++;
    }
    
    /**
     * Increments the tasks executed on Cloud count
     */
    public void incrementTasksExecutedOnCloud() {
        tasksExecutedOnCloud++;
    }
    
    /**
     * Adds to the total data transferred
     * @param dataSize Data size in MB
     */
    public void addDataTransferred(double dataSize) {
        totalDataTransferred += dataSize;
    }
    
    /**
     * Adds a network latency measurement
     * @param latency Latency in milliseconds
     */
    public void addNetworkLatency(double latency) {
        networkLatencies.add(latency);
    }
    
    /**
     * Increments the network congestion events count
     */
    public void incrementNetworkCongestionEvents() {
        networkCongestionEvents++;
    }
    
    /**
     * Increments the packet loss events count
     */
    public void incrementPacketLossEvents() {
        packetLossEvents++;
    }
    
    /**
     * Increments the total attack attempts count
     */
    public void incrementAttackAttempts() {
        totalAttackAttempts++;
    }
    
    /**
     * Increments the successful attacks count
     */
    public void incrementSuccessfulAttacks() {
        successfulAttacks++;
    }
    
    /**
     * Increments the detected attacks count
     */
    public void incrementDetectedAttacks() {
        detectedAttacks++;
    }
    
    /**
     * Increments the mitigated attacks count
     */
    public void incrementMitigatedAttacks() {
        mitigatedAttacks++;
    }
    
    /**
     * Records an attack by type
     * @param attackType The type of attack
     */
    public void recordAttackByType(String attackType) {
        attacksByType.put(attackType, attacksByType.getOrDefault(attackType, 0) + 1);
    }
    
    /**
     * Adds to the total energy consumed
     * @param energy Energy in mWh
     */
    public void addTotalEnergyConsumed(double energy) {
        totalEnergyConsumed += energy;
    }
    
    /**
     * Adds to the IoT energy consumed
     * @param energy Energy in mWh
     */
    public void addIoTEnergyConsumed(double energy) {
        ioTEnergyConsumed += energy;
    }
    
    /**
     * Adds to the Edge energy consumed
     * @param energy Energy in mWh
     */
    public void addEdgeEnergyConsumed(double energy) {
        edgeEnergyConsumed += energy;
    }
    
    /**
     * Adds to the Fog energy consumed
     * @param energy Energy in mWh
     */
    public void addFogEnergyConsumed(double energy) {
        fogEnergyConsumed += energy;
    }
    
    /**
     * Adds to the Cloud energy consumed
     * @param energy Energy in mWh
     */
    public void addCloudEnergyConsumed(double energy) {
        cloudEnergyConsumed += energy;
    }
    
    /**
     * Calculates final metrics from collected data
     */
    public void calculateFinalMetrics() {
        // Calculate average network latency from collected measurements
        if (!networkLatencies.isEmpty()) {
            double sum = 0;
            for (double latency : networkLatencies) {
                sum += latency;
            }
            averageNetworkLatency = sum / networkLatencies.size();
        }
    }
    
    /**
     * Initialize the results object
     */
    public void initialize() {
        // Reset all metrics to initial values
        totalTasksGenerated = 0;
        totalTasksCompleted = 0;
        totalTasksFailed = 0;
        tasksExecutedOnIoT = 0;
        tasksExecutedOnEdge = 0;
        tasksExecutedOnFog = 0;
        tasksExecutedOnCloud = 0;
        
        totalDataTransferred = 0.0;
        networkLatencies = new ArrayList<>();
        averageNetworkLatency = 0.0;
        networkCongestionEvents = 0;
        packetLossEvents = 0;
        
        totalAttackAttempts = 0;
        successfulAttacks = 0;
        detectedAttacks = 0;
        mitigatedAttacks = 0;
        attacksByType = new HashMap<>();
        
        totalEnergyConsumed = 0.0;
        ioTEnergyConsumed = 0.0;
        edgeEnergyConsumed = 0.0;
        fogEnergyConsumed = 0.0;
        cloudEnergyConsumed = 0.0;
    }
    
    /**
     * Set the total number of ticks the simulation ran for
     * @param totalTicks Total number of ticks
     */
    public void setTotalTicks(int totalTicks) {
        // Store this information if needed
    }
    
    /**
     * Set the average resource utilization
     * @param avgUtilization Average resource utilization
     */
    public void setAverageResourceUtilization(double avgUtilization) {
        // Store this information if needed
    }
    
    /**
     * Set the average energy level
     * @param avgEnergyLevel Average energy level
     */
    public void setAverageEnergyLevel(double avgEnergyLevel) {
        // Store this information if needed
    }
    
    /**
     * Set the average network bandwidth
     * @param avgBandwidth Average network bandwidth in Mbps
     */
    public void setAverageNetworkBandwidth(double avgBandwidth) {
        // Store the average network bandwidth
        // This would be used for reporting and analysis
    }
    
    /**
     * Set the average network latency
     * @param avgLatency Average network latency in milliseconds
     */
    public void setAverageNetworkLatency(double avgLatency) {
        this.averageNetworkLatency = avgLatency;
    }
    
    /**
     * Set the number of compromised devices
     * @param compromisedCount Number of compromised devices
     */
    public void setCompromisedDeviceCount(int compromisedCount) {
        // Store this information if needed
    }
    
    /**
     * Get the total number of tasks
     * @return Total number of tasks
     */
    public int getTotalTasksCount() {
        return totalTasksGenerated;
    }
    
    /**
     * Get the total number of tasks generated
     * @return Total number of tasks generated
     */
    public int getTotalTasksGenerated() {
        return totalTasksGenerated;
    }
    
    /**
     * Get the number of completed tasks
     * @return Number of completed tasks
     */
    public int getCompletedTasksCount() {
        return totalTasksCompleted;
    }
    
    /**
     * Get the total number of tasks completed
     * @return Total number of tasks completed
     */
    public int getTotalTasksCompleted() {
        return totalTasksCompleted;
    }
    
    /**
     * Get the number of failed tasks
     * @return Number of failed tasks
     */
    public int getFailedTasksCount() {
        return totalTasksFailed;
    }
    
    /**
     * Get the total number of tasks failed
     * @return Total number of tasks failed
     */
    public int getTotalTasksFailed() {
        return totalTasksFailed;
    }
    
    /**
     * Get the number of offloaded tasks
     * @return Number of offloaded tasks
     */
    public int getOffloadedTasksCount() {
        // This is the sum of tasks executed on Edge, Fog, and Cloud
        return tasksExecutedOnEdge + tasksExecutedOnFog + tasksExecutedOnCloud;
    }
    
    /**
     * Increments the offloaded tasks count
     */
    public void incrementOffloadedTasks() {
        // This is tracked separately in the TaskManager
        // The actual count is calculated as the sum of tasks on Edge, Fog, and Cloud
    }
    
    /**
     * Adds a task execution time measurement
     * @param executionTime Execution time in milliseconds
     */
    public void addTaskExecutionTime(long executionTime) {
        // Implementation would depend on how task execution times are tracked
    }
    
    /**
     * Adds a task waiting time measurement
     * @param waitingTime Waiting time in milliseconds
     */
    public void addTaskWaitingTime(long waitingTime) {
        // Implementation would depend on how task waiting times are tracked
    }
    
    /**
     * Adds a task response time measurement
     * @param responseTime Response time in milliseconds
     */
    public void addTaskResponseTime(long responseTime) {
        // Implementation would depend on how task response times are tracked
    }
    
    /**
     * Get the number of tasks executed on IoT devices
     * @return Number of tasks executed on IoT devices
     */
    public int getTasksExecutedOnIoT() {
        return tasksExecutedOnIoT;
    }
    
    /**
     * Get the number of tasks executed on Edge nodes
     * @return Number of tasks executed on Edge nodes
     */
    public int getTasksExecutedOnEdge() {
        return tasksExecutedOnEdge;
    }
    
    /**
     * Get the number of tasks executed on Fog nodes
     * @return Number of tasks executed on Fog nodes
     */
    public int getTasksExecutedOnFog() {
        return tasksExecutedOnFog;
    }
    
    /**
     * Get the number of tasks executed on Cloud
     * @return Number of tasks executed on Cloud
     */
    public int getTasksExecutedOnCloud() {
        return tasksExecutedOnCloud;
    }
    
    /**
     * Get the total data transferred
     * @return Total data transferred in MB
     */
    public double getTotalDataTransferred() {
        return totalDataTransferred;
    }
    
    /**
     * Get the average network latency
     * @return Average network latency in ms
     */
    public double getAverageNetworkLatency() {
        return averageNetworkLatency;
    }
    
    /**
     * Get the number of network congestion events
     * @return Number of network congestion events
     */
    public int getNetworkCongestionEvents() {
        return networkCongestionEvents;
    }
    
    /**
     * Get the number of packet loss events
     * @return Number of packet loss events
     */
    public int getPacketLossEvents() {
        return packetLossEvents;
    }
    
    /**
     * Get the total number of attack attempts
     * @return Total number of attack attempts
     */
    public int getTotalAttackAttempts() {
        return totalAttackAttempts;
    }
    
    /**
     * Get the number of successful attacks
     * @return Number of successful attacks
     */
    public int getSuccessfulAttacks() {
        return successfulAttacks;
    }
    
    /**
     * Get the number of detected attacks
     * @return Number of detected attacks
     */
    public int getDetectedAttacks() {
        return detectedAttacks;
    }
    
    /**
     * Get the number of mitigated attacks
     * @return Number of mitigated attacks
     */
    public int getMitigatedAttacks() {
        return mitigatedAttacks;
    }
    
    /**
     * Get the total energy consumed
     * @return Total energy consumed in mWh
     */
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    /**
     * Get the IoT energy consumed
     * @return IoT energy consumed in mWh
     */
    public double getIoTEnergyConsumed() {
        return ioTEnergyConsumed;
    }
    
    /**
     * Get the Edge energy consumed
     * @return Edge energy consumed in mWh
     */
    public double getEdgeEnergyConsumed() {
        return edgeEnergyConsumed;
    }
    
    /**
     * Get the Fog energy consumed
     * @return Fog energy consumed in mWh
     */
    public double getFogEnergyConsumed() {
        return fogEnergyConsumed;
    }
    
    /**
     * Get the Cloud energy consumed
     * @return Cloud energy consumed in mWh
     */
    public double getCloudEnergyConsumed() {
        return cloudEnergyConsumed;
    }
    
    /**
     * Get the total simulation time
     * @return Total simulation time in milliseconds
     */
    public long getTotalSimulationTime() {
        return totalSimulationTime;
    }
    
    /**
     * Set the task completion rate
     * @param completionRate Task completion rate
     */
    public void setTaskCompletionRate(double completionRate) {
        // Store this information if needed
    }
    
    /**
     * Calculate the average task execution time
     */
    public void calculateAverageTaskExecutionTime() {
        // Implementation would depend on how task execution times are tracked
    }
    
    /**
     * Set the average network bandwidth
     * @param avgBandwidth Average network bandwidth in Mbps
     */
    public void setAverageNetworkBandwidth(double avgBandwidth) {
        // Store the average network bandwidth for reporting and analysis
    }
    
    /**
     * Set the average network latency
     * @param avgLatency Average network latency in milliseconds
     */
    public void setAverageNetworkLatency(double avgLatency) {
        this.averageNetworkLatency = avgLatency;
    }
    
    /**
     * Set the active device count
     * @param activeCount Number of active devices
     */
    public void setActiveDeviceCount(int activeCount) {
        // Store this information if needed
    }
    
    /**
     * Set the inactive device count
     * @param inactiveCount Number of inactive devices
     */
    public void setInactiveDeviceCount(int inactiveCount) {
        // Store this information if needed
    }
    
    /**
     * Adds a task execution time measurement
     * @param executionTime Execution time in ticks
     */
    public void addTaskExecutionTime(int executionTime) {
        // Store task execution time for metrics calculation
    }
    
    /**
     * Adds a task waiting time measurement
     * @param waitingTime Waiting time in ticks
     */
    public void addTaskWaitingTime(int waitingTime) {
        // Store task waiting time for metrics calculation
    }
    
    /**
     * Adds a task response time measurement
     * @param responseTime Response time in ticks
     */
    public void addTaskResponseTime(int responseTime) {
        // Store task response time for metrics calculation
    }
    
    /**
     * Saves the results to a CSV file
     * @param filePath The path to save the file to
     * @return True if the file was saved successfully, false otherwise
     */
    public boolean saveToFile(String filePath) {
        try (FileWriter writer = new FileWriter(filePath)) {
            // Write header
            writer.write("Metric,Value\n");
            
            // Write task metrics
            writer.write("Total Tasks Generated," + totalTasksGenerated + "\n");
            writer.write("Total Tasks Completed," + totalTasksCompleted + "\n");
            writer.write("Total Tasks Failed," + totalTasksFailed + "\n");
            writer.write("Tasks Executed on IoT," + tasksExecutedOnIoT + "\n");
            writer.write("Tasks Executed on Edge," + tasksExecutedOnEdge + "\n");
            writer.write("Tasks Executed on Fog," + tasksExecutedOnFog + "\n");
            writer.write("Tasks Executed on Cloud," + tasksExecutedOnCloud + "\n");
            
            // Write network metrics
            writer.write("Total Data Transferred (MB)," + totalDataTransferred + "\n");
            writer.write("Average Network Latency (ms)," + averageNetworkLatency + "\n");
            writer.write("Network Congestion Events," + networkCongestionEvents + "\n");
            writer.write("Packet Loss Events," + packetLossEvents + "\n");
            
            // Write security metrics
            writer.write("Total Attack Attempts," + totalAttackAttempts + "\n");
            writer.write("Successful Attacks," + successfulAttacks + "\n");
            writer.write("Detected Attacks," + detectedAttacks + "\n");
            writer.write("Mitigated Attacks," + mitigatedAttacks + "\n");
            
            // Write attack types
            writer.write("\nAttack Types,Count\n");
            for (Map.Entry<String, Integer> entry : attacksByType.entrySet()) {
                writer.write(entry.getKey() + "," + entry.getValue() + "\n");
            }
            
            // Write energy metrics
            writer.write("\nEnergy Metrics\n");
            writer.write("Total Energy Consumed (mWh)," + totalEnergyConsumed + "\n");
            writer.write("IoT Energy Consumed (mWh)," + ioTEnergyConsumed + "\n");
            writer.write("Edge Energy Consumed (mWh)," + edgeEnergyConsumed + "\n");
            writer.write("Fog Energy Consumed (mWh)," + fogEnergyConsumed + "\n");
            writer.write("Cloud Energy Consumed (mWh)," + cloudEnergyConsumed + "\n");
            
            return true;
        } catch (IOException e) {
            System.err.println("Error saving results to file: " + e.getMessage());
            return false;
        }
    }
    
    // Getters for all metrics
    
    public long getTotalSimulationTime() {
        return totalSimulationTime;
    }
    
    public int getTotalTasksGenerated() {
        return totalTasksGenerated;
    }
    
    public int getTotalTasksCompleted() {
        return totalTasksCompleted;
    }
    
    public int getTotalTasksFailed() {
        return totalTasksFailed;
    }
    
    public int getTasksExecutedOnIoT() {
        return tasksExecutedOnIoT;
    }
    
    public int getTasksExecutedOnEdge() {
        return tasksExecutedOnEdge;
    }
    
    public int getTasksExecutedOnFog() {
        return tasksExecutedOnFog;
    }
    
    public int getTasksExecutedOnCloud() {
        return tasksExecutedOnCloud;
    }
    
    public double getTotalDataTransferred() {
        return totalDataTransferred;
    }
    
    public double getAverageNetworkLatency() {
        return averageNetworkLatency;
    }
    
    public int getNetworkCongestionEvents() {
        return networkCongestionEvents;
    }
    
    public int getPacketLossEvents() {
        return packetLossEvents;
    }
    
    public int getTotalAttackAttempts() {
        return totalAttackAttempts;
    }
    
    public int getSuccessfulAttacks() {
        return successfulAttacks;
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
    
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    public double getIoTEnergyConsumed() {
        return ioTEnergyConsumed;
    }
    
    public double getEdgeEnergyConsumed() {
        return edgeEnergyConsumed;
    }
    
    public double getFogEnergyConsumed() {
        return fogEnergyConsumed;
    }
    
    public double getCloudEnergyConsumed() {
        return cloudEnergyConsumed;
    }
    
    /**
     * Returns a comprehensive string representation of the simulation results
     * @return String representation of the simulation results
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("===== FOG-EDGE SIMULATION RESULTS =====\n\n");
        
        // Simulation metrics
        sb.append("Simulation Time: ").append(totalSimulationTime).append(" ms\n\n");
        
        // Task metrics
        sb.append("--- TASK METRICS ---\n");
        sb.append("Total Tasks Generated: ").append(totalTasksGenerated).append("\n");
        sb.append("Total Tasks Completed: ").append(totalTasksCompleted).append("\n");
        sb.append("Total Tasks Failed: ").append(totalTasksFailed).append("\n");
        sb.append("Task Success Rate: ").append(totalTasksGenerated > 0 ? 
                String.format("%.2f%%", (double)totalTasksCompleted / totalTasksGenerated * 100) : "0.00%").append("\n\n");
        
        sb.append("Task Distribution:\n");
        sb.append("  - IoT Devices: ").append(tasksExecutedOnIoT);
        if (totalTasksCompleted > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)tasksExecutedOnIoT / totalTasksCompleted * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("  - Edge Nodes: ").append(tasksExecutedOnEdge);
        if (totalTasksCompleted > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)tasksExecutedOnEdge / totalTasksCompleted * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("  - Fog Nodes: ").append(tasksExecutedOnFog);
        if (totalTasksCompleted > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)tasksExecutedOnFog / totalTasksCompleted * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("  - Cloud: ").append(tasksExecutedOnCloud);
        if (totalTasksCompleted > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)tasksExecutedOnCloud / totalTasksCompleted * 100)).append(")");
        }
        sb.append("\n\n");
        
        // Network metrics
        sb.append("--- NETWORK METRICS ---\n");
        sb.append("Total Data Transferred: ").append(String.format("%.2f MB", totalDataTransferred)).append("\n");
        sb.append("Average Network Latency: ").append(String.format("%.2f ms", averageNetworkLatency)).append("\n");
        sb.append("Network Congestion Events: ").append(networkCongestionEvents).append("\n");
        sb.append("Packet Loss Events: ").append(packetLossEvents).append("\n\n");
        
        // Security metrics
        sb.append("--- SECURITY METRICS ---\n");
        sb.append("Total Attack Attempts: ").append(totalAttackAttempts).append("\n");
        sb.append("Successful Attacks: ").append(successfulAttacks);
        if (totalAttackAttempts > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)successfulAttacks / totalAttackAttempts * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("Detected Attacks: ").append(detectedAttacks);
        if (totalAttackAttempts > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)detectedAttacks / totalAttackAttempts * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("Mitigated Attacks: ").append(mitigatedAttacks);
        if (totalAttackAttempts > 0) {
            sb.append(" (").append(String.format("%.2f%%", (double)mitigatedAttacks / totalAttackAttempts * 100)).append(")");
        }
        sb.append("\n\n");
        
        // Attack types
        if (!attacksByType.isEmpty()) {
            sb.append("Attack Types Distribution:\n");
            for (Map.Entry<String, Integer> entry : attacksByType.entrySet()) {
                sb.append("  - ").append(entry.getKey()).append(": ").append(entry.getValue());
                if (totalAttackAttempts > 0) {
                    sb.append(" (").append(String.format("%.2f%%", (double)entry.getValue() / totalAttackAttempts * 100)).append(")");
                }
                sb.append("\n");
            }
            sb.append("\n");
        }
        
        // Energy metrics
        sb.append("--- ENERGY METRICS ---\n");
        sb.append("Total Energy Consumed: ").append(String.format("%.2f mWh", totalEnergyConsumed)).append("\n");
        sb.append("Energy Distribution:\n");
        sb.append("  - IoT Devices: ").append(String.format("%.2f mWh", ioTEnergyConsumed));
        if (totalEnergyConsumed > 0) {
            sb.append(" (").append(String.format("%.2f%%", ioTEnergyConsumed / totalEnergyConsumed * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("  - Edge Nodes: ").append(String.format("%.2f mWh", edgeEnergyConsumed));
        if (totalEnergyConsumed > 0) {
            sb.append(" (").append(String.format("%.2f%%", edgeEnergyConsumed / totalEnergyConsumed * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("  - Fog Nodes: ").append(String.format("%.2f mWh", fogEnergyConsumed));
        if (totalEnergyConsumed > 0) {
            sb.append(" (").append(String.format("%.2f%%", fogEnergyConsumed / totalEnergyConsumed * 100)).append(")");
        }
        sb.append("\n");
        
        sb.append("  - Cloud: ").append(String.format("%.2f mWh", cloudEnergyConsumed));
        if (totalEnergyConsumed > 0) {
            sb.append(" (").append(String.format("%.2f%%", cloudEnergyConsumed / totalEnergyConsumed * 100)).append(")");
        }
        sb.append("\n");
        
        return sb.toString();
    }
}
