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
     * Increments the total tasks completed count
     */
    public void incrementTasksCompleted() {
        totalTasksCompleted++;
    }
    
    /**
     * Increments the total tasks failed count
     */
    public void incrementTasksFailed() {
        totalTasksFailed++;
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
            averageNetworkLatency = networkLatencies.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);
        }
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
