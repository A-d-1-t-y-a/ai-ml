package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a Fog Node in the simulation.
 * Fog nodes are intermediate computing devices between edge and cloud.
 * They have significant computing resources and are typically located in local networks.
 */
public class FogNode extends Device {
    // Fog node specific properties
    private double networkBandwidth; // Mbps
    private double networkLatency; // ms
    private int maxConnections; // Maximum number of connected devices
    private List<String> connectedEdgeNodeIds; // IDs of connected edge nodes
    private Map<String, Double> resourceAllocation; // Resource allocation per edge node (percentage)
    private double securityLevel; // 0-1, higher means more secure
    private boolean hasBackupPower; // Whether the fog node has backup power
    
    /**
     * Constructor for a Fog Node
     * 
     * @param id Unique identifier for the device
     * @param processingPower Processing power in MIPS
     * @param memory Memory in MB
     * @param storage Storage in GB
     * @param batteryCapacity Battery capacity in mAh (0 for always-on devices)
     * @param xPos Initial X position
     * @param yPos Initial Y position
     * @param networkBandwidth Network bandwidth in Mbps
     * @param networkLatency Network latency in ms
     * @param maxConnections Maximum number of connected devices
     * @param securityLevel Security level (0-1)
     * @param hasBackupPower Whether the fog node has backup power
     */
    public FogNode(String id, double processingPower, double memory, double storage,
                  double batteryCapacity, double xPos, double yPos, double networkBandwidth,
                  double networkLatency, int maxConnections, double securityLevel,
                  boolean hasBackupPower) {
        super(id, DeviceType.FOG_NODE, processingPower, memory, storage, batteryCapacity, xPos, yPos);
        this.networkBandwidth = networkBandwidth;
        this.networkLatency = networkLatency;
        this.maxConnections = maxConnections;
        this.securityLevel = Math.max(0, Math.min(1, securityLevel)); // Ensure between 0 and 1
        this.hasBackupPower = hasBackupPower;
        this.connectedEdgeNodeIds = new ArrayList<>();
        this.resourceAllocation = new HashMap<>();
    }
    
    /**
     * Connects an edge node to this fog node
     * 
     * @param edgeNodeId ID of the edge node to connect
     * @param allocatedResources Percentage of resources allocated to this edge node (0-100)
     * @return True if the connection was successful, false otherwise
     */
    public boolean connectEdgeNode(String edgeNodeId, double allocatedResources) {
        // Check if the fog node can accept more connections
        if (connectedEdgeNodeIds.size() >= maxConnections) {
            return false;
        }
        
        // Check if the edge node is already connected
        if (connectedEdgeNodeIds.contains(edgeNodeId)) {
            // Update resource allocation
            resourceAllocation.put(edgeNodeId, Math.max(0, Math.min(100, allocatedResources)));
            return true;
        }
        
        // Add the edge node to the connected devices list
        connectedEdgeNodeIds.add(edgeNodeId);
        resourceAllocation.put(edgeNodeId, Math.max(0, Math.min(100, allocatedResources)));
        return true;
    }
    
    /**
     * Disconnects an edge node from this fog node
     * 
     * @param edgeNodeId ID of the edge node to disconnect
     * @return True if the edge node was disconnected, false if it wasn't connected
     */
    public boolean disconnectEdgeNode(String edgeNodeId) {
        boolean removed = connectedEdgeNodeIds.remove(edgeNodeId);
        if (removed) {
            resourceAllocation.remove(edgeNodeId);
        }
        return removed;
    }
    
    /**
     * Checks if an edge node is connected to this fog node
     * 
     * @param edgeNodeId ID of the edge node to check
     * @return True if the edge node is connected, false otherwise
     */
    public boolean isEdgeNodeConnected(String edgeNodeId) {
        return connectedEdgeNodeIds.contains(edgeNodeId);
    }
    
    /**
     * Gets the number of connected edge nodes
     * 
     * @return Number of connected edge nodes
     */
    public int getConnectedEdgeNodeCount() {
        return connectedEdgeNodeIds.size();
    }
    
    /**
     * Gets the list of connected edge node IDs
     * 
     * @return List of connected edge node IDs
     */
    public List<String> getConnectedEdgeNodeIds() {
        return new ArrayList<>(connectedEdgeNodeIds); // Return a copy to prevent modification
    }
    
    /**
     * Gets the resource allocation for a specific edge node
     * 
     * @param edgeNodeId ID of the edge node
     * @return Resource allocation percentage, or 0 if the edge node is not connected
     */
    public double getResourceAllocation(String edgeNodeId) {
        return resourceAllocation.getOrDefault(edgeNodeId, 0.0);
    }
    
    /**
     * Calculates the available bandwidth per connected edge node
     * 
     * @return Available bandwidth per edge node in Mbps
     */
    public double getAvailableBandwidthPerEdgeNode() {
        int connectedEdgeNodes = Math.max(1, connectedEdgeNodeIds.size());
        return networkBandwidth / connectedEdgeNodes;
    }
    
    /**
     * Executes a task on this fog node
     * 
     * @param task The task to execute
     * @return True if the task was executed successfully, false otherwise
     */
    @Override
    public boolean executeTask(Task task) {
        // Check if the device is active
        if (!isActive) {
            return false;
        }
        
        // Get the source edge node ID from the task
        String sourceEdgeNodeId = task.getSourceDeviceId();
        
        // Check if the source edge node is connected to this fog node
        if (!connectedEdgeNodeIds.contains(sourceEdgeNodeId)) {
            return false;
        }
        
        // Get the resource allocation for this edge node
        double allocation = resourceAllocation.getOrDefault(sourceEdgeNodeId, 0.0);
        
        // Calculate available processing power based on allocation
        double availableProcessingPower = processingPower * (allocation / 100.0);
        
        // Check if the fog node has enough resources to execute the task
        double taskExecutionTime = task.getLength() / availableProcessingPower;
        
        // Check if the fog node has enough memory
        double memoryRequired = task.getInputSize() / 1024.0; // Convert KB to MB
        double availableMemory = memory * (allocation / 100.0) * (1 - resourceUtilization / 100);
        
        if (memoryRequired > availableMemory) {
            return false; // Not enough available memory
        }
        
        // Calculate energy consumption for task execution
        double energyConsumption = 0.03 + taskExecutionTime * 0.005;
        
        // Attempt to consume energy (if the device has a battery)
        if (batteryCapacity > 0 && !consumeEnergy(energyConsumption)) {
            // If there's backup power, use it instead
            if (!hasBackupPower) {
                return false; // No backup power and not enough battery
            }
        }
        
        // Update resource utilization
        double newUtilization = resourceUtilization + (taskExecutionTime / 10) * 100 * (allocation / 100.0);
        updateResourceUtilization(Math.min(100, newUtilization));
        
        return true;
    }
    
    /**
     * Offloads a task to the cloud
     * 
     * @param task The task to offload
     * @return Estimated offloading time in ms, or -1 if offloading is not possible
     */
    public double offloadTaskToCloud(Task task) {
        // Check if the device is active
        if (!isActive) {
            return -1;
        }
        
        // Calculate data transfer time based on available bandwidth
        double dataSize = task.getInputSize() + task.getOutputSize(); // Total data size in KB
        double dataSizeMb = dataSize / 1024; // Convert to MB
        double transferTime = (dataSizeMb * 8) / (networkBandwidth * 0.7); // Time in seconds, assume 70% of bandwidth available for cloud
        
        // Convert to milliseconds and add network latency
        return (transferTime * 1000) + networkLatency * 2; // Double latency for cloud communication
    }
    
    /**
     * Applies security measures to protect against attacks
     * Higher security level means better protection but more overhead
     * 
     * @return Security overhead (additional processing time as a percentage)
     */
    public double applySecurityMeasures() {
        // Calculate security overhead based on security level
        // Higher security level means more overhead
        double overhead = securityLevel * 0.15; // 0-15% overhead
        
        // Apply the overhead to resource utilization
        double newUtilization = resourceUtilization + overhead * 100;
        updateResourceUtilization(Math.min(100, newUtilization));
        
        return overhead;
    }
    
    /**
     * Detects if there's an attack on this fog node
     * Higher security level means better detection rate
     * 
     * @param actualAttack True if there's an actual attack, false otherwise
     * @return True if the attack was detected, false otherwise
     */
    public boolean detectAttack(boolean actualAttack) {
        if (!actualAttack) {
            // No attack, but might have false positive
            double falsePositiveRate = 0.03 * (1 - securityLevel); // 0-3% false positive rate
            return Math.random() < falsePositiveRate;
        } else {
            // There is an attack, detection depends on security level
            // Fog nodes have better detection capabilities than edge nodes
            return Math.random() < (securityLevel * 1.2); // Up to 120% detection rate (capped at 100%)
        }
    }
    
    /**
     * Handles a power outage event
     * 
     * @return True if the fog node remains operational, false otherwise
     */
    public boolean handlePowerOutage() {
        if (hasBackupPower) {
            // Fog node has backup power, so it remains operational
            return true;
        } else if (batteryCapacity > 0 && remainingBattery > batteryCapacity * 0.1) {
            // Fog node has enough battery to continue operation
            // Consume 10% of battery capacity for the power outage
            consumeEnergy(batteryCapacity * 0.1);
            return true;
        } else {
            // Fog node has no backup power and insufficient battery
            isActive = false;
            return false;
        }
    }
    
    // Getters and setters
    
    public double getNetworkBandwidth() {
        return networkBandwidth;
    }
    
    public double getNetworkLatency() {
        return networkLatency;
    }
    
    public int getMaxConnections() {
        return maxConnections;
    }
    
    public double getSecurityLevel() {
        return securityLevel;
    }
    
    public void setSecurityLevel(double securityLevel) {
        this.securityLevel = Math.max(0, Math.min(1, securityLevel));
    }
    
    public boolean hasBackupPower() {
        return hasBackupPower;
    }
    
    public void setBackupPower(boolean hasBackupPower) {
        this.hasBackupPower = hasBackupPower;
    }
}
