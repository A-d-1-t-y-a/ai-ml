package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a Cloud Datacenter in the simulation.
 * Cloud datacenters are large-scale computing facilities with abundant resources.
 * They are typically located far from IoT devices and have high latency but high computing power.
 */
public class CloudDatacenter extends Device {
    // Cloud datacenter specific properties
    private double networkBandwidth; // Mbps
    private double networkLatency; // ms
    private int maxConnections; // Maximum number of connected fog nodes
    private List<String> connectedFogNodeIds; // IDs of connected fog nodes
    private Map<String, Double> resourceAllocation; // Resource allocation per fog node (percentage)
    private double securityLevel; // 0-1, higher means more secure
    private boolean hasRedundancy; // Whether the cloud datacenter has redundancy
    private double costPerCpuHour; // Cost per CPU hour
    private double costPerGbStorage; // Cost per GB of storage per hour
    private double costPerGbNetwork; // Cost per GB of network transfer
    
    /**
     * Constructor for a Cloud Datacenter
     * 
     * @param id Unique identifier for the device
     * @param name Human-readable name for the device
     * @param xPos Initial X position
     * @param yPos Initial Y position
     * @param cpuCapacity CPU capacity in MIPS
     * @param ramCapacity RAM capacity in MB
     * @param storageCapacity Storage capacity in GB
     * @param maxConnections Maximum number of connected fog nodes
     */
    public CloudDatacenter(String id, String name, double xPos, double yPos, int cpuCapacity,
                          int ramCapacity, int storageCapacity, int maxConnections) {
        // Cloud datacenters don't have batteries, they're always connected to power
        super(id, DeviceType.CLOUD_DATACENTER, name, xPos, yPos, cpuCapacity, ramCapacity, storageCapacity, 0);
        this.networkBandwidth = 10000; // Default 10 Gbps
        this.networkLatency = 100; // Default 100ms latency (higher due to distance)
        this.maxConnections = maxConnections;
        this.securityLevel = 0.9; // Default very high security level
        this.hasRedundancy = true; // Default has redundancy
        this.costPerCpuHour = 0.05; // Default $0.05 per CPU hour
        this.costPerGbStorage = 0.02; // Default $0.02 per GB storage per hour
        this.costPerGbNetwork = 0.10; // Default $0.10 per GB network transfer
        this.connectedFogNodeIds = new ArrayList<>();
        this.resourceAllocation = new HashMap<>();
        
        // Cloud datacenters are always active
        this.isActive = true;
    }
    
    /**
     * Connects a fog node to this cloud datacenter
     * 
     * @param fogNodeId ID of the fog node to connect
     * @param allocatedResources Percentage of resources allocated to this fog node (0-100)
     * @return True if the connection was successful, false otherwise
     */
    public boolean connectFogNode(String fogNodeId, double allocatedResources) {
        // Check if the cloud datacenter can accept more connections
        if (connectedFogNodeIds.size() >= maxConnections) {
            return false;
        }
        
        // Check if the fog node is already connected
        if (connectedFogNodeIds.contains(fogNodeId)) {
            // Update resource allocation
            resourceAllocation.put(fogNodeId, Math.max(0, Math.min(100, allocatedResources)));
            return true;
        }
        
        // Add the fog node to the connected devices list
        connectedFogNodeIds.add(fogNodeId);
        resourceAllocation.put(fogNodeId, Math.max(0, Math.min(100, allocatedResources)));
        return true;
    }
    
    /**
     * Disconnects a fog node from this cloud datacenter
     * 
     * @param fogNodeId ID of the fog node to disconnect
     * @return True if the fog node was disconnected, false if it wasn't connected
     */
    public boolean disconnectFogNode(String fogNodeId) {
        boolean removed = connectedFogNodeIds.remove(fogNodeId);
        if (removed) {
            resourceAllocation.remove(fogNodeId);
        }
        return removed;
    }
    
    /**
     * Checks if a fog node is connected to this cloud datacenter
     * 
     * @param fogNodeId ID of the fog node to check
     * @return True if the fog node is connected, false otherwise
     */
    public boolean isFogNodeConnected(String fogNodeId) {
        return connectedFogNodeIds.contains(fogNodeId);
    }
    
    /**
     * Gets the number of connected fog nodes
     * 
     * @return Number of connected fog nodes
     */
    public int getConnectedFogNodeCount() {
        return connectedFogNodeIds.size();
    }
    
    /**
     * Gets the list of connected fog node IDs
     * 
     * @return List of connected fog node IDs
     */
    public List<String> getConnectedFogNodeIds() {
        return new ArrayList<>(connectedFogNodeIds); // Return a copy to prevent modification
    }
    
    /**
     * Gets the resource allocation for a specific fog node
     * 
     * @param fogNodeId ID of the fog node
     * @return Resource allocation percentage, or 0 if the fog node is not connected
     */
    public double getResourceAllocation(String fogNodeId) {
        return resourceAllocation.getOrDefault(fogNodeId, 0.0);
    }
    
    /**
     * Calculates the available bandwidth per connected fog node
     * 
     * @return Available bandwidth per fog node in Mbps
     */
    public double getAvailableBandwidthPerFogNode() {
        int connectedFogNodes = Math.max(1, connectedFogNodeIds.size());
        return networkBandwidth / connectedFogNodes;
    }
    
    /**
     * Executes a task on this cloud datacenter
     * 
     * @param task The task to execute
     * @return True if the task was executed successfully, false otherwise
     */
    @Override
    public boolean executeTask(Task task) {
        // Cloud datacenters are always active
        
        // Get the source fog node ID from the task
        String sourceFogNodeId = task.getSourceDeviceId();
        
        // Check if the source fog node is connected to this cloud datacenter
        if (!connectedFogNodeIds.contains(sourceFogNodeId)) {
            return false;
        }
        
        // Get the resource allocation for this fog node
        double allocation = resourceAllocation.getOrDefault(sourceFogNodeId, 0.0);
        
        // Calculate available processing power based on allocation
        double availableProcessingPower = processingPower * (allocation / 100.0);
        
        // Check if the cloud datacenter has enough resources to execute the task
        double taskExecutionTime = task.getLength() / availableProcessingPower;
        
        // Check if the cloud datacenter has enough memory
        double memoryRequired = task.getInputSize() / 1024.0; // Convert KB to MB
        double availableMemory = memory * (allocation / 100.0) * (1 - resourceUtilization / 100);
        
        if (memoryRequired > availableMemory) {
            return false; // Not enough available memory
        }
        
        // Update resource utilization
        double newUtilization = resourceUtilization + (taskExecutionTime / 10) * 100 * (allocation / 100.0);
        updateResourceUtilization(Math.min(100, newUtilization));
        
        return true;
    }
    
    /**
     * Calculates the cost of executing a task
     * 
     * @param task The task to execute
     * @return Cost in currency units
     */
    public double calculateTaskExecutionCost(Task task) {
        // Calculate CPU usage cost
        double cpuHours = task.getLength() / (processingPower * 3600); // Convert MIPS to hours
        double cpuCost = cpuHours * costPerCpuHour;
        
        // Calculate storage cost
        double storageGb = task.getInputSize() / (1024 * 1024); // Convert KB to GB
        double storageCost = storageGb * costPerGbStorage * (cpuHours / 24.0); // Prorated to hours
        
        // Calculate network cost
        double networkGb = (task.getInputSize() + task.getOutputSize()) / (1024 * 1024); // Convert KB to GB
        double networkCost = networkGb * costPerGbNetwork;
        
        return cpuCost + storageCost + networkCost;
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
        double overhead = securityLevel * 0.1; // 0-10% overhead
        
        // Apply the overhead to resource utilization
        double newUtilization = resourceUtilization + overhead * 100;
        updateResourceUtilization(Math.min(100, newUtilization));
        
        return overhead;
    }
    
    /**
     * Detects if there's an attack on this cloud datacenter
     * Higher security level means better detection rate
     * 
     * @param actualAttack True if there's an actual attack, false otherwise
     * @return True if the attack was detected, false otherwise
     */
    public boolean detectAttack(boolean actualAttack) {
        if (!actualAttack) {
            // No attack, but might have false positive
            double falsePositiveRate = 0.01 * (1 - securityLevel); // 0-1% false positive rate
            return Math.random() < falsePositiveRate;
        } else {
            // There is an attack, detection depends on security level
            // Cloud datacenters have the best detection capabilities
            return Math.random() < (securityLevel * 1.5); // Up to 150% detection rate (capped at 100%)
        }
    }
    
    /**
     * Handles a failure event
     * 
     * @return True if the cloud datacenter remains operational, false otherwise
     */
    public boolean handleFailure() {
        if (hasRedundancy) {
            // Cloud datacenter has redundancy, so it remains operational
            return true;
        } else {
            // Cloud datacenter has no redundancy
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
    
    public boolean hasRedundancy() {
        return hasRedundancy;
    }
    
    public void setRedundancy(boolean hasRedundancy) {
        this.hasRedundancy = hasRedundancy;
    }
    
    public double getCostPerCpuHour() {
        return costPerCpuHour;
    }
    
    public double getCostPerGbStorage() {
        return costPerGbStorage;
    }
    
    public double getCostPerGbNetwork() {
        return costPerGbNetwork;
    }
    
    /**
     * Increments the count of connected fog nodes
     * This is a utility method used by the TopologyManager
     */
    public void incrementConnectedFogNodesCount() {
        // This method is intentionally left empty as the count is already managed by the connectedFogNodeIds list
        // The actual count can be retrieved using getConnectedFogNodeCount()
    }
}
