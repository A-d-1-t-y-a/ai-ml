package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents an Edge Node in the simulation.
 * Edge nodes are devices located at the edge of the network, close to IoT devices.
 * They have more resources than IoT devices but less than fog nodes.
 */
public class EdgeNode extends Device {
    // Edge node specific properties
    private double networkBandwidth; // Mbps
    private double networkLatency; // ms
    private int maxConnections; // Maximum number of connected devices
    private List<String> connectedDeviceIds; // IDs of connected devices
    private double securityLevel; // 0-1, higher means more secure
    
    /**
     * Constructor for an Edge Node
     * 
     * @param id Unique identifier for the device
     * @param name Human-readable name for the device
     * @param xPos Initial X position
     * @param yPos Initial Y position
     * @param cpuCapacity CPU capacity in MIPS
     * @param ramCapacity RAM capacity in MB
     * @param storageCapacity Storage capacity in GB
     * @param batteryCapacity Battery capacity in mAh (0 for always-on devices)
     * @param maxConnections Maximum number of connected devices
     */
    public EdgeNode(String id, String name, double xPos, double yPos, int cpuCapacity,
                   int ramCapacity, int storageCapacity, int batteryCapacity,
                   int maxConnections) {
        super(id, DeviceType.EDGE_NODE, name, xPos, yPos, cpuCapacity, ramCapacity, storageCapacity, batteryCapacity);
        this.networkBandwidth = 500; // Default 500 Mbps
        this.networkLatency = 10; // Default 10ms latency
        this.maxConnections = maxConnections;
        this.securityLevel = 0.7; // Default medium-high security level
        this.connectedDeviceIds = new ArrayList<>();
    }
    
    /**
     * Connects a device to this edge node
     * 
     * @param deviceId ID of the device to connect
     * @return True if the connection was successful, false otherwise
     */
    public boolean connectDevice(String deviceId) {
        // Check if the edge node can accept more connections
        if (connectedDeviceIds.size() >= maxConnections) {
            return false;
        }
        
        // Check if the device is already connected
        if (connectedDeviceIds.contains(deviceId)) {
            return true; // Already connected
        }
        
        // Add the device to the connected devices list
        connectedDeviceIds.add(deviceId);
        return true;
    }
    
    /**
     * Disconnects a device from this edge node
     * 
     * @param deviceId ID of the device to disconnect
     * @return True if the device was disconnected, false if it wasn't connected
     */
    public boolean disconnectDevice(String deviceId) {
        return connectedDeviceIds.remove(deviceId);
    }
    
    /**
     * Checks if a device is connected to this edge node
     * 
     * @param deviceId ID of the device to check
     * @return True if the device is connected, false otherwise
     */
    public boolean isDeviceConnected(String deviceId) {
        return connectedDeviceIds.contains(deviceId);
    }
    
    /**
     * Gets the number of connected devices
     * 
     * @return Number of connected devices
     */
    public int getConnectedDeviceCount() {
        return connectedDeviceIds.size();
    }
    
    /**
     * Gets the list of connected device IDs
     * 
     * @return List of connected device IDs
     */
    public List<String> getConnectedDeviceIds() {
        return new ArrayList<>(connectedDeviceIds); // Return a copy to prevent modification
    }
    
    /**
     * Calculates the available bandwidth per connected device
     * 
     * @return Available bandwidth per device in Mbps
     */
    public double getAvailableBandwidthPerDevice() {
        int connectedDevices = Math.max(1, connectedDeviceIds.size());
        return networkBandwidth / connectedDevices;
    }
    
    /**
     * Executes a task on this edge node
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
        
        // Check if the task's source device is connected to this edge node
        if (!isDeviceConnected(task.getSourceDeviceId())) {
            return false;
        }
        
        // Check if the edge node has enough resources to execute the task
        double taskExecutionTime = task.getLength() / processingPower;
        
        // Check if the edge node has enough memory
        double memoryRequired = task.getInputSize() / 1024.0; // Convert KB to MB
        if (memoryRequired > memory * (1 - resourceUtilization / 100)) {
            return false; // Not enough available memory
        }
        
        // Calculate energy consumption for task execution
        double energyConsumption = 0.05 + taskExecutionTime * 0.01;
        
        // Attempt to consume energy (if the device has a battery)
        if (batteryCapacity > 0 && !consumeEnergy(energyConsumption)) {
            return false; // Not enough energy
        }
        
        // Update resource utilization
        double newUtilization = resourceUtilization + (taskExecutionTime / 10) * 100;
        updateResourceUtilization(Math.min(100, newUtilization));
        
        return true;
    }
    
    /**
     * Offloads a task to another device (fog node or cloud)
     * 
     * @param task The task to offload
     * @param targetDeviceId ID of the target device
     * @return Estimated offloading time in ms, or -1 if offloading is not possible
     */
    public double offloadTask(Task task, String targetDeviceId) {
        // Check if the device is active
        if (!isActive) {
            return -1;
        }
        
        // Calculate data transfer time based on available bandwidth
        double dataSize = task.getInputSize() + task.getOutputSize(); // Total data size in KB
        double dataSizeMb = dataSize / 1024; // Convert to MB
        double transferTime = (dataSizeMb * 8) / getAvailableBandwidthPerDevice(); // Time in seconds
        
        // Convert to milliseconds and add network latency
        return (transferTime * 1000) + networkLatency;
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
        double overhead = securityLevel * 0.2; // 0-20% overhead
        
        // Apply the overhead to resource utilization
        double newUtilization = resourceUtilization + overhead * 100;
        updateResourceUtilization(Math.min(100, newUtilization));
        
        return overhead;
    }
    
    /**
     * Detects if there's an attack on this edge node
     * Higher security level means better detection rate
     * 
     * @param actualAttack True if there's an actual attack, false otherwise
     * @return True if the attack was detected, false otherwise
     */
    public boolean detectAttack(boolean actualAttack) {
        if (!actualAttack) {
            // No attack, but might have false positive
            double falsePositiveRate = 0.05 * (1 - securityLevel); // 0-5% false positive rate
            return Math.random() < falsePositiveRate;
        } else {
            // There is an attack, detection depends on security level
            return Math.random() < securityLevel;
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
    
    /**
     * Increments the count of connected devices
     * This is a utility method used by the TopologyManager
     */
    public void incrementConnectedDevicesCount() {
        // This method is intentionally left empty as the count is already managed by the connectedDeviceIds list
        // The actual count can be retrieved using getConnectedDeviceCount()
    }
}
