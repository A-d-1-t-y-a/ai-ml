package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a Fog computing node in the simulation.
 * Fog nodes are intermediate between edge nodes and cloud datacenters,
 * providing more resources than edge nodes but with higher latency.
 */
public class FogNode extends Device {
    private int maxConnections;
    private List<Device> connectedDevices;
    private List<EdgeNode> connectedEdgeNodes;
    private int connectedDevicesCount;
    private int connectedEdgeNodesCount;
    private List<Task> queuedTasks;
    private List<Task> completedTasks;
    private double costPerCpuMips;
    private double costPerRamMb;
    private double costPerStorageGb;
    private double energyEfficiency; // MIPS per mAh
    
    /**
     * Constructor for a Fog node
     * 
     * @param id Unique identifier for the device
     * @param name Human-readable name for the device
     * @param xPos Initial X position
     * @param yPos Initial Y position
     * @param processingPower Processing power in MIPS
     * @param memory Memory in MB
     * @param storage Storage in GB
     * @param batteryCapacity Battery capacity in mAh
     * @param maxConnections Maximum number of devices that can connect to this fog node
     * @param costPerCpuMips Cost per CPU MIPS
     * @param costPerRamMb Cost per RAM MB
     * @param costPerStorageGb Cost per storage GB
     * @param energyEfficiency Energy efficiency in MIPS per mAh
     */
    public FogNode(String id, String name, double xPos, double yPos,
                  double processingPower, double memory, double storage, double batteryCapacity,
                  int maxConnections, double costPerCpuMips, double costPerRamMb,
                  double costPerStorageGb, double energyEfficiency) {
        super(id, DeviceType.FOG_NODE, name, xPos, yPos, processingPower, memory, storage, batteryCapacity);
        this.maxConnections = maxConnections;
        this.connectedDevices = new ArrayList<>();
        this.connectedEdgeNodes = new ArrayList<>();
        this.connectedDevicesCount = 0;
        this.connectedEdgeNodesCount = 0;
        this.queuedTasks = new ArrayList<>();
        this.completedTasks = new ArrayList<>();
        this.costPerCpuMips = costPerCpuMips;
        this.costPerRamMb = costPerRamMb;
        this.costPerStorageGb = costPerStorageGb;
        this.energyEfficiency = energyEfficiency;
    }
    
    /**
     * Connects a device to this fog node
     * 
     * @param device The device to connect
     * @return True if the connection was successful, false otherwise
     */
    public boolean connectDevice(Device device) {
        // Check if the fog node is active
        if (!isActive) {
            return false;
        }
        
        // Check if the fog node has reached its maximum connections
        if (connectedDevices.size() >= maxConnections) {
            return false;
        }
        
        // Check if the device is already connected
        if (connectedDevices.contains(device)) {
            return false;
        }
        
        // Add the device to the list of connected devices
        connectedDevices.add(device);
        connectedDevicesCount++;
        
        // If the device is an edge node, also add it to the list of connected edge nodes
        if (device instanceof EdgeNode) {
            connectedEdgeNodes.add((EdgeNode) device);
            connectedEdgeNodesCount++;
        }
        
        return true;
    }
    
    /**
     * Disconnects a device from this fog node
     * 
     * @param device The device to disconnect
     * @return True if the disconnection was successful, false otherwise
     */
    public boolean disconnectDevice(Device device) {
        boolean removed = connectedDevices.remove(device);
        if (removed) {
            connectedDevicesCount--;
            
            // If the device is an edge node, also remove it from the list of connected edge nodes
            if (device instanceof EdgeNode) {
                connectedEdgeNodes.remove(device);
                connectedEdgeNodesCount--;
            }
        }
        return removed;
    }
    
    /**
     * Queues a task for execution on this fog node
     * 
     * @param task The task to queue
     * @return True if the task was queued successfully, false otherwise
     */
    public boolean queueTask(Task task) {
        // Check if the fog node is active
        if (!isActive) {
            return false;
        }
        
        // Check if the fog node has enough resources to eventually execute the task
        if (task.getCpuRequirement() > processingPower ||
            task.getRamRequirement() > memory ||
            task.getStorageRequirement() > storage) {
            return false;
        }
        
        // Add the task to the queue
        queuedTasks.add(task);
        
        return true;
    }
    
    /**
     * Processes queued tasks
     * 
     * @param timeStep Time step in seconds
     * @return Number of tasks processed
     */
    public int processTasks(double timeStep) {
        // Check if the fog node is active
        if (!isActive) {
            return 0;
        }
        
        int processedTasks = 0;
        double availableMips = processingPower * timeStep;
        double totalEnergyConsumed = 0.0;
        
        // Process tasks until we run out of MIPS or tasks
        List<Task> tasksToRemove = new ArrayList<>();
        
        for (Task task : queuedTasks) {
            // Calculate MIPS required for this task in this time step
            double mipsRequired = task.getCpuRequirement();
            
            // Check if we have enough MIPS available
            if (mipsRequired <= availableMips) {
                // Calculate energy consumption
                double energyConsumption = mipsRequired / energyEfficiency;
                
                // Check if we have enough energy
                if (consumeEnergy(energyConsumption)) {
                    // Execute the task
                    if (executeTask(task)) {
                        processedTasks++;
                        tasksToRemove.add(task);
                        availableMips -= mipsRequired;
                        totalEnergyConsumed += energyConsumption;
                    }
                } else {
                    // Not enough energy
                    break;
                }
            } else {
                // Not enough MIPS available
                break;
            }
        }
        
        // Remove processed tasks from the queue
        queuedTasks.removeAll(tasksToRemove);
        
        // Update resource utilization based on MIPS used
        double utilizationPercentage = ((processingPower * timeStep) - availableMips) / (processingPower * timeStep) * 100.0;
        updateResourceUtilization(utilizationPercentage);
        
        return processedTasks;
    }
    
    /**
     * Executes a task on this device
     * 
     * @param task The task to execute
     * @return True if the task was executed successfully, false otherwise
     */
    @Override
    public boolean executeTask(Task task) {
        // Check if the fog node is active
        if (!isActive) {
            return false;
        }
        
        // Check if the fog node has enough resources
        if (task.getCpuRequirement() > processingPower ||
            task.getRamRequirement() > memory ||
            task.getStorageRequirement() > storage) {
            return false;
        }
        
        // Mark task as completed
        completedTasks.add(task);
        
        return true;
    }
    
    /**
     * Calculates the cost of executing a task on this fog node
     * 
     * @param task The task to calculate cost for
     * @return The cost of executing the task
     */
    public double calculateTaskCost(Task task) {
        return (task.getCpuRequirement() * costPerCpuMips) +
               (task.getRamRequirement() * costPerRamMb) +
               (task.getStorageRequirement() * costPerStorageGb);
    }
    
    /**
     * Increments the count of connected edge nodes
     */
    public void incrementConnectedEdgeNodesCount() {
        connectedEdgeNodesCount++;
    }
    
    /**
     * Gets the maximum number of connections
     * 
     * @return The maximum number of connections
     */
    public int getMaxConnections() {
        return maxConnections;
    }
    
    /**
     * Gets the list of connected devices
     * 
     * @return The list of connected devices
     */
    public List<Device> getConnectedDevices() {
        return new ArrayList<>(connectedDevices);
    }
    
    /**
     * Gets the list of connected edge nodes
     * 
     * @return The list of connected edge nodes
     */
    public List<EdgeNode> getConnectedEdgeNodes() {
        return new ArrayList<>(connectedEdgeNodes);
    }
    
    /**
     * Gets the number of connected devices
     * 
     * @return The number of connected devices
     */
    public int getConnectedDevicesCount() {
        return connectedDevicesCount;
    }
    
    /**
     * Gets the number of connected edge nodes
     * 
     * @return The number of connected edge nodes
     */
    public int getConnectedEdgeNodesCount() {
        return connectedEdgeNodesCount;
    }
    
    /**
     * Gets the list of queued tasks
     * 
     * @return The list of queued tasks
     */
    public List<Task> getQueuedTasks() {
        return new ArrayList<>(queuedTasks);
    }
    
    /**
     * Gets the list of completed tasks
     * 
     * @return The list of completed tasks
     */
    public List<Task> getCompletedTasks() {
        return new ArrayList<>(completedTasks);
    }
    
    /**
     * Gets the cost per CPU MIPS
     * 
     * @return The cost per CPU MIPS
     */
    public double getCostPerCpuMips() {
        return costPerCpuMips;
    }
    
    /**
     * Gets the cost per RAM MB
     * 
     * @return The cost per RAM MB
     */
    public double getCostPerRamMb() {
        return costPerRamMb;
    }
    
    /**
     * Gets the cost per storage GB
     * 
     * @return The cost per storage GB
     */
    public double getCostPerStorageGb() {
        return costPerStorageGb;
    }
    
    /**
     * Gets the energy efficiency
     * 
     * @return The energy efficiency in MIPS per mAh
     */
    public double getEnergyEfficiency() {
        return energyEfficiency;
    }
    
    /**
     * Returns a string representation of the fog node
     * 
     * @return String representation of the fog node
     */
    @Override
    public String toString() {
        return "FogNode{" +
               "id='" + id + '\'' +
               ", name='" + name + '\'' +
               ", isActive=" + isActive +
               ", isCompromised=" + isCompromised +
               ", position=(" + xPos + ", " + yPos + ")" +
               ", processingPower=" + processingPower +
               ", memory=" + memory +
               ", storage=" + storage +
               ", batteryCapacity=" + batteryCapacity +
               ", remainingBattery=" + remainingBattery +
               ", resourceUtilization=" + resourceUtilization +
               ", maxConnections=" + maxConnections +
               ", connectedDevicesCount=" + connectedDevicesCount +
               ", connectedEdgeNodesCount=" + connectedEdgeNodesCount +
               ", queuedTasks=" + queuedTasks.size() +
               ", completedTasks=" + completedTasks.size() +
               '}';
    }
}
