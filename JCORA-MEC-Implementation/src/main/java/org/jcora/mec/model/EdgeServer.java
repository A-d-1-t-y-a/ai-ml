package org.jcora.mec.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents an Edge Server in the MEC environment.
 * Edge servers receive offloaded tasks from IoT devices and process them.
 */
public class EdgeServer {
    private final int id;
    private final String name;
    private final double processingPower;      // in MIPS (Million Instructions Per Second)
    private final double energyConsumption;    // in Watts when processing at full capacity
    private final double idleEnergyConsumption; // in Watts when idle
    private final double maxBandwidth;         // in Mbps
    private final int maxConnections;          // maximum number of simultaneous connections
    
    private final Map<Integer, Double> deviceBandwidthMap; // Maps device ID to allocated bandwidth
    private final List<Task> taskQueue;
    private final List<Task> processingTasks;
    private double totalEnergyConsumed;
    private double totalProcessingTime;
    private int completedTasks;
    private int failedTasks;
    private double currentLoad;               // current load as a percentage (0-100%)
    
    /**
     * Constructor for creating a new Edge Server.
     * 
     * @param id Unique identifier for the server
     * @param name Name of the server
     * @param processingPower Processing power in MIPS
     * @param energyConsumption Energy consumption in Watts when processing
     * @param idleEnergyConsumption Energy consumption in Watts when idle
     * @param maxBandwidth Maximum bandwidth in Mbps
     * @param maxConnections Maximum number of simultaneous connections
     */
    public EdgeServer(int id, String name, double processingPower, double energyConsumption,
                     double idleEnergyConsumption, double maxBandwidth, int maxConnections) {
        this.id = id;
        this.name = name;
        this.processingPower = processingPower;
        this.energyConsumption = energyConsumption;
        this.idleEnergyConsumption = idleEnergyConsumption;
        this.maxBandwidth = maxBandwidth;
        this.maxConnections = maxConnections;
        
        this.deviceBandwidthMap = new HashMap<>();
        this.taskQueue = new ArrayList<>();
        this.processingTasks = new ArrayList<>();
        this.totalEnergyConsumed = 0.0;
        this.totalProcessingTime = 0.0;
        this.completedTasks = 0;
        this.failedTasks = 0;
        this.currentLoad = 0.0;
    }
    
    // Getters and setters
    
    public int getId() {
        return id;
    }
    
    public String getName() {
        return name;
    }
    
    public double getProcessingPower() {
        return processingPower;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    public double getIdleEnergyConsumption() {
        return idleEnergyConsumption;
    }
    
    public double getMaxBandwidth() {
        return maxBandwidth;
    }
    
    public int getMaxConnections() {
        return maxConnections;
    }
    
    public Map<Integer, Double> getDeviceBandwidthMap() {
        return new HashMap<>(deviceBandwidthMap);
    }
    
    public List<Task> getTaskQueue() {
        return new ArrayList<>(taskQueue);
    }
    
    public List<Task> getProcessingTasks() {
        return new ArrayList<>(processingTasks);
    }
    
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    public double getTotalProcessingTime() {
        return totalProcessingTime;
    }
    
    public int getCompletedTasks() {
        return completedTasks;
    }
    
    public int getFailedTasks() {
        return failedTasks;
    }
    
    public double getCurrentLoad() {
        return currentLoad;
    }
    
    public void setCurrentLoad(double currentLoad) {
        this.currentLoad = currentLoad;
    }
    
    /**
     * Allocate bandwidth to a device.
     * 
     * @param deviceId ID of the device
     * @param bandwidth Bandwidth to allocate in Mbps
     * @return True if bandwidth was successfully allocated, false otherwise
     */
    public boolean allocateBandwidth(int deviceId, double bandwidth) {
        // Check if the server has enough available bandwidth
        double currentAllocated = deviceBandwidthMap.values().stream().mapToDouble(Double::doubleValue).sum();
        if (currentAllocated + bandwidth > maxBandwidth) {
            return false;
        }
        
        // Check if the server has enough available connections
        if (!deviceBandwidthMap.containsKey(deviceId) && deviceBandwidthMap.size() >= maxConnections) {
            return false;
        }
        
        // Allocate bandwidth
        deviceBandwidthMap.put(deviceId, bandwidth);
        return true;
    }
    
    /**
     * Release bandwidth allocated to a device.
     * 
     * @param deviceId ID of the device
     */
    public void releaseBandwidth(int deviceId) {
        deviceBandwidthMap.remove(deviceId);
    }
    
    /**
     * Get the bandwidth allocated to a device.
     * 
     * @param deviceId ID of the device
     * @return Allocated bandwidth in Mbps, or 0 if no bandwidth is allocated
     */
    public double getAllocatedBandwidth(int deviceId) {
        return deviceBandwidthMap.getOrDefault(deviceId, 0.0);
    }
    
    /**
     * Add a task to the server's queue.
     * 
     * @param task Task to be added
     */
    public void addTask(Task task) {
        taskQueue.add(task);
        task.setStatus(Task.TaskStatus.WAITING);
    }
    
    /**
     * Process a task on the server.
     * 
     * @param task Task to be processed
     * @param currentTime Current simulation time
     * @return True if the task was successfully processed, false otherwise
     */
    public boolean processTask(Task task, double currentTime) {
        // Calculate processing time based on current load
        double effectiveProcessingPower = processingPower * (1 - currentLoad / 100.0);
        double processingTime = task.calculateProcessingTime(effectiveProcessingPower);
        double energyRequired = processingTime * energyConsumption;
        
        // Process the task
        task.setStartTime(currentTime);
        task.setFinishTime(currentTime + processingTime);
        task.setStatus(Task.TaskStatus.PROCESSING);
        task.setAssignedDeviceId(this.id);
        
        // Update server state
        processingTasks.add(task);
        totalEnergyConsumed += energyRequired;
        totalProcessingTime += processingTime;
        
        // Update server load
        updateServerLoad();
        
        return true;
    }
    
    /**
     * Complete a task that has finished processing.
     * 
     * @param task Task that has finished processing
     */
    public void completeTask(Task task) {
        task.setStatus(Task.TaskStatus.COMPLETED);
        processingTasks.remove(task);
        completedTasks++;
        
        // Update server load
        updateServerLoad();
    }
    
    /**
     * Mark a task as failed.
     * 
     * @param task Task that has failed
     */
    public void failTask(Task task) {
        task.setStatus(Task.TaskStatus.FAILED);
        processingTasks.remove(task);
        failedTasks++;
        
        // Update server load
        updateServerLoad();
    }
    
    /**
     * Update the server's load based on the number of tasks being processed.
     */
    private void updateServerLoad() {
        // Simple load calculation based on the number of tasks being processed
        currentLoad = (processingTasks.size() / (double) maxConnections) * 100.0;
    }
    
    /**
     * Calculate the energy consumption for processing a task.
     * 
     * @param task Task to be processed
     * @return Energy consumption in Joules
     */
    public double calculateProcessingEnergy(Task task) {
        double effectiveProcessingPower = processingPower * (1 - currentLoad / 100.0);
        double processingTime = task.calculateProcessingTime(effectiveProcessingPower);
        return processingTime * energyConsumption;
    }
    
    /**
     * Update the server's energy consumption during idle time.
     * 
     * @param idleTime Time spent idle in seconds
     */
    public void consumeIdleEnergy(double idleTime) {
        double energyConsumed = idleTime * idleEnergyConsumption;
        totalEnergyConsumed += energyConsumed;
    }
    
    @Override
    public String toString() {
        return "EdgeServer{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", processingPower=" + processingPower +
                ", currentLoad=" + currentLoad + "%" +
                ", completedTasks=" + completedTasks +
                ", failedTasks=" + failedTasks +
                '}';
    }
}
