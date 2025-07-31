package com.fog.eedto.model;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a Cloud Server in the EEDTO system.
 * Cloud servers have high resources but are located farther from IoT devices,
 * resulting in higher latency compared to edge servers.
 */
public class CloudServer extends Device {
    private final double powerConsumption; // Power consumption in Watts
    private final double latency; // Network latency in milliseconds
    private final int maxConcurrentTasks; // Maximum number of tasks that can be executed concurrently
    private final List<Task> activeTasks; // Tasks currently being executed
    private final double costPerMI; // Cost per Million Instructions executed
    private final double scalingFactor; // Factor by which resources can be scaled up
    
    /**
     * Constructor for the CloudServer class
     * 
     * @param id Unique identifier for the cloud server
     * @param name Name of the cloud server
     * @param mips Processing speed in Million Instructions Per Second
     * @param ram RAM in MB
     * @param storage Storage in MB
     * @param bandwidth Bandwidth in Mbps
     * @param energyEfficiency Energy efficiency in MIPS per Watt
     * @param powerConsumption Power consumption in Watts
     * @param latency Network latency in milliseconds
     * @param maxConcurrentTasks Maximum number of tasks that can be executed concurrently
     * @param costPerMI Cost per Million Instructions executed
     * @param scalingFactor Factor by which resources can be scaled up
     */
    public CloudServer(int id, String name, double mips, int ram, long storage, 
                      double bandwidth, double energyEfficiency, double powerConsumption, 
                      double latency, int maxConcurrentTasks, double costPerMI,
                      double scalingFactor) {
        super(id, name, mips, ram, storage, bandwidth, energyEfficiency);
        this.powerConsumption = powerConsumption;
        this.latency = latency;
        this.maxConcurrentTasks = maxConcurrentTasks;
        this.costPerMI = costPerMI;
        this.scalingFactor = scalingFactor;
        this.activeTasks = new ArrayList<>();
    }

    // Getters and setters
    public double getPowerConsumption() {
        return powerConsumption;
    }

    public double getLatency() {
        return latency;
    }

    public int getMaxConcurrentTasks() {
        return maxConcurrentTasks;
    }

    public List<Task> getActiveTasks() {
        return activeTasks;
    }
    
    public double getCostPerMI() {
        return costPerMI;
    }
    
    public double getScalingFactor() {
        return scalingFactor;
    }

    /**
     * Calculate the cost of executing a task
     * 
     * @param task Task to be executed
     * @return Cost in monetary units
     */
    public double calculateCost(Task task) {
        return task.getLength() * costPerMI;
    }

    /**
     * Add a task to the active tasks list
     * 
     * @param task Task to be added
     * @return true if the task was added, false if the server is at capacity
     */
    public boolean addActiveTask(Task task) {
        if (activeTasks.size() < maxConcurrentTasks) {
            activeTasks.add(task);
            return true;
        }
        return false;
    }

    /**
     * Remove a task from the active tasks list
     * 
     * @param task Task to be removed
     * @return true if the task was removed, false otherwise
     */
    public boolean removeActiveTask(Task task) {
        return activeTasks.remove(task);
    }

    /**
     * Calculate the total latency for executing a task (network latency + execution time)
     * 
     * @param task Task to be executed
     * @return Total latency in seconds
     */
    public double calculateTotalLatency(Task task) {
        // Cloud servers have higher processing power, so execution time is reduced by scaling factor
        double executionTime = task.calculateExecutionTime(getMips() * scalingFactor);
        // Convert latency from milliseconds to seconds
        return executionTime + (latency / 1000.0);
    }

    /**
     * Scale resources based on the current load
     * 
     * @param currentLoad Current load as a percentage (0-1)
     * @return Scaled MIPS
     */
    public double getScaledMips(double currentLoad) {
        // Scale MIPS based on current load and scaling factor
        double scaleFactor = 1.0 + (scalingFactor - 1.0) * currentLoad;
        return getMips() * scaleFactor;
    }

    @Override
    public double executeTask(Task task, double currentTime) {
        if (!canExecuteTask(task)) {
            throw new IllegalStateException("Cloud server cannot execute this task due to resource constraints");
        }
        
        // Add task to active tasks
        addActiveTask(task);
        
        // Set task status to executing
        task.setStatus(Task.TaskStatus.EXECUTING);
        task.setStartTime(currentTime);
        task.setExecutionLocation(Task.DeviceType.CLOUD_SERVER);
        
        // Calculate current load and scaled MIPS
        double currentLoad = (double) activeTasks.size() / maxConcurrentTasks;
        double scaledMips = getScaledMips(currentLoad);
        
        // Calculate execution time (including network latency)
        double executionTime = task.getLength() / scaledMips;
        double totalTime = executionTime + (latency / 1000.0); // Convert latency from ms to seconds
        
        // Calculate energy consumption
        double energyConsumption = powerConsumption * executionTime;
        task.setEnergyConsumed(energyConsumption);
        
        // Update total energy consumed
        setEnergyConsumed(getEnergyConsumed() + energyConsumption);
        
        // Set task finish time
        double finishTime = currentTime + totalTime;
        task.setFinishTime(finishTime);
        
        // Set task status to completed
        task.setStatus(Task.TaskStatus.COMPLETED);
        
        // Remove task from active tasks
        removeActiveTask(task);
        
        return finishTime;
    }

    @Override
    public boolean canExecuteTask(Task task) {
        // Cloud servers can virtually handle any task due to their scalability,
        // but we still check for basic constraints
        boolean hasSufficientMips = getMips() > 0;
        boolean hasCapacity = activeTasks.size() < maxConcurrentTasks;
        
        return isActive() && hasSufficientMips && hasCapacity;
    }

    @Override
    public String toString() {
        return "CloudServer{" +
                "id=" + getId() +
                ", name='" + getName() + '\'' +
                ", mips=" + getMips() +
                ", ram=" + getRam() +
                ", storage=" + getStorage() +
                ", bandwidth=" + getBandwidth() +
                ", energyEfficiency=" + getEnergyEfficiency() +
                ", powerConsumption=" + powerConsumption +
                ", latency=" + latency +
                ", maxConcurrentTasks=" + maxConcurrentTasks +
                ", activeTasks=" + activeTasks.size() +
                ", scalingFactor=" + scalingFactor +
                ", active=" + isActive() +
                '}';
    }
}
