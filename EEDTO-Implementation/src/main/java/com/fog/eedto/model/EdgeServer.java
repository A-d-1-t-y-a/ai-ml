package com.fog.eedto.model;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents an Edge Server in the EEDTO system.
 * Edge servers have moderate resources and are located closer to IoT devices,
 * providing lower latency compared to cloud servers.
 */
public class EdgeServer extends Device {
    private final double powerConsumption; // Power consumption in Watts
    private final double latency; // Network latency in milliseconds
    private final int maxConcurrentTasks; // Maximum number of tasks that can be executed concurrently
    private final List<Task> activeTasks; // Tasks currently being executed
    private final double costPerMI; // Cost per Million Instructions executed
    
    /**
     * Constructor for the EdgeServer class
     * 
     * @param id Unique identifier for the edge server
     * @param name Name of the edge server
     * @param mips Processing speed in Million Instructions Per Second
     * @param ram RAM in MB
     * @param storage Storage in MB
     * @param bandwidth Bandwidth in Mbps
     * @param energyEfficiency Energy efficiency in MIPS per Watt
     * @param powerConsumption Power consumption in Watts
     * @param latency Network latency in milliseconds
     * @param maxConcurrentTasks Maximum number of tasks that can be executed concurrently
     * @param costPerMI Cost per Million Instructions executed
     */
    public EdgeServer(int id, String name, double mips, int ram, long storage, 
                     double bandwidth, double energyEfficiency, double powerConsumption, 
                     double latency, int maxConcurrentTasks, double costPerMI) {
        super(id, name, mips, ram, storage, bandwidth, energyEfficiency);
        this.powerConsumption = powerConsumption;
        this.latency = latency;
        this.maxConcurrentTasks = maxConcurrentTasks;
        this.costPerMI = costPerMI;
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
        double executionTime = task.calculateExecutionTime(getMips());
        // Convert latency from milliseconds to seconds
        return executionTime + (latency / 1000.0);
    }

    @Override
    public double executeTask(Task task, double currentTime) {
        if (!canExecuteTask(task)) {
            throw new IllegalStateException("Edge server cannot execute this task due to resource constraints");
        }
        
        // Add task to active tasks
        addActiveTask(task);
        
        // Set task status to executing
        task.setStatus(Task.TaskStatus.EXECUTING);
        task.setStartTime(currentTime);
        task.setExecutionLocation(Task.DeviceType.EDGE_SERVER);
        
        // Calculate execution time (including network latency)
        double executionTime = task.calculateExecutionTime(getMips());
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
        // Check if the edge server has sufficient resources to execute the task
        boolean hasSufficientMips = getMips() > 0;
        boolean hasSufficientRam = task.getInputSize() / 1024 <= getRam(); // Convert bytes to KB
        boolean hasSufficientStorage = task.getOutputSize() / 1024 <= getStorage(); // Convert bytes to KB
        boolean hasCapacity = activeTasks.size() < maxConcurrentTasks;
        
        return isActive() && hasSufficientMips && hasSufficientRam && 
               hasSufficientStorage && hasCapacity;
    }

    @Override
    public String toString() {
        return "EdgeServer{" +
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
                ", active=" + isActive() +
                '}';
    }
}
