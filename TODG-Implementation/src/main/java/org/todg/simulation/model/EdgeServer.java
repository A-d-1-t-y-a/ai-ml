package org.todg.simulation.model;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Comparator;

/**
 * Represents an Edge Server in the TODG simulation.
 * Edge servers process offloaded tasks from IoT devices.
 * 
 * Based on the TODG paper: "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021)
 */
public class EdgeServer {
    private int serverId;
    private String serverName;
    private double mips; // Processing capability in Million Instructions Per Second
    private double availableMemory; // in MB
    private double totalMemory; // in MB
    private double availableStorage; // in GB
    private double totalStorage; // in GB
    private double powerConsumption; // in Watts
    private double totalEnergyConsumed; // in Joules
    
    // Server location
    private double xCoordinate;
    private double yCoordinate;
    
    // Task management
    private PriorityQueue<Task> taskQueue;
    private List<Task> completedTasks;
    private List<Task> failedTasks;
    private int currentLoad; // Number of tasks currently being processed
    private int maxLoad; // Maximum number of tasks that can be processed concurrently
    
    /**
     * Constructor for creating a new Edge Server.
     * 
     * @param serverId The unique identifier for this server
     * @param serverName The name of this server
     * @param mips The processing capability in Million Instructions Per Second
     * @param totalMemory The total memory in MB
     * @param totalStorage The total storage in GB
     * @param powerConsumption The power consumption in Watts
     * @param maxLoad The maximum number of tasks that can be processed concurrently
     * @param xCoordinate The x-coordinate of the server location
     * @param yCoordinate The y-coordinate of the server location
     */
    public EdgeServer(int serverId, String serverName, double mips, double totalMemory,
                     double totalStorage, double powerConsumption, int maxLoad,
                     double xCoordinate, double yCoordinate) {
        this.serverId = serverId;
        this.serverName = serverName;
        this.mips = mips;
        this.totalMemory = totalMemory;
        this.availableMemory = totalMemory;
        this.totalStorage = totalStorage;
        this.availableStorage = totalStorage;
        this.powerConsumption = powerConsumption;
        this.totalEnergyConsumed = 0.0;
        this.xCoordinate = xCoordinate;
        this.yCoordinate = yCoordinate;
        this.maxLoad = maxLoad;
        this.currentLoad = 0;
        
        // Initialize task queue with priority based on deadline
        this.taskQueue = new PriorityQueue<>(Comparator.comparingDouble(task -> 
                task.getArrivalTime() + task.getDeadline()));
        this.completedTasks = new ArrayList<>();
        this.failedTasks = new ArrayList<>();
    }
    
    /**
     * Receives a task for processing.
     * 
     * @param task The task to be processed
     * @param currentTime The current simulation time
     * @return true if the task was accepted, false otherwise
     */
    public boolean receiveTask(Task task, double currentTime) {
        // Check if server has capacity to accept the task
        if (currentLoad >= maxLoad) {
            return false;
        }
        
        // Check if server has enough memory for the task
        if (task.getDataSize() > availableMemory) {
            return false;
        }
        
        // Update task status and add to queue
        task.setStatus(Task.TaskStatus.QUEUED);
        task.setAssignedServerId(serverId);
        taskQueue.add(task);
        
        // Update server resources
        availableMemory -= task.getDataSize();
        currentLoad++;
        
        return true;
    }
    
    /**
     * Processes tasks in the queue for a specified time interval.
     * 
     * @param currentTime The current simulation time
     * @param timeInterval The time interval for which tasks are processed
     * @return The number of tasks processed
     */
    public int processTasks(double currentTime, double timeInterval) {
        int tasksProcessed = 0;
        double timeRemaining = timeInterval;
        double energyUsed = 0.0;
        
        while (!taskQueue.isEmpty() && timeRemaining > 0) {
            // Get the highest priority task
            Task task = taskQueue.peek();
            
            // Calculate processing time for this task
            double processingTime = task.getComputationalRequirement() / mips;
            
            if (processingTime <= timeRemaining) {
                // Task can be completed in this interval
                taskQueue.poll(); // Remove task from queue
                
                // Update task status
                task.setStartTime(currentTime + (timeInterval - timeRemaining));
                task.setCompletionTime(task.getStartTime() + processingTime);
                task.setStatus(Task.TaskStatus.COMPLETED);
                
                // Check if task met its deadline
                if (task.getCompletionTime() <= (task.getArrivalTime() + task.getDeadline())) {
                    completedTasks.add(task);
                } else {
                    task.setStatus(Task.TaskStatus.FAILED);
                    failedTasks.add(task);
                }
                
                // Update server resources
                availableMemory += task.getDataSize();
                currentLoad--;
                
                // Update time and energy
                timeRemaining -= processingTime;
                energyUsed += (powerConsumption * processingTime);
                
                tasksProcessed++;
            } else {
                // Task cannot be completed in this interval
                // Process it partially and continue in the next interval
                energyUsed += (powerConsumption * timeRemaining);
                timeRemaining = 0;
            }
        }
        
        // Update total energy consumed
        totalEnergyConsumed += energyUsed;
        
        return tasksProcessed;
    }
    
    /**
     * Calculates the distance to another node (device or server).
     * 
     * @param x The x-coordinate of the other node
     * @param y The y-coordinate of the other node
     * @return The Euclidean distance
     */
    public double calculateDistance(double x, double y) {
        double dx = xCoordinate - x;
        double dy = yCoordinate - y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Calculates the estimated processing time for a task.
     * 
     * @param task The task to process
     * @return The estimated processing time in seconds
     */
    public double calculateProcessingTime(Task task) {
        return task.getComputationalRequirement() / mips;
    }
    
    /**
     * Calculates the estimated energy consumption for processing a task.
     * 
     * @param task The task to process
     * @return The estimated energy consumption in Joules
     */
    public double calculateEnergyConsumption(Task task) {
        double processingTime = calculateProcessingTime(task);
        return powerConsumption * processingTime;
    }
    
    /**
     * Calculates the current load percentage of the server.
     * 
     * @return The load percentage (0-100)
     */
    public double getLoadPercentage() {
        return (currentLoad * 100.0) / maxLoad;
    }
    
    /**
     * Calculates the success rate of task processing.
     * 
     * @return The success rate as a percentage
     */
    public double getSuccessRate() {
        int totalTasks = completedTasks.size() + failedTasks.size();
        if (totalTasks == 0) {
            return 100.0; // No tasks processed yet
        }
        return (completedTasks.size() * 100.0) / totalTasks;
    }
    
    // Getters and setters
    
    public int getServerId() {
        return serverId;
    }
    
    public String getServerName() {
        return serverName;
    }
    
    public double getMips() {
        return mips;
    }
    
    public double getAvailableMemory() {
        return availableMemory;
    }
    
    public double getTotalMemory() {
        return totalMemory;
    }
    
    public double getAvailableStorage() {
        return availableStorage;
    }
    
    public double getTotalStorage() {
        return totalStorage;
    }
    
    public double getPowerConsumption() {
        return powerConsumption;
    }
    
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    public double getXCoordinate() {
        return xCoordinate;
    }
    
    public double getYCoordinate() {
        return yCoordinate;
    }
    
    public int getCurrentLoad() {
        return currentLoad;
    }
    
    public int getMaxLoad() {
        return maxLoad;
    }
    
    public PriorityQueue<Task> getTaskQueue() {
        return taskQueue;
    }
    
    public List<Task> getCompletedTasks() {
        return completedTasks;
    }
    
    public List<Task> getFailedTasks() {
        return failedTasks;
    }
}
