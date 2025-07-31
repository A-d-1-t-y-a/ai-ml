package org.todg.simulation.model;

/**
 * Represents a computational task in the TODG simulation.
 * Each task has properties such as data size, computational requirements, 
 * deadline, and arrival time.
 * 
 * Based on the TODG paper: "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021)
 */
public class Task {
    private int taskId;
    private double arrivalTime;
    private double dataSize; // in MB
    private double computationalRequirement; // in Million Instructions (MI)
    private double deadline; // in seconds
    private double startTime; // Time when task processing begins
    private double completionTime; // Time when task processing completes
    private int sourceDeviceId;
    private int assignedServerId; // ID of the edge server this task is assigned to
    private TaskStatus status;
    private TaskPriority priority;
    
    /**
     * Enum representing the possible statuses of a task.
     */
    public enum TaskStatus {
        CREATED,
        QUEUED,
        TRANSMITTING,
        PROCESSING,
        COMPLETED,
        FAILED
    }
    
    /**
     * Enum representing the priority levels of a task.
     */
    public enum TaskPriority {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }
    
    /**
     * Constructor for creating a new task.
     * 
     * @param taskId The unique identifier for this task
     * @param arrivalTime The time at which the task arrives in the system
     * @param dataSize The size of the input data in MB
     * @param computationalRequirement The computational requirement in MI
     * @param deadline The deadline for task completion in seconds
     * @param sourceDeviceId The ID of the device that generated this task
     */
    public Task(int taskId, double arrivalTime, double dataSize, 
                double computationalRequirement, double deadline, int sourceDeviceId) {
        this.taskId = taskId;
        this.arrivalTime = arrivalTime;
        this.dataSize = dataSize;
        this.computationalRequirement = computationalRequirement;
        this.deadline = deadline;
        this.sourceDeviceId = sourceDeviceId;
        this.status = TaskStatus.CREATED;
        
        // Set priority based on deadline tightness
        setPriorityBasedOnDeadline();
    }
    
    /**
     * Sets the task priority based on how tight the deadline is.
     * Tasks with tighter deadlines get higher priority.
     */
    private void setPriorityBasedOnDeadline() {
        // Estimate execution time (simplified model)
        double estimatedExecutionTime = computationalRequirement / 1000; // Assuming 1000 MIPS as a baseline
        
        // Calculate deadline tightness as ratio of deadline to estimated execution time
        double deadlineTightness = deadline / estimatedExecutionTime;
        
        if (deadlineTightness < 1.2) {
            this.priority = TaskPriority.CRITICAL;
        } else if (deadlineTightness < 2.0) {
            this.priority = TaskPriority.HIGH;
        } else if (deadlineTightness < 5.0) {
            this.priority = TaskPriority.MEDIUM;
        } else {
            this.priority = TaskPriority.LOW;
        }
    }
    
    /**
     * Checks if the task can meet its deadline given the current time and estimated processing time.
     * 
     * @param currentTime The current simulation time
     * @param transmissionTime The estimated time to transmit the task data
     * @param processingTime The estimated time to process the task
     * @return true if the task can meet its deadline, false otherwise
     */
    public boolean canMeetDeadline(double currentTime, double transmissionTime, double processingTime) {
        double estimatedCompletionTime = currentTime + transmissionTime + processingTime;
        return estimatedCompletionTime <= (arrivalTime + deadline);
    }
    
    /**
     * Calculates the slack time for this task.
     * Slack time is the difference between the deadline and the estimated completion time.
     * 
     * @param currentTime The current simulation time
     * @param transmissionTime The estimated time to transmit the task data
     * @param processingTime The estimated time to process the task
     * @return The slack time in seconds
     */
    public double calculateSlackTime(double currentTime, double transmissionTime, double processingTime) {
        double estimatedCompletionTime = currentTime + transmissionTime + processingTime;
        return (arrivalTime + deadline) - estimatedCompletionTime;
    }
    
    // Getters and setters
    
    public int getTaskId() {
        return taskId;
    }
    
    public double getArrivalTime() {
        return arrivalTime;
    }
    
    public double getDataSize() {
        return dataSize;
    }
    
    public double getComputationalRequirement() {
        return computationalRequirement;
    }
    
    public double getDeadline() {
        return deadline;
    }
    
    public double getStartTime() {
        return startTime;
    }
    
    public void setStartTime(double startTime) {
        this.startTime = startTime;
    }
    
    public double getCompletionTime() {
        return completionTime;
    }
    
    public void setCompletionTime(double completionTime) {
        this.completionTime = completionTime;
    }
    
    public int getSourceDeviceId() {
        return sourceDeviceId;
    }
    
    public int getAssignedServerId() {
        return assignedServerId;
    }
    
    public void setAssignedServerId(int assignedServerId) {
        this.assignedServerId = assignedServerId;
    }
    
    public TaskStatus getStatus() {
        return status;
    }
    
    public void setStatus(TaskStatus status) {
        this.status = status;
    }
    
    public TaskPriority getPriority() {
        return priority;
    }
    
    public void setPriority(TaskPriority priority) {
        this.priority = priority;
    }
    
    @Override
    public String toString() {
        return "Task{" +
                "taskId=" + taskId +
                ", arrivalTime=" + arrivalTime +
                ", dataSize=" + dataSize +
                ", computationalRequirement=" + computationalRequirement +
                ", deadline=" + deadline +
                ", status=" + status +
                ", priority=" + priority +
                '}';
    }
}
