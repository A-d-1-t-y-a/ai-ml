package org.jcora.mec.model;

/**
 * Represents a computational task in the MEC environment.
 * Each task has specific requirements for computation, data transfer, and deadlines.
 */
public class Task {
    private final int id;
    private final double inputDataSize;    // in MB
    private final double outputDataSize;   // in MB
    private final long computationalRequirement; // in Million Instructions (MI)
    private final double deadline;         // in seconds
    private final double arrivalTime;      // in seconds
    
    private double startTime;              // in seconds
    private double finishTime;             // in seconds
    private boolean isOffloaded;           // whether task is offloaded to edge server
    private int assignedDeviceId;          // ID of the device/server processing this task
    private TaskStatus status;             // current status of the task
    
    /**
     * Enum representing the possible states of a task.
     */
    public enum TaskStatus {
        CREATED,
        WAITING,
        PROCESSING,
        COMPLETED,
        FAILED
    }
    
    /**
     * Constructor for creating a new task.
     * 
     * @param id Unique identifier for the task
     * @param inputDataSize Size of input data in MB
     * @param outputDataSize Size of output data in MB
     * @param computationalRequirement Computational requirement in MI
     * @param deadline Maximum time allowed for completion in seconds
     * @param arrivalTime Time when the task arrives in the system in seconds
     */
    public Task(int id, double inputDataSize, double outputDataSize, 
                long computationalRequirement, double deadline, double arrivalTime) {
        this.id = id;
        this.inputDataSize = inputDataSize;
        this.outputDataSize = outputDataSize;
        this.computationalRequirement = computationalRequirement;
        this.deadline = deadline;
        this.arrivalTime = arrivalTime;
        this.status = TaskStatus.CREATED;
    }
    
    // Getters and setters
    
    public int getId() {
        return id;
    }
    
    public double getInputDataSize() {
        return inputDataSize;
    }
    
    public double getOutputDataSize() {
        return outputDataSize;
    }
    
    public long getComputationalRequirement() {
        return computationalRequirement;
    }
    
    public double getDeadline() {
        return deadline;
    }
    
    public double getArrivalTime() {
        return arrivalTime;
    }
    
    public double getStartTime() {
        return startTime;
    }
    
    public void setStartTime(double startTime) {
        this.startTime = startTime;
    }
    
    public double getFinishTime() {
        return finishTime;
    }
    
    public void setFinishTime(double finishTime) {
        this.finishTime = finishTime;
    }
    
    public boolean isOffloaded() {
        return isOffloaded;
    }
    
    public void setOffloaded(boolean offloaded) {
        isOffloaded = offloaded;
    }
    
    public int getAssignedDeviceId() {
        return assignedDeviceId;
    }
    
    public void setAssignedDeviceId(int assignedDeviceId) {
        this.assignedDeviceId = assignedDeviceId;
    }
    
    public TaskStatus getStatus() {
        return status;
    }
    
    public void setStatus(TaskStatus status) {
        this.status = status;
    }
    
    /**
     * Calculate the processing time based on the processing power.
     * 
     * @param mips Processing power in Million Instructions Per Second
     * @return Processing time in seconds
     */
    public double calculateProcessingTime(double mips) {
        return computationalRequirement / mips;
    }
    
    /**
     * Calculate the transmission time based on the bandwidth.
     * 
     * @param bandwidth Available bandwidth in Mbps
     * @return Transmission time in seconds
     */
    public double calculateTransmissionTime(double bandwidth) {
        return inputDataSize / bandwidth;
    }
    
    /**
     * Calculate the response time (total time from arrival to completion).
     * 
     * @return Response time in seconds
     */
    public double calculateResponseTime() {
        return finishTime - arrivalTime;
    }
    
    /**
     * Check if the task meets its deadline.
     * 
     * @return True if the task meets its deadline, false otherwise
     */
    public boolean meetsDeadline() {
        return calculateResponseTime() <= deadline;
    }
    
    @Override
    public String toString() {
        return "Task{" +
                "id=" + id +
                ", inputDataSize=" + inputDataSize +
                ", outputDataSize=" + outputDataSize +
                ", computationalRequirement=" + computationalRequirement +
                ", deadline=" + deadline +
                ", arrivalTime=" + arrivalTime +
                ", status=" + status +
                ", isOffloaded=" + isOffloaded +
                '}';
    }
}
