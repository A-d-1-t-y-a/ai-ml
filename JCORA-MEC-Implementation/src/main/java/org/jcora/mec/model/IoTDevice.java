package org.jcora.mec.model;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents an IoT device in the MEC environment.
 * IoT devices generate tasks and can either process them locally or offload them to edge servers.
 */
public class IoTDevice {
    private final int id;
    private final String name;
    private final double processingPower;      // in MIPS (Million Instructions Per Second)
    private final double energyConsumption;    // in Watts when processing at full capacity
    private final double idleEnergyConsumption; // in Watts when idle
    private final double transmissionPower;    // in Watts when transmitting data
    private final double batteryCapacity;      // in Joules
    private double remainingBattery;           // in Joules
    
    private final List<Task> taskQueue;
    private Task currentTask;
    private double totalEnergyConsumed;
    private double totalProcessingTime;
    private int completedTasks;
    private int failedTasks;
    
    /**
     * Constructor for creating a new IoT device.
     * 
     * @param id Unique identifier for the device
     * @param name Name of the device
     * @param processingPower Processing power in MIPS
     * @param energyConsumption Energy consumption in Watts when processing
     * @param idleEnergyConsumption Energy consumption in Watts when idle
     * @param transmissionPower Power consumption in Watts when transmitting
     * @param batteryCapacity Battery capacity in Joules
     */
    public IoTDevice(int id, String name, double processingPower, double energyConsumption,
                    double idleEnergyConsumption, double transmissionPower, double batteryCapacity) {
        this.id = id;
        this.name = name;
        this.processingPower = processingPower;
        this.energyConsumption = energyConsumption;
        this.idleEnergyConsumption = idleEnergyConsumption;
        this.transmissionPower = transmissionPower;
        this.batteryCapacity = batteryCapacity;
        this.remainingBattery = batteryCapacity;
        
        this.taskQueue = new ArrayList<>();
        this.totalEnergyConsumed = 0.0;
        this.totalProcessingTime = 0.0;
        this.completedTasks = 0;
        this.failedTasks = 0;
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
    
    public double getTransmissionPower() {
        return transmissionPower;
    }
    
    public double getBatteryCapacity() {
        return batteryCapacity;
    }
    
    public double getRemainingBattery() {
        return remainingBattery;
    }
    
    public List<Task> getTaskQueue() {
        return new ArrayList<>(taskQueue);
    }
    
    public Task getCurrentTask() {
        return currentTask;
    }
    
    public void setCurrentTask(Task currentTask) {
        this.currentTask = currentTask;
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
    
    /**
     * Add a task to the device's queue.
     * 
     * @param task Task to be added
     */
    public void addTask(Task task) {
        taskQueue.add(task);
    }
    
    /**
     * Process a task locally on the device.
     * 
     * @param task Task to be processed
     * @param currentTime Current simulation time
     * @return True if the task was successfully processed, false otherwise
     */
    public boolean processTaskLocally(Task task, double currentTime) {
        double processingTime = task.calculateProcessingTime(processingPower);
        double energyRequired = processingTime * energyConsumption;
        
        // Check if there's enough battery to process the task
        if (remainingBattery < energyRequired) {
            task.setStatus(Task.TaskStatus.FAILED);
            failedTasks++;
            return false;
        }
        
        // Process the task
        task.setStartTime(currentTime);
        task.setFinishTime(currentTime + processingTime);
        task.setStatus(Task.TaskStatus.COMPLETED);
        task.setOffloaded(false);
        task.setAssignedDeviceId(this.id);
        
        // Update device state
        remainingBattery -= energyRequired;
        totalEnergyConsumed += energyRequired;
        totalProcessingTime += processingTime;
        completedTasks++;
        
        return true;
    }
    
    /**
     * Offload a task to an edge server.
     * 
     * @param task Task to be offloaded
     * @param bandwidth Available bandwidth in Mbps
     * @param currentTime Current simulation time
     * @return Energy consumed for offloading in Joules
     */
    public double offloadTask(Task task, double bandwidth, double currentTime) {
        double transmissionTime = task.calculateTransmissionTime(bandwidth);
        double energyRequired = transmissionTime * transmissionPower;
        
        // Check if there's enough battery to transmit the task
        if (remainingBattery < energyRequired) {
            task.setStatus(Task.TaskStatus.FAILED);
            failedTasks++;
            return 0.0;
        }
        
        // Offload the task
        task.setStartTime(currentTime);
        task.setOffloaded(true);
        
        // Update device state
        remainingBattery -= energyRequired;
        totalEnergyConsumed += energyRequired;
        
        return energyRequired;
    }
    
    /**
     * Calculate the energy consumption for processing a task locally.
     * 
     * @param task Task to be processed
     * @return Energy consumption in Joules
     */
    public double calculateLocalProcessingEnergy(Task task) {
        double processingTime = task.calculateProcessingTime(processingPower);
        return processingTime * energyConsumption;
    }
    
    /**
     * Calculate the energy consumption for offloading a task.
     * 
     * @param task Task to be offloaded
     * @param bandwidth Available bandwidth in Mbps
     * @return Energy consumption in Joules
     */
    public double calculateOffloadingEnergy(Task task, double bandwidth) {
        double transmissionTime = task.calculateTransmissionTime(bandwidth);
        return transmissionTime * transmissionPower;
    }
    
    /**
     * Update the device's energy consumption during idle time.
     * 
     * @param idleTime Time spent idle in seconds
     */
    public void consumeIdleEnergy(double idleTime) {
        double energyConsumed = idleTime * idleEnergyConsumption;
        remainingBattery -= energyConsumed;
        totalEnergyConsumed += energyConsumed;
    }
    
    @Override
    public String toString() {
        return "IoTDevice{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", processingPower=" + processingPower +
                ", remainingBattery=" + remainingBattery + "/" + batteryCapacity +
                ", completedTasks=" + completedTasks +
                ", failedTasks=" + failedTasks +
                '}';
    }
}
