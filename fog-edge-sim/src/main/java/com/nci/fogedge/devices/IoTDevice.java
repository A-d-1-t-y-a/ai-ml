package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Represents an IoT device in the simulation.
 * IoT devices generate tasks and can either process them locally or offload them.
 */
public class IoTDevice extends Device {
    private WirelessType wirelessType;
    private boolean isMobile;
    private double taskGenerationRate; // Tasks per second
    private double mobilitySpeed; // Meters per second (for mobile devices)
    private List<Task> generatedTasks;
    private List<Task> completedTasks;
    private double dataGenerationRate; // KB per second
    
    /**
     * Constructor for an IoT device
     * 
     * @param id Unique identifier for the device
     * @param name Human-readable name for the device
     * @param xPos Initial X position
     * @param yPos Initial Y position
     * @param processingPower Processing power in MIPS
     * @param memory Memory in MB
     * @param storage Storage in GB
     * @param batteryCapacity Battery capacity in mAh
     * @param wirelessType Wireless communication type
     * @param isMobile Whether the device is mobile
     * @param taskGenerationRate Task generation rate in tasks per second
     * @param dataGenerationRate Data generation rate in KB per second
     */
    public IoTDevice(String id, String name, double xPos, double yPos,
                     double processingPower, double memory, double storage, double batteryCapacity,
                     WirelessType wirelessType, boolean isMobile, double taskGenerationRate,
                     double dataGenerationRate) {
        super(id, DeviceType.IOT_DEVICE, name, xPos, yPos, processingPower, memory, storage, batteryCapacity);
        this.wirelessType = wirelessType;
        this.isMobile = isMobile;
        this.taskGenerationRate = taskGenerationRate;
        this.dataGenerationRate = dataGenerationRate;
        this.mobilitySpeed = isMobile ? (new Random().nextDouble() * 2.0 + 0.5) : 0.0; // 0.5-2.5 m/s for mobile devices
        this.generatedTasks = new ArrayList<>();
        this.completedTasks = new ArrayList<>();
    }
    
    /**
     * Generates a new task based on the device's task generation rate
     * 
     * @param currentTime Current simulation time in seconds
     * @param random Random number generator
     * @param minCpu Minimum CPU requirement for tasks
     * @param maxCpu Maximum CPU requirement for tasks
     * @param minRam Minimum RAM requirement for tasks
     * @param maxRam Maximum RAM requirement for tasks
     * @param minStorage Minimum storage requirement for tasks
     * @param maxStorage Maximum storage requirement for tasks
     * @param minDuration Minimum task duration
     * @param maxDuration Maximum task duration
     * @return The generated task, or null if no task was generated
     */
    public Task generateTask(double currentTime, Random random,
                            double minCpu, double maxCpu,
                            double minRam, double maxRam,
                            double minStorage, double maxStorage,
                            double minDuration, double maxDuration) {
        // Check if a task should be generated based on the task generation rate
        if (random.nextDouble() > taskGenerationRate) {
            return null;
        }
        
        // Generate random task requirements
        double cpuReq = minCpu + random.nextDouble() * (maxCpu - minCpu);
        double ramReq = minRam + random.nextDouble() * (maxRam - minRam);
        double storageReq = minStorage + random.nextDouble() * (maxStorage - minStorage);
        double duration = minDuration + random.nextDouble() * (maxDuration - minDuration);
        
        // Create the task
        Task task = new Task(
            "TASK_" + id + "_" + generatedTasks.size(),
            this,
            cpuReq,
            ramReq,
            storageReq,
            duration,
            currentTime
        );
        
        generatedTasks.add(task);
        return task;
    }
    
    /**
     * Moves the device if it is mobile
     * 
     * @param timeStep Time step in seconds
     * @param random Random number generator
     * @param areaWidth Width of the simulation area
     * @param areaHeight Height of the simulation area
     */
    public void move(double timeStep, Random random, double areaWidth, double areaHeight) {
        if (!isMobile || !isActive) {
            return;
        }
        
        // Calculate distance to move based on speed and time step
        double distance = mobilitySpeed * timeStep;
        
        // Generate random direction
        double angle = random.nextDouble() * 2 * Math.PI;
        double dx = distance * Math.cos(angle);
        double dy = distance * Math.sin(angle);
        
        // Calculate new position
        double newX = xPos + dx;
        double newY = yPos + dy;
        
        // Ensure the device stays within the simulation area (bounce off the edges)
        if (newX < 0) {
            newX = -newX;
        } else if (newX > areaWidth) {
            newX = 2 * areaWidth - newX;
        }
        
        if (newY < 0) {
            newY = -newY;
        } else if (newY > areaHeight) {
            newY = 2 * areaHeight - newY;
        }
        
        // Update position
        updatePosition(newX, newY);
        
        // Consume energy for movement
        consumeEnergy(0.1 * timeStep); // Simple model: 0.1 mAh per second of movement
    }
    
    /**
     * Executes a task on this device
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
        
        // Check if the device has enough resources
        if (task.getCpuRequirement() > processingPower ||
            task.getRamRequirement() > memory ||
            task.getStorageRequirement() > storage) {
            return false;
        }
        
        // Calculate energy consumption for task execution
        // Simple model: energy consumption is proportional to task requirements and duration
        double energyConsumption = (task.getCpuRequirement() / processingPower) * 
                                   (task.getDuration() / 60.0) * 10.0; // mAh
        
        // Consume energy
        if (!consumeEnergy(energyConsumption)) {
            return false;
        }
        
        // Update resource utilization
        double utilization = (task.getCpuRequirement() / processingPower) * 100.0;
        updateResourceUtilization(utilization);
        
        // Mark task as completed
        completedTasks.add(task);
        
        return true;
    }
    
    /**
     * Gets the wireless type of the device
     * 
     * @return The wireless type
     */
    public WirelessType getWirelessType() {
        return wirelessType;
    }
    
    /**
     * Checks if the device is mobile
     * 
     * @return True if the device is mobile, false otherwise
     */
    public boolean isMobile() {
        return isMobile;
    }
    
    /**
     * Gets the task generation rate of the device
     * 
     * @return The task generation rate in tasks per second
     */
    public double getTaskGenerationRate() {
        return taskGenerationRate;
    }
    
    /**
     * Gets the mobility speed of the device
     * 
     * @return The mobility speed in meters per second
     */
    public double getMobilitySpeed() {
        return mobilitySpeed;
    }
    
    /**
     * Gets the list of tasks generated by this device
     * 
     * @return The list of generated tasks
     */
    public List<Task> getGeneratedTasks() {
        return new ArrayList<>(generatedTasks);
    }
    
    /**
     * Gets the list of tasks completed by this device
     * 
     * @return The list of completed tasks
     */
    public List<Task> getCompletedTasks() {
        return new ArrayList<>(completedTasks);
    }
    
    /**
     * Gets the data generation rate of the device
     * 
     * @return The data generation rate in KB per second
     */
    public double getDataGenerationRate() {
        return dataGenerationRate;
    }
    
    /**
     * Sets the wireless type of the device
     * 
     * @param wirelessType The wireless type
     */
    public void setWirelessType(WirelessType wirelessType) {
        this.wirelessType = wirelessType;
    }
    
    /**
     * Sets the task generation rate of the device
     * 
     * @param taskGenerationRate The task generation rate in tasks per second
     */
    public void setTaskGenerationRate(double taskGenerationRate) {
        this.taskGenerationRate = taskGenerationRate;
    }
    
    /**
     * Sets the data generation rate of the device
     * 
     * @param dataGenerationRate The data generation rate in KB per second
     */
    public void setDataGenerationRate(double dataGenerationRate) {
        this.dataGenerationRate = dataGenerationRate;
    }
    
    /**
     * Returns a string representation of the IoT device
     * 
     * @return String representation of the IoT device
     */
    @Override
    public String toString() {
        return "IoTDevice{" +
               "id='" + id + '\'' +
               ", name='" + name + '\'' +
               ", wirelessType=" + wirelessType +
               ", isMobile=" + isMobile +
               ", isActive=" + isActive +
               ", isCompromised=" + isCompromised +
               ", position=(" + xPos + ", " + yPos + ")" +
               ", processingPower=" + processingPower +
               ", memory=" + memory +
               ", storage=" + storage +
               ", batteryCapacity=" + batteryCapacity +
               ", remainingBattery=" + remainingBattery +
               ", resourceUtilization=" + resourceUtilization +
               ", taskGenerationRate=" + taskGenerationRate +
               ", dataGenerationRate=" + dataGenerationRate +
               '}';
    }
}
