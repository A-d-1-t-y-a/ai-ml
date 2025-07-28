package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;
import java.util.Random;

/**
 * Represents an IoT device in the simulation.
 * IoT devices are typically resource-constrained and generate data/tasks.
 */
public class IoTDevice extends Device {
    // IoT device specific properties
    private double mobilityFactor; // How mobile is this device (0-1)
    private double taskGenerationRate; // Tasks per 100 ticks
    private double dataGenerationRate; // KB per tick
    private WirelessType wirelessType;
    private Random random;
    
    /**
     * Constructor for an IoT device
     * 
     * @param id Unique identifier for the device
     * @param name Human-readable name for the device
     * @param xPos Initial X position
     * @param yPos Initial Y position
     * @param cpuCapacity CPU capacity in MIPS
     * @param ramCapacity RAM capacity in MB
     * @param storageCapacity Storage capacity in GB
     * @param batteryCapacity Battery capacity in mAh
     * @param wirelessType Type of wireless connection
     * @param isMobile Whether the device is mobile
     * @param taskGenerationRate Task generation rate (tasks per 100 ticks)
     * @param dataGenerationRate Data generation rate (KB per tick)
     */
    public IoTDevice(String id, String name, double xPos, double yPos, int cpuCapacity,
                    int ramCapacity, int storageCapacity, int batteryCapacity, WirelessType wirelessType,
                    boolean isMobile, double taskGenerationRate, double dataGenerationRate) {
        super(id, DeviceType.IOT_DEVICE, name, xPos, yPos, cpuCapacity, ramCapacity, storageCapacity, batteryCapacity);
        this.mobilityFactor = isMobile ? 0.5 : 0.0; // Set mobility factor based on isMobile flag
        this.taskGenerationRate = taskGenerationRate;
        this.dataGenerationRate = dataGenerationRate;
        this.wirelessType = wirelessType;
        this.random = new Random();
    }
    
    /**
     * Updates the device's position based on mobility factor
     * Higher mobility factor means more movement
     * 
     * @param maxX Maximum X coordinate of the simulation area
     * @param maxY Maximum Y coordinate of the simulation area
     */
    public void move(double maxX, double maxY) {
        if (mobilityFactor > 0) {
            // Calculate movement based on mobility factor
            double movementRange = 10.0 * mobilityFactor; // 10 meters max movement per tick
            
            // Generate random movement
            double deltaX = (random.nextDouble() * 2 - 1) * movementRange; // -movementRange to +movementRange
            double deltaY = (random.nextDouble() * 2 - 1) * movementRange; // -movementRange to +movementRange
            
            // Update position, ensuring it stays within bounds
            double newX = Math.max(0, Math.min(maxX, xPos + deltaX));
            double newY = Math.max(0, Math.min(maxY, yPos + deltaY));
            
            // Update position
            updatePosition(newX, newY);
            
            // Moving consumes energy - more energy for higher mobility
            consumeEnergy(0.01 * mobilityFactor);
        }
    }
    
    /**
     * Generates a new task based on the device's task generation rate
     * 
     * @param currentTick Current simulation tick
     * @param taskLength Task length in MI
     * @param inputSize Task input size in KB
     * @param outputSize Task output size in KB
     * @return A new Task if one should be generated, null otherwise
     */
    public Task generateTask(int currentTick, int taskLength, int inputSize, int outputSize) {
        // Check if a task should be generated this tick
        // taskGenerationRate is tasks per 100 ticks
        double probability = taskGenerationRate / 100.0;
        
        if (random.nextDouble() < probability) {
            // Generate a task
            String taskId = id + "_task_" + currentTick;
            
            // Vary task parameters slightly to simulate real-world variation
            int actualTaskLength = (int) (taskLength * (0.8 + random.nextDouble() * 0.4)); // 80-120% of base length
            int actualInputSize = (int) (inputSize * (0.8 + random.nextDouble() * 0.4)); // 80-120% of base input size
            int actualOutputSize = (int) (outputSize * (0.8 + random.nextDouble() * 0.4)); // 80-120% of base output size
            
            // Creating a task consumes energy
            consumeEnergy(0.05);
            
            // Return the new task
            return new Task(taskId, this.id, actualTaskLength, actualInputSize, actualOutputSize, currentTick);
        }
        
        return null;
    }
    
    /**
     * Generates data (e.g., sensor readings) based on the data generation rate
     * 
     * @param currentTick Current simulation tick
     * @return Amount of data generated in KB
     */
    public double generateData(int currentTick) {
        // Generate data based on the data generation rate
        double dataGenerated = dataGenerationRate;
        
        // Vary data generation slightly to simulate real-world variation
        dataGenerated *= (0.8 + random.nextDouble() * 0.4); // 80-120% of base rate
        
        // Generating data consumes energy
        consumeEnergy(0.01);
        
        return dataGenerated;
    }
    
    /**
     * Executes a task on this IoT device
     * IoT devices have limited resources, so they may not be able to execute all tasks
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
        
        // Check if the device has enough resources to execute the task
        if (task.getLength() > processingPower * 10) { // Assume task needs 10 ticks to complete
            return false;
        }
        
        // Check if the device has enough memory
        if (task.getInputSize() / 1024.0 > memory * 0.1) { // Assume task needs 10% of memory
            return false;
        }
        
        // Calculate energy consumption for task execution
        // More complex tasks consume more energy
        double energyConsumption = 0.1 + (task.getLength() / processingPower) * 0.01;
        
        // Attempt to consume energy
        if (!consumeEnergy(energyConsumption)) {
            return false; // Not enough energy
        }
        
        // Update resource utilization
        double utilization = (task.getLength() / processingPower) * 100;
        updateResourceUtilization(utilization);
        
        return true;
    }
    
    // Getters and setters
    
    public double getMobilityFactor() {
        return mobilityFactor;
    }
    
    public double getTaskGenerationRate() {
        return taskGenerationRate;
    }
    
    public double getDataGenerationRate() {
        return dataGenerationRate;
    }
    
    public WirelessType getWirelessType() {
        return wirelessType;
    }
    
    public void setWirelessType(WirelessType wirelessType) {
        this.wirelessType = wirelessType;
    }
    
    /**
     * Checks if this IoT device is mobile
     * 
     * @return True if the device is mobile (mobility factor > 0), false otherwise
     */
    public boolean isMobile() {
        return mobilityFactor > 0;
    }
}
