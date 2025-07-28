package com.nci.fogedge.devices;

import com.nci.fogedge.tasks.Task;

/**
 * Abstract base class for all devices in the simulation.
 * This includes IoT devices, edge nodes, fog nodes, and cloud data centers.
 */
public abstract class Device {
    // Device identification
    protected String id;
    protected DeviceType type;
    
    // Device capabilities
    protected double processingPower; // MIPS (Million Instructions Per Second)
    protected double memory; // MB
    protected double storage; // GB
    protected double batteryCapacity; // mAh
    protected double remainingBattery; // mAh
    
    // Device status
    protected boolean isActive;
    protected boolean isCompromised;
    protected double resourceUtilization; // percentage (0-100)
    
    // Device location
    protected double xPos;
    protected double yPos;
    
    /**
     * Constructor for a device
     * 
     * @param id Unique identifier for the device
     * @param type Type of the device
     * @param processingPower Processing power in MIPS
     * @param memory Memory in MB
     * @param storage Storage in GB
     * @param batteryCapacity Battery capacity in mAh
     * @param xPos Initial X position
     * @param yPos Initial Y position
     */
    public Device(String id, DeviceType type, double processingPower, double memory, 
                  double storage, double batteryCapacity, double xPos, double yPos) {
        this.id = id;
        this.type = type;
        this.processingPower = processingPower;
        this.memory = memory;
        this.storage = storage;
        this.batteryCapacity = batteryCapacity;
        this.remainingBattery = batteryCapacity;
        this.isActive = true;
        this.isCompromised = false;
        this.resourceUtilization = 0.0;
        this.xPos = xPos;
        this.yPos = yPos;
    }
    
    /**
     * Executes a task on this device
     * 
     * @param task The task to execute
     * @return True if the task was executed successfully, false otherwise
     */
    public abstract boolean executeTask(Task task);
    
    /**
     * Consumes energy based on the current operation
     * 
     * @param amount Amount of energy to consume in mAh
     * @return True if the device has enough energy, false otherwise
     */
    public boolean consumeEnergy(double amount) {
        if (remainingBattery >= amount) {
            remainingBattery -= amount;
            
            // Check if the device has run out of battery
            if (remainingBattery <= 0) {
                isActive = false;
                remainingBattery = 0;
            }
            
            return true;
        }
        
        return false;
    }
    
    /**
     * Updates the device's position (for mobile devices)
     * 
     * @param newX New X position
     * @param newY New Y position
     */
    public void updatePosition(double newX, double newY) {
        this.xPos = newX;
        this.yPos = newY;
    }
    
    /**
     * Calculates the distance to another device
     * 
     * @param other The other device
     * @return The distance in meters
     */
    public double distanceTo(Device other) {
        double dx = this.xPos - other.xPos;
        double dy = this.yPos - other.yPos;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Updates the resource utilization of the device
     * 
     * @param utilization The new utilization percentage (0-100)
     */
    public void updateResourceUtilization(double utilization) {
        this.resourceUtilization = Math.max(0, Math.min(100, utilization));
    }
    
    /**
     * Sets the compromised status of the device
     * 
     * @param compromised True if the device is compromised, false otherwise
     */
    public void setCompromised(boolean compromised) {
        this.isCompromised = compromised;
    }
    
    // Getters and setters
    
    public String getId() {
        return id;
    }
    
    public DeviceType getType() {
        return type;
    }
    
    public double getProcessingPower() {
        return processingPower;
    }
    
    public double getMemory() {
        return memory;
    }
    
    public double getStorage() {
        return storage;
    }
    
    public double getBatteryCapacity() {
        return batteryCapacity;
    }
    
    public double getRemainingBattery() {
        return remainingBattery;
    }
    
    public boolean isActive() {
        return isActive;
    }
    
    public void setActive(boolean active) {
        this.isActive = active;
    }
    
    public boolean isCompromised() {
        return isCompromised;
    }
    
    public double getResourceUtilization() {
        return resourceUtilization;
    }
    
    public double getXPos() {
        return xPos;
    }
    
    public double getYPos() {
        return yPos;
    }
}
