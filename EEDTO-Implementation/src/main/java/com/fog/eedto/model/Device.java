package com.fog.eedto.model;

import java.util.ArrayList;
import java.util.List;

import com.fog.eedto.model.Task;

/**
 * Abstract base class for all devices in the EEDTO system (IoT devices, edge servers, cloud servers).
 * Contains common properties and methods for all device types.
 */
public abstract class Device {
    private final int id;
    private final String name;
    private final double mips; // Processing speed in Million Instructions Per Second
    private final int ram; // RAM in MB
    private final long storage; // Storage in MB
    private final double bandwidth; // Bandwidth in Mbps
    private final double energyEfficiency; // Energy efficiency in MIPS per Watt
    private final List<Task> taskQueue;
    private double energyConsumed; // Total energy consumed in Joules
    private double uptime; // Total uptime in seconds
    private boolean active;

    /**
     * Constructor for the Device class
     * 
     * @param id Unique identifier for the device
     * @param name Name of the device
     * @param mips Processing speed in Million Instructions Per Second
     * @param ram RAM in MB
     * @param storage Storage in MB
     * @param bandwidth Bandwidth in Mbps
     * @param energyEfficiency Energy efficiency in MIPS per Watt
     */
    public Device(int id, String name, double mips, int ram, long storage, 
                  double bandwidth, double energyEfficiency) {
        this.id = id;
        this.name = name;
        this.mips = mips;
        this.ram = ram;
        this.storage = storage;
        this.bandwidth = bandwidth;
        this.energyEfficiency = energyEfficiency;
        this.taskQueue = new ArrayList<>();
        this.energyConsumed = 0;
        this.uptime = 0;
        this.active = true;
    }

    // Getters and setters
    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public double getMips() {
        return mips;
    }

    public int getRam() {
        return ram;
    }

    public long getStorage() {
        return storage;
    }

    public double getBandwidth() {
        return bandwidth;
    }

    public double getEnergyEfficiency() {
        return energyEfficiency;
    }

    public List<Task> getTaskQueue() {
        return taskQueue;
    }

    public double getEnergyConsumed() {
        return energyConsumed;
    }

    public void setEnergyConsumed(double energyConsumed) {
        this.energyConsumed = energyConsumed;
    }

    public double getUptime() {
        return uptime;
    }

    public void setUptime(double uptime) {
        this.uptime = uptime;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    /**
     * Add a task to the device's task queue
     * 
     * @param task Task to be added
     */
    public void addTask(Task task) {
        taskQueue.add(task);
    }

    /**
     * Remove a task from the device's task queue
     * 
     * @param task Task to be removed
     * @return true if the task was removed, false otherwise
     */
    public boolean removeTask(Task task) {
        return taskQueue.remove(task);
    }

    /**
     * Calculate the energy consumption for executing a task
     * 
     * @param task Task to be executed
     * @return Energy consumption in Joules
     */
    public double calculateEnergyConsumption(Task task) {
        double executionTime = task.calculateExecutionTime(mips);
        double power = mips / energyEfficiency; // Power in Watts
        return power * executionTime; // Energy in Joules
    }

    /**
     * Calculate the transmission time for sending a task to another device
     * 
     * @param task Task to be transmitted
     * @param targetDevice Target device
     * @return Transmission time in seconds
     */
    public double calculateTransmissionTime(Task task, Device targetDevice) {
        // Calculate the minimum bandwidth between the two devices
        double minBandwidth = Math.min(this.bandwidth, targetDevice.getBandwidth());
        
        // Convert bandwidth from Mbps to Bytes per second (1 Mbps = 125000 Bytes/s)
        double bandwidthBytesPerSecond = minBandwidth * 125000;
        
        // Calculate transmission time for input data
        return task.getInputSize() / bandwidthBytesPerSecond;
    }

    /**
     * Calculate the response time for a task (execution time + transmission time)
     * 
     * @param task Task to be executed
     * @param targetDevice Target device (null if executed locally)
     * @return Response time in seconds
     */
    public double calculateResponseTime(Task task, Device targetDevice) {
        double executionTime = task.calculateExecutionTime(targetDevice != null ? 
                                                         targetDevice.getMips() : mips);
        double transmissionTime = targetDevice != null ? 
                                calculateTransmissionTime(task, targetDevice) : 0;
        return executionTime + transmissionTime;
    }

    /**
     * Execute a task on this device
     * 
     * @param task Task to be executed
     * @param currentTime Current simulation time
     * @return Finish time of the task
     */
    public abstract double executeTask(Task task, double currentTime);

    /**
     * Check if the device can execute a task based on its resource constraints
     * 
     * @param task Task to be checked
     * @return true if the device can execute the task, false otherwise
     */
    public abstract boolean canExecuteTask(Task task);

    @Override
    public String toString() {
        return "Device{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", mips=" + mips +
                ", ram=" + ram +
                ", storage=" + storage +
                ", bandwidth=" + bandwidth +
                ", energyEfficiency=" + energyEfficiency +
                ", active=" + active +
                '}';
    }
}
