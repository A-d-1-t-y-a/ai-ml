package com.fog.eedto.model;

import java.util.Random;

/**
 * Represents an IoT device in the EEDTO system.
 * IoT devices have limited resources and generate tasks that can be executed locally or offloaded.
 */
public class IoTDevice extends Device {
    private final double batteryCapacity; // Battery capacity in Joules
    private double remainingBattery; // Remaining battery in Joules
    private final double idlePower; // Power consumption when idle in Watts
    private final Random random;

    /**
     * Constructor for the IoTDevice class
     * 
     * @param id Unique identifier for the IoT device
     * @param name Name of the IoT device
     * @param mips Processing speed in Million Instructions Per Second
     * @param ram RAM in MB
     * @param storage Storage in MB
     * @param bandwidth Bandwidth in Mbps
     * @param energyEfficiency Energy efficiency in MIPS per Watt
     * @param batteryCapacity Battery capacity in Joules
     * @param idlePower Power consumption when idle in Watts
     */
    public IoTDevice(int id, String name, double mips, int ram, long storage, 
                    double bandwidth, double energyEfficiency, double batteryCapacity, 
                    double idlePower) {
        super(id, name, mips, ram, storage, bandwidth, energyEfficiency);
        this.batteryCapacity = batteryCapacity;
        this.remainingBattery = batteryCapacity;
        this.idlePower = idlePower;
        this.random = new Random();
    }

    // Getters and setters
    public double getBatteryCapacity() {
        return batteryCapacity;
    }

    public double getRemainingBattery() {
        return remainingBattery;
    }

    public void setRemainingBattery(double remainingBattery) {
        this.remainingBattery = Math.max(0, Math.min(batteryCapacity, remainingBattery));
    }

    public double getIdlePower() {
        return idlePower;
    }

    /**
     * Generate a new task with random parameters
     * 
     * @param taskId Unique identifier for the task
     * @param currentTime Current simulation time
     * @return Generated task
     */
    public Task generateTask(int taskId, double currentTime) {
        // Generate random task parameters based on realistic values
        long length = 1000 + random.nextInt(9000); // 1000-10000 MI
        long inputSize = 10 + random.nextInt(990); // 10-1000 KB
        long outputSize = 1 + random.nextInt(99); // 1-100 KB
        double deadline = currentTime + (5 + random.nextInt(16)); // 5-20 seconds from now
        
        // Convert KB to bytes
        inputSize *= 1024;
        outputSize *= 1024;
        
        // Determine task type based on length
        Task.TaskType taskType;
        if (length < 3000) {
            taskType = Task.TaskType.LIGHTWEIGHT;
        } else if (length < 7000) {
            taskType = Task.TaskType.MEDIUM;
        } else {
            taskType = Task.TaskType.INTENSIVE;
        }
        
        return new Task(taskId, length, inputSize, outputSize, deadline, currentTime, taskType);
    }

    /**
     * Update the battery level based on the time elapsed and power consumption
     * 
     * @param elapsedTime Time elapsed in seconds
     * @param isIdle Whether the device is idle or executing a task
     */
    public void updateBatteryLevel(double elapsedTime, boolean isIdle) {
        double powerConsumption = isIdle ? idlePower : (getMips() / getEnergyEfficiency());
        double energyConsumed = powerConsumption * elapsedTime;
        
        // Update remaining battery
        setRemainingBattery(remainingBattery - energyConsumed);
        
        // Update total energy consumed
        setEnergyConsumed(getEnergyConsumed() + energyConsumed);
    }

    /**
     * Check if the battery level is sufficient to execute a task
     * 
     * @param task Task to be executed
     * @return true if the battery level is sufficient, false otherwise
     */
    public boolean hasSufficientBattery(Task task) {
        double energyRequired = calculateEnergyConsumption(task);
        return remainingBattery >= energyRequired;
    }

    @Override
    public double executeTask(Task task, double currentTime) {
        if (!canExecuteTask(task)) {
            throw new IllegalStateException("Device cannot execute this task due to resource constraints");
        }
        
        // Set task status to executing
        task.setStatus(Task.TaskStatus.EXECUTING);
        task.setStartTime(currentTime);
        task.setExecutionLocation(Task.DeviceType.IOT_DEVICE);
        
        // Calculate execution time
        double executionTime = task.calculateExecutionTime(getMips());
        
        // Calculate energy consumption
        double energyConsumption = calculateEnergyConsumption(task);
        task.setEnergyConsumed(energyConsumption);
        
        // Update battery level
        setRemainingBattery(remainingBattery - energyConsumption);
        
        // Update total energy consumed
        setEnergyConsumed(getEnergyConsumed() + energyConsumption);
        
        // Set task finish time
        double finishTime = currentTime + executionTime;
        task.setFinishTime(finishTime);
        
        // Set task status to completed
        task.setStatus(Task.TaskStatus.COMPLETED);
        
        return finishTime;
    }

    @Override
    public boolean canExecuteTask(Task task) {
        // Check if the device has sufficient resources to execute the task
        boolean hasSufficientMips = getMips() > 0;
        boolean hasSufficientRam = task.getInputSize() / 1024 <= getRam(); // Convert bytes to KB
        boolean hasSufficientStorage = task.getOutputSize() / 1024 <= getStorage(); // Convert bytes to KB
        boolean hasSufficientBattery = hasSufficientBattery(task);
        
        return isActive() && hasSufficientMips && hasSufficientRam && 
               hasSufficientStorage && hasSufficientBattery;
    }

    @Override
    public String toString() {
        return "IoTDevice{" +
                "id=" + getId() +
                ", name='" + getName() + '\'' +
                ", mips=" + getMips() +
                ", ram=" + getRam() +
                ", storage=" + getStorage() +
                ", bandwidth=" + getBandwidth() +
                ", energyEfficiency=" + getEnergyEfficiency() +
                ", batteryCapacity=" + batteryCapacity +
                ", remainingBattery=" + remainingBattery +
                ", active=" + isActive() +
                '}';
    }
}
