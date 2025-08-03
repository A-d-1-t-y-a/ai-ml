package com.fog.eedto.model;

import java.util.Random;
import com.fog.eedto.util.ConfigurationManager;

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
        // Determine task type based on probabilities from configuration
        double lightweightProb = ConfigurationManager.getDouble("task.lightweightProbability", 0.6);
        double mediumProb = ConfigurationManager.getDouble("task.mediumProbability", 0.3);
        double randomValue = random.nextDouble();
        
        Task.TaskType taskType;
        int minSize, maxSize, minMI, maxMI, minDeadline, maxDeadline;
        
        if (randomValue < lightweightProb) {
            taskType = Task.TaskType.LIGHTWEIGHT;
            minSize = ConfigurationManager.getInt("task.lightweight.size.min", 10);
            maxSize = ConfigurationManager.getInt("task.lightweight.size.max", 100);
            minMI = ConfigurationManager.getInt("task.lightweight.mi.min", 100);
            maxMI = ConfigurationManager.getInt("task.lightweight.mi.max", 1000);
            minDeadline = ConfigurationManager.getInt("task.lightweight.deadline.min", 1);
            maxDeadline = ConfigurationManager.getInt("task.lightweight.deadline.max", 5);
        } else if (randomValue < lightweightProb + mediumProb) {
            taskType = Task.TaskType.MEDIUM;
            minSize = ConfigurationManager.getInt("task.medium.size.min", 100);
            maxSize = ConfigurationManager.getInt("task.medium.size.max", 1000);
            minMI = ConfigurationManager.getInt("task.medium.mi.min", 1000);
            maxMI = ConfigurationManager.getInt("task.medium.mi.max", 10000);
            minDeadline = ConfigurationManager.getInt("task.medium.deadline.min", 5);
            maxDeadline = ConfigurationManager.getInt("task.medium.deadline.max", 20);
        } else {
            taskType = Task.TaskType.INTENSIVE;
            minSize = ConfigurationManager.getInt("task.heavyweight.size.min", 1000);
            maxSize = ConfigurationManager.getInt("task.heavyweight.size.max", 10000);
            minMI = ConfigurationManager.getInt("task.heavyweight.mi.min", 10000);
            maxMI = ConfigurationManager.getInt("task.heavyweight.mi.max", 100000);
            minDeadline = ConfigurationManager.getInt("task.heavyweight.deadline.min", 20);
            maxDeadline = ConfigurationManager.getInt("task.heavyweight.deadline.max", 60);
        }
        
        // Generate random task parameters based on configuration values
        long length = minMI + random.nextInt(maxMI - minMI + 1);
        long inputSize = minSize + random.nextInt(maxSize - minSize + 1);
        long outputSize = Math.max(1, inputSize / 2); // Output is typically smaller than input
        double deadline = currentTime + (minDeadline + random.nextInt(maxDeadline - minDeadline + 1));
        
        // Convert KB to bytes
        inputSize *= 1024;
        outputSize *= 1024;
        
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
