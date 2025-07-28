package com.nci.fogedge.tasks;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;

import java.util.*;

/**
 * Manages tasks in the simulation, including creation, assignment, and tracking.
 * This class is responsible for orchestrating task execution across devices.
 */
public class TaskManager {
    private SimulationConfig config;
    private SimulationResults results;
    private Map<String, Task> tasks; // All tasks indexed by ID
    private Queue<Task> taskQueue; // Queue of tasks waiting to be assigned
    private Map<String, List<Task>> deviceTasks; // Tasks assigned to each device
    private Map<String, List<Task>> completedTasks; // Completed tasks for each device
    private Map<String, List<Task>> failedTasks; // Failed tasks for each device
    private Random random;
    
    /**
     * Constructor for TaskManager
     * 
     * @param config Simulation configuration
     * @param results Simulation results collector
     */
    public TaskManager(SimulationConfig config, SimulationResults results) {
        this.config = config;
        this.results = results;
        this.tasks = new HashMap<>();
        this.taskQueue = new PriorityQueue<>(Comparator.comparingInt(Task::getPriority).reversed());
        this.deviceTasks = new HashMap<>();
        this.completedTasks = new HashMap<>();
        this.failedTasks = new HashMap<>();
        this.random = new Random();
    }
    
    /**
     * Initializes the TaskManager
     */
    public void initialize() {
        tasks.clear();
        taskQueue.clear();
        deviceTasks.clear();
        completedTasks.clear();
        failedTasks.clear();
    }
    
    /**
     * Generates tasks from IoT devices based on their task generation rates
     * 
     * @param iotDevices List of IoT devices
     * @param currentTick Current simulation tick
     */
    public void generateTasks(List<IoTDevice> iotDevices, int currentTick) {
        for (IoTDevice device : iotDevices) {
            // Check if the device is active
            if (!device.isActive()) {
                continue;
            }
            
            // Generate task based on the device's task generation rate
            int taskLength = config.getTaskLength();
            int inputSize = config.getTaskInputSize();
            int outputSize = config.getTaskOutputSize();
            
            Task task = device.generateTask(currentTick, taskLength, inputSize, outputSize);
            
            if (task != null) {
                // Add task to the task list and queue
                tasks.put(task.getId(), task);
                taskQueue.add(task);
                
                // Initialize device task list if needed
                deviceTasks.computeIfAbsent(device.getId(), k -> new ArrayList<>()).add(task);
                
                // Update statistics
                results.incrementTotalTasksGenerated();
            }
        }
    }
    
    /**
     * Assigns tasks to devices based on the task scheduling policy
     * 
     * @param devices Map of all devices indexed by ID
     * @param currentTick Current simulation tick
     */
    public void assignTasks(Map<String, Device> devices, int currentTick) {
        // Process the task queue
        List<Task> unassignedTasks = new ArrayList<>();
        
        while (!taskQueue.isEmpty()) {
            Task task = taskQueue.poll();
            
            // Skip tasks that are not in CREATED or READY state
            if (task.getStatus() != TaskStatus.CREATED && task.getStatus() != TaskStatus.READY) {
                continue;
            }
            
            // Set task to READY state
            task.setReady();
            
            // Get the source device
            String sourceDeviceId = task.getSourceDeviceId();
            Device sourceDevice = devices.get(sourceDeviceId);
            
            if (sourceDevice == null) {
                // Source device not found, mark task as failed
                task.setFailed();
                failedTasks.computeIfAbsent(sourceDeviceId, k -> new ArrayList<>()).add(task);
                results.incrementFailedTasks();
                continue;
            }
            
            // Determine where to execute the task based on the scheduling policy
            String executorDeviceId = selectExecutorDevice(task, sourceDevice, devices);
            
            if (executorDeviceId == null) {
                // No suitable executor found, put back in queue for next tick
                unassignedTasks.add(task);
                continue;
            }
            
            // Get the executor device
            Device executorDevice = devices.get(executorDeviceId);
            
            if (executorDevice == null) {
                // Executor device not found, mark task as failed
                task.setFailed();
                failedTasks.computeIfAbsent(sourceDeviceId, k -> new ArrayList<>()).add(task);
                results.incrementFailedTasks();
                continue;
            }
            
            // Try to execute the task on the selected device
            boolean executed = executorDevice.executeTask(task);
            
            if (executed) {
                // Task execution started
                task.setRunning(currentTick, executorDeviceId);
                
                // Add to device tasks if not the source device (offloaded)
                if (!executorDeviceId.equals(sourceDeviceId)) {
                    deviceTasks.computeIfAbsent(executorDeviceId, k -> new ArrayList<>()).add(task);
                    task.setOffloaded(executorDeviceId);
                    results.incrementOffloadedTasks();
                }
            } else {
                // Execution failed, put back in queue for next tick
                unassignedTasks.add(task);
            }
        }
        
        // Put unassigned tasks back in the queue
        taskQueue.addAll(unassignedTasks);
    }
    
    /**
     * Selects an executor device for a task based on the scheduling policy
     * 
     * @param task Task to be executed
     * @param sourceDevice Device that generated the task
     * @param devices Map of all devices indexed by ID
     * @return ID of the selected executor device, or null if no suitable device found
     */
    private String selectExecutorDevice(Task task, Device sourceDevice, Map<String, Device> devices) {
        String schedulingPolicy = config.getTaskSchedulingPolicy();
        
        switch (schedulingPolicy) {
            case "LOCAL_ONLY":
                // Execute on the source device only
                return sourceDevice.getId();
                
            case "EDGE_ONLY":
                // Find the nearest edge node
                return findNearestDeviceOfType(sourceDevice, devices, DeviceType.EDGE_NODE);
                
            case "FOG_ONLY":
                // Find the nearest fog node
                return findNearestDeviceOfType(sourceDevice, devices, DeviceType.FOG_NODE);
                
            case "CLOUD_ONLY":
                // Find a cloud datacenter
                return findNearestDeviceOfType(sourceDevice, devices, DeviceType.CLOUD_DATACENTER);
                
            case "NEAREST_DEVICE":
                // Try local execution first
                if (canExecuteTask(task, sourceDevice)) {
                    return sourceDevice.getId();
                }
                
                // Try edge node
                String edgeNodeId = findNearestDeviceOfType(sourceDevice, devices, DeviceType.EDGE_NODE);
                if (edgeNodeId != null && canExecuteTask(task, devices.get(edgeNodeId))) {
                    return edgeNodeId;
                }
                
                // Try fog node
                String fogNodeId = findNearestDeviceOfType(sourceDevice, devices, DeviceType.FOG_NODE);
                if (fogNodeId != null && canExecuteTask(task, devices.get(fogNodeId))) {
                    return fogNodeId;
                }
                
                // Try cloud datacenter
                String cloudId = findNearestDeviceOfType(sourceDevice, devices, DeviceType.CLOUD_DATACENTER);
                if (cloudId != null && canExecuteTask(task, devices.get(cloudId))) {
                    return cloudId;
                }
                
                // No suitable device found
                return null;
                
            case "RANDOM":
                // Select a random device
                List<String> deviceIds = new ArrayList<>(devices.keySet());
                if (deviceIds.isEmpty()) {
                    return null;
                }
                return deviceIds.get(random.nextInt(deviceIds.size()));
                
            case "LOAD_BALANCING":
                // Select the device with the lowest resource utilization
                return findDeviceWithLowestUtilization(devices);
                
            default:
                // Default to local execution
                return sourceDevice.getId();
        }
    }
    
    /**
     * Finds the nearest device of a specific type to the source device
     * 
     * @param sourceDevice Source device
     * @param devices Map of all devices indexed by ID
     * @param deviceType Type of device to find
     * @return ID of the nearest device, or null if no device of that type found
     */
    private String findNearestDeviceOfType(Device sourceDevice, Map<String, Device> devices, DeviceType deviceType) {
        double minDistance = Double.MAX_VALUE;
        String nearestDeviceId = null;
        
        for (Map.Entry<String, Device> entry : devices.entrySet()) {
            Device device = entry.getValue();
            
            // Skip inactive devices and devices of the wrong type
            if (!device.isActive() || device.getType() != deviceType) {
                continue;
            }
            
            // Calculate distance
            double distance = calculateDistance(sourceDevice, device);
            
            // Update nearest device if this one is closer
            if (distance < minDistance) {
                minDistance = distance;
                nearestDeviceId = entry.getKey();
            }
        }
        
        return nearestDeviceId;
    }
    
    /**
     * Calculates the Euclidean distance between two devices
     * 
     * @param device1 First device
     * @param device2 Second device
     * @return Distance between the devices
     */
    private double calculateDistance(Device device1, Device device2) {
        double dx = device1.getXPos() - device2.getXPos();
        double dy = device1.getYPos() - device2.getYPos();
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Finds the device with the lowest resource utilization
     * 
     * @param devices Map of all devices indexed by ID
     * @return ID of the device with the lowest utilization, or null if no active devices
     */
    private String findDeviceWithLowestUtilization(Map<String, Device> devices) {
        double minUtilization = Double.MAX_VALUE;
        String lowestUtilizationDeviceId = null;
        
        for (Map.Entry<String, Device> entry : devices.entrySet()) {
            Device device = entry.getValue();
            
            // Skip inactive devices
            if (!device.isActive()) {
                continue;
            }
            
            // Get resource utilization
            double utilization = device.getResourceUtilization();
            
            // Update device with lowest utilization
            if (utilization < minUtilization) {
                minUtilization = utilization;
                lowestUtilizationDeviceId = entry.getKey();
            }
        }
        
        return lowestUtilizationDeviceId;
    }
    
    /**
     * Checks if a device can execute a task
     * 
     * @param task Task to be executed
     * @param device Device to check
     * @return True if the device can execute the task, false otherwise
     */
    private boolean canExecuteTask(Task task, Device device) {
        // Check if the device is active
        if (!device.isActive()) {
            return false;
        }
        
        // Simple check based on task length and device processing power
        return task.getLength() <= device.getProcessingPower() * 10;
    }
    
    /**
     * Updates the status of running tasks
     * 
     * @param devices Map of all devices indexed by ID
     * @param currentTick Current simulation tick
     */
    public void updateTaskStatus(Map<String, Device> devices, int currentTick) {
        // Iterate through all tasks
        for (Task task : tasks.values()) {
            // Only process running tasks
            if (task.getStatus() != TaskStatus.RUNNING) {
                continue;
            }
            
            // Get the executor device
            String executorDeviceId = task.getExecutorDeviceId();
            Device executorDevice = devices.get(executorDeviceId);
            
            if (executorDevice == null || !executorDevice.isActive()) {
                // Executor device not found or inactive, mark task as failed
                task.setFailed();
                failedTasks.computeIfAbsent(task.getSourceDeviceId(), k -> new ArrayList<>()).add(task);
                results.incrementFailedTasks();
                continue;
            }
            
            // Calculate task execution time based on task length and device processing power
            int executionTime = (int) Math.ceil(task.getLength() / executorDevice.getProcessingPower());
            
            // Check if the task has completed
            if (currentTick - task.getStartTick() >= executionTime) {
                // Task completed
                task.setCompleted(currentTick);
                
                // Move from device tasks to completed tasks
                List<Task> deviceTaskList = deviceTasks.getOrDefault(executorDeviceId, new ArrayList<>());
                deviceTaskList.remove(task);
                
                completedTasks.computeIfAbsent(executorDeviceId, k -> new ArrayList<>()).add(task);
                
                // Update statistics
                results.incrementCompletedTasks();
                results.addTaskExecutionTime(task.getExecutionTime());
                results.addTaskWaitingTime(task.getWaitingTime());
                results.addTaskResponseTime(task.getResponseTime());
            }
        }
    }
    
    /**
     * Gets all tasks
     * 
     * @return Map of all tasks indexed by ID
     */
    public Map<String, Task> getTasks() {
        return tasks;
    }
    
    /**
     * Gets the task queue
     * 
     * @return Queue of tasks waiting to be assigned
     */
    public Queue<Task> getTaskQueue() {
        return taskQueue;
    }
    
    /**
     * Gets tasks assigned to a specific device
     * 
     * @param deviceId ID of the device
     * @return List of tasks assigned to the device
     */
    public List<Task> getDeviceTasks(String deviceId) {
        return deviceTasks.getOrDefault(deviceId, new ArrayList<>());
    }
    
    /**
     * Gets completed tasks for a specific device
     * 
     * @param deviceId ID of the device
     * @return List of completed tasks for the device
     */
    public List<Task> getCompletedTasks(String deviceId) {
        return completedTasks.getOrDefault(deviceId, new ArrayList<>());
    }
    
    /**
     * Gets failed tasks for a specific device
     * 
     * @param deviceId ID of the device
     * @return List of failed tasks for the device
     */
    public List<Task> getFailedTasks(String deviceId) {
        return failedTasks.getOrDefault(deviceId, new ArrayList<>());
    }
    
    /**
     * Gets the total number of tasks
     * 
     * @return Total number of tasks
     */
    public int getTotalTaskCount() {
        return tasks.size();
    }
    
    /**
     * Gets the number of completed tasks
     * 
     * @return Number of completed tasks
     */
    public int getCompletedTaskCount() {
        int count = 0;
        for (List<Task> taskList : completedTasks.values()) {
            count += taskList.size();
        }
        return count;
    }
    
    /**
     * Gets the number of failed tasks
     * 
     * @return Number of failed tasks
     */
    public int getFailedTaskCount() {
        int count = 0;
        for (List<Task> taskList : failedTasks.values()) {
            count += taskList.size();
        }
        return count;
    }
    
    /**
     * Gets the number of tasks in the queue
     * 
     * @return Number of tasks in the queue
     */
    public int getQueuedTaskCount() {
        return taskQueue.size();
    }
}
