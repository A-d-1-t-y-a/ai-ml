package com.nci.fogedge.tasks;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.core.SimulationConfig;
import com.nci.fogedge.core.SimulationResults;
import com.nci.fogedge.network.NetworkModel;
import com.nci.fogedge.security.SecurityManager;

import java.util.*;

/**
 * Manages tasks in the simulation, including generation, scheduling, and execution.
 */
public class TaskManager {
    private SimulationConfig config;
    private SimulationResults results;
    private NetworkModel networkModel;
    private SecurityManager securityManager;
    
    private List<Task> allTasks;
    private List<Task> pendingTasks;
    private List<Task> executingTasks;
    private List<Task> completedTasks;
    private List<Task> failedTasks;
    
    private Random random;
    private TaskSchedulingPolicy schedulingPolicy;
    
    /**
     * Constructor for TaskManager
     * 
     * @param config Simulation configuration
     * @param results Simulation results
     * @param networkModel Network model
     * @param securityManager Security manager
     */
    public TaskManager(SimulationConfig config, SimulationResults results, 
                      NetworkModel networkModel, SecurityManager securityManager) {
        this.config = config;
        this.results = results;
        this.networkModel = networkModel;
        this.securityManager = securityManager;
        
        this.allTasks = new ArrayList<>();
        this.pendingTasks = new ArrayList<>();
        this.executingTasks = new ArrayList<>();
        this.completedTasks = new ArrayList<>();
        this.failedTasks = new ArrayList<>();
        
        this.random = new Random(config.getRandomSeed());
        this.schedulingPolicy = config.getTaskSchedulingPolicy();
    }
    
    /**
     * Generates tasks from IoT devices
     * 
     * @param iotDevices List of IoT devices
     * @param currentTime Current simulation time
     */
    public void generateTasks(List<IoTDevice> iotDevices, double currentTime) {
        for (IoTDevice device : iotDevices) {
            if (!device.isActive()) {
                continue;
            }
            
            Task task = device.generateTask(
                currentTime,
                random,
                config.getMinTaskCpuRequirement(),
                config.getMaxTaskCpuRequirement(),
                config.getMinTaskRamRequirement(),
                config.getMaxTaskRamRequirement(),
                config.getMinTaskStorageRequirement(),
                config.getMaxTaskStorageRequirement(),
                config.getMinTaskDuration(),
                config.getMaxTaskDuration()
            );
            
            if (task != null) {
                // Set task data sizes
                task.setDataInputSize(config.getTaskInputSize());
                task.setDataOutputSize(config.getTaskOutputSize());
                
                // Set security criticality based on configuration
                if (random.nextDouble() < config.getSecurityCriticalTaskPercentage()) {
                    task.setSecurityCritical(true);
                }
                
                // Submit the task
                task.submit(currentTime);
                
                // Add to task lists
                allTasks.add(task);
                pendingTasks.add(task);
                
                // Update metrics
                results.incrementTotalTasksGenerated();
            }
        }
    }
    
    /**
     * Schedules pending tasks to appropriate devices
     * 
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @param currentTime Current simulation time
     */
    public void scheduleTasks(List<IoTDevice> iotDevices, List<EdgeNode> edgeNodes,
                             List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters,
                             double currentTime) {
        // Sort pending tasks based on scheduling policy
        sortPendingTasks();
        
        List<Task> scheduledTasks = new ArrayList<>();
        
        for (Task task : pendingTasks) {
            Device sourceDevice = task.getSourceDevice();
            
            // Get the best device to execute this task
            Device targetDevice = selectTargetDevice(task, iotDevices, edgeNodes, fogNodes, cloudDatacenters);
            
            if (targetDevice != null) {
                // Calculate network transfer time if the target is different from the source
                double transferTime = 0.0;
                if (targetDevice != sourceDevice) {
                    transferTime = networkModel.calculateTransferTime(sourceDevice, targetDevice, 
                                                                     task.getDataInputSize());
                }
                
                // Start task execution with delay for network transfer
                task.start(currentTime + transferTime, targetDevice);
                
                // Add to executing tasks
                executingTasks.add(task);
                scheduledTasks.add(task);
                
                // Update metrics
                results.addTaskWaitingTime(task.getWaitingTime());
            }
        }
        
        // Remove scheduled tasks from pending list
        pendingTasks.removeAll(scheduledTasks);
    }
    
    /**
     * Processes executing tasks
     * 
     * @param currentTime Current simulation time
     * @param timeStep Time step in seconds
     */
    public void processTasks(double currentTime, double timeStep) {
        List<Task> tasksToComplete = new ArrayList<>();
        List<Task> tasksToFail = new ArrayList<>();
        
        for (Task task : executingTasks) {
            Device executingDevice = task.getExecutingDevice();
            
            // Check if the device is still active
            if (!executingDevice.isActive()) {
                tasksToFail.add(task);
                continue;
            }
            
            // Check if the device is compromised and the task is security-critical
            if (executingDevice.isCompromised() && task.isSecurityCritical() && 
                securityManager.shouldFailCompromisedTasks()) {
                tasksToFail.add(task);
                continue;
            }
            
            // Check if the task has completed its execution time
            double executionTime = currentTime - task.getStartTime();
            if (executionTime >= task.getDuration()) {
                // Execute the task on the device
                boolean success = executingDevice.executeTask(task);
                
                if (success) {
                    // Complete the task
                    task.complete(currentTime);
                    tasksToComplete.add(task);
                    
                    // Update metrics
                    results.incrementCompletedTasks();
                    results.addTaskExecutionTime(task.getExecutionTime());
                    results.addTaskResponseTime(task.getResponseTime());
                    
                    // Calculate and add return transfer time for output data if needed
                    if (executingDevice != task.getSourceDevice()) {
                        double returnTransferTime = networkModel.calculateTransferTime(
                            executingDevice, task.getSourceDevice(), task.getDataOutputSize());
                        
                        // Add return transfer time to response time metrics
                        results.addTaskResponseTime(returnTransferTime);
                    }
                } else {
                    // Task execution failed
                    task.fail(currentTime);
                    tasksToFail.add(task);
                }
            }
        }
        
        // Move completed tasks
        for (Task task : tasksToComplete) {
            executingTasks.remove(task);
            completedTasks.add(task);
        }
        
        // Move failed tasks
        for (Task task : tasksToFail) {
            executingTasks.remove(task);
            failedTasks.add(task);
            results.incrementFailedTasks();
        }
    }
    
    /**
     * Selects the best device to execute a task based on the scheduling policy
     * 
     * @param task The task to schedule
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectTargetDevice(Task task, List<IoTDevice> iotDevices,
                                     List<EdgeNode> edgeNodes, List<FogNode> fogNodes,
                                     List<CloudDatacenter> cloudDatacenters) {
        switch (schedulingPolicy) {
            case LOCAL_ONLY:
                return selectLocalDevice(task);
                
            case EDGE_FIRST:
                return selectEdgeFirstDevice(task, edgeNodes, fogNodes, cloudDatacenters);
                
            case FOG_FIRST:
                return selectFogFirstDevice(task, edgeNodes, fogNodes, cloudDatacenters);
                
            case CLOUD_FIRST:
                return selectCloudFirstDevice(task, edgeNodes, fogNodes, cloudDatacenters);
                
            case RESOURCE_AWARE:
                return selectResourceAwareDevice(task, iotDevices, edgeNodes, fogNodes, cloudDatacenters);
                
            case SECURITY_AWARE:
                return selectSecurityAwareDevice(task, iotDevices, edgeNodes, fogNodes, cloudDatacenters);
                
            case ENERGY_AWARE:
                return selectEnergyAwareDevice(task, iotDevices, edgeNodes, fogNodes, cloudDatacenters);
                
            case COST_AWARE:
                return selectCostAwareDevice(task, edgeNodes, fogNodes, cloudDatacenters);
                
            case LATENCY_AWARE:
                return selectLatencyAwareDevice(task, iotDevices, edgeNodes, fogNodes, cloudDatacenters);
                
            default:
                return selectResourceAwareDevice(task, iotDevices, edgeNodes, fogNodes, cloudDatacenters);
        }
    }
    
    /**
     * Selects a local device (the source device) for task execution
     * 
     * @param task The task to schedule
     * @return The source device if it can execute the task, null otherwise
     */
    private Device selectLocalDevice(Task task) {
        Device sourceDevice = task.getSourceDevice();
        
        // Check if the source device can execute the task
        if (sourceDevice.isActive() && 
            sourceDevice.getProcessingPower() >= task.getCpuRequirement() &&
            sourceDevice.getMemory() >= task.getRamRequirement() &&
            sourceDevice.getStorage() >= task.getStorageRequirement()) {
            return sourceDevice;
        }
        
        return null;
    }
    
    /**
     * Selects a device with an edge-first policy
     * 
     * @param task The task to schedule
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectEdgeFirstDevice(Task task, List<EdgeNode> edgeNodes,
                                        List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters) {
        // Try edge nodes first
        Device selectedDevice = findSuitableDevice(task, edgeNodes);
        if (selectedDevice != null) {
            return selectedDevice;
        }
        
        // Try fog nodes next
        selectedDevice = findSuitableDevice(task, fogNodes);
        if (selectedDevice != null) {
            return selectedDevice;
        }
        
        // Try cloud datacenters last
        return findSuitableDevice(task, cloudDatacenters);
    }
    
    /**
     * Selects a device with a fog-first policy
     * 
     * @param task The task to schedule
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectFogFirstDevice(Task task, List<EdgeNode> edgeNodes,
                                       List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters) {
        // Try fog nodes first
        Device selectedDevice = findSuitableDevice(task, fogNodes);
        if (selectedDevice != null) {
            return selectedDevice;
        }
        
        // Try edge nodes next
        selectedDevice = findSuitableDevice(task, edgeNodes);
        if (selectedDevice != null) {
            return selectedDevice;
        }
        
        // Try cloud datacenters last
        return findSuitableDevice(task, cloudDatacenters);
    }
    
    /**
     * Selects a device with a cloud-first policy
     * 
     * @param task The task to schedule
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectCloudFirstDevice(Task task, List<EdgeNode> edgeNodes,
                                         List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters) {
        // Try cloud datacenters first
        Device selectedDevice = findSuitableDevice(task, cloudDatacenters);
        if (selectedDevice != null) {
            return selectedDevice;
        }
        
        // Try fog nodes next
        selectedDevice = findSuitableDevice(task, fogNodes);
        if (selectedDevice != null) {
            return selectedDevice;
        }
        
        // Try edge nodes last
        return findSuitableDevice(task, edgeNodes);
    }
    
    /**
     * Selects a device based on available resources
     * 
     * @param task The task to schedule
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectResourceAwareDevice(Task task, List<IoTDevice> iotDevices,
                                           List<EdgeNode> edgeNodes, List<FogNode> fogNodes,
                                           List<CloudDatacenter> cloudDatacenters) {
        List<Device> allDevices = new ArrayList<>();
        
        // Add all devices to the list
        allDevices.addAll(iotDevices);
        allDevices.addAll(edgeNodes);
        allDevices.addAll(fogNodes);
        allDevices.addAll(cloudDatacenters);
        
        // Find the device with the most available resources
        Device bestDevice = null;
        double bestResourceRatio = 0.0;
        
        for (Device device : allDevices) {
            if (!device.isActive()) {
                continue;
            }
            
            // Check if the device can execute the task
            if (device.getProcessingPower() >= task.getCpuRequirement() &&
                device.getMemory() >= task.getRamRequirement() &&
                device.getStorage() >= task.getStorageRequirement()) {
                
                // Calculate resource ratio (higher is better)
                double cpuRatio = device.getProcessingPower() / task.getCpuRequirement();
                double memoryRatio = device.getMemory() / task.getRamRequirement();
                double storageRatio = device.getStorage() / task.getStorageRequirement();
                
                // Use the minimum ratio as the limiting factor
                double resourceRatio = Math.min(cpuRatio, Math.min(memoryRatio, storageRatio));
                
                // Consider current utilization (lower is better)
                resourceRatio = resourceRatio * (1.0 - (device.getResourceUtilization() / 100.0));
                
                if (bestDevice == null || resourceRatio > bestResourceRatio) {
                    bestDevice = device;
                    bestResourceRatio = resourceRatio;
                }
            }
        }
        
        return bestDevice;
    }
    
    /**
     * Selects a device based on security considerations
     * 
     * @param task The task to schedule
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectSecurityAwareDevice(Task task, List<IoTDevice> iotDevices,
                                           List<EdgeNode> edgeNodes, List<FogNode> fogNodes,
                                           List<CloudDatacenter> cloudDatacenters) {
        List<Device> allDevices = new ArrayList<>();
        
        // Add all devices to the list
        allDevices.addAll(iotDevices);
        allDevices.addAll(edgeNodes);
        allDevices.addAll(fogNodes);
        allDevices.addAll(cloudDatacenters);
        
        // Filter out compromised devices for security-critical tasks
        if (task.isSecurityCritical()) {
            allDevices.removeIf(Device::isCompromised);
        }
        
        // Find the device with the most available resources
        Device bestDevice = null;
        double bestResourceRatio = 0.0;
        
        for (Device device : allDevices) {
            if (!device.isActive()) {
                continue;
            }
            
            // Check if the device can execute the task
            if (device.getProcessingPower() >= task.getCpuRequirement() &&
                device.getMemory() >= task.getRamRequirement() &&
                device.getStorage() >= task.getStorageRequirement()) {
                
                // Calculate resource ratio (higher is better)
                double cpuRatio = device.getProcessingPower() / task.getCpuRequirement();
                double memoryRatio = device.getMemory() / task.getRamRequirement();
                double storageRatio = device.getStorage() / task.getStorageRequirement();
                
                // Use the minimum ratio as the limiting factor
                double resourceRatio = Math.min(cpuRatio, Math.min(memoryRatio, storageRatio));
                
                // Consider current utilization (lower is better)
                resourceRatio = resourceRatio * (1.0 - (device.getResourceUtilization() / 100.0));
                
                if (bestDevice == null || resourceRatio > bestResourceRatio) {
                    bestDevice = device;
                    bestResourceRatio = resourceRatio;
                }
            }
        }
        
        return bestDevice;
    }
    
    /**
     * Selects a device based on energy efficiency
     * 
     * @param task The task to schedule
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectEnergyAwareDevice(Task task, List<IoTDevice> iotDevices,
                                         List<EdgeNode> edgeNodes, List<FogNode> fogNodes,
                                         List<CloudDatacenter> cloudDatacenters) {
        List<Device> allDevices = new ArrayList<>();
        
        // Add all devices to the list
        allDevices.addAll(iotDevices);
        allDevices.addAll(edgeNodes);
        allDevices.addAll(fogNodes);
        allDevices.addAll(cloudDatacenters);
        
        // Find the device with the best energy efficiency
        Device bestDevice = null;
        double bestEnergyEfficiency = 0.0;
        
        for (Device device : allDevices) {
            if (!device.isActive()) {
                continue;
            }
            
            // Check if the device can execute the task
            if (device.getProcessingPower() >= task.getCpuRequirement() &&
                device.getMemory() >= task.getRamRequirement() &&
                device.getStorage() >= task.getStorageRequirement()) {
                
                // Calculate energy efficiency
                double energyEfficiency = 0.0;
                
                if (device instanceof EdgeNode) {
                    energyEfficiency = ((EdgeNode) device).getEnergyEfficiency();
                } else if (device instanceof FogNode) {
                    energyEfficiency = ((FogNode) device).getEnergyEfficiency();
                } else if (device instanceof CloudDatacenter) {
                    energyEfficiency = ((CloudDatacenter) device).getEnergyEfficiency();
                } else {
                    // For IoT devices, use a simple model based on battery capacity
                    energyEfficiency = device.getRemainingBattery() / 100.0;
                }
                
                if (bestDevice == null || energyEfficiency > bestEnergyEfficiency) {
                    bestDevice = device;
                    bestEnergyEfficiency = energyEfficiency;
                }
            }
        }
        
        return bestDevice;
    }
    
    /**
     * Selects a device based on execution cost
     * 
     * @param task The task to schedule
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectCostAwareDevice(Task task, List<EdgeNode> edgeNodes,
                                       List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters) {
        Device bestDevice = null;
        double lowestCost = Double.MAX_VALUE;
        
        // Check edge nodes
        for (EdgeNode node : edgeNodes) {
            if (!node.isActive()) {
                continue;
            }
            
            // Check if the node can execute the task
            if (node.getProcessingPower() >= task.getCpuRequirement() &&
                node.getMemory() >= task.getRamRequirement() &&
                node.getStorage() >= task.getStorageRequirement()) {
                
                double cost = node.calculateTaskCost(task);
                
                if (cost < lowestCost) {
                    bestDevice = node;
                    lowestCost = cost;
                }
            }
        }
        
        // Check fog nodes
        for (FogNode node : fogNodes) {
            if (!node.isActive()) {
                continue;
            }
            
            // Check if the node can execute the task
            if (node.getProcessingPower() >= task.getCpuRequirement() &&
                node.getMemory() >= task.getRamRequirement() &&
                node.getStorage() >= task.getStorageRequirement()) {
                
                double cost = node.calculateTaskCost(task);
                
                if (cost < lowestCost) {
                    bestDevice = node;
                    lowestCost = cost;
                }
            }
        }
        
        // Check cloud datacenters
        for (CloudDatacenter datacenter : cloudDatacenters) {
            if (!datacenter.isActive()) {
                continue;
            }
            
            // Check if the datacenter can execute the task
            if (datacenter.getProcessingPower() >= task.getCpuRequirement() &&
                datacenter.getMemory() >= task.getRamRequirement() &&
                datacenter.getStorage() >= task.getStorageRequirement()) {
                
                double cost = datacenter.calculateTaskCost(task);
                
                if (cost < lowestCost) {
                    bestDevice = datacenter;
                    lowestCost = cost;
                }
            }
        }
        
        return bestDevice;
    }
    
    /**
     * Selects a device based on network latency
     * 
     * @param task The task to schedule
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return The selected device, or null if no suitable device is found
     */
    private Device selectLatencyAwareDevice(Task task, List<IoTDevice> iotDevices,
                                          List<EdgeNode> edgeNodes, List<FogNode> fogNodes,
                                          List<CloudDatacenter> cloudDatacenters) {
        Device sourceDevice = task.getSourceDevice();
        Device bestDevice = null;
        double lowestLatency = Double.MAX_VALUE;
        
        List<Device> allDevices = new ArrayList<>();
        allDevices.addAll(iotDevices);
        allDevices.addAll(edgeNodes);
        allDevices.addAll(fogNodes);
        allDevices.addAll(cloudDatacenters);
        
        for (Device device : allDevices) {
            if (!device.isActive()) {
                continue;
            }
            
            // Check if the device can execute the task
            if (device.getProcessingPower() >= task.getCpuRequirement() &&
                device.getMemory() >= task.getRamRequirement() &&
                device.getStorage() >= task.getStorageRequirement()) {
                
                // Calculate network latency
                double latency = networkModel.calculateLatency(sourceDevice, device);
                
                if (latency < lowestLatency) {
                    bestDevice = device;
                    lowestLatency = latency;
                }
            }
        }
        
        return bestDevice;
    }
    
    /**
     * Finds a suitable device from a list of devices
     * 
     * @param task The task to schedule
     * @param devices List of devices
     * @return The selected device, or null if no suitable device is found
     */
    private Device findSuitableDevice(Task task, List<? extends Device> devices) {
        for (Device device : devices) {
            if (!device.isActive()) {
                continue;
            }
            
            // Check if the device can execute the task
            if (device.getProcessingPower() >= task.getCpuRequirement() &&
                device.getMemory() >= task.getRamRequirement() &&
                device.getStorage() >= task.getStorageRequirement()) {
                return device;
            }
        }
        
        return null;
    }
    
    /**
     * Sorts pending tasks based on the scheduling policy
     */
    private void sortPendingTasks() {
        switch (schedulingPolicy) {
            case FIFO:
                // Sort by submission time (ascending)
                pendingTasks.sort(Comparator.comparingDouble(Task::getSubmissionTime));
                break;
                
            case PRIORITY:
                // Sort by priority (descending) and then by submission time (ascending)
                pendingTasks.sort(
                    Comparator.comparingInt(Task::getPriority).reversed()
                              .thenComparingDouble(Task::getSubmissionTime)
                );
                break;
                
            case SHORTEST_JOB_FIRST:
                // Sort by duration (ascending)
                pendingTasks.sort(Comparator.comparingDouble(Task::getDuration));
                break;
                
            case SECURITY_FIRST:
                // Sort by security criticality (security-critical first) and then by submission time
                pendingTasks.sort(
                    Comparator.comparing(Task::isSecurityCritical).reversed()
                              .thenComparingDouble(Task::getSubmissionTime)
                );
                break;
                
            default:
                // Default to FIFO
                pendingTasks.sort(Comparator.comparingDouble(Task::getSubmissionTime));
                break;
        }
    }
    
    /**
     * Gets all tasks in the simulation
     * 
     * @return List of all tasks
     */
    public List<Task> getAllTasks() {
        return new ArrayList<>(allTasks);
    }
    
    /**
     * Gets pending tasks in the simulation
     * 
     * @return List of pending tasks
     */
    public List<Task> getPendingTasks() {
        return new ArrayList<>(pendingTasks);
    }
    
    /**
     * Gets executing tasks in the simulation
     * 
     * @return List of executing tasks
     */
    public List<Task> getExecutingTasks() {
        return new ArrayList<>(executingTasks);
    }
    
    /**
     * Gets completed tasks in the simulation
     * 
     * @return List of completed tasks
     */
    public List<Task> getCompletedTasks() {
        return new ArrayList<>(completedTasks);
    }
    
    /**
     * Gets failed tasks in the simulation
     * 
     * @return List of failed tasks
     */
    public List<Task> getFailedTasks() {
        return new ArrayList<>(failedTasks);
    }
    
    /**
     * Gets the current scheduling policy
     * 
     * @return The scheduling policy
     */
    public TaskSchedulingPolicy getSchedulingPolicy() {
        return schedulingPolicy;
    }
    
    /**
     * Sets the scheduling policy
     * 
     * @param schedulingPolicy The scheduling policy to set
     */
    public void setSchedulingPolicy(TaskSchedulingPolicy schedulingPolicy) {
        this.schedulingPolicy = schedulingPolicy;
    }
}
