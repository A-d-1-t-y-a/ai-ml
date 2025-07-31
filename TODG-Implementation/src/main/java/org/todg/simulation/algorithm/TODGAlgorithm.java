package org.todg.simulation.algorithm;

import org.todg.simulation.model.*;
import java.util.*;

/**
 * Implementation of the TODG (Task Offloading with Delay Guarantees) algorithm.
 * This class provides the core logic for distributed task offloading with delay guarantees
 * as described in the paper "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021).
 */
public class TODGAlgorithm {
    // Algorithm parameters
    private double alpha; // Weight for delay in utility function
    private double beta; // Weight for energy in utility function
    private double gamma; // Weight for load balancing in utility function
    private double V; // Control parameter for delay-energy tradeoff
    
    // Simulation components
    private List<IoTDevice> devices;
    private List<EdgeServer> servers;
    private List<Channel> channels;
    
    // Statistics
    private int totalTasksGenerated;
    private int totalTasksOffloaded;
    private int totalTasksProcessedLocally;
    private int totalTasksCompleted;
    private int totalTasksFailed;
    private double totalEnergyConsumed;
    private double averageDelay;
    
    /**
     * Constructor for the TODG algorithm.
     * 
     * @param alpha Weight for delay in utility function
     * @param beta Weight for energy in utility function
     * @param gamma Weight for load balancing in utility function
     * @param V Control parameter for delay-energy tradeoff
     */
    public TODGAlgorithm(double alpha, double beta, double gamma, double V) {
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.V = V;
        this.devices = new ArrayList<>();
        this.servers = new ArrayList<>();
        this.channels = new ArrayList<>();
        resetStatistics();
    }
    
    /**
     * Resets all statistics to zero.
     */
    public void resetStatistics() {
        totalTasksGenerated = 0;
        totalTasksOffloaded = 0;
        totalTasksProcessedLocally = 0;
        totalTasksCompleted = 0;
        totalTasksFailed = 0;
        totalEnergyConsumed = 0.0;
        averageDelay = 0.0;
    }
    
    /**
     * Sets the IoT devices for the simulation.
     * 
     * @param devices List of IoT devices
     */
    public void setDevices(List<IoTDevice> devices) {
        this.devices = devices;
    }
    
    /**
     * Sets the edge servers for the simulation.
     * 
     * @param servers List of edge servers
     */
    public void setServers(List<EdgeServer> servers) {
        this.servers = servers;
    }
    
    /**
     * Sets the communication channels for the simulation.
     * 
     * @param channels List of communication channels
     */
    public void setChannels(List<Channel> channels) {
        this.channels = channels;
    }
    
    /**
     * Executes the TODG algorithm for a single time step.
     * 
     * @param currentTime The current simulation time
     * @param timeStep The time step duration
     * @return Statistics for this time step
     */
    public Map<String, Object> executeTimeStep(double currentTime, double timeStep) {
        Map<String, Object> stepStats = new HashMap<>();
        int tasksGenerated = 0;
        int tasksOffloaded = 0;
        int tasksProcessedLocally = 0;
        int tasksCompleted = 0;
        int tasksFailed = 0;
        double energyConsumed = 0.0;
        
        // Update all channels to reflect current network conditions
        for (Channel channel : channels) {
            channel.updateChannel(currentTime);
        }
        
        // Generate tasks for each device
        int taskIdCounter = totalTasksGenerated;
        for (IoTDevice device : devices) {
            int deviceTasksGenerated = device.generateTasks(currentTime, timeStep, taskIdCounter);
            taskIdCounter += deviceTasksGenerated;
            tasksGenerated += deviceTasksGenerated;
        }
        
        // Process tasks for each device
        for (IoTDevice device : devices) {
            Queue<Task> localQueue = device.getLocalTaskQueue();
            List<Task> tasksToRemove = new ArrayList<>();
            
            for (Task task : localQueue) {
                // Make offloading decision
                EdgeServer targetServer = makeOffloadingDecision(device, task, currentTime);
                
                if (targetServer != null) {
                    // Offload task to edge server
                    Channel channel = findChannelBySourceAndDestination(device, targetServer);
                    
                    if (targetServer != null && channel != null) {
                        // Simulate transmission
                        double transmissionTime = channel.transmitTask(task, currentTime);
                        
                        if (transmissionTime > 0) {
                            // Transmission successful
                            task.setStatus(Task.TaskStatus.TRANSMITTING);
                            
                            // Calculate arrival time at server
                            double arrivalTimeAtServer = currentTime + transmissionTime;
                            
                            // Try to assign task to server
                            if (targetServer.receiveTask(task, arrivalTimeAtServer)) {
                                tasksOffloaded++;
                                tasksToRemove.add(task);
                                
                                // Calculate energy consumed for transmission
                                double transmissionEnergy = 0.5 * transmissionTime; // Simplified energy model
                                device.consumeEnergy(transmissionEnergy);
                                energyConsumed += transmissionEnergy;
                            }
                        }
                    }
                } else {
                    // Process task locally
                    task.setStatus(Task.TaskStatus.PROCESSING);
                    double completionTime = device.processTaskLocally(task, currentTime);
                    
                    if (completionTime <= (task.getArrivalTime() + task.getDeadline())) {
                        tasksCompleted++;
                    } else {
                        tasksFailed++;
                    }
                    
                    tasksProcessedLocally++;
                    tasksToRemove.add(task);
                    
                    // Add energy consumed
                    double processingEnergy = device.calculateLocalEnergy(task);
                    energyConsumed += processingEnergy;
                }
            }
            
            // Remove processed tasks from queue
            localQueue.removeAll(tasksToRemove);
        }
        
        // Process tasks on each edge server
        for (EdgeServer server : servers) {
            int serverTasksProcessed = server.processTasks(currentTime, timeStep);
            tasksCompleted += serverTasksProcessed;
            energyConsumed += server.getTotalEnergyConsumed();
        }
        
        // Update global statistics
        totalTasksGenerated += tasksGenerated;
        totalTasksOffloaded += tasksOffloaded;
        totalTasksProcessedLocally += tasksProcessedLocally;
        totalTasksCompleted += tasksCompleted;
        totalTasksFailed += tasksFailed;
        totalEnergyConsumed += energyConsumed;
        
        // Calculate average delay for completed tasks
        double totalDelay = 0.0;
        int completedTaskCount = 0;
        
        for (EdgeServer server : servers) {
            for (Task task : server.getCompletedTasks()) {
                totalDelay += (task.getCompletionTime() - task.getArrivalTime());
                completedTaskCount++;
            }
        }
        
        if (completedTaskCount > 0) {
            averageDelay = totalDelay / completedTaskCount;
        }
        
        // Prepare statistics for this time step
        stepStats.put("currentTime", currentTime);
        stepStats.put("tasksGenerated", tasksGenerated);
        stepStats.put("tasksOffloaded", tasksOffloaded);
        stepStats.put("tasksProcessedLocally", tasksProcessedLocally);
        stepStats.put("tasksCompleted", tasksCompleted);
        stepStats.put("tasksFailed", tasksFailed);
        stepStats.put("energyConsumed", energyConsumed);
        
        return stepStats;
    }
    
    /**
     * Makes an offloading decision for a task using the TODG algorithm.
     * 
     * @param device The source IoT device
     * @param task The task to offload
     * @param currentTime The current simulation time
     * @return The selected edge server, or null if the task should be processed locally
     */
    public EdgeServer makeOffloadingDecision(IoTDevice device, Task task, double currentTime) {
        // Calculate local processing metrics
        double localProcessingTime = task.getComputationalRequirement() / device.getMips();
        double localEnergyConsumption = device.calculateLocalEnergy(task);
        
        // Check if the task can meet its deadline locally
        boolean canMeetDeadlineLocally = (localProcessingTime <= task.getDeadline());
        
        // Calculate local utility
        double localUtility = calculateUtility(localProcessingTime, localEnergyConsumption, 0.0);
        
        // Default is to process locally
        EdgeServer bestServer = null;
        double bestUtility = canMeetDeadlineLocally ? localUtility : Double.NEGATIVE_INFINITY;
        
        // Evaluate each edge server
        for (EdgeServer server : servers) {
            // Find the best available channel for this server
            Channel bestChannel = findBestChannel(device, server);
            
            if (bestChannel != null && bestChannel.isAvailable()) {
                // Calculate offloading metrics
                double transmissionTime = bestChannel.calculateTransmissionTime(task.getDataSize());
                double processingTime = task.getComputationalRequirement() / server.getMips();
                double totalTime = transmissionTime + processingTime;
                double offloadingEnergyConsumption = 0.5 * transmissionTime; // Simplified energy model
                double serverLoad = server.getLoadPercentage() / 100.0; // Normalize to 0-1 range
                
                // Check if the task can meet its deadline if offloaded to this server
                boolean canMeetDeadline = task.canMeetDeadline(currentTime, transmissionTime, processingTime);
                
                if (canMeetDeadline) {
                    // Calculate utility of offloading to this server
                    double utility = calculateUtility(totalTime, offloadingEnergyConsumption, serverLoad);
                    
                    // Update best server if this one has better utility
                    if (utility > bestUtility) {
                        bestUtility = utility;
                        bestServer = server;
                    }
                }
            }
        }
        
        return bestServer;
    }
    
    /**
     * Calculates the utility of an offloading decision based on time, energy, and server load.
     * 
     * @param time The processing time
     * @param energy The energy consumption
     * @param serverLoad The server load (0.0 - 1.0)
     * @return The utility value
     */
    public double calculateUtility(double time, double energy, double serverLoad) {
        // Higher utility is better (negative values because we want to minimize time, energy, and load)
        return -(alpha * time + beta * energy + gamma * serverLoad);
    }
    
    /**
     * Finds the best available channel for communication between a device and a server.
     * 
     * @param device The source IoT device
     * @param server The destination edge server
     * @return The best available channel, or null if no suitable channel is found
     */
    public Channel findBestChannel(IoTDevice device, EdgeServer server) {
        Channel bestChannel = null;
        double bestQuality = Double.NEGATIVE_INFINITY;
        
        for (Channel channel : channels) {
            if (channel.isAvailable() && 
                channel.getSourceDeviceId() == device.getDeviceId() && 
                channel.getDestinationServerId() == server.getServerId()) {
                
                double quality = channel.getBandwidth() / (1 + channel.getInterference());
                
                if (quality > bestQuality) {
                    bestQuality = quality;
                    bestChannel = channel;
                }
            }
        }
        
        return bestChannel;
    }
    
    /**
     * Finds an edge server by its ID.
     * 
     * @param serverId The server ID to find
     * @return The edge server, or null if not found
     */
    public EdgeServer findServerById(int serverId) {
        for (EdgeServer server : servers) {
            if (server.getServerId() == serverId) {
                return server;
            }
        }
        return null;
    }
    
    /**
     * Finds a channel by its source device and destination server.
     * 
     * @param device The source IoT device
     * @param server The destination edge server
     * @return The channel, or null if not found
     */
    public Channel findChannelBySourceAndDestination(IoTDevice device, EdgeServer server) {
        for (Channel channel : channels) {
            if (channel.getSourceDeviceId() == device.getDeviceId() && 
                channel.getDestinationServerId() == server.getServerId()) {
                return channel;
            }
        }
        return null;
    }
    
    // Getters for statistics
    
    public int getTotalTasksGenerated() {
        return totalTasksGenerated;
    }
    
    public int getTotalTasksOffloaded() {
        return totalTasksOffloaded;
    }
    
    public int getTotalTasksProcessedLocally() {
        return totalTasksProcessedLocally;
    }
    
    public int getTotalTasksCompleted() {
        return totalTasksCompleted;
    }
    
    public int getTotalTasksFailed() {
        return totalTasksFailed;
    }
    
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    public double getAverageDelay() {
        return averageDelay;
    }
    
    public double getTaskCompletionRate() {
        if (totalTasksGenerated == 0) {
            return 100.0;
        }
        return (totalTasksCompleted * 100.0) / totalTasksGenerated;
    }
}
