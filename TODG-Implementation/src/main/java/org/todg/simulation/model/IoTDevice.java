package org.todg.simulation.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.LinkedList;
import java.util.Random;

/**
 * Represents an IoT device in the TODG simulation.
 * IoT devices generate tasks and make offloading decisions.
 * 
 * Based on the TODG paper: "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021)
 */
public class IoTDevice {
    private int deviceId;
    private String deviceName;
    private double mips; // Processing capability in Million Instructions Per Second
    private double availableMemory; // in MB
    private double energyCapacity; // in Joules
    private double energyConsumption; // in Joules
    private double taskGenerationRate; // Tasks per second (Poisson distribution parameter)
    private Queue<Task> localTaskQueue;
    private List<Task> generatedTasks;
    private Random random;
    
    // Network-related attributes
    private double uploadBandwidth; // in Mbps
    private double downloadBandwidth; // in Mbps
    private double latency; // in ms
    
    // Location coordinates (for mobility modeling)
    private double xCoordinate;
    private double yCoordinate;
    
    /**
     * Constructor for creating a new IoT device.
     * 
     * @param deviceId The unique identifier for this device
     * @param deviceName The name of this device
     * @param mips The processing capability in Million Instructions Per Second
     * @param availableMemory The available memory in MB
     * @param energyCapacity The energy capacity in Joules
     * @param taskGenerationRate The task generation rate (tasks per second)
     * @param uploadBandwidth The upload bandwidth in Mbps
     * @param downloadBandwidth The download bandwidth in Mbps
     * @param latency The network latency in ms
     * @param xCoordinate The x-coordinate of the device location
     * @param yCoordinate The y-coordinate of the device location
     */
    public IoTDevice(int deviceId, String deviceName, double mips, double availableMemory,
                    double energyCapacity, double taskGenerationRate,
                    double uploadBandwidth, double downloadBandwidth, double latency,
                    double xCoordinate, double yCoordinate) {
        this.deviceId = deviceId;
        this.deviceName = deviceName;
        this.mips = mips;
        this.availableMemory = availableMemory;
        this.energyCapacity = energyCapacity;
        this.energyConsumption = 0.0;
        this.taskGenerationRate = taskGenerationRate;
        this.localTaskQueue = new LinkedList<>();
        this.generatedTasks = new ArrayList<>();
        this.random = new Random();
        this.uploadBandwidth = uploadBandwidth;
        this.downloadBandwidth = downloadBandwidth;
        this.latency = latency;
        this.xCoordinate = xCoordinate;
        this.yCoordinate = yCoordinate;
    }
    
    /**
     * Generates tasks based on a Poisson distribution.
     * 
     * @param currentTime The current simulation time
     * @param timeInterval The time interval for which tasks are being generated
     * @param taskIdCounter The counter for task IDs
     * @return The number of tasks generated
     */
    public int generateTasks(double currentTime, double timeInterval, int taskIdCounter) {
        // Calculate expected number of tasks in this interval
        double expectedTasks = taskGenerationRate * timeInterval;
        
        // Generate random number of tasks using Poisson distribution
        int numTasksToGenerate = getPoissonRandom(expectedTasks);
        
        for (int i = 0; i < numTasksToGenerate; i++) {
            // Generate task with random characteristics
            double arrivalTime = currentTime + (random.nextDouble() * timeInterval);
            double dataSize = 0.5 + (random.nextDouble() * 9.5); // 0.5-10 MB
            double computationalRequirement = 100 + (random.nextDouble() * 900); // 100-1000 MI
            double deadline = 2.0 + (random.nextDouble() * 8.0); // 2-10 seconds
            
            Task task = new Task(taskIdCounter++, arrivalTime, dataSize, 
                                computationalRequirement, deadline, deviceId);
            
            generatedTasks.add(task);
            localTaskQueue.add(task);
        }
        
        return numTasksToGenerate;
    }
    
    /**
     * Generates a random number from a Poisson distribution.
     * 
     * @param lambda The expected number of occurrences
     * @return A random number from a Poisson distribution
     */
    private int getPoissonRandom(double lambda) {
        double L = Math.exp(-lambda);
        double p = 1.0;
        int k = 0;
        
        do {
            k++;
            p *= random.nextDouble();
        } while (p > L);
        
        return k - 1;
    }
    
    /**
     * Makes an offloading decision for a task based on the TODG algorithm.
     * 
     * @param task The task to offload
     * @param edgeServers The list of available edge servers
     * @param channels The list of available communication channels
     * @param currentTime The current simulation time
     * @return The ID of the selected edge server, or -1 if the task should be processed locally
     */
    public int makeOffloadingDecision(Task task, List<EdgeServer> edgeServers, 
                                     List<Channel> channels, double currentTime) {
        // If no edge servers or channels are available, process locally
        if (edgeServers == null || edgeServers.isEmpty() || channels == null || channels.isEmpty()) {
            return -1;
        }
        
        // Calculate local processing metrics
        double localProcessingTime = task.getComputationalRequirement() / mips;
        double localEnergyConsumption = calculateLocalEnergy(task);
        
        // Check if the task can meet its deadline locally
        boolean canMeetDeadlineLocally = (localProcessingTime <= task.getDeadline());
        
        // Initialize variables to track the best offloading option
        int bestServerId = -1;
        double bestUtility = canMeetDeadlineLocally ? calculateUtility(localProcessingTime, localEnergyConsumption) : Double.NEGATIVE_INFINITY;
        
        // Evaluate each edge server
        for (EdgeServer server : edgeServers) {
            // Find the best available channel for this server
            Channel bestChannel = findBestChannel(channels, server);
            
            if (bestChannel != null) {
                // Calculate offloading metrics
                double transmissionTime = calculateTransmissionTime(task, bestChannel);
                double processingTime = task.getComputationalRequirement() / server.getMips();
                double totalTime = transmissionTime + processingTime;
                double offloadingEnergyConsumption = calculateOffloadingEnergy(task, bestChannel);
                
                // Check if the task can meet its deadline if offloaded to this server
                boolean canMeetDeadline = task.canMeetDeadline(currentTime, transmissionTime, processingTime);
                
                if (canMeetDeadline) {
                    // Calculate utility of offloading to this server
                    double utility = calculateUtility(totalTime, offloadingEnergyConsumption);
                    
                    // Update best server if this one has better utility
                    if (utility > bestUtility) {
                        bestUtility = utility;
                        bestServerId = server.getServerId();
                    }
                }
            }
        }
        
        return bestServerId;
    }
    
    /**
     * Finds the best available channel for communicating with a specific edge server.
     * 
     * @param channels The list of available communication channels
     * @param server The target edge server
     * @return The best available channel, or null if no suitable channel is found
     */
    private Channel findBestChannel(List<Channel> channels, EdgeServer server) {
        Channel bestChannel = null;
        double bestQuality = Double.NEGATIVE_INFINITY;
        
        for (Channel channel : channels) {
            if (channel.isAvailable() && channel.getDestinationServerId() == server.getServerId()) {
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
     * Calculates the transmission time for a task over a specific channel.
     * 
     * @param task The task to transmit
     * @param channel The communication channel
     * @return The transmission time in seconds
     */
    private double calculateTransmissionTime(Task task, Channel channel) {
        // Convert data size from MB to Mb (megabytes to megabits)
        double dataSizeInMb = task.getDataSize() * 8;
        
        // Calculate transmission time (data size / effective bandwidth)
        double effectiveBandwidth = channel.getBandwidth() / (1 + channel.getInterference());
        double transmissionTime = dataSizeInMb / effectiveBandwidth;
        
        // Add network latency (convert from ms to seconds)
        transmissionTime += (latency / 1000.0);
        
        return transmissionTime;
    }
    
    /**
     * Calculates the energy consumption for local processing of a task.
     * 
     * @param task The task to process
     * @return The energy consumption in Joules
     */
    private double calculateLocalEnergy(Task task) {
        // Simplified energy model: energy = power * time
        // Assume power consumption is proportional to MIPS
        double processingPower = 0.5 + (mips * 0.001); // Watts
        double processingTime = task.getComputationalRequirement() / mips;
        
        return processingPower * processingTime;
    }
    
    /**
     * Calculates the energy consumption for offloading a task.
     * 
     * @param task The task to offload
     * @param channel The communication channel used
     * @return The energy consumption in Joules
     */
    private double calculateOffloadingEnergy(Task task, Channel channel) {
        // Energy for transmission = transmission power * transmission time
        double transmissionPower = 0.9; // Watts
        double transmissionTime = calculateTransmissionTime(task, channel);
        
        return transmissionPower * transmissionTime;
    }
    
    /**
     * Calculates the utility of an offloading decision based on time and energy.
     * 
     * @param time The processing time
     * @param energy The energy consumption
     * @return The utility value
     */
    private double calculateUtility(double time, double energy) {
        // Utility is inversely proportional to time and energy
        // Alpha and beta are weighting factors for time and energy
        double alpha = 0.7;
        double beta = 0.3;
        
        return -(alpha * time + beta * energy);
    }
    
    /**
     * Updates the device's energy consumption.
     * 
     * @param energyUsed The amount of energy used
     */
    public void consumeEnergy(double energyUsed) {
        this.energyConsumption += energyUsed;
    }
    
    /**
     * Processes a task locally.
     * 
     * @param task The task to process
     * @param currentTime The current simulation time
     * @return The completion time of the task
     */
    public double processTaskLocally(Task task, double currentTime) {
        double processingTime = task.getComputationalRequirement() / mips;
        double energyUsed = calculateLocalEnergy(task);
        
        // Update task status
        task.setStartTime(currentTime);
        task.setCompletionTime(currentTime + processingTime);
        task.setStatus(Task.TaskStatus.COMPLETED);
        
        // Update device energy consumption
        consumeEnergy(energyUsed);
        
        return task.getCompletionTime();
    }
    
    // Getters and setters
    
    public int getDeviceId() {
        return deviceId;
    }
    
    public String getDeviceName() {
        return deviceName;
    }
    
    public double getMips() {
        return mips;
    }
    
    public double getAvailableMemory() {
        return availableMemory;
    }
    
    public double getEnergyCapacity() {
        return energyCapacity;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    public double getTaskGenerationRate() {
        return taskGenerationRate;
    }
    
    public Queue<Task> getLocalTaskQueue() {
        return localTaskQueue;
    }
    
    public List<Task> getGeneratedTasks() {
        return generatedTasks;
    }
    
    public double getUploadBandwidth() {
        return uploadBandwidth;
    }
    
    public double getDownloadBandwidth() {
        return downloadBandwidth;
    }
    
    public double getLatency() {
        return latency;
    }
    
    public double getXCoordinate() {
        return xCoordinate;
    }
    
    public double getYCoordinate() {
        return yCoordinate;
    }
    
    public void setXCoordinate(double xCoordinate) {
        this.xCoordinate = xCoordinate;
    }
    
    public void setYCoordinate(double yCoordinate) {
        this.yCoordinate = yCoordinate;
    }
}
