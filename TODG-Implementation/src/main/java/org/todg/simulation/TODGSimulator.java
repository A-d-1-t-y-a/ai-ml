package org.todg.simulation;

import org.todg.simulation.algorithm.TODGAlgorithm;
import org.todg.simulation.model.*;
import org.todg.simulation.metrics.MetricsCollector;
import org.todg.simulation.util.SimulationConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Main simulator class for the TODG implementation.
 * This class manages the overall simulation of the TODG system.
 * 
 * Based on the TODG paper: "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021)
 */
public class TODGSimulator {
    private static final Logger logger = LoggerFactory.getLogger(TODGSimulator.class);
    
    // Simulation components
    private SimulationConfig config;
    private TODGAlgorithm algorithm;
    private List<IoTDevice> devices;
    private List<EdgeServer> servers;
    private List<Channel> channels;
    private MetricsCollector metricsCollector;
    
    // Simulation state
    private double currentTime;
    private double simulationEndTime;
    private double timeStep;
    private boolean isRunning;
    
    /**
     * Constructor for the TODG simulator.
     * 
     * @param config The simulation configuration
     */
    public TODGSimulator(SimulationConfig config) {
        this.config = config;
        this.currentTime = 0.0;
        this.simulationEndTime = config.getSimulationDuration();
        this.timeStep = config.getTimeStep();
        this.isRunning = false;
        
        // Initialize components
        initializeComponents();
    }
    
    /**
     * Initializes all simulation components.
     */
    private void initializeComponents() {
        // Initialize algorithm
        algorithm = new TODGAlgorithm(
            config.getAlpha(),
            config.getBeta(),
            config.getGamma(),
            config.getV()
        );
        
        // Initialize devices
        devices = new ArrayList<>();
        for (int i = 0; i < config.getNumDevices(); i++) {
            IoTDevice device = createIoTDevice(i);
            devices.add(device);
        }
        
        // Initialize edge servers
        servers = new ArrayList<>();
        for (int i = 0; i < config.getNumServers(); i++) {
            EdgeServer server = createEdgeServer(i);
            servers.add(server);
        }
        
        // Initialize channels
        channels = new ArrayList<>();
        int channelId = 0;
        for (IoTDevice device : devices) {
            for (EdgeServer server : servers) {
                Channel channel = createChannel(channelId++, device.getDeviceId(), server.getServerId());
                channels.add(channel);
            }
        }
        
        // Set components in algorithm
        algorithm.setDevices(devices);
        algorithm.setServers(servers);
        algorithm.setChannels(channels);
        
        // Initialize metrics collector
        metricsCollector = new MetricsCollector();
    }
    
    /**
     * Creates an IoT device with parameters from the configuration.
     * 
     * @param deviceId The device ID
     * @return A new IoT device
     */
    private IoTDevice createIoTDevice(int deviceId) {
        String deviceName = "Device-" + deviceId;
        double mips = config.getDeviceMipsMin() + Math.random() * (config.getDeviceMipsMax() - config.getDeviceMipsMin());
        double memory = config.getDeviceMemoryMin() + Math.random() * (config.getDeviceMemoryMax() - config.getDeviceMemoryMin());
        double energy = config.getDeviceEnergyMin() + Math.random() * (config.getDeviceEnergyMax() - config.getDeviceEnergyMin());
        double taskRate = config.getTaskGenerationRateMin() + Math.random() * (config.getTaskGenerationRateMax() - config.getTaskGenerationRateMin());
        double uploadBw = config.getDeviceUploadBwMin() + Math.random() * (config.getDeviceUploadBwMax() - config.getDeviceUploadBwMin());
        double downloadBw = config.getDeviceDownloadBwMin() + Math.random() * (config.getDeviceDownloadBwMax() - config.getDeviceDownloadBwMin());
        double latency = config.getNetworkLatencyMin() + Math.random() * (config.getNetworkLatencyMax() - config.getNetworkLatencyMin());
        
        // Random position within simulation area
        double x = Math.random() * config.getSimulationAreaWidth();
        double y = Math.random() * config.getSimulationAreaHeight();
        
        return new IoTDevice(deviceId, deviceName, mips, memory, energy, taskRate, uploadBw, downloadBw, latency, x, y);
    }
    
    /**
     * Creates an edge server with parameters from the configuration.
     * 
     * @param serverId The server ID
     * @return A new edge server
     */
    private EdgeServer createEdgeServer(int serverId) {
        String serverName = "Server-" + serverId;
        double mips = config.getServerMipsMin() + Math.random() * (config.getServerMipsMax() - config.getServerMipsMin());
        double memory = config.getServerMemoryMin() + Math.random() * (config.getServerMemoryMax() - config.getServerMemoryMin());
        double storage = config.getServerStorageMin() + Math.random() * (config.getServerStorageMax() - config.getServerStorageMin());
        double power = config.getServerPowerMin() + Math.random() * (config.getServerPowerMax() - config.getServerPowerMin());
        int maxLoad = config.getServerMaxLoadMin() + (int)(Math.random() * (config.getServerMaxLoadMax() - config.getServerMaxLoadMin()));
        
        // Random position within simulation area (edge servers are typically at the edge of the network)
        double x, y;
        if (Math.random() < 0.5) {
            // Place on horizontal edge
            x = Math.random() * config.getSimulationAreaWidth();
            y = Math.random() < 0.5 ? 0 : config.getSimulationAreaHeight();
        } else {
            // Place on vertical edge
            x = Math.random() < 0.5 ? 0 : config.getSimulationAreaWidth();
            y = Math.random() * config.getSimulationAreaHeight();
        }
        
        return new EdgeServer(serverId, serverName, mips, memory, storage, power, maxLoad, x, y);
    }
    
    /**
     * Creates a communication channel with parameters from the configuration.
     * 
     * @param channelId The channel ID
     * @param sourceDeviceId The source device ID
     * @param destinationServerId The destination server ID
     * @return A new communication channel
     */
    private Channel createChannel(int channelId, int sourceDeviceId, int destinationServerId) {
        double bandwidth = config.getChannelBandwidthMin() + Math.random() * (config.getChannelBandwidthMax() - config.getChannelBandwidthMin());
        double interference = config.getChannelInterferenceMin() + Math.random() * (config.getChannelInterferenceMax() - config.getChannelInterferenceMin());
        double reliability = config.getChannelReliabilityMin() + Math.random() * (config.getChannelReliabilityMax() - config.getChannelReliabilityMin());
        double dynamicsInterval = config.getChannelDynamicsInterval();
        double interferenceVariability = config.getChannelInterferenceVariability();
        
        return new Channel(channelId, sourceDeviceId, destinationServerId, bandwidth, interference, reliability, dynamicsInterval, interferenceVariability);
    }
    
    /**
     * Runs the simulation for the configured duration.
     */
    public void runSimulation() {
        logger.info("Starting TODG simulation with {} devices and {} edge servers", devices.size(), servers.size());
        isRunning = true;
        currentTime = 0.0;
        
        // Reset algorithm statistics
        algorithm.resetStatistics();
        
        // Main simulation loop
        while (currentTime < simulationEndTime && isRunning) {
            // Execute one time step
            Map<String, Object> stepStats = algorithm.executeTimeStep(currentTime, timeStep);
            
            // Collect metrics
            metricsCollector.collectMetrics(currentTime, stepStats);
            
            // Log progress periodically
            if ((int)(currentTime / timeStep) % 100 == 0) {
                logger.info("Simulation time: {}/{} seconds", String.format("%.2f", currentTime), simulationEndTime);
                logger.info("Tasks generated: {}, completed: {}, failed: {}", 
                    algorithm.getTotalTasksGenerated(), 
                    algorithm.getTotalTasksCompleted(), 
                    algorithm.getTotalTasksFailed());
            }
            
            // Advance time
            currentTime += timeStep;
        }
        
        // Simulation complete
        isRunning = false;
        logger.info("Simulation completed at time: {} seconds", String.format("%.2f", currentTime));
        printFinalStatistics();
    }
    
    /**
     * Stops the simulation.
     */
    public void stopSimulation() {
        isRunning = false;
        logger.info("Simulation stopped at time: {} seconds", String.format("%.2f", currentTime));
    }
    
    /**
     * Prints the final simulation statistics.
     */
    private void printFinalStatistics() {
        logger.info("=== Final Simulation Statistics ===");
        logger.info("Total tasks generated: {}", algorithm.getTotalTasksGenerated());
        logger.info("Tasks offloaded to edge servers: {}", algorithm.getTotalTasksOffloaded());
        logger.info("Tasks processed locally: {}", algorithm.getTotalTasksProcessedLocally());
        logger.info("Tasks completed successfully: {}", algorithm.getTotalTasksCompleted());
        logger.info("Tasks failed (missed deadline): {}", algorithm.getTotalTasksFailed());
        logger.info("Task completion rate: {}%", String.format("%.2f", algorithm.getTaskCompletionRate()));
        logger.info("Average task delay: {} seconds", String.format("%.4f", algorithm.getAverageDelay()));
        logger.info("Total energy consumed: {} Joules", String.format("%.2f", algorithm.getTotalEnergyConsumed()));
    }
    
    /**
     * Gets the metrics collector for this simulation.
     * 
     * @return The metrics collector
     */
    public MetricsCollector getMetricsCollector() {
        return metricsCollector;
    }
    
    /**
     * Gets the current simulation time.
     * 
     * @return The current simulation time
     */
    public double getCurrentTime() {
        return currentTime;
    }
    
    /**
     * Gets the simulation end time.
     * 
     * @return The simulation end time
     */
    public double getSimulationEndTime() {
        return simulationEndTime;
    }
    
    /**
     * Gets the simulation time step.
     * 
     * @return The simulation time step
     */
    public double getTimeStep() {
        return timeStep;
    }
    
    /**
     * Checks if the simulation is running.
     * 
     * @return true if the simulation is running, false otherwise
     */
    public boolean isRunning() {
        return isRunning;
    }
    
    /**
     * Gets the list of IoT devices in the simulation.
     * 
     * @return The list of IoT devices
     */
    public List<IoTDevice> getDevices() {
        return devices;
    }
    
    /**
     * Gets the list of edge servers in the simulation.
     * 
     * @return The list of edge servers
     */
    public List<EdgeServer> getServers() {
        return servers;
    }
    
    /**
     * Gets the list of communication channels in the simulation.
     * 
     * @return The list of communication channels
     */
    public List<Channel> getChannels() {
        return channels;
    }
    
    /**
     * Gets the TODG algorithm instance.
     * 
     * @return The TODG algorithm
     */
    public TODGAlgorithm getAlgorithm() {
        return algorithm;
    }
}
