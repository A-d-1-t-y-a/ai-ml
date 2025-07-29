package org.fog.edge.computing.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Utility class to load and store simulation parameters from configuration files.
 * This class is responsible for reading and parsing various configuration files
 * (properties files, XML files) and providing access to the simulation parameters
 * through getter and setter methods.
 * 
 * The parameters include general simulation settings, network configurations,
 * device specifications, and application characteristics needed for the
 * PureEdgeSim-based fog and edge computing simulation.
 * 
 * @author Student
 * @version 1.0
 */
public class SimulationParameters {
    // General simulation parameters
    private int numberOfEdgeDevices;
    private int numberOfEdgeDataCenters;
    private int numberOfCloudDataCenters;
    private int simulationDuration;
    private int updateInterval;
    
    // Network parameters
    private double wanBandwidth;
    private double wanLatency;
    private double lanBandwidth;
    private double lanLatency;
    
    // Mobility parameters
    private boolean mobilityEnabled;
    private double minMobilitySpeed;
    private double maxMobilitySpeed;
    private int pauseTime;
    
    // Applications parameters
    private int taskGenerationRate;
    private int taskLength;
    private int taskInputSize;
    private int taskOutputSize;
    
    /**
     * Constructor for SimulationParameters
     */
    public SimulationParameters() {
        // Set default values
        this.numberOfEdgeDevices = 100;
        this.numberOfEdgeDataCenters = 10;
        this.numberOfCloudDataCenters = 1;
        this.simulationDuration = 3600; // 1 hour in seconds
        this.updateInterval = 1; // 1 second
        
        this.wanBandwidth = 100.0; // 100 Mbps
        this.wanLatency = 100.0; // 100 ms
        this.lanBandwidth = 1000.0; // 1 Gbps
        this.lanLatency = 5.0; // 5 ms
        
        this.mobilityEnabled = true;
        this.minMobilitySpeed = 1.0; // 1 m/s
        this.maxMobilitySpeed = 1.4; // 1.4 m/s (walking speed)
        this.pauseTime = 10; // 10 seconds
        
        this.taskGenerationRate = 6; // 6 tasks per minute
        this.taskLength = 1000; // 1000 MI
        this.taskInputSize = 100; // 100 KB
        this.taskOutputSize = 10; // 10 KB
    }
    
    /**
     * Loads parameters from configuration files
     * 
     * @param files Array of paths to configuration files
     * @throws IOException if there's an error reading the files
     */
    public void loadFromFiles(String[] files) throws IOException {
        for (String file : files) {
            if (file.endsWith(".properties")) {
                loadPropertiesFile(file);
            }
            // Other file types (XML, etc.) would be handled by specific loaders
        }
    }
    
    /**
     * Loads parameters from a properties file
     * 
     * @param filePath Path to the properties file
     * @throws IOException if there's an error reading the file
     */
    private void loadPropertiesFile(String filePath) throws IOException {
        Properties properties = new Properties();
        try (FileInputStream fis = new FileInputStream(filePath)) {
            properties.load(fis);
            
            // Load general simulation parameters
            if (properties.containsKey("number_of_edge_devices")) {
                this.numberOfEdgeDevices = Integer.parseInt(properties.getProperty("number_of_edge_devices"));
            }
            
            if (properties.containsKey("number_of_edge_datacenters")) {
                this.numberOfEdgeDataCenters = Integer.parseInt(properties.getProperty("number_of_edge_datacenters"));
            }
            
            if (properties.containsKey("number_of_cloud_datacenters")) {
                this.numberOfCloudDataCenters = Integer.parseInt(properties.getProperty("number_of_cloud_datacenters"));
            }
            
            if (properties.containsKey("simulation_duration")) {
                this.simulationDuration = Integer.parseInt(properties.getProperty("simulation_duration"));
            }
            
            if (properties.containsKey("update_interval")) {
                this.updateInterval = Integer.parseInt(properties.getProperty("update_interval"));
            }
            
            // Load network parameters
            if (properties.containsKey("wan_bandwidth")) {
                this.wanBandwidth = Double.parseDouble(properties.getProperty("wan_bandwidth"));
            }
            
            if (properties.containsKey("wan_latency")) {
                this.wanLatency = Double.parseDouble(properties.getProperty("wan_latency"));
            }
            
            if (properties.containsKey("lan_bandwidth")) {
                this.lanBandwidth = Double.parseDouble(properties.getProperty("lan_bandwidth"));
            }
            
            if (properties.containsKey("lan_latency")) {
                this.lanLatency = Double.parseDouble(properties.getProperty("lan_latency"));
            }
            
            // Load mobility parameters
            if (properties.containsKey("mobility_enabled")) {
                this.mobilityEnabled = Boolean.parseBoolean(properties.getProperty("mobility_enabled"));
            }
            
            if (properties.containsKey("min_mobility_speed")) {
                this.minMobilitySpeed = Double.parseDouble(properties.getProperty("min_mobility_speed"));
            }
            
            if (properties.containsKey("max_mobility_speed")) {
                this.maxMobilitySpeed = Double.parseDouble(properties.getProperty("max_mobility_speed"));
            }
            
            if (properties.containsKey("pause_time")) {
                this.pauseTime = Integer.parseInt(properties.getProperty("pause_time"));
            }
            
            // Load applications parameters
            if (properties.containsKey("task_generation_rate")) {
                this.taskGenerationRate = Integer.parseInt(properties.getProperty("task_generation_rate"));
            }
            
            if (properties.containsKey("task_length")) {
                this.taskLength = Integer.parseInt(properties.getProperty("task_length"));
            }
            
            if (properties.containsKey("task_input_size")) {
                this.taskInputSize = Integer.parseInt(properties.getProperty("task_input_size"));
            }
            
            if (properties.containsKey("task_output_size")) {
                this.taskOutputSize = Integer.parseInt(properties.getProperty("task_output_size"));
            }
        }
    }
    
    // Getters and setters
    
    public int getNumberOfEdgeDevices() {
        return numberOfEdgeDevices;
    }
    
    public void setNumberOfEdgeDevices(int numberOfEdgeDevices) {
        this.numberOfEdgeDevices = numberOfEdgeDevices;
    }
    
    public int getNumberOfEdgeDataCenters() {
        return numberOfEdgeDataCenters;
    }
    
    public void setNumberOfEdgeDataCenters(int numberOfEdgeDataCenters) {
        this.numberOfEdgeDataCenters = numberOfEdgeDataCenters;
    }
    
    public int getNumberOfCloudDataCenters() {
        return numberOfCloudDataCenters;
    }
    
    public void setNumberOfCloudDataCenters(int numberOfCloudDataCenters) {
        this.numberOfCloudDataCenters = numberOfCloudDataCenters;
    }
    
    public int getSimulationDuration() {
        return simulationDuration;
    }
    
    public void setSimulationDuration(int simulationDuration) {
        this.simulationDuration = simulationDuration;
    }
    
    public int getUpdateInterval() {
        return updateInterval;
    }
    
    public void setUpdateInterval(int updateInterval) {
        this.updateInterval = updateInterval;
    }
    
    public double getWanBandwidth() {
        return wanBandwidth;
    }
    
    public void setWanBandwidth(double wanBandwidth) {
        this.wanBandwidth = wanBandwidth;
    }
    
    public double getWanLatency() {
        return wanLatency;
    }
    
    public void setWanLatency(double wanLatency) {
        this.wanLatency = wanLatency;
    }
    
    public double getLanBandwidth() {
        return lanBandwidth;
    }
    
    public void setLanBandwidth(double lanBandwidth) {
        this.lanBandwidth = lanBandwidth;
    }
    
    public double getLanLatency() {
        return lanLatency;
    }
    
    public void setLanLatency(double lanLatency) {
        this.lanLatency = lanLatency;
    }
    
    public boolean isMobilityEnabled() {
        return mobilityEnabled;
    }
    
    public void setMobilityEnabled(boolean mobilityEnabled) {
        this.mobilityEnabled = mobilityEnabled;
    }
    
    public double getMinMobilitySpeed() {
        return minMobilitySpeed;
    }
    
    public void setMinMobilitySpeed(double minMobilitySpeed) {
        this.minMobilitySpeed = minMobilitySpeed;
    }
    
    public double getMaxMobilitySpeed() {
        return maxMobilitySpeed;
    }
    
    public void setMaxMobilitySpeed(double maxMobilitySpeed) {
        this.maxMobilitySpeed = maxMobilitySpeed;
    }
    
    public int getPauseTime() {
        return pauseTime;
    }
    
    public void setPauseTime(int pauseTime) {
        this.pauseTime = pauseTime;
    }
    
    public int getTaskGenerationRate() {
        return taskGenerationRate;
    }
    
    public void setTaskGenerationRate(int taskGenerationRate) {
        this.taskGenerationRate = taskGenerationRate;
    }
    
    public int getTaskLength() {
        return taskLength;
    }
    
    public void setTaskLength(int taskLength) {
        this.taskLength = taskLength;
    }
    
    public int getTaskInputSize() {
        return taskInputSize;
    }
    
    public void setTaskInputSize(int taskInputSize) {
        this.taskInputSize = taskInputSize;
    }
    
    public int getTaskOutputSize() {
        return taskOutputSize;
    }
    
    public void setTaskOutputSize(int taskOutputSize) {
        this.taskOutputSize = taskOutputSize;
    }
}
