package org.fog.edge.computing.orchestration;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.fog.edge.computing.simulation.SimulationManager.DeviceProperties;
import org.fog.edge.computing.simulation.SimulationManager.TaskProperties;
import org.fog.edge.computing.simulation.SimulationScenario;
import org.fog.edge.computing.utils.SimulationParameters;
import org.fog.edge.computing.utils.SimulationResults;

/**
 * Implementation of the Fuzzy Decision Tree Orchestrator as described in the paper.
 * This orchestrator uses a two-stage fuzzy decision tree approach for task offloading decisions:
 * 1. First stage: Classify tasks into Cloud, Fog, or Mist tasks based on multiple criteria
 * 2. Second stage: If Mist is chosen, find the most suitable edge device
 * 
 * @author Student
 * @version 1.0
 */
public class FuzzyDecisionTreeOrchestrator implements CustomOrchestrator {
    
    // Simulation scenario containing all entities
    private SimulationScenario scenario;
    
    // Simulation parameters and results
    private SimulationParameters parameters;
    private SimulationResults results;
    
    // Fuzzy decision tree parameters
    private Map<String, Double> fogUtilization;
    private double currentWanBandwidth;
    private Map<Integer, Double> deviceMobilityMap;
    private Map<String, Double> wanBandwidth;
    private Map<String, Double> deviceMobility;
    
    /**
     * Constructor for the FuzzyDecisionTreeOrchestrator
     */
    public FuzzyDecisionTreeOrchestrator() {
        // Initialize fuzzy parameters
        this.fogUtilization = new HashMap<>();
        this.wanBandwidth = new HashMap<>();
        this.deviceMobility = new HashMap<>();
        this.deviceMobilityMap = new HashMap<>();
        this.currentWanBandwidth = 5.0; // Default value in Mbps
    }
    
    @Override
    public void configure(SimulationScenario scenario, SimulationParameters parameters, 
                        SimulationResults results) {
        this.scenario = scenario;
        this.parameters = parameters;
        this.results = results;
        
        // Initialize fuzzy decision tree parameters
        initializeFuzzyParameters();
    }
    
    /**
     * Initializes the fuzzy parameters for the decision tree
     */
    private void initializeFuzzyParameters() {
        // Initialize fog utilization (example values)
        fogUtilization.put("low", 0.3);
        fogUtilization.put("medium", 0.6);
        fogUtilization.put("high", 0.9);
        
        // Initialize WAN bandwidth (example values in Mbps)
        wanBandwidth.put("low", 1.0);
        wanBandwidth.put("medium", 5.0);
        wanBandwidth.put("high", 10.0);
        
        // Initialize device mobility (example values)
        deviceMobility.put("stationary", 0.0);
        deviceMobility.put("walking", 1.4);
        deviceMobility.put("vehicle", 10.0);
    }
    
    /**
     * Classifies a task as Cloud, Fog, or Mist based on task properties and source device
     * 
     * @param task The task to classify
     * @param sourceDevice The device that generated the task
     * @return String representing the task classification ("Cloud", "Fog", or "Mist")
     */
    public String classifyTask(Object task, Object sourceDevice) {
        return classifyTaskFirstStage(task, sourceDevice);
    }
    
    @Override
    public Object findDestination(Object task, Object sourceDevice) {
        // Start timing for orchestration decision
        long startTime = System.currentTimeMillis();
        
        // Implementation of the two-stage fuzzy decision tree algorithm
        
        // First stage: Classify the task as Cloud, Fog, or Mist
        String taskType = classifyTaskFirstStage(task, sourceDevice);
        
        // Second stage: Find the most suitable resource based on classification
        Object destination = null;
        
        if ("Cloud".equals(taskType)) {
            destination = findBestCloudDataCenter();
        } else if ("Fog".equals(taskType)) {
            destination = findBestFogNode(task, sourceDevice);
        } else { // Mist
            destination = findBestEdgeDevice(task, sourceDevice);
        }
        
        // Calculate orchestration decision time
        long endTime = System.currentTimeMillis();
        double decisionTime = (endTime - startTime) / 1000.0; // Convert to seconds
        
        // Record metrics for this orchestration decision
        recordOrchestrationMetrics(task, sourceDevice, destination, taskType, decisionTime);
        
        return destination;
    }
    
    /**
     * First stage of the fuzzy decision tree: classify task as Cloud, Fog, or Mist
     */
    private String classifyTaskFirstStage(Object task, Object sourceDevice) {
        // Get fuzzy parameters
        double latencySensitivity = getTaskLatencySensitivity(task);
        double fogUtilization = getCurrentFogUtilization();
        double deviceMobility = getDeviceMobility(sourceDevice);
        double wanBandwidth = getCurrentWanBandwidth();
        
        // Apply fuzzy logic rules
        double cloudScore = calculateCloudScore(latencySensitivity, fogUtilization, deviceMobility, wanBandwidth);
        double fogScore = calculateFogScore(latencySensitivity, fogUtilization, deviceMobility, wanBandwidth);
        double mistScore = calculateMistScore(latencySensitivity, fogUtilization, deviceMobility, wanBandwidth);
        
        // Select the option with the highest score
        if (cloudScore >= fogScore && cloudScore >= mistScore) {
            return "Cloud";
        } else if (fogScore >= mistScore) {
            return "Fog";
        } else {
            return "Mist";
        }
    }
    
    /**
     * Calculate Cloud computing score based on fuzzy parameters
     */
    private double calculateCloudScore(double latencySensitivity, double fogUtilization, 
                                     double deviceMobility, double wanBandwidth) {
        // Cloud is preferred when:
        // - Task is not latency-sensitive (low latency sensitivity)
        // - Fog nodes are highly utilized
        // - Device is mobile (for stability)
        // - WAN bandwidth is high
        
        double score = 0.0;
        
        // Latency sensitivity (inverse relationship)
        score += (1.0 - latencySensitivity) * 0.3;
        
        // Fog utilization (direct relationship)
        score += fogUtilization * 0.3;
        
        // Device mobility (direct relationship)
        score += deviceMobility * 0.2;
        
        // WAN bandwidth (direct relationship)
        score += (wanBandwidth / 10.0) * 0.2; // Normalize assuming max 10 Mbps
        
        return Math.min(1.0, score);
    }
    
    /**
     * Calculate Fog computing score based on fuzzy parameters
     */
    private double calculateFogScore(double latencySensitivity, double fogUtilization, 
                                   double deviceMobility, double wanBandwidth) {
        // Fog is preferred when:
        // - Task has moderate latency sensitivity
        // - Fog nodes are not highly utilized
        // - Device mobility is moderate
        // - WAN bandwidth is moderate
        
        double score = 0.0;
        
        // Latency sensitivity (moderate is best)
        if (latencySensitivity >= 0.3 && latencySensitivity <= 0.7) {
            score += 0.4;
        } else {
            score += 0.2;
        }
        
        // Fog utilization (inverse relationship)
        score += (1.0 - fogUtilization) * 0.3;
        
        // Device mobility (inverse relationship for stability)
        score += (1.0 - deviceMobility) * 0.2;
        
        // WAN bandwidth (moderate is preferred)
        double normalizedBandwidth = wanBandwidth / 10.0;
        if (normalizedBandwidth >= 0.3 && normalizedBandwidth <= 0.7) {
            score += 0.1;
        }
        
        return Math.min(1.0, score);
    }
    
    /**
     * Calculate Mist computing score based on fuzzy parameters
     */
    private double calculateMistScore(double latencySensitivity, double fogUtilization, 
                                    double deviceMobility, double wanBandwidth) {
        // Mist is preferred when:
        // - Task is highly latency-sensitive
        // - Device is stationary or has low mobility
        // - Local processing is beneficial
        
        double score = 0.0;
        
        // Latency sensitivity (direct relationship)
        score += latencySensitivity * 0.4;
        
        // Device mobility (inverse relationship)
        score += (1.0 - deviceMobility) * 0.3;
        
        // Fog utilization (moderate influence)
        score += fogUtilization * 0.2;
        
        // WAN bandwidth (low bandwidth favors local processing)
        score += (1.0 - (wanBandwidth / 10.0)) * 0.1;
        
        return Math.min(1.0, score);
    }
    
    /**
     * Find the best cloud data center for the task
     */
    private Object findBestCloudDataCenter() {
        List<Object> cloudDCs = scenario.getCloudDatacenters();
        
        if (cloudDCs == null || cloudDCs.isEmpty()) {
            System.out.println("No cloud datacenters available.");
            return null;
        }
        
        // For simplicity, return the first available cloud datacenter
        // In a real implementation, this would consider factors like:
        // - Geographic proximity
        // - Current load
        // - Available resources
        // - Network latency
        
        return cloudDCs.get(0);
    }
    
    /**
     * Find the best fog node for the task
     */
    private Object findBestFogNode(Object task, Object sourceDevice) {
        List<Object> fogDCs = scenario.getFogDatacenters();
        
        if (fogDCs == null || fogDCs.isEmpty()) {
            System.out.println("No fog nodes available, falling back to cloud.");
            return findBestCloudDataCenter();
        }
        
        // For simplicity, return the first available fog datacenter
        // In a real implementation, this would consider factors like:
        // - Proximity to source device
        // - Current resource utilization
        // - Network bandwidth
        // - Task requirements
        
        return fogDCs.get(0);
    }
    
    /**
     * Find the best edge device for the task
     */
    private Object findBestEdgeDevice(Object task, Object sourceDevice) {
        List<DeviceInfo> edgeDevices = scenario.getEdgeDevices();
        
        if (edgeDevices == null || edgeDevices.isEmpty()) {
            System.out.println("No edge devices available, falling back to fog.");
            return findBestFogNode(task, sourceDevice);
        }
        
        // For simplicity, return the first available edge device
        // In a real implementation, this would consider factors like:
        // - Device capabilities
        // - Current utilization
        // - Energy constraints
        // - Mobility compatibility
        
        return edgeDevices.get(0);
    }
    
    /**
     * Record orchestration metrics for this decision
     */
    private void recordOrchestrationMetrics(Object task, Object sourceDevice, Object destination, 
                                          String taskType, double decisionTime) {
        // Record orchestration decision time
        results.recordOrchestrationTime(decisionTime);
        
        // Record task type distribution
        results.recordTaskTypeDistribution(taskType);
        
        System.out.println("Task classified as: " + taskType + " (Decision time: " + 
                          String.format("%.3f", decisionTime) + "s)");
    }
    
    /**
     * Get task latency sensitivity
     */
    private double getTaskLatencySensitivity(Object task) {
        if (task instanceof TaskProperties) {
            TaskProperties taskProps = (TaskProperties) task;
            
            // Calculate latency sensitivity based on task characteristics
            long length = taskProps.getLength();
            int pesNumber = taskProps.getPesNumber();
            long fileSize = taskProps.getFileSize();
            long outputSize = taskProps.getOutputSize();
            
            // Tasks with high PE requirements relative to length are typically latency-sensitive
            double peRatio = Math.min(1.0, pesNumber / 4.0); // Normalize to 0-1 range
            
            // Tasks with high data transfer relative to computation are typically latency-sensitive
            double dataRatio = Math.min(1.0, (fileSize + outputSize) / (double)length * 0.01);
            
            // Calculate overall latency sensitivity (weighted average)
            double latencySensitivity = (0.6 * peRatio + 0.4 * dataRatio);
            
            // Ensure the result is in the range [0.0, 1.0]
            return Math.max(0.0, Math.min(1.0, latencySensitivity));
        } else {
            // Default moderate sensitivity if task is not of expected type
            return 0.5;
        }
    }
    
    /**
     * Get current fog utilization
     */
    private double getCurrentFogUtilization() {
        // Get utilization data from simulation results
        Map<String, Double> utilizationData = results.getResourceUtilizationData();
        
        // If no data is available yet, use a default value based on simulation progress
        if (utilizationData == null || utilizationData.isEmpty()) {
            // Dynamic default that increases over time to simulate growing load
            // Start at 30% and grow to 70% over time
            // Use a simple time-based progression (assuming we're partway through simulation)
            double simulationProgress = 0.5; // Default to 50% progress
            return 0.3 + (0.4 * simulationProgress);
        }
        
        // Calculate average utilization for fog nodes
        double totalUtilization = 0.0;
        int fogNodeCount = 0;
        
        for (Map.Entry<String, Double> entry : utilizationData.entrySet()) {
            // Only consider fog nodes (VM IDs 8-19 are fog VMs as per our VM creation logic)
            if (entry.getKey().startsWith("VM_") && 
                entry.getKey().length() > 3) {
                try {
                    int vmId = Integer.parseInt(entry.getKey().substring(3));
                    if (vmId >= 8 && vmId < 20) { // Fog VM ID range
                        totalUtilization += entry.getValue();
                        fogNodeCount++;
                    }
                } catch (NumberFormatException e) {
                    // Skip this entry if it doesn't have a valid VM ID
                }
            }
        }
        
        // Calculate average utilization
        if (fogNodeCount > 0) {
            return totalUtilization / fogNodeCount;
        } else {
            // Default moderate utilization if no fog nodes found
            return 0.5;
        }
    }
    
    /**
     * Get device mobility
     */
    private double getDeviceMobility(Object device) {
        if (device instanceof DeviceProperties) {
            DeviceProperties deviceProps = (DeviceProperties) device;
            
            // Get mobility information from device properties
            boolean isMobile = deviceProps.isMobile();
            
            // For mobile devices, mobility value depends on battery level
            // Devices with lower battery are considered more "mobile" in the sense
            // that they are more likely to disconnect or move away
            if (isMobile) {
                double batteryLevel = deviceProps.getBatteryLevel();
                
                // Normalize battery level (0-100) to mobility value (1.0-0.5)
                // Lower battery = higher mobility value
                return 1.0 - (batteryLevel / 200.0); // Range: 0.5-1.0
            } else {
                // Stationary devices have low mobility value
                return 0.0;
            }
        } else {
            // Default to stationary if device is not of expected type
            return 0.0;
        }
    }
    
    /**
     * Get current WAN bandwidth
     */
    private double getCurrentWanBandwidth() {
        // Get network usage data from simulation results
        Map<String, Double> networkData = results.getNetworkUsageData();
        
        // If no data is available yet, use the parameter value
        if (networkData == null || networkData.isEmpty()) {
            return parameters.getWanBandwidth();
        }
        
        // Get cloud network usage to calculate congestion
        Double cloudNetworkUsage = networkData.get("Network_Cloud");
        
        if (cloudNetworkUsage != null) {
            // Calculate available bandwidth based on usage
            // As usage increases, available bandwidth decreases
            double maxBandwidth = parameters.getWanBandwidth();
            double usageThreshold = 10000.0; // Threshold where congestion starts
            
            // Calculate congestion factor (0.0-1.0)
            double congestionFactor = Math.min(1.0, cloudNetworkUsage / usageThreshold);
            
            // Calculate available bandwidth (decreases with congestion)
            // At max congestion, bandwidth drops to 20% of maximum
            double availableBandwidth = maxBandwidth * (1.0 - (0.8 * congestionFactor));
            
            return availableBandwidth;
        } else {
            // Default to parameter value if no cloud network data found
            return parameters.getWanBandwidth();
        }
    }
    
    /**
     * Inner interface to represent edge device information
     */
    public interface DeviceInfo {
        int getDeviceId();
        String getDeviceName();
        double getCpuUtilization();
        double getMemoryUtilization();
        double getStorageUtilization();
        boolean isBatteryPowered();
        double getBatteryLevel();
        boolean isMobile();
        double getAvailableMips();
        int getNumberOfCores();
        long getAvailableStorage();
    }
}
