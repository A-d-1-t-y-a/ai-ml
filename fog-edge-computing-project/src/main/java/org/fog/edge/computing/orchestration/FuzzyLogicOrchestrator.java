package org.fog.edge.computing.orchestration;

import java.util.List;

import org.fog.edge.computing.simulation.SimulationManager;
import org.fog.edge.computing.simulation.SimulationScenario;
import org.fog.edge.computing.utils.SimulationParameters;
import org.fog.edge.computing.utils.SimulationResults;

/**
 * Fuzzy Logic Orchestrator for task offloading decisions
 * 
 * This class implements a Fuzzy Logic based orchestration algorithm 
 * as described in the literature for comparison with the Fuzzy Decision Tree approach.
 * 
 * The algorithm uses fuzzy logic rules to make offloading decisions based on
 * network bandwidth, device capabilities, and task characteristics.
 * 
 * Reference: Sonmez, C., Ozgovde, A., & Ersoy, C. (2019). Fuzzy workload 
 * orchestration for edge computing. IEEE Transactions on Network and Service Management.
 * 
 * @author Student
 * @version 1.0
 */
public class FuzzyLogicOrchestrator implements CustomOrchestrator {
    
    private SimulationScenario scenario;
    private SimulationParameters parameters;
    private SimulationResults results;
    
    // Fuzzy logic parameters
    private double lowBandwidthThreshold = 5.0;    // Mbps
    private double highBandwidthThreshold = 15.0;  // Mbps
    private double lowLatencyThreshold = 50.0;     // ms
    private double highLatencyThreshold = 200.0;   // ms
    
    /**
     * Constructor for FuzzyLogicOrchestrator
     */
    public FuzzyLogicOrchestrator() {
        // Initialize with default parameters
    }
    
    @Override
    public void configure(SimulationScenario scenario, SimulationParameters parameters, 
                        SimulationResults results) {
        this.scenario = scenario;
        this.parameters = parameters;
        this.results = results;
    }
    
    @Override
    public Object findDestination(Object task, Object sourceDevice) {
        // Start timing for orchestration decision
        long startTime = System.currentTimeMillis();
        
        // Fuzzy logic algorithm implementation
        String taskType = classifyTaskFuzzyLogic(task, sourceDevice);
        
        Object destination = null;
        
        if ("Cloud".equals(taskType)) {
            destination = findBestCloudDataCenter();
        } else if ("Fog".equals(taskType)) {
            destination = findBestFogNode();
        } else { // Edge/Mist
            destination = findBestEdgeDevice();
        }
        
        // Calculate orchestration decision time
        long endTime = System.currentTimeMillis();
        double decisionTime = (endTime - startTime) / 1000.0;
        
        // Record metrics for this orchestration decision
        recordOrchestrationMetrics(task, sourceDevice, destination, taskType, decisionTime);
        
        return destination;
    }
    
    /**
     * Fuzzy Logic task classification algorithm
     * 
     * @param task The task to classify
     * @param sourceDevice The source device
     * @return The classification result ("Cloud", "Fog", or "Mist")
     */
    private String classifyTaskFuzzyLogic(Object task, Object sourceDevice) {
        try {
            if (task instanceof SimulationManager.TaskProperties && 
                sourceDevice instanceof SimulationManager.DeviceProperties) {
                
                SimulationManager.TaskProperties taskProps = (SimulationManager.TaskProperties) task;
                SimulationManager.DeviceProperties deviceProps = (SimulationManager.DeviceProperties) sourceDevice;
                
                // Calculate fuzzy membership values
                double bandwidthMembership = calculateBandwidthMembership();
                double latencyMembership = calculateLatencyMembership(taskProps);
                double resourceMembership = calculateResourceMembership(deviceProps);
                
                // Apply fuzzy rules
                double cloudScore = applyCloudRules(bandwidthMembership, latencyMembership, resourceMembership);
                double fogScore = applyFogRules(bandwidthMembership, latencyMembership, resourceMembership);
                double mistScore = applyMistRules(bandwidthMembership, latencyMembership, resourceMembership);
                
                // Defuzzification - choose the option with highest score
                if (cloudScore >= fogScore && cloudScore >= mistScore) {
                    return "Cloud";
                } else if (fogScore >= mistScore) {
                    return "Fog";
                } else {
                    return "Mist";
                }
            }
            
            // Default fallback
            return "Fog";
            
        } catch (Exception e) {
            System.err.println("ERROR in Fuzzy Logic classification: " + e.getMessage());
            return "Cloud"; // Safe fallback
        }
    }
    
    /**
     * Calculate bandwidth membership function
     * 
     * @return Bandwidth membership value (0.0 to 1.0)
     */
    private double calculateBandwidthMembership() {
        // Simulate current network bandwidth (in reality, this would be measured)
        double currentBandwidth = parameters.getWanBandwidth();
        
        if (currentBandwidth <= lowBandwidthThreshold) {
            return 0.0; // Low bandwidth
        } else if (currentBandwidth >= highBandwidthThreshold) {
            return 1.0; // High bandwidth
        } else {
            // Linear membership function between low and high thresholds
            return (currentBandwidth - lowBandwidthThreshold) / 
                   (highBandwidthThreshold - lowBandwidthThreshold);
        }
    }
    
    /**
     * Calculate latency membership function
     * 
     * @param taskProps Task properties
     * @return Latency membership value (0.0 to 1.0)
     */
    private double calculateLatencyMembership(SimulationManager.TaskProperties taskProps) {
        // Estimate task latency sensitivity based on task characteristics
        double estimatedLatency = taskProps.getLength() / 1000.0; // Simple heuristic
        
        if (estimatedLatency <= lowLatencyThreshold) {
            return 1.0; // Low latency requirement (latency-sensitive)
        } else if (estimatedLatency >= highLatencyThreshold) {
            return 0.0; // High latency tolerance
        } else {
            // Linear membership function
            return 1.0 - ((estimatedLatency - lowLatencyThreshold) / 
                         (highLatencyThreshold - lowLatencyThreshold));
        }
    }
    
    /**
     * Calculate resource membership function
     * 
     * @param deviceProps Device properties
     * @return Resource membership value (0.0 to 1.0)
     */
    private double calculateResourceMembership(SimulationManager.DeviceProperties deviceProps) {
        if (deviceProps.isMobile()) {
            // Mobile devices have limited resources
            double batteryLevel = deviceProps.getBatteryLevel();
            return batteryLevel / 100.0; // Normalize battery level
        } else {
            // Non-mobile devices typically have more resources
            return 0.8;
        }
    }
    
    /**
     * Apply fuzzy rules for Cloud offloading
     * 
     * @param bandwidth Bandwidth membership
     * @param latency Latency membership
     * @param resource Resource membership
     * @return Cloud offloading score
     */
    private double applyCloudRules(double bandwidth, double latency, double resource) {
        // Rule 1: If bandwidth is high AND latency tolerance is high, then Cloud is good
        double rule1 = Math.min(bandwidth, 1.0 - latency);
        
        // Rule 2: If resource is low, then Cloud is preferred
        double rule2 = 1.0 - resource;
        
        // Aggregate rules using maximum operator
        return Math.max(rule1, rule2);
    }
    
    /**
     * Apply fuzzy rules for Fog offloading
     * 
     * @param bandwidth Bandwidth membership
     * @param latency Latency membership
     * @param resource Resource membership
     * @return Fog offloading score
     */
    private double applyFogRules(double bandwidth, double latency, double resource) {
        // Rule 1: If bandwidth is medium AND latency is medium, then Fog is good
        double mediumBandwidth = 1.0 - Math.abs(bandwidth - 0.5) * 2.0;
        double mediumLatency = 1.0 - Math.abs(latency - 0.5) * 2.0;
        double rule1 = Math.min(mediumBandwidth, mediumLatency);
        
        // Rule 2: If resource is medium, then Fog is preferred
        double mediumResource = 1.0 - Math.abs(resource - 0.5) * 2.0;
        double rule2 = mediumResource;
        
        // Aggregate rules
        return Math.max(rule1, rule2);
    }
    
    /**
     * Apply fuzzy rules for Mist/Edge offloading
     * 
     * @param bandwidth Bandwidth membership
     * @param latency Latency membership
     * @param resource Resource membership
     * @return Mist offloading score
     */
    private double applyMistRules(double bandwidth, double latency, double resource) {
        // Rule 1: If latency requirement is high (latency-sensitive), then Mist is good
        double rule1 = latency;
        
        // Rule 2: If resource is high, then Mist is preferred
        double rule2 = resource;
        
        // Rule 3: If bandwidth is low, then local processing (Mist) is better
        double rule3 = 1.0 - bandwidth;
        
        // Aggregate rules
        return Math.max(Math.max(rule1, rule2), rule3);
    }
    
    /**
     * Find the best cloud data center
     */
    private Object findBestCloudDataCenter() {
        List<Object> cloudDCs = scenario.getCloudDatacenters();
        return cloudDCs.isEmpty() ? null : cloudDCs.get(0);
    }
    
    /**
     * Find the best fog node
     */
    private Object findBestFogNode() {
        List<Object> fogDCs = scenario.getFogDatacenters();
        return fogDCs.isEmpty() ? null : fogDCs.get(0);
    }
    
    /**
     * Find the best edge device
     */
    private Object findBestEdgeDevice() {
        List<org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator.DeviceInfo> edgeDevices = 
            scenario.getEdgeDevices();
        return edgeDevices.isEmpty() ? null : edgeDevices.get(0);
    }
    
    /**
     * Record orchestration metrics
     */
    private void recordOrchestrationMetrics(Object task, Object sourceDevice, Object destination, 
                                          String taskType, double decisionTime) {
        if (results != null && task instanceof SimulationManager.TaskProperties) {
            SimulationManager.TaskProperties taskProps = (SimulationManager.TaskProperties) task;
            
            // Record the orchestration decision
            results.recordTaskResult(
                taskProps.getId(),
                0, // sourceDeviceId
                1, // destinationDeviceId
                decisionTime,
                0.1, // executionTime (placeholder)
                0.05, // waitingTime (placeholder)
                true, // success
                taskType
            );
        }
    }
}
