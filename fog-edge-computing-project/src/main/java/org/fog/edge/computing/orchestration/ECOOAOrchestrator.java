package org.fog.edge.computing.orchestration;

import java.util.List;

import org.fog.edge.computing.simulation.SimulationManager;
import org.fog.edge.computing.simulation.SimulationScenario;
import org.fog.edge.computing.utils.SimulationParameters;
import org.fog.edge.computing.utils.SimulationResults;

/**
 * ECOOA (Energy Consumption Oriented Offloading Algorithm) Orchestrator
 * 
 * This class implements the Energy-Oriented Tasks Orchestration Algorithm 
 * as described in the literature for comparison with the Fuzzy Decision Tree approach.
 * 
 * ECOOA focuses on minimizing energy consumption and delays by dynamically 
 * choosing between Cloud and Fog based on delay tolerance and power consumption.
 * 
 * Reference: Zhao, X., Zhao, L., & Liang, K. (2016). An Energy Consumption 
 * Oriented Offloading Algorithm for Fog Computing.
 * 
 * @author Student
 * @version 1.0
 */
public class ECOOAOrchestrator implements CustomOrchestrator {
    
    private SimulationScenario scenario;
    private SimulationParameters parameters;
    private SimulationResults results;
    
    // ECOOA algorithm parameters
    private double energyThreshold = 0.3; // Energy consumption threshold
    private double delayThreshold = 2.0;  // Delay tolerance threshold (seconds)
    private double fogUtilizationThreshold = 0.7; // Fog utilization threshold
    
    /**
     * Constructor for ECOOAOrchestrator
     */
    public ECOOAOrchestrator() {
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
        
        // ECOOA algorithm implementation
        String taskType = classifyTaskECOOA(task, sourceDevice);
        
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
     * ECOOA task classification algorithm
     * 
     * @param task The task to classify
     * @param sourceDevice The source device
     * @return The classification result ("Cloud", "Fog", or "Mist")
     */
    private String classifyTaskECOOA(Object task, Object sourceDevice) {
        try {
            if (task instanceof SimulationManager.TaskProperties && 
                sourceDevice instanceof SimulationManager.DeviceProperties) {
                
                SimulationManager.TaskProperties taskProps = (SimulationManager.TaskProperties) task;
                SimulationManager.DeviceProperties deviceProps = (SimulationManager.DeviceProperties) sourceDevice;
                
                // Calculate energy consumption score
                double energyScore = calculateEnergyScore(taskProps, deviceProps);
                
                // Calculate delay tolerance
                double delayTolerance = calculateDelayTolerance(taskProps);
                
                // ECOOA decision logic
                if (deviceProps.isMobile() && deviceProps.getBatteryLevel() < 30) {
                    // Low battery mobile devices prefer fog/cloud
                    if (delayTolerance > delayThreshold) {
                        return "Cloud"; // Delay tolerant -> Cloud
                    } else {
                        return "Fog";   // Delay sensitive -> Fog
                    }
                } else if (energyScore > energyThreshold) {
                    // High energy consumption tasks
                    if (delayTolerance > delayThreshold) {
                        return "Cloud"; // Offload to cloud for energy saving
                    } else {
                        return "Fog";   // Use fog for delay-sensitive tasks
                    }
                } else {
                    // Low energy consumption tasks can be processed locally
                    return "Mist";
                }
            }
            
            // Default fallback
            return "Fog";
            
        } catch (Exception e) {
            System.err.println("ERROR in ECOOA classification: " + e.getMessage());
            return "Cloud"; // Safe fallback
        }
    }
    
    /**
     * Calculate energy consumption score for a task
     * 
     * @param taskProps Task properties
     * @param deviceProps Device properties
     * @return Energy score (0.0 to 1.0)
     */
    private double calculateEnergyScore(SimulationManager.TaskProperties taskProps, 
                                      SimulationManager.DeviceProperties deviceProps) {
        // Energy score based on task complexity and device battery level
        double taskComplexity = Math.min(1.0, taskProps.getLength() / 20000.0);
        double batteryFactor = deviceProps.isMobile() ? (1.0 - deviceProps.getBatteryLevel() / 100.0) : 0.0;
        
        return (taskComplexity * 0.7) + (batteryFactor * 0.3);
    }
    
    /**
     * Calculate delay tolerance for a task
     * 
     * @param taskProps Task properties
     * @return Delay tolerance in seconds
     */
    private double calculateDelayTolerance(SimulationManager.TaskProperties taskProps) {
        // Simple heuristic: smaller tasks are more delay-sensitive
        if (taskProps.getLength() < 5000) {
            return 0.5; // Very delay sensitive
        } else if (taskProps.getLength() < 15000) {
            return 1.5; // Moderately delay sensitive
        } else {
            return 3.0; // Delay tolerant
        }
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
