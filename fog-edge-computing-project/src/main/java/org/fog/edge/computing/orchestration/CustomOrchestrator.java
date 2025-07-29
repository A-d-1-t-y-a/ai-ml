package org.fog.edge.computing.orchestration;

import java.util.List;

import org.fog.edge.computing.simulation.SimulationScenario.CloudDataCenter;
import org.fog.edge.computing.simulation.SimulationScenario.EdgeDataCenter;
import org.fog.edge.computing.simulation.SimulationScenario.EdgeDevice;
import org.fog.edge.computing.simulation.SimulationScenario.IoTDevice;
import org.fog.edge.computing.utils.SimulationParameters;
import org.fog.edge.computing.utils.SimulationResults;

/**
 * Interface for custom orchestrators in the Fog and Edge Computing simulation.
 * Orchestrators are responsible for making intelligent task offloading decisions
 * in the multi-tier computing environment (Cloud, Fog, and Mist).
 * 
 * This interface defines the contract that all orchestration algorithms must follow.
 * Implementations of this interface can use different strategies for task placement,
 * such as the Fuzzy Decision Tree approach described in the PureEdgeSim paper.
 * 
 * The orchestrator plays a critical role in the system by:
 * 1. Analyzing task requirements (e.g., latency sensitivity)
 * 2. Evaluating available resources across the computing continuum
 * 3. Considering network conditions and device mobility
 * 4. Making optimal offloading decisions to maximize performance and efficiency
 * 
 * @author Student
 * @version 1.0
 */
public interface CustomOrchestrator {
    
    /**
     * Configures the orchestrator with simulation entities and parameters
     * 
     * @param cloudDataCenters List of cloud data centers
     * @param edgeDataCenters List of edge data centers (fog nodes)
     * @param edgeDevices List of edge devices (mist computing nodes)
     * @param iotDevices List of IoT devices (sensors)
     * @param parameters Simulation parameters
     * @param results Results collector
     */
    void configure(
            List<?> cloudDataCenters,
            List<?> edgeDataCenters,
            List<?> edgeDevices,
            List<?> iotDevices,
            SimulationParameters parameters,
            SimulationResults results);
    
    /**
     * Makes a decision about where to offload a task
     * 
     * @param task The task to be offloaded
     * @param sourceDevice The device that generated the task
     * @return The destination device for the task
     */
    Object findDestination(Object task, Object sourceDevice);
}
