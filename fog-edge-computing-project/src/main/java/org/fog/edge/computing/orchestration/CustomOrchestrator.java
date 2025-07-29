package org.fog.edge.computing.orchestration;

import java.util.List;

import org.fog.edge.computing.simulation.SimulationScenario;
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
     * @param scenario The simulation scenario containing all simulation entities
     * @param parameters Simulation parameters
     * @param results Results collector
     */
    void configure(
            SimulationScenario scenario,
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
