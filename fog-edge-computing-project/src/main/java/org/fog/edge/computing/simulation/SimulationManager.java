package org.fog.edge.computing.simulation;

import java.io.IOException;

import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.edge.computing.orchestration.CustomOrchestrator;
import org.fog.edge.computing.utils.SimulationParameters;
import org.fog.edge.computing.utils.SimulationResults;

/**
 * Manages the simulation lifecycle including initialization, execution, and result collection.
 * This class serves as the central coordinator for the PureEdgeSim-based simulation,
 * handling the initialization of CloudSim Plus, loading configuration parameters,
 * setting up the simulation scenario, and managing the simulation execution.
 * 
 * The SimulationManager follows the Facade design pattern, providing a simplified
 * interface to the complex simulation backend while handling all the necessary
 * setup and coordination between different components.
 * 
 * @author Student
 * @version 1.0
 */
public class SimulationManager {
    /**
     * Array of configuration file paths for the simulation
     */
    private String[] settingsFiles;
    
    /**
     * Output directory path for simulation results
     */
    private String outputFolder;
    
    /**
     * Parameters loaded from configuration files
     */
    private SimulationParameters simulationParameters;
    
    /**
     * The simulation scenario containing entities and topology
     */
    private SimulationResults simulationResults;
    
    /**
     * Custom orchestrator class to be used for task offloading decisions
     */
    private Class<? extends CustomOrchestrator> orchestratorClass;
    
    /**
     * Constructor for the SimulationManager
     * 
     * @param settingsFiles  Array of configuration file paths including simulation parameters,
     *                    applications, edge devices, edge datacenters, and cloud datacenters
     * @param outputFolder Output directory path for storing simulation results
     */
    public SimulationManager(String[] settingsFiles, String outputFolder) {
        this.settingsFiles = settingsFiles;
        this.outputFolder = outputFolder;
        this.simulationParameters = new SimulationParameters();
        this.simulationResults = new SimulationResults(outputFolder);
    }
    
    /**
     * Sets the custom orchestrator class to use for task offloading decisions.
     * The orchestrator is responsible for determining where tasks should be executed
     * (Cloud, Fog, or Mist) based on various parameters like latency sensitivity,
     * resource availability, and network conditions.
     * 
     * @param orchestratorClass The orchestrator class that implements the CustomOrchestrator interface
     */
    public void setCustomOrchestrator(Class<? extends CustomOrchestrator> orchestratorClass) {
        this.orchestratorClass = orchestratorClass;
    }
    
    /**
     * Starts the simulation with the configured settings
     * 
     * This method orchestrates the complete simulation lifecycle:
     * 
     * 1. Initializes the CloudSim Plus simulation engine with timing and user settings
     * 2. Loads all simulation parameters from the configuration files
     * 3. Creates the simulation scenario with all entities (cloud, fog, mist, IoT)
     * 4. Executes the simulation, which involves:
     *    - Task generation by IoT devices
     *    - Task orchestration decisions by the FuzzyDecisionTreeOrchestrator
     *    - Task execution on selected resources
     *    - Network data transfers between entities
     *    - Energy consumption tracking
     * 5. Processes and saves the simulation results to the output folder
     * 
     * During simulation execution, the CloudSim Plus discrete event simulator advances
     * the simulation clock and processes events in chronological order. The simulation
     * continues until all scheduled events are processed or the configured simulation
     * duration is reached.
     * 
     * @throws Exception if there's an error during simulation initialization, execution, or result processing
     */
    public void startSimulation() throws Exception {
        // Initialize CloudSim
        int numUsers = 1;
        Calendar calendar = Calendar.getInstance();
        boolean traceEvents = false;
        CloudSim.init(numUsers, calendar, traceEvents);
        
        // Load simulation parameters from settings files
        loadSimulationParameters();
        
        // Create the simulation scenario
        SimulationScenario scenario = new SimulationScenario(
                simulationParameters, 
                orchestratorClass, 
                simulationResults);
        
        // Start the simulation
        Log.printLine("Starting simulation...");
        CloudSim.startSimulation();
        
        // Process and save results
        simulationResults.processResults();
        
        Log.printLine("Simulation finished!");
    }
    
    /**
     * Loads simulation parameters from the settings files
     * 
     * This method reads and parses all configuration files specified in the settingsFiles array,
     * loading parameters for:
     * 
     * 1. General simulation settings (simulation_parameters.properties)
     *    - Simulation duration, logging level, random seed
     *    - Network parameters (bandwidth, latency)
     *    - Mobility models and patterns
     *    - Energy consumption models
     * 
     * 2. Device configurations (edge_devices.xml)
     *    - Device types, capabilities, and quantities
     *    - CPU, RAM, and storage specifications
     *    - Mobility characteristics
     *    - Battery and energy parameters
     * 
     * 3. Edge data center configurations (edge_datacenters.xml)
     *    - Fog node locations and capabilities
     *    - Host and VM specifications
     *    - Network connectivity
     * 
     * 4. Cloud data center configurations (cloud.xml)
     *    - Remote cloud resources and capabilities
     *    - Cost models
     * 
     * 5. Application configurations (applications.xml)
     *    - Task types and characteristics
     *    - Data sizes and computational requirements
     *    - Latency sensitivity
     * 
     * The loaded parameters are stored in the simulationParameters object for use
     * throughout the simulation. This configuration-driven approach allows for
     * flexible scenario definition without code changes.
     * 
     * @throws IOException if there's an error reading or parsing the settings files
     */
    private void loadSimulationParameters() throws IOException {
        simulationParameters.loadFromFiles(settingsFiles);
    }
}
