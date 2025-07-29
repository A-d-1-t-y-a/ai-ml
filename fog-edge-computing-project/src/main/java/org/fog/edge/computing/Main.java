package org.fog.edge.computing;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import org.fog.edge.computing.orchestration.FuzzyDecisionTreeOrchestrator;
import org.fog.edge.computing.simulation.SimulationManager;

/**
 * Main class for the Fog and Edge Computing project based on PureEdgeSim.
 * This implementation is based on the paper:
 * "PureEdgeSim: A Simulation Framework for Performance Evaluation of Cloud, Edge and Mist Computing Environments"
 * by Charafeddine Mechalikh, Hajer Taktak, and Faouzi Moussa
 * 
 * This class serves as the entry point for the simulation and is responsible for:
 * 1. Setting up the simulation environment and configuration files
 * 2. Creating a unique output directory for simulation results
 * 3. Initializing the SimulationManager with the appropriate settings
 * 4. Configuring the custom FuzzyDecisionTreeOrchestrator for task offloading decisions
 * 5. Starting the simulation execution
 * 
 * The simulation implements a smart campus scenario with a three-tier computing architecture
 * (Cloud-Fog-Mist) and heterogeneous devices. It demonstrates the effectiveness of the
 * fuzzy decision tree approach for task orchestration in edge computing environments.
 * 
 * This proof-of-concept implementation showcases the collaborative interaction among
 * Big Data processing, IoT/Wireless technologies, and service distribution in IoT/Edge
 * environments as required by the assignment specifications.
 * 
 * @author Student
 * @version 1.0
 */
public class Main {

    /**
     * Main method to start the simulation
     * 
     * This method performs the following steps to set up and execute the simulation:
     * 
     * 1. Displays welcome and information messages about the simulation
     * 2. Creates a timestamp-based directory structure for organizing simulation results
     *    (each simulation run gets its own directory with date and time)
     * 3. Defines the paths to all configuration files needed for the simulation:
     *    - simulation_parameters.properties: General simulation settings
     *    - applications.xml: Task types and characteristics
     *    - edge_devices.xml: Edge device specifications (mist computing nodes)
     *    - edge_datacenters.xml: Fog node specifications
     *    - cloud.xml: Cloud data center specifications
     * 4. Initializes the SimulationManager with these configuration files
     * 5. Sets the FuzzyDecisionTreeOrchestrator as the custom orchestration algorithm
     * 6. Starts the simulation and handles any exceptions that might occur
     * 
     * After successful execution, the simulation results will be available in the
     * created output directory, including metrics on task execution times, energy
     * consumption, network usage, and resource utilization across the Cloud-Fog-Mist
     * computing continuum.
     * 
     * @param args command line arguments (not used)
     */
    public static void main(String[] args) {
        // Print welcome message
        System.out.println("Starting PureEdgeSim-based Fog and Edge Computing Simulation...");
        System.out.println("Implementation of Smart Campus scenario from the paper");
        
        // Create simulation timestamp for output files
        String simStartTime = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(Calendar.getInstance().getTime());
        
        // Create output directory if it doesn't exist
        File outputFolder = new File("simulation_results/" + simStartTime);
        if (!outputFolder.exists())
            outputFolder.mkdirs();
        
        // Define simulation settings files
        String[] settingsFiles = {
            "d:/projects/ai-ml/fog-edge-computing-project/src/main/resources/simulation_parameters.properties",
            "d:/projects/ai-ml/fog-edge-computing-project/src/main/resources/applications.xml",
            "d:/projects/ai-ml/fog-edge-computing-project/src/main/resources/edge_devices.xml",
            "d:/projects/ai-ml/fog-edge-computing-project/src/main/resources/edge_datacenters.xml",
            "d:/projects/ai-ml/fog-edge-computing-project/src/main/resources/cloud.xml"
        };
        
        try {
            // Initialize and run the simulation with our custom orchestrator
            SimulationManager simulationManager = new SimulationManager(settingsFiles, outputFolder.getAbsolutePath() + "/");
            simulationManager.setCustomOrchestrator(FuzzyDecisionTreeOrchestrator.class);
            simulationManager.startSimulation();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
