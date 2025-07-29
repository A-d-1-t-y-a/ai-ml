package org.fog.edge.computing;

import java.io.File;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import org.fog.edge.computing.simulation.SimulationManager;

/**
 * Main class for the Fog and Edge Computing project based on CloudSim Plus.
 * This implementation is inspired by the paper:
 * "PureEdgeSim: A Simulation Framework for Performance Evaluation of Cloud, Edge and Mist Computing Environments"
 * by Charafeddine Mechalikh, Hajer Taktak, and Faouzi Moussa
 * 
 * Migrated to use CloudSim Plus as the underlying simulation framework.
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
     */
    public static void main(String[] args) throws Exception {
        // Create output directories if they don't exist
        createOutputDirectories();
        
        // Set up configuration file paths
        setupConfigurationPaths();
        
        // Start the simulation
        startSimulation();
    }
    
    /**
     * Creates necessary output directories for simulation results
     */
    private static void createOutputDirectories() {
        String outputPath = "./output";
        File outputDir = new File(outputPath);
        if (!outputDir.exists()) {
            outputDir.mkdir();
            System.out.println("Created output directory: " + outputDir.getAbsolutePath());
        }
    }
    
    /**
     * Sets up configuration file paths
     */
    private static void setupConfigurationPaths() {
        // Configuration files are now located in the ./config directory
        String configPath = "./config";
        File configDir = new File(configPath);
        
        // Create config directory if it doesn't exist
        if (!configDir.exists()) {
            configDir.mkdir();
            System.out.println("Created configuration directory: " + configDir.getAbsolutePath());
        }
    }
    
    /**
     * Starts the simulation process
     * 
     * @throws Exception if there's an error during simulation
     */
    private static void startSimulation() throws Exception {
        System.out.println("Starting fog and edge computing simulation...");
        
        // Define output folder for results
        String outputFolder = Paths.get(".", "output").toString();
        System.out.println("Results will be saved to: " + new File(outputFolder).getAbsolutePath());
        
        // Create and initialize the simulation manager
        SimulationManager simulationManager = new SimulationManager(outputFolder);
        
        // Start the simulation
        simulationManager.startSimulation();
        
        System.out.println("Simulation completed successfully.");
    }
}
