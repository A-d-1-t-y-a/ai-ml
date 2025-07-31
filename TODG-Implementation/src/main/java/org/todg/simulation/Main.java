package org.todg.simulation;

import org.todg.simulation.util.SimulationConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main entry point for the TODG simulation.
 * This class initializes and runs the simulation with the specified configuration.
 * 
 * Based on the TODG paper: "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021)
 */
public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    
    /**
     * Main method to run the TODG simulation.
     * 
     * @param args Command line arguments
     *             args[0] (optional): Path to configuration file
     */
    public static void main(String[] args) {
        logger.info("Starting TODG simulation");
        
        // Load configuration
        SimulationConfig config;
        if (args.length > 0) {
            String configFile = args[0];
            logger.info("Loading configuration from file: {}", configFile);
            config = new SimulationConfig(configFile);
        } else {
            logger.info("Using default configuration");
            config = new SimulationConfig();
        }
        
        // Create and run simulation
        TODGSimulator simulator = new TODGSimulator(config);
        
        try {
            // Run simulation
            simulator.runSimulation();
            
            // Generate metrics and charts
            simulator.getMetricsCollector().generateCharts();
            simulator.getMetricsCollector().exportToCSV();
            simulator.getMetricsCollector().exportSummary();
            
            logger.info("Simulation completed successfully");
        } catch (Exception e) {
            logger.error("Error running simulation: {}", e.getMessage(), e);
        }
    }
}
