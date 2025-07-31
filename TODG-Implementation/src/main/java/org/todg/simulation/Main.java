package org.todg.simulation;

import org.todg.simulation.util.SimulationConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;

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
        
        // Create output directory if it doesn't exist
        File outputDir = new File("output");
        if (!outputDir.exists()) {
            outputDir.mkdirs();
            logger.info("Created output directory: {}", outputDir.getAbsolutePath());
        }
        
        // Load configuration
        SimulationConfig config;
        try {
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
                logger.info("Results available in: {}", outputDir.getAbsolutePath());
            } catch (Exception e) {
                logger.error("Error running simulation: {}", e.getMessage(), e);
                System.err.println("Error running simulation: " + e.getMessage());
                e.printStackTrace();
            }
        } catch (Exception e) {
            logger.error("Error loading configuration: {}", e.getMessage(), e);
            System.err.println("Error loading configuration: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
