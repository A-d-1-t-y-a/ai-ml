package com.fog.eedto;

import java.util.logging.Logger;
import java.util.logging.Level;
import java.io.File;

import com.fog.eedto.simulation.Simulation;
import com.fog.eedto.simulation.SimulationResults;

/**
 * Simplified Main class for the EEDTO system that focuses on core functionality
 * without complex visualization dependencies.
 */
public class SimpleMain {
    private static final Logger logger = Logger.getLogger(SimpleMain.class.getName());
    
    public static void main(String[] args) {
        logger.info("Starting EEDTO simulation");
        
        // Create output directory if it doesn't exist
        File outputDir = new File("output");
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        // Create logs directory if it doesn't exist
        File logsDir = new File("logs");
        if (!logsDir.exists()) {
            logsDir.mkdirs();
        }
        
        try {
            // Run baseline simulation
            logger.info("Running baseline simulation");
            SimulationResults baselineResults = runSimulation(
                "Baseline",
                10, 3, 1, 300, 0.1,
                0.33, 0.33, 0.33, 0.2, 5, 3, 2
            );
            
            // Log results
            logResults("Baseline", baselineResults);
            
            // Run energy-focused simulation
            logger.info("Running energy-focused simulation");
            SimulationResults energyResults = runSimulation(
                "Energy-Focused",
                10, 3, 1, 300, 0.1,
                0.6, 0.2, 0.2, 0.2, 5, 3, 2
            );
            
            // Log results
            logResults("Energy-Focused", energyResults);
            
            // Run latency-focused simulation
            logger.info("Running latency-focused simulation");
            SimulationResults latencyResults = runSimulation(
                "Latency-Focused",
                10, 3, 1, 300, 0.1,
                0.2, 0.6, 0.2, 0.2, 5, 3, 2
            );
            
            // Log results
            logResults("Latency-Focused", latencyResults);
            
            // Run security-focused simulation
            logger.info("Running security-focused simulation");
            SimulationResults securityResults = runSimulation(
                "Security-Focused",
                10, 3, 1, 300, 0.1,
                0.2, 0.2, 0.6, 0.2, 5, 3, 2
            );
            
            // Log results
            logResults("Security-Focused", securityResults);
            
            logger.info("EEDTO simulation completed successfully");
            logger.info("Check the logs/ directory for detailed simulation logs");
            logger.info("Check the output/ directory for any generated files");
            
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error during simulation: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Run a simulation with the specified parameters
     */
    private static SimulationResults runSimulation(String name, int numIoTDevices, int numEdgeServers, 
                                                  int numCloudServers, double simulationEndTime, 
                                                  double taskGenerationRate, double energyWeight, 
                                                  double latencyWeight, double securityWeight,
                                                  double energyThreshold, double latencyThreshold, 
                                                  int securityLevel, int blockchainDifficulty) {
        
        Simulation simulation = new Simulation(
            numIoTDevices, numEdgeServers, numCloudServers,
            simulationEndTime, taskGenerationRate,
            energyWeight, latencyWeight, securityWeight,
            energyThreshold, latencyThreshold, securityLevel,
            blockchainDifficulty
        );
        
        simulation.run();
        return simulation.getResults();
    }
    
    /**
     * Log simulation results
     */
    private static void logResults(String name, SimulationResults results) {
        logger.info(String.format("=== %s Simulation Results ===", name));
        logger.info(String.format("Total tasks generated: %d", results.getTotalTasksGenerated()));
        logger.info(String.format("Total tasks completed: %d", results.getTotalTasksCompleted()));
        logger.info(String.format("Total tasks rejected: %d", results.getTotalTasksRejected()));
        logger.info(String.format("Task completion rate: %.2f%%", 
                   results.getTotalTasksGenerated() > 0 ? 
                   (double) results.getTotalTasksCompleted() / results.getTotalTasksGenerated() * 100 : 0));
        logger.info(String.format("Average energy consumed: %.2f J", results.getAverageEnergyConsumption()));
        logger.info(String.format("Average response time: %.2f s", results.getAverageResponseTime()));
        logger.info(String.format("Average execution cost: $%.2f", results.getAverageExecutionCost()));
        logger.info(String.format("Local executions: %d", results.getLocalExecutions()));
        logger.info(String.format("Edge offloads: %d", results.getEdgeOffloads()));
        logger.info(String.format("Cloud offloads: %d", results.getCloudOffloads()));
        logger.info(String.format("Failed offloads: %d", results.getFailedOffloads()));
        logger.info(String.format("Blockchain size: %d blocks", results.getBlockchainSize()));
        logger.info("=====================================");
    }
}
