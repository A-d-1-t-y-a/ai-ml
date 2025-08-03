package com.fog.eedto;

import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.FileHandler;
import java.util.logging.SimpleFormatter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

import com.fog.eedto.simulation.Simulation;
import com.fog.eedto.simulation.SimulationResults;
import com.fog.eedto.util.ChartGenerator;
import com.fog.eedto.util.ConfigurationManager;

/**
 * Simplified Main class for the EEDTO system that focuses on core functionality
 * without complex visualization dependencies.
 */
public class SimpleMain {
    private static final Logger logger = Logger.getLogger(SimpleMain.class.getName());
    private static List<SimulationResults> allResults = new ArrayList<>();
    private static List<String> simulationNames = new ArrayList<>();
    
    public static void main(String[] args) {
        // Initialize configuration
        if (!ConfigurationManager.initialize()) {
            logger.severe("Failed to initialize configuration. Exiting.");
            return;
        }
        
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
        
        // Set up file logging
        setupFileLogging();
        
        logger.info("Starting EEDTO simulation");
        
        try {
            // Get common simulation parameters
            int numIoTDevices = ConfigurationManager.getInt("devices.iot", 10);
            int numEdgeServers = ConfigurationManager.getInt("devices.edge", 3);
            int numCloudServers = ConfigurationManager.getInt("devices.cloud", 1);
            double simulationDuration = ConfigurationManager.getDouble("simulation.duration", 300);
            double timeStep = ConfigurationManager.getDouble("simulation.timeStep", 0.1);
            
            // Run baseline simulation
            logger.info("Running baseline simulation");
            SimulationResults baselineResults = runSimulation(
                "Baseline",
                numIoTDevices, numEdgeServers, numCloudServers, simulationDuration, timeStep,
                ConfigurationManager.getDouble("baseline.energyWeight", 0.33),
                ConfigurationManager.getDouble("baseline.latencyWeight", 0.33),
                ConfigurationManager.getDouble("baseline.securityWeight", 0.33),
                ConfigurationManager.getDouble("baseline.energyThreshold", 0.2),
                ConfigurationManager.getDouble("baseline.latencyThreshold", 5),
                ConfigurationManager.getInt("baseline.securityLevel", 3),
                ConfigurationManager.getInt("baseline.blockchainDifficulty", 2)
            );
            
            // Store results and log
            allResults.add(baselineResults);
            simulationNames.add("Baseline");
            logResults("Baseline", baselineResults);
            
            // Run energy-focused simulation
            logger.info("Running energy-focused simulation");
            SimulationResults energyResults = runSimulation(
                "Energy-Focused",
                numIoTDevices, numEdgeServers, numCloudServers, simulationDuration, timeStep,
                ConfigurationManager.getDouble("energy.energyWeight", 0.6),
                ConfigurationManager.getDouble("energy.latencyWeight", 0.2),
                ConfigurationManager.getDouble("energy.securityWeight", 0.2),
                ConfigurationManager.getDouble("energy.energyThreshold", 0.2),
                ConfigurationManager.getDouble("energy.latencyThreshold", 5),
                ConfigurationManager.getInt("energy.securityLevel", 3),
                ConfigurationManager.getInt("energy.blockchainDifficulty", 2)
            );
            
            // Store results and log
            allResults.add(energyResults);
            simulationNames.add("Energy-Focused");
            logResults("Energy-Focused", energyResults);
            
            // Run latency-focused simulation
            logger.info("Running latency-focused simulation");
            SimulationResults latencyResults = runSimulation(
                "Latency-Focused",
                numIoTDevices, numEdgeServers, numCloudServers, simulationDuration, timeStep,
                ConfigurationManager.getDouble("latency.energyWeight", 0.2),
                ConfigurationManager.getDouble("latency.latencyWeight", 0.6),
                ConfigurationManager.getDouble("latency.securityWeight", 0.2),
                ConfigurationManager.getDouble("latency.energyThreshold", 0.2),
                ConfigurationManager.getDouble("latency.latencyThreshold", 5),
                ConfigurationManager.getInt("latency.securityLevel", 3),
                ConfigurationManager.getInt("latency.blockchainDifficulty", 2)
            );
            
            // Store results and log
            allResults.add(latencyResults);
            simulationNames.add("Latency-Focused");
            logResults("Latency-Focused", latencyResults);
            
            // Run security-focused simulation
            logger.info("Running security-focused simulation");
            SimulationResults securityResults = runSimulation(
                "Security-Focused",
                numIoTDevices, numEdgeServers, numCloudServers, simulationDuration, timeStep,
                ConfigurationManager.getDouble("security.energyWeight", 0.2),
                ConfigurationManager.getDouble("security.latencyWeight", 0.2),
                ConfigurationManager.getDouble("security.securityWeight", 0.6),
                ConfigurationManager.getDouble("security.energyThreshold", 0.2),
                ConfigurationManager.getDouble("security.latencyThreshold", 5),
                ConfigurationManager.getInt("security.securityLevel", 3),
                ConfigurationManager.getInt("security.blockchainDifficulty", 2)
            );
            
            // Store results and log
            allResults.add(securityResults);
            simulationNames.add("Security-Focused");
            logResults("Security-Focused", securityResults);
            
            // Generate all output files
            logger.info("Generating output files...");
            generateCSVResults();
            generateCharts();
            generateSummaryReport();
            
            logger.info("EEDTO simulation completed successfully");
            logger.info("Check the logs/ directory for detailed simulation logs");
            logger.info("Check the output/ directory for CSV files, charts, and reports");
            
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
        logger.info(String.format("Average energy consumed: %.2f J", results.getAverageEnergyPerTask()));
        logger.info(String.format("Average response time: %.2f s", results.getAverageResponseTime()));
        logger.info(String.format("Average execution cost: $%.2f", results.getAverageExecutionCost()));
        logger.info(String.format("Local executions: %d", results.getLocalExecutions()));
        logger.info(String.format("Edge offloads: %d", results.getEdgeOffloads()));
        logger.info(String.format("Cloud offloads: %d", results.getCloudOffloads()));
        logger.info(String.format("Failed offloads: %d", results.getFailedOffloads()));
        logger.info(String.format("Blockchain size: %d blocks", results.getBlockchainSize()));
        logger.info("=====================================");
    }
    
    /**
     * Set up file logging to write logs to files
     */
    private static void setupFileLogging() {
        try {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
            FileHandler fileHandler = new FileHandler("logs/eedto_simulation_" + timestamp + ".log");
            fileHandler.setFormatter(new SimpleFormatter());
            logger.addHandler(fileHandler);
            logger.info("File logging initialized: logs/eedto_simulation_" + timestamp + ".log");
        } catch (IOException e) {
            logger.log(Level.WARNING, "Could not set up file logging: " + e.getMessage());
        }
    }
    
    /**
     * Generate CSV files with simulation results
     */
    private static void generateCSVResults() {
        try {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
            
            // Generate detailed results CSV
            try (PrintWriter writer = new PrintWriter(new FileWriter("output/simulation_results_" + timestamp + ".csv"))) {
                writer.println("Simulation,Tasks_Generated,Tasks_Completed,Tasks_Rejected,Completion_Rate_%," +
                             "Avg_Energy_J,Avg_Response_Time_s,Avg_Cost_$,Local_Executions,Edge_Offloads," +
                             "Cloud_Offloads,Failed_Offloads,Blockchain_Size");
                
                for (int i = 0; i < allResults.size(); i++) {
                    SimulationResults results = allResults.get(i);
                    String name = simulationNames.get(i);
                    
                    writer.printf("%s,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d%n",
                        name,
                        results.getTotalTasksGenerated(),
                        results.getTotalTasksCompleted(),
                        results.getTotalTasksRejected(),
                        results.getTaskCompletionRate(),
                        results.getAverageEnergyPerTask(),
                        results.getAverageResponseTime(),
                        results.getAverageExecutionCost(),
                        results.getLocalExecutions(),
                        results.getEdgeOffloads(),
                        results.getCloudOffloads(),
                        results.getFailedOffloads(),
                        results.getBlockchainSize()
                    );
                }
            }
            
            // Generate comparison metrics CSV
            try (PrintWriter writer = new PrintWriter(new FileWriter("output/comparison_metrics_" + timestamp + ".csv"))) {
                writer.println("Metric,Baseline,Energy_Focused,Latency_Focused,Security_Focused");
                
                writer.printf("Task_Completion_Rate_%%,%.2f,%.2f,%.2f,%.2f%n",
                    allResults.get(0).getTaskCompletionRate(),
                    allResults.get(1).getTaskCompletionRate(),
                    allResults.get(2).getTaskCompletionRate(),
                    allResults.get(3).getTaskCompletionRate());
                    
                writer.printf("Avg_Energy_Per_Task_J,%.2f,%.2f,%.2f,%.2f%n",
                    allResults.get(0).getAverageEnergyPerTask(),
                    allResults.get(1).getAverageEnergyPerTask(),
                    allResults.get(2).getAverageEnergyPerTask(),
                    allResults.get(3).getAverageEnergyPerTask());
                    
                writer.printf("Avg_Response_Time_s,%.2f,%.2f,%.2f,%.2f%n",
                    allResults.get(0).getAverageResponseTime(),
                    allResults.get(1).getAverageResponseTime(),
                    allResults.get(2).getAverageResponseTime(),
                    allResults.get(3).getAverageResponseTime());
                    
                writer.printf("Avg_Execution_Cost_$,%.2f,%.2f,%.2f,%.2f%n",
                    allResults.get(0).getAverageExecutionCost(),
                    allResults.get(1).getAverageExecutionCost(),
                    allResults.get(2).getAverageExecutionCost(),
                    allResults.get(3).getAverageExecutionCost());
            }
            
            logger.info("CSV files generated successfully in output/ directory");
            
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error generating CSV files: " + e.getMessage());
        }
    }
    
    /**
     * Generate charts and visualizations
     */
    private static void generateCharts() {
        try {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
            
            // Convert list to array for chart generation
            String[] simNames = simulationNames.toArray(new String[0]);
            List<SimulationResults> resultsList = allResults;
            
            // Generate comparative energy consumption chart
            ChartGenerator.generateComparativeEnergyChart(
                simNames,
                resultsList,
                "output/energy_comparison_" + timestamp + ".png"
            );
            
            // Generate comparative response time chart
            ChartGenerator.generateComparativeResponseTimeChart(
                simNames,
                resultsList,
                "output/response_time_comparison_" + timestamp + ".png"
            );
            
            // Generate comparative offloading distribution chart
            ChartGenerator.generateComparativeOffloadingChart(
                simNames,
                resultsList,
                "output/offloading_comparison_" + timestamp + ".png"
            );
            
            // Generate comparative cost chart
            ChartGenerator.generateComparativeCostChart(
                simNames,
                resultsList,
                "output/cost_comparison_" + timestamp + ".png"
            );
            
            // Generate individual charts for each simulation
            for (int i = 0; i < allResults.size(); i++) {
                ChartGenerator.generateAllCharts(
                    simulationNames.get(i),
                    allResults.get(i)
                );
            }
            
            logger.info("Charts generated successfully in output/ directory");
            
        } catch (Exception e) {
            logger.log(Level.SEVERE, "Error generating charts: " + e.getMessage());
        }
    }
    
    /**
     * Generate summary report
     */
    private static void generateSummaryReport() {
        try {
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));
            
            try (PrintWriter writer = new PrintWriter(new FileWriter("output/simulation_summary_report_" + timestamp + ".txt"))) {
                writer.println("===============================================");
                writer.println("   EEDTO SIMULATION SUMMARY REPORT");
                writer.println("===============================================");
                writer.println("Generated: " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
                writer.println();
                
                writer.println("SIMULATION CONFIGURATIONS:");
                writer.println("- Baseline: Equal weights (33% each for energy, latency, security)");
                writer.println("- Energy-Focused: 60% energy, 20% latency, 20% security");
                writer.println("- Latency-Focused: 20% energy, 60% latency, 20% security");
                writer.println("- Security-Focused: 20% energy, 20% latency, 60% security");
                writer.println();
                
                writer.println("DETAILED RESULTS:");
                writer.println();
                
                for (int i = 0; i < allResults.size(); i++) {
                    SimulationResults results = allResults.get(i);
                    String name = simulationNames.get(i);
                    
                    writer.println("=== " + name + " Simulation ===");
                    writer.println("Tasks Generated: " + results.getTotalTasksGenerated());
                    writer.println("Tasks Completed: " + results.getTotalTasksCompleted());
                    writer.println("Tasks Rejected: " + results.getTotalTasksRejected());
                    writer.printf("Completion Rate: %.2f%%%n", results.getTaskCompletionRate());
                    writer.printf("Average Energy per Task: %.2f J%n", results.getAverageEnergyPerTask());
                    writer.printf("Average Response Time: %.2f s%n", results.getAverageResponseTime());
                    writer.printf("Average Execution Cost: $%.2f%n", results.getAverageExecutionCost());
                    writer.println("Local Executions: " + results.getLocalExecutions());
                    writer.println("Edge Offloads: " + results.getEdgeOffloads());
                    writer.println("Cloud Offloads: " + results.getCloudOffloads());
                    writer.println("Failed Offloads: " + results.getFailedOffloads());
                    writer.println("Blockchain Size: " + results.getBlockchainSize() + " blocks");
                    writer.println();
                }
                
                writer.println("PERFORMANCE COMPARISON:");
                writer.println();
                
                // Find best performing simulation for each metric
                int bestCompletion = 0, bestEnergy = 0, bestLatency = 0, bestCost = 0;
                for (int i = 1; i < allResults.size(); i++) {
                    if (allResults.get(i).getTaskCompletionRate() > allResults.get(bestCompletion).getTaskCompletionRate()) {
                        bestCompletion = i;
                    }
                    if (allResults.get(i).getAverageEnergyPerTask() < allResults.get(bestEnergy).getAverageEnergyPerTask()) {
                        bestEnergy = i;
                    }
                    if (allResults.get(i).getAverageResponseTime() < allResults.get(bestLatency).getAverageResponseTime()) {
                        bestLatency = i;
                    }
                    if (allResults.get(i).getAverageExecutionCost() < allResults.get(bestCost).getAverageExecutionCost()) {
                        bestCost = i;
                    }
                }
                
                writer.println("Best Task Completion Rate: " + simulationNames.get(bestCompletion) + 
                             String.format(" (%.2f%%)", allResults.get(bestCompletion).getTaskCompletionRate()));
                writer.println("Best Energy Efficiency: " + simulationNames.get(bestEnergy) + 
                             String.format(" (%.2f J/task)", allResults.get(bestEnergy).getAverageEnergyPerTask()));
                writer.println("Best Response Time: " + simulationNames.get(bestLatency) + 
                             String.format(" (%.2f s)", allResults.get(bestLatency).getAverageResponseTime()));
                writer.println("Best Cost Efficiency: " + simulationNames.get(bestCost) + 
                             String.format(" ($%.2f/task)", allResults.get(bestCost).getAverageExecutionCost()));
                
                writer.println();
                writer.println("===============================================");
                writer.println("   END OF REPORT");
                writer.println("===============================================");
            }
            
            logger.info("Summary report generated successfully in output/ directory");
            
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error generating summary report: " + e.getMessage());
        }
    }
}
