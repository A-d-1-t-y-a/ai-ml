package org.jcora.mec;

import org.jcora.mec.config.ConfigurationLoader;
import org.jcora.mec.drl.DRLAgent;
import org.jcora.mec.model.EdgeServer;
import org.jcora.mec.model.IoTDevice;
import org.jcora.mec.simulation.MECEnvironment;
import org.jcora.mec.util.LoggingUtil;
import org.jcora.mec.util.VisualizationUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Main class for the JCORA-MEC simulation.
 * This class coordinates the simulation of the Mobile Edge Computing environment
 * with the Deep Reinforcement Learning agent for joint computation offloading and resource allocation.
 */
public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    
    /**
     * Main method to run the simulation.
     * 
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        logger.info("Starting JCORA-MEC simulation");
        
        // Parse command line arguments
        String configPath = "config/simulation.properties";
        if (args.length > 0) {
            configPath = args[0];
        }
        
        // Load configuration
        ConfigurationLoader config = new ConfigurationLoader(configPath);
        
        // Run simulation
        runSimulation(config);
        
        logger.info("JCORA-MEC simulation completed");
    }
    
    /**
     * Run the simulation with the specified configuration.
     * 
     * @param config Configuration loader
     */
    private static void runSimulation(ConfigurationLoader config) {
        // Create IoT devices and edge servers
        List<IoTDevice> devices = config.createIoTDevices();
        List<EdgeServer> servers = config.createEdgeServers();
        
        // Get DRL agent parameters
        Object[] drlParams = config.getDRLAgentParameters();
        int stateSize = (int) drlParams[0];
        int actionSize = (int) drlParams[1];
        double gamma = (double) drlParams[2];
        double epsilon = (double) drlParams[3];
        double epsilonMin = (double) drlParams[4];
        double epsilonDecay = (double) drlParams[5];
        int batchSize = (int) drlParams[6];
        int replayMemorySize = (int) drlParams[7];
        int targetNetworkUpdateFreq = (int) drlParams[8];
        
        // Create DRL agent
        DRLAgent agent = new DRLAgent(stateSize, actionSize, gamma, epsilon, epsilonMin, epsilonDecay,
                                     batchSize, replayMemorySize, targetNetworkUpdateFreq);
        
        // Get simulation parameters
        double simulationDuration = config.getSimulationDuration();
        double timeStep = config.getTimeStep();
        double taskGenerationProbability = config.getTaskGenerationProbability();
        
        // Create MEC environment
        MECEnvironment environment = new MECEnvironment(devices, servers, agent, simulationDuration,
                                                      timeStep, taskGenerationProbability);
        
        // Run simulation
        environment.runSimulation();
        
        // Generate logs and visualizations
        String outputDir = config.getOutputDirectory();
        String scenarioName = config.getScenarioName();
        
        // Generate CSV files
        LoggingUtil.generateMetricsCSV(environment, outputDir, scenarioName);
        LoggingUtil.generateDeviceStatsCSV(devices, outputDir, scenarioName);
        LoggingUtil.generateServerStatsCSV(servers, outputDir, scenarioName);
        LoggingUtil.generateSummaryReport(environment, outputDir, scenarioName);
        
        // Generate charts
        VisualizationUtil.generateEnergyConsumptionChart(environment, outputDir, scenarioName);
        VisualizationUtil.generateResponseTimeChart(environment, outputDir, scenarioName);
        VisualizationUtil.generateDeadlineMissRateChart(environment, outputDir, scenarioName);
        VisualizationUtil.generateTaskCompletionRateChart(environment, outputDir, scenarioName);
        
        // Log summary
        logger.info("Simulation results:");
        logger.info("Total tasks: {}", environment.getTotalTasks());
        logger.info("Completed tasks: {} ({}%)", environment.getCompletedTasks(), 
                   String.format("%.2f", environment.getTaskCompletionRate()));
        logger.info("Failed tasks: {} ({}%)", environment.getFailedTasks(), 
                   String.format("%.2f", 100.0 - environment.getTaskCompletionRate()));
        logger.info("Total energy consumed: {} J", String.format("%.2f", environment.getTotalEnergyConsumed()));
        logger.info("Average response time: {} s", String.format("%.2f", environment.getAverageResponseTime()));
        logger.info("Deadline miss rate: {}%", String.format("%.2f", environment.getDeadlineMissRate()));
    }
    
    /**
     * Run multiple simulation scenarios for comparison.
     * 
     * @param configPaths List of configuration file paths for different scenarios
     */
    private static void runComparisonSimulations(List<String> configPaths) {
        List<String> scenarioNames = new ArrayList<>();
        List<Double> energyValues = new ArrayList<>();
        List<Double> responseTimeValues = new ArrayList<>();
        List<Double> deadlineMissRateValues = new ArrayList<>();
        List<Double> taskCompletionRateValues = new ArrayList<>();
        
        String outputDir = "output";
        
        // Run each scenario
        for (String configPath : configPaths) {
            // Load configuration
            ConfigurationLoader config = new ConfigurationLoader(configPath);
            
            // Create IoT devices and edge servers
            List<IoTDevice> devices = config.createIoTDevices();
            List<EdgeServer> servers = config.createEdgeServers();
            
            // Get DRL agent parameters
            Object[] drlParams = config.getDRLAgentParameters();
            int stateSize = (int) drlParams[0];
            int actionSize = (int) drlParams[1];
            double gamma = (double) drlParams[2];
            double epsilon = (double) drlParams[3];
            double epsilonMin = (double) drlParams[4];
            double epsilonDecay = (double) drlParams[5];
            int batchSize = (int) drlParams[6];
            int replayMemorySize = (int) drlParams[7];
            int targetNetworkUpdateFreq = (int) drlParams[8];
            
            // Create DRL agent
            DRLAgent agent = new DRLAgent(stateSize, actionSize, gamma, epsilon, epsilonMin, epsilonDecay,
                                         batchSize, replayMemorySize, targetNetworkUpdateFreq);
            
            // Get simulation parameters
            double simulationDuration = config.getSimulationDuration();
            double timeStep = config.getTimeStep();
            double taskGenerationProbability = config.getTaskGenerationProbability();
            
            // Create MEC environment
            MECEnvironment environment = new MECEnvironment(devices, servers, agent, simulationDuration,
                                                          timeStep, taskGenerationProbability);
            
            // Run simulation
            environment.runSimulation();
            
            // Generate logs and visualizations
            String scenarioName = config.getScenarioName();
            outputDir = config.getOutputDirectory();
            
            // Generate CSV files
            LoggingUtil.generateMetricsCSV(environment, outputDir, scenarioName);
            LoggingUtil.generateDeviceStatsCSV(devices, outputDir, scenarioName);
            LoggingUtil.generateServerStatsCSV(servers, outputDir, scenarioName);
            LoggingUtil.generateSummaryReport(environment, outputDir, scenarioName);
            
            // Generate charts
            VisualizationUtil.generateEnergyConsumptionChart(environment, outputDir, scenarioName);
            VisualizationUtil.generateResponseTimeChart(environment, outputDir, scenarioName);
            VisualizationUtil.generateDeadlineMissRateChart(environment, outputDir, scenarioName);
            VisualizationUtil.generateTaskCompletionRateChart(environment, outputDir, scenarioName);
            
            // Collect metrics for comparison
            scenarioNames.add(scenarioName);
            energyValues.add(environment.getTotalEnergyConsumed());
            responseTimeValues.add(environment.getAverageResponseTime());
            deadlineMissRateValues.add(environment.getDeadlineMissRate());
            taskCompletionRateValues.add(environment.getTaskCompletionRate());
        }
        
        // Generate comparison charts
        VisualizationUtil.generateComparisonChart(scenarioNames, energyValues, responseTimeValues,
                                                deadlineMissRateValues, taskCompletionRateValues, outputDir);
    }
}
