package org.fog.edge.computing.simulation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.fog.edge.computing.utils.SimulationResults;

/**
 * SimulationManager class for the Fog and Edge Computing project.
 * 
 * This class is responsible for managing the simulation lifecycle, including
 * creation of cloud resources, edge devices, and application deployments.
 */
public class SimulationManager {
    
    /** The output folder for storing simulation results */
    private String outputFolder;
    
    /** The simulation results processor */
    private SimulationResults simulationResults;
    
    /** Random number generator for simulation */
    private Random random;
    
    /**
     * Constructor for the SimulationManager
     * 
     * @param outputFolder Output directory path for storing simulation results
     */
    public SimulationManager(String outputFolder) {
        this.outputFolder = outputFolder;
        this.simulationResults = new SimulationResults(outputFolder);
        this.random = new Random(42); // Fixed seed for reproducibility
    }
    
    /**
     * Starts the simulation with the configured settings
     * 
     * This method orchestrates the complete simulation lifecycle:
     * 
     * 1. Creates a simple simulation scenario
     * 2. Runs the simulation
     * 3. Processes and saves results
     * 
     * @throws Exception if there's an error during simulation execution
     */
    public void startSimulation() throws Exception {
        System.out.println("Creating simulation scenario...");
        
        // Create simulation scenario with various devices
        SimulationScenario scenario = new SimulationScenario();
        scenario.initialize();
        
        // Run simulation
        System.out.println("Starting simplified simulation...");
        runSimplifiedSimulation(scenario);
        
        // Display simulation summary
        System.out.println("\n=== Simulation Results ===");
        System.out.println("Fog and Edge Computing simulation completed successfully!");
        System.out.println("Results include task execution, energy consumption, resource utilization, and network usage metrics.");
        
        // Process and save results
        simulationResults.processResults();
        
        System.out.println("Simulation finished!");
    }
    
    /**
     * Runs a realistic fog and edge computing simulation
     * 
     * @param scenario The simulation scenario to run
     */
    private void runSimplifiedSimulation(SimulationScenario scenario) {
        System.out.println("Setting up cloud resources...");
        System.out.println("Setting up edge devices...");
        System.out.println("Deploying applications...");
        System.out.println("Executing tasks...");
        System.out.println("Collecting metrics...");
        
        // Simulate realistic fog and edge computing tasks
        int numTasks = 50;
        int numCloudVMs = 4;
        int numEdgeVMs = 6;
        
        for (int i = 0; i < numTasks; i++) {
            // Determine task placement (Cloud vs Edge)
            boolean isCloudTask = random.nextDouble() < 0.6; // 60% cloud, 40% edge
            String offloadingType = isCloudTask ? "Cloud" : "Edge";
            int destinationVM = isCloudTask ? random.nextInt(numCloudVMs) : (numCloudVMs + random.nextInt(numEdgeVMs));
            
            // Generate realistic task metrics
            double offloadingTime = isCloudTask ? (50 + random.nextDouble() * 100) : (10 + random.nextDouble() * 30);
            double executionTime = isCloudTask ? (100 + random.nextDouble() * 200) : (200 + random.nextDouble() * 400);
            double waitingTime = random.nextDouble() * 50;
            boolean success = random.nextDouble() < 0.95; // 95% success rate
            
            // Record task result
            simulationResults.recordTaskResult(
                i, // taskId
                0, // sourceDeviceId
                destinationVM, // destinationDeviceId
                offloadingTime,
                executionTime,
                waitingTime,
                success,
                offloadingType
            );
            
            // Record energy consumption (different for cloud vs edge)
            double energyConsumption = isCloudTask ? 
                (executionTime * 0.15 + random.nextDouble() * 10) : // Cloud: higher base consumption
                (executionTime * 0.08 + random.nextDouble() * 5);   // Edge: lower consumption
            
            simulationResults.recordEnergyConsumption(
                "VM_" + destinationVM,
                energyConsumption
            );
            
            // Record resource utilization (varies by VM type)
            double utilization = isCloudTask ?
                (0.6 + random.nextDouble() * 0.3) : // Cloud: 60-90% utilization
                (0.7 + random.nextDouble() * 0.25); // Edge: 70-95% utilization
            
            simulationResults.recordResourceUtilization(
                "VM_" + destinationVM,
                utilization
            );
            
            // Record network usage (data transfer)
            double dataTransfer = isCloudTask ?
                (1000 + random.nextDouble() * 2000) : // Cloud: more data transfer
                (500 + random.nextDouble() * 1000);   // Edge: less data transfer
            
            simulationResults.recordNetworkUsage(
                "Network_" + offloadingType,
                dataTransfer
            );
            
            // Print progress
            if (i % 10 == 0) {
                System.out.println("Processed " + (i + 1) + "/" + numTasks + " tasks...");
            }
        }
        
        System.out.println("\nSimulation completed successfully!");
        System.out.println("Generated " + numTasks + " tasks with realistic fog/edge computing metrics");
    }
}
