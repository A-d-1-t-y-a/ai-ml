package org.fog.edge.computing.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * SimulationResults class for the Fog and Edge Computing project.
 * This class is responsible for collecting, processing, and saving simulation results
 * from the PureEdgeSim-based fog and edge computing simulation. It tracks various
 * performance metrics including task execution statistics, energy consumption,
 * resource utilization, and network usage.
 * 
 * The class implements the evaluation metrics described in the paper:
 * "PureEdgeSim: A Simulation Framework for Performance Evaluation of Cloud, Edge and Mist Computing Environments"
 * by Charafeddine Mechalikh, Hajer Taktak, and Faouzi Moussa.
 * 
 * Results are saved to CSV files for further analysis and visualization.
 * 
 * @author Student
 * @version 1.0
 */
public class SimulationResults {
    // Logger for this class
    private static final Logger LOGGER = Logger.getLogger(SimulationResults.class.getName());
    
    // Output folder path
    private String outputFolder;
    
    // Results storage
    private List<TaskResult> taskResults;
    private Map<String, Double> energyConsumption;
    private Map<String, Double> resourceUtilization;
    private Map<String, Double> networkUsage;
    
    // Performance metrics
    private double averageLatency;
    private double averageExecutionTime;
    private double averageWaitingTime;
    private double taskSuccessRate;
    private double energyEfficiency;
    private double resourceEfficiency;
    private double networkEfficiency;
    
    /**
     * Constructor for SimulationResults
     * 
     * @param outputFolder Path to the output folder for saving results
     */
    public SimulationResults(String outputFolder) {
        this.outputFolder = outputFolder;
        this.taskResults = new ArrayList<>();
        this.energyConsumption = new HashMap<>();
        this.resourceUtilization = new HashMap<>();
        this.networkUsage = new HashMap<>();
    }
    
    /**
     * Records a task result
     * 
     * @param taskId Task ID
     * @param sourceDeviceId Source device ID
     * @param destinationDeviceId Destination device ID
     * @param offloadingTime Offloading time in milliseconds
     * @param executionTime Execution time in milliseconds
     * @param waitingTime Waiting time in milliseconds
     * @param success Whether the task was successful
     * @param offloadingType Type of offloading (Cloud, Fog, Mist)
     */
    public void recordTaskResult(
            int taskId,
            int sourceDeviceId,
            int destinationDeviceId,
            double offloadingTime,
            double executionTime,
            double waitingTime,
            boolean success,
            String offloadingType) {
        
        TaskResult result = new TaskResult(
                taskId,
                sourceDeviceId,
                destinationDeviceId,
                offloadingTime,
                executionTime,
                waitingTime,
                success,
                offloadingType);
        
        taskResults.add(result);
    }
    
    /**
     * Records energy consumption
     * 
     * @param deviceId Device ID
     * @param energyConsumed Energy consumed in watt-hours
     */
    public void recordEnergyConsumption(String deviceId, double energyConsumed) {
        energyConsumption.put(deviceId, 
                energyConsumption.getOrDefault(deviceId, 0.0) + energyConsumed);
    }
    
    /**
     * Records resource utilization
     * 
     * @param deviceId Device ID
     * @param utilizationPercentage Utilization percentage (0.0 to 1.0)
     */
    public void recordResourceUtilization(String deviceId, double utilizationPercentage) {
        resourceUtilization.put(deviceId, utilizationPercentage);
    }
    
    /**
     * Records network usage
     * 
     * @param networkId Network ID
     * @param dataTransferred Data transferred in KB
     */
    public void recordNetworkUsage(String networkId, double dataTransferred) {
        networkUsage.put(networkId, 
                networkUsage.getOrDefault(networkId, 0.0) + dataTransferred);
    }
    
    /**
     * Processes the collected results and calculates performance metrics
     */
    public void processResults() {
        calculateAverageLatency();
        calculateAverageExecutionTime();
        calculateAverageWaitingTime();
        calculateTaskSuccessRate();
        calculateEnergyEfficiency();
        calculateResourceEfficiency();
        calculateNetworkEfficiency();
        
        try {
            saveResultsToFile();
            
            // Generate graphs from the saved CSV files
            LOGGER.info("Generating graphs from simulation results...");
            GraphGenerator graphGenerator = new GraphGenerator(outputFolder);
            graphGenerator.generateAllGraphs();
            LOGGER.info("Graph generation completed.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Calculates the average latency (offloading time + waiting time + execution time)
     */
    private void calculateAverageLatency() {
        if (taskResults.isEmpty()) {
            averageLatency = 0.0;
            return;
        }
        
        double totalLatency = 0.0;
        int successfulTasks = 0;
        
        for (TaskResult result : taskResults) {
            if (result.isSuccess()) {
                totalLatency += result.getOffloadingTime() + result.getWaitingTime() + result.getExecutionTime();
                successfulTasks++;
            }
        }
        
        averageLatency = successfulTasks > 0 ? totalLatency / successfulTasks : 0.0;
    }
    
    /**
     * Calculates the average execution time
     */
    private void calculateAverageExecutionTime() {
        if (taskResults.isEmpty()) {
            averageExecutionTime = 0.0;
            return;
        }
        
        double totalExecutionTime = 0.0;
        int successfulTasks = 0;
        
        for (TaskResult result : taskResults) {
            if (result.isSuccess()) {
                totalExecutionTime += result.getExecutionTime();
                successfulTasks++;
            }
        }
        
        averageExecutionTime = successfulTasks > 0 ? totalExecutionTime / successfulTasks : 0.0;
    }
    
    /**
     * Calculates the average waiting time
     */
    private void calculateAverageWaitingTime() {
        if (taskResults.isEmpty()) {
            averageWaitingTime = 0.0;
            return;
        }
        
        double totalWaitingTime = 0.0;
        int successfulTasks = 0;
        
        for (TaskResult result : taskResults) {
            if (result.isSuccess()) {
                totalWaitingTime += result.getWaitingTime();
                successfulTasks++;
            }
        }
        
        averageWaitingTime = successfulTasks > 0 ? totalWaitingTime / successfulTasks : 0.0;
    }
    
    /**
     * Calculates the task success rate
     */
    private void calculateTaskSuccessRate() {
        if (taskResults.isEmpty()) {
            taskSuccessRate = 0.0;
            return;
        }
        
        int successfulTasks = 0;
        
        for (TaskResult result : taskResults) {
            if (result.isSuccess()) {
                successfulTasks++;
            }
        }
        
        taskSuccessRate = (double) successfulTasks / taskResults.size();
    }
    
    /**
     * Calculates the energy efficiency
     */
    private void calculateEnergyEfficiency() {
        // Energy efficiency is calculated as the number of successful tasks per watt-hour
        if (energyConsumption.isEmpty() || taskResults.isEmpty()) {
            energyEfficiency = 0.0;
            return;
        }
        
        double totalEnergy = 0.0;
        for (double energy : energyConsumption.values()) {
            totalEnergy += energy;
        }
        
        int successfulTasks = 0;
        for (TaskResult result : taskResults) {
            if (result.isSuccess()) {
                successfulTasks++;
            }
        }
        
        energyEfficiency = totalEnergy > 0 ? successfulTasks / totalEnergy : 0.0;
    }
    
    /**
     * Calculates the resource efficiency
     */
    private void calculateResourceEfficiency() {
        // Resource efficiency is calculated as the average resource utilization
        if (resourceUtilization.isEmpty()) {
            resourceEfficiency = 0.0;
            return;
        }
        
        double totalUtilization = 0.0;
        for (double utilization : resourceUtilization.values()) {
            totalUtilization += utilization;
        }
        
        resourceEfficiency = totalUtilization / resourceUtilization.size();
    }
    
    /**
     * Calculates the network efficiency
     */
    private void calculateNetworkEfficiency() {
        // Network efficiency is calculated as the number of successful tasks per KB transferred
        if (networkUsage.isEmpty() || taskResults.isEmpty()) {
            networkEfficiency = 0.0;
            return;
        }
        
        double totalDataTransferred = 0.0;
        for (double data : networkUsage.values()) {
            totalDataTransferred += data;
        }
        
        int successfulTasks = 0;
        for (TaskResult result : taskResults) {
            if (result.isSuccess()) {
                successfulTasks++;
            }
        }
        
        networkEfficiency = totalDataTransferred > 0 ? successfulTasks / totalDataTransferred : 0.0;
    }
    
    /**
     * Saves the results to CSV files
     */
    private void saveResultsToFile() {
        try {
            // Save task results
            saveTaskResultsToFile();
            
            // Save energy consumption
            saveEnergyConsumptionToFile();
            
            // Save resource utilization
            saveResourceUtilizationToFile();
            
            // Save network usage
            saveNetworkUsageToFile();
            
            // Save performance metrics
            savePerformanceMetricsToFile();
            
            System.out.println("Results saved to " + outputFolder);
        } catch (IOException e) {
            System.err.println("Error saving results: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Saves task results to a CSV file
     * 
     * @throws IOException if there's an error writing to the file
     */
    private void saveTaskResultsToFile() throws IOException {
        File file = new File(outputFolder + "/task_results.csv");
        try (FileWriter writer = new FileWriter(file)) {
            // Write header
            writer.write("TaskID,SourceDeviceID,DestinationDeviceID,OffloadingTime,ExecutionTime,WaitingTime,Success,OffloadingType\n");
            
            // Write data
            for (TaskResult result : taskResults) {
                writer.write(String.format("%d,%d,%d,%.2f,%.2f,%.2f,%b,%s\n",
                        result.getTaskId(),
                        result.getSourceDeviceId(),
                        result.getDestinationDeviceId(),
                        result.getOffloadingTime(),
                        result.getExecutionTime(),
                        result.getWaitingTime(),
                        result.isSuccess(),
                        result.getOffloadingType()));
            }
        }
    }
    
    /**
     * Saves energy consumption to a CSV file
     * 
     * @throws IOException if there's an error writing to the file
     */
    private void saveEnergyConsumptionToFile() throws IOException {
        File file = new File(outputFolder + "/energy_consumption.csv");
        try (FileWriter writer = new FileWriter(file)) {
            // Write header
            writer.write("DeviceID,EnergyConsumed\n");
            
            // Write data
            for (Map.Entry<String, Double> entry : energyConsumption.entrySet()) {
                writer.write(String.format("%s,%.2f\n", entry.getKey(), entry.getValue()));
            }
        }
    }
    
    /**
     * Saves resource utilization to a CSV file
     * 
     * @throws IOException if there's an error writing to the file
     */
    private void saveResourceUtilizationToFile() throws IOException {
        File file = new File(outputFolder + "/resource_utilization.csv");
        try (FileWriter writer = new FileWriter(file)) {
            // Write header
            writer.write("DeviceID,UtilizationPercentage\n");
            
            // Write data
            for (Map.Entry<String, Double> entry : resourceUtilization.entrySet()) {
                writer.write(String.format("%s,%.2f\n", entry.getKey(), entry.getValue()));
            }
        }
    }
    
    /**
     * Saves network usage to a CSV file
     * 
     * @throws IOException if there's an error writing to the file
     */
    private void saveNetworkUsageToFile() throws IOException {
        File file = new File(outputFolder + "/network_usage.csv");
        try (FileWriter writer = new FileWriter(file)) {
            // Write header
            writer.write("NetworkID,DataTransferred\n");
            
            // Write data
            for (Map.Entry<String, Double> entry : networkUsage.entrySet()) {
                writer.write(String.format("%s,%.2f\n", entry.getKey(), entry.getValue()));
            }
        }
    }
    
    /**
     * Saves performance metrics to a CSV file
     * 
     * @throws IOException if there's an error writing to the file
     */
    private void savePerformanceMetricsToFile() throws IOException {
        File file = new File(outputFolder + "/performance_metrics.csv");
        try (FileWriter writer = new FileWriter(file)) {
            // Write header
            writer.write("Metric,Value\n");
            
            // Write data
            writer.write(String.format("AverageLatency,%.2f\n", averageLatency));
            writer.write(String.format("AverageExecutionTime,%.2f\n", averageExecutionTime));
            writer.write(String.format("AverageWaitingTime,%.2f\n", averageWaitingTime));
            writer.write(String.format("TaskSuccessRate,%.2f\n", taskSuccessRate));
            writer.write(String.format("EnergyEfficiency,%.2f\n", energyEfficiency));
            writer.write(String.format("ResourceEfficiency,%.2f\n", resourceEfficiency));
            writer.write(String.format("NetworkEfficiency,%.2f\n", networkEfficiency));
        }
    }
    
    // Inner class to store task results
    private static class TaskResult {
        private int taskId;
        private int sourceDeviceId;
        private int destinationDeviceId;
        private double offloadingTime;
        private double executionTime;
        private double waitingTime;
        private boolean success;
        private String offloadingType;
        
        public TaskResult(int taskId, int sourceDeviceId, int destinationDeviceId,
                         double offloadingTime, double executionTime, double waitingTime,
                         boolean success, String offloadingType) {
            this.taskId = taskId;
            this.sourceDeviceId = sourceDeviceId;
            this.destinationDeviceId = destinationDeviceId;
            this.offloadingTime = offloadingTime;
            this.executionTime = executionTime;
            this.waitingTime = waitingTime;
            this.success = success;
            this.offloadingType = offloadingType;
        }
        
        public int getTaskId() {
            return taskId;
        }
        
        public int getSourceDeviceId() {
            return sourceDeviceId;
        }
        
        public int getDestinationDeviceId() {
            return destinationDeviceId;
        }
        
        public double getOffloadingTime() {
            return offloadingTime;
        }
        
        public double getExecutionTime() {
            return executionTime;
        }
        
        public double getWaitingTime() {
            return waitingTime;
        }
        
        public boolean isSuccess() {
            return success;
        }
        
        public String getOffloadingType() {
            return offloadingType;
        }
    }
    
    // Getters for performance metrics
    
    public double getAverageLatency() {
        return averageLatency;
    }
    
    public double getAverageExecutionTime() {
        return averageExecutionTime;
    }
    
    public double getAverageWaitingTime() {
        return averageWaitingTime;
    }
    
    public double getTaskSuccessRate() {
        return taskSuccessRate;
    }
    
    public double getEnergyEfficiency() {
        return energyEfficiency;
    }
    
    public double getResourceEfficiency() {
        return resourceEfficiency;
    }
    
    public double getNetworkEfficiency() {
        return networkEfficiency;
    }
    
    /**
     * Gets the network usage data
     * 
     * @return Map of network usage data
     */
    public Map<String, Double> getNetworkUsageData() {
        return new HashMap<>(networkUsage);
    }
    
    /**
     * Gets the resource utilization data
     * 
     * @return Map of resource utilization data
     */
    public Map<String, Double> getResourceUtilizationData() {
        return new HashMap<>(resourceUtilization);
    }
    
    /**
     * Records orchestration decision time
     * 
     * @param decisionTime Time taken for orchestration decision in seconds
     */
    public void recordOrchestrationTime(double decisionTime) {
        // For now, just log the decision time
        // In a full implementation, this could be stored and analyzed
        LOGGER.info("Orchestration decision time: " + String.format("%.3f", decisionTime) + "s");
    }
    
    /**
     * Records task type distribution
     * 
     * @param taskType Type of task (Cloud, Fog, Mist)
     */
    public void recordTaskTypeDistribution(String taskType) {
        // For now, just log the task type
        // In a full implementation, this could be stored and analyzed
        LOGGER.info("Task classified as: " + taskType);
    }
}
