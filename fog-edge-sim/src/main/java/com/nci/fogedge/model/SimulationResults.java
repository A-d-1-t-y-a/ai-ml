package com.nci.fogedge.model;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class for storing and analyzing simulation results.
 * Collects metrics throughout the simulation and provides methods for analysis.
 */
public class SimulationResults {
    // Performance metrics
    private List<Double> executionTimes;
    private List<Double> networkDelays;
    private List<Double> energyConsumption;
    private List<Double> resourceUtilization;
    private List<Double> taskSuccessRates;
    
    // Security metrics
    private List<Double> attackDetectionRates;
    private List<Double> securityOverheads;
    
    // Final calculated metrics
    private double averageExecutionTime;
    private double averageNetworkDelay;
    private double totalEnergyConsumption;
    private double averageResourceUtilization;
    private double taskSuccessRate;
    private double attackDetectionRate;
    private double securityOverhead;
    
    /**
     * Constructor initializes all metric lists
     */
    public SimulationResults() {
        executionTimes = new ArrayList<>();
        networkDelays = new ArrayList<>();
        energyConsumption = new ArrayList<>();
        resourceUtilization = new ArrayList<>();
        taskSuccessRates = new ArrayList<>();
        attackDetectionRates = new ArrayList<>();
        securityOverheads = new ArrayList<>();
    }
    
    /**
     * Updates the execution time metric
     * @param executionTime The execution time to add
     */
    public void updateExecutionTime(double executionTime) {
        executionTimes.add(executionTime);
    }
    
    /**
     * Updates the network delay metric
     * @param networkDelay The network delay to add
     */
    public void updateNetworkDelay(double networkDelay) {
        networkDelays.add(networkDelay);
    }
    
    /**
     * Updates the energy consumption metric
     * @param energy The energy consumption to add
     */
    public void updateEnergyConsumption(double energy) {
        energyConsumption.add(energy);
    }
    
    /**
     * Updates the resource utilization metric
     * @param utilization The resource utilization to add
     */
    public void updateResourceUtilization(double utilization) {
        resourceUtilization.add(utilization);
    }
    
    /**
     * Updates the task success rate metric
     * @param successRate The task success rate to add
     */
    public void updateTaskSuccessRate(double successRate) {
        taskSuccessRates.add(successRate);
    }
    
    /**
     * Updates the attack detection rate metric
     * @param detectionRate The attack detection rate to add
     */
    public void updateAttackDetectionRate(double detectionRate) {
        attackDetectionRates.add(detectionRate);
    }
    
    /**
     * Updates the security overhead metric
     * @param overhead The security overhead to add
     */
    public void updateSecurityOverhead(double overhead) {
        securityOverheads.add(overhead);
    }
    
    /**
     * Calculates final metrics from collected data
     */
    public void calculateFinalMetrics() {
        // Calculate average execution time
        averageExecutionTime = executionTimes.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
        
        // Calculate average network delay
        averageNetworkDelay = networkDelays.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
        
        // Calculate total energy consumption
        totalEnergyConsumption = energyConsumption.stream()
                .mapToDouble(Double::doubleValue)
                .sum();
        
        // Calculate average resource utilization
        averageResourceUtilization = resourceUtilization.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
        
        // Calculate average task success rate
        taskSuccessRate = taskSuccessRates.stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
        
        // Calculate security metrics if available
        if (!attackDetectionRates.isEmpty()) {
            attackDetectionRate = attackDetectionRates.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);
        }
        
        if (!securityOverheads.isEmpty()) {
            securityOverhead = securityOverheads.stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);
        }
    }
    
    /**
     * Saves the results to a CSV file
     * @param filePath The path to save the file to
     * @return True if the file was saved successfully, false otherwise
     */
    public boolean saveToFile(String filePath) {
        try (FileWriter writer = new FileWriter(filePath)) {
            // Write header
            writer.write("Metric,Value\n");
            
            // Write performance metrics
            writer.write("Average Execution Time (ms)," + averageExecutionTime + "\n");
            writer.write("Average Network Delay (ms)," + averageNetworkDelay + "\n");
            writer.write("Total Energy Consumption (J)," + totalEnergyConsumption + "\n");
            writer.write("Average Resource Utilization (%)," + averageResourceUtilization + "\n");
            writer.write("Task Success Rate (%)," + taskSuccessRate + "\n");
            
            // Write security metrics if available
            if (!attackDetectionRates.isEmpty()) {
                writer.write("Attack Detection Rate (%)," + attackDetectionRate + "\n");
            }
            
            if (!securityOverheads.isEmpty()) {
                writer.write("Security Overhead (%)," + securityOverhead + "\n");
            }
            
            return true;
        } catch (IOException e) {
            System.err.println("Error saving results to file: " + e.getMessage());
            return false;
        }
    }
    
    // Getters for final metrics
    
    public double getAverageExecutionTime() {
        return averageExecutionTime;
    }
    
    public double getAverageNetworkDelay() {
        return averageNetworkDelay;
    }
    
    public double getTotalEnergyConsumption() {
        return totalEnergyConsumption;
    }
    
    public double getAverageResourceUtilization() {
        return averageResourceUtilization;
    }
    
    public double getTaskSuccessRate() {
        return taskSuccessRate;
    }
    
    public double getAttackDetectionRate() {
        return attackDetectionRate;
    }
    
    public double getSecurityOverhead() {
        return securityOverhead;
    }
}
