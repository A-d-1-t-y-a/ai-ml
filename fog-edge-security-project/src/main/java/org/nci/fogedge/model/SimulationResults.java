package org.nci.fogedge.model;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.security.AttackSimulator;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.topology.NetworkTopology;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

/**
 * Class to collect and analyze simulation metrics
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class SimulationResults {
    private static final Logger logger = LogManager.getLogger(SimulationResults.class);
    
    // Performance metrics
    private Map<Integer, Double> dataGeneratedPerStep;
    private Map<Integer, Double> dataProcessedAtEdgePerStep;
    private Map<Integer, Double> dataProcessedAtFogPerStep;
    private Map<Integer, Double> processingTimePerStep;
    
    // Security metrics
    private Map<Integer, Integer> attacksDetectedPerStep;
    private Map<Integer, Integer> attacksPreventedPerStep;
    private Map<Integer, Double> securityOverheadPerStep;
    
    // Energy metrics
    private Map<Integer, Double> energyConsumptionPerStep;
    private Map<Integer, Double> securityEnergyOverheadPerStep;
    
    // Summary metrics
    private double totalDataGenerated;
    private double totalDataProcessedAtEdge;
    private double totalDataProcessedAtFog;
    private double averageProcessingTime;
    private int totalAttacksDetected;
    private int totalAttacksPrevented;
    private double averageSecurityOverhead;
    private double totalEnergyConsumption;
    private double securityEnergyPercentage;
    
    public SimulationResults() {
        this.dataGeneratedPerStep = new HashMap<>();
        this.dataProcessedAtEdgePerStep = new HashMap<>();
        this.dataProcessedAtFogPerStep = new HashMap<>();
        this.processingTimePerStep = new HashMap<>();
        this.attacksDetectedPerStep = new HashMap<>();
        this.attacksPreventedPerStep = new HashMap<>();
        this.securityOverheadPerStep = new HashMap<>();
        this.energyConsumptionPerStep = new HashMap<>();
        this.securityEnergyOverheadPerStep = new HashMap<>();
    }
    
    public void collectPerformanceMetrics(NetworkTopology topology, int step) {
        // Collect data generation and processing metrics
        double dataGenerated = topology.calculateTotalDataGenerated();
        double dataProcessedAtEdge = topology.calculateDataProcessedAtEdge();
        double dataProcessedAtFog = topology.calculateDataProcessedAtFog();
        double processingTime = topology.calculateProcessingTime();
        
        dataGeneratedPerStep.put(step, dataGenerated);
        dataProcessedAtEdgePerStep.put(step, dataProcessedAtEdge);
        dataProcessedAtFogPerStep.put(step, dataProcessedAtFog);
        processingTimePerStep.put(step, processingTime);
        
        logger.debug("Step {}: Generated {}, Processed at Edge {}, Processed at Fog {}, Processing Time {}",
                step, dataGenerated, dataProcessedAtEdge, dataProcessedAtFog, processingTime);
    }
    
    public void collectSecurityMetrics(SecurityManager securityManager, AttackSimulator attackSimulator, int step) {
        // Collect security metrics
        int attacksDetected = securityManager.getDetectedAttacks();
        int attacksPrevented = securityManager.getPreventedAttacks();
        double securityOverhead = securityManager.calculateSecurityOverhead();
        
        attacksDetectedPerStep.put(step, attacksDetected);
        attacksPreventedPerStep.put(step, attacksPrevented);
        securityOverheadPerStep.put(step, securityOverhead);
        
        logger.debug("Step {}: Attacks Detected {}, Attacks Prevented {}, Security Overhead {}",
                step, attacksDetected, attacksPrevented, securityOverhead);
    }
    
    public void collectEnergyMetrics(NetworkTopology topology, SecurityManager securityManager, int step) {
        // Collect energy consumption metrics
        double energyConsumption = topology.calculateTotalEnergyConsumption();
        double securityEnergyOverhead = securityManager.calculateEnergyOverhead();
        
        energyConsumptionPerStep.put(step, energyConsumption);
        securityEnergyOverheadPerStep.put(step, securityEnergyOverhead);
        
        logger.debug("Step {}: Energy Consumption {}, Security Energy Overhead {}",
                step, energyConsumption, securityEnergyOverhead);
    }
    
    public void finalizeResults() {
        // Calculate summary metrics
        totalDataGenerated = dataGeneratedPerStep.values().stream().mapToDouble(Double::doubleValue).sum();
        totalDataProcessedAtEdge = dataProcessedAtEdgePerStep.values().stream().mapToDouble(Double::doubleValue).sum();
        totalDataProcessedAtFog = dataProcessedAtFogPerStep.values().stream().mapToDouble(Double::doubleValue).sum();
        averageProcessingTime = processingTimePerStep.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        totalAttacksDetected = attacksDetectedPerStep.values().stream().mapToInt(Integer::intValue).sum();
        totalAttacksPrevented = attacksPreventedPerStep.values().stream().mapToInt(Integer::intValue).sum();
        averageSecurityOverhead = securityOverheadPerStep.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        totalEnergyConsumption = energyConsumptionPerStep.values().stream().mapToDouble(Double::doubleValue).sum();
        double totalSecurityEnergy = securityEnergyOverheadPerStep.values().stream().mapToDouble(Double::doubleValue).sum();
        securityEnergyPercentage = (totalEnergyConsumption > 0) ? (totalSecurityEnergy / totalEnergyConsumption) * 100.0 : 0.0;
        
        logger.info("Finalized simulation results");
    }
    
    public void printResults() {
        System.out.println("\n=== Fog and Edge Computing Security Simulation Results ===");
        System.out.println("\nPerformance Metrics:");
        System.out.println("Total Data Generated: " + totalDataGenerated + " KB");
        System.out.println("Total Data Processed at Edge: " + totalDataProcessedAtEdge + " KB");
        System.out.println("Total Data Processed at Fog: " + totalDataProcessedAtFog + " KB");
        System.out.println("Data Reduction at Edge: " + calculatePercentage(totalDataProcessedAtEdge, totalDataGenerated) + "%");
        System.out.println("Data Reduction at Fog: " + calculatePercentage(totalDataProcessedAtFog, totalDataProcessedAtEdge) + "%");
        System.out.println("Average Processing Time: " + averageProcessingTime + " ms");
        
        System.out.println("\nSecurity Metrics:");
        System.out.println("Total Attacks Detected: " + totalAttacksDetected);
        System.out.println("Total Attacks Prevented: " + totalAttacksPrevented);
        System.out.println("Attack Prevention Rate: " + 
                (totalAttacksDetected > 0 ? (totalAttacksPrevented * 100.0 / totalAttacksDetected) : 0.0) + "%");
        System.out.println("Average Security Overhead: " + averageSecurityOverhead + " ms");
        
        System.out.println("\nEnergy Metrics:");
        System.out.println("Total Energy Consumption: " + totalEnergyConsumption + " mJ");
        System.out.println("Security Energy Overhead: " + securityEnergyPercentage + "%");
        
        System.out.println("\n=== Analysis ===");
        analyzeResults();
    }
    
    private void analyzeResults() {
        // Analyze the trade-off between security and performance
        System.out.println("Security vs. Performance Trade-off:");
        if (averageSecurityOverhead < 10.0) {
            System.out.println("- Low security overhead with minimal impact on performance");
        } else if (averageSecurityOverhead < 30.0) {
            System.out.println("- Moderate security overhead with acceptable performance impact");
        } else {
            System.out.println("- High security overhead with significant performance impact");
        }
        
        // Analyze the effectiveness of security measures
        System.out.println("\nSecurity Effectiveness:");
        double preventionRate = totalAttacksDetected > 0 ? (totalAttacksPrevented * 100.0 / totalAttacksDetected) : 0.0;
        if (preventionRate > 90.0) {
            System.out.println("- Excellent attack prevention rate");
        } else if (preventionRate > 70.0) {
            System.out.println("- Good attack prevention rate");
        } else {
            System.out.println("- Needs improvement in attack prevention");
        }
        
        // Analyze energy efficiency
        System.out.println("\nEnergy Efficiency:");
        if (securityEnergyPercentage < 15.0) {
            System.out.println("- Security measures are energy efficient");
        } else if (securityEnergyPercentage < 30.0) {
            System.out.println("- Security measures have moderate energy impact");
        } else {
            System.out.println("- Security measures have high energy impact");
        }
        
        // Analyze data reduction efficiency
        double edgeReduction = calculatePercentage(totalDataProcessedAtEdge, totalDataGenerated);
        double fogReduction = calculatePercentage(totalDataProcessedAtFog, totalDataProcessedAtEdge);
        
        System.out.println("\nData Reduction Efficiency:");
        System.out.println("- Edge layer reduced data by " + edgeReduction + "%");
        System.out.println("- Fog layer reduced data by " + fogReduction + "%");
        System.out.println("- Total data reduction: " + 
                calculatePercentage(totalDataProcessedAtFog, totalDataGenerated) + "%");
    }
    
    private double calculatePercentage(double part, double whole) {
        if (whole == 0) return 0;
        return 100.0 - ((part / whole) * 100.0);
    }
    
    /**
     * Save simulation results to a file
     * @param filename The filename to save results to
     */
    public void saveToFile(String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("=== Fog and Edge Computing Security Simulation Results ===");
            writer.println("Based on the paper: \"An Overview of Fog Computing and Edge Computing Security and Privacy Issues\"");
            writer.println("(Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)");
            
            writer.println("\nPerformance Metrics:");
            writer.println("Total Data Generated: " + totalDataGenerated + " KB");
            writer.println("Total Data Processed at Edge: " + totalDataProcessedAtEdge + " KB");
            writer.println("Total Data Processed at Fog: " + totalDataProcessedAtFog + " KB");
            writer.println("Data Reduction at Edge: " + calculatePercentage(totalDataProcessedAtEdge, totalDataGenerated) + "%");
            writer.println("Data Reduction at Fog: " + calculatePercentage(totalDataProcessedAtFog, totalDataProcessedAtEdge) + "%");
            writer.println("Average Processing Time: " + averageProcessingTime + " ms");
            
            writer.println("\nSecurity Metrics:");
            writer.println("Total Attacks Detected: " + totalAttacksDetected);
            writer.println("Total Attacks Prevented: " + totalAttacksPrevented);
            writer.println("Attack Prevention Rate: " + 
                    (totalAttacksDetected > 0 ? (totalAttacksPrevented * 100.0 / totalAttacksDetected) : 0.0) + "%");
            writer.println("Average Security Overhead: " + averageSecurityOverhead + " ms");
            
            writer.println("\nEnergy Metrics:");
            writer.println("Total Energy Consumption: " + totalEnergyConsumption + " mJ");
            writer.println("Security Energy Overhead: " + securityEnergyPercentage + "%");
            
            writer.println("\n=== Detailed Metrics Per Step ===");
            
            writer.println("\nStep,Data Generated (KB),Data Processed Edge (KB),Data Processed Fog (KB),Processing Time (ms)," +
                    "Attacks Detected,Attacks Prevented,Security Overhead (ms),Energy Consumption (mJ),Security Energy (mJ)");
            
            // Print per-step metrics in CSV format
            for (int step : dataGeneratedPerStep.keySet()) {
                writer.printf("%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f,%.2f,%.2f\n",
                        step,
                        dataGeneratedPerStep.getOrDefault(step, 0.0),
                        dataProcessedAtEdgePerStep.getOrDefault(step, 0.0),
                        dataProcessedAtFogPerStep.getOrDefault(step, 0.0),
                        processingTimePerStep.getOrDefault(step, 0.0),
                        attacksDetectedPerStep.getOrDefault(step, 0),
                        attacksPreventedPerStep.getOrDefault(step, 0),
                        securityOverheadPerStep.getOrDefault(step, 0.0),
                        energyConsumptionPerStep.getOrDefault(step, 0.0),
                        securityEnergyOverheadPerStep.getOrDefault(step, 0.0));
            }
            
            writer.println("\n=== Analysis ===");
            
            // Security vs. Performance Trade-off
            writer.println("Security vs. Performance Trade-off:");
            if (averageSecurityOverhead < 10.0) {
                writer.println("- Low security overhead with minimal impact on performance");
            } else if (averageSecurityOverhead < 30.0) {
                writer.println("- Moderate security overhead with acceptable performance impact");
            } else {
                writer.println("- High security overhead with significant performance impact");
            }
            
            // Security Effectiveness
            writer.println("\nSecurity Effectiveness:");
            double preventionRate = totalAttacksDetected > 0 ? (totalAttacksPrevented * 100.0 / totalAttacksDetected) : 0.0;
            if (preventionRate > 90.0) {
                writer.println("- Excellent attack prevention rate");
            } else if (preventionRate > 70.0) {
                writer.println("- Good attack prevention rate");
            } else {
                writer.println("- Needs improvement in attack prevention");
            }
            
            // Energy Efficiency
            writer.println("\nEnergy Efficiency:");
            if (securityEnergyPercentage < 15.0) {
                writer.println("- Security measures are energy efficient");
            } else if (securityEnergyPercentage < 30.0) {
                writer.println("- Security measures have moderate energy impact");
            } else {
                writer.println("- Security measures have high energy impact");
            }
            
            // Data Reduction Efficiency
            double edgeReduction = calculatePercentage(totalDataProcessedAtEdge, totalDataGenerated);
            double fogReduction = calculatePercentage(totalDataProcessedAtFog, totalDataProcessedAtEdge);
            
            writer.println("\nData Reduction Efficiency:");
            writer.println("- Edge layer reduced data by " + edgeReduction + "%");
            writer.println("- Fog layer reduced data by " + fogReduction + "%");
            writer.println("- Total data reduction: " + 
                    calculatePercentage(totalDataProcessedAtFog, totalDataGenerated) + "%");
            
            logger.info("Simulation results saved to file: {}", filename);
        } catch (IOException e) {
            logger.error("Error saving results to file: {}", e.getMessage());
            System.err.println("Error saving results to file: " + e.getMessage());
        }
    }
}
