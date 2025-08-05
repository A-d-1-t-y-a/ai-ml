package org.jcora.mec.util;

import org.jcora.mec.model.EdgeServer;
import org.jcora.mec.model.IoTDevice;
import org.jcora.mec.model.Task;
import org.jcora.mec.simulation.MECEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Utility class for logging simulation results and generating CSV files.
 */
public class LoggingUtil {
    private static final Logger logger = LoggerFactory.getLogger(LoggingUtil.class);
    
    /**
     * Create the output directory if it doesn't exist.
     * 
     * @param outputDir Path to the output directory
     * @return True if the directory exists or was created successfully, false otherwise
     */
    public static boolean createOutputDirectory(String outputDir) {
        Path path = Paths.get(outputDir);
        if (!Files.exists(path)) {
            try {
                Files.createDirectories(path);
                logger.info("Created output directory: {}", outputDir);
                return true;
            } catch (IOException e) {
                logger.error("Failed to create output directory: {}", e.getMessage());
                return false;
            }
        }
        return true;
    }
    
    /**
     * Generate a CSV file with simulation metrics over time.
     * 
     * @param environment MEC environment with simulation results
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateMetricsCSV(MECEnvironment environment, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_metrics_%s.csv", outputDir, scenarioName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write header
            writer.write("TimeStep,EnergyConsumption,ResponseTime,DeadlineMissRate,TaskCompletionRate");
            writer.newLine();
            
            // Get metrics history
            List<Double> energyHistory = environment.getEnergyConsumptionHistory();
            List<Double> responseTimeHistory = environment.getResponseTimeHistory();
            List<Double> deadlineMissRateHistory = environment.getDeadlineMissRateHistory();
            List<Double> taskCompletionRateHistory = environment.getTaskCompletionRateHistory();
            
            // Write data
            int steps = energyHistory.size();
            for (int i = 0; i < steps; i++) {
                writer.write(String.format("%d,%.6f,%.6f,%.6f,%.6f",
                        i,
                        energyHistory.get(i),
                        responseTimeHistory.get(i),
                        deadlineMissRateHistory.get(i),
                        taskCompletionRateHistory.get(i)));
                writer.newLine();
            }
            
            logger.info("Generated metrics CSV file: {}", filename);
        } catch (IOException e) {
            logger.error("Failed to generate metrics CSV file: {}", e.getMessage());
        }
    }
    
    /**
     * Generate a CSV file with device statistics.
     * 
     * @param devices List of IoT devices
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateDeviceStatsCSV(List<IoTDevice> devices, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_device_stats_%s.csv", outputDir, scenarioName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write header
            writer.write("DeviceID,Name,ProcessingPower,BatteryCapacity,RemainingBattery,EnergyConsumed,CompletedTasks,FailedTasks");
            writer.newLine();
            
            // Write data for each device
            for (IoTDevice device : devices) {
                writer.write(String.format("%d,%s,%.2f,%.2f,%.2f,%.2f,%d,%d",
                        device.getId(),
                        device.getName(),
                        device.getProcessingPower(),
                        device.getBatteryCapacity(),
                        device.getRemainingBattery(),
                        device.getTotalEnergyConsumed(),
                        device.getCompletedTasks(),
                        device.getFailedTasks()));
                writer.newLine();
            }
            
            logger.info("Generated device stats CSV file: {}", filename);
        } catch (IOException e) {
            logger.error("Failed to generate device stats CSV file: {}", e.getMessage());
        }
    }
    
    /**
     * Generate a CSV file with server statistics.
     * 
     * @param servers List of edge servers
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateServerStatsCSV(List<EdgeServer> servers, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_server_stats_%s.csv", outputDir, scenarioName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write header
            writer.write("ServerID,Name,ProcessingPower,MaxBandwidth,MaxConnections,CurrentLoad,EnergyConsumed,CompletedTasks,FailedTasks");
            writer.newLine();
            
            // Write data for each server
            for (EdgeServer server : servers) {
                writer.write(String.format("%d,%s,%.2f,%.2f,%d,%.2f,%.2f,%d,%d",
                        server.getId(),
                        server.getName(),
                        server.getProcessingPower(),
                        server.getMaxBandwidth(),
                        server.getMaxConnections(),
                        server.getCurrentLoad(),
                        server.getTotalEnergyConsumed(),
                        server.getCompletedTasks(),
                        server.getFailedTasks()));
                writer.newLine();
            }
            
            logger.info("Generated server stats CSV file: {}", filename);
        } catch (IOException e) {
            logger.error("Failed to generate server stats CSV file: {}", e.getMessage());
        }
    }
    
    /**
     * Generate a summary report of the simulation results.
     * 
     * @param environment MEC environment with simulation results
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateSummaryReport(MECEnvironment environment, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_summary_%s.txt", outputDir, scenarioName, timestamp);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write summary header
            writer.write("==========================================================");
            writer.newLine();
            writer.write(String.format("JCORA-MEC Simulation Summary - %s", scenarioName));
            writer.newLine();
            writer.write(String.format("Generated on: %s", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))));
            writer.newLine();
            writer.write("==========================================================");
            writer.newLine();
            writer.newLine();
            
            // Write overall metrics
            writer.write("Overall Metrics:");
            writer.newLine();
            writer.write(String.format("Total Tasks: %d", environment.getTotalTasks()));
            writer.newLine();
            writer.write(String.format("Completed Tasks: %d (%.2f%%)", 
                    environment.getCompletedTasks(), environment.getTaskCompletionRate()));
            writer.newLine();
            writer.write(String.format("Failed Tasks: %d (%.2f%%)", 
                    environment.getFailedTasks(), 
                    100.0 - environment.getTaskCompletionRate()));
            writer.newLine();
            writer.write(String.format("Total Energy Consumed: %.2f J", environment.getTotalEnergyConsumed()));
            writer.newLine();
            writer.write(String.format("Average Response Time: %.2f s", environment.getAverageResponseTime()));
            writer.newLine();
            writer.write(String.format("Deadline Miss Rate: %.2f%%", environment.getDeadlineMissRate()));
            writer.newLine();
            writer.newLine();
            
            // Write conclusion
            writer.write("Conclusion:");
            writer.newLine();
            writer.write("The JCORA-MEC system demonstrates the effectiveness of using Deep Reinforcement Learning");
            writer.newLine();
            writer.write("for joint computation offloading and resource allocation in Mobile Edge Computing environments.");
            writer.newLine();
            writer.write("The system balances energy consumption, response time, and task completion rate");
            writer.newLine();
            writer.write("to optimize the overall performance of the IoT-Edge computing system.");
            writer.newLine();
            
            logger.info("Generated summary report: {}", filename);
        } catch (IOException e) {
            logger.error("Failed to generate summary report: {}", e.getMessage());
        }
    }
}
