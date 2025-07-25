package org.nci.fogedge.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Map;

import org.nci.fogedge.model.SimulationResults;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.topology.NetworkTopology;

/**
 * Report Generator for the fog computing simulation
 * 
 * This class generates various reports based on simulation results,
 * including performance metrics, security incidents, and network statistics.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class ReportGenerator {
    
    private SimulationResults results;
    private NetworkTopology topology;
    private SecurityManager securityManager;
    private String outputDirectory;
    
    /**
     * Constructor with parameters
     * @param results Simulation results
     * @param topology Network topology
     * @param securityManager Security manager
     */
    public ReportGenerator(SimulationResults results, NetworkTopology topology, SecurityManager securityManager) {
        this.results = results;
        this.topology = topology;
        this.securityManager = securityManager;
        this.outputDirectory = "reports";
        
        // Create output directory if it doesn't exist
        File dir = new File(outputDirectory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    /**
     * Generate all reports
     */
    public void generateAllReports() {
        generatePerformanceReport();
        generateSecurityReport("security_report.txt");
        generateNetworkReport();
        generateSummaryReport();
        generateEnergyReport("energy_report.txt");
        generateLatencyReport("latency_report.txt");
    }
    
    /**
     * Generate performance report
     * @return Path to generated report
     */
    public String generatePerformanceReport() {
        String filename = outputDirectory + "/performance_report_" + getTimestamp() + ".txt";
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("=======================================================");
            writer.println("             PERFORMANCE REPORT                        ");
            writer.println("=======================================================");
            writer.println();
            
            writer.println("PACKET STATISTICS:");
            writer.println("------------------");
            writer.println("Total packets generated: " + results.getTotalPacketsGenerated());
            writer.println("Packets processed at edge: " + results.getPacketsProcessedAtEdge());
            writer.println("Packets processed at fog: " + results.getPacketsProcessedAtFog());
            writer.println("Packets processed at cloud: " + results.getPacketsProcessedAtCloud());
            writer.println("Packets processed locally (not forwarded): " + results.getPacketsProcessedLocally());
            writer.println();
            
            writer.println("LATENCY STATISTICS:");
            writer.println("------------------");
            writer.println("Average end-to-end latency: " + formatDouble(results.getAverageEndToEndLatency()) + " ms");
            writer.println("Average processing time at edge: " + formatDouble(results.getAverageEdgeProcessingTime()) + " ms");
            writer.println("Average processing time at fog: " + formatDouble(results.getAverageFogProcessingTime()) + " ms");
            writer.println("Average processing time at cloud: " + formatDouble(results.getAverageCloudProcessingTime()) + " ms");
            writer.println();
            
            writer.println("BANDWIDTH STATISTICS:");
            writer.println("--------------------");
            writer.println("Bandwidth saved by edge processing: " + formatDouble(results.getBandwidthSavedByEdgeProcessing()) + " MB");
            writer.println("Bandwidth saved by fog processing: " + formatDouble(results.getBandwidthSavedByFogProcessing()) + " MB");
            writer.println("Total bandwidth saved: " + formatDouble(results.getTotalBandwidthSaved()) + " MB");
            writer.println();
            
            writer.println("ENERGY STATISTICS:");
            writer.println("-----------------");
            writer.println("Energy consumed by IoT devices: " + formatDouble(results.getEnergyConsumedByIoT()) + " J");
            writer.println("Energy consumed by edge nodes: " + formatDouble(results.getEnergyConsumedByEdgeNodes()) + " J");
            writer.println("Energy consumed by fog nodes: " + formatDouble(results.getEnergyConsumedByFogNodes()) + " J");
            writer.println("Energy consumed by cloud: " + formatDouble(results.getEnergyConsumedByCloud()) + " J");
            writer.println("Total energy consumed: " + formatDouble(results.getTotalEnergyConsumed()) + " J");
            writer.println();
            
            writer.println("SECURITY OVERHEAD:");
            writer.println("-----------------");
            writer.println("Average security overhead at IoT: " + formatDouble(results.getAverageSecurityOverheadIoT()) + " ms");
            writer.println("Average security overhead at edge: " + formatDouble(results.getAverageSecurityOverheadEdge()) + " ms");
            writer.println("Average security overhead at fog: " + formatDouble(results.getAverageSecurityOverheadFog()) + " ms");
            writer.println("Total security overhead: " + formatDouble(results.getTotalSecurityOverhead()) + " ms");
            
            System.out.println("Performance report generated: " + filename);
            return filename;
        } catch (IOException e) {
            System.err.println("Error generating performance report: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate security report
     * @param filename Output filename
     * @return Path to generated report
     */
    public String generateSecurityReport(String filename) {
        String outputPath = outputDirectory + "/" + filename.replace(".pdf", "_" + getTimestamp() + ".txt");
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("=======================================================");
            writer.println("             SECURITY REPORT                           ");
            writer.println("=======================================================");
            writer.println();
            
            writer.println("SECURITY INCIDENT STATISTICS:");
            writer.println("----------------------------");
            writer.println("Total security incidents detected: " + results.getSecurityIncidentsDetected());
            writer.println("Security incidents mitigated: " + results.getSecurityIncidentsMitigated());
            writer.println("Security incidents unmitigated: " + results.getSecurityIncidentsUnmitigated());
            writer.println("Mitigation success rate: " + formatDouble(results.getMitigationSuccessRate() * 100) + "%");
            writer.println();
            
            writer.println("SECURITY INCIDENTS BY TYPE:");
            writer.println("---------------------------");
            Map<String, Integer> incidentsByType = securityManager.getIncidentsByType();
            for (Map.Entry<String, Integer> entry : incidentsByType.entrySet()) {
                writer.println(entry.getKey() + ": " + entry.getValue());
            }
            writer.println();
            
            writer.println("SECURITY COUNTERMEASURES EFFECTIVENESS:");
            writer.println("--------------------------------------");
            writer.println("Encryption effectiveness: " + formatDouble(results.getEncryptionEffectiveness() * 100) + "%");
            writer.println("Intrusion detection effectiveness: " + formatDouble(results.getIntrusionDetectionEffectiveness() * 100) + "%");
            writer.println("Blockchain effectiveness: " + formatDouble(results.getBlockchainEffectiveness() * 100) + "%");
            writer.println("Decoy technique effectiveness: " + formatDouble(results.getDecoyTechniqueEffectiveness() * 100) + "%");
            writer.println();
            
            writer.println("SECURITY RESPONSE TIMES:");
            writer.println("-----------------------");
            writer.println("Average incident detection time: " + formatDouble(results.getAverageIncidentDetectionTime()) + " ms");
            writer.println("Average incident mitigation time: " + formatDouble(results.getAverageIncidentMitigationTime()) + " ms");
            writer.println("Average security response time: " + formatDouble(results.getAverageSecurityResponseTime()) + " ms");
            
            System.out.println("Security report generated: " + outputPath);
            return outputPath;
        } catch (IOException e) {
            System.err.println("Error generating security report: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate network report
     * @return Path to generated report
     */
    public String generateNetworkReport() {
        String filename = outputDirectory + "/network_report_" + getTimestamp() + ".txt";
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("=======================================================");
            writer.println("             NETWORK REPORT                            ");
            writer.println("=======================================================");
            writer.println();
            
            writer.println("NETWORK TOPOLOGY:");
            writer.println("----------------");
            writer.println("IoT devices: " + topology.getIotDevices().size());
            writer.println("Edge nodes: " + topology.getEdgeNodes().size());
            writer.println("Fog nodes: " + topology.getFogNodes().size());
            writer.println("Cloud datacenter: " + (topology.getCloudDatacenter() != null ? "1" : "0"));
            writer.println();
            
            writer.println("NETWORK STATISTICS:");
            writer.println("------------------");
            writer.println("Average IoT devices per edge node: " + formatDouble(calculateAverageIoTDevicesPerEdgeNode()));
            writer.println("Average edge nodes per fog node: " + formatDouble(calculateAverageEdgeNodesPerFogNode()));
            writer.println();
            
            writer.println("PROCESSING DISTRIBUTION:");
            writer.println("----------------------");
            writer.println("Edge processing ratio: " + formatDouble(calculateEdgeProcessingRatio() * 100) + "%");
            writer.println("Fog processing ratio: " + formatDouble(calculateFogProcessingRatio() * 100) + "%");
            writer.println("Cloud processing ratio: " + formatDouble(calculateCloudProcessingRatio() * 100) + "%");
            writer.println("Local processing ratio: " + formatDouble(calculateLocalProcessingRatio() * 100) + "%");
            writer.println();
            
            writer.println("EFFICIENCY METRICS:");
            writer.println("-----------------");
            writer.println("Edge efficiency: " + formatDouble(calculateEdgeEfficiency() * 100) + "%");
            writer.println("Fog efficiency: " + formatDouble(calculateFogEfficiency() * 100) + "%");
            writer.println("Cloud efficiency: " + formatDouble(calculateCloudEfficiency() * 100) + "%");
            
            System.out.println("Network report generated: " + filename);
            return filename;
        } catch (IOException e) {
            System.err.println("Error generating network report: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate summary report
     * @return Path to generated report
     */
    public String generateSummaryReport() {
        String filename = outputDirectory + "/summary_report_" + getTimestamp() + ".txt";
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("=======================================================");
            writer.println("             SUMMARY REPORT                            ");
            writer.println("=======================================================");
            writer.println();
            
            writer.println("SIMULATION OVERVIEW:");
            writer.println("-------------------");
            writer.println("Total IoT devices: " + topology.getIotDevices().size());
            writer.println("Total edge nodes: " + topology.getEdgeNodes().size());
            writer.println("Total fog nodes: " + topology.getFogNodes().size());
            writer.println("Cloud datacenter: " + (topology.getCloudDatacenter() != null ? "Present" : "None"));
            writer.println();
            
            writer.println("PERFORMANCE SUMMARY:");
            writer.println("-------------------");
            writer.println("Total packets processed: " + results.getTotalPacketsGenerated());
            writer.println("Average end-to-end latency: " + formatDouble(results.getAverageEndToEndLatency()) + " ms");
            writer.println("Bandwidth saved: " + formatDouble(results.getBandwidthSaved()) + " MB");
            writer.println("Energy consumption: " + formatDouble(results.getEnergyConsumption()) + " kWh");
            writer.println();
            
            writer.println("SECURITY SUMMARY:");
            writer.println("----------------");
            writer.println("Security incidents detected: " + results.getSecurityIncidentsDetected());
            writer.println("Security incidents mitigated: " + results.getSecurityIncidentsMitigated());
            writer.println("Mitigation success rate: " + formatDouble(results.getMitigationSuccessRate() * 100) + "%");
            writer.println("Average security response time: " + formatDouble(results.getAverageSecurityResponseTime()) + " ms");
            writer.println();
            
            writer.println("EFFICIENCY SUMMARY:");
            writer.println("------------------");
            writer.println("Edge processing ratio: " + formatDouble(calculateEdgeProcessingRatio() * 100) + "%");
            writer.println("Fog processing ratio: " + formatDouble(calculateFogProcessingRatio() * 100) + "%");
            writer.println("Cloud processing ratio: " + formatDouble(calculateCloudProcessingRatio() * 100) + "%");
            
            System.out.println("Summary report generated: " + filename);
            return filename;
        } catch (IOException e) {
            System.err.println("Error generating summary report: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Calculate average IoT devices per edge node
     * @return Average IoT devices per edge node
     */
    private double calculateAverageIoTDevicesPerEdgeNode() {
        if (topology.getEdgeNodes().size() == 0) {
            return 0;
        }
        return (double) topology.getIotDevices().size() / topology.getEdgeNodes().size();
    }
    
    /**
     * Calculate average edge nodes per fog node
     * @return Average edge nodes per fog node
     */
    private double calculateAverageEdgeNodesPerFogNode() {
        if (topology.getFogNodes().size() == 0) {
            return 0;
        }
        return (double) topology.getEdgeNodes().size() / topology.getFogNodes().size();
    }
    
    /**
     * Calculate edge processing ratio
     * @return Edge processing ratio
     */
    private double calculateEdgeProcessingRatio() {
        if (results.getTotalPacketsGenerated() == 0) {
            return 0;
        }
        return (double) results.getPacketsProcessedAtEdge() / results.getTotalPacketsGenerated();
    }
    
    /**
     * Calculate fog processing ratio
     * @return Fog processing ratio
     */
    private double calculateFogProcessingRatio() {
        if (results.getTotalPacketsGenerated() == 0) {
            return 0;
        }
        return (double) results.getPacketsProcessedAtFog() / results.getTotalPacketsGenerated();
    }
    
    /**
     * Calculate cloud processing ratio
     * @return Cloud processing ratio
     */
    private double calculateCloudProcessingRatio() {
        if (results.getTotalPacketsGenerated() == 0) {
            return 0;
        }
        return (double) results.getPacketsProcessedAtCloud() / results.getTotalPacketsGenerated();
    }
    
    /**
     * Calculate local processing ratio
     * @return Local processing ratio
     */
    private double calculateLocalProcessingRatio() {
        if (results.getTotalPacketsGenerated() == 0) {
            return 0;
        }
        return (double) results.getPacketsProcessedLocally() / results.getTotalPacketsGenerated();
    }
    
    /**
     * Calculate edge efficiency
     * @return Edge efficiency
     */
    private double calculateEdgeEfficiency() {
        if (results.getPacketsProcessedAtEdge() == 0) {
            return 0;
        }
        // Efficiency is calculated as (packets processed / energy consumed)
        return results.getPacketsProcessedAtEdge() / (results.getEnergyConsumedByEdgeNodes() + 0.001);
    }
    
    /**
     * Calculate fog efficiency
     * @return Fog efficiency
     */
    private double calculateFogEfficiency() {
        if (results.getPacketsProcessedAtFog() == 0) {
            return 0;
        }
        // Efficiency is calculated as (packets processed / energy consumed)
        return results.getPacketsProcessedAtFog() / (results.getEnergyConsumedByFogNodes() + 0.001);
    }
    
    /**
     * Calculate cloud efficiency
     * @return Cloud efficiency
     */
    private double calculateCloudEfficiency() {
        if (results.getPacketsProcessedAtCloud() == 0) {
            return 0;
        }
        // Efficiency is calculated as (packets processed / energy consumed)
        return results.getPacketsProcessedAtCloud() / (results.getEnergyConsumedByCloud() + 0.001);
    }
    
    /**
     * Format a double value to 2 decimal places
     * @param value Value to format
     * @return Formatted value
     */
    private String formatDouble(double value) {
        return String.format("%.2f", value);
    }
    
    /**
     * Get timestamp for file naming
     * @return Timestamp string
     */
    private String getTimestamp() {
        return new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
    }
    
    /**
     * Generate energy report
     * @param filename Output filename
     * @return Path to generated report
     */
    public String generateEnergyReport(String filename) {
        String outputPath = outputDirectory + "/" + filename.replace(".pdf", "_" + getTimestamp() + ".txt");
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("=======================================================");
            writer.println("             ENERGY CONSUMPTION REPORT                 ");
            writer.println("=======================================================");
            writer.println();
            
            writer.println("ENERGY CONSUMPTION BY LAYER:");
            writer.println("--------------------------");
            writer.println("IoT layer energy consumption: " + formatDouble(results.getIoTLayerEnergyConsumption()) + " J");
            writer.println("Edge layer energy consumption: " + formatDouble(results.getEdgeLayerEnergyConsumption()) + " J");
            writer.println("Fog layer energy consumption: " + formatDouble(results.getFogLayerEnergyConsumption()) + " J");
            writer.println("Cloud layer energy consumption: " + formatDouble(results.getCloudLayerEnergyConsumption()) + " J");
            writer.println("Total energy consumption: " + formatDouble(results.getTotalEnergyConsumption()) + " J");
            writer.println();
            
            writer.println("ENERGY CONSUMPTION BY COMPONENT:");
            writer.println("------------------------------");
            writer.println("Processing energy consumption: " + formatDouble(results.getProcessingEnergyConsumption()) + " J");
            writer.println("Transmission energy consumption: " + formatDouble(results.getTransmissionEnergyConsumption()) + " J");
            writer.println("Storage energy consumption: " + formatDouble(results.getStorageEnergyConsumption()) + " J");
            writer.println("Security energy consumption: " + formatDouble(results.getSecurityEnergyConsumption()) + " J");
            writer.println();
            
            writer.println("ENERGY EFFICIENCY METRICS:");
            writer.println("------------------------");
            writer.println("Energy per bit: " + formatDouble(results.getEnergyPerBit()) + " J/bit");
            writer.println("Energy per packet: " + formatDouble(results.getEnergyPerPacket()) + " J/packet");
            writer.println();
            
            System.out.println("Energy report generated: " + outputPath);
            return outputPath;
        } catch (IOException e) {
            System.err.println("Error generating energy report: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate latency report
     * @param filename Output filename
     * @return Path to generated report
     */
    public String generateLatencyReport(String filename) {
        String outputPath = outputDirectory + "/" + filename.replace(".pdf", "_" + getTimestamp() + ".txt");
        
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            writer.println("=======================================================");
            writer.println("             LATENCY ANALYSIS REPORT                  ");
            writer.println("=======================================================");
            writer.println();
            
            writer.println("END-TO-END LATENCY:");
            writer.println("------------------");
            writer.println("Average end-to-end latency: " + formatDouble(results.getAverageEndToEndLatency()) + " ms");
            writer.println("Minimum end-to-end latency: " + formatDouble(results.getMinEndToEndLatency()) + " ms");
            writer.println("Maximum end-to-end latency: " + formatDouble(results.getMaxEndToEndLatency()) + " ms");
            writer.println("Latency standard deviation: " + formatDouble(results.getLatencyStandardDeviation()) + " ms");
            writer.println();
            
            writer.println("PROCESSING LATENCY BY LAYER:");
            writer.println("---------------------------");
            writer.println("Average processing time at IoT: " + formatDouble(results.getAverageIoTProcessingTime()) + " ms");
            writer.println("Average processing time at edge: " + formatDouble(results.getAverageEdgeProcessingTime()) + " ms");
            writer.println("Average processing time at fog: " + formatDouble(results.getAverageFogProcessingTime()) + " ms");
            writer.println("Average processing time at cloud: " + formatDouble(results.getAverageCloudProcessingTime()) + " ms");
            writer.println();
            
            writer.println("TRANSMISSION LATENCY:");
            writer.println("--------------------");
            writer.println("Average IoT to edge transmission time: " + formatDouble(results.getAverageIoTToEdgeTransmissionTime()) + " ms");
            writer.println("Average edge to fog transmission time: " + formatDouble(results.getAverageEdgeToFogTransmissionTime()) + " ms");
            writer.println("Average fog to cloud transmission time: " + formatDouble(results.getAverageFogToCloudTransmissionTime()) + " ms");
            writer.println();
            
            writer.println("SECURITY OVERHEAD LATENCY:");
            writer.println("------------------------");
            writer.println("Average encryption overhead: " + formatDouble(results.getAverageEncryptionOverhead()) + " ms");
            writer.println("Average authentication overhead: " + formatDouble(results.getAverageAuthenticationOverhead()) + " ms");
            writer.println("Average intrusion detection overhead: " + formatDouble(results.getAverageIntrusionDetectionOverhead()) + " ms");
            writer.println("Average blockchain overhead: " + formatDouble(results.getAverageBlockchainOverhead()) + " ms");
            writer.println("Average decoy technique overhead: " + formatDouble(results.getAverageDecoyTechniqueOverhead()) + " ms");
            writer.println();
            
            System.out.println("Latency report generated: " + outputPath);
            return outputPath;
        } catch (IOException e) {
            System.err.println("Error generating latency report: " + e.getMessage());
            return null;
        }
    }
}
