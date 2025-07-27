package org.nci.fogedge.model;

import java.util.logging.Logger;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.util.LoggingUtil;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * Collects and analyzes simulation results from the fog computing architecture.
 * Provides methods for aggregating metrics and generating reports.
 */
public class SimulationResults {
    
    private boolean securityEnabled;
    private int totalIoTDevices;
    private int totalEdgeNodes;
    private int totalFogNodes;
    private double totalDataGenerated;    // KB
    private double totalDataProcessed;    // KB
    private double totalEnergyConsumption; // Joules
    private double totalProcessingTime;   // ms
    private double totalSecurityOverhead; // ms
    private int detectedAttacks;
    
    private Map<String, Double> deviceEnergyConsumption;
    private Map<String, Double> edgeProcessingTimes;
    private Map<String, Double> fogProcessingTimes;
    private Map<IoTDevice.WirelessType, Integer> wirelessTypeDistribution;
    
    private String resultsLogFile;
    
    /**
     * Creates a new SimulationResults instance
     * 
     * @param securityEnabled Whether security was enabled in the simulation
     */
    public SimulationResults(boolean securityEnabled) {
        this.securityEnabled = securityEnabled;
        this.totalIoTDevices = 0;
        this.totalEdgeNodes = 0;
        this.totalFogNodes = 0;
        this.totalDataGenerated = 0.0;
        this.totalDataProcessed = 0.0;
        this.totalEnergyConsumption = 0.0;
        this.totalProcessingTime = 0.0;
        this.totalSecurityOverhead = 0.0;
        this.detectedAttacks = 0;
        
        this.deviceEnergyConsumption = new HashMap<>();
        this.edgeProcessingTimes = new HashMap<>();
        this.fogProcessingTimes = new HashMap<>();
        this.wirelessTypeDistribution = new HashMap<>();
        
        // Initialize wireless type distribution
        for (IoTDevice.WirelessType type : IoTDevice.WirelessType.values()) {
            wirelessTypeDistribution.put(type, 0);
        }
        
        // Create results log file
        this.resultsLogFile = LoggingUtil.createSimulationResultsLog("SecureFogSim");
    }
    
    /**
     * Collects metrics from IoT devices
     * 
     * @param devices List of IoT devices
     */
    public void collectIoTMetrics(List<IoTDevice> devices) {
        this.totalIoTDevices = devices.size();
        
        for (IoTDevice device : devices) {
            this.totalDataGenerated += device.getDataGenerationRate();
            this.totalEnergyConsumption += device.getEnergyConsumption();
            this.totalSecurityOverhead += device.getSecurityOverhead();
            
            deviceEnergyConsumption.put(device.getDeviceId(), device.getEnergyConsumption());
            
            // Count wireless type distribution
            IoTDevice.WirelessType type = device.getWirelessType();
            wirelessTypeDistribution.put(type, wirelessTypeDistribution.get(type) + 1);
        }
        
        Log.printLine("Collected metrics from " + devices.size() + " IoT devices");
    }
    
    /**
     * Collects metrics from edge nodes
     * 
     * @param edgeNodes List of edge nodes
     */
    public void collectEdgeMetrics(List<EdgeNode> edgeNodes) {
        this.totalEdgeNodes = edgeNodes.size();
        
        for (EdgeNode edge : edgeNodes) {
            this.totalEnergyConsumption += edge.getEnergyConsumption();
            this.totalProcessingTime += edge.getProcessingTime();
            this.totalSecurityOverhead += edge.getSecurityOverhead();
            
            edgeProcessingTimes.put(edge.getNodeId(), edge.getProcessingTime());
        }
        
        Log.printLine("Collected metrics from " + edgeNodes.size() + " edge nodes");
    }
    
    /**
     * Collects metrics from fog nodes
     * 
     * @param fogNodes List of fog nodes
     */
    public void collectFogMetrics(List<FogNode> fogNodes) {
        this.totalFogNodes = fogNodes.size();
        
        for (FogNode fog : fogNodes) {
            this.totalDataProcessed += fog.calculateDataVolume();
            this.totalEnergyConsumption += fog.getEnergyConsumption();
            this.totalProcessingTime += fog.getProcessingTime();
            this.totalSecurityOverhead += fog.getSecurityOverhead();
            
            fogProcessingTimes.put(fog.getNodeId(), fog.getProcessingTime());
        }
        
        Log.printLine("Collected metrics from " + fogNodes.size() + " fog nodes");
    }
    
    /**
     * Sets the number of detected attacks
     * 
     * @param detectedAttacks Number of detected attacks
     */
    public void setDetectedAttacks(int detectedAttacks) {
        this.detectedAttacks = detectedAttacks;
    }
    
    /**
     * Logs the simulation results to the console and results file
     */
    public void logResults() {
        Log.printLine("\n============ SIMULATION RESULTS ============");
        Log.printLine("Security Enabled: " + securityEnabled);
        Log.printLine("Total IoT Devices: " + totalIoTDevices);
        Log.printLine("Total Edge Nodes: " + totalEdgeNodes);
        Log.printLine("Total Fog Nodes: " + totalFogNodes);
        Log.printLine("Total Data Generated: " + String.format("%.2f", totalDataGenerated) + " KB");
        Log.printLine("Total Data Processed: " + String.format("%.2f", totalDataProcessed) + " KB");
        Log.printLine("Total Energy Consumption: " + String.format("%.2f", totalEnergyConsumption) + " J");
        Log.printLine("Total Processing Time: " + String.format("%.2f", totalProcessingTime) + " ms");
        Log.printLine("Total Security Overhead: " + String.format("%.2f", totalSecurityOverhead) + " ms");
        Log.printLine("Security Overhead Percentage: " + 
                String.format("%.2f", (totalSecurityOverhead / totalProcessingTime) * 100) + "%");
        Log.printLine("Detected Attacks: " + detectedAttacks);
        Log.printLine("==========================================\n");
        
        // Log to results file
        if (resultsLogFile != null) {
            LoggingUtil.appendSimulationResult(
                    resultsLogFile,
                    securityEnabled,
                    totalIoTDevices,
                    totalEdgeNodes,
                    totalFogNodes,
                    totalDataGenerated,
                    totalDataProcessed,
                    totalEnergyConsumption,
                    totalProcessingTime,
                    totalSecurityOverhead,
                    detectedAttacks
            );
        }
    }
    
    /**
     * Analyzes the energy efficiency of the system
     * 
     * @return Energy efficiency in KB processed per Joule
     */
    public double analyzeEnergyEfficiency() {
        if (totalEnergyConsumption == 0) {
            return 0.0;
        }
        
        double efficiency = totalDataProcessed / totalEnergyConsumption;
        Log.printLine("Energy Efficiency: " + String.format("%.2f", efficiency) + " KB/J");
        return efficiency;
    }
    
    /**
     * Analyzes the security overhead impact
     * 
     * @return Security overhead percentage
     */
    public double analyzeSecurityOverhead() {
        if (totalProcessingTime == 0) {
            return 0.0;
        }
        
        double overheadPercentage = (totalSecurityOverhead / totalProcessingTime) * 100;
        Log.printLine("Security Overhead: " + String.format("%.2f", overheadPercentage) + "%");
        return overheadPercentage;
    }
    
    /**
     * Analyzes the data reduction from IoT to fog
     * 
     * @return Data reduction percentage
     */
    public double analyzeDataReduction() {
        if (totalDataGenerated == 0) {
            return 0.0;
        }
        
        double reductionPercentage = ((totalDataGenerated - totalDataProcessed) / totalDataGenerated) * 100;
        Log.printLine("Data Reduction: " + String.format("%.2f", reductionPercentage) + "%");
        return reductionPercentage;
    }
    
    /**
     * Generates a detailed report of the simulation results
     * 
     * @return Detailed report as a string
     */
    public String generateDetailedReport() {
        StringBuilder report = new StringBuilder();
        
        report.append("# Secure Fog Computing Simulation Report\n\n");
        report.append("## Configuration\n");
        report.append("- Security Enabled: ").append(securityEnabled).append("\n");
        report.append("- Total IoT Devices: ").append(totalIoTDevices).append("\n");
        report.append("- Total Edge Nodes: ").append(totalEdgeNodes).append("\n");
        report.append("- Total Fog Nodes: ").append(totalFogNodes).append("\n");
        
        // Add wireless technology distribution
        report.append("- Wireless Technology Distribution:\n");
        for (Map.Entry<IoTDevice.WirelessType, Integer> entry : wirelessTypeDistribution.entrySet()) {
            report.append("  - ").append(entry.getKey()).append(": ")
                  .append(entry.getValue()).append(" devices\n");
        }
        report.append("\n");
        
        report.append("## Performance Metrics\n");
        report.append("- Total Data Generated: ").append(String.format("%.2f", totalDataGenerated)).append(" KB\n");
        report.append("- Total Data Processed: ").append(String.format("%.2f", totalDataProcessed)).append(" KB\n");
        report.append("- Total Energy Consumption: ").append(String.format("%.2f", totalEnergyConsumption)).append(" J\n");
        report.append("- Total Processing Time: ").append(String.format("%.2f", totalProcessingTime)).append(" ms\n");
        report.append("- Total Security Overhead: ").append(String.format("%.2f", totalSecurityOverhead)).append(" ms\n");
        report.append("- Security Overhead Percentage: ").append(
                String.format("%.2f", (totalSecurityOverhead / totalProcessingTime) * 100)).append("%\n");
        report.append("- Detected Attacks: ").append(detectedAttacks).append("\n\n");
        
        report.append("## Analysis\n");
        report.append("- Energy Efficiency: ").append(String.format("%.2f", analyzeEnergyEfficiency())).append(" KB/J\n");
        report.append("- Security Overhead: ").append(String.format("%.2f", analyzeSecurityOverhead())).append("%\n");
        report.append("- Data Reduction: ").append(String.format("%.2f", analyzeDataReduction())).append("%\n\n");
        
        report.append("## Conclusion\n");
        if (securityEnabled) {
            report.append("The simulation with security enabled demonstrates the trade-off between ");
            report.append("security and performance. The security overhead of ");
            report.append(String.format("%.2f", analyzeSecurityOverhead())).append("% ");
            report.append("shows the cost of implementing encryption and authentication in the fog architecture. ");
            report.append("However, this overhead is justified by the protection against potential attacks ");
            report.append("and the secure processing of sensitive data.\n\n");
        } else {
            report.append("The simulation without security features shows better performance metrics ");
            report.append("but lacks protection against potential attacks. In real-world deployments, ");
            report.append("this configuration would be vulnerable to various security threats.\n\n");
        }
        
        report.append("The data reduction of ").append(String.format("%.2f", analyzeDataReduction())).append("% ");
        report.append("demonstrates the effectiveness of edge processing in reducing the data volume ");
        report.append("that needs to be transmitted to higher layers, which is a key benefit of fog computing.\n");
        
        return report.toString();
    }
    
    // Getters
    public boolean isSecurityEnabled() {
        return securityEnabled;
    }
    
    public int getTotalIoTDevices() {
        return totalIoTDevices;
    }
    
    public int getTotalEdgeNodes() {
        return totalEdgeNodes;
    }
    
    public int getTotalFogNodes() {
        return totalFogNodes;
    }
    
    public double getTotalDataGenerated() {
        return totalDataGenerated;
    }
    
    public double getTotalDataProcessed() {
        return totalDataProcessed;
    }
    
    public double getTotalEnergyConsumption() {
        return totalEnergyConsumption;
    }
    
    public double getTotalProcessingTime() {
        return totalProcessingTime;
    }
    
    public double getTotalSecurityOverhead() {
        return totalSecurityOverhead;
    }
    
    public int getDetectedAttacks() {
        return detectedAttacks;
    }
    
    public String getResultsLogFile() {
        return resultsLogFile;
    }
}
