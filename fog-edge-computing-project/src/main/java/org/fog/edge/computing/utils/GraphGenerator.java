package org.fog.edge.computing.utils;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

/**
 * GraphGenerator class for the Fog and Edge Computing project.
 * This class is responsible for generating graphs from the simulation results
 * stored in CSV files. It uses JFreeChart library to create various types of
 * charts including bar charts, line charts, and pie charts.
 * 
 * @author Student
 * @version 1.0
 */
public class GraphGenerator {
    // Output folder path
    private String resultsFolder;
    private String graphsFolder;
    
    // Chart dimensions
    private static final int CHART_WIDTH = 800;
    private static final int CHART_HEIGHT = 600;
    
    /**
     * Constructor for GraphGenerator
     * 
     * @param resultsFolder Path to the folder containing CSV result files
     */
    public GraphGenerator(String resultsFolder) {
        this.resultsFolder = resultsFolder;
        this.graphsFolder = resultsFolder + "/graphs";
        
        // Create graphs directory if it doesn't exist
        File graphsDir = new File(graphsFolder);
        if (!graphsDir.exists()) {
            graphsDir.mkdirs();
        }
    }
    
    /**
     * Generates all graphs from the simulation results
     * 
     * @throws IOException if there's an error reading CSV files or writing graph images
     */
    public void generateAllGraphs() throws IOException {
        System.out.println("Generating graphs from simulation results...");
        
        // Generate task execution graphs
        generateTaskExecutionGraphs();
        
        // Generate energy consumption graphs
        generateEnergyConsumptionGraphs();
        
        // Generate resource utilization graphs
        generateResourceUtilizationGraphs();
        
        // Generate network usage graphs
        generateNetworkUsageGraphs();
        
        // Generate performance metrics graphs
        generatePerformanceMetricsGraphs();
        
        System.out.println("Graph generation completed. Graphs saved to: " + graphsFolder);
    }
    
    /**
     * Generates graphs related to task execution
     * 
     * @throws IOException if there's an error reading CSV files or writing graph images
     */
    private void generateTaskExecutionGraphs() throws IOException {
        File taskResultsFile = new File(resultsFolder + "/task_execution_summary.csv");
        if (!taskResultsFile.exists()) {
            System.out.println("Warning: Task execution summary file not found.");
            return;
        }
        
        // Read task execution data
        List<Map<String, String>> taskData = readCsvFile(taskResultsFile);
        
        // Generate task success rate pie chart
        generateTaskSuccessRatePieChart(taskData);
        
        // Generate offloading distribution pie chart
        generateOffloadingDistributionPieChart(taskData);
        
        // Generate execution time by offloading type bar chart
        generateExecutionTimeByOffloadingTypeBarChart(taskData);
    }
    
    /**
     * Generates a pie chart showing task success rate
     * 
     * @param taskData List of task data records
     * @throws IOException if there's an error writing the chart image
     */
    private void generateTaskSuccessRatePieChart(List<Map<String, String>> taskData) throws IOException {
        DefaultPieDataset<String> dataset = new DefaultPieDataset<>();
        
        int successCount = 0;
        int failureCount = 0;
        
        for (Map<String, String> record : taskData) {
            if (Boolean.parseBoolean(record.get("Success"))) {
                successCount++;
            } else {
                failureCount++;
            }
        }
        
        dataset.setValue("Success", successCount);
        dataset.setValue("Failure", failureCount);
        
        JFreeChart chart = ChartFactory.createPieChart(
                "Task Success Rate",
                dataset,
                true,
                true,
                false);
        
        PiePlot<String> plot = (PiePlot<String>) chart.getPlot();
        plot.setSectionPaint("Success", new Color(0, 153, 51));
        plot.setSectionPaint("Failure", new Color(204, 0, 0));
        plot.setLabelFont(new Font("SansSerif", Font.PLAIN, 12));
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/task_success_rate.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Generates a pie chart showing offloading distribution
     * 
     * @param taskData List of task data records
     * @throws IOException if there's an error writing the chart image
     */
    private void generateOffloadingDistributionPieChart(List<Map<String, String>> taskData) throws IOException {
        DefaultPieDataset<String> dataset = new DefaultPieDataset<>();
        
        Map<String, Integer> offloadingCounts = new HashMap<>();
        
        for (Map<String, String> record : taskData) {
            String offloadingType = record.get("OffloadingType");
            offloadingCounts.put(offloadingType, offloadingCounts.getOrDefault(offloadingType, 0) + 1);
        }
        
        for (Map.Entry<String, Integer> entry : offloadingCounts.entrySet()) {
            dataset.setValue(entry.getKey(), entry.getValue());
        }
        
        JFreeChart chart = ChartFactory.createPieChart(
                "Task Offloading Distribution",
                dataset,
                true,
                true,
                false);
        
        PiePlot<String> plot = (PiePlot<String>) chart.getPlot();
        plot.setLabelFont(new Font("SansSerif", Font.PLAIN, 12));
        
        // Set colors for different offloading types
        if (offloadingCounts.containsKey("Cloud")) {
            plot.setSectionPaint("Cloud", new Color(51, 102, 255));
        }
        if (offloadingCounts.containsKey("Fog")) {
            plot.setSectionPaint("Fog", new Color(255, 153, 51));
        }
        if (offloadingCounts.containsKey("Mist")) {
            plot.setSectionPaint("Mist", new Color(0, 204, 102));
        }
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/offloading_distribution.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Generates a bar chart showing execution time by offloading type
     * 
     * @param taskData List of task data records
     * @throws IOException if there's an error writing the chart image
     */
    private void generateExecutionTimeByOffloadingTypeBarChart(List<Map<String, String>> taskData) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        Map<String, List<Double>> executionTimesByType = new HashMap<>();
        
        for (Map<String, String> record : taskData) {
            String offloadingType = record.get("OffloadingType");
            double executionTime = Double.parseDouble(record.get("ExecutionTime"));
            
            if (!executionTimesByType.containsKey(offloadingType)) {
                executionTimesByType.put(offloadingType, new ArrayList<>());
            }
            executionTimesByType.get(offloadingType).add(executionTime);
        }
        
        for (Map.Entry<String, List<Double>> entry : executionTimesByType.entrySet()) {
            double average = entry.getValue().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            dataset.addValue(average, "Average Execution Time", entry.getKey());
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
                "Average Execution Time by Offloading Type",
                "Offloading Type",
                "Execution Time (ms)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false);
        
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);
        
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(79, 129, 189));
        renderer.setDrawBarOutline(false);
        
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryMargin(0.2);
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/execution_time_by_offloading_type.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Generates graphs related to energy consumption
     * 
     * @throws IOException if there's an error reading CSV files or writing graph images
     */
    private void generateEnergyConsumptionGraphs() throws IOException {
        File energyFile = new File(resultsFolder + "/energy_consumption.csv");
        if (!energyFile.exists()) {
            System.out.println("Warning: Energy consumption file not found.");
            return;
        }
        
        // Read energy consumption data
        List<Map<String, String>> energyData = readCsvFile(energyFile);
        
        // Generate energy consumption by device type bar chart
        generateEnergyConsumptionBarChart(energyData);
    }
    
    /**
     * Generates a bar chart showing energy consumption by device
     * 
     * @param energyData List of energy consumption records
     * @throws IOException if there's an error writing the chart image
     */
    private void generateEnergyConsumptionBarChart(List<Map<String, String>> energyData) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (Map<String, String> record : energyData) {
            String deviceId = record.get("DeviceID");
            double energyConsumed = Double.parseDouble(record.get("EnergyConsumed"));
            
            // Extract device type from device ID (assuming format like "Cloud_1", "Fog_2", etc.)
            String deviceType = deviceId.split("_")[0];
            
            dataset.addValue(energyConsumed, deviceType, deviceId);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
                "Energy Consumption by Device",
                "Device ID",
                "Energy Consumed (Wh)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false);
        
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);
        
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setDrawBarOutline(false);
        
        // Set colors for different device types
        renderer.setSeriesPaint(0, new Color(51, 102, 255));  // Cloud
        renderer.setSeriesPaint(1, new Color(255, 153, 51));  // Fog
        renderer.setSeriesPaint(2, new Color(0, 204, 102));   // Mist
        
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryMargin(0.2);
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/energy_consumption.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Generates graphs related to resource utilization
     * 
     * @throws IOException if there's an error reading CSV files or writing graph images
     */
    private void generateResourceUtilizationGraphs() throws IOException {
        File utilizationFile = new File(resultsFolder + "/resource_utilization.csv");
        if (!utilizationFile.exists()) {
            System.out.println("Warning: Resource utilization file not found.");
            return;
        }
        
        // Read resource utilization data
        List<Map<String, String>> utilizationData = readCsvFile(utilizationFile);
        
        // Generate resource utilization bar chart
        generateResourceUtilizationBarChart(utilizationData);
    }
    
    /**
     * Generates a bar chart showing resource utilization by device
     * 
     * @param utilizationData List of resource utilization records
     * @throws IOException if there's an error writing the chart image
     */
    private void generateResourceUtilizationBarChart(List<Map<String, String>> utilizationData) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (Map<String, String> record : utilizationData) {
            String deviceId = record.get("DeviceID");
            double utilization = Double.parseDouble(record.get("UtilizationPercentage")) * 100;  // Convert to percentage
            
            // Extract device type from device ID (assuming format like "Cloud_1", "Fog_2", etc.)
            String deviceType = deviceId.split("_")[0];
            
            dataset.addValue(utilization, deviceType, deviceId);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
                "Resource Utilization by Device",
                "Device ID",
                "Utilization (%)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false);
        
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);
        
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setDrawBarOutline(false);
        
        // Set colors for different device types
        renderer.setSeriesPaint(0, new Color(51, 102, 255));  // Cloud
        renderer.setSeriesPaint(1, new Color(255, 153, 51));  // Fog
        renderer.setSeriesPaint(2, new Color(0, 204, 102));   // Mist
        
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryMargin(0.2);
        
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setRange(0, 100);
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/resource_utilization.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Generates graphs related to network usage
     * 
     * @throws IOException if there's an error reading CSV files or writing graph images
     */
    private void generateNetworkUsageGraphs() throws IOException {
        File networkFile = new File(resultsFolder + "/network_usage.csv");
        if (!networkFile.exists()) {
            System.out.println("Warning: Network usage file not found.");
            return;
        }
        
        // Read network usage data
        List<Map<String, String>> networkData = readCsvFile(networkFile);
        
        // Generate network usage bar chart
        generateNetworkUsageBarChart(networkData);
    }
    
    /**
     * Generates a bar chart showing network usage by network link
     * 
     * @param networkData List of network usage records
     * @throws IOException if there's an error writing the chart image
     */
    private void generateNetworkUsageBarChart(List<Map<String, String>> networkData) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (Map<String, String> record : networkData) {
            String networkId = record.get("NetworkID");
            double dataTransferred = Double.parseDouble(record.get("DataTransferred"));
            
            // Extract network type from network ID (assuming format like "Cloud-Fog", "Fog-Mist", etc.)
            String networkType = networkId.contains("-") ? networkId.split("-")[0] + "-" + networkId.split("-")[1] : networkId;
            
            dataset.addValue(dataTransferred, networkType, networkId);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
                "Network Usage by Link",
                "Network Link",
                "Data Transferred (KB)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false);
        
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);
        
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setDrawBarOutline(false);
        
        // Set colors for different network types
        renderer.setSeriesPaint(0, new Color(51, 102, 255));  // Cloud-Fog
        renderer.setSeriesPaint(1, new Color(255, 153, 51));  // Fog-Mist
        
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryMargin(0.2);
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/network_usage.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Generates graphs related to performance metrics
     * 
     * @throws IOException if there's an error reading CSV files or writing graph images
     */
    private void generatePerformanceMetricsGraphs() throws IOException {
        File metricsFile = new File(resultsFolder + "/performance_metrics.csv");
        if (!metricsFile.exists()) {
            System.out.println("Warning: Performance metrics file not found.");
            return;
        }
        
        // Read performance metrics data
        List<Map<String, String>> metricsData = readCsvFile(metricsFile);
        
        // Generate performance metrics bar chart
        generatePerformanceMetricsBarChart(metricsData);
    }
    
    /**
     * Generates a bar chart showing performance metrics
     * 
     * @param metricsData List of performance metrics records
     * @throws IOException if there's an error writing the chart image
     */
    private void generatePerformanceMetricsBarChart(List<Map<String, String>> metricsData) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (Map<String, String> record : metricsData) {
            String metric = record.get("Metric");
            double value = Double.parseDouble(record.get("Value"));
            
            dataset.addValue(value, "Value", metric);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
                "Performance Metrics",
                "Metric",
                "Value",
                dataset,
                PlotOrientation.VERTICAL,
                false,
                true,
                false);
        
        CategoryPlot plot = chart.getCategoryPlot();
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.GRAY);
        
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(79, 129, 189));
        renderer.setDrawBarOutline(false);
        
        CategoryAxis domainAxis = plot.getDomainAxis();
        domainAxis.setCategoryMargin(0.2);
        
        ChartUtils.saveChartAsPNG(new File(graphsFolder + "/performance_metrics.png"), chart, CHART_WIDTH, CHART_HEIGHT);
    }
    
    /**
     * Reads a CSV file and returns its contents as a list of maps
     * 
     * @param file CSV file to read
     * @return List of maps, where each map represents a row with column name as key
     * @throws IOException if there's an error reading the file
     */
    private List<Map<String, String>> readCsvFile(File file) throws IOException {
        List<Map<String, String>> data = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line = reader.readLine();
            if (line == null) {
                return data;
            }
            
            // Parse header
            String[] headers = line.split(",");
            
            // Parse data rows
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                Map<String, String> row = new HashMap<>();
                
                for (int i = 0; i < headers.length && i < values.length; i++) {
                    row.put(headers[i], values[i]);
                }
                
                data.add(row);
            }
        }
        
        return data;
    }
}
