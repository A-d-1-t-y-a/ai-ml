package org.todg.simulation.metrics;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Collects and stores metrics from the TODG simulation.
 * This class is responsible for collecting metrics during simulation,
 * storing them, and generating visualizations.
 */
public class MetricsCollector {
    private static final Logger logger = LoggerFactory.getLogger(MetricsCollector.class);
    
    // Time series data
    private List<Double> timePoints;
    private List<Integer> tasksGenerated;
    private List<Integer> tasksOffloaded;
    private List<Integer> tasksProcessedLocally;
    private List<Integer> tasksCompleted;
    private List<Integer> tasksFailed;
    private List<Double> energyConsumed;
    private List<Double> averageDelay;
    private List<Double> taskCompletionRate;
    private List<Double> serverUtilization;
    
    // Cumulative metrics
    private int totalTasksGenerated;
    private int totalTasksOffloaded;
    private int totalTasksProcessedLocally;
    private int totalTasksCompleted;
    private int totalTasksFailed;
    private double totalEnergyConsumed;
    
    // Output directory for charts and logs
    private String outputDirectory;
    
    /**
     * Constructor for the MetricsCollector.
     */
    public MetricsCollector() {
        this("output");
    }
    
    /**
     * Constructor for the MetricsCollector with custom output directory.
     * 
     * @param outputDirectory The directory where charts and logs will be saved
     */
    public MetricsCollector(String outputDirectory) {
        this.outputDirectory = outputDirectory;
        
        // Initialize time series data
        timePoints = new ArrayList<>();
        tasksGenerated = new ArrayList<>();
        tasksOffloaded = new ArrayList<>();
        tasksProcessedLocally = new ArrayList<>();
        tasksCompleted = new ArrayList<>();
        tasksFailed = new ArrayList<>();
        energyConsumed = new ArrayList<>();
        averageDelay = new ArrayList<>();
        taskCompletionRate = new ArrayList<>();
        serverUtilization = new ArrayList<>();
        
        // Initialize cumulative metrics
        totalTasksGenerated = 0;
        totalTasksOffloaded = 0;
        totalTasksProcessedLocally = 0;
        totalTasksCompleted = 0;
        totalTasksFailed = 0;
        totalEnergyConsumed = 0.0;
        
        // Create output directory if it doesn't exist
        File directory = new File(outputDirectory);
        if (!directory.exists()) {
            directory.mkdirs();
        }
    }
    
    /**
     * Collects metrics for a simulation time step.
     * 
     * @param time The current simulation time
     * @param metrics The metrics for this time step
     */
    public void collectMetrics(double time, Map<String, Object> metrics) {
        // Add time point
        timePoints.add(time);
        
        // Extract metrics
        int stepTasksGenerated = getIntValue(metrics, "tasksGenerated");
        int stepTasksOffloaded = getIntValue(metrics, "tasksOffloaded");
        int stepTasksProcessedLocally = getIntValue(metrics, "tasksProcessedLocally");
        int stepTasksCompleted = getIntValue(metrics, "tasksCompleted");
        int stepTasksFailed = getIntValue(metrics, "tasksFailed");
        double stepEnergyConsumed = getDoubleValue(metrics, "energyConsumed");
        double stepServerUtilization = getDoubleValue(metrics, "serverUtilization", 0.0);
        
        // Update cumulative metrics
        totalTasksGenerated += stepTasksGenerated;
        totalTasksOffloaded += stepTasksOffloaded;
        totalTasksProcessedLocally += stepTasksProcessedLocally;
        totalTasksCompleted += stepTasksCompleted;
        totalTasksFailed += stepTasksFailed;
        totalEnergyConsumed += stepEnergyConsumed;
        
        // Calculate derived metrics
        double stepAverageDelay = 0.0;
        if (stepTasksCompleted > 0 && metrics.containsKey("totalDelay")) {
            double totalDelay = getDoubleValue(metrics, "totalDelay");
            stepAverageDelay = totalDelay / stepTasksCompleted;
        }
        
        double stepCompletionRate = 0.0;
        if (totalTasksGenerated > 0) {
            stepCompletionRate = (totalTasksCompleted * 100.0) / totalTasksGenerated;
        }
        
        // Add metrics to time series
        tasksGenerated.add(stepTasksGenerated);
        tasksOffloaded.add(stepTasksOffloaded);
        tasksProcessedLocally.add(stepTasksProcessedLocally);
        tasksCompleted.add(stepTasksCompleted);
        tasksFailed.add(stepTasksFailed);
        energyConsumed.add(stepEnergyConsumed);
        averageDelay.add(stepAverageDelay);
        taskCompletionRate.add(stepCompletionRate);
        serverUtilization.add(stepServerUtilization);
        
        // Log metrics periodically
        if (timePoints.size() % 100 == 0) {
            logger.info("Metrics at time {}: Generated={}, Completed={}, Failed={}, Energy={}, CompletionRate={}%",
                String.format("%.2f", time),
                totalTasksGenerated,
                totalTasksCompleted,
                totalTasksFailed,
                String.format("%.2f", totalEnergyConsumed),
                String.format("%.2f", stepCompletionRate));
        }
    }
    
    /**
     * Safely extracts an integer value from a metrics map.
     * 
     * @param metrics The metrics map
     * @param key The key to extract
     * @return The integer value, or 0 if not found
     */
    private int getIntValue(Map<String, Object> metrics, String key) {
        if (metrics.containsKey(key)) {
            Object value = metrics.get(key);
            if (value instanceof Integer) {
                return (Integer) value;
            } else if (value instanceof Number) {
                return ((Number) value).intValue();
            }
        }
        return 0;
    }
    
    /**
     * Safely extracts a double value from a metrics map.
     * 
     * @param metrics The metrics map
     * @param key The key to extract
     * @return The double value, or 0.0 if not found
     */
    private double getDoubleValue(Map<String, Object> metrics, String key) {
        return getDoubleValue(metrics, key, 0.0);
    }
    
    /**
     * Safely extracts a double value from a metrics map with a default value.
     * 
     * @param metrics The metrics map
     * @param key The key to extract
     * @param defaultValue The default value if not found
     * @return The double value, or the default value if not found
     */
    private double getDoubleValue(Map<String, Object> metrics, String key, double defaultValue) {
        if (metrics.containsKey(key)) {
            Object value = metrics.get(key);
            if (value instanceof Double) {
                return (Double) value;
            } else if (value instanceof Number) {
                return ((Number) value).doubleValue();
            }
        }
        return defaultValue;
    }
    
    /**
     * Generates all charts and saves them to the output directory.
     */
    public void generateCharts() {
        logger.info("Generating charts in directory: {}", outputDirectory);
        
        try {
            // Generate time series charts
            generateTimeSeriesChart("task_generation", "Task Generation Over Time",
                "Time (s)", "Number of Tasks", tasksGenerated);
                
            generateTimeSeriesChart("task_offloading", "Task Offloading Over Time",
                "Time (s)", "Number of Tasks", tasksOffloaded);
                
            generateTimeSeriesChart("task_local_processing", "Local Task Processing Over Time",
                "Time (s)", "Number of Tasks", tasksProcessedLocally);
                
            generateTimeSeriesChart("task_completion", "Task Completion Over Time",
                "Time (s)", "Number of Tasks", tasksCompleted);
                
            generateTimeSeriesChart("task_failure", "Task Failures Over Time",
                "Time (s)", "Number of Tasks", tasksFailed);
                
            generateTimeSeriesChart("energy_consumption", "Energy Consumption Over Time",
                "Time (s)", "Energy (Joules)", energyConsumed);
                
            generateTimeSeriesChart("average_delay", "Average Task Delay Over Time",
                "Time (s)", "Delay (s)", averageDelay);
                
            generateTimeSeriesChart("completion_rate", "Task Completion Rate Over Time",
                "Time (s)", "Completion Rate (%)", taskCompletionRate);
                
            generateTimeSeriesChart("server_utilization", "Server Utilization Over Time",
                "Time (s)", "Utilization (%)", serverUtilization);
                
            // Generate summary charts
            generateSummaryChart();
            
            // Generate comparison charts
            generateComparisonChart();
            
            logger.info("Chart generation completed successfully");
        } catch (IOException e) {
            logger.error("Error generating charts: {}", e.getMessage());
        }
    }
    
    /**
     * Generates a time series chart.
     * 
     * @param filename The output filename (without extension)
     * @param title The chart title
     * @param xAxisLabel The x-axis label
     * @param yAxisLabel The y-axis label
     * @param data The data series
     * @throws IOException If an I/O error occurs
     */
    private void generateTimeSeriesChart(String filename, String title, String xAxisLabel, String yAxisLabel,
            List<? extends Number> data) throws IOException {
        XYSeries series = new XYSeries(title);
        
        for (int i = 0; i < timePoints.size(); i++) {
            series.add(timePoints.get(i), data.get(i));
        }
        
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        JFreeChart chart = ChartFactory.createXYLineChart(
            title,
            xAxisLabel,
            yAxisLabel,
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        File outputFile = new File(outputDirectory, filename + ".png");
        ChartUtils.saveChartAsPNG(outputFile, chart, 800, 600);
        logger.info("Chart saved: {}", outputFile.getAbsolutePath());
    }
    
    /**
     * Generates a summary chart with overall metrics.
     * 
     * @throws IOException If an I/O error occurs
     */
    private void generateSummaryChart() throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        dataset.addValue(totalTasksGenerated, "Tasks", "Generated");
        dataset.addValue(totalTasksOffloaded, "Tasks", "Offloaded");
        dataset.addValue(totalTasksProcessedLocally, "Tasks", "Local");
        dataset.addValue(totalTasksCompleted, "Tasks", "Completed");
        dataset.addValue(totalTasksFailed, "Tasks", "Failed");
        
        JFreeChart chart = ChartFactory.createBarChart(
            "Task Processing Summary",
            "Category",
            "Number of Tasks",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        File outputFile = new File(outputDirectory, "task_summary.png");
        ChartUtils.saveChartAsPNG(outputFile, chart, 800, 600);
        logger.info("Summary chart saved: {}", outputFile.getAbsolutePath());
    }
    
    /**
     * Generates a comparison chart for offloaded vs. local processing.
     * 
     * @throws IOException If an I/O error occurs
     */
    private void generateComparisonChart() throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        double offloadedPercentage = (totalTasksGenerated > 0) ? 
            (totalTasksOffloaded * 100.0) / totalTasksGenerated : 0;
        double localPercentage = (totalTasksGenerated > 0) ? 
            (totalTasksProcessedLocally * 100.0) / totalTasksGenerated : 0;
        
        dataset.addValue(offloadedPercentage, "Percentage", "Offloaded");
        dataset.addValue(localPercentage, "Percentage", "Local");
        
        JFreeChart chart = ChartFactory.createBarChart(
            "Task Processing Distribution",
            "Processing Location",
            "Percentage of Tasks",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        File outputFile = new File(outputDirectory, "processing_distribution.png");
        ChartUtils.saveChartAsPNG(outputFile, chart, 800, 600);
        logger.info("Comparison chart saved: {}", outputFile.getAbsolutePath());
    }
    
    /**
     * Exports all metrics to a CSV file.
     */
    public void exportToCSV() {
        try {
            File outputFile = new File(outputDirectory, "simulation_metrics.csv");
            FileWriter writer = new FileWriter(outputFile);
            
            // Write header
            writer.write("Time,TasksGenerated,TasksOffloaded,TasksProcessedLocally," +
                "TasksCompleted,TasksFailed,EnergyConsumed,AverageDelay," +
                "TaskCompletionRate,ServerUtilization\n");
            
            // Write data rows
            for (int i = 0; i < timePoints.size(); i++) {
                writer.write(String.format("%.2f,%d,%d,%d,%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                    timePoints.get(i),
                    tasksGenerated.get(i),
                    tasksOffloaded.get(i),
                    tasksProcessedLocally.get(i),
                    tasksCompleted.get(i),
                    tasksFailed.get(i),
                    energyConsumed.get(i),
                    averageDelay.get(i),
                    taskCompletionRate.get(i),
                    serverUtilization.get(i)));
            }
            
            writer.close();
            logger.info("Metrics exported to CSV: {}", outputFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Error exporting metrics to CSV: {}", e.getMessage());
        }
    }
    
    /**
     * Exports summary statistics to a text file.
     */
    public void exportSummary() {
        try {
            File outputFile = new File(outputDirectory, "simulation_summary.txt");
            FileWriter writer = new FileWriter(outputFile);
            
            writer.write("TODG Simulation Summary\n");
            writer.write("=======================\n\n");
            
            writer.write(String.format("Total simulation time: %.2f seconds\n", 
                timePoints.isEmpty() ? 0 : timePoints.get(timePoints.size() - 1)));
            writer.write(String.format("Total tasks generated: %d\n", totalTasksGenerated));
            writer.write(String.format("Tasks offloaded to edge servers: %d (%.2f%%)\n", 
                totalTasksOffloaded, 
                totalTasksGenerated > 0 ? (totalTasksOffloaded * 100.0) / totalTasksGenerated : 0));
            writer.write(String.format("Tasks processed locally: %d (%.2f%%)\n", 
                totalTasksProcessedLocally, 
                totalTasksGenerated > 0 ? (totalTasksProcessedLocally * 100.0) / totalTasksGenerated : 0));
            writer.write(String.format("Tasks completed successfully: %d (%.2f%%)\n", 
                totalTasksCompleted, 
                totalTasksGenerated > 0 ? (totalTasksCompleted * 100.0) / totalTasksGenerated : 0));
            writer.write(String.format("Tasks failed (missed deadline): %d (%.2f%%)\n", 
                totalTasksFailed, 
                totalTasksGenerated > 0 ? (totalTasksFailed * 100.0) / totalTasksGenerated : 0));
            writer.write(String.format("Total energy consumed: %.2f Joules\n", totalEnergyConsumed));
            
            // Calculate average metrics
            double avgDelay = calculateAverage(averageDelay);
            double avgCompletionRate = calculateAverage(taskCompletionRate);
            double avgServerUtilization = calculateAverage(serverUtilization);
            
            writer.write(String.format("Average task delay: %.4f seconds\n", avgDelay));
            writer.write(String.format("Average task completion rate: %.2f%%\n", avgCompletionRate));
            writer.write(String.format("Average server utilization: %.2f%%\n", avgServerUtilization));
            
            writer.close();
            logger.info("Summary exported to: {}", outputFile.getAbsolutePath());
        } catch (IOException e) {
            logger.error("Error exporting summary: {}", e.getMessage());
        }
    }
    
    /**
     * Calculates the average of a list of numbers.
     * 
     * @param list The list of numbers
     * @return The average value
     */
    private double calculateAverage(List<? extends Number> list) {
        if (list.isEmpty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (Number num : list) {
            sum += num.doubleValue();
        }
        
        return sum / list.size();
    }
    
    /**
     * Gets a map of all cumulative metrics.
     * 
     * @return A map of cumulative metrics
     */
    public Map<String, Object> getCumulativeMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        metrics.put("totalTasksGenerated", totalTasksGenerated);
        metrics.put("totalTasksOffloaded", totalTasksOffloaded);
        metrics.put("totalTasksProcessedLocally", totalTasksProcessedLocally);
        metrics.put("totalTasksCompleted", totalTasksCompleted);
        metrics.put("totalTasksFailed", totalTasksFailed);
        metrics.put("totalEnergyConsumed", totalEnergyConsumed);
        
        // Calculate average metrics
        metrics.put("averageDelay", calculateAverage(averageDelay));
        metrics.put("averageCompletionRate", calculateAverage(taskCompletionRate));
        metrics.put("averageServerUtilization", calculateAverage(serverUtilization));
        
        return metrics;
    }
    
    /**
     * Gets the output directory.
     * 
     * @return The output directory
     */
    public String getOutputDirectory() {
        return outputDirectory;
    }
    
    /**
     * Sets the output directory.
     * 
     * @param outputDirectory The output directory
     */
    public void setOutputDirectory(String outputDirectory) {
        this.outputDirectory = outputDirectory;
        
        // Create directory if it doesn't exist
        File directory = new File(outputDirectory);
        if (!directory.exists()) {
            directory.mkdirs();
        }
    }
}
