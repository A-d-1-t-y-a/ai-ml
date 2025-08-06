package org.jcora.mec.util;

import org.jcora.mec.simulation.MECEnvironment;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

// Use simple System.out logging instead of SLF4J for now
// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

/**
 * Utility class for generating visualizations of simulation results.
 */
public class VisualizationUtil {
    // Use simple System.out logging instead of SLF4J for now
    // private static final Logger logger = LoggerFactory.getLogger(VisualizationUtil.class);
    
    // Chart dimensions
    private static final int CHART_WIDTH = 800;
    private static final int CHART_HEIGHT = 600;
    
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
                System.out.println("Created output directory: " + outputDir);
                return true;
            } catch (IOException e) {
                System.err.println("Failed to create output directory: " + e.getMessage());
                return false;
            }
        }
        return true;
    }
    
    /**
     * Generate a line chart for energy consumption over time.
     * 
     * @param environment MEC environment with simulation results
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateEnergyConsumptionChart(MECEnvironment environment, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Get energy consumption history
        List<Double> energyHistory = environment.getEnergyConsumptionHistory();
        
        // Create dataset
        XYSeries series = new XYSeries("Energy Consumption");
        for (int i = 0; i < energyHistory.size(); i++) {
            series.add(i, energyHistory.get(i));
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Energy Consumption Over Time - " + scenarioName,
                "Time Step",
                "Energy Consumption (J)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Customize chart
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.BLUE);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_energy_consumption_%s.png", outputDir, scenarioName, timestamp);
        
        // Save chart to file
        try {
            ChartUtils.saveChartAsPNG(new File(filename), chart, CHART_WIDTH, CHART_HEIGHT);
            System.out.println("Generated energy consumption chart: " + filename);
        } catch (IOException e) {
            System.err.println("Failed to generate energy consumption chart: " + e.getMessage());
        }
    }
    
    /**
     * Generate a line chart for response time over time.
     * 
     * @param environment MEC environment with simulation results
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateResponseTimeChart(MECEnvironment environment, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Get response time history
        List<Double> responseTimeHistory = environment.getResponseTimeHistory();
        
        // Create dataset
        XYSeries series = new XYSeries("Response Time");
        for (int i = 0; i < responseTimeHistory.size(); i++) {
            series.add(i, responseTimeHistory.get(i));
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Response Time Over Time - " + scenarioName,
                "Time Step",
                "Response Time (s)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Customize chart
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_response_time_%s.png", outputDir, scenarioName, timestamp);
        
        // Save chart to file
        try {
            ChartUtils.saveChartAsPNG(new File(filename), chart, CHART_WIDTH, CHART_HEIGHT);
            System.out.println("Generated response time chart: " + filename);
        } catch (IOException e) {
            System.err.println("Failed to generate response time chart: " + e.getMessage());
        }
    }
    
    /**
     * Generate a line chart for deadline miss rate over time.
     * 
     * @param environment MEC environment with simulation results
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateDeadlineMissRateChart(MECEnvironment environment, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Get deadline miss rate history
        List<Double> deadlineMissRateHistory = environment.getDeadlineMissRateHistory();
        
        // Create dataset
        XYSeries series = new XYSeries("Deadline Miss Rate");
        for (int i = 0; i < deadlineMissRateHistory.size(); i++) {
            series.add(i, deadlineMissRateHistory.get(i));
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Deadline Miss Rate Over Time - " + scenarioName,
                "Time Step",
                "Deadline Miss Rate (%)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Customize chart
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.ORANGE);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_deadline_miss_rate_%s.png", outputDir, scenarioName, timestamp);
        
        // Save chart to file
        try {
            ChartUtils.saveChartAsPNG(new File(filename), chart, CHART_WIDTH, CHART_HEIGHT);
            System.out.println("Generated deadline miss rate chart: " + filename);
        } catch (IOException e) {
            System.err.println("Failed to generate deadline miss rate chart: " + e.getMessage());
        }
    }
    
    /**
     * Generate a line chart for task completion rate over time.
     * 
     * @param environment MEC environment with simulation results
     * @param outputDir Path to the output directory
     * @param scenarioName Name of the simulation scenario
     */
    public static void generateTaskCompletionRateChart(MECEnvironment environment, String outputDir, String scenarioName) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Get task completion rate history
        List<Double> taskCompletionRateHistory = environment.getTaskCompletionRateHistory();
        
        // Create dataset
        XYSeries series = new XYSeries("Task Completion Rate");
        for (int i = 0; i < taskCompletionRateHistory.size(); i++) {
            series.add(i, taskCompletionRateHistory.get(i));
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Task Completion Rate Over Time - " + scenarioName,
                "Time Step",
                "Task Completion Rate (%)",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Customize chart
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.GREEN);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String filename = String.format("%s/%s_task_completion_rate_%s.png", outputDir, scenarioName, timestamp);
        
        // Save chart to file
        try {
            ChartUtils.saveChartAsPNG(new File(filename), chart, CHART_WIDTH, CHART_HEIGHT);
            System.out.println("Generated task completion rate chart: " + filename);
        } catch (IOException e) {
            System.err.println("Failed to generate task completion rate chart: " + e.getMessage());
        }
    }
    
    /**
     * Generate a comparison chart for multiple scenarios.
     * 
     * @param scenarioNames List of scenario names
     * @param energyValues List of energy consumption values for each scenario
     * @param responseTimeValues List of response time values for each scenario
     * @param deadlineMissRateValues List of deadline miss rate values for each scenario
     * @param taskCompletionRateValues List of task completion rate values for each scenario
     * @param outputDir Path to the output directory
     */
    public static void generateComparisonChart(List<String> scenarioNames, 
                                             List<Double> energyValues,
                                             List<Double> responseTimeValues,
                                             List<Double> deadlineMissRateValues,
                                             List<Double> taskCompletionRateValues,
                                             String outputDir) {
        // Create output directory if it doesn't exist
        if (!createOutputDirectory(outputDir)) {
            return;
        }
        
        // Generate timestamp for the filename
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        
        // Generate energy consumption comparison chart
        generateMetricComparisonChart(scenarioNames, energyValues, "Energy Consumption (J)",
                "Energy Consumption Comparison", outputDir, "energy_comparison_" + timestamp);
        
        // Generate response time comparison chart
        generateMetricComparisonChart(scenarioNames, responseTimeValues, "Response Time (s)",
                "Response Time Comparison", outputDir, "response_time_comparison_" + timestamp);
        
        // Generate deadline miss rate comparison chart
        generateMetricComparisonChart(scenarioNames, deadlineMissRateValues, "Deadline Miss Rate (%)",
                "Deadline Miss Rate Comparison", outputDir, "deadline_miss_rate_comparison_" + timestamp);
        
        // Generate task completion rate comparison chart
        generateMetricComparisonChart(scenarioNames, taskCompletionRateValues, "Task Completion Rate (%)",
                "Task Completion Rate Comparison", outputDir, "task_completion_rate_comparison_" + timestamp);
    }
    
    /**
     * Generate a comparison chart for a specific metric across multiple scenarios.
     * 
     * @param scenarioNames List of scenario names
     * @param metricValues List of metric values for each scenario
     * @param metricLabel Label for the metric
     * @param chartTitle Title of the chart
     * @param outputDir Path to the output directory
     * @param filenameSuffix Suffix for the filename
     */
    private static void generateMetricComparisonChart(List<String> scenarioNames,
                                                    List<Double> metricValues,
                                                    String metricLabel,
                                                    String chartTitle,
                                                    String outputDir,
                                                    String filenameSuffix) {
        // Create dataset
        XYSeries series = new XYSeries(metricLabel);
        for (int i = 0; i < scenarioNames.size(); i++) {
            series.add(i, metricValues.get(i));
        }
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(series);
        
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                chartTitle,
                "Scenario",
                metricLabel,
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        
        // Customize chart
        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.BLUE);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        plot.setRenderer(renderer);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        
        // Save chart to file
        String filename = String.format("%s/%s.png", outputDir, filenameSuffix);
        try {
            ChartUtils.saveChartAsPNG(new File(filename), chart, CHART_WIDTH, CHART_HEIGHT);
            System.out.println("Generated comparison chart: " + filename);
        } catch (IOException e) {
            System.err.println("Failed to generate comparison chart: " + e.getMessage());
        }
    }
}
