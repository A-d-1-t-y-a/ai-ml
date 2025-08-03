package com.fog.eedto.util;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import java.util.logging.Logger;
import java.util.logging.Level;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import com.fog.eedto.simulation.SimulationResults;

/**
 * Utility class for generating charts and graphs for EEDTO simulation results.
 */
public class ChartGenerator {
    private static final Logger logger = Logger.getLogger(ChartGenerator.class.getName());
    
    // Directory for storing output files
    private static final String OUTPUT_DIR = "output";
    
    // Chart dimensions from configuration
    private static final int CHART_WIDTH = ConfigurationManager.getInt("chart.width", 800);
    private static final int CHART_HEIGHT = ConfigurationManager.getInt("chart.height", 600);
    
    // Chart colors from configuration
    private static final Color LOCAL_EXECUTION_COLOR = ConfigurationManager.getColor("chart.localExecutionColor", new Color(0, 153, 51));
    private static final Color EDGE_OFFLOAD_COLOR = ConfigurationManager.getColor("chart.edgeOffloadColor", new Color(51, 153, 255));
    private static final Color CLOUD_OFFLOAD_COLOR = ConfigurationManager.getColor("chart.cloudOffloadColor", new Color(153, 51, 255));
    private static final Color FAILED_OFFLOAD_COLOR = ConfigurationManager.getColor("chart.failedOffloadColor", new Color(255, 51, 51));
    private static final Color COST_COLOR = ConfigurationManager.getColor("chart.costColor", new Color(255, 153, 0));
    
    /**
     * Generate all charts for a single simulation
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @return Array of generated file paths
     */
    public static String[] generateAllCharts(String name, SimulationResults results) {
        // Create output directory if it doesn't exist
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filePrefix = OUTPUT_DIR + File.separator + name + "_" + timestamp;
        
        String taskDistributionChart = generateTaskDistributionPieChart(name, results, filePrefix + "_task_distribution.png");
        String energyConsumptionChart = generateEnergyConsumptionBarChart(name, results, filePrefix + "_energy_consumption.png");
        String responseTimeChart = generateResponseTimeBarChart(name, results, filePrefix + "_response_time.png");
        String costChart = generateCostBarChart(name, results, filePrefix + "_cost.png");
        
        logger.info(String.format("Generated charts for simulation: %s", name));
        
        return new String[] {
            taskDistributionChart,
            energyConsumptionChart,
            responseTimeChart,
            costChart
        };
    }
    
    /**
     * Generate task distribution pie chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateTaskDistributionPieChart(String name, SimulationResults results, String outputFile) {
        DefaultPieDataset<String> dataset = new DefaultPieDataset<>();
        dataset.setValue("Local Execution", results.getLocalExecutions());
        dataset.setValue("Edge Offload", results.getEdgeOffloads());
        dataset.setValue("Cloud Offload", results.getCloudOffloads());
        dataset.setValue("Failed Offload", results.getFailedOffloads());
        
        JFreeChart chart = ChartFactory.createPieChart(
            name + " - Task Distribution",
            dataset,
            true,
            true,
            false
        );
        
        PiePlot<String> plot = (PiePlot<String>) chart.getPlot();
        plot.setSectionPaint("Local Execution", LOCAL_EXECUTION_COLOR);
        plot.setSectionPaint("Edge Offload", EDGE_OFFLOAD_COLOR);
        plot.setSectionPaint("Cloud Offload", CLOUD_OFFLOAD_COLOR);
        plot.setSectionPaint("Failed Offload", FAILED_OFFLOAD_COLOR);
        
        plot.setLabelFont(new Font("SansSerif", Font.PLAIN, 12));
        plot.setNoDataMessage("No data available");
        plot.setCircular(true);
        plot.setLabelGap(0.02);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated task distribution chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.severe("Error saving task distribution chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate energy consumption bar chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateEnergyConsumptionBarChart(String name, SimulationResults results, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(results.getAverageEnergyPerTask(), "Average Energy (J)", name);
        
        JFreeChart chart = ChartFactory.createBarChart(
            name + " - Energy Consumption",
            "Simulation",
            "Average Energy per Task (Joules)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(0, 153, 51));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated energy consumption chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.severe("Error saving energy consumption chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate response time bar chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateResponseTimeBarChart(String name, SimulationResults results, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(results.getAverageResponseTime(), "Average Response Time (s)", name);
        
        JFreeChart chart = ChartFactory.createBarChart(
            name + " - Response Time",
            "Simulation",
            "Average Response Time (seconds)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(51, 153, 255));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated response time chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.severe("Error saving response time chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate cost bar chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateCostBarChart(String name, SimulationResults results, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.addValue(results.getAverageExecutionCost(), "Average Cost ($)", name);
        
        JFreeChart chart = ChartFactory.createBarChart(
            name + " - Execution Cost",
            "Simulation",
            "Average Cost per Task ($)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(255, 153, 0));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated cost chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.severe("Error saving cost chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate comparative charts for multiple simulations
     * 
     * @param simulationNames Names of the simulations
     * @param allResults List of simulation results
     * @return Array of generated file paths
     */
    public static String[] generateComparativeCharts(String[] simulationNames, List<SimulationResults> allResults) {
        // Create output directory if it doesn't exist
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filePrefix = OUTPUT_DIR + File.separator + "Comparative_" + timestamp;
        
        String energyChart = generateComparativeEnergyChart(simulationNames, allResults, filePrefix + "_energy.png");
        String responseTimeChart = generateComparativeResponseTimeChart(simulationNames, allResults, filePrefix + "_response_time.png");
        String offloadingChart = generateComparativeOffloadingChart(simulationNames, allResults, filePrefix + "_offloading.png");
        String costChart = generateComparativeCostChart(simulationNames, allResults, filePrefix + "_cost.png");
        
        logger.info("Generated comparative charts");
        
        return new String[] {
            energyChart,
            responseTimeChart,
            offloadingChart,
            costChart
        };
    }
    
    /**
     * Generate comparative energy consumption chart
     * 
     * @param simulationNames Names of the simulations
     * @param allResults List of simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateComparativeEnergyChart(String[] simulationNames, List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (int i = 0; i < allResults.size(); i++) {
            dataset.addValue(allResults.get(i).getAverageEnergyPerTask(), "Average Energy (J)", simulationNames[i]);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
            "Comparative Energy Consumption",
            "Simulation Configuration",
            "Average Energy per Task (Joules)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(0, 153, 51));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated comparative energy chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error saving comparative energy chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate comparative response time chart
     * 
     * @param simulationNames Names of the simulations
     * @param allResults List of simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateComparativeResponseTimeChart(String[] simulationNames, List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (int i = 0; i < allResults.size(); i++) {
            dataset.addValue(allResults.get(i).getAverageResponseTime(), "Average Response Time (s)", simulationNames[i]);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
            "Comparative Response Time",
            "Simulation Configuration",
            "Average Response Time (seconds)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(51, 153, 255));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated comparative response time chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error saving comparative response time chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate comparative offloading distribution chart
     * 
     * @param simulationNames Names of the simulations
     * @param allResults List of simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateComparativeOffloadingChart(String[] simulationNames, List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (int i = 0; i < allResults.size(); i++) {
            SimulationResults results = allResults.get(i);
            dataset.addValue(results.getLocalExecutionPercentage(), "Local Execution (%)", simulationNames[i]);
            dataset.addValue(results.getEdgeOffloadPercentage(), "Edge Offload (%)", simulationNames[i]);
            dataset.addValue(results.getCloudOffloadPercentage(), "Cloud Offload (%)", simulationNames[i]);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
            "Comparative Offloading Distribution",
            "Simulation Configuration",
            "Percentage (%)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, LOCAL_EXECUTION_COLOR);
        renderer.setSeriesPaint(1, EDGE_OFFLOAD_COLOR);
        renderer.setSeriesPaint(2, CLOUD_OFFLOAD_COLOR);
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated comparative offloading chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error saving comparative offloading chart: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate comparative cost chart
     * 
     * @param simulationNames Names of the simulations
     * @param allResults List of simulation results
     * @param outputFile Output file path
     * @return Path to the generated chart file
     */
    public static String generateComparativeCostChart(String[] simulationNames, List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        for (int i = 0; i < allResults.size(); i++) {
            dataset.addValue(allResults.get(i).getAverageExecutionCost(), "Average Cost ($)", simulationNames[i]);
        }
        
        JFreeChart chart = ChartFactory.createBarChart(
            "Comparative Execution Cost",
            "Simulation Configuration",
            "Average Cost per Task ($)",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        );
        
        CategoryPlot plot = chart.getCategoryPlot();
        BarRenderer renderer = (BarRenderer) plot.getRenderer();
        renderer.setSeriesPaint(0, new Color(255, 153, 0));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, CHART_WIDTH, CHART_HEIGHT);
            logger.info(String.format("Generated comparative cost chart: %s", outputFile));
            return outputFile;
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error saving comparative cost chart: " + e.getMessage());
            return null;
        }
    }
}
