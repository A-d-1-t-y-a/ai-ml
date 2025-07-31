package com.fog.eedto;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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

import com.fog.eedto.simulation.Simulation;
import com.fog.eedto.simulation.SimulationResults;

/**
 * Main class for the EEDTO system.
 * This class runs the simulation with different parameters and generates visualizations of the results.
 */
public class Main {
    private static final Logger logger = LogManager.getLogger(Main.class);
    
    // Directory for storing output files
    private static final String OUTPUT_DIR = "output";
    
    public static void main(String[] args) {
        logger.info("Starting EEDTO simulation");
        
        // Create output directory if it doesn't exist
        File outputDir = new File(OUTPUT_DIR);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        
        // Run simulations with different parameters
        List<SimulationResults> allResults = new ArrayList<>();
        
        // Baseline simulation
        logger.info("Running baseline simulation");
        SimulationResults baselineResults = runSimulation(
            "Baseline",
            10, 3, 1, 300, 0.1,
            0.33, 0.33, 0.33, 0.2, 5, 3, 2
        );
        allResults.add(baselineResults);
        
        // Energy-focused simulation
        logger.info("Running energy-focused simulation");
        SimulationResults energyResults = runSimulation(
            "Energy-Focused",
            10, 3, 1, 300, 0.1,
            0.6, 0.2, 0.2, 0.2, 5, 3, 2
        );
        allResults.add(energyResults);
        
        // Latency-focused simulation
        logger.info("Running latency-focused simulation");
        SimulationResults latencyResults = runSimulation(
            "Latency-Focused",
            10, 3, 1, 300, 0.1,
            0.2, 0.6, 0.2, 0.2, 5, 3, 2
        );
        allResults.add(latencyResults);
        
        // Security-focused simulation
        logger.info("Running security-focused simulation");
        SimulationResults securityResults = runSimulation(
            "Security-Focused",
            10, 3, 1, 300, 0.1,
            0.2, 0.2, 0.6, 0.2, 5, 3, 2
        );
        allResults.add(securityResults);
        
        // Generate comparative visualizations
        generateComparativeVisualizations(allResults);
        
        logger.info("EEDTO simulation completed successfully");
    }
    
    /**
     * Run a simulation with the specified parameters
     * 
     * @param name Simulation name
     * @param numIoTDevices Number of IoT devices
     * @param numEdgeServers Number of edge servers
     * @param numCloudServers Number of cloud servers
     * @param simulationEndTime Simulation end time in seconds
     * @param taskGenerationRate Task generation rate per second per IoT device
     * @param energyWeight Weight factor for energy efficiency in decision-making
     * @param latencyWeight Weight factor for latency in decision-making
     * @param securityWeight Weight factor for security in decision-making
     * @param energyThreshold Minimum battery level for IoT devices (percentage)
     * @param latencyThreshold Maximum acceptable latency in seconds
     * @param securityLevel Required security level (1-5)
     * @param blockchainDifficulty Mining difficulty for the blockchain
     * @return Simulation results
     */
    private static SimulationResults runSimulation(String name, int numIoTDevices, int numEdgeServers, 
                                                 int numCloudServers, double simulationEndTime, 
                                                 double taskGenerationRate, double energyWeight, 
                                                 double latencyWeight, double securityWeight, 
                                                 double energyThreshold, double latencyThreshold, 
                                                 int securityLevel, int blockchainDifficulty) {
        // Create and run simulation
        Simulation simulation = new Simulation(
            numIoTDevices, numEdgeServers, numCloudServers,
            simulationEndTime, taskGenerationRate,
            energyWeight, latencyWeight, securityWeight,
            energyThreshold, latencyThreshold, securityLevel,
            blockchainDifficulty
        );
        
        simulation.run();
        
        // Get results
        SimulationResults results = simulation.getResults();
        
        // Generate visualizations
        generateVisualizations(name, results);
        
        return results;
    }
    
    /**
     * Generate visualizations for a single simulation
     * 
     * @param name Simulation name
     * @param results Simulation results
     */
    private static void generateVisualizations(String name, SimulationResults results) {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filePrefix = OUTPUT_DIR + File.separator + name + "_" + timestamp;
        
        // Generate task distribution pie chart
        generateTaskDistributionPieChart(name, results, filePrefix + "_task_distribution.png");
        
        // Generate energy consumption bar chart
        generateEnergyConsumptionBarChart(name, results, filePrefix + "_energy_consumption.png");
        
        // Generate response time bar chart
        generateResponseTimeBarChart(name, results, filePrefix + "_response_time.png");
        
        logger.info("Generated visualizations for simulation: {}", name);
    }
    
    /**
     * Generate task distribution pie chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     */
    private static void generateTaskDistributionPieChart(String name, SimulationResults results, String outputFile) {
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
        plot.setSectionPaint("Local Execution", new Color(0, 153, 51));
        plot.setSectionPaint("Edge Offload", new Color(51, 153, 255));
        plot.setSectionPaint("Cloud Offload", new Color(153, 51, 255));
        plot.setSectionPaint("Failed Offload", new Color(255, 51, 51));
        
        plot.setLabelFont(new Font("SansSerif", Font.PLAIN, 12));
        plot.setNoDataMessage("No data available");
        plot.setCircular(true);
        plot.setLabelGap(0.02);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 800, 600);
        } catch (IOException e) {
            logger.error("Error saving pie chart: {}", e.getMessage());
        }
    }
    
    /**
     * Generate energy consumption bar chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     */
    private static void generateEnergyConsumptionBarChart(String name, SimulationResults results, String outputFile) {
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
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 800, 600);
        } catch (IOException e) {
            logger.error("Error saving energy consumption chart: {}", e.getMessage());
        }
    }
    
    /**
     * Generate response time bar chart
     * 
     * @param name Simulation name
     * @param results Simulation results
     * @param outputFile Output file path
     */
    private static void generateResponseTimeBarChart(String name, SimulationResults results, String outputFile) {
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
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 800, 600);
        } catch (IOException e) {
            logger.error("Error saving response time chart: {}", e.getMessage());
        }
    }
    
    /**
     * Generate comparative visualizations for multiple simulations
     * 
     * @param allResults List of simulation results
     */
    private static void generateComparativeVisualizations(List<SimulationResults> allResults) {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String filePrefix = OUTPUT_DIR + File.separator + "Comparative_" + timestamp;
        
        // Generate comparative energy consumption chart
        generateComparativeEnergyChart(allResults, filePrefix + "_energy.png");
        
        // Generate comparative response time chart
        generateComparativeResponseTimeChart(allResults, filePrefix + "_response_time.png");
        
        // Generate comparative offloading distribution chart
        generateComparativeOffloadingChart(allResults, filePrefix + "_offloading.png");
        
        logger.info("Generated comparative visualizations");
    }
    
    /**
     * Generate comparative energy consumption chart
     * 
     * @param allResults List of simulation results
     * @param outputFile Output file path
     */
    private static void generateComparativeEnergyChart(List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        String[] names = {"Baseline", "Energy-Focused", "Latency-Focused", "Security-Focused"};
        for (int i = 0; i < allResults.size(); i++) {
            dataset.addValue(allResults.get(i).getAverageEnergyPerTask(), "Average Energy (J)", names[i]);
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
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 800, 600);
        } catch (IOException e) {
            logger.error("Error saving comparative energy chart: {}", e.getMessage());
        }
    }
    
    /**
     * Generate comparative response time chart
     * 
     * @param allResults List of simulation results
     * @param outputFile Output file path
     */
    private static void generateComparativeResponseTimeChart(List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        String[] names = {"Baseline", "Energy-Focused", "Latency-Focused", "Security-Focused"};
        for (int i = 0; i < allResults.size(); i++) {
            dataset.addValue(allResults.get(i).getAverageResponseTime(), "Average Response Time (s)", names[i]);
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
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 800, 600);
        } catch (IOException e) {
            logger.error("Error saving comparative response time chart: {}", e.getMessage());
        }
    }
    
    /**
     * Generate comparative offloading distribution chart
     * 
     * @param allResults List of simulation results
     * @param outputFile Output file path
     */
    private static void generateComparativeOffloadingChart(List<SimulationResults> allResults, String outputFile) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        
        String[] names = {"Baseline", "Energy-Focused", "Latency-Focused", "Security-Focused"};
        for (int i = 0; i < allResults.size(); i++) {
            SimulationResults results = allResults.get(i);
            dataset.addValue(results.getLocalExecutionPercentage(), "Local Execution (%)", names[i]);
            dataset.addValue(results.getEdgeOffloadPercentage(), "Edge Offload (%)", names[i]);
            dataset.addValue(results.getCloudOffloadPercentage(), "Cloud Offload (%)", names[i]);
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
        renderer.setSeriesPaint(0, new Color(0, 153, 51));
        renderer.setSeriesPaint(1, new Color(51, 153, 255));
        renderer.setSeriesPaint(2, new Color(153, 51, 255));
        renderer.setDrawBarOutline(false);
        renderer.setItemMargin(0.1);
        
        try {
            ChartUtils.saveChartAsPNG(new File(outputFile), chart, 800, 600);
        } catch (IOException e) {
            logger.error("Error saving comparative offloading chart: {}", e.getMessage());
        }
    }
}
