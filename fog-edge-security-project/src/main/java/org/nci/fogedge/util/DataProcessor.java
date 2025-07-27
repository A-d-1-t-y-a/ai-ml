package org.nci.fogedge.util;

import java.util.logging.Logger;

import java.util.Arrays;
import java.util.Random;

/**
 * Utility class for processing data at different levels of the fog architecture.
 * Implements data filtering, aggregation, and analytics algorithms.
 */
public class DataProcessor {
    
    private static final Random random = new Random();
    
    /**
     * Processes data at the edge level.
     * Edge processing typically involves filtering, basic aggregation, and preprocessing.
     * 
     * @param data The raw data to process
     * @return Processed data
     */
    public static byte[] processDataAtEdge(byte[] data) {
        // In a real implementation, this would perform actual data processing
        // For simulation purposes, we'll just reduce the data size by filtering
        
        // Simulate filtering out 30% of the data
        int newSize = (int) (data.length * 0.7);
        byte[] processedData = new byte[newSize];
        
        // Copy a subset of the data
        System.arraycopy(data, 0, processedData, 0, newSize);
        
        // Log processing information
        Log.printLine("Edge processing: Filtered data from " + data.length + 
                " bytes to " + processedData.length + " bytes");
        
        return processedData;
    }
    
    /**
     * Processes data at the fog level.
     * Fog processing typically involves more complex analytics, aggregation, and decision making.
     * 
     * @param data The data to process (already preprocessed by edge)
     * @return Processed data
     */
    public static byte[] processDataAtFog(byte[] data) {
        // In a real implementation, this would perform actual data analytics
        // For simulation purposes, we'll aggregate and transform the data
        
        // Simulate data transformation and feature extraction
        // In this case, we'll just create a summary of the data (e.g., statistical features)
        byte[] processedData = new byte[100]; // Fixed size summary
        
        // Fill with simulated statistical features
        for (int i = 0; i < processedData.length; i++) {
            if (i < 20) {
                // First 20 bytes represent mean values of data segments
                processedData[i] = calculateMean(data, i * (data.length / 20), (i + 1) * (data.length / 20));
            } else if (i < 40) {
                // Next 20 bytes represent variance of data segments
                processedData[i] = calculateVariance(data, (i - 20) * (data.length / 20), (i - 19) * (data.length / 20));
            } else {
                // Remaining bytes represent other statistical features
                processedData[i] = (byte) random.nextInt(256);
            }
        }
        
        // Log processing information
        Log.printLine("Fog processing: Transformed data from " + data.length + 
                " bytes to " + processedData.length + " bytes (statistical summary)");
        
        return processedData;
    }
    
    /**
     * Performs big data analytics on aggregated data.
     * This would typically run on fog nodes or in the cloud.
     * 
     * @param dataMap Map of data from different sources
     * @return Analytics results
     */
    public static byte[] performBigDataAnalytics(byte[][] dataArray) {
        // In a real implementation, this would perform complex analytics
        // For simulation purposes, we'll just aggregate the data
        
        // Calculate total size of all data
        int totalSize = 0;
        for (byte[] data : dataArray) {
            totalSize += data.length;
        }
        
        // Create aggregated data array
        byte[] aggregatedData = new byte[totalSize];
        
        // Copy all data into the aggregated array
        int offset = 0;
        for (byte[] data : dataArray) {
            System.arraycopy(data, 0, aggregatedData, offset, data.length);
            offset += data.length;
        }
        
        // Simulate analytics processing
        // For simulation, we'll just create a small result set
        byte[] analyticsResults = new byte[50];
        
        // Fill with simulated analytics results
        for (int i = 0; i < analyticsResults.length; i++) {
            analyticsResults[i] = (byte) (aggregatedData[i % aggregatedData.length] & 0xFF);
        }
        
        // Log analytics information
        Log.printLine("Big data analytics: Processed " + totalSize + 
                " bytes of aggregated data into " + analyticsResults.length + " bytes of results");
        
        return analyticsResults;
    }
    
    /**
     * Calculates the mean value of a segment of data
     * 
     * @param data The data array
     * @param start Start index
     * @param end End index
     * @return Mean value as a byte
     */
    private static byte calculateMean(byte[] data, int start, int end) {
        if (start >= end || start >= data.length) {
            return 0;
        }
        
        end = Math.min(end, data.length);
        
        int sum = 0;
        for (int i = start; i < end; i++) {
            sum += data[i] & 0xFF; // Convert to unsigned
        }
        
        return (byte) (sum / (end - start));
    }
    
    /**
     * Calculates the variance of a segment of data
     * 
     * @param data The data array
     * @param start Start index
     * @param end End index
     * @return Variance value as a byte
     */
    private static byte calculateVariance(byte[] data, int start, int end) {
        if (start >= end || start >= data.length) {
            return 0;
        }
        
        end = Math.min(end, data.length);
        
        // Calculate mean first
        byte mean = calculateMean(data, start, end);
        
        // Calculate variance
        int variance = 0;
        for (int i = start; i < end; i++) {
            int diff = (data[i] & 0xFF) - (mean & 0xFF);
            variance += diff * diff;
        }
        
        return (byte) (variance / (end - start));
    }
}
