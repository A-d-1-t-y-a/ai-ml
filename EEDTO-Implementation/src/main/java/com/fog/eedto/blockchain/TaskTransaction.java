package com.fog.eedto.blockchain;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;

/**
 * Represents a transaction in the blockchain for the EEDTO system.
 * Each transaction records a task offloading decision from one device to another.
 */
public class TaskTransaction {
    private final String transactionId;
    private final int taskId;
    private final int sourceDeviceId;
    private final int targetDeviceId;
    private final long timestamp;
    private final long taskLength;
    private final long taskInputSize;
    private final long taskOutputSize;
    
    /**
     * Constructor for the TaskTransaction class
     * 
     * @param taskId ID of the task being offloaded
     * @param sourceDeviceId ID of the source device (offloading the task)
     * @param targetDeviceId ID of the target device (receiving the task)
     * @param timestamp Transaction timestamp
     * @param taskLength Computational length of the task in Million Instructions (MI)
     * @param taskInputSize Input data size of the task in bytes
     * @param taskOutputSize Output data size of the task in bytes
     */
    public TaskTransaction(int taskId, int sourceDeviceId, int targetDeviceId, 
                          long timestamp, long taskLength, long taskInputSize, 
                          long taskOutputSize) {
        this.taskId = taskId;
        this.sourceDeviceId = sourceDeviceId;
        this.targetDeviceId = targetDeviceId;
        this.timestamp = timestamp;
        this.taskLength = taskLength;
        this.taskInputSize = taskInputSize;
        this.taskOutputSize = taskOutputSize;
        this.transactionId = calculateTransactionId();
    }
    
    /**
     * Calculate a unique transaction ID based on the transaction data
     * 
     * @return SHA-256 hash of the transaction data
     */
    private String calculateTransactionId() {
        String dataToHash = taskId + sourceDeviceId + targetDeviceId + timestamp + 
                           taskLength + taskInputSize + taskOutputSize + UUID.randomUUID().toString();
        
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hashBytes = digest.digest(dataToHash.getBytes(StandardCharsets.UTF_8));
            
            // Convert byte array to hexadecimal string
            StringBuilder hexString = new StringBuilder();
            for (byte hashByte : hashBytes) {
                String hex = Integer.toHexString(0xff & hashByte);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("Error calculating transaction ID: " + e.getMessage(), e);
        }
    }
    
    // Getters
    public String getTransactionId() {
        return transactionId;
    }
    
    public int getTaskId() {
        return taskId;
    }
    
    public int getSourceDeviceId() {
        return sourceDeviceId;
    }
    
    public int getTargetDeviceId() {
        return targetDeviceId;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public long getTaskLength() {
        return taskLength;
    }
    
    public long getTaskInputSize() {
        return taskInputSize;
    }
    
    public long getTaskOutputSize() {
        return taskOutputSize;
    }
    
    @Override
    public String toString() {
        return "TaskTransaction{" +
                "transactionId='" + transactionId + '\'' +
                ", taskId=" + taskId +
                ", sourceDeviceId=" + sourceDeviceId +
                ", targetDeviceId=" + targetDeviceId +
                ", timestamp=" + timestamp +
                ", taskLength=" + taskLength +
                ", taskInputSize=" + taskInputSize +
                ", taskOutputSize=" + taskOutputSize +
                '}';
    }
}
