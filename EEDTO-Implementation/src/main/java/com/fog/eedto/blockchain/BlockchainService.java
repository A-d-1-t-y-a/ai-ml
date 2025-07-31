package com.fog.eedto.blockchain;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import com.fog.eedto.model.Task;
import com.fog.eedto.model.Device;

/**
 * Provides blockchain functionality for secure task offloading in the EEDTO system.
 * This simplified blockchain implementation records task offloading decisions and
 * ensures transparency and security in the offloading process.
 */
public class BlockchainService {
    private final List<Block> blockchain;
    private final int difficulty;
    private final List<TaskTransaction> pendingTransactions;
    
    /**
     * Constructor for the BlockchainService class
     * 
     * @param difficulty Mining difficulty (number of leading zeros required in hash)
     */
    public BlockchainService(int difficulty) {
        this.blockchain = new ArrayList<>();
        this.difficulty = difficulty;
        this.pendingTransactions = new CopyOnWriteArrayList<>();
        
        // Create genesis block
        createGenesisBlock();
    }
    
    /**
     * Create the genesis block (first block in the blockchain)
     */
    private void createGenesisBlock() {
        Block genesisBlock = new Block(0, System.currentTimeMillis(), new ArrayList<>(), "0");
        genesisBlock.mineBlock(difficulty);
        blockchain.add(genesisBlock);
        System.out.println("Genesis block created: " + genesisBlock.getHash());
    }
    
    /**
     * Get the latest block in the blockchain
     * 
     * @return Latest block
     */
    public Block getLatestBlock() {
        return blockchain.get(blockchain.size() - 1);
    }
    
    /**
     * Add a new task offloading transaction to the pending transactions list
     * 
     * @param task Task being offloaded
     * @param sourceDevice Source device (offloading the task)
     * @param targetDevice Target device (receiving the task)
     * @return Transaction ID
     */
    public String addTaskOffloadingTransaction(Task task, Device sourceDevice, Device targetDevice) {
        TaskTransaction transaction = new TaskTransaction(
            task.getId(),
            sourceDevice.getId(),
            targetDevice.getId(),
            System.currentTimeMillis(),
            task.getLength(),
            task.getInputSize(),
            task.getOutputSize()
        );
        
        pendingTransactions.add(transaction);
        return transaction.getTransactionId();
    }
    
    /**
     * Mine pending transactions into a new block
     * 
     * @return true if mining was successful, false otherwise
     */
    public boolean minePendingTransactions() {
        if (pendingTransactions.isEmpty()) {
            System.out.println("No pending transactions to mine");
            return false;
        }
        
        // Create a new block with all pending transactions
        Block newBlock = new Block(
            blockchain.size(),
            System.currentTimeMillis(),
            new ArrayList<>(pendingTransactions),
            getLatestBlock().getHash()
        );
        
        // Mine the block
        newBlock.mineBlock(difficulty);
        
        // Add the block to the blockchain
        blockchain.add(newBlock);
        
        // Clear pending transactions
        pendingTransactions.clear();
        
        System.out.println("Block mined and added to blockchain: " + newBlock.getHash());
        return true;
    }
    
    /**
     * Verify the integrity of the blockchain
     * 
     * @return true if the blockchain is valid, false otherwise
     */
    public boolean isChainValid() {
        for (int i = 1; i < blockchain.size(); i++) {
            Block currentBlock = blockchain.get(i);
            Block previousBlock = blockchain.get(i - 1);
            
            // Check if the current block's hash is valid
            if (!currentBlock.getHash().equals(currentBlock.calculateHash())) {
                System.out.println("Current hash is invalid");
                return false;
            }
            
            // Check if the current block's previousHash matches the previous block's hash
            if (!currentBlock.getPreviousHash().equals(previousBlock.getHash())) {
                System.out.println("Previous hash is invalid");
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Get the entire blockchain
     * 
     * @return List of blocks in the blockchain
     */
    public List<Block> getBlockchain() {
        return new ArrayList<>(blockchain);
    }
    
    /**
     * Get the number of blocks in the blockchain
     * 
     * @return Number of blocks
     */
    public int getBlockchainSize() {
        return blockchain.size();
    }
    
    /**
     * Get the list of pending transactions
     * 
     * @return List of pending transactions
     */
    public List<TaskTransaction> getPendingTransactions() {
        return new ArrayList<>(pendingTransactions);
    }
    
    /**
     * Get the number of pending transactions
     * 
     * @return Number of pending transactions
     */
    public int getPendingTransactionsCount() {
        return pendingTransactions.size();
    }
    
    /**
     * Find all transactions related to a specific task
     * 
     * @param taskId Task ID
     * @return List of transactions related to the task
     */
    public List<TaskTransaction> findTransactionsByTaskId(int taskId) {
        List<TaskTransaction> result = new ArrayList<>();
        
        // Search in all blocks
        for (Block block : blockchain) {
            for (TaskTransaction transaction : block.getTransactions()) {
                if (transaction.getTaskId() == taskId) {
                    result.add(transaction);
                }
            }
        }
        
        // Search in pending transactions
        for (TaskTransaction transaction : pendingTransactions) {
            if (transaction.getTaskId() == taskId) {
                result.add(transaction);
            }
        }
        
        return result;
    }
}
