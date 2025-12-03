package com.example.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.RemoveDuplicates;


public class Cleaner {
    
    /*
    Clean data: Handle missing values + Remove duplicates
    @param rawData Dataset gốc
    @return Cleaned dataset
    @throws Exception
     */
    public Instances cleanData(Instances rawData) throws Exception {
        System.out.println("\nCleaning Data");
        //Handle missing values
        rawData = handleMissingValues(rawData);
        //Remove duplicates
        rawData = removeDuplicates(rawData);
        System.out.println("Data cleaned successfully");
        return rawData;
    }

    private Instances handleMissingValues(Instances data) throws Exception {
        System.out.println("\nHandling missing values...");
        // Count missing values trước khi xử lý
        int totalMissing = countMissingValues(data);
        
        if (totalMissing > 0) {
            System.out.println("Found " + totalMissing + " missing values");
            
            // Xử lý từng attribute
            for (int i = 0; i < data.numAttributes(); i++) {
                Attribute attr = data.attribute(i);
                
                if (attr.isNumeric()) {
                    // Tính median cho numeric attributes
                    double[] values = data.attributeToDoubleArray(i);
                    double median = calculateMedian(values);
                    
                    // Replace missing với median
                    for (int j = 0; j < data.numInstances(); j++) {
                        if (data.instance(j).isMissing(i)) {
                            data.instance(j).setValue(i, median);
                        }
                    }
                    System.out.println("  " + attr.name() + ": replaced with median = " + median);
                    
                } else if (attr.isNominal()) {
                    // Tính mode cho nominal attributes
                    int[] counts = new int[attr.numValues()];
                    
                    for (int j = 0; j < data.numInstances(); j++) {
                        if (!data.instance(j).isMissing(i)) {
                            counts[(int) data.instance(j).value(i)]++;
                        }
                    }
                    
                    // Tìm mode
                    int modeIndex = 0;
                    for (int k = 1; k < counts.length; k++) {
                        if (counts[k] > counts[modeIndex]) {
                            modeIndex = k;
                        }
                    }
                    
                    // Replace missing với mode
                    for (int j = 0; j < data.numInstances(); j++) {
                        if (data.instance(j).isMissing(i)) {
                            data.instance(j).setValue(i, modeIndex);
                        }
                    }
                    System.out.println("  " + attr.name() + ": replaced with mode = " + attr.value(modeIndex));
                }
            }
            
            System.out.println("Completed: Replaced with median (numeric) / mode (nominal)");
        } else {
            System.out.println("No missing values found");
        }
        
        return data;
    }

    // Hàm tính median
    private double calculateMedian(double[] values) {
        // Lọc bỏ missing values (NaN)
        List<Double> validValues = new ArrayList<>();
        for (double v : values) {
            if (!Double.isNaN(v)) {
                validValues.add(v);
            }
        }
        
        if (validValues.isEmpty()) {
            return 0.0;
        }
        
        Collections.sort(validValues);
        int size = validValues.size();
        
        if (size % 2 == 0) {
            return (validValues.get(size/2 - 1) + validValues.get(size/2)) / 2.0;
        } else {
            return validValues.get(size/2);
        }
    }
    
    private Instances removeDuplicates(Instances data) throws Exception {
        System.out.println("\nRemoving duplicates...");
        int beforeCount = data.numInstances();
        
        RemoveDuplicates filter = new RemoveDuplicates();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        int removed = beforeCount - data.numInstances();
        if (removed > 0) {
            System.out.println("  Removed: " + removed + " duplicates");
        } else {
            System.out.println("No duplicates found");
        } 
        System.out.println("Remaining instances: " + data.numInstances());
        
        return data;
    }
    
    /*
    Remove outliers using IQR method
    @param data Dataset
    @return Dataset without outliers
    @throws Exception
     */
    public Instances removeOutliers(Instances data) throws Exception {
        System.out.println("\nRemoving Outliers");
        int beforeCount = data.numInstances();
        //Manual outlier removal using IQR
        data = removeOutliersIQR(data);
        int removed = beforeCount - data.numInstances();
        System.out.println("  Removed: " + removed + " outliers (" + 
            String.format("%.1f%%", removed * 100.0 / beforeCount) + ")");
        System.out.println("  Remaining: " + data.numInstances() + " instances");
        return data;
    }
    

    private Instances removeOutliersIQR(Instances data) throws Exception {
        Instances cleanData = new Instances(data);
        
        //Iterate through each numeric attribute (except class)
        for (int i = 0; i < cleanData.numAttributes() - 1; i++) {
            if (cleanData.attribute(i).isNumeric()) {
                double[] values = cleanData.attributeToDoubleArray(i);
                //Calculate Q1, Q3, IQR
                java.util.Arrays.sort(values);
                int n = values.length;
                double q1 = values[n / 4];
                double q3 = values[3 * n / 4];
                double iqr = q3 - q1;    
                double lowerBound = q1 - 1.5 * iqr;
                double upperBound = q3 + 1.5 * iqr;
                //Remove instances with outliers
                for (int j = cleanData.numInstances() - 1; j >= 0; j--) {
                    double value = cleanData.instance(j).value(i);
                    if (value < lowerBound || value > upperBound) {
                        cleanData.delete(j);
                    }
                }
            }
        }
        return cleanData;
    }
    
    /*
    Normalize numeric attributes to [0,1] range
    @param data Dataset
    @return Normalized dataset
    @throws Exception
     */
    public Instances normalize(Instances data) throws Exception {
        System.out.println("\nNormalizing Data");
        Normalize filter = new Normalize();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        System.out.println(" All numeric attributes normalized to [0, 1]");
        return data;
    }
    
    /*
    Apply SMOTE (Synthetic Minority Over-sampling Technique)
    Để cân bằng class distribution
    @param data Dataset
    @return Balanced dataset
    @throws Exception
     */
    public Instances applySMOTE(Instances data) throws Exception {
        System.out.println("\nApplying SMOTE");
        //Check class distribution trước SMOTE
        int[] beforeCounts = getClassCounts(data);
        System.out.println("  Before SMOTE:");
        System.out.println("    Class 0 (No Disease): " + beforeCounts[0]);
        System.out.println("    Class 1 (Disease): " + beforeCounts[1]);
        double ratio = Math.max(beforeCounts[0], beforeCounts[1]) / 
        (double) Math.min(beforeCounts[0], beforeCounts[1]);
        System.out.println("    Imbalance ratio: " + String.format("%.2f", ratio));
        //Apply SMOTE
        SMOTE smote = new SMOTE();
        smote.setInputFormat(data);
        smote.setPercentage(100.0); // Increase minority class by 100%
        smote.setNearestNeighbors(5); // Use 5 nearest neighbors
        Instances balancedData = Filter.useFilter(data, smote);
        //Check sau SMOTE
        int[] afterCounts = getClassCounts(balancedData);
        System.out.println("\n  After SMOTE:");
        System.out.println("    Class 0: " + afterCounts[0]);
        System.out.println("    Class 1: " + afterCounts[1]);
        System.out.println("    Total instances: " + beforeCounts[0] + beforeCounts[1] + 
            " → " + balancedData.numInstances());
        double newRatio = Math.max(afterCounts[0], afterCounts[1]) / 
        (double) Math.min(afterCounts[0], afterCounts[1]);
        System.out.println("New ratio: " + String.format("%.2f", newRatio));
        System.out.println("Class distribution balanced");
        return balancedData;
    }
    
    /*
    Feature Selection using CFS (Correlation-based Feature Selection)
    @param data Dataset
    @return Dataset with selected features
    @throws Exception
     */
    public Instances selectFeatures(Instances data) throws Exception {
        System.out.println("\nFeature Selection (CFS)");
        int originalFeatures = data.numAttributes() - 1; // Exclude class
        //CfsSubsetEval - Correlation-based Feature Selection
        CfsSubsetEval eval = new CfsSubsetEval();
        //GreedyStepwise search
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        //Attribute selection
        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(eval);
        selector.setSearch(search);
        selector.SelectAttributes(data);
        //Get selected attributes
        int[] selectedIndices = selector.selectedAttributes();
        System.out.println("  Original features: " + originalFeatures);
        System.out.println("  Selected features: " + (selectedIndices.length - 1)); // -1 for class
        System.out.println("\n  Selected attributes:");
        for (int i = 0; i < selectedIndices.length - 1; i++) { // -1 to exclude class
        System.out.println("    • " + data.attribute(selectedIndices[i]).name());
        }
        //Reduce dimensionality
        Instances reducedData = selector.reduceDimensionality(data);
        System.out.println("\nFeature selection completed");
        System.out.println("Reduced: " + originalFeatures + " → " + 
        (reducedData.numAttributes() - 1) + " features");
        return reducedData;
    }
    
    private int countMissingValues(Instances data) {
        int count = 0;
        for (int i = 0; i < data.numAttributes(); i++) {
            count += data.attributeStats(i).missingCount;
        }
        return count;
    }
    

    private int[] getClassCounts(Instances data) {
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        //Handle both binary and multi-class
        if (counts == null || counts.length < 2) {
            return new int[]{0, 0};
        }
        int class0 = counts[0];
        int class1 = 0;
        for (int i = 1; i < counts.length; i++) {
            class1 += counts[i];
        }
        return new int[]{class0, class1};
    }
    

    public void printCleaningSummary(Instances before, Instances after) {
        System.out.println("\nCleaning Summary");
        System.out.println("Instances: " + before.numInstances() + " → " + after.numInstances());
        System.out.println("Attributes: " + before.numAttributes() + " → " + after.numAttributes());
        System.out.println("Missing values: " + countMissingValues(before) + " → " + 
            countMissingValues(after));
    }
}