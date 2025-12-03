package com.example.data;

import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;

/*Data Analyzer cho Heart Disease Dataset
 Phân tích: class distribution, attributes, correlations, feature importance
 */
public class Analyzer {
    /*
     Phân tích tổng quan dataset
     @param data Dataset cần phân tích
     */
    public void analyzeData(Instances data) {
        System.out.println(" DATA ANALYSIS REPORT");
        //Basic Information
        printBasicInfo(data);
        //Class Distribution
        analyzeClassDistribution(data);
        //Attribute Statistics
        analyzeAttributes(data);
        // Missing Values Analysis
        analyzeMissingValues(data);
    }
    
    private void printBasicInfo(Instances data) {
        System.out.println("\n--- Dataset Overview ---");
        System.out.println("Total instances: " + data.numInstances());
        System.out.println("Total attributes: " + data.numAttributes());
        System.out.println("Class attribute: " + data.classAttribute().name());
        System.out.println("Class values: " + data.classAttribute().numValues());
        System.out.println("Relation name: " + data.relationName());
    }
    
    private void analyzeClassDistribution(Instances data) {
        System.out.println("\nClass Distribution");
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        if (counts == null || counts.length == 0) {
            System.out.println("Warning: No class distribution available");
            return;
        }
        //Binary classification: 0 vs 1-4
        int class0 = counts[0];
        int classDisease = 0;
        for (int i = 1; i < counts.length; i++) {
            classDisease += counts[i];
        }
        int total = data.numInstances();
        System.out.println("\n  Class 0 (No Disease):");
        System.out.printf("    Count: %d (%.1f%%)%n", 
            class0, class0 * 100.0 / total);
        System.out.println("\n  Class 1-4 (Disease Present):");
        System.out.printf("    Count: %d (%.1f%%)%n", 
            classDisease, classDisease * 100.0 / total);
        //Check imbalance
        double ratio = Math.max(class0, classDisease) / (double) Math.min(class0, classDisease);
        System.out.printf("\nImbalance Ratio: %.2f%n", ratio);
        if (ratio > 1.5) {
            System.out.println("IMBALANCED dataset detected!");
            System.out.println("Recommendation: Apply SMOTE or class weights");
        } else {
            System.out.println("Relatively balanced dataset");
        }
        //Detailed class breakdown (if multi-class)
        if (counts.length > 2) {
            System.out.println("\nDetailed Class Breakdown:");
            for (int i = 0; i < counts.length; i++) {
                System.out.printf("    Class %d: %d (%.1f%%)%n", 
                    i, counts[i], counts[i] * 100.0 / total);
            }
        }
    }
    

    private void analyzeAttributes(Instances data) {
        System.out.println("\nAttribute Statistics");
        int numericCount = 0;
        int nominalCount = 0;
        System.out.println("\nNumeric Attributes:");
        System.out.println(String.format("%-20s %10s %10s %10s %10s", 
            "Attribute", "Min", "Max", "Mean", "StdDev"));
        System.out.println("-".repeat(70));
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr.isNumeric() && i != data.classIndex()) {
                AttributeStats stats = data.attributeStats(i);
                System.out.printf("%-20s %10.2f %10.2f %10.2f %10.2f%n",
                    truncate(attr.name(), 20),
                    stats.numericStats.min,
                    stats.numericStats.max,
                    stats.numericStats.mean,
                    stats.numericStats.stdDev);
                numericCount++;
            } else if (attr.isNominal() && i != data.classIndex()) {
                nominalCount++;
            }
        }
        System.out.println("\nAttribute Type Summary:");
        System.out.println(" Numeric attributes: " + numericCount);
        System.out.println(" Nominal attributes: " + nominalCount);
        System.out.println(" Total (excluding class): " + (numericCount + nominalCount));
    }
    

    private void analyzeMissingValues(Instances data) {
        System.out.println("\nMissing Values Analysis");
        int totalMissing = 0;
        boolean hasMissing = false; 
        for (int i = 0; i < data.numAttributes(); i++) {
            int missing = data.attributeStats(i).missingCount;
            
            if (missing > 0) {
                if (!hasMissing) {
                    System.out.println("\nAttributes with missing values:");
                    hasMissing = true;
                }
                double percentage = missing * 100.0 / data.numInstances();
                System.out.printf("  %-20s: %4d (%.1f%%)%n", 
                    truncate(data.attribute(i).name(), 20), 
                    missing, 
                    percentage);
                
                totalMissing += missing;
            }
        }
        if (!hasMissing) {
            System.out.println("  No missing values detected");
        } else {
            System.out.println("\n Total missing values: " + totalMissing);
            double totalPercentage = totalMissing * 100.0 / 
                (data.numInstances() * data.numAttributes());
            System.out.printf(" Overall missing rate: %.2f%%%n", totalPercentage);
        }
    }
    
    /*
     Feature Importance Analysis using Information Gain
     @param data Dataset
     @throws Exception
     */
//    public void featureImportance(Instances data) throws Exception {
//        System.out.println("FEATURE IMPORTANCE ANALYSIS");
//        //Information Gain
//        analyzeInfoGain(data);
//        //Gain Ratio
//        analyzeGainRatio(data);
//        System.out.println("\n Interpretation:");
//        System.out.println("   -Higher score = More important for prediction");
//        System.out.println("   -Focus on top 5-10 features for model training");
//        System.out.println("   -Consider removing features with score < 0.01");
//    }
    
//    private void analyzeInfoGain(Instances data) throws Exception {
//        System.out.println("\nInformation Gain Ranking");
//        InfoGainAttributeEval eval = new InfoGainAttributeEval();
//        eval.buildEvaluator(data);
//        //Calculate scores
//        double[][] scores = new double[data.numAttributes() - 1][2];
//        for (int i = 0; i < data.numAttributes() - 1; i++) {
//            scores[i][0] = i; // attribute index
//            scores[i][1] = eval.evaluateAttribute(i);
//        }
//        //Sort by score (descending)
//        java.util.Arrays.sort(scores, (a, b) -> Double.compare(b[1], a[1]));
//        //Print top features
//        System.out.println("\nTop 10 Most Important Features:");
//        System.out.println(String.format("%-4s %-25s %12s", "Rank", "Attribute", "InfoGain"));
//        System.out.println("-".repeat(45));
//        for (int i = 0; i < Math.min(10, scores.length); i++) {
//            int attrIndex = (int) scores[i][0];
//            System.out.printf("%-4d %-25s %12.4f%n",
//                i + 1,
//                truncate(data.attribute(attrIndex).name(), 25),
//                scores[i][1]);
//        }
//        //Print least important (bottom 5)
//        if (scores.length > 10) {
//            System.out.println("\nLeast Important Features (Bottom 5):");
//            for (int i = Math.max(0, scores.length - 5); i < scores.length; i++) {
//                int attrIndex = (int) scores[i][0];
//                System.out.printf("%-4d %-25s %12.4f%n",
//                    i + 1,
//                    truncate(data.attribute(attrIndex).name(), 25),
//                    scores[i][1]);
//            }
//        }
//    }
//
//
//    private void analyzeGainRatio(Instances data) throws Exception {
//        System.out.println("\nGain Ratio Ranking (Top 5):");
//        GainRatioAttributeEval eval = new GainRatioAttributeEval();
//        eval.buildEvaluator(data);
//        //Calculate scores
//        double[][] scores = new double[data.numAttributes() - 1][2];
//        for (int i = 0; i < data.numAttributes() - 1; i++) {
//            scores[i][0] = i;
//            scores[i][1] = eval.evaluateAttribute(i);
//        }
//        //Sort
//        java.util.Arrays.sort(scores, (a, b) -> Double.compare(b[1], a[1]));
//        System.out.println(String.format("%-4s %-25s %12s", "Rank", "Attribute", "GainRatio"));
//        System.out.println("-".repeat(45));
//        for (int i = 0; i < Math.min(5, scores.length); i++) {
//            int attrIndex = (int) scores[i][0];
//            System.out.printf("%-4d %-25s %12.4f%n",
//                i + 1,
//                truncate(data.attribute(attrIndex).name(), 25),
//                scores[i][1]);
//        }
//    }
//

    /**
     * Helper: Truncate string to max length
     */
    private String truncate(String str, int maxLength) {
        if (str.length() <= maxLength) {
            return str;
        }
        return str.substring(0, maxLength - 3) + "...";
    }
    
    /**
     * Test method
     */
    public static void main(String[] args) {
        try {
            System.out.println("Data Analyzer Test\n");
            
            // Load sample data
            weka.core.converters.ConverterUtils.DataSource source = 
                new weka.core.converters.ConverterUtils.DataSource(
                    "src/main/resources/heart_disease.csv");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            
            // Analyze
            Analyzer analyzer = new Analyzer();
            analyzer.analyzeData(data);
//            analyzer.featureImportance(data);
            
            System.out.println("\nAnalysis completed!");
            
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}