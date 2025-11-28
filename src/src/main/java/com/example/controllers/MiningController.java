package com.example.controllers;

import com.example.data.DataAnalyzer;
import com.example.data.DataCleaner;
import com.example.data.DataLoader;
import com.example.data.FeatureEngineer;
import com.example.evaluation.ModelEvaluator;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

//Main controller cho Data Mining Pipeline
//Quản lý toàn bộ quy trình từ preprocessing đến model evaluation

public class MiningController {
    private final DataCleaner cleaner = new DataCleaner();
    private final DataAnalyzer analyzer = new DataAnalyzer();
    private final DataLoader loader = new DataLoader();
    private final FeatureEngineer engineer = new FeatureEngineer();
    private final ModelEvaluator evaluator = new ModelEvaluator();

    /*
    Chạy toàn bộ pipeline: Preprocessing → Training → Evaluation 
    @param rawPath Đường dẫn file dataset gốc (.csv hoặc .arff)
    @param reportPath Đường dẫn file báo cáo kết quả
     @throws Exception Lỗi trong quá trình xử lý
     */
    public void runPipeline(String rawPath, String reportPath) throws Exception {
        printHeader("HEART DISEASE RISK PREDICTOR - DATA MINING PIPELINE");
        printSectionHeader("STEP 1: DATA PREPROCESSING");
        //Load Dataset
        System.out.println("\nLoading dataset...");
        Instances data = loader.loadDataset(rawPath);
        exploreDataset(data);
        //Handle Missing Values & Remove Duplicates
        System.out.println("\nCleaning data...");
        data = cleaner.cleanData(data);
        //Remove Outliers
        System.out.println("\nRemoving outliers...");
        data = cleaner.removeOutliers(data);
        //Feature Engineering
        System.out.println("\nEngineering features...");
        data = engineer.createFeatures(data);
        //Normalize Data
        System.out.println("\nNormalizing data...");
        data = cleaner.normalize(data);
        //Data Analysis
        System.out.println("\nAnalyzing dataset...");
        analyzer.analyzeData(data);
        //Feature Importance Analysis
        System.out.println("\nCalculating feature importance...");
        analyzer.featureImportance(data);
        //Save Preprocessed Data
        System.out.println("\nSaving preprocessed data...");
        String cleanedPath = "src/resources/data_cleaned.arff";
        loader.saveARFF(data, cleanedPath);

        System.out.println("\nSTEP 1 COMPLETED: Data preprocessing finished!");
        System.out.println("   Preprocessed data saved to: " + cleanedPath);
        printSectionHeader("STEP 2: J48 DECISION TREE - BASELINE MODEL");
        weka.classifiers.trees.J48 j48 = new weka.classifiers.trees.J48();
        j48.setConfidenceFactor(0.25f);
        j48.setMinNumObj(2);
        j48.setUnpruned(false);
        evaluator.evaluateModel(j48, data, reportPath);
        System.out.println("\nSTEP 2 COMPLETED: J48 baseline model evaluated!");
        printSectionHeader("STEP 3: RANDOM FOREST - IMPROVED MODEL");
        System.out.println("\nApplying SMOTE for class balancing...");
        Instances balancedData = cleaner.applySMOTE(data);
        //Feature Selection
        System.out.println("\nPerforming feature selection...");
        Instances selectedData = cleaner.selectFeatures(balancedData);
        //Save improved dataset
        String improvedPath = "src/resources/data_improved.arff";
        loader.saveARFF(selectedData, improvedPath);
        System.out.println("   Improved data saved to: " + improvedPath);
        
        // Train Random Forest
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setSeed(1);
        evaluator.evaluateModel(rf, data, reportPath);
        
        System.out.println("\n✅ STEP 3 COMPLETED: Random Forest improved model evaluated!");
        
        // ========================================
        // STEP 4: MODEL COMPARISON
        // ========================================
        printSectionHeader("STEP 4: MODEL COMPARISON & FINAL REPORT");
        
        System.out.println("\n📈 Generating comparison report...");
        evaluator.compareModels(reportPath);
        
        // Final Summary
        printFinalSummary(reportPath);
    }
    
    /**
     * Explore dataset - In thông tin tổng quan
     */
    private void exploreDataset(Instances data) {
        System.out.println("\n--- Dataset Overview ---");
        System.out.println("📁 Total instances: " + data.numInstances());
        System.out.println("📊 Total attributes: " + data.numAttributes());
        System.out.println("🎯 Class attribute: " + data.classAttribute().name());
        System.out.println("📋 Class values: " + data.classAttribute().numValues());
        
        System.out.println("\n--- Attributes List ---");
        for (int i = 0; i < Math.min(10, data.numAttributes()); i++) {
            String type = data.attribute(i).isNumeric() ? "Numeric" : "Nominal";
            System.out.printf("  %2d. %-20s [%s]\n", 
                i + 1, data.attribute(i).name(), type);
        }
        if (data.numAttributes() > 10) {
            System.out.println("  ... and " + (data.numAttributes() - 10) + " more");
        }
    }
    
    /**
     * Print header cho toàn bộ pipeline
     */
    private void printHeader(String title) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("  " + title);
        System.out.println("=".repeat(60));
    }
    
    /**
     * Print header cho từng section
     */
    private void printSectionHeader(String title) {
        System.out.println("\n\n" + "═".repeat(60));
        System.out.println("  " + title);
        System.out.println("═".repeat(60));
    }
    
    /**
     * Print final summary
     */
    private void printFinalSummary(String reportPath) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("  🎉 PIPELINE COMPLETED SUCCESSFULLY!");
        System.out.println("=".repeat(60));
        System.out.println("\n📄 Reports generated:");
        System.out.println("   • Evaluation report: " + reportPath);
        System.out.println("   • Preprocessed data: src/resources/data_cleaned.arff");
        System.out.println("   • Improved data: src/resources/data_improved.arff");
        System.out.println("\n📊 Models trained:");
        System.out.println("   • J48 Decision Tree (Baseline)");
        System.out.println("   • Random Forest + SMOTE + Feature Selection (Improved)");
        System.out.println("\n💡 Next steps:");
        System.out.println("   1. Review evaluation_report.txt for detailed metrics");
        System.out.println("   2. Analyze confusion matrices and ROC curves");
        System.out.println("   3. Compare model performance for clinical deployment");
        System.out.println("\n" + "=".repeat(60) + "\n");
    }
    
    /**
     * Main method để chạy pipeline
     */
    public static void main(String[] args) {
        try {
            MiningController controller = new MiningController();
            
            // Đường dẫn files
            String rawDataPath = "src/resources/heart_disease.csv";
            String reportPath = "src/resources/evaluation_report.txt";
            
            // Chạy pipeline
            controller.runPipeline(rawDataPath, reportPath);
            
        } catch (Exception e) {
            System.err.println("\n❌ ERROR: Pipeline failed!");
            System.err.println("Error message: " + e.getMessage());
            e.printStackTrace();
        }
    }
}