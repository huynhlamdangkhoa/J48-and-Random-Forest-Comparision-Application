package com.example.controllers;

import com.example.algorithms.J48Classifier;
import com.example.algorithms.RandomForestClassifier;
import com.example.data.Analyzer;
import com.example.data.Cleaner;
import com.example.data.Loader;
import com.example.data.FeatureEngineer;
import com.example.evaluation.ModelEvaluator;



import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

//Main controller cho Data Mining Pipeline
//Quản lý toàn bộ quy trình từ preprocessing đến model evaluation

public class MiningController {
    private final Cleaner cleaner = new Cleaner();
    private final Analyzer analyzer = new Analyzer();
    private final Loader loader = new Loader();
    private final FeatureEngineer engineer = new FeatureEngineer();
    private final ModelEvaluator evaluator = new ModelEvaluator();

    /*
    Chạy toàn bộ pipeline: Preprocessing → Training → Evaluation 
    @param rawPath Đường dẫn file dataset gốc (.csv hoặc .arff)
    @param reportPath Đường dẫn file báo cáo kết quả
    @throws Exception Lỗi trong quá trình xử lý
     */
    public void runPipeline(String rawPath, String reportPath) throws Exception {
        printHeader("\nEVALUATE MODELS WITH RAW DATA FOR COMPARE LATER WITH THE MODELS THAT ARE EVALUATED WITH PREPROCESSED DATA");
        Instances data = loader.loadDataset(rawPath);
        J48Classifier j48Raw = new J48Classifier();
        j48Raw.train(data);
        Instances j48RawData = j48Raw.getTrainingData();
        evaluator.evaluateModel(j48Raw.getClassifier(), j48RawData, reportPath);

//        RandomForestClassifier rfRaw = new RandomForestClassifier();
//        rfRaw.train(data);
//        Instances rfRawData = rfRaw.getTrainingData();
//        evaluator.evaluateModel(rfRaw.getClassifier(), rfRawData, reportPath);

        printHeader("HEART DISEASE RISK PREDICTOR - DATA MINING PIPELINE");
        printSectionHeader("STEP 1: DATA PREPROCESSING");
        //Load Dataset
        System.out.println("\nLoading dataset...");
//        Instances data = loader.loadDataset(rawPath);
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
//        //Normalize Data
//        System.out.println("\nNormalizing data...");
//        data = cleaner.normalize(data);
        //Data Analysis
        System.out.println("\nAnalyzing dataset...");
        analyzer.analyzeData(data);
        //Feature Importance Analysis
//        System.out.println("\nCalculating feature importance...");
//        analyzer.featureImportance(data);
        //Save Preprocessed Data
        System.out.println("\nSaving preprocessed data...");
        String cleanedPath = "output/heart_data_cleaned.arff";
        loader.saveARFF(data, cleanedPath);

        System.out.println("\nSTEP 1 COMPLETED: Data preprocessing finished!");
        System.out.println("   Preprocessed data saved to: " + cleanedPath);

        printSectionHeader("STEP 2: J48 CLASSIFIER");

        J48Classifier customJ48 = new J48Classifier();
        customJ48.train(data); // data ở đây đã qua tất cả bước preprocess
        Instances j48ReadyData = customJ48.getTrainingData();
        evaluator.evaluateModel(customJ48.getClassifier(), j48ReadyData, reportPath); // true = preprocessed

        evaluator.compareJ48BeforeAfterPreprocessingComparision(reportPath);

        System.out.println("\nSTEP 2 COMPLETED: Custom J48 pipeline evaluated!");
        
        printSectionHeader("STEP 3: RANDOM FOREST CLASSIFIER");

        System.out.println("\nApplying SMOTE for class balancing...");
        Instances balancedData = cleaner.applySMOTE(data);

        // Feature Selection
        System.out.println("\nPerforming feature selection...");
        Instances selectedData = cleaner.selectFeatures(balancedData);

        // (optional safety) make sure class attribute is set
        if (selectedData.classIndex() < 0) {
            selectedData.setClassIndex(selectedData.numAttributes() - 1);
        }

        // Save improved dataset
        String improvedPath = "output/heart_data_improved.arff";
        loader.saveARFF(selectedData, improvedPath);
        System.out.println("   Improved data saved to: " + improvedPath);

        // Train Random Forest on the *improved* data
        RandomForest rf = new RandomForest();
        rf.setNumIterations(200);
        rf.setSeed(1);
        evaluator.evaluateModel(rf, selectedData, reportPath);

        evaluator.compareRFBeforeAfterPreprocessingComparision(reportPath);

        System.out.println("\nSTEP 3 COMPLETED: Random Forest improved model evaluated!");

        printSectionHeader("STEP 4: MODEL COMPARISON");
        
        System.out.println("\nGenerating comparison report...");
        evaluator.compareModels(reportPath);

        
        // Final Summary
        printFinalSummary(reportPath);
    }
    
    /**
     * Explore dataset - In thông tin tổng quan
     */
    private void exploreDataset(Instances data) {
        System.out.println("\n--- Dataset Overview ---");
        System.out.println("Total instances: " + data.numInstances());
        System.out.println("Total attributes: " + data.numAttributes());
        System.out.println("Class attribute: " + data.classAttribute().name());
        System.out.println("Class values: " + data.classAttribute().numValues());
        
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
        System.out.println("PIPELINE COMPLETED SUCCESSFULLY!");
        System.out.println("=".repeat(60));
        System.out.println("\nReports generated:");
        System.out.println("   • Evaluation report: " + reportPath);
    }


    
    /**
     * Main method để chạy pipeline
     */
    public static void main(String[] args) {
        try {
            MiningController controller = new MiningController();
            
            // Đường dẫn files
            String rawDataPath = "src/main/resources/heart_disease.csv";
            String reportPath = "output/evaluation_report.txt";
            
            // Chạy pipeline
            controller.runPipeline(rawDataPath, reportPath);
            
        } catch (Exception e) {
            System.err.println("\nERROR: Pipeline failed!");
            System.err.println("Error message: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
