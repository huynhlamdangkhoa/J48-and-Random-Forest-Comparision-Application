package com.example.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.InputStream;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;


public class DataLoader {
    
    /*
    Load dataset từ file path (CSV hoặc ARFF)
    Auto-detect format và set class index
    @param filepath Đường dẫn đến file dataset
    @return Instances with class index set
    @throws Exception
     */
    public Instances loadDataset(String filepath) throws Exception {
        System.out.println("\n=== Loading Dataset ===");
        System.out.println("File path: " + filepath);
        Instances data = null;
        //Auto-detect file type
        if (filepath.toLowerCase().endsWith(".csv")) {
            data = loadCSV(filepath);
        } else if (filepath.toLowerCase().endsWith(".arff")) {
            data = loadARFF(filepath);
        } else {
            //Try DataSource (support multiple formats)
            DataSource source = new DataSource(filepath);
            data = source.getDataSet();
        }
        if (data == null) {
            throw new Exception("Failed to load dataset from: " + filepath);
        }
        //Set class index (last attribute by default)
        if (data.classIndex() == -1) {
            data = setClassAttribute(data);
        }
        System.out.println("✓ Loaded successfully");
        System.out.println("  Instances: " + data.numInstances());
        System.out.println("  Attributes: " + data.numAttributes());
        System.out.println("  Class: " + data.classAttribute().name());
        return data;
    }
    
    private Instances loadCSV(String filepath) throws Exception {
        System.out.println("  Format: CSV"); 
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filepath));
        Instances data = loader.getDataSet();
        return data;
    }
    
    private Instances loadARFF(String filepath) throws Exception {
        System.out.println("  Format: ARFF");
        
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        Instances data = new Instances(reader);
        reader.close();
        
        return data;
    }

    public Instances loadDatasetFromResources(String resourceName) throws Exception {
        System.out.println("\n=== Loading from Resources ===");
        System.out.println("Resource: " + resourceName);
        InputStream inputStream = getClass().getClassLoader().getResourceAsStream(resourceName);
        if (inputStream == null) {
            throw new FileNotFoundException("Dataset not found in resources: " + resourceName);
        }
        ArffLoader loader = new ArffLoader();
        loader.setSource(inputStream);
        Instances data = loader.getDataSet();
        
        if (data.classIndex() == -1) {
            data = setClassAttribute(data);
        }
        System.out.println("✓ Loaded from resources");
        return data;
    }
    
    private Instances setClassAttribute(Instances data) throws Exception {
        //Common class attribute names cho Heart Disease
        String[] possibleClassNames = {
            "num",           //UCI Heart Disease dataset
            "target",        //Kaggle versions
            "heart_disease",
            "diagnosis",
            "disease",
            "condition",
            "class"
        };
        //Try to find class attribute by name
        for (String className : possibleClassNames) {
            if (data.attribute(className) != null) {
                int idx = data.attribute(className).index();
                data.setClassIndex(idx);
                System.out.println("  ✓ Class attribute detected: " + className);
                // Convert to nominal if numeric
                if (data.classAttribute().isNumeric()) {
                    data = convertClassToNominal(data);
                }
                return data;
            }
        }
        //If not found, use last attribute as class (default)
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("  ℹ Using last attribute as class: " + 
            data.classAttribute().name());
        //Convert to nominal if needed
        if (data.classAttribute().isNumeric()) {
            data = convertClassToNominal(data);
        }
        
        return data;
    }
    
    private Instances convertClassToNominal(Instances data) throws Exception {
        System.out.println("  ℹ Converting class from numeric to nominal...");
        
        NumericToNominal filter = new NumericToNominal();
        filter.setInputFormat(data);
        filter.setAttributeIndices("" + (data.classIndex() + 1));
        data = Filter.useFilter(data, filter);
        
        System.out.println("  ✓ Class converted to nominal");
        
        return data;
    }
    

    public void saveARFF(Instances data, String filepath) throws Exception {
        System.out.println("\n=== Saving ARFF ===");
        System.out.println("Path: " + filepath);
        //Create parent directories if needed
        File file = new File(filepath);
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
            System.out.println("  Created directory: " + parentDir.getPath());
        }
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(file);
        saver.writeBatch();
        System.out.println("  Instances: " + data.numInstances());
        System.out.println("  Attributes: " + data.numAttributes());
    }
    
    public void saveCSV(Instances data, String filepath) throws Exception {
        System.out.println("\nSaving CSV");
        System.out.println("Path: " + filepath);
        weka.core.converters.CSVSaver saver = new weka.core.converters.CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(filepath));
        saver.writeBatch();
    }
    
    public Instances loadCSVDataset(String csvPath, String arffPath) throws Exception {
        System.out.println("\n=== Loading CSV & Converting to ARFF ===");
        //Load CSV
        Instances data = loadCSV(csvPath);
        //Set class index
        data = setClassAttribute(data);
        //Save to ARFF
        saveARFF(data, arffPath); 
        return data;
    }
    

    public void printDatasetInfo(Instances data) {
        System.out.println("\n--- Dataset Information ---");
        System.out.println("Relation: " + data.relationName());
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());
        System.out.println("Class: " + data.classAttribute().name() + 
            " (index: " + data.classIndex() + ")");
        System.out.println("Class type: " + 
            (data.classAttribute().isNumeric() ? "Numeric" : "Nominal"));
        if (data.classAttribute().isNominal()) {
            System.out.println("Class values: " + data.classAttribute().numValues());
        }
    }
    

    public boolean validateDataset(Instances data) {
        System.out.println("\nValidating Dataset");
        
        boolean valid = true;
        
        //Check if empty
        if (data.numInstances() == 0) {
            System.err.println("Error: Dataset is empty!");
            return false;
        }
        //check if class is set
        if (data.classIndex() == -1) {
            System.err.println("Error: Class attribute not set!");
            return false;
        }
        //check for all missing values
        boolean hasData = false;
        for (int i = 0; i < data.numInstances(); i++) {
            if (!data.instance(i).hasMissingValue()) {
                hasData = true;
                break;
            }
        }
        if (!hasData) {
            System.err.println("⚠️  Warning: All instances have missing values!");
        }
        System.out.println("✓ Dataset validation passed");
        return valid;
    }
    
    public static void main(String[] args) {
        try {
            System.out.println("=== DataLoader Test ===\n");
            DataLoader loader = new DataLoader();
            //Test loading
            Instances data = loader.loadDataset("src/resources/heart_disease.csv");
            //Print info
            loader.printDatasetInfo(data);
            //Validate
            loader.validateDataset(data);
            //Test saving
            loader.saveARFF(data, "src/resources/test_output.arff");

            System.out.println("\nAll tests passed!");
            
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}