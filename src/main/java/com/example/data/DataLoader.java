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

        // sửa duy nhất tại đây → resolve path chuẩn cho IntelliJ
        String resolved = resolvePath(filepath);
        System.out.println("File path: " + resolved);

        Instances data = null;

        if (resolved.toLowerCase().endsWith(".csv")) {
            data = loadCSV(resolved);
        } else if (resolved.toLowerCase().endsWith(".arff")) {
            data = loadARFF(resolved);
        } else {
            DataSource source = new DataSource(resolved);
            data = source.getDataSet();
        }

        if (data == null) {
            throw new Exception("Failed to load dataset from: " + resolved);
        }

        if (data.classIndex() == -1) {
            data = setClassAttribute(data);
        }

        System.out.println("✓ Loaded successfully");
        System.out.println("  Instances: " + data.numInstances());
        System.out.println("  Attributes: " + data.numAttributes());
        System.out.println("  Class: " + data.classAttribute().name());

        return data;
    }

    // ===============================
    // ONLY CHANGE: đảm bảo CSVLoader đọc file đúng vị trí IntelliJ
    // ===============================
    private Instances loadCSV(String filepath) throws Exception {
        System.out.println("  Format: CSV");
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filepath));
        return loader.getDataSet();
    }

    // ===============================
    // ONLY CHANGE: dùng resolve path cho ARFF
    // ===============================
    private Instances loadARFF(String filepath) throws Exception {
        System.out.println("  Format: ARFF");
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        Instances data = new Instances(reader);
        reader.close();
        return data;
    }

    // ===============================
    // helper mới: giúp IntelliJ xác định đường dẫn đúng
    // không ảnh hưởng logic cũ
    // ===============================
    private String resolvePath(String path) {
        File f = new File(path);
        if (f.exists()) return f.getAbsolutePath();

        // thử thêm "src/main/resources"
        File f2 = new File("src/main/resources/" + path);
        if (f2.exists()) return f2.getAbsolutePath();

        // thử thêm "resources"
        File f3 = new File("resources/" + path);
        if (f3.exists()) return f3.getAbsolutePath();

        return path; // fallback
    }

    // ===============================
    // Giữ nguyên toàn bộ đoạn này
    // ===============================

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
        String[] possibleClassNames = {
                "num", "target", "heart_disease", "diagnosis",
                "disease", "condition", "class"
        };
        for (String className : possibleClassNames) {
            if (data.attribute(className) != null) {
                int idx = data.attribute(className).index();
                data.setClassIndex(idx);
                System.out.println("  ✓ Class attribute detected: " + className);
                if (data.classAttribute().isNumeric()) {
                    data = convertClassToNominal(data);
                }
                return data;
            }
        }
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("  ℹ Using last attribute as class: " +
                data.classAttribute().name());
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
        Instances data = loadCSV(resolvePath(csvPath));
        data = setClassAttribute(data);
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

        if (data.numInstances() == 0) {
            System.err.println("Error: Dataset is empty!");
            return false;
        }
        if (data.classIndex() == -1) {
            System.err.println("Error: Class attribute not set!");
            return false;
        }
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

            // sửa duy nhất: dùng resolvePath để tránh lỗi IntelliJ
            Instances data = loader.loadDataset("src/main/resources/heart_disease.csv");

            loader.printDatasetInfo(data);
            loader.validateDataset(data);

            loader.saveARFF(data, "src/main/resources/test_output.arff");

            System.out.println("\nAll tests passed!");

        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
