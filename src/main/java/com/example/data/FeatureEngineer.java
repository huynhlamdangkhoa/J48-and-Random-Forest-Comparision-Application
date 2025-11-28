package com.example.data;

import java.util.Locale;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddExpression;

/*
Feature Engineering cho Heart Disease dataset
Tạo các features mới dựa trên domain knowledge y tế
 */
public class FeatureEngineer {
    /*
    Main method - Tạo tất cả features mới
    @param data Dataset gốc
    @return Dataset với các features mới
    @throws Exception Lỗi khi tạo features
    */
    public Instances createFeatures(Instances data) throws Exception {
        System.out.println("\nFeature Engineering");
        int originalFeatures = data.numAttributes();
        //Age Risk Group
        data = createAgeRiskGroup(data);
        //Cholesterol Category
        data = createCholesterolCategory(data);
        //Blood Pressure Category
        data = createBPCategory(data);
        //Risk Score (Composite feature)
        data = createRiskScore(data);
        
        int newFeatures = data.numAttributes() - originalFeatures;
        System.out.println("✓ Created " + newFeatures + " new features");
        System.out.println("  Total attributes: " + originalFeatures + " → " + data.numAttributes());
        return data;
    }
    
    /**
    -Tạo Age Risk Group
    -Phân nhóm tuổi theo nguy cơ tim mạch:
    -Low risk (<40)
    -Medium risk (40-55)
    -High risk (56-70)
    -Very high risk (>70)
     */
    private Instances createAgeRiskGroup(Instances data) throws Exception {
        System.out.println("\n[1/4] Creating age_risk_group...");
        //Tìm index của age attribute
        Attribute ageAttr = findAttribute(data, "age", "Age");
        if (ageAttr == null) {
            System.err.println("⚠️  Warning: 'age' attribute not found, skipping...");
            return data;
        }

        int ageIdx = ageAttr.index();
        //Tạo expression: ifelse nested
        AddExpression filter = new AddExpression();
        filter.setExpression(
            "ifelse(a" + (ageIdx + 1) + "<40, 1, " +
            "ifelse(a" + (ageIdx + 1) + "<56, 2, " +
            "ifelse(a" + (ageIdx + 1) + "<71, 3, 4)))"
        );
        filter.setName("age_risk_group");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        System.out.println("  ✓ age_risk_group created");
        System.out.println("    1=Low(<40), 2=Medium(40-55), 3=High(56-70), 4=VeryHigh(>70)");
        return data;
    }
    
    /*
    -Tạo Cholesterol Category
    -Phân loại theo American Heart Association (AHA):
    -Normal (<200 mg/dL)
    -Borderline High (200-239 mg/dL)
    -High (≥240 mg/dL)
     */
    private Instances createCholesterolCategory(Instances data) throws Exception {
        System.out.println("\nCreating chol_category...");
        //Tìm cholesterol attribute
        Attribute cholAttr = findAttribute(data, 
            "chol", "cholesterol", "cholesterol_level", "Cholesterol Level", "cholesterol level");
        if (cholAttr == null) {
            System.err.println("Warning: 'cholesterol' attribute not found, skipping...");
            return data;
        }
        int cholIdx = cholAttr.index();
        AddExpression filter = new AddExpression();
        filter.setExpression(
            "ifelse(a" + (cholIdx + 1) + "<200, 1, " +
            "ifelse(a" + (cholIdx + 1) + "<240, 2, 3))"
        );
        filter.setName("chol_category");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        
        System.out.println("  ✓ chol_category created");
        System.out.println("    1=Normal(<200), 2=Borderline(200-239), 3=High(≥240)");
        
        return data;
    }
    
    /*
    Tạo Blood Pressure Category
    Phân loại theo AHA guidelines:
    -Normal (<120 mmHg)
    -Elevated (120-129 mmHg)
    -Stage 1 Hypertension (130-139 mmHg)
    -Stage 2 Hypertension (≥140 mmHg)
     */
    private Instances createBPCategory(Instances data) throws Exception {
        System.out.println("\n[3/4] Creating bp_category...");

        //Tìm blood pressure attribute
        Attribute bpAttr = findAttribute(data, 
            "trestbps", "bp", "blood_pressure", "Blood Pressure", "blood pressure");
        if (bpAttr == null) {
            System.err.println("Warning: blood pressure attribute not found, skipping...");
            return data;
        }
        int bpIdx = bpAttr.index();
        AddExpression filter = new AddExpression();
        filter.setExpression(
            "ifelse(a" + (bpIdx + 1) + "<120, 1, " +
            "ifelse(a" + (bpIdx + 1) + "<130, 2, " +
            "ifelse(a" + (bpIdx + 1) + "<140, 3, 4)))"
        );
        filter.setName("bp_category");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);

        System.out.println(" bp_category created");
        System.out.println("  1=Normal(<120), 2=Elevated(120-129), 3=Stage1(130-139), 4=Stage2(≥140)");
        
        return data;
    }
    
    /*
    Tạo Composite Risk Score
    Kết hợp nhiều yếu tố nguy cơ với trọng số:
    -Age risk: 30%
    -Cholesterol: 30%
    -Blood pressure: 40%
    Score càng cao = nguy cơ càng lớn
     */
    private Instances createRiskScore(Instances data) throws Exception {
        System.out.println("\nCreating composite risk_score...");
        
        //Tìm các features đã tạo
        Attribute ageRiskAttr = data.attribute("age_risk_group");
        Attribute cholCatAttr = data.attribute("chol_category");
        Attribute bpCatAttr = data.attribute("bp_category");
        
        //Kiểm tra xem có đủ features không
        if (ageRiskAttr == null || cholCatAttr == null || bpCatAttr == null) {
            System.err.println("⚠️  Warning: Required features not found, skipping risk_score...");
            return data;
        }
    
        int ageRiskIdx = ageRiskAttr.index();
        int cholCatIdx = cholCatAttr.index();
        int bpCatIdx = bpCatAttr.index();   
        //Weighted composite score
        AddExpression filter = new AddExpression();
        filter.setExpression(
            "(a" + (ageRiskIdx + 1) + "*0.3 + " +
            "a" + (cholCatIdx + 1) + "*0.3 + " +
            "a" + (bpCatIdx + 1) + "*0.4)"
        );
        filter.setName("risk_score");
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        System.out.println("risk_score created");
        System.out.println("    Formula: (age_risk×0.3 + chol_cat×0.3 + bp_cat×0.4)");
        System.out.println("    Range: ~1.0 (low risk) to ~4.0 (very high risk)");
        
        return data;
    }
    
    public void printFeatureStats(Instances data) {
        System.out.println("\nNew Features Summary");
        
        String[] newFeatures = {"age_risk_group", "chol_category", "bp_category", "risk_score"};
        
        for (String featureName : newFeatures) {
            Attribute attr = data.attribute(featureName);
            if (attr != null && attr.isNumeric()) {
                double[] values = data.attributeToDoubleArray(attr.index());
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                double sum = 0;

                for (double v : values) {
                    if (!Double.isNaN(v)) {
                        min = Math.min(min, v);
                        max = Math.max(max, v);
                        sum += v;
                    }
                }

                double mean = sum / values.length;
                System.out.printf("%-20s: min=%.2f, max=%.2f, mean=%.2f\n",
                    featureName, min, max, mean);
            }
        }
    }
    
    public static void main(String[] args) {
        try {
            System.out.println("Feature Engineer Test\n");
            //Load sample data (cần có file test)
            weka.core.converters.ConverterUtils.DataSource source = 
                new weka.core.converters.ConverterUtils.DataSource("src/resources/heart_disease.csv");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            System.out.println("Original attributes: " + data.numAttributes());
            //Apply feature engineering
            FeatureEngineer engineer = new FeatureEngineer();
            Instances newData = engineer.createFeatures(data);
            
            System.out.println("\nNew attributes: " + newData.numAttributes());
            
            //Print stats
            engineer.printFeatureStats(newData);
            
            System.out.println("\nFeature engineering test completed!");
        } catch (Exception e) {
            System.err.println("Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private Attribute findAttribute(Instances data, String... candidates) {
        if (candidates == null) {
            return null;
        }
        for (String candidate : candidates) {
            if (candidate == null) {
                continue;
            }
            Attribute direct = data.attribute(candidate);
            if (direct != null) {
                return direct;
            }
        }
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr == null) {
                continue;
            }
            String normalized = normalize(attr.name());
            for (String candidate : candidates) {
                if (candidate != null && normalized.equals(normalize(candidate))) {
                    return attr;
                }
            }
        }
        return null;
    }

    private String normalize(String name) {
        return name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
    }
}