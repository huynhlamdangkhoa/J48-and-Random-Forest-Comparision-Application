<div align="center">
  
# 📈 E-Commerce Ad Boost Classifier
A modular Java + Weka project comparing classification models to predict advertisement boost usage in e-commerce listings

![GitHub repo size](https://img.shields.io/github/repo-size/koitran14/AdBoost-Classifier)
![GitHub contributors](https://img.shields.io/github/contributors/koitran14/AdBoost-Classifier)
![GitHub stars](https://img.shields.io/github/stars/koitran14/AdBoost-Classifier?style=social)
![GitHub forks](https://img.shields.io/github/forks/koitran14/AdBoost-Classifier?style=social)
![GitHub issues](https://img.shields.io/github/issues/koitran14/AdBoost-Classifier)

</div>



## 👥 Contributors

| Name                    | Role         |
| ----------------------- | ------------ |
| **Trần Ngọc Đăng Khôi** | Project Lead |
| Huỳnh Lâm Đăng Khoa     | Team Member  |
| Nguyễn Xuân Trâm Anh    | Team Member  |

---

## 🧠 Overview

> Predict whether a product uses ad boosts based on features like price, ratings, and product text using 3 classic machine learning models.

* **Language**: Java
* **Library**: Weka
* **Dataset**: 1,573 product listings, 43 attributes
* **Target Feature**: `uses_ad_boosts`

---

## ⚙️ Features

* 🔧 Preprocessing: missing value imputation, outlier handling, duplicates removal
* ⚖️ SMOTE: for balancing the imbalanced class
* 🔤 Text Vectorization: StringToWordVector with TF-IDF & n-grams
* 🧪 Feature Selection: CfsSubsetEval, WrapperSubsetEval
* 🤖 Models: J48, Naive Bayes, Random Forest

---

## 🧪 Model Performance

| Model                | Accuracy   | F1-Score  |
| -------------------- | ---------- | --------- |
| 🌳 Random Forest     | **75.11%** | **0.735** |
| 📙 Naive Bayes       | 72.06%     | 0.721     |
| 🌲 J48 Decision Tree | 66.93%     | 0.618     |

> 🏆 **Random Forest** performed the best in overall accuracy and F1-score.

---

## 🚀 How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/koitran14/Data-Mining-Proj.git
cd Data-Mining-Proj
```

2. **Set Up Java and Weka**

* Java JDK 8 or higher
* Download [Weka 3.8+](https://www.cs.waikato.ac.nz/ml/weka/)

3. **Compile and Run**

```bash
javac -cp weka.jar src/**/*.java
java -cp .:weka.jar Main
```

---

## 🗂️ Project Structure

```
Data-Mining-Proj/
├── classifiers/           # J48, NaiveBayes, RandomForest classes
├── preprocessing/         # Text filters, SMOTE, discretizers
├── evaluation/            # Evaluation & ROC metrics
└── Main.java              # Program entry point
```

---

## 🚧 Future Improvements

* 🧬 Ensemble models or hybrid pipelines
* 🗣️ Improve text encoding with embeddings
* 🔍 Add interpretability tools (e.g., SHAP, LIME)

