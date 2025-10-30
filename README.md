# 🧬 Disease Variant Prediction with Genomics

### 🔍 Project Overview
This ongoing project aims to build a **machine learning pipeline to predict the clinical significance of genomic variants** using publicly available **ClinVar** and **VCF** (Variant Call Format) data.  
The primary goal is to analyze DNA variant information and identify potential **pathogenic (disease-causing) mutations** to support healthcare and precision medicine research.

---

### ⚙️ Technologies Used
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, scikit-learn, TensorFlow/Keras  
- **Genomic Tools:** cyvcf2 (for parsing VCF files)  
- **Data Source:** ClinVar (NCBI)  
- **Environment:** Anaconda, Jupyter Notebook  

---

### 🧩 Project Workflow
1. **Data Acquisition**
   - Download and parse ClinVar `.vcf.gz` files containing genomic variant details.  
   - Extract metadata such as chromosome, position, reference and alternate alleles, and gene symbol.

2. **Data Preprocessing & Cleaning**
   - Handle missing values and inconsistent genomic attributes.  
   - Map chromosome and clinical significance labels for supervised learning.  

3. **Feature Engineering**
   - Derive biologically meaningful features such as:
     - Chromosome mapping (categorical → numerical)
     - Gene frequency count
     - Allele-type variations  
   - Evaluate importance and correlation of each feature.

4. **Model Development**
   - Train baseline ML models (Random Forest, Decision Tree).  
   - Implement **Neural Network** (Keras) with **EarlyStopping** for better generalization.  
   - Compare traditional ML vs NN performance using classification metrics.

5. **Model Evaluation**
   - Use metrics such as **Precision, Recall, F1-Score**, and **Confusion Matrix**.  
   - Analyze model interpretability and biological relevance.

6. **Next Steps (Ongoing)**
   - Integrate variant effect prediction with external annotation datasets.  
   - Deploy scalable prediction API or web dashboard for genomic interpretation.

---

### 📊 Sample Output
```text
Classification Report (Neural Network with EarlyStopping)

              precision    recall  f1-score   support
           0     0.47       0.77      0.58    253225
           1     0.82       0.43      0.56    387994
           2     0.64       0.66      0.65     88450








1. conda create -n genomics python=3.10 -y
2. conda activate genomics
3. conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn
4. conda install pytorch torchvision torchaudio -c pytorch -c conda-forge
5. conda install -c bioconda cyvcf2 pyfaidx biopython
6. pip install transformers datasets sentencepiece


bcftools view clinvar_20250923.vcf -Ov -o cleaned.vcf
