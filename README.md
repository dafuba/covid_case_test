# COVID-19 Pakistan Dataset – Machine Learning Regression Project

This project explores the **Pakistan COVID-19 synthetic dataset** and builds a machine learning model to **predict daily New Cases** using features such as tests, vaccinations, recoveries, hospitalizations, and province-level data.

The dataset is clean, well-structured, and ideal for practicing:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature encoding
- Correlation and visualization
- Regression modeling
- Evaluation and model interpretation

---

## 📁 Dataset Description

Each row represents a **daily report for a specific province**.  
Columns include:

| Feature | Description |
|---------|-------------|
| Province | Province name (categorical) |
| New_Cases | Reported new COVID-19 cases that day |
| Recoveries | Daily recoveries |
| Deaths | Daily reported deaths |
| Vaccinations | Daily vaccinations given |
| Hospitalized | Active hospitalized cases for that day |
| Tests_Conducted | Number of tests performed |

This dataset was generated synthetically to simulate real COVID-19 trends in Pakistan while remaining privacy-safe.

---

## 🔍 Project Goals

1. Perform full **Exploratory Data Analysis (EDA)**  
2. Visualize feature distributions and relationships  
3. Encode categorical variables (Province → one-hot)  
4. Build a **regression model predicting New Cases**  
5. Evaluate performance using regression metrics  
6. Visualize model performance (Predicted vs Actual)  
7. Plot feature importance  
8. Derive insights from model results

---

## 🧪 Exploratory Data Analysis (EDA)

The notebook performs:

### ✔ Distribution plots  
- New Cases  
- Tests  
- Hospitalizations  
- Vaccinations  
- Recoveries  
- Deaths  

### ✔ Correlation heatmap  
Using only numeric columns to avoid errors.

### ✔ Province-level analysis  
- Average New Cases per province  
- Total vaccinations per province  
- Hospitalization comparisons  

These visualizations help understand feature relationships before modeling.

---

## 🛠 Preprocessing

### ✔ One-hot encoding  
`Province` is converted into dummy variables:

```python
df_encoded = pd.get_dummies(df, columns=["Province"], drop_first=False)
