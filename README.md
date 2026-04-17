# Diabetes Data Visualization

Analyzes diabetes patient data to identify which health factors (glucose, BMI, age, blood pressure, etc.) are most linked to diabetes. Creates correlation heatmaps, distribution charts, and bar graphs to visualize patterns in medical data.

---

## Dataset

Pima Indians Diabetes Dataset from Kaggle. Contains medical data from 768 female patients aged 21+.

| Column | Description | Example |
|--------|-------------|---------|
| Pregnancies | Number of times pregnant | 6 |
| Glucose | Plasma glucose concentration after fasting (mg/dL) | 148 |
| BloodPressure | Diastolic blood pressure (mm Hg) | 72 |
| SkinThickness | Triceps skin fold thickness (mm) | 35 |
| Insulin | 2-hour serum insulin (mu U/ml) | 0 |
| BMI | Body mass index - weight/height² (kg/m²) | 33.6 |
| DiabetesPedigreeFunction | Genetic diabetes likelihood (higher = more risk) | 0.627 |
| Age | Age in years | 50 |
| Outcome | 0 = No diabetes, 1 = Has diabetes | 1 |

---

## Key Findings

| Factor | Correlation | Meaning |
|--------|------------|---------|
| Glucose | 0.47 | Strongest link - higher blood sugar = more diabetes |
| BMI | 0.29 | Moderate - higher weight = more risk |
| Age | 0.24 | Moderate - older people have more diabetes |
| Pregnancies | 0.22 | Weak link |
| Insulin | 0.13 | Weak link |
| BloodPressure | 0.07 | Very weak link |

---

## Visualizations

### 1. Correlation Heatmap
![Correlations with Diabetes](./Correlations%20with%20Diabetes.png)

### 2. Feature Importance Pie Chart
![Correlation Pie Chart](./DiabetesCorrelationPiechart.png)

### 3. Age Group Distribution
![Age Group Distribution](./Diabetes%20Distribution%20Across%20Age%20Groups.png)

### 4. Blood Pressure Distribution
![Blood Pressure Distribution](./Diabetes%20Distribution%20Across%20Blood%20Pressure%20Levels.png)

### 5. BMI Distribution
![BMI Distribution](./Diabetes%20Distribution%20Across%20BMI%20Levels.png)

### 6. Glucose Distribution
![Glucose Distribution](./Diabetes%20Distribution%20Across%20Glucose%20Levels.png)

### 7. Insulin Distribution
![Insulin Distribution](./Diabetes%20Distribution%20Across%20Insulin%20Levels.png)

---

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

---

## Project Structure

```
DiabetesDataViz/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── diabetes.csv
├── *.png
├── source/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── visualizer.py
│   └── analyzer.py
└── notebooks/
    └── Diabetes_Analysis_Workflow.ipynb
```

---

## Files

| File | Purpose |
|------|---------|
| main.py | Runs analysis and generates charts |
| source/data_loader.py | Load and prepare data |
| source/visualizer.py | Create charts |
| source/analyzer.py | Statistical analysis |
| notebooks/*.ipynb | Step-by-step guide |

---

## Tips

1. Run `python main.py` to generate all charts
2. Open `notebooks/Diabetes_Analysis_Workflow.ipynb` for interactive analysis
3. Charts are saved as PNG files in project root

---

## Troubleshooting

**Module not found:**
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

**File not found:** Make sure `diabetes.csv` is in the project root.

---

## Credits

**Dataset**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Libraries**: pandas, matplotlib, seaborn, numpy
