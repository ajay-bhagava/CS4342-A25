# Predicting Student Stress Levels

Authors (in alphabetical order by last name): Sean Arackal, Ajay Bhagavatula, Matthew Franco, Zach Szeto, Ryan Zhang

CS 4342 Machine Learning

## Project Overview

This project focuses on predicting student stress levels using machine learning techniques. We analyze various factors that contribute to student stress and build predictive models to classify stress levels into categories. The project implements and compares two different classification algorithms: Random Forest and LightGBM (Light Gradient Boosting Machine).

## Dataset

Kaggle Dataset: https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets
We use the `StressLevelDataset.csv` dataset which contains 20 features related to student well-being and academic performance:

### Features:

- **Mental Health Indicators**: anxiety_level, self_esteem, mental_health_history, depression
- **Physical Health**: headache, blood_pressure, sleep_quality, breathing_problem
- **Environmental Factors**: noise_level, living_conditions, safety, basic_needs
- **Academic Factors**: academic_performance, study_load, teacher_student_relationship, future_career_concerns
- **Social Factors**: social_support, peer_pressure, extracurricular_activities, bullying

### Target Variable:

- **stress_level**: Categories include "No Stress", "Normal/Moderate Stress", and "High Stress"

## Models Implemented

1. **Random Forest Classifier**: An ensemble method that combines multiple decision trees
2. **LightGBM Classifier**: A gradient boosting framework optimized for speed and memory efficiency

Both models are evaluated using accuracy metrics and confusion matrices for performance comparison.

## Requirements

The project requires the following Python libraries:

- pandas
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- numpy

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install pandas scikit-learn lightgbm matplotlib seaborn numpy
```

## How to Run

1. Ensure you have Python 3.7+ installed on your system
2. Navigate to the project directory:

```bash
cd CS4342-A25
```

3. Run the main script:

```bash
python StudentStress_RandomForest.py
```

## Output

The script will:

1. Load and preprocess the stress level dataset
2. Split the data into training and testing sets (80/20 split)
3. Apply feature scaling using StandardScaler
4. Train both Random Forest and LightGBM classifiers
5. Display accuracy scores for both models
6. Show confusion matrix heatmaps for model evaluation
7. Display feature importance rankings
8. Generate a bar chart showing the top 8 most important features
