# Diamond Price Prediction - ML Coursework 1

**Student ID:** k24063781  
**Evaluation Metric:** Out-of-sample $R^2$

## Project Overview
This repository contains the implementation for a diamond price prediction task. The goal was to develop a robust machine learning pipeline capable of handling non-linear relationships and a high volume of synthetic noise features ($a1-b10$).

## Repository Structure
The project is split into three standalone scripts to demonstrate the model selection process:

1. `linearReg.py` [CW1_submission_k24063781_Linear.csv]: Baseline implementation using Ordinary Least Squares (OLS) regression.
2. `randomForest.py` [CW1_submission_k24063781_RF.csv]: An ensemble-based approach using 100 decision trees.
3. `gradientBoost.py` [CW1_submission_k24063781.csv]: The final optimized model using **XGBoost** with specialized noise-reduction hyperparameters.

## Installation
```bash
pip install -r requirements.txt