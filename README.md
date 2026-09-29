# ♻️ Waste Generation Prediction System

An ML-based waste generation prediction system that forecasts daily municipal waste generation using historical waste collection data and provides an estimated number of collection trips required.

The project uses **Machine Learning with Random Forest Regression** and provides an interactive **Streamlit web application** for making predictions.

---

## 🚀 Project Overview

Efficient waste collection requires better estimation of how much waste will be generated.

This project analyzes historical waste collection data and predicts future waste generation based on:

* Day of the week
* Month
* Day of the month
* Previous day's waste
* Previous 7-day average waste

The system can also estimate the number of collection trips required based on the predicted waste quantity and truck capacity.

---

## 🎯 Objectives

* Predict daily waste generation using Machine Learning.
* Compare multiple regression algorithms.
* Identify important factors affecting waste generation.
* Provide an interactive prediction interface.
* Estimate the number of waste collection trips required.
* Support data-driven waste collection planning.

---

## 🛠️ Technologies Used

| Technology    | Purpose              |
| ------------- | -------------------- |
| Python        | Programming language |
| Pandas        | Data processing      |
| NumPy         | Numerical operations |
| Scikit-learn  | Machine Learning     |
| Random Forest | Waste prediction     |
| Matplotlib    | Data visualization   |
| Streamlit     | Web application      |
| Git & GitHub  | Version control      |

---

## 📂 Project Structure

```text
Waste_Prediction_Project/
│
├── data/
│   └── austin_waste.csv
│
├── Screenshots/
│   ├── main_dashboard.png
│   ├── prediction_result.png
│   └── waste_graph.png
│
├── app.py
├── load_dataset.py
├── predict_waste.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The project uses historical municipal waste collection data containing information about waste collection activities.

The dataset is processed by:

1. Converting dates into a standard datetime format.
2. Removing records with missing waste quantities.
3. Aggregating waste generation by date.
4. Creating time-based and historical features.
5. Preparing the data for regression models.

---

## 🧠 Feature Engineering

The following features are used for prediction:

| Feature              | Description                                        |
| -------------------- | -------------------------------------------------- |
| `Day_of_Week`        | Day of the week                                    |
| `Month`              | Month of the year                                  |
| `Day`                | Day of the month                                   |
| `Previous_Day_Waste` | Waste generated on the previous day                |
| `Previous_7_Day_Avg` | Average waste generated during the previous 7 days |

Feature engineering helps the model capture both **calendar patterns** and **recent waste-generation trends**.

---

## 🤖 Machine Learning Models

Three regression models were compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The data was divided chronologically into:

* **80% Training Data**
* **20% Testing Data**

A chronological split was used because waste prediction is a time-dependent problem.

---

## 📈 Model Performance

The Random Forest model produced the following test results:

| Metric   |     Result |
| -------- | ---------: |
| MAE      |  70,646.40 |
| RMSE     | 100,710.88 |
| R² Score |     0.9386 |

### Model Comparison

| Model             | R² Score |
| ----------------- | -------: |
| Linear Regression |   0.5439 |
| Decision Tree     |   0.8879 |
| Random Forest     |   0.9385 |

The values above are based on the project's chronological test split.

---

## 🌲 Random Forest Configuration

The Random Forest model was trained using:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The model uses multiple decision trees and combines their predictions to produce the final regression output.

---

## 🔍 Feature Importance

The trained Random Forest model identified the following feature importance values:

| Feature                | Importance |
| ---------------------- | ---------: |
| Day of Week            |     0.8641 |
| Previous Day Waste     |     0.0563 |
| Previous 7-Day Average |     0.0332 |
| Day                    |     0.0256 |
| Month                  |     0.0209 |

These values indicate how much each feature contributed to the model's predictions within this trained model.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

The user can:

1. Select a prediction date.
2. Enter the available truck capacity.
3. Generate the predicted waste quantity.
4. View estimated collection trips.
5. View historical waste-generation trends.
6. View model performance information.

The application combines:

* Random Forest prediction
* Same-day-of-week historical average
* Recent 30-day average

to produce the displayed prediction.

---

## 📸 Application Screenshots

### Main Dashboard

![Main Dashboard](Screenshots/main_dashboard.png)

### Prediction Result

![Prediction Result](Screenshots/prediction_result.png)

### Historical Waste Graph

![Waste Generation Graph](Screenshots/waste_graph.png)

---

## 🔄 Project Workflow

```text
Historical Waste Dataset
          ↓
Data Cleaning
          ↓
Date Processing
          ↓
Daily Waste Aggregation
          ↓
Feature Engineering
          ↓
Train/Test Split
          ↓
Model Training
          ↓
Model Evaluation
          ↓
Random Forest Prediction
          ↓
Streamlit Application
          ↓
Predicted Waste Generation
          ↓
Estimated Collection Trips
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Maithili0131/waste-generation-prediction.git
```

### 2. Open the project folder

```bash
cd waste-generation-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📌 Files Description

### `app.py`

Runs the Streamlit web application and provides the interactive prediction interface.

### `load_dataset.py`

Loads, cleans, processes, and analyzes the dataset and compares the regression models.

### `train_model.py`

Trains the Random Forest model and evaluates its performance and feature importance.

### `predict_waste.py`

Provides a command-line based method for predicting waste generation and estimating collection trips.

---

## ⚠️ Limitations

* Predictions depend on the quality and representativeness of historical data.
* The dataset may not represent the waste-generation patterns of every city.
* External factors such as festivals, weather, construction activity, or sudden events are not directly modeled.
* The project is intended as a prediction and planning prototype rather than a complete municipal waste-management system.

---

## 🔮 Future Scope

Possible future improvements include:

* Using a location-specific dataset for Navi Mumbai.
* Adding weather and holiday information.
* Testing advanced time-series models.
* Adding real-time municipal data.
* Creating location-wise waste prediction.
* Developing a collection-route optimization module.
* Adding an alert mechanism for predicted high-waste periods.
* Deploying the application online.

---

## 👩‍💻 Author

**Maithili Madhavi**

B.E. Computer Engineering

GitHub: [Maithili0131](https://github.com/Maithili0131)

LinkedIn: [Maithili Madhavi](https://www.linkedin.com/in/maithilimadhavi321/)

---

## ⭐ Project Highlights

* Machine Learning based waste prediction
* Random Forest Regression
* Feature engineering
* Chronological train/test split
* Model comparison
* Streamlit interactive application
* Collection-trip estimation
* Data visualization
* GitHub-ready project structure
