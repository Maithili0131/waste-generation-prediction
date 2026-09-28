# ♻️ Waste Generation Prediction System

## 📌 Overview

The **Waste Generation Prediction System** is a machine learning-based project that predicts daily waste generation using historical waste collection data.

The system uses historical waste patterns, previous-day waste, previous 7-day average waste, day of the week, month, and day of the month as input features.

A **Random Forest Regression** model is used to predict future waste generation. The project also includes a **Streamlit web application** where users can select a prediction date and enter truck capacity to estimate the number of collection trips required.

---

## 🎯 Objectives

* Predict future daily waste generation using historical data.
* Analyze patterns in waste generation.
* Compare different machine learning regression models.
* Use Random Forest Regression for waste prediction.
* Estimate the number of collection trips required based on truck capacity.
* Provide an easy-to-use web interface using Streamlit.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and preprocessing
* **NumPy** – Numerical calculations
* **Scikit-learn** – Machine learning models and evaluation
* **Random Forest Regression** – Main prediction model
* **Streamlit** – Web-based dashboard
* **Matplotlib / Streamlit Charts** – Data visualization

---

## 📂 Project Structure

```text
Waste_Prediction_Project/
│
├── data/
│   └── austin_waste.csv
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

The project uses historical municipal waste collection data.

The dataset contains waste collection records with information including the **Report Date** and **Load Weight**.

The individual collection records are aggregated by date to calculate the total waste generated each day.

### Data Processing

The dataset goes through the following preprocessing steps:

1. Load the CSV dataset.
2. Convert `Report Date` into datetime format.
3. Remove records with missing `Load Weight`.
4. Group waste collection records by date.
5. Calculate total daily waste.
6. Sort the data chronologically.

---

## 🔧 Feature Engineering

The following features are created for prediction:

| Feature              | Description                                      |
| -------------------- | ------------------------------------------------ |
| `Day_of_Week`        | Day of the week represented numerically          |
| `Month`              | Month of the prediction date                     |
| `Day`                | Day of the month                                 |
| `Previous_Day_Waste` | Waste generated on the previous available day    |
| `Previous_7_Day_Avg` | Average waste from the previous 7 available days |

### Target Variable

```text
Total_Waste
```

The model predicts the total waste generated for a particular day.

---

## 🤖 Machine Learning Models

The project compares three regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The data is divided chronologically into:

* **80% training data**
* **20% testing data**

A chronological split is used because this is time-based data and future information should not be used to train the model.

---

## 🌳 Random Forest Model

The main model used in the project is **Random Forest Regressor**.

Configuration:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

The final model is trained using the available historical data for future prediction.

---

## 📈 Model Evaluation

The Random Forest model is evaluated using:

### MAE

**Mean Absolute Error** measures the average absolute difference between actual and predicted waste.

### MSE

**Mean Squared Error** measures the average squared difference between actual and predicted values.

### RMSE

**Root Mean Squared Error** is the square root of MSE and represents prediction error in the same unit as the target variable.

### R² Score

**R² Score** indicates how much of the variation in waste generation is explained by the model.

---

## 🔮 Prediction System

The prediction system accepts:

* Prediction date
* Truck capacity

The Random Forest model predicts the expected waste generation.

The Streamlit application additionally considers:

* Random Forest prediction
* Same day-of-week historical average
* Recent 30-day average

The final prediction is calculated using:

```text
Final Prediction =
70% Random Forest Prediction
+ 20% Same Day-of-Week Average
+ 10% Recent 30-Day Average
```

---

## 🚛 Collection Trip Estimation

The system estimates the number of collection trips using:

```text
Required Trips = Ceiling(Predicted Waste / Truck Capacity)
```

For example, if the predicted waste is 1,000,000 units and the truck capacity is 400,000 units:

```text
Required Trips = ceil(1,000,000 / 400,000)
               = 3 trips
```

---

## 💻 Streamlit Dashboard

The Streamlit application provides:

* 📅 Prediction date selection
* 🚛 Truck capacity input
* 🔮 Waste prediction
* 📊 Required collection trips
* 🤖 Model performance metrics
* 📈 Historical waste generation chart
* 📁 Dataset information
* Prediction calculation details

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/waste-generation-prediction.git
```

### 2. Open the project folder

```bash
cd waste-generation-prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ▶️ Run Individual Python Programs

### Train and evaluate the model

```bash
python train_model.py
```

### Compare regression models

```bash
python load_dataset.py
```

### Make a command-line prediction

```bash
python predict_waste.py
```

---

## 🔮 Future Scope

The project can be extended by:

* Using a dataset specific to a particular city or municipal corporation.
* Including weather and seasonal information.
* Including holidays and special events.
* Adding waste-category information.
* Using additional time-series forecasting techniques.
* Developing a municipal corporation dashboard.
* Adding automated alerts for high predicted waste generation.
* Integrating real-time or regularly updated waste collection data.

---

## ⚠️ Limitations

* The current dataset is historical and may not represent every locality.
* Prediction quality depends on the quality and coverage of the historical data.
* The current system does not use real-time bin-level sensor data.
* Truck-trip estimation depends on the truck capacity entered by the user.
* The prediction is intended as a data-driven estimate and not as a guaranteed future value.

---

## 👩‍💻 Project

**Waste Generation Prediction System**

Developed as a machine learning project using Python, Scikit-learn and Streamlit.
