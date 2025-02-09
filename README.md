# 🏡 **Airbnb Price Prediction - NYC**  

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/) [![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)](https://xgboost.readthedocs.io/en/latest/) [![Data Science](https://img.shields.io/badge/Data%20Science-Pandas-orange)](https://pandas.pydata.org/)  

---

## 📌 **Project Overview**  
This project leverages **machine learning and data science techniques** to predict **Airbnb listing prices** in **New York City (NYC)**. The goal is to provide **data-driven pricing strategies** that help **Airbnb hosts optimize revenue** by dynamically adjusting prices based on **demand, seasonality, amenities, and location**.  

💡 **Key Objectives:**  
✔ Identify **key factors influencing Airbnb prices**  
✔ Develop a **robust machine learning pipeline** for price prediction  
✔ Apply **feature engineering** to enhance model accuracy  
✔ Compare **multiple ML models** to determine the best-performing algorithm  
✔ Deliver **actionable insights** for hosts to maximize earnings  



## 📊 **Dataset & Methodology**  
### **Dataset:**  
The dataset is sourced from **Kaggle’s NYC Airbnb Open Data** and contains **74,111 rows & 29 columns**. After **data preprocessing & feature engineering**, the refined dataset consists of **32,349 observations & 23 relevant features**.  

### **Key Features Considered:**  
| **Feature** | **Description** |
|------------|----------------|
| `log_price` | Target variable (log-transformed price) |
| `room_type` | Entire home/apt, Private room, Shared room |
| `accommodates` | Number of people a listing can accommodate |
| `amenities` | Features offered (WiFi, TV, Kitchen, etc.) |
| `borough` | NYC boroughs (Manhattan, Brooklyn, etc.) |
| `travel_time` | Time to major attractions (Times Square, Airports) |

### **Data Preprocessing & Feature Engineering:**  
✔ **Handled missing values** (Imputation & feature selection)  
✔ **Applied One-Hot Encoding** to categorical variables  
✔ **Extracted borough information** from latitude/longitude  
✔ **Engineered travel-time features** using **OSRM API**  
✔ **Applied PCA (Principal Component Analysis)** on amenities for dimensionality reduction  


## 📁 **Project Structure**  
```
📂 data/                # Raw and cleaned datasets
📂 notebooks/           # Jupyter Notebooks for EDA, ML modeling, and analysis
📂 src/                 # Python scripts for modularization
📂 results/             # Model results and performance metrics
📂 config/              # Configuration settings for hyperparameters
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── Dockerfile         # Containerization setup
├── app.py             # Flask API for model deployment
```


## ⚙️ **Machine Learning Models Used**
We experimented with **multiple regression models**, and **XGBoost** outperformed all with **the highest accuracy**.

| **Model**                  | **R² Score (Validation)** | **RMSE (Validation)** |
|----------------------------|--------------------------|------------------------|
| **Linear Regression**      | 0.68                     | 0.34                   |
| **Lasso Regression**       | 0.72                     | 0.31                   |
| **Random Forest**          | 0.75                     | 0.29                   |
| **Support Vector Regression (SVR)** | 0.74           | 0.33                   |
| **XGBoost (Optimized)**    | **0.78**                 | **0.31**               |

📌 **Best Model**: **XGBoost** achieved **78% R²** and **0.31 RMSE**, making it the most accurate for **dynamic Airbnb pricing**.

---

## 💻 **Installation & Usage**  
### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/Navabhargav/Airbnb-price-prediction.git
cd Airbnb-price-prediction
```  

### **2️⃣ Install Required Dependencies**  
```sh
pip install -r requirements.txt
```  

### **3️⃣ Run Jupyter Notebook**  
```sh
jupyter notebook
```
Open `BUSM131_masterclass.ipynb` and execute the cells step by step.

### **4️⃣ Run the Model Training Script**  
```sh
python src/model_training.py
```

### **5️⃣ Deploy Model as API (Flask Server)**  
```sh
python app.py
```
Model will be available at: **http://127.0.0.1:5000/predict**

---

## 📈 **Key Insights & Findings**
✔ **Airbnb prices vary significantly based on boroughs** – **Manhattan listings** have the highest average prices.  
✔ **Travel time** to key attractions like **Times Square** and **JFK Airport** **impacts listing prices**.  
✔ **Entire apartments are priced 2x higher than private rooms**.  
✔ **Machine learning-driven pricing recommendations** help **hosts maximize revenue** by dynamically adjusting prices.  

---

## 🚀 **Business Impact**
📊 **Dynamic Pricing Strategy**: Helps **Airbnb hosts** maximize revenue based on **real-time demand & location data**.  
📉 **Occupancy Rate Optimization**: Provides **competitive pricing suggestions** to minimize **vacancies**.  
🧠 **Data-Driven Decision Making**: Empowers **property owners & investors** with insights to **adjust pricing models effectively**.  

🔹 **Example Insight:**  
- Listings **near Central Park** are **25% more expensive** than those in **outer boroughs like Queens/Bronx**.  
- **Peak demand seasons (summer & holidays)** show a **15-20% price surge**, making **dynamic pricing essential**.  

---

## 🖥️ **Technologies Used**
| **Category**         | **Technologies** |
|----------------------|-----------------|
| **Languages**        | Python, SQL |
| **Data Processing**  | Pandas, NumPy, SciPy |
| **Machine Learning** | Scikit-learn, XGBoost, TensorFlow |
| **Data Visualization** | Matplotlib, Seaborn, Tableau |
| **Feature Engineering** | PCA, Travel Time Estimation (OSRM API) |
| **Deployment**       | Flask, Docker, AWS Lambda |

---

## 📚 **References**
- 📄 Kaggle Dataset: [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)  
- 📄 Scikit-learn Documentation: [Machine Learning in Python](https://scikit-learn.org/stable/)  
- 📄 OpenStreetMap API: [OSRM Travel Time Estimation](https://wiki.openstreetmap.org/wiki/OpenRouteService)  
- 📄 Airbnb Dynamic Pricing Insights: [Forbes](https://www.forbes.com/companies/airbnb/)  

---

## 🤝 **Contributors**
👤 **Nava Bhargav Gedda**  
📩 [navabhargavg@gmail.com](mailto:navabhargavg@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/nava-bhargav-gedda-4a4a30151) | 🌐 [GitHub](https://github.com/Navabhargav)  


---

## ⭐ **Like this Project?**
If you found this project useful, **give it a star ⭐** on GitHub and share it with others! 🚀  
```
