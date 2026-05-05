## Software System Overview

This project implements the software layer of an **Intelligent Flooring System** designed for energy generation and smart space monitoring using electromagnetic generators.

The system processes sensor data generated from footstep activity and applies machine learning techniques to extract meaningful insights such as step detection and crowd estimation.

### Key Functionalities

* **Data Acquisition & Simulation**
  Generates synthetic INA219 sensor data representing voltage and current produced from footsteps.

* **Signal Processing**
  Converts raw sensor readings into usable features such as power and temporal patterns.

* **Step Detection (CNN Model)**
  Identifies individual footsteps from time-series sensor data.

* **People Counting (ML Model)**
  Estimates the number of people based on detected step patterns.

* **Dashboard Visualization**
  Displays crowd activity and energy generation metrics using a web-based interface.

### Technologies Used

* Python (NumPy, Pandas)
* TensorFlow / Keras
* HTML, CSS, JavaScript (Dashboard)
* JSON for data exchange

### Workflow

1. Generate or load sensor dataset
2. Train machine learning models
3. Perform predictions on session data
4. Visualize results in the dashboard

This software system bridges hardware-generated energy signals with intelligent analytics for smart infrastructure applications.
