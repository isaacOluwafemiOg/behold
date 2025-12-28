# 🔭 Behold: Hunting for Exoplanets with AI
### NASA Space Apps Challenge 2025 Submission | Team Behold

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://behold-exoplanets.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"A World Away"**: Empowering astronomers and enthusiasts to identify exoplanet candidates from Kepler mission data using explainable ensemble Machine Learning.

---

## 📖 Overview
**Behold** is a web application designed to predict the disposition of **Threshold Crossing Events (TCEs)** observed by the Kepler Space Telescope. By leveraging an ensemble of **CatBoost classifiers**, the application distinguishes between legitimate **Candidate Exoplanets** and **False Positives** with high accuracy.

Beyond simple prediction, Behold emphasizes **explainability**. It allows users to understand *why* a model made a specific decision using SHAP (SHapley Additive exPlanations) values, making the "black box" of AI transparent for scientific inquiry.

**[🚀 View Short Demo](https://www.youtube.com/watch?v=v-rNypRc4fE)**

---

## ✨ Key Features

### 1. 🔍 Explorer Mode (EDA)
Designed for both novices and experts to familiarize themselves with the Kepler Object of Interest (KOI) dataset.
*   **Visualizations:** Feature correlation heatmaps, scatter plots, and target distributions.
*   **Insights:** Understand the 14 critical features used for training.

### 2. 🎯 Single Prediction with Explainability
Input specific parameters of an observation to get an instant classification.
*   **Result:** Classifies as `CANDIDATE` or `FALSE POSITIVE`.
*   **Interpretability:** Visualizes feature importance using SHAP plots, showing exactly which variables (e.g., Planet-Star Radius Ratio, Transit Duration) pushed the model toward its decision.

### 3. 📂 Batch Prediction
Upload large datasets to process multiple observations simultaneously.
*   **Input:** Accepts CSV files formatted with standard Kepler data columns.
*   **Output:** Generates a downloadable summary of predictions.

---

## 🧠 How It Works & Performance

The core prediction engine is built on an ensemble of **5 CatBoost Classifiers**, optimized using **Optuna**. The models were trained, validated, and tested on data from the NASA Exoplanet Archive.

### The Methodology
1.  **Data Processing:** Utilized thousands of transit observations from the Kepler Mission.
2.  **Feature Selection:** Narrowed down to the **14 most impactful features** to ensure model efficiency without sacrificing accuracy.
3.  **Ensemble Voting:** When a new observation is input, all 5 models analyze the data. The final output is determined by a majority vote, ensuring robustness.

### Model Metrics
The system demonstrates high generalizability, minimizing the risk of overfitting:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Train F1 Score** | **97.22%** | High fidelity on learning data. |
| **CV F1 Score** | **90.09%** | Robustness across 5-fold cross-validation. |
| **Test F1 Score** | **89.77%** | Proven accuracy on unseen data. |

---

## 🛠️ Tech Stack

*   **Language:** Python
*   **Web Framework:** [Streamlit](https://streamlit.io/)
*   **Machine Learning:** CatBoost, Scikit-Learn
*   **Optimization:** Optuna
*   **Explainability:** SHAP (SHapley Additive exPlanations)
*   **Visualization:** Plotly, Matplotlib
*   **Deployment:** Docker, Render

---

## 💻 Installation & Local Usage

You can run Behold locally using standard Python installation or Docker.

### Option 1: Standard Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/isaacOluwafemiOg/behold.git
    cd behold
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**
    ```bash
    streamlit run app.py
    ```

### Option 2: Docker

1.  **Build the image**
    ```bash
    docker build -t behold-app .
    ```

2.  **Run the container**
    ```bash
    docker run -p 8501:8501 behold-app
    ```
    Access the app at `http://localhost:8501`.

--

## 👥 Team Behold

*   **Isaac Oluwafemi Ogunniyi** - *Machine Learning & Aerospace Engineer*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

*   **NASA Space Apps Challenge** for the opportunity.
*   **NASA Exoplanet Archive** for the publicly available Kepler mission data.
