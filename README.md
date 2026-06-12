# 🏥 Healthcare Cost Prediction using Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

This project builds a deep neural network regression model to predict patient healthcare expenses based on demographic and health data.

> **Requirement:** Achieve a Mean Absolute Error (MAE) of under **$3,500** on the unseen test dataset.
>
> ✅ **Result:** The model successfully passed the challenge, consistently achieving an MAE of **$1351.94**.


![Preview](https://drive.google.com/file/d/1hfhckYDFp6rLRRphowqYUhQ-NvTB0U4_/view?usp=sharing)

---

## 📊 Methodology & Pipeline

Rather than simply feeding raw data into a model, this project applies a rigorous data science pipeline to maximize predictive accuracy and model robustness.

### 1. Exploratory Data Analysis (EDA)

Before modeling, the dataset was visualized to understand underlying distributions and correlations.

**Key Insight:** A scatter plot of BMI vs. Expenses (segmented by smoker status) revealed that high BMI combined with smoking causes an **exponential increase** in costs, rather than a linear one.

---

### 2. Feature Engineering

Based on the EDA insights, new features were manually engineered to give the neural network explicit context it might struggle to learn on its own:

| Feature | Description |
|---|---|
| `is_obese` | Binary flag — `1` if BMI > 30, else `0` |
| `bmi_smoker_interaction` | Product of BMI and smoker status, capturing the compound effect of both risk factors |

---

### 3. Robust Preprocessing

- **One-Hot Encoding:** Categorical variables (`sex`, `smoker`, `region`) were one-hot encoded using `dtype=float` to avoid a known Keras bug where boolean arrays fail to scale correctly in the `Normalization` layer.
- **Normalization:** A `tf.keras.layers.Normalization` layer was added as the first layer of the model to automatically shift and scale inputs to a mean of 0 and variance of 1, preventing gradient explosion from mixed-scale features (e.g., Age vs. Binary columns).

---

### 4. Model Architecture & Regularization

To prevent overfitting and ensure the model generalizes well to unseen data:

- **Dropout Layers:** `Dropout(0.2)` was implemented between dense layers, randomly deactivating 20% of neurons during training to force the network to learn generalized patterns.
- **MAE Loss Function:** Mean Absolute Error was chosen over Mean Squared Error (MSE). Medical expense data contains severe high-end outliers (bills > $50k). MSE squares these errors, creating massive gradient spikes that destabilize training; MAE treats errors linearly, making training robust to outliers.

---

### 5. Advanced Evaluation (Residual Analysis)

Instead of solely relying on the final MAE score, the **Residuals** (Actual Cost − Predicted Cost) were plotted. This verifies that the model's errors are normally distributed around zero, confirming there is no systemic bias in the predictions.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Framework | TensorFlow / Keras |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colaboratory |

---

## 🚀 How to Run

1. Click the **"Open in Colab"** badge at the top of this README.
2. In Colab, go to **Runtime > Run all**.
3. The dataset will be automatically downloaded via `wget`, the model will train, and the final evaluation cell will output the MAE and prediction scatter plot.

---

## 📁 File Structure

```
├── Health_Costs_Predictor.ipynb   # Complete, runnable Colab notebook (EDA, preprocessing, training, evaluation)
└── README.md                      # You are here!
```

