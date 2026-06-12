🏥 Healthcare Cost Prediction using Deep Learning
Open In Colab

This project builds a deep neural network regression model to predict patient healthcare expenses based on demographic and health data. It is certified by freeCodeCamp as part of the Machine Learning with Python Certification.

Challenge Requirement: Achieve a Mean Absolute Error (MAE) of under $3,500 on the unseen test dataset.

✅ Result: The model successfully passed the challenge, consistently achieving an MAE of ~$3,100 — $3,300.

📊 Methodology & Pipeline
Rather than simply feeding raw data into a model, this project applies a rigorous data science pipeline to maximize predictive accuracy and model robustness.

1. Exploratory Data Analysis (EDA)
Before modeling, I visualized the dataset to understand underlying distributions and correlations.

Key Insight: A scatter plot of BMI vs. Expenses (segmented by smoker status) revealed that high BMI combined with smoking causes an exponential increase in costs, rather than a linear one.
2. Feature Engineering
Based on the EDA insights, I manually engineered new features to give the neural network explicit context it might struggle to learn on its own:

is_obese: A binary flag (1 if BMI > 30, else 0).
bmi_smoker_interaction: The product of BMI and smoker status, capturing the compound effect of both risk factors simultaneously.
3. Robust Preprocessing
One-Hot Encoding: Categorical variables (sex, smoker, region) were one-hot encoded using dtype=float to avoid a known Keras bug where boolean arrays fail to scale correctly in the Normalization layer.
Normalization: A tf.keras.layers.Normalization layer was added as the first layer of the model to automatically shift and scale inputs to a mean of 0 and variance of 1, preventing gradient explosion from mixed-scale features (e.g., Age vs. Binary columns).
4. Model Architecture & Regularization
To prevent overfitting and ensure the model generalizes well to unseen data:

Dropout Layers: Implemented Dropout(0.2) between dense layers, randomly deactivating 20% of neurons during training to force the network to learn generalized patterns.
MAE Loss Function: Chose Mean Absolute Error over Mean Squared Error (MSE). Medical expense data contains severe high-end outliers (bills > $50k). MSE squares these errors, creating massive gradient spikes that destabilize training; MAE treats errors linearly, making the training process robust to outliers.
5. Advanced Evaluation (Residual Analysis)
Instead of solely relying on the final MAE score, I plotted the Residuals (Actual Cost - Predicted Cost). This verifies that the model's errors are normally distributed around zero, confirming there is no systemic bias in the predictions.

🛠️ Tech Stack
Language: Python
Framework: TensorFlow / Keras
Data Manipulation: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Environment: Google Colaboratory
🚀 How to Run
Click the "Open in Colab" badge at the top of this README.
In Colab, go to Runtime > Run all.
The dataset will be automatically downloaded via wget, the model will train, and the final evaluation cell will output the MAE and prediction scatter plot.
📁 File Structure
Health_Costs_Predictor.ipynb - The complete, runnable Google Colab notebook containing EDA, preprocessing, model training, and evaluation.
README.md - You are here!
