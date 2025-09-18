# file: churn_gui.py
import tkinter as tk
import numpy as np, joblib
from tensorflow.keras.models import load_model

# Feature columns (surname removed)
cols = [
    'CreditScore','Age','Tenure','Balance','NumOfProducts',
    'HasCrCard\n(If yes = 1 else 0)','IsActiveMember\n(If yes = 1 else 0)','EstimatedSalary',
    'France\n(If yes = 1 else 0)','Germany\n(If yes = 1 else 0)','Spain\n(If yes = 1 else 0)','Female\n(If yes = 1 else 0)',
    'Male\n(If yes = 1 else 0)','Mem__no__Products','Age_Tenure_product'
]

# Load trained model and scaler
m = load_model("ChurnPredictor.h5")
s = joblib.load("churnScaler.pkl")

# GUI setup
root = tk.Tk()
root.title("Churn Predictor")
root.configure(bg="#f0f4f7")   # Light background
root.geometry("500x750")       # Set window size

# Title label
title = tk.Label(root, text="Bank Customer Churn Predictor", 
                 font=("Helvetica", 16, "bold"), 
                 fg="#ffffff", bg="#2c3e50", pady=10)
title.pack(fill="x")

# Framing for form
form_frame = tk.Frame(root, bg="#f0f4f7", padx=15, pady=10)
form_frame.pack(fill="both", expand=True)

es = {}

# Creating input fields
for i, c in enumerate(cols):
    tk.Label(form_frame, text=c, width=20, anchor='w',
             font=("Arial", 10), bg="#f0f4f7").grid(row=i, column=0, sticky='w', pady=4)
    e = tk.Entry(form_frame, width=25, font=("Arial", 10))
    e.grid(row=i, column=1, pady=4, padx=8)
    es[c] = e

# OUTPUT HEADING & LABEL 
output_heading = tk.Label(root, text="Output", 
                          font=("Helvetica", 14, "bold"),
                          fg="#2c3e50", bg="#f0f4f7", pady=10)
output_heading.pack()

output_label = tk.Label(root, text="", 
                        font=("Arial", 12), 
                        bg="#ecf0f1", fg="black", 
                        width=50, height=5, 
                        relief="sunken", justify="center", wraplength=400)
output_label.pack(pady=10)

# Prediction function with 5 stages
def pred():
    try:
        # Collecting inputs
        v = []
        for c in cols:
            t = es[c].get().strip()
            v.append(float(t))

        # Convert to numpy array
        X = np.array(v).reshape(1, -1)

        # Scale input
        if s.n_features_in_ == X.shape[1]:
            X_scaled = s.transform(X)
        else:
            X_scaled = X   # fallback if scaler wrong

        # Predicting churn probability
        p = float(m.predict(X_scaled, verbose=0)[0][0])

        # Defining 5 stages
        if p < 0.2:
            stage = "❌ Very Low Risk (Not Likely to Churn)"
        elif p < 0.4:
            stage = "⚪ Low Risk (Unlikely to Churn)"
        elif p < 0.6:
            stage = "🟡 Medium Risk (Monitor Closely)"
        elif p < 0.8:
            stage = "🟠 High Risk (Likely to Churn)"
        else:
            stage = "✅ Very High Risk (Almost Certain to Churn)"

        #Show result inside the GUI
        result_text = f"Predicted Churn Probability: {p:.4f}\nStage: {stage}"
        output_label.config(text=result_text)

    except Exception as e:
        output_label.config(text=f"Error: {str(e)}")

#button
btn = tk.Button(root, text="Predict", command=pred,
                font=("Helvetica", 12, "bold"), 
                bg="#27ae60", fg="white", 
                activebackground="#2ecc71", activeforeground="white",
                relief="raised", bd=3, width=20, pady=8)
btn.pack(pady=15)

root.mainloop()
