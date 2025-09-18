# Bank Customer Churn Prediction Using Deep Learning- Build during Data Scientist Intern at Milestone PLM Solutions Pvt. Ltd., Thane

[![GitHub stars](https://img.shields.io/github/stars/abubakarpungiwale/bank-churn-prediction-dl?style=social)](https://github.com/abubakarpungiwale/bank-churn-prediction-dl/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/abubakarpungiwale/bank-churn-prediction-dl?style=social)](https://github.com/abubakarpungiwale/bank-churn-prediction-dl/network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This repository implements a deep learning solution for predicting bank customer churn using an Artificial Neural Network (ANN). Built on a dataset of 10,000+ records with 16 features (e.g., CreditScore, Age, Balance), it classifies customers as churners (1) or retainers (0). Developed during a Data Scientist internship at **Milestone PLM Solutions Pvt. Ltd., Thane**, it showcases end-to-end DL pipelines for binary classification in finance.

**Key Highlights**:
- **Neural Network Architecture**: Multi-layer ANN with Dense layers, BatchNormalization, Dropout, and L1 regularization for robust, overfitting-resistant predictions.
- **Deep Learning Focus**: TensorFlow/Keras Sequential model with EarlyStopping for optimal training.
- **Production-Ready**: Model saved as `.h5` with scaler for deployment.

Ideal portfolio piece demonstrating DL expertise for data science/ML engineering roles.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies](#key-technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Technologies

- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn (EDA/visualization); Scikit-learn (preprocessing, splitting); TensorFlow/Keras (ANN modeling); Joblib (model saving).
- **Model**: Sequential ANN (input: 15 features → hidden: Dense(128/64/32 with ReLU) → output: Dense(1, sigmoid) for binary classification.
- **Techniques**: StandardScaler normalization, train_test_split (80/20), L1 regularization, Dropout (0.3-0.5), EarlyStopping (patience=10).
- **Metrics**: Accuracy, binary predictions (threshold: 0.5).

## Installation

```bash
git clone https://github.com/yourusername/bank-churn-prediction-dl.git
cd bank-churn-prediction-dl
pip install -r requirements.txt
```

## Usage

1. Place `Bank Customer Churn.csv` in the root directory.
2. Run:
   ```bash
   jupyter notebook Bank_Customer_Churn_Prediction_using_DL.ipynb
   ```
3. Execute cells for EDA, training, evaluation. Load saved model: `ChurnPredictor.h5` & `churnScaler.pkl`.

## Methodology

- **Preprocessing**: EDA (descriptions, correlations), feature engineering (e.g., Mem__no__Products, Age_Tenure_product), StandardScaler, 80/20 split.
- **Model Building**: Sequential ANN with 3 hidden layers (128-64-32 neurons), ReLU activation, sigmoid output; compiled with Adam optimizer & binary_crossentropy.
- **Training**: Fit on scaled data with EarlyStopping; predictions thresholded at 0.5 for churn classification.
- **Evaluation**: Accuracy on test set; sample predictions (e.g., "Not Likely to Churn").

## Performance Metrics

- **Test Accuracy**: ~85% (ANN outperforms baselines by 5-8% via regularization).
- **Insights**: High recall on churners; confusion matrix/logs in notebook. DL architecture handles non-linearity effectively.
- **Extensions**: Potential for LSTM/ensemble hybrids.

## Contributing

Fork and submit pull requests for enhancements like hyperparameter tuning or CNN integration.

## License

MIT License - see [LICENSE](LICENSE).

## Contact

- **Author**: Abubakar Maulani Pungiwale
- **Email**: abubakarp496@gmail.com
- **LinkedIn**: [linkedin.com/in/abubakarpungiwale](https://linkedin.com/in/abubakarpungiwale)
- **Portfolio**: +91 9321782858

Eager to discuss DL projects or data science opportunities—reach out!

---
