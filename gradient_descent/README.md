## Gradient Descent Implementations and Linear Regression Analysis

This repository contains a completed **homework assignment** focusing on the implementation and analysis of various **Gradient Descent (GD)** optimization algorithms applied to a **Linear Regression** model.

### Key Files and Components

* **`descent.py`**: This file contains the core implementations of several Gradient Descent variants from scratch.
    * `**VanillaGradientDescent**`: The standard, basic batch GD algorithm.
    * `**StochasticGradientDescent** (SGD)`: Implements the stochastic version, updating parameters using a single, randomly chosen data point.
    * `**StochasticAverageGradient** (SAG)`: A memory-efficient, accelerated variant of SGD that uses average gradients.
    * `**MomentumDescent**`: Implements the Momentum optimization technique to accelerate GD in the relevant direction and dampen oscillations.
    * `**Adam**`: Implementation of the popular **Adaptive Moment Estimation (Adam)** optimizer, which combines ideas from Momentum and RMSprop.

* **`linear_regression.py`**: This file defines the `**LinearRegression**` class, which is used to model the relationship between variables. It uses the implemented descent algorithms for training.

* **`linear_regression_gradient_descent.ipynb`**: This **Jupyter Notebook** provides a comprehensive analysis:
    1.  **Data Preprocessing**: Steps taken to prepare the dataset for training.
    2.  **Model Training**: Training of the `LinearRegression` model using each of the implemented gradient descent methods (`VanillaGradientDescent`, `SGD`, `SAG`, `MomentumDescent`, `Adam`).
    3.  **Comparative Analysis**: Detailed examination and visualization of how the different optimization algorithms influence the **model's convergence, training speed, and final performance**.
