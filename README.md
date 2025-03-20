# Housing Price Prediction using Gradient Descent Methods

## Project Overview
This project explores different gradient descent optimization techniques to train a linear regression model for predicting housing prices. We compare Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), Mini-Batch Gradient Descent (MBGD), and Momentum Gradient Descent (MGD) on the Housing Price Dataset.

The project aims to:
- Implement and compare different gradient descent algorithms.
- Analyze the effect of batch size, momentum, and learning rate on convergence.
- Optimize model parameters for better performance.

## Dataset Information
- **Dataset Size**: 545 rows × 13 columns
- **Target Variable**: `price` (housing price)
- **Feature Types**:
  - **Numerical Features (int64)**: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  - **Categorical Features (object)**: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

### Columns & Description
| Column Name        | Data Type | Description |
|-------------------|----------|-------------|
| `price` | int64 | The target variable (house price) |
| `area` | int64 | Size of the house in square feet |
| `bedrooms` | int64 | Number of bedrooms |
| `bathrooms` | int64 | Number of bathrooms |
| `stories` | int64 | Number of floors in the house |
| `mainroad` | object | Whether the house is near a main road (`yes`/`no`) |
| `guestroom` | object | Whether there is a guest room (`yes`/`no`) |
| `basement` | object | Whether the house has a basement (`yes`/`no`) |
| `hotwaterheating` | object | Availability of hot water heating (`yes`/`no`) |
| `airconditioning` | object | Presence of air conditioning (`yes`/`no`) |
| `parking` | int64 | Number of parking spaces available |
| `prefarea` | object | Whether the house is in a preferred area (`yes`/`no`) |
| `furnishingstatus` | object | Type of furnishing (`furnished` / `semi-furnished` / `unfurnished`) |

## Preprocessing Steps
1. **Handling Categorical Data**:
   - Converted categorical columns (`yes/no`, `furnished/unfurnished`) into numerical form using One-Hot Encoding.
   
2. **Feature Scaling**:
   - Standardized numerical features using MinMax Scaling to improve gradient descent convergence.

3. **Adding a Bias Term**:
   - Added a column of ones to `X_train` and `X_test` to account for the intercept term in linear regression.

## Gradient Descent Implementations
We implemented four different optimization techniques:

### 1. Batch Gradient Descent (BGD)
- Updates model parameters after computing the gradient across the entire dataset.
- Stable convergence, but slower for large datasets.

### 2. Stochastic Gradient Descent (SGD)
- Updates parameters after each training example, leading to faster updates but more variance.
- Good for large datasets, but can have noisy convergence.

### 3. Mini-Batch Gradient Descent (MBGD)
- Combines BGD and SGD by updating parameters on small random subsets (batches) of the dataset.
- Faster than BGD but less noisy than SGD.

### 4. Momentum Gradient Descent (MGD)
- Introduces momentum to help smooth updates and speed up convergence.
- Reduces oscillations and helps avoid local minima.

## Results & Analysis
### Convergence Analysis key Observations:
- **Batch Gradient Descent (BGD)**: Smooth, stable convergence but slow.  
- **Stochastic Gradient Descent (SGD)**: Fast convergence, but noisy.  
- **Mini-Batch Gradient Descent (MBGD)**: A good balance between BGD and SGD.  
- **Momentum Gradient Descent (MGD)**: Shows initial oscillations but quickly converges.

