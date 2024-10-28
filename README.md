# Sake Recommendation Model

This project implements a sake recommendation system based on user preferences for specific sake attributes, including Price, SMV (Sake Meter Value), and Brewery. It calculates weighted preferences, scales feature values based on user likes and dislikes, and uses the k-Nearest Neighbors (k-NN) algorithm to recommend similar sakes.

## Project Overview

The model analyzes a small dataset with four sakes, considering three main features—Price, SMV, and Brewery (encoded numerically). By taking into account the sakes the user likes and dislikes, the system calculates preference weights, combines them, and identifies similar options in the dataset based on Euclidean distances in the weighted feature space.

## Key Steps and Workflow

### 1. Dataset and Features
   - **Dataset:** The dataset consists of four sakes (A, B, C, and D), with attributes such as Price, SMV, and Brewery (numerically encoded).
   - **User Preferences:** The user likes Sake A and Sake B, and dislikes Sake C. Sake D’s similarity to others will be assessed based on these preferences.

### 2. Calculating Like and Dislike Weights
   - **Objective:** Compute separate weights based on features of liked and disliked sakes to emphasize preferred attributes.
   - **Steps:**
     - **Like Weights:** Calculate the average of each feature for liked sakes:
       ```python
       Mean Price (Like) = (Price of Sake A + Price of Sake B) / 2
       Mean SMV (Like) = (SMV of Sake A + SMV of Sake B) / 2
       Mean Brewery (Like) = (Brewery of Sake A + Brewery of Sake B) / 2
       ```
       Example Calculation:
       - `Like Weights = [150, 2.5, 0.5]` for Price, SMV, and Brewery.
     
     - **Dislike Weights:** Compute the inverse of feature values for disliked sakes to minimize their influence:
       ```python
       Mean Price (Dislike) = 1 / (Price of Sake C + 1e-10)
       Mean SMV (Dislike) = 1 / (SMV of Sake C + 1e-10)
       Mean Brewery (Dislike) = 1 / (Brewery of Sake C + 1e-10)
       ```
       Example Calculation:
       - `Dislike Weights = [0.0067, 0.25, 0.5]` for Price, SMV, and Brewery.

### 3. Combining and Normalizing Weights
   - **Objective:** Merge like and dislike weights to reflect user preferences accurately.
   - **Steps:**
     - **Weight Multiplication:** For each feature, multiply like and dislike weights to balance user preferences:
       ```python
       Combined Weights = Like Weights * Dislike Weights
       ```
       Example Calculation:
       - `Combined Weights = [1, 0.625, 0.25]`

     - **Normalization:** Scale the combined weights between 0 and 1 to ensure consistent feature importance:
       ```python
       Min Weight = min(Combined Weights)
       Max Weight = max(Combined Weights)
       Range = Max Weight - Min Weight
       Normalized Weights = (Combined Weights - Min Weight) / Range
       ```
       Result:
       - `Normalized Weights = [1, 0.5, 0]`

### 4. Feature Scaling Using Weighted Preferences
   - **Objective:** Apply normalized weights to each feature in the dataset, highlighting user preferences.
   - **Implementation:** Multiply each feature of all sakes by the normalized weights:
     ```python
     Scaled Features (Sake A) = Original Features (Sake A) * Normalized Weights
     ```
   - **Example Calculations:**
     - Sake A: `[100 * 1, 3 * 0.5, 0 * 0] = [100, 1.5, 0]`
     - Sake B: `[200 * 1, 2 * 0.5, 1 * 0] = [200, 1, 0]`

### 5. k-NN for Sake Recommendation
   - **Objective:** Use the k-Nearest Neighbors algorithm to find the sakes most similar to those the user likes.
   - **Steps:**
     - Calculate the Euclidean distance between each sake and others based on scaled features.
     - Select the top `k` sakes with the smallest distances as recommendations.
   - **Example Use Case:** To find the 2 sakes most similar to Sake A, the model computes the distances to all other sakes and picks the 2 closest.

## Code Summary

1. **Data Loading and Preprocessing:**
   - Loads the dataset and preprocesses features, converting price and SMV to numerical values, and encoding categorical values (e.g., Brewery) numerically.

2. **Like and Dislike Weights Calculation:**
   - Computes the mean for liked sakes and the inverse for disliked sakes to create `like weights` and `dislike weights`.

3. **Combine and Normalize Weights:**
   - Balances preferences by combining like and dislike weights, followed by normalization.

4. **Feature Scaling:**
   - Applies normalized weights to each sake’s features to create weighted feature vectors.

5. **k-NN Recommendation System:**
   - Uses k-NN to find similar sakes based on Euclidean distances in the weighted feature space.

## Usage Instructions

### Installation
Ensure required libraries are installed:
```bash
pip install pandas scikit-learn
