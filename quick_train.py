#!/usr/bin/env python3
"""
Quick model retrain for Streamlit Cloud compatibility
This ensures the model is compatible with the deployment environment
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def quick_train():
    """Quick training with compatible SimpleImputer"""
    print("Loading data...")
    df = pd.read_csv('Thales_Group_Manufacturing.csv')
    
    # Basic preprocessing
    df = df.dropna()
    df = df.head(10000)  # Use subset for quick training
    
    # Features and target
    feature_cols = [c for c in df.columns if c not in ['Date', 'Timestamp', 'Efficiency_Status']]
    X = df[feature_cols]
    y = df['Efficiency_Status']
    
    # Encode categorical variables
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Create preprocess pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', Pipeline([
                    ('imputer2', SimpleImputer(strategy='most_frequent'))
                ]))
            ]), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Train model
    print("Training model...")
    X_processed = preprocessor.fit_transform(X)
    
    rf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    rf.fit(X_processed, y_encoded)
    
    # Create pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])
    
    # Save model
    artifacts = {
        'model': model,
        'classes': le.classes_,
        'best_model_name': 'rf',
        'feature_names': feature_cols
    }
    
    joblib.dump(artifacts, 'model.joblib')
    print("Model saved successfully!")
    print(f"Classes: {le.classes_}")
    
    # Test accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    quick_train()
