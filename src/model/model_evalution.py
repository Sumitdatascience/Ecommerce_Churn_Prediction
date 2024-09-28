import pickle
import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load the trained model
def load_model(model_path: str):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {model_path}")
    return model


# Load test data
def load_test_data(x_test_path: str, y_test_path: str):
    x_test = pd.read_csv(
        x_test_path,
        usecols=['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferredOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromLastYear', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']
    )
    y_test = pd.read_csv(y_test_path)
    print("Test data loaded successfully")
    return x_test, y_test


# Make predictions using the model
def predict(model, x_test):
    y_pred = model.predict(x_test)
    print("Predictions generated successfully")
    return y_pred


# Calculate and save metrics
def calculate_metrics(y_test, y_pred, output_path: str):
    acc_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    metrics_dict = {
        'accuracy': acc_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    # Save metrics to a JSON file
    with open(output_path, 'w') as file:
        json.dump(metrics_dict, file, indent=4)
    
    print(f"Metrics saved to {output_path}")
    return metrics_dict


# Main function to load model, test data, and calculate metrics
def main():
    # Load model
    model = load_model('model.pkl')
    
    # Load test data
    x_test, y_test = load_test_data(
        r'.\data\processed\x_test.csv',
        r'.\data\processed\y_test.csv'
    )
    
    # Generate predictions
    y_pred = predict(model, x_test)
    
    # Calculate and save metrics
    metrics_dict = calculate_metrics(y_test, y_pred, "metrics.json")
    print("Metrics calculated:", metrics_dict)


if __name__ == "__main__":
    main()
