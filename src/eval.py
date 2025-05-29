import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

from data_processing import JobPostingData
from model import JobPostingClassifier

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    data_processor = JobPostingData('data/job_postings.csv')
    data = data_processor.get_data()
    
    # Convert test data to PyTorch tensor
    from scipy.sparse import csr_matrix

    if isinstance(data['X_test'], csr_matrix):
        X_test = torch.FloatTensor(data['X_test'].toarray())  # Convert sparse matrix to dense tensor
    else:
        X_test = torch.FloatTensor(data['X_test'])  # If already dense, directly convert to tensor

    y_test = torch.LongTensor(data['y_test'])  # Correct the dtype for y_test
    
    # Create test data loader
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load trained model
    input_size = X_test.shape[1]
    model = JobPostingClassifier(input_size)
    model.load_state_dict(torch.load('models/job_posting_classifier.pth'))  # Fixed model loading
    model = model.to(device)
    
    # Evaluate model
    print('Evaluating model...')
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            predictions = torch.sigmoid(predictions)  # For binary classification
            predictions = (predictions > 0.5).int()  # Thresholding at 0.5 for binary classification
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(labels, predictions))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions)
    
    # Print feature importance
    feature_importance = model.model[0].weight.data.cpu().numpy()  # Assuming model[0] is the first layer
    top_features = np.argsort(np.abs(feature_importance).mean(axis=0))[-10:]
    
    print('\nTop 10 Most Important Features:')
    for idx in reversed(top_features):
        feature_name = data['feature_names'][idx]
        importance = np.abs(feature_importance).mean(axis=0)[idx]
        print(f'{feature_name}: {importance:.4f}')

if __name__ == '__main__':
    main()
