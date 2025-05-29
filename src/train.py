import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_processing import JobPostingData
from model import JobPostingClassifier

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the model"""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions = model.predict(batch_X)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    return np.array(all_predictions), np.array(all_labels)

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
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    data_processor = JobPostingData('data/job_postings.csv')
    data = data_processor.get_data()
    
    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(data['X_train'])
    X_test = torch.FloatTensor(data['X_test'])
    y_train = torch.FloatTensor(data['y_train'])
    y_test = torch.FloatTensor(data['y_test'])
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = JobPostingClassifier(input_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print('Starting training...')
    train_model(model, train_loader, criterion, optimizer, device)
    
    # Evaluate model
    print('\nEvaluating model...')
    predictions, labels = evaluate_model(model, test_loader, device)
    
    # Print classification report
    print('\nClassification Report:')
    print(classification_report(labels, predictions))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/job_posting_classifier.pth')
    print('\nModel saved to models/job_posting_classifier.pth')

if __name__ == '__main__':
    main() 