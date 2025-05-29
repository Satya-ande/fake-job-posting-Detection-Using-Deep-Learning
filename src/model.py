import torch
import torch.nn as nn
import torch.nn.functional as F

class JobPostingClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32]):
        super(JobPostingClassifier, self).__init__()
        
        # Create layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        """Make predictions on input data"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            predictions = torch.sigmoid(outputs)
            return (predictions > 0.5).float()
    
    def save(self, path):
        """Save model state"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load model state"""
        self.load_state_dict(torch.load(path))
        self.eval() 