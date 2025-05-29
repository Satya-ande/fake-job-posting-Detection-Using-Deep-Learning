import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class JobPostingData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def load_data(self):
        """Load the job posting dataset"""
        df = pd.read_csv(self.data_path)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the job posting data"""
        # Combine relevant text fields
        text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        df['text'] = df[text_fields].fillna('').apply(lambda x: ' '.join(x), axis=1)
        
        # Convert fraudulent column to label (0 for real, 1 for fake)
        df['label'] = df['fraudulent']
        
        # Add numerical features
        df['has_company_logo'] = df['has_company_logo'].fillna(0)
        df['has_questions'] = df['has_questions'].fillna(0)
        df['telecommuting'] = df['telecommuting'].fillna(0)
        
        # Convert salary range to numerical (if available)
        df['salary_range'] = df['salary_range'].fillna('')
        df['has_salary'] = df['salary_range'].apply(lambda x: 1 if x else 0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features using TF-IDF vectorization"""
        # Fit and transform the text data
        X_text = self.vectorizer.fit_transform(df['text'])
        
        # Add numerical features
        numerical_features = ['has_company_logo', 'has_questions', 'telecommuting', 'has_salary']
        X_numerical = df[numerical_features].values
        
        # Combine text and numerical features
        X = np.hstack([X_text.toarray(), X_numerical])
        y = df['label'].values
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def get_data(self):
        """Main method to get processed data"""
        # Load and preprocess data
        df = self.load_data()
        df = self.preprocess_data(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.vectorizer.get_feature_names_out()
        } 