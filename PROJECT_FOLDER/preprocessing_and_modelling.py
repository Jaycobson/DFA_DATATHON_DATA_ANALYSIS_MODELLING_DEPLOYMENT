import os
import pickle
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from collecting_data_from_db import getting

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'datasets')
data = os.path.join(dataset_dir,'school_dataset.csv')
encoder = os.path.join(current_dir,'encoders')
models_dir = os.path.join(current_dir,'models')
metrics_dir = os.path.join(current_dir,'metrics_result/confusion_matrix')
metrics_dir_model = os.path.join(current_dir, 'metrics_result/model_metrics')


# Creating a directory to store encoders and models if it doesn't exist
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(encoder, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(metrics_dir_model, exist_ok=True)  # Ensure the directory exists






class StudentPerformanceModel:
    def __init__(self, df):
        self.df = df
        self.target_column = 'target'
        self.encoders = {}
        self.scaler = StandardScaler()
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
        }
        self.metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
        
    def preprocess(self):
        # Removing columns containing 'id'
        self.df = self.df.drop(columns=[col for col in self.df.columns if 'id' in col.lower()])

         # Encoding categorical features
        for col in self.df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
            # Saving encoder
            with open(f'{encoder}/{col}_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
        
        # Separating target and features
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

       
        # Standardizing numerical features
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns)
        with open(f'{encoder}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
    
    def plot_and_save_confusion_matrix(self, y_test, y_pred, model_name):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{metrics_dir}/confusion_matrix_{model_name}.png')
        plt.close()

    def train_and_evaluate(self):
        best_accuracy = 0
        best_model = None
        best_model_name = ""

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Plotting and save confusion matrix
            self.plot_and_save_confusion_matrix(self.y_test, y_pred, model_name)

            # Appending metrics to dataframe
            result = pd.DataFrame({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }, index = [1])

            self.metrics_df = pd.concat([self.metrics_df, result], axis = 0)

            print(f"\n{model_name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Classification Report:\n", classification_report(self.y_test, y_pred))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

        # Saving the best model
        with open(f'{models_dir}/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print(f"\nBest model is {best_model_name} with accuracy {best_accuracy:.4f}. Model saved to 'models' directory.")
    
    def save_metrics(self):
        # Saving metrics dataframe to CSV
        self.metrics_df.to_csv(f'{metrics_dir_model}/model_metrics.csv', index=False)
        print("Model metrics saved to 'metrics_result/model_metrics.csv'.")
    
    def evaluate_prediction(self, prob):
        if prob > 0.75:
            print("You will pass strongly!")
        elif 0.5 < prob <= 0.75:
            print("You will pass but you need to study more.")
        elif 0.25 < prob <= 0.5:
            print("You may fail, but with more effort you can pass.")
        else:
            print("You are at risk of failing. You need to work a lot harder.")
            
    def run_pipeline(self):
        # Preprocessing the data
        self.preprocess()
        # Training and evaluate models
        self.train_and_evaluate()
        # Saving the metrics
        self.save_metrics()


if __name__ == '__main__':
    getting()
    
    df = pd.read_csv(data)

    ls_to_drop = ['student_id', 'student_name','parent_id', 'parent_name','phone_number',
                'email','comments','subject_id',
                'academic_record_id', 'qualification', 'evaluation_date'
    ]
    df = df.drop(ls_to_drop,axis = 1)
    df['target'] = df['target'].map({'Fail':0,'Pass':1})
    # Instantiating and running the model pipeline
    
    model_pipeline = StudentPerformanceModel(df)
    model_pipeline.run_pipeline()
    print('successful')

