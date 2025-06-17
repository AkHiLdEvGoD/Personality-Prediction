import unittest
import joblib
import mlflow
import pandas as pd
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dagshub_token = os.getenv("PERSONALITY_TEST")
        if not dagshub_token:
            raise EnvironmentError("PERSONALITY_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "AkHiLdEvGoD"
        repo_name = "Personality-Prediction"

        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        cls.preprcessing_pipeline = joblib.load('local_Storage/models/preprocessing_pipeline.pkl')
        cls.test_data = pd.read_csv("local_Storage/data/processed/test.csv")

    @staticmethod
    def get_latest_model_version(model_name,stage='Staging'):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)
    
    def test_model_signature(self):
        input_df = pd.DataFrame({
            'Time_spent_Alone' : 4.0,
            'Stage_fear' : 'No',
            'Social_event_attendance' : 4.0,
            'Going_outside':6.0,
            'Drained_after_socializing': 'No',
            'Friends_circle_size' : 13.0,
            'Post_frequency':5.0,
        })
        input_df['Offline_social_activity'] = input_df['Social_event_attendance'] * input_df['Going_outside']

        preprocess_data = self.preprcessing_pipeline.transform(input_df)
        prediction = self.new_model.predict(preprocess_data)

        self.assertEqual(input_df.shape[1], len(self.preprcessing_pipeline.get_feature_names_out()))

        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        X = self.test_data.drop(columns=['target'])
        y = self.test_data['target']
        
        y_pred = self.new_model.predict(X)

        accuracy_new = accuracy_score(y, y_pred)
        precision_new = precision_score(y, y_pred)
        recall_new = recall_score(y, y_pred)
        f1_new = f1_score(y, y_pred)

        expected_accuracy = 0.80
        expected_precision = 0.80
        expected_recall = 0.80
        expected_f1 = 0.80

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')


if __name__ == "__main__":
    unittest.main()