from api.app import app
import unittest
from fastapi.testclient import TestClient

class FastAPITest(unittest.TestCase):

    def test_health_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "up"})

    def test_predict_endpoint(self):
        payload = {
            'time_spend_alone': 4,
            'stage_fear': False,
            'social_event_attendance': 4,
            'going_outside': 6,
            'drained_after_socializing': False,
            'friends_circle_size': 13,
            'post_frequency': 5
        }

        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Predicted Personality", response.json())

if __name__ == "__main__":
    unittest.main()