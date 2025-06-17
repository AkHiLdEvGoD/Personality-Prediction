from api.app import app
import unittest
from fastapi.testclient import TestClient

class FastAPITest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "up"})

    def test_predict_endpoint(self):
        payload = {
            'time_spent_Alone': 4,
            'stage_fear': False,
            'social_event_attendance': 4,
            'going_outside': 6,
            'drained_after_socializing': False,
            'friends_circle_size': 13,
            'post_frequency': 5
        }

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())

if __name__ == "__main__":
    unittest.main()