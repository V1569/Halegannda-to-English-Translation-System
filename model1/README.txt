
Translator v2 package
---------------------
Contents:
- translator_v2_artifacts/ (vectorizer.pkl, data.pkl)
- translator_v2_app.py (Flask app)
- hale_hosa_dataset.json (dataset)

How to run locally:
1. pip install flask scikit-learn
2. python translator_v2_app.py
3. Open http://localhost:7861

API:
POST /translate
Payload: { "text": "Hale sentence", "mode": "simple"|"poetic"|"elevated" }
Response: { "hosa": "...", "english": "...", "similarity": 0.XXX }
