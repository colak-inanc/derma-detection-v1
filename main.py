from src.md.predict_router import route_prediction
from src.md.explain_with_gemini import explain_result

sample_input = {
    "image_path": "data-sets/Dataset/s-test/Acne and Rosacea Photos/img123.jpg",
    "cbc_data": {
        "HCT": 50.5,
        "MCV": 84.7,
        "PDW": 17.8,

    },
    "symptoms": ["itching", "scaling"]
}
prediction_results = route_prediction(sample_input)

summary_prompt = f"Aşağıdaki verileri değerlendir:\n\n{prediction_results}\n\nDurumu açıkla."
explanation = explain_result(summary_prompt)

print("🔍 Tahminler:")
print(prediction_results)

print("\nAI Açıklaması:")
print(explanation)
