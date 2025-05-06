import google.generativeai as genai

genai.configure(api_key="AIzaSyCdgUpKijJBSLQ3_BWjy0HQDPcxKzVNRyM")

def explain_result(prompt_text):
    model = genai.GenerativeModel("gemini-2.0-flash")  # Model adı düzeltildi
    response = model.generate_content(prompt_text)
    return response.text


# Örnek kullanım
# print(explain_result("The CBC values indicate elevated HCT and low MCV. What could be the possible interpretation?"))
