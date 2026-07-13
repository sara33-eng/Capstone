# PART 4 - LLM POWERED MODEL EXPLANATION PIPELINE

import json
import os
import re

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from jsonschema import validate, ValidationError

URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4.1-mini"
load_dotenv()
api_key = os.getenv("LLM_API_KEY")

if api_key is None:
    raise ValueError("LLM_API_KEY not found. Set it as environment variable.")
print("API Key Loaded Successfully")

def call_llm(system_prompt,user_prompt,temperature=0.0,max_tokens=512):
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Churn Explanation Pipeline"
    }
    response = requests.post(URL,headers=headers,json=payload)
    if response.status_code != 200:
        print("LLM API Error")
        print(response.status_code)
        print(response.text)
        return None
    result = response.json()
    return result["choices"][0]["message"]["content"]


def has_pii(text):
    email_pattern = (r'[a-zA-Z0-9_.+-]+@'r'[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    phone_pattern = (r'\b\d{10}\b|'r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b')
    return bool(re.search(email_pattern, text)or re.search(phone_pattern, text))

def safe_call_llm(system_prompt,user_prompt,temperature=0.0,max_tokens=512):
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
    return call_llm(system_prompt,user_prompt,temperature,max_tokens)

# 4. TEST LLM CONNECTION
print("\nTesting LLM Connection")
test_response = safe_call_llm("You are a helpful assistant.","Reply only with the word hello",temperature=0)
print(test_response)

# 5. LOAD BEST MODEL
print("\nLoading Best Model")
model = joblib.load("best_model.pkl")
print("Model Loaded:")
print(model)

# 6. CREATE THREE HANDCRAFTED INPUT RECORDS
customer_records = [
    {

        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "TotalCharges": 6500.50

    },

    {

        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "TotalCharges": 450.75

    },

    {

        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 30,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "TotalCharges": 2200.30

    }

]
customers_df = pd.DataFrame(customer_records)
print(customers_df)
# 7. MODEL PREDICTION
predictions = model.predict(customers_df)
probabilities = model.predict_proba(customers_df)[:, 1]
prediction_inputs = []
for i, row in customers_df.iterrows():
    prediction_inputs.append(
        {
            "features": row.to_dict(),
            "predicted_class": int(predictions[i]),
            "probability": float(probabilities[i])
        }
    )
print("\nPredictions")
print(pd.DataFrame(
        [
            {
                "Prediction": x["predicted_class"],
                "Probability": x["probability"]
            }
            for x in prediction_inputs
        ]
    )
)
# 8. LLM PROMPT + SCHEMA
SYSTEM_PROMPT = """
You are an AI assistant explaining machine learning predictions.
You explain customer churn predictions.
Input contains:
- customer features
- predicted class
- churn probability
Return ONLY valid JSON.
Required JSON fields:
prediction_label:
string
confidence_level:
one of low, medium, high
top_reason:
main reason influencing prediction
second_reason:
secondary reason
next_step:
recommended business action

Do not include markdown.
Do not include additional text.
"""

USER_TEMPLATE = """
Explain this churn prediction.
Customer Features:
{features}
Model Prediction:
Class:
{prediction}
Probability:
{probability}
Return JSON only.
"""

SCHEMA = {
   "type": "object",
    "properties": {
        "prediction_label":
            {
                "type": "string"
            },
        "confidence_level":
            {
                "type": "string",
                "enum":
                    [
                        "low",
                        "medium",
                        "high"
                    ]
            },
        "top_reason":
            {
                "type": "string"
            },
        "second_reason":
            {
                "type": "string"
            },
        "next_step":
            {
                "type": "string"
            }
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step"
    ]
}

def validate_response(response):
    fallback = {
        "prediction_label": None,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "next_step": None
    }
    try:
        data = json.loads(response.strip())
    except Exception as e:
        print("JSON Parsing Failed",e)
        return fallback
    try:
        validate(instance=data,schema=SCHEMA)
        return data
    except ValidationError as e:
        print(
            "Schema Error",
            e.message
        )

        return fallback

# 9. RUN LLM EXPLANATION PIPELINE
final_results = []
for item in prediction_inputs:
    user_prompt = USER_TEMPLATE.format(
        features=json.dumps(item["features"],indent=2),
        prediction=item["predicted_class"],
        probability=item["probability"]
    )
    print("\nUSER INPUT")
    print(user_prompt)
    llm_response = safe_call_llm(SYSTEM_PROMPT,user_prompt,temperature=0)
    print("\nRAW RESPONSE")
    print(llm_response)
    if llm_response:
        explanation = validate_response(llm_response)
        status = "PASS"
    else:
        explanation = validate_response("{}")
        status = "BLOCKED"
    final_results.append(
        {
           "feature_input": item["features"],
            "predicted_class":item["predicted_class"],
            "probability":item["probability"],
            "explanation":explanation,
            "validation_status":status
        }
    )
results_df = pd.DataFrame(final_results)
print("\nFINAL RESULTS")
print(results_df)
results_df.to_json("llm_explanation_results.json",indent=4)

# 10. PII TEST
print("\nPII TEST")
safe_call_llm(SYSTEM_PROMPT,"Customer email is test@gmail.com")
safe_call_llm(SYSTEM_PROMPT,"Customer tenure is 24 months")

# 11. TEMPERATURE COMPARISON
temperature_results = []
for index, item in enumerate(prediction_inputs,start=1):
    user_prompt = USER_TEMPLATE.format(
        features=json.dumps(item["features"],indent=2),
        prediction=item["predicted_class"],
        probability=item["probability"]
    )
    output_0 = safe_call_llm(SYSTEM_PROMPT,user_prompt,temperature=0)
    output_07 = safe_call_llm(SYSTEM_PROMPT,user_prompt,temperature=0.7)
    temperature_results.append(
        {
            "Input":f"Customer {index}",
            "Output_temp_0":output_0,
            "Output_temp_0.7":output_07,
            "Key Difference":
                "Same output"
                if output_0 == output_07
                else
                "Different wording/reasoning"
        }
    )
temperature_df = pd.DataFrame(temperature_results)
print(temperature_df)
temperature_df.to_csv("temperature_comparison.csv",index=False)
print("\nTemperature comparison saved.")

# KEY DIFFERENCE SUMMARY
temperature_df["Key Difference"] = ""
for i in range(len(temperature_df)):
    output0 = str(temperature_df.loc[i, "Output_Temp_0"])
    output07 = str(temperature_df.loc[i, "Output_Temp_0.7"])
    if output0 == output07:
        difference = "Outputs are identical"
    else:
        difference = ("Output variation observed. ""Temperature 0.7 generated different wording or reasoning.")
    temperature_df.loc[i, "Key Difference"] = difference
print(temperature_df[["Input", "Output_Temp_0", "Output_Temp_0.7", "Key Difference"]])
temperature_df.to_csv("temperature_analysis.csv", index=False)
