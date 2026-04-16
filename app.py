from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import random
import os

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 HF API
HF_API = "https://api-inference.huggingface.co/models/Mounish79/mindcare-emotion-model"

headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

# INPUT
class TextInput(BaseModel):
    text: str


# 🔥 LABEL MAPPING (VERY IMPORTANT)
label_map = {
    "LABEL_0": "angry",
    "LABEL_1": "fear",
    "LABEL_2": "happy",
    "LABEL_3": "neutral",
    "LABEL_4": "sad",
    "LABEL_5": "stressed"
}


# 🔥 EMOTION DETECTION (FIXED)
def predict_emotion(text):
    t = text.lower()

    # 🔥 Crisis override (important for demo)
    if any(x in t for x in ["hurt myself","suicide","kill myself"]):
        return "sad", 0.95

    try:
        payload = {"inputs": text}
        response = requests.post(HF_API, headers=headers, json=payload)
        result = response.json()

        print("HF RESPONSE:", result)

        # Handle API error
        if isinstance(result, dict) and "error" in result:
            return "neutral", 0.5

        scores = result[0]
        best = max(scores, key=lambda x: x['score'])

        emotion = label_map.get(best["label"], "neutral")

        return emotion, float(best["score"])

    except Exception as e:
        print("ERROR:", e)
        return "neutral", 0.5


# 🔥 HUMAN RESPONSES
responses = {
    "sad": [
        "I'm really sorry you're feeling this way… I'm here with you.",
        "That sounds heavy… you don’t have to go through it alone.",
        "It’s okay to feel low sometimes… tell me more."
    ],
    "stressed": [
        "That sounds overwhelming… let's slow things down together.",
        "You're dealing with a lot right now… take a deep breath.",
        "One step at a time… you’ll get through this."
    ],
    "happy": [
        "That’s great to hear 😊",
        "I love that energy — keep it up!",
        "That sounds like a great moment!"
    ],
    "angry": [
        "I understand that frustration…",
        "Let’s pause before reacting.",
        "That sounds intense… want to talk it out?"
    ],
    "fear": [
        "That sounds scary… but you’re safe.",
        "Let’s slow down and breathe together.",
        "You’re not alone in this feeling."
    ],
    "neutral": [
        "I'm listening… tell me more.",
        "What's on your mind?",
        "I'm here for you."
    ]
}


coping = {
    "sad":["Talk to someone","Write feelings","Listen to music"],
    "stressed":["Deep breathing","Take a walk","Short break"],
    "angry":["Pause","Count to 10","Step away"],
    "fear":["Grounding","Breathing exercise"],
    "happy":["Enjoy the moment"],
    "neutral":["Stay mindful"]
}


relaxation = {
    "sad":["Meditation","Soft music"],
    "stressed":["4-4 breathing","Stretching"],
    "angry":["Cold water splash","Deep breathing"],
    "fear":["5-4-3-2-1 grounding"],
    "happy":["Celebrate"],
    "neutral":["Relax"]
}


youtube = {
    "sad":["https://youtu.be/2OEL4P1Rz04"],
    "stressed":["https://youtu.be/ZToicYcHIOU"],
    "angry":["https://youtu.be/9ZfN87gSjvI"],
    "fear":["https://youtu.be/VZ2rKq9c3qM"],
    "happy":["https://youtu.be/3GwjfUFyY6M"],
    "neutral":["https://youtu.be/inpok4MKVLM"]
}


# 🔥 RISK DETECTION
def detect_risk(text):
    return any(x in text.lower() for x in [
        "hurt myself","suicide","end my life"
    ])


# 🔥 TREND TRACKING
emotion_history = []

def mental_state(hist):
    if hist.count("sad") >= 3:
        return "⚠️ You seem emotionally low recently. Consider talking to someone."
    return "Stable 👍"


# 🚀 MAIN API
@app.post("/analyze")
def analyze(data: TextInput):

    emotion, conf = predict_emotion(data.text)

    emotion_history.append(emotion)
    if len(emotion_history) > 5:
        emotion_history.pop(0)

    crisis = detect_risk(data.text)

    reply = random.choice(responses.get(emotion, responses["neutral"]))

    if crisis:
        reply += " 🚨 Please seek professional help immediately."

    return {
        "emotion": emotion,
        "confidence": round(conf*100,2),
        "response": reply,
        "coping": coping.get(emotion, []),
        "relaxation": relaxation.get(emotion, []),
        "youtube": youtube.get(emotion, []),
        "is_crisis": crisis,
        "trend": emotion_history,
        "mental_state": mental_state(emotion_history)
    }