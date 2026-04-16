from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import random

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔑 HUGGING FACE API
HF_API = "https://api-inference.huggingface.co/models/Mounish79/mindcare-emotion-model"

headers = {
    "Authorization": "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"
}

class TextInput(BaseModel):
    text: str


# 🔥 EMOTION DETECTION (HF API)
def predict_emotion(text):
    t = text.lower()

    # Crisis override
    if any(x in t for x in ["hurt myself","suicide","kill myself"]):
        return "sad", 0.95

    try:
        payload = {"inputs": text}
        response = requests.post(HF_API, headers=headers, json=payload)
        result = response.json()

        scores = result[0]
        best = max(scores, key=lambda x: x['score'])

        return best['label'].lower(), float(best['score'])

    except:
        return "neutral", 0.5


# 🔥 RESPONSES
responses = {
    "sad": [
        "I'm really sorry you're feeling like this… I'm here with you.",
        "That sounds heavy… you don’t have to carry it alone.",
        "I can feel this matters to you… tell me more."
    ],
    "stressed": [
        "That sounds overwhelming… let's slow things down together.",
        "You're handling a lot… take a deep breath.",
        "One step at a time… you're not alone."
    ],
    "happy": [
        "That's amazing 😊 keep that energy!",
        "I love hearing this — tell me more!",
        "That’s a great moment for you!"
    ],
    "angry": [
        "I understand that frustration…",
        "Let’s pause before reacting.",
        "That sounds intense… want to talk it out?"
    ],
    "fear": [
        "That sounds scary… but you're safe.",
        "Let's breathe slowly together.",
        "You're not alone in this feeling."
    ],
    "neutral": [
        "I'm here… tell me more.",
        "What's on your mind?",
        "I'm listening."
    ]
}

coping = {
    "sad":["Talk to someone","Write feelings","Music"],
    "stressed":["Deep breathing","Short walk","Take a break"],
    "angry":["Pause","Count to 10"],
    "fear":["Grounding","Breathing"],
    "happy":["Enjoy the moment"],
    "neutral":["Stay mindful"]
}

relaxation = {
    "sad":["Meditation","Soft music"],
    "stressed":["4-4 breathing","Stretch"],
    "angry":["Cold water","Step away"],
    "fear":["5-4-3-2-1 technique"],
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

def detect_risk(text):
    return any(x in text.lower() for x in [
        "hurt myself","suicide","end my life"
    ])

emotion_history = []

def mental_state(hist):
    if hist.count("sad") >= 3:
        return "⚠️ You seem emotionally low recently. Consider talking to someone."
    return "Stable 👍"


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