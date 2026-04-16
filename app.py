# ===== FINAL BACKEND (FULLY MERGED) =====

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F
import random

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ MODEL LOAD
model = DistilBertForSequenceClassification.from_pretrained("emotion_model")
tokenizer = DistilBertTokenizer.from_pretrained("emotion_model")
model.eval()

labels = ['angry','fear','happy','neutral','sad','stressed']


# =========================
# 📥 INPUT MODELS
# =========================
class TextInput(BaseModel):
    text: str



# =========================
# 🔥 EMOTION DETECTION
# =========================
def predict_emotion(text):
    t = text.lower()

    # 🔴 strong rules (fixes your earlier issue)
    if any(x in t for x in ["hurt myself","kill myself","suicide","end my life"]):
        return "sad",0.95
    if "anxious" in t or "confused" in t:
        return "stressed",0.9

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return labels[pred.item()], float(conf.item())


# =========================
# 💬 HUMAN-LIKE RESPONSES
# =========================
responses = {
    "sad":[
        "I'm really sorry you're feeling like this… you don’t have to go through it alone.",
        "That sounds really heavy… I’m here with you, tell me more.",
        "It’s okay to feel low sometimes… I’m listening."
    ],
    "stressed":[
        "That sounds overwhelming… try slowing things down for a moment.",
        "You're dealing with a lot right now… take a deep breath.",
        "Let’s take it one step at a time… you’re not alone."
    ],
    "happy":[
        "That’s amazing to hear! 😊",
        "I love that energy — keep it going!",
        "That sounds like a great moment for you!"
    ],
    "angry":[
        "I can understand that frustration…",
        "Let’s pause for a moment before reacting.",
        "That sounds intense… want to talk it out?"
    ],
    "fear":[
        "That sounds scary… but you’re safe right now.",
        "Let’s slow down and breathe together.",
        "You’re not alone in this feeling."
    ],
    "neutral":[
        "I’m here… tell me more.",
        "What’s on your mind?",
        "I’m listening to you."
    ]
}


# =========================
# 💡 COPING + RELAXATION
# =========================
coping = {
    "sad":["Talk to someone","Write feelings","Music"],
    "stressed":["Deep breathing","Short walk","Break"],
    "angry":["Pause","Count 10"],
    "fear":["Grounding","Breathing"],
    "happy":["Enjoy moment"],
    "neutral":["Mindfulness"]
}

relaxation = {
    "sad":["Meditation","Soft music"],
    "stressed":["4-4 breathing","Stretch"],
    "angry":["Cold water","Step away"],
    "fear":["5-4-3-2-1 grounding"],
    "happy":["Celebrate"],
    "neutral":["Relax"]
}


# =========================
# 🎥 YOUTUBE SUPPORT
# =========================
youtube = {
    "sad":["https://youtu.be/2OEL4P1Rz04"],
    "stressed":["https://youtu.be/ZToicYcHIOU"],
    "angry":["https://youtu.be/9ZfN87gSjvI"],
    "fear":["https://youtu.be/VZ2rKq9c3qM"],
    "happy":["https://youtu.be/3GwjfUFyY6M"],
    "neutral":["https://youtu.be/inpok4MKVLM"]
}


# =========================
# ⚠️ CRISIS DETECTION
# =========================
def detect_risk(text):
    return any(x in text.lower() for x in [
        "hurt myself","kill myself","suicide","end my life"
    ])


# =========================
# 📊 TREND TRACKING
# =========================
emotion_history = []


# =========================
# 🧠 MENTAL STATE
# =========================
def mental_state(hist):
    if len(hist) < 3:
        return None

    if hist.count("sad") >= 3:
        return "⚠️ You seem emotionally low lately. Consider talking to someone."

    if hist.count("stressed") >= 3:
        return "⚠️ You seem stressed recently. Try relaxation."

    return None


# =========================
# 🚀 MAIN CHAT API
# =========================
@app.post("/analyze")
def analyze(data:TextInput):

    emotion,conf = predict_emotion(data.text)

    emotion_history.append(emotion)
    if len(emotion_history)>5:
        emotion_history.pop(0)

    crisis = detect_risk(data.text)

    reply = random.choice(responses[emotion])

    if crisis:
        reply += " 🚨 Please seek professional help immediately."

    return {
        "emotion":emotion,
        "confidence":round(conf*100,2),
        "response":reply,
        "coping":coping[emotion],
        "relaxation":relaxation[emotion],
        "youtube":youtube[emotion],
        "is_crisis":crisis,
        "trend":emotion_history,
        "mental_state":mental_state(emotion_history),
        "show_assessment": emotion=="neutral" or conf<0.6
    }


