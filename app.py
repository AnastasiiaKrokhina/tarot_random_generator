import datetime
import os
import requests
import pathlib
import streamlit as st
import pandas as pd

from cards_meta import pick_daily_cards

########################
# CONFIG / CONSTANTS
########################

# Важно: положи свой Hugging Face token в переменную окружения
# HF_TOKEN="hf_...."
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# (пример: любая доступная инструктажная модель. ты можешь поменять на ту, что бесплатна/доступна у тебя)

DISCLAIMER_TEXT = (
    "👹Это развлекательный контент. "
    "Ничего ниже не является медицинской, психологической, юридической или финансовой рекомендацией."
    "All tarot card images were generated using AI (ChatGPT/DALL·E) for artistic and entertainment purposes."
)

st.set_page_config(
    page_title="Tarot of the Day",
    page_icon="🔮",
    layout="centered",
)

@st.cache_data
def load_tarot_dataset():
    df = pd.read_csv("tarot_readings.csv", encoding="utf-8")
    return df
########################
# HELPERS
########################

# def find_reading_for(cards, df):
#     """
#     cards: список имён карт (например ["The Fool", "The Magician", "The Empress"])
#     df: pandas.DataFrame с колонками 'Card 1', 'Card 2', 'Card 3', 'Reading'
#     """
#     target_set = {c.lower().strip() for c in cards}

#     for _, row in df.iterrows():
#         row_set = {str(row["Card 1"]).lower().strip(),
#                    str(row["Card 2"]).lower().strip(),
#                    str(row["Card 3"]).lower().strip()}
#         if row_set == target_set:
#             return row["Reading"]

#     return None

def build_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        key = frozenset([
            str(row["Card 1"]).lower().strip(),
            str(row["Card 2"]).lower().strip(),
            str(row["Card 3"]).lower().strip()
        ])
        lookup[key] = row["Reading"]
    return lookup


def find_reading_for(cards, lookup):
    key = frozenset(c.lower().strip() for c in cards)
    return lookup.get(key)


def build_prompt(cards):
    """
    cards: список карт из pick_daily_cards
    Мы сформируем подсказку для LLM.
    """
    lines = []
    lines.append("You are a playful tarot storyteller.")
    lines.append("You give uplifting, empathetic, magical-feeling guidance.")
    lines.append("You NEVER give medical, legal or financial advice.")
    lines.append("Write in Russian, warm, feminine tone, 150-220 words total.")
    lines.append("Structure: 1) короткий общий вайб дня, 2) что делать, 3) мягкий совет про заботу о себе.")
    lines.append("Don't mention 'upright' or 'reversed' literally. Just reflect the vibe subtly.")
    lines.append("Cards pulled:")

    for idx, c in enumerate(cards, start=1):
        orientation = "reversed" if c["is_reversed"] else "upright"
        lines.append(
            f"{idx}. {c['name']} ({orientation}) "
            f"- keywords: {', '.join(c[orientation])}. "
            f"blurb: {c['blurb']}"
        )

    prompt = "\n".join(lines)
    return prompt

def call_hf_inference(prompt: str) -> str:
    """
    Вызов Hugging Face Inference API.
    Для бесплатного юзкейса: можешь создать бесплатный аккаунт HF и получить токен.
    В проде лучше делать через secrets.
    """
    hf_token = os.environ.get("HF_TOKEN", None)
    if not hf_token:
        # Без токена: fallback - просто вернём заглушку.
        return (
            "Сегодняшняя энергия просит тебя быть мягкой к себе. "
            "Ты не обязана бежать быстрее всех. "
            "Сделай один шаг, но сделай его осознанно, с уважением к своим границам. "
            "Замечай сигналы тела и не игнорируй усталость. "
            "Ты не теряешь время — ты выращиваешь устойчивость."
        )

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 220,
            "temperature": 0.7,
        },
    }

    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        return f"LLM недоступна сейчас. Но карта дня говорит: доверься процессу. ({resp.status_code})"

    data = resp.json()
    # HF иногда возвращает список [{generated_text: "..."}], иногда string
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]

    # Если модель вернула просто текст
    if isinstance(data, str):
        return data

    # Если модель вернула tokens массив
    return str(data)


def show_card(card, set_name):
    """
    card: один dict карты
    Показываем картинку и подпись.
    """
    orientation_label = "перевернута" if card["is_reversed"] else "прямая"

    st.image(pathlib.Path("cards",set_name, card["image"]), width=200)
    st.markdown(
        f"**{card['name']}** ({orientation_label})  \n"
        f"_Ключевые идеи:_ {', '.join(card['reversed' if card['is_reversed'] else 'upright'])}  \n"
        f"{card['blurb']}"
    )

########################
# UI LAYOUT
########################

st.title("🔮 Tarot of the Day")
st.caption(DISCLAIMER_TEXT)

# сид = сегодняшняя дата, чтобы был стабильный прогноз на день
today = datetime.date.today().isoformat()  # '2025-10-28' и т.д.
st.sidebar.header("Настройки")
st.sidebar.write("Этот расклад основан на сегодняшней дате:")
st.sidebar.code(today)

num_cards = st.sidebar.slider("Сколько карт тянуть?", min_value=3, max_value=4, value=3)
# (держим 3 фиксированно, но слайдер даёт чувство интерактива; можешь расширить позже)

cards_today = pick_daily_cards(seed_str=today
                               , n_cards=num_cards)

st.subheader("Твои карты сегодня")
cols = st.columns(len(cards_today))
for col, c in zip(cols, cards_today):
    with col:
        show_card(c, "minecraft")

st.subheader("Сообщение дня 🌙")
prompt = build_prompt(cards_today)
reading_text = call_hf_inference(prompt)
st.write(reading_text)
df = load_tarot_dataset()
card_names = [c["name"] for c in cards_today]
reading_from_csv = find_reading_for(card_names, df)

if reading_from_csv:
    st.subheader("Сообщение дня 🌙")
    st.write(reading_from_csv)
else:
    # fallback — если не найдено, вызвать LLM
    prompt = build_prompt(cards_today)
    reading_text = call_hf_inference(prompt)
    st.subheader("Сообщение дня 🌙")
    st.write(reading_from_csv)


st.markdown("---")
st.caption("Это не совет по здоровью, финансам или юриспруденции. Это мягкая подсказка-вдохновение ✨")
