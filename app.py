import datetime
import os
import requests
import pathlib
import streamlit as st
import pandas as pd

from cards_meta import pick_daily_cards

import streamlit as st
import base64

st.set_page_config(
    page_title="Tarot of the Day",
    page_icon="🔮",
    layout="centered",
)

# Функция для добавления фона
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* делаем хэдер прозрачным */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* делаем сайдбар с полупрозрачным фоном чтобы текст читался */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(0,0,0,0.35);
        color: white;
        border-radius: 8px;
        padding: 1rem;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

add_bg_from_local("minecraf_background.png")


def white_text_with_black_outline():
    st.markdown("""
    <style>
    /* Общие стили для текста */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: white !important;
        text-shadow:
            -1px -1px 0 black,
             1px -1px 0 black,
            -1px  1px 0 black,
             1px  1px 0 black;
    }
    </style>
    """, unsafe_allow_html=True)

white_text_with_black_outline()

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



import json
from pathlib import Path

CACHE_FILE = Path("tarot_cache.json")  # локальный кэш предсказаний

def make_spread_key(cards):
    """
    cards: список карт как в cards_today.
    Возвращает строку-ключ, уникальную для конкретного расклада.
    Пример: 'the fool|upright || the moon|reversed || the empress|upright'
    """
    parts = []
    for c in cards:
        orientation = "reversed" if c["is_reversed"] else "upright"
        parts.append(f"{c['name'].lower().strip()}|{orientation}")
    # порядок важен: карта1||карта2||карта3
    return " || ".join(parts)

def load_cache():
    """
    Читает локальный json-файл с кэшем.
    Формат файла:
    {
        "the fool|upright || the moon|reversed || the empress|upright": "текст предсказания ...",
        "strength|upright || the sun|upright || death|reversed": "..."
    }
    """
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # если файл битый — начнём заново
                return {}
    return {}


def save_cache(cache_dict):
    """
    Перезаписывает json-файл свежим словарём.
    """
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, ensure_ascii=False, indent=2)
def get_reading_for_spread(cards, prompt_builder):
    """
    cards: список карт (cards_today)
    prompt_builder: функция, которая строит промпт (build_prompt)

    Возвращает готовый текст прогнозика.
    Использует и обновляет локальный кэш tarot_cache.json.
    """
    cache = load_cache()
    spread_key = make_spread_key(cards)

    # 1. Есть ли уже в кэше?
    if spread_key in cache:
        return cache[spread_key], True  # True -> это было из кэша

    # 2. Если нет — генерим с моделью
    prompt = prompt_builder(cards)

    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        # нет токена = offline fallback
        reading_text = (
            "Сегодняшняя энергия просит быть бережной к себе. "
            "Сделай что-то маленькое и приятное для тела, не из чувства вины, а из любви. 🫧"
        )
    else:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )

            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct:groq",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=220,
            )

            # ответ модели
            # у клиента openai-style message хранится как объект с полем 'content'
            reading_text = completion.choices[0].message.content

        except Exception as e:
            # Если облако упало / лимит / интернет
            reading_text = (
                "Сегодня важен покой. "
                "Ты не обязана всё контролировать. "
                "Твоё тело — не враг, а союзник, который устал и просит мягкости. 🌙"
            )

    # 3. Кладём в кэш и сохраняем на диск
    cache[spread_key] = reading_text
    save_cache(cache)

    return reading_text, False  # False -> только что сгенерили


########################
# HELPERS
########################

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
def build_prompt(cards):
    """
    cards = список словарей карт из твоего TAROT_CARDS (с полями name, upright, reversed, blurb, is_reversed)
    Мы объясняем модели контекст и просим её говорить как тёплая девочка-таролог.
    """
    lines = []
    lines.append("You are a soft, empathetic tarot reader. Write in Russian.")
    lines.append("Tone: supportive, feminine, intimate, not judgmental.")
    lines.append("Do NOT give medical, legal or financial advice.")
    lines.append("Write ~150 words total.")
    lines.append("Make it feel like a daily emotional check-in, not fortune-telling.")
    lines.append("Cards:")

    for idx, c in enumerate(cards, start=1):
        orientation = "reversed" if c["is_reversed"] else "upright"
        keywords = ", ".join(c[orientation])
        lines.append(
            f"{idx}. {c['name']} ({orientation}), ключевые идеи: {keywords}. "
            f"Описание карты: {c['blurb']}"
        )

    lines.append("Give one reading in Russian and English, in 2 short paragraphs.")
    prompt = "\n".join(lines)
    return prompt

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

prompt = build_prompt(cards_today)

import os
from openai import OpenAI

reading_text, from_cache = get_reading_for_spread(cards_today, build_prompt)

st.subheader("Сообщение дня 🌙")
st.write(reading_text)

if from_cache:
    st.caption("✨ (Это сообщение уже сохранено для такого расклада. Без вызова модели.)")
else:
    st.caption("✨ (Новое сообщение создано ИИ и добавлено в локальную библиотеку.)")



st.caption("Это развлекательный контент. Не медицинская, не финансовая и не юридическая рекомендация.")

st.markdown("---")
st.caption("Это не совет по здоровью, финансам или юриспруденции. Это мягкая подсказка-вдохновение ✨")
