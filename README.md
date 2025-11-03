---
### Disclaimer / Personal Note

This project is **a personal, non-commercial side experiment** created purely for fun and creative exploration.  
It is **not related to my professional work** in any way.

The app uses simple logic and a language model to generate *entertainment-style tarot readings*.  
It does **not provide real advice**, medical or otherwise.  
All tarot card images were **AI-generated** (ChatGPT / DALL·E) and are free for use under OpenAI’s content policy.

I built this project as a small creative break — a playful way to combine coding and design with a "girly" aesthetic 🌸✨

---

# Tarot of the Day 🔮

MVP is located here: https://taro-random-generator.streamlit.app/

Streamlit-app generates 3 tarot set based on the date and 150 words LLM-generated tarot readings based on tarot cards' names, orientation, description and key meanings. 

To generate tarot readings I used __meta-llama/Llama-3.1-8B-Instruct__ model on HuggingFace. Firstly code checks if the reading for the selected cards set existis in my json cash database, otherwise it makes inference from HuggingFace and then saves this reading in json cash database to future use. 

Minecraft tarot cards generated with ChatGPT-5, as well as a wallpaper.

## Run localy

1. Build dependencies:
   ```bash
   pip install -r requirements.txt

    (Необязательно) Получи Hugging Face API токен и экспортируй:
    export HF_TOKEN="hf_xxx..."
    ```

2. Run:

    ``` bash
    streamlit run app.py
    ```

Откроется браузер с приложением.
