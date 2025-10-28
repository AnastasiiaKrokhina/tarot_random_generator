# cards_meta.py

import random

# Список всех доступных карт. Каждая карта = dict.
# Поля:
# - id: внутренний id для файла
# - name: имя карты
# - image: путь к картинке
# - upright / reversed: краткие ключевые смыслы
# - blurb: короткое описание в человеческом стиле

TAROT_CARDS = [
    {
        "id": "0_the_fool",
        "name": "The Fool",
        "image": "cards/0_the_fool.png",
        "upright": ["новый старт", "спонтанность", "доверие миру"],
        "reversed": ["импульсивность", "наивность", "риск без плана"],
        "blurb": "Энергия начала. Шаг вперёд даже если страшно.",
    },
    {
        "id": "1_the_magician",
        "name": "The Magician",
        "image": "cards/1_the_magician.png",
        "upright": ["сила воли", "фокус", "проявление идеи в реальность"],
        "reversed": ["манипуляция", "сомнение в себе", "распыление энергии"],
        "blurb": "Ты можешь превратить мысль в действие. Ресурсы уже есть.",
    },
    {
        "id": "2_the_high_priestess",
        "name": "The High Priestess",
        "image": "cards/2_the_high_priestess.png",
        "upright": ["интуиция", "тишина", "секреты"],
        "reversed": ["подавленная интуиция", "самообман", "нерешительность"],
        "blurb": "Информация уже внутри тебя. Просто стань тише.",
    },
    {
        "id": "3_the_empress",
        "name": "The Empress",
        "image": "cards/3_the_empress.png",
        "upright": ["забота", "изобилие", "тело и удовольствие"],
        "reversed": ["истощение", "самокритика", "передержка других"],
        "blurb": "Красота и тело требуют внимания, не спешки.",
    },
    {
        "id": "10_wheel_of_fortune",
        "name": "Wheel of Fortune",
        "image": "cards/10_wheel_of_fortune.png",
        "upright": ["цикл", "удача", "неожиданное в плюс"],
        "reversed": ["застревание", "чувство что тебя несёт", "хаос"],
        "blurb": "Колесо крутится. Это момент сдвига, не статики.",
    },
    {
        "id": "21_the_world",
        "name": "The World",
        "image": "cards/21_the_world.png",
        "upright": ["завершение", "уровень ап", "гармония"],
        "reversed": ["почти-конец но не до конца", "незакрытый гештальт"],
        "blurb": "Цикл подходит к концу. Тебя ждёт следующая версия тебя.",
    },
    # ...добавь остальные карты в этом же стиле...
]

def pick_daily_cards(seed_str: str, n_cards: int = 3):
    """
    Детерминированно выбираем n_cards карт для заданного seed (обычно дата).
    Каждая карта может быть upright или reversed.
    Возвращаем список dict с добавленным полем "is_reversed".
    """
    rng = random.Random(seed_str)

    cards = rng.sample(TAROT_CARDS, n_cards)
    result = []
    for c in cards:
        reversed_flag = rng.random() < 0.5
        result.append({
            **c,
            "is_reversed": reversed_flag,
        })
    return result
