# ДЗ1. Collaborative filtering / нейроранжирование

Соревнование: https://cups.online/ru/contests/education_2889  
Метрика: NDCG. Нужно побить LightFM-бейзлайн.

## Что положить

Скачайте с All Cups и положите сюда:

- `data/train.csv` — `user,track,time`
- `data/test.csv` — `user,track`

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_and_submit.py
```

На выходе:

- `data/submit.csv` — файл для сабмита (`user,track,score`)
- `data/val_metrics.json` — локальный hold-out NDCG
- текст для платформы — `description.md` (отправлять один раз, когда скор устраивает)

Проверка пайплайна без данных соревнования:

```bash
python train_and_submit.py --synthetic
```

Сабмит на cups.online я не отправляю — это нужно сделать вручную после проверки.
