import pandas as pd
import re
import math
import pickle

# ── 1. Chargement du modèle ────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

positive_words_dict = model["positive_words_dict"]
negative_words_dict = model["negative_words_dict"]
V         = model["vocab_size"]
total_pos = model["total_pos"]
total_neg = model["total_neg"]
p_positive = model["p_positive"]
p_negative = model["p_negative"]

# ── 2. Classificateur ──────────────────────────────────────────

import numpy as np

def smooth_proba(p: float, alpha: float = 700.0) -> float:
    """
    Transforme une probabilité issue d'un Naive Bayes.
    
    - alpha > 1 : écrase le centre (≈0.5), amplifie les extrêmes
    - alpha = 1 : identité
    - alpha < 1 : effet inverse
    """
    sign = 1 if p >= 0.5 else -1
    return 0.5 + sign * 0.5 * abs(2 * p - 1) ** alpha


def classify(review):
    words = re.findall(r'\b\w+\b', review.lower())

    log_score_pos = math.log(p_positive)
    log_score_neg = math.log(p_negative)

    for word in words:
        p_word_pos = (positive_words_dict.get(word, 0) + 1) / (total_pos + V)
        p_word_neg = (negative_words_dict.get(word, 0) + 1) / (total_neg + V)

        log_score_pos += math.log(smooth_proba(p_word_pos))
        log_score_neg += math.log(smooth_proba(p_word_neg))

    return "positive" if log_score_pos > log_score_neg else "negative"

# ── 3. Évaluation sur le jeu de test ──────────────────────────
df = pd.read_csv("IMDBDataset.csv")
split = int(0.90 * len(df))
test_df = df[split:]

correct = sum(
    classify(row['review']) == row['sentiment']
    for _, row in test_df.iterrows()
)
print(f"Précision sur {len(test_df)} exemples de test : {correct / len(test_df):.2%}\n")

# ── 4. Tests manuels ───────────────────────────────────────────
tests = [
    "This movie was absolutely fantastic, I loved every minute!",
    "Terrible film, waste of time, awful acting.",
    """This film is far from perfect. The pacing drags in the second act, and a couple of 
    supporting characters feel underdeveloped. However, the lead performance is genuinely 
    captivating and I walked out feeling it was absolutely worth watching."""
]

for text in tests:
    print(f"'{text[:60].strip()}...'\n  → {classify(text)}\n")

