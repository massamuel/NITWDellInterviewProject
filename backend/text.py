import re

LABELS = sorted(["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
                 "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"])
TYPE_PATTERN = re.compile(r"\b(?:" + "|".join(LABELS) + r")\b", re.I)

def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text)
    text = TYPE_PATTERN.sub(" ", text)  # Remove direct label mentions.
    return " ".join(re.sub(r"[^a-zA-Z\s]", " ", text).lower().split())
