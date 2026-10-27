from nltk.stem import PorterStemmer
import re


with open("src/lib/stop_word.txt", "r") as f:
    stop_word_raw = f.read()
    stop_word = stop_word_raw.split()


def term_toknizer(term: str) -> list:
    # pattern = r"[A-Z]+|[A-Z](?:[a-z]+)|[a-z]+\d*|\d+"
    # pattern = r"[A-Z]+(?![a-z])|[A-Z][a-z]+(?:-\d+)?|[a-z]+|\d+"
    pattern = r"(?u)\b\w\w+\b"

    matched = re.findall(pattern, term)
    stemmer = PorterStemmer()
    term_toknizer = []

    for m in matched:
        m = stemmer.stem(m)
        if m not in stop_word:
            term_toknizer.append(m)
    return term_toknizer


def tokenizer(string: str) -> list:
    list_string = string.split()
    tokens = []

    for word in list_string:
        tokens += term_toknizer(word)
    return tokens
