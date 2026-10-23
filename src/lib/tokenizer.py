from nltk.stem import PorterStemmer
import re

def term_toknizer(term: str) -> list:
    pattern = r"[A-Z]+|[a-z]+|/d+"
    matched = re.findall(pattern, term)
    stemmer = PorterStemmer()
    term_toknizer = []

    with open("src/lib/stop_word.txt", "r") as f:
            stop_word = f.read()
            stop_word = stop_word.split()

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

