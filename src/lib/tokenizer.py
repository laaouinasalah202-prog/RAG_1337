from nltk.stem import PorterStemmer
import re


with open("src/lib/stop_word.txt", "r") as f:
    stop_word_raw = f.read()
    stop_word = stop_word_raw.split()


def term_toknizer(term: str) -> list:
    """Split a string into stemmed tokens without stop words.

    Extract words of two or more characters with a regular expression,
    reduce each one with the Porter stemmer, and drop those found in
    the stop word list.

    Args:
        term (str): The text to tokenize.

    Returns:
        list[str]: The stemmed tokens that are not stop words.
    """

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
    """Tokenize a string word by word.

    Split the string on whitespace and apply ``term_toknizer`` to each
    word.

    Args:
        string (str): The text to tokenize.

    Returns:
        list[str]: The stemmed tokens of all the words, in order,
            without stop words.
    """

    list_string = string.split()
    tokens = []

    for word in list_string:
        tokens += term_toknizer(word)
    return tokens
