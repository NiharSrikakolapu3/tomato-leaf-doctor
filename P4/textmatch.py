"""Small word-level tools shared by the source check and the gate's care-overlap test."""
import re

from nltk.stem import PorterStemmer

STOPWORDS = set(
    """a an the and or but if then than so as of to in on at by for from with into onto
    over under up down out off is are was were be been being it its this that these those
    there their them they you your yours we our i me my he she his her him which who whom
    what when where while how why either other such own same too very can will just
    should would could may might must do does did
    done have has had having also about above below after before again further once here
    s t one's e.g i.e etc""".split()
)

# Words that change meaning when flipped. "avoid" counts, since "avoid X" negates X.
NEGATIONS = {"not", "no", "never", "nor", "cannot", "can't", "don't", "doesn't", "shouldn't",
             "won't", "isn't", "aren't", "avoid", "without", "none"}

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "half": "1/2", "single": "1", "once": "1", "twice": "2", "double": "2",
}
FRACTIONS = {"¼": " 1/4 ", "½": " 1/2 ", "¾": " 3/4 ", "⅛": " 1/8 ", "⅓": " 1/3 "}
IRREGULAR = {"leaves": "leaf", "eaten": "eat", "ate": "eat", "left": "leav", "dead": "kill",
             "died": "kill", "dying": "kill", "grown": "grow", "grew": "grow"}
# Words a faithful rewrite swaps freely, mapped to one shared stem (after Porter stemming).
SAME_MEANING = {
    "infect": "sick", "affect": "sick", "diseas": "sick", "symptomat": "sick",
    "sunlight": "sun", "sunni": "sun", "die": "kill", "garbag": "trash",
    "whitish": "white", "yellowish": "yellow", "brownish": "brown",
}
PORTER = PorterStemmer()
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[./]\d+)?%?")


def tokens(text):
    """Lowercase word and number tokens, in order."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    for symbol, spelled in FRACTIONS.items():
        text = text.replace(symbol, spelled)
    return TOKEN.findall(text)


def stem(word):
    if word in NUMBER_WORDS:
        return NUMBER_WORDS[word]
    if word[0].isdigit():
        return word
    base = IRREGULAR.get(word) or PORTER.stem(word)
    if base.endswith("li") and len(base) > 5:
        base = base[:-2]  # Porter leaves "freshly" as "freshli"
    return SAME_MEANING.get(base, base)


def content_stems(text):
    """Stems of the words that carry meaning (no stopwords, no negations)."""
    kept = []
    for word in tokens(text):
        if word in STOPWORDS or word in NEGATIONS:
            continue
        kept.append(stem(word))
    return kept


def numbers(text):
    found = set()
    for word in tokens(text):
        if word[0].isdigit():
            found.add(word.rstrip("%"))
        elif word in NUMBER_WORDS:
            found.add(NUMBER_WORDS[word])
    return found


def negation_count(words):
    return sum(w in NEGATIONS or w.endswith("n't") for w in words)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))
