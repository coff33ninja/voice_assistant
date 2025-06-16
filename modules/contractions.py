# contractions.py
# Utility for expanding contractions and normalizing pronunciations in text

import re

CONTRACTIONS = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he will / he shall",
    "he'll've": "he will have / he shall have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "i'd": "i had / i would",
    "i'd've": "i would have",
    "i'll": "i will / i shall",
    "i'll've": "i will have / i shall have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it will / it shall",
    "it'll've": "it will have / it shall have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she will / she shall",
    "she'll've": "she will have / she shall have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they will / they shall",
    "they'll've": "they will have / they shall have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "what're": "what are",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who will / who shall",
    "who'll've": "who will have / who shall have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you're": "you are",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you will / you shall",
    "you'll've": "you will have / you shall have",
    "you've": "you have"
}

CONTRACTION_RE = re.compile(r"\\b(" + "|".join(map(re.escape, CONTRACTIONS.keys())) + r")\\b", re.IGNORECASE)

COMMON_MISSPELLINGS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "lemme": "let me",
    "gimme": "give me",
    "kinda": "kind of",
    "sorta": "sort of",
    "hafta": "have to",
    "oughta": "ought to",
    "prolly": "probably",
    "probly": "probably",
    "dunno": "do not know",
    "whatcha": "what are you",
    "ya": "you",
    "imma": "i am going to",
    "finna": "fixing to / going to",
    "sup": "what is up",
    "cuz": "because",
    "cos": "because",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "musta": "must have",
    "lotta": "lot of",
    "outta": "out of",
    "sortof": "sort of",
    "kindof": "kind of"
    # Add more common misspellings or STT quirks
}

MISSPELLING_RE = re.compile(r"\\b(" + "|".join(map(re.escape, COMMON_MISSPELLINGS.keys())) + r")\\b", re.IGNORECASE)

def normalize_text(text: str) -> str:
    def replace_contraction(match):
        word = match.group(0)
        expanded = CONTRACTIONS.get(word.lower())
        if expanded:
            return expanded
        return word

    def replace_misspelling(match):
        word = match.group(0)
        corrected = COMMON_MISSPELLINGS.get(word.lower())
        return corrected if corrected else word

    text = CONTRACTION_RE.sub(replace_contraction, text)
    text = MISSPELLING_RE.sub(replace_misspelling, text)
    return text
