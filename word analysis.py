import PyPDF2
from time import time
from collections import Counter

targets = [
    "conj",
    "art",
    "adj",
    "adv",
    "prep",
    "noun",
    "verb",
    "dpron",
    "indpron",
    "intpron",
    "opron",
    "ppron",
    "refpron",
    "relpron",
    "spron",
]

start_time = time()
print("Fingerprinting Edgar Allan Poe...")


def init_dictionary():
    dictionary = {}
    for target in targets:
        with open(f"Dict/{target}.exc", "r") as f:
            dictionary[target] = [word.strip() for line in f for word in line.split()]
    return dictionary


def get_assignment(word, dictionary):
    for _ in range(3):
        for target, words in dictionary.items():
            if word in words:
                return target
        if word.endswith("es"):
            word = word[:-2]
        elif word.endswith("s"):
            word = word[:-1]
    return None


def pdf_to_text(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages[10:907])
        print(f"Analyzing {len(pdf_reader.pages[10:907])} pages of their work...")
        return text


words_total = Counter(
    word.lower()
    for line in pdf_to_text(
        r"edgar_allan_poe__complete_tales_-_edgar_allan_poe.pdf"
    ).split("\n")
    for word in line.split()
    if word.isalpha()
)

sentence = [
    "indpron",  # All
    "shprep",  # of
    "ldpron",  # these
    "conj",  # and
    "madv",  # there
    "verb",  # was
    "ladv",  # never
    "madj",  # more
]


result = []
hits = []
dictionary = init_dictionary()

prefix_conditions = {
    "sh": lambda k: len(k) <= 2,
    "mm": lambda k: len(k) == 3,
    "ml": lambda k: len(k) == 4,
    "m": lambda k: len(k) >= 4,
    "ll": lambda k: len(k) >= 7,
    "l": lambda k: len(k) >= 5,
}

for tar in sentence:
    for k, v in words_total.most_common():
        if k in hits:
            continue
        prefix = next((p for p in prefix_conditions if tar.startswith(p)), None)
        if prefix and not prefix_conditions[prefix](k):
            continue
        if not prefix and len(k) <= 2:
            continue
        if get_assignment(k, dictionary) == tar.replace("sh", "").replace(
            "l", ""
        ).replace("m", ""):
            result.append((tar, k, v))
            hits.append(k)
            break

biggest = max(len(line[1]) for line in result)
result[0] = (result[0][0], result[0][1].capitalize(), result[0][2])

print("\nFingerprint: ", end="")
print(" ".join(line[1] for line in result) + ".")

print("\n                       Words: ", end="")
print(" | ".join(f"{line[1]:<{biggest}}" for line in result))

print("Number of times word appears: ", end="")
print(" | ".join(f"{line[2]:<{biggest}}" for line in result))

print(f"\nCompleted fingerprint in {round(time()-start_time,2)} seconds")
