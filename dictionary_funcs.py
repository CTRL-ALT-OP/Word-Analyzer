targets = [
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


def init_dictionary():
    dictionary = {}

    for target in targets:
        tmp = []
        with open(f"Dict/{target}.exc", "r") as f:
            lines = f.readlines()
            for line in lines:
                for word in line.split(" "):
                    tmp.append(word.strip())

        dictionary[target] = tmp

    return dictionary


def get_assignment(word, dictionary):
    for target in dictionary.keys():
        if word in dictionary[target]:
            return target


dictionary = init_dictionary()
print(get_assignment("another", dictionary))
