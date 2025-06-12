dictionary = {}

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

for target in targets:
    tmp = []
    with open(f"Dict/{target}.exc", "r") as f:
        lines = f.readlines()
        for line in lines:
            for word in line.split(" "):
                tmp.append(word.strip().lower())

        dictionary[target] = tmp


for target in targets:
    if target.find("art") != -1:
        continue
    init_len = len(dictionary[target])
    dictionary[target] = list(set(dictionary[target]))
    print(
        f"removed {init_len-len(dictionary[target])} duplicates from {target} dictionary"
    )
    for target2 in targets:
        if target2.find("art") == -1:
            continue
        for word in dictionary[target]:
            if word in dictionary[target2]:
                dictionary[target].remove(word)
                print(
                    f"{word} removed from {target} dictionary due to inclusion in {target2} dictionary"
                )


for target in targets:
    with open(f"Dict/{target}.exc", "w") as f:
        for word in dictionary[target]:
            f.write(f"{word}\n")
