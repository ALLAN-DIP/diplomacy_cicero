def and_items(items):
    if len(items) == 1:
        return str(items[0]) + " "
    elif len(items) == 2:
        return str(items[0]) + " and " + str(items[1]) + " "
    else:
        return ", ".join([str(item) for item in items[:-1]]) + ", and " + str(items[-1]) + " "

def or_items(items):
    if len(items) == 1:
        return str(items[0]) + " "
    elif len(items) == 2:
        return str(items[0]) + " or " + str(items[1]) + " "
    else:
        return ", ".join([str(item) for item in items[:-1]]) + ", or " + str(items[-1]) + " "

power_dict = {
    "AUSTRIA": "AUSTRIA",
    "ENGLANG": "ENGLAND",
    "FRANCE": "FRANCE",
    "GERMANY": "GERMANY",
    "ITALY": "ITALY",
    "RUSSIA": "RUSSIA",
    "TURKEY": "TURKEY",
    "AUS": "AUSTRIA",
    "ENG": "ENGLAND",
    "FRA": "FRANCE",
    "GER": "GERMANY",
    "ITA": "ITALY",
    "RUS": "RUSSIA",
    "TUR": "TURKEY",
}

power_list = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

unit_dict = {
    "FLT": "fleet",
    "AMY": "army",
}    