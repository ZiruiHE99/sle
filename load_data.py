import json

parenthesis = ["-LRB-", "-LSB-", "-RRB-", "-RSB-"]

def containsNonAscii(s):
    return any(ord(i) > 127 for i in s)
def prep_sent(sentence):
    temp = ' '.join([x for x in sentence.split() if x and x not in parenthesis])
    words = temp.split()
    cleaned_words = [word for word in words if not containsNonAscii(word)]
    cleaned_sentence = ' '.join(cleaned_words)
    return cleaned_sentence


def prep_id(fname):
    s1 = []
    s2 = []
    id_label = []
    data = [json.loads(line) for line in open(fname, 'r')]
    for i in data:
        s1.append(prep_sent(i['evidence']))
        s2.append(prep_sent(i['claim']))
        id_label.append(i['gold_label'])
    label_to_num1 = {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}
    id_label = list(map(lambda label: label_to_num1[label], id_label))
    return s1,s2,id_label

def prep_sym1(fname):
    s1 = []
    s2 = []
    sym1_label = []
    with open(fname, 'r') as file:
        for line in file:
            obj = json.loads(line)
            s1.append(prep_sent(obj['evidence_sentence']))
            s2.append(prep_sent(obj['claim']))
            sym1_label.append(obj['label'])
    label_to_num2 = {"SUPPORTS": 0, "NEI": 1, "REFUTES": 2}
    sym1_label = list(map(lambda label: label_to_num2[label], sym1_label))
    return s1,s2,sym1_label

def prep_sym2(fname):
    s1 = []
    s2 = []
    sym2_label = []
    with open(fname, 'r') as file:
        for line in file:
            obj = json.loads(line)
            s1.append(prep_sent(obj['sentence1']))
            s2.append(prep_sent(obj['sentence2']))
            sym2_label.append(obj['label'])
    label_to_num2 = {"SUPPORTS": 0, "NEI": 1, "REFUTES": 2}
    sym2_label = list(map(lambda label: label_to_num2[label], sym2_label))
    return s1,s2,sym2_label

