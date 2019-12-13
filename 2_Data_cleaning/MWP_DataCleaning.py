import pandas as pd
import numpy as np
import re
import json
import nltk
from word2number import w2n
from nltk.stem.snowball import SnowballStemmer

df = pd.read_csv('./Cleaned Data/trainData_univariable.csv')

df = df[np.invert(np.array(df['text'].isna()))]
numMap = {"twice": 2, "double": 2, "thrice": 3, "half": "1/2", "tenth": "1/10", "quarter": "1/4", "fifth": "1/5"}
fraction = {"third": "/3", "half": "/2", "fourth": "/4", "sixth": "/6", "fifth": "/5", "seventh": "/7", "eighth": "/8",
            "ninth": "/9", "tenth": "/10", "eleventh": "/11", "twelfth": "/12", "thirteenth": "/13",
            "fourteenth": "/14", "fifteenth": "/15", "sixteenth": "/16", "seventeenth": "/17", "eighteenth": "/18",
            "nineteenth": "/19", "twentieth": "/20", "%": "/100"}

stemmer = SnowballStemmer(language='english')


# Convert fractions to floating point numbers
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


# Convert infix expression to suffix expression
def postfix_equation(equ_list):
    stack = []
    post_equ = []
    op_list = ['+', '-', '*', '/', '^']
    priori = {'^': 3, '*': 2, '/': 2, '+': 1, '-': 1}
    for elem in equ_list:
        if elem == '(':
            stack.append('(')
        elif elem == ')':
            while 1:
                op = stack.pop()
                if op == '(':
                    break
                else:
                    post_equ.append(op)
        elif elem in op_list:
            while 1:
                if not stack:
                    break
                elif stack[-1] == '(':
                    break
                elif priori[elem] > priori[stack[-1]]:
                    break
                else:
                    op = stack.pop()
                    post_equ.append(op)
            stack.append(elem)
        else:
            post_equ.append(elem)
    while stack:
        post_equ.append(stack.pop())
    return post_equ


# Identification and filtering of univariable equations
char_set = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '+', '-', '/', '*'}
df_cleaned = pd.DataFrame()
for id_, row in df.iterrows():
    l, r = row['equations'].split("=", 1)
    lSet, rSet = set(l.replace(" ", "")), set(r.replace(" ", ""))
    flagl = (len(l.strip()) == 1 and not l.strip().isdigit() and len(rSet - char_set) == 0)
    flagr = (len(r.strip()) == 1 and not r.strip().isdigit() and len(lSet - char_set) == 0)
    if flagl or flagr:
        if flagr:
            row['equations'] = r + '=' + l

        df_cleaned = df_cleaned.append(row)

k = 0

numLists = {}
numLists_idx = {}
eqLists_idx = {}
eqLists = {}
texts = {}
equations_List = {}
final_ans = {}
numListMAP = {}
final_replaced_text = {}
final_replaced_eq = {}
final_replaced_eq_post = {}
final_number_list = {}
final_num_postn_list = {}
numtemp_order = {}

for id_, row in df_cleaned.iterrows():
    # Converting fractions and ordinals to values through appropriate sub-routines and string builder ops
    if not bool(re.search(r'[\d]+', row['text'])):
        continue
    sb = ""
    numSb = ""
    val = 0
    prevToken = ""
    for tokens in nltk.word_tokenize((row['text'])):
        try:
            val += w2n.word_to_num(tokens)
        except ValueError:
            if val > 0:
                sb = sb + " " + str(val)
                if tokens in fraction:
                    sb = sb + fraction[tokens]
                elif stemmer.stem(tokens) in fraction:
                    sb = sb + fraction[stemmer.stem(tokens)]
                else:
                    sb = sb + " " + tokens
                val = 0
            else:
                if tokens in numMap.keys():
                    sb = sb + " " + str(numMap[tokens])
                else:
                    sb = sb + " " + tokens
        prevToken = tokens

    re.sub(' +', ' ', sb)
    sb = sb.strip()

    # Re-structure equation anomalies for normalization
    row['equations'] = row['equations'].replace(' ', '')
    eqs = row['equations'][0]

    for i in range(1, len(row['equations']) - 1):
        if row['equations'][i] == '(':
            if row['equations'][i - 1] == ')' or row['equations'][i - 1].isdigit():
                eqs += '*' + row['equations'][i]
            else:
                eqs += row['equations'][i]
        elif row['equations'][i + 1].isdigit() and row['equations'][i] == ')':
            eqs += row['equations'][i] + "*"

        elif row['equations'][i] == '-' and (row['equations'][i - 1].isdigit() or row['equations'][i - 1] == ')') and \
                row['equations'][i + 1].isdigit():
            eqs += "+" + row['equations'][i]
        else:
            eqs += row['equations'][i]
    eqs += row['equations'][-1]

    # Extract different number types from the text in order
    numList = re.findall(r'-?[\d]* *[\d]+ *\/ *?[\d]*|-?[\d]+\.?[\d]*', sb)
    if len(numList) == 0:
        continue
    # Extract different number types from the equation in order
    eqNumList = re.findall(r'-?[\d]* *[\d]+ *\/ *?[\d]*|-?[\d]+\.?[\d]*', eqs)
    # Get the positions of the numbers in the text
    id_pattern = [m.span() for m in re.finditer(r'-?[\d]* *[\d]+ *\/ *?[\d]*|-?[\d]+\.?[\d]*', sb)]
    # Get the positions of the numbers in the equation
    eqid_pattern = [m.span() for m in re.finditer(r'-?[\d]* *[\d]+ *\/ *?[\d]*|-?[\d]+\.?[\d]*', eqs)]
    if len(set(numList)) < len(set(eqNumList)) or len(set(eqNumList) - set(numList)) != 0:
        continue

    numLists[id_] = numList
    numLists_idx[id_] = id_pattern
    eqLists_idx[id_] = eqid_pattern
    texts[id_] = sb
    eqLists[id_] = eqNumList
    equations_List[id_] = eqs
    final_ans[id_] = row['ans']

# Adding space between numbers for ease of identification
for id, text in texts.items():
    modified_text = ''
    prev = 0
    for (start, end) in numLists_idx[id]:
        modified_text += text[prev:start] + ' ' + text[start:end] + ' '
        prev = end
    modified_text += text[end:]

    # Re-evaluate positions of variables in the modified text
    numList = re.findall(r'-?[\d]* *[\d]+ *\/ *?[\d]*|-?[\d]+\.?[\d]*', modified_text)
    if len(numList) == 0:
        continue
    id_pattern = [m.span() for m in re.finditer(r'-?[\d]* *[\d]+ *\/ *?[\d]*|-?[\d]+\.?[\d]*', modified_text)]

    # assert (len(numList) == len(numLists[id]))
    # for i in range(len(numList)):
    #     assert (numList[i].strip() == numLists[id][i].strip())

    numLists[id] = numList
    numLists_idx[id] = id_pattern
    texts[id] = modified_text

# Replace questions text with template variables
for id, vals in numLists.items():
    numListMap = {}
    num_list = []
    num_postion = []
    k = 0
    for i in range(len(vals)):
        tmp = convert_to_float(vals[i]) if vals[i].find("/") != -1 else vals[i]
        if tmp not in numListMap.keys():
            num_list.append(str(tmp))
            num_postion.append(numLists_idx[id][i][0])
            numListMap[tmp] = 'temp_' + str(k)
            k += 1
    final_number_list[id] = num_list
    final_num_postn_list[id] = num_postion

    numListMAP[id] = numListMap
    replaced_text = ''
    prev_index = 0
    final_num_list = []
    num_temps = []
    for i in range(len(vals)):
        tmp = convert_to_float(vals[i]) if vals[i].find("/") != -1 else vals[i]
        final_num_list.append(tmp)
        start = numLists_idx[id][i][0]
        end = numLists_idx[id][i][1]
        unchanged = texts[id][prev_index:start]
        changed = texts[id][start:end]
        replaced_text = replaced_text + unchanged + numListMap[tmp]
        num_temps.append(numListMap[tmp])
        prev_index = end
    replaced_text += texts[id][numLists_idx[id][i][-1]:]
    final_replaced_text[id] = replaced_text
    numtemp_order[id] = num_temps

# Replace equations with template variables
for id, vals in eqLists.items():
    equation = equations_List[id]
    replaced_eq = ''
    prev_index = 0
    for i in range(len(vals)):
        tmp = convert_to_float(vals[i]) if vals[i].find("/") != -1 else vals[i]
        start = eqLists_idx[id][i][0]
        end = eqLists_idx[id][i][1]
        unchanged = equation[prev_index:start]
        changed = equation[start:end]
        replaced_eq = replaced_eq + " " + unchanged + " " + numListMAP[id][tmp]
        prev_index = end

    replaced_eq += " " + equation[eqLists_idx[id][i][-1]:]

    list_eqs = replaced_eq.split(" ")
    equation_normalized = []

    for i in range(len(list_eqs)):
        if list_eqs[i] == '':
            continue
        elif list_eqs[i].find('temp') == -1 and len(list_eqs[i]) > 1:
            equation_normalized.extend(list(list_eqs[i]))
        else:
            equation_normalized.append(list_eqs[i])

    final_replaced_eq[id] = equation_normalized

    eq_norm_postfix = postfix_equation(equation_normalized)
    final_replaced_eq_post[id] = eq_norm_postfix

# Calculate word position for numbers in text
for id, text in texts.items():
    words = [i.replace(' ', '') for i in text.split()]
    numpos = []

    min_ind = {i.replace(' ', ''): float('inf') for i in final_number_list[id]}

    for i in range(len(words)):
        if words[i] in [j.replace(' ', '') for j in numLists[id]]:
            tmp = convert_to_float(words[i]) if words[i].find("/") != -1 else words[i]
            min_ind[str(tmp)] = min(min_ind[str(tmp)], i)

    for val in final_number_list[id]:
        numpos.append(min_ind[str(val).replace(' ', '')])

    final_num_postn_list[id] = numpos

# Create json dump from the updated dictionaries
data = {}
for id in final_replaced_text.keys():
    data_template = {}
    data_template["template_text"] = final_replaced_text[id]
    data_template["expression"] = equations_List[id]
    data_template["mid_template"] = final_replaced_eq[id]
    data_template["num_list"] = final_number_list[id]
    data_template["index"] = str(id)
    data_template["numtemp_order"] = numtemp_order[id]
    data_template["post_template"] = final_replaced_eq_post[id][2:]
    if final_ans[id].find("|") != -1:
        data_template["ans"] = final_ans[id].split("|")[1].strip()
    elif final_ans[id].find("/") != -1:
        data_template["ans"] = str(convert_to_float(final_ans[id]))
    else:
        data_template["ans"] = final_ans[id]
    data_template["text"] = texts[id]
    data_template["num_position"] = final_num_postn_list[id]
    data[id] = data_template

with open('final_dolphin_data.json', 'w') as f:
    json.dump(data, f)
