import re
import sys
import math

parameters = len(sys.argv)
if len(sys.argv) != 2:
    print("Invalid input, please enter for example 'python MLProject.py EN'")
else:
    running = sys.argv[1]

# running = "EN"

filetrain = "{}/train".format(running)
filetest = "{}/dev.in".format(running)
filep2out = "{}/dev.p2.out".format(running)

# Train Data Cleaning -----------------------------
def probability_creation(inputfile):
    f = open(inputfile,encoding="UTF-8")
    e_dict = {}
    t_dict = {}
    previous_tag = "START"
    for line in f:
        if len(line) == 0: continue
        cols = re.split('\s+(?=\S+$)',line) #Using the last whitespace as separator
        if len(cols) > 1:
            tag = cols[1].strip()
            word = cols[0].strip()
            if tag not in e_dict:
                e_dict[tag] = {}
            if word not in e_dict[tag]:
                e_dict[tag][word] = 1
            else:
                e_dict[tag][word] += 1
                    
            if previous_tag not in t_dict:
                t_dict[previous_tag] = {}
            if tag not in t_dict[previous_tag]:
                t_dict[previous_tag][tag] = 1
            else:
                t_dict[previous_tag][tag] += 1
            previous_tag = tag
        else:
            tag = "STOP"
            if previous_tag not in t_dict:
                t_dict[previous_tag] = {}
            if tag not in t_dict[previous_tag]:
                t_dict[previous_tag][tag] = 1
            else:
                t_dict[previous_tag][tag] += 1
            previous_tag = "START"
            
    t_count = {}
    for key in t_dict:
        count = 0
        for value in t_dict[key]:
            count+= t_dict[key][value]
        t_count[key] = count
    return t_dict, t_count, e_dict

# Part 2---------------------------------------------
# Without Unknown
def get_emission_probability(x, y, e_dict, t_count):
    try:
        total_y_words = t_count[y]
        total_tag_to_word = e_dict[y][x]
        return total_tag_to_word/total_y_words
    except:
        return 0.0

# With Unknown
def get_kemission_probability(x, y, e_dict, t_count):
    global_counter=0
    for key in e_dict:
        if x in e_dict[key]:
            global_counter+=1
    try:
        total_y_words = t_count[y]
        if global_counter == 0:
            calculatedprob = float(1 / (total_y_words + 1))
            return calculatedprob
        else:
            if x in e_dict[y]:
                total_tag_to_word = e_dict[y][x]
            else:
                total_tag_to_word = 0
            calculatedprob = float(total_tag_to_word / (total_y_words + 1))
            return calculatedprob
    except:
        calculatedprob = float(1 / (total_y_words + 1))
        return 0.0

def emission_tag_creation(x, e_dict, t_count):
    highest_prob = 0
    most_prob = ""
    for tag in e_dict:
        prob = get_kemission_probability(x, tag, e_dict, t_count)
        if prob > highest_prob:
            highest_prob = prob
            most_prob = tag
    return most_prob

def emission_output(inputfile, outputfile, e_dict, t_count):
    f = open(inputfile,encoding="UTF-8")
    w = open(outputfile, 'w' ,encoding="UTF-8")
    for line in f:
        word = line.strip()
        if word == "":
            w.write("\n")
            continue
        counter = 0
        for tag in e_dict:
            if word in e_dict[tag]:
                counter += 1
        if counter == 0:
            word = '#UNK#'
        tag = emission_tag_creation(word, e_dict, t_count)
        w.write("{} {}\n".format(word, tag))

t_dict, t_count, e_dict = probability_creation(filetrain)
emission_output(filetest, filep2out, e_dict, t_count)