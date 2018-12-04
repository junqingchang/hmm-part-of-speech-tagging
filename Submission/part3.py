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
filep3out = "{}/dev.p3.out".format(running)

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

# Part 3-------------------------------------------------
def get_transition_probability(y1, y2, t_dict, t_count):
    try:
        count_y1_y2 = t_dict[y1][y2]
        count_y1 = t_count[y1] # to be replaced by y1
        probability = count_y1_y2 /count_y1
        return float(probability)
    except:
        return 0.0

# Viterbi
def viterbi(sentence, e_dict, t_dict, t_count):
    tag_track = []
    for tag in t_dict:
        tag_track.append(tag)
    tag_track.remove("START")
    prob_table = {}
    for i in range(len(sentence)+1):
        # row column downwards, key is {i : probabilties}
        row = {}
        for j in range(len(tag_track)):
            #First word in the sentence
            if i == 0:
                ij_transition = get_transition_probability("START",tag_track[j], t_dict,t_count)
                ij_emission = get_kemission_probability(sentence[i],tag_track[j], e_dict, t_count)
                ij_value = ij_transition*ij_emission
                ##Fix for finding probability 
                if ij_value != 0:
                    ij_value = -1*math.log(ij_value)
                row[j] = (ij_value, "START")
            #End Of the Sentence
            elif i == len(sentence):
                ij_prev = prob_table[i-1][j][0] # take the 0th element in the tuple
                if ij_prev == 0:
                    row[j] = (0,j)
                    continue
                ij_transition = get_transition_probability(tag_track[j], "STOP", t_dict, t_count)
                ##Fix for finding probability
                if ij_transition != 0:
                    ij_value = -1*math.log(ij_transition)+ij_prev
                else:
                    ij_value=0
                row[j] = (ij_value,j)
            else:
                largest_value = sys.maxsize
                largest_index = 0
                for k in range(len(tag_track)):
                    kj_prev = prob_table[i-1][k][0]
                    if kj_prev == 0:
                        continue
                    kj_transition = get_transition_probability(tag_track[k],tag_track[j],t_dict,t_count)
                    kj_emission = get_kemission_probability(sentence[i],tag_track[j],e_dict, t_count)
                    kj_value = kj_transition * kj_emission
                    if kj_value != 0:
                        kj_value = -1*math.log(kj_value)+kj_prev
                    if kj_value < largest_value and kj_value != 0:
                        largest_value = kj_value
                        largest_index = k
                row[j] = (largest_value,largest_index)
        prob_table[i] = row
    sequence = []
    highest_prob = sys.maxsize
    previous_tag = 0
    for i in range(len(tag_track)):
        compare_prob = prob_table[len(sentence)][i][0]
        if compare_prob < highest_prob and compare_prob != 0:
            highest_prob = compare_prob
            previous_tag = prob_table[len(sentence)][i][1]
    for i in range(len(prob_table)-1):
        sequence.append(tag_track[previous_tag])
        previous_tag = prob_table[len(prob_table)-i-2][previous_tag][1]
    sequence.reverse()
    
    return sequence

def viterbi_on_input(inputfile, outputfile, e_dict, t_dict, t_count):
    f = open(inputfile, encoding="UTF-8")
    w = open(outputfile, 'w' ,encoding="UTF-8")
    sentence = []
    for line in f:
        word = line.strip()
        if word == "":
            tags = viterbi(sentence, e_dict, t_dict, t_count)
            for i in range(len(sentence)):
                w.write('{} {}\n'.format(sentence[i], tags[i]))
            w.write("\n")
            sentence = []
            continue
        counter = 0
        for tag in e_dict:
            if word in e_dict[tag]:
                counter += 1
        if counter == 0:
            word = '#UNK#'
        sentence.append(word)

t_dict, t_count, e_dict = probability_creation(filetrain)
viterbi_on_input(filetest, filep3out, e_dict, t_dict, t_count)




