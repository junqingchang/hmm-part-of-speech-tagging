import pandas as pd
import re
import numpy as np

####### Data preprocessing ####### 
filename = "SG/train"
fin = open(filename)

rawframe=[]
all_tags = [] # List holding All unique Tags
all_words = {} # dictionary of all words with tag as key
k_of_dict={}

#Part 2a
##Algorithm
#for loop tweets,tags:
#add each tag into a dictionary. (key:tags,values:[words])
for line in fin:
    if len(line) == 0: continue
    cols = re.split('\s+(?=\S+$)',line) #Using the last whitespace as separator
    if len(cols) > 1:
        tag = cols[1].strip()
        word = cols[0].strip()
        if tag not in all_tags:
            all_tags.append(tag)
        if tag not in all_words:
            all_words[tag] = [word]
            k_of_dict[tag] = 1
        else:
            all_words[tag].append(word)
    rawframe.append(cols)

df = pd.DataFrame(rawframe, columns = ["Word", "Tag"])
df["transit"] = None # Create an extra column for transition start/end for states
# Printing out the dataframe
# print(df)
# print(df.describe())


# For Emission 
def get_emission_probability(x,y): 

    '''
    Returns float probability of emitting x from y
    If invalid parameters, return None
    x: String value which is the emitted word 
    y: String value which is the given tag
    '''

    try:
        total_y_words = len(all_words[y])
        total_tag_to_word = all_words[y].count(x)
        return total_tag_to_word/total_y_words
    except:
        return None

## Test out #MK1
# print(get_emission_probability("trump","B-positive"))

#Part 2b
#During the testing phase, if the word does not appear in the training set, we replace that word with the
#special word token #UNK#
def test_get_emission_probability(x,y):
    '''
    Returns float probability of emitting x from y, accounting in #UNKN#
    If invalid parameters, return None
    x: String value which is the emitted word 
    y: String value which is the given tag
    '''
    global k_of_dict
    try:
        total_y_words = len(all_words[y])
        total_tag_to_word = all_words[y].count(x)
        if total_tag_to_word == 0:
            #replace word with #UNK#
            x="#UNK#"
            k_of_dict[y]+=1
            return (k_of_dict[y]/(total_y_words+k_of_dict[y]))
        else:
            return total_tag_to_word/(total_y_words+k_of_dict[y])
    except:
        return None

## Testing out test_get_emission_probability(x,y):
# print(test_get_emission_probability("kahwee","B-positive"))

# Part 2c
# argmax word to tag
def preprocess_word_prediction(in_file):
    '''
    Returns a dataframe after processing input file
    in_file: input text file with 1 column of words 
    '''
    inputList=[]
    inputFile = open(in_file, 'r')
    for line in inputFile:
        line = line.strip()
        if len(line) == 0:continue
        inputList.append(line)
    #print(inputList)
    df = pd.DataFrame(inputList, columns = ["Word"])
    return df

def tag_creator(x):
    '''
    Returns the most probable word, calls test_get_emission_probability inside
    x: word to predict tag 
    '''
    highest_prob = 0
    most_probable = ""
    for key in all_words:
        if x in all_words[key]:
            curr_prob = test_get_emission_probability(x, key)
            if curr_prob > highest_prob:
                highest_prob = curr_prob
                most_probable = key
    return most_probable


filename = "SG/dev.in"
filechina="CN/dev.in"
fileen="EN/dev.in"
filefr="FR/dev.in"

# File Generation #TODO Uncomment later
# outframe = preprocess_word_prediction(filename)
# outframe['Tag'] = outframe['Word'].apply(tag_creator)
# outframe.to_csv("SG/devSG.out", sep=" ", index=False, header=False)



### The below is the transition from state to state

# For Transition
counter = 0
def startEndCol(df):
    """
    This function label the new column as either start or end based on the position of the "None" tag in the Tag Column.
    df: This is the dataframe you want to input. Dataframe needs to have columns Word, Tag, transit
    """
    dataframeSize = len(df.index)
    df.loc[0]['transit']= "Start"
    df.loc[dataframeSize-2]['transit']='End'
    global counter
    for rows in df.iterrows():
        if rows[1][1]==None and counter<dataframeSize-1:
            df.loc[counter-1]['transit'] = "End"
            df.loc[counter+1]['transit'] = "Start"
            counter+=1
        else: counter+=1
    return df


transitionframe = startEndCol(df)

#print(precision(x,y))
#print(recall(x,y))
#print(F_score())

# Print after processing
# print(transitionframe)

#Part 3

###
transition_dict = {"Start": [], "End":[]}
transitionframe = transitionframe.replace('\n','', regex=True)
previous_Tag = "Starter"
for index, row in transitionframe.iterrows():
    current_Tag = row["Tag"]
    if previous_Tag == "Starter":
        previous_Tag == current_Tag
    if row['transit'] == "Start":
        transition_dict["Start"].append(current_Tag)
    elif row["transit"]=="End":
	    transition_dict[current_Tag].append("End")
    else:
        if previous_Tag not in transition_dict:
            transition_dict[previous_Tag] = [current_Tag]
        else:
            transition_dict[previous_Tag].append(current_Tag)
    previous_Tag = current_Tag

# Print after dictionary
# print(transition_dict)



# Part 3b
# Predict Label
def viterbi(curr_word, prev_tag):
    unique_set = set(transition_dict[prev_tag])
    unique_tags = list(unique_set)
    probability_dict = {}
    for tag in unique_tags:
        probability_dict[tag] = transition_dict[prev_tag].count(tag)/len(transition_dict[previous_Tag])
    for tag in unique_tags:
        probability_dict[tag] = probability_dict[tag]*test_get_emission_probability(curr_word, tag)
    
    highest_probability = 0
    most_probable = ""
    for probability_tag in probability_dict:
        if probability_dict[probability_tag] > highest_probability:
            highest_probability = probability_dict[probability_tag]
            most_probable = probability_tag
    return most_probable

outframe = preprocess_word_prediction(filename)
outframe['Tag'] = outframe['Word'].apply()
outframe.to_csv("SG/devSG.out", sep=" ", index=False, header=False)



# KIV DESIGN CHALLENGE
# class node(object):
#     def __init__(self):
#         self.previous_tag = None
#         self.highest_previous = -1
#         self.next_tag = None
#         self.highest_next = -1

#     def set_previous(self, )


