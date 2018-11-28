import pandas as pd
import re
import numpy as np
import sys
import math

running = "FR"

filetrain = "{}/train".format(running)
filetest = "{}/dev.in".format(running)

def file_to_df(inputfile):
    """
    Function to process input file to dataframe
    inputfile: File to be processed

    Returns:
    df: Output dataframe
    all_tags: unique tags
    all_words: dictionary of all words with tag as key
    k_of_dict = dictionary of unknown var k
    """

    fin = open(inputfile,encoding="UTF-8")
    rawframe=[]
    all_tags = [] # List holding All unique Tags
    all_words = {} # dictionary of all words with tag as key
    k_of_dict={}
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
    print("File Processing Completed")
    return df, all_tags, all_words, k_of_dict

def startEndCol(df):
    """
    This function label the new column as either start or end based on the position of the "None" tag in the Tag Column.
    df: This is the dataframe generated from Part 2 without start and end. Dataframe needs to have columns Word, Tag, transit
    """
    dataframeSize = len(df.index)
    df.loc[0]['transit']= "Start"
    df.loc[dataframeSize-2]['transit']='End'
    counter = 0
    for rows in df.iterrows():
        if rows[1][1]==None and counter<dataframeSize-1:
            df.loc[counter-1]['transit'] = "End"
            df.loc[counter+1]['transit'] = "Start"
            counter+=1
        else: counter+=1
    print("Start End Columns Assigned")
    return df


def transition_2_viterbi(transitionframe):
    '''
    Creates a dictionary of transitions
    transitionframe: dataframe to be that transitions will be based on
    
    Returns:
    transition_dict: dictionary with transitions from transition frame
    '''
    transition_dict = {"Start": [], "End":[]}
    transitionframe = transitionframe.replace('\n','', regex=True)
    previous_Tag = "Start"
    previous_previous_Tag = "Start"
    # print(transitionframe.head(3410))
    for index, row in transitionframe.iterrows():
        if row["Word"].strip() == "":
            continue
        current_Tag = row["Tag"]
        #This is meant for one word situation. One word only give End       
        if row["transit"]=="End":
            if transitionframe.loc[index-1]["Word"].strip() =="":
                transition_dict["Start"].append(current_Tag)
                continue
        #This is meant for firstword situation
        if row["transit"]=="Start": 
            transition_dict["Start"].append(current_Tag)
        elif transitionframe.loc[index-1]["transit"] == "Start": #2nd word
            layer2key = ("Start", previous_Tag)
            if layer2key in transition_dict:
                transition_dict[layer2key].append(current_Tag)
            else:
                transition_dict[layer2key] = [current_Tag]
        else: # 3rd word onwards
            layer2key = (previous_previous_Tag, previous_Tag)
            if layer2key in transition_dict:
                transition_dict[layer2key].append(current_Tag)
            else:
                transition_dict[layer2key] = [current_Tag]
            if row["transit"] == "End":
                layer2key = (previous_Tag,current_Tag)
                if layer2key in transition_dict:
                    transition_dict[layer2key].append("End")
                else:
                    transition_dict[layer2key] = ["End"]
        previous_previous_Tag = previous_Tag
        previous_Tag = current_Tag
    print("Transition Dictionary Created")
    return transition_dict

df, all_tags, all_words, k_of_dict = file_to_df(filetrain)

transitionframe = startEndCol(df)
pd.set_option("display.max_rows", 3410)

# Removing columns with [Tag] Column=None
# transitionframe = transitionframe[~transitionframe['Tag'].isin([None])]
transitionframe.reset_index(drop=True)
transition_dict = transition_2_viterbi(transitionframe)

# for key in transition_dict:
#     print("{}: {}".format(key, len(transition_dict[key])))


def layer2_get_transition_probability(y1,y2,y3,transition_dict):
    '''
    Calculates the probability of y1 and y2 going into y3
    y1: previous previous tag
    y2: previous tag
    y3: probable tag
    transition_dict: dictionary of transitions
    
    Returns:
    probability of y1 and y2 going into y3
    
    '''
    try:
        if y1=="":
            if y2 == "Start": # 2nd word
                count_y1_y2_y3 = transition_dict["Start"].count(y3)
                count_y1_y2 = len(transition_dict["Start"])
            else:
                print("Wrong tag")
                return 0.0
        else:
            count_y1_y2_y3 = transition_dict[(y1,y2)].count(y3)
            count_y1_y2 = len(transition_dict[(y1,y2)])
        probability = count_y1_y2_y3/count_y1_y2
        return float(probability)
    except:
        return 0.0

print(layer2_get_transition_probability("", "Start", "O", transition_dict))


def viterbi2(sentence, sentence_index, outframe, transition_dict, k_of_dict, all_words):
    pass