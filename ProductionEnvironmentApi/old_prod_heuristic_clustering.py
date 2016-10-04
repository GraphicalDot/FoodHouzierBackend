#!/usr/bin/env python

"""
Author: Kaali
Dated: 9 march, 2015
Purpose: This module deals with the clustering of the noun phrases, Evverything it uses are heuristic rules because
till now i am unable to find any good clutering algorithms which suits our needs.

Edit 1: 15 May to 21 May


"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import requests
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import Levenshtein
import codecs
import nltk
from compiler.ast import flatten
import time
from collections import Counter
import re
import math
import jaro
import os
import sys
from nltk.tag.hunpos import HunposTagger
this_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_file_path)
from Text_Processing.PosTaggers import PosTaggerDirPath, HunPosModelPath, HunPosTagPath
from Text_Processing.colored_print import bcolors
from GlobalConfigs import DEBUG
from Text_Processing import SentenceTokenizationOnRegexOnInterjections


def encoding_helper(__object):
        if isinstance(__object, unicode):
                obj  = unicode(__object)
        obj.encode("ascii", "xmlcharrefreplace")
        return obj


class SimilarityMatrices:
        
        @staticmethod
        def levenshtein_ratio(__str1, __str2):
                ratio = 'Levenshtein.ratio("{1}", "{0}")'.format(__str1, __str2)
                return eval(ratio)


        @staticmethod
        def modified_dice_cofficient(__str1, __str2):
                __str1, __str2 = __str1.replace(" ", ""), __str2.replace(" ", "")
                __ngrams = lambda __str: ["".join(e) for e in list(nltk.ngrams(__str, 2))]
                __l = len(set.intersection(set(__ngrams(__str1)), set(__ngrams(__str2))))
                total = len(__ngrams(__str1)) + len(__ngrams(__str2))
                """
                if len(set.intersection(set(__str1.split(" ")), set(__str2.split(" ")))) \
                        >= min(len(__str1.split(" ")), len(__str2.split(" "))):
                        print "New matric found  beyween %s and %s\n"%(__str1, __str2)
                        return 0.9
                """
                try:
                    return  float(__l*2)/total
                except Exception as e:
                    return 0

        @staticmethod
        def get_cosine(__str1, __str2):
                """
                Returns 0.0 if both string doesnt have any word common
                for example
                In[#]: get_cosine(text_to_vector('uttappams'), text_to_vector('appams'))
                Out[#]: 0.0
                
                In[#]: get_cosine(text_to_vector('love masala dosai'), text_to_vector('onion rawa masala dosa'))
                Out[#]: 0.288
                
                In[#]: get_cosine(text_to_vector('awesme tast'), text_to_vector('good taste'))
                Out[#]: 0.0
                
                In[#]: get_cosine(text_to_vector('awesme taste'), text_to_vector('good taste'))
                Out[#]: 0.5
                """
                vector1 = text_to_vector(__str1)
                vector2 = text_to_vector(__str2)

                WORD = re.compile(r'\w+')

                def text_to_vector(text):
                        words = WORD.findall(text)
                        return Counter(words)
                
                intersection = set(vector1.keys()) & set(vector2.keys())
                numerator = sum([vector1[x] * vector2[x] for x in intersection])
                
                sum1 = sum([vec1[x]**2 for x in vector1.keys()])
                sum2 = sum([vec2[x]**2 for x in vector2.keys()])
                denominator = math.sqrt(sum1) * math.sqrt(sum2)
                
                if not denominator:
                        return 0.0
                else:
                        return float(numerator) / denominator


        @staticmethod
        def jaro_winkler(__str1, __str2):
                def to_unicode(__str):
                        if isinstance(__str, unicode):
                                return __str
                        return unicode(__str)

                return jaro.jaro_winkler_metric(to_unicode(__str1), to_unicode(__str2))



        @staticmethod
        def check_if_shortform(self, str1, str2):
                """
                To identify if "bbq nation" is similar to "barbeque nation"

                """
                if bool(set.intersection(set(str1.split()), set(str2.split()))):

                        if set.issubset(set(str1), set(str2)):
                                return True

                        if set.issubset(set(str2), set(str1)):
                                return 

                return False

class ProductionHeuristicClustering:
        def __init__(self, sentiment_np_time, sub_category=None, sentences=None, eatery_name=None):
                """ 
                Args:
                    sentiment_nps:
                        [[u'positive',[u'paneer chilli pepper starter']], [u'positive', []],
                         [u'positive', [u'friday night']], [u'positive', []],                                   
                         [u'super-positive', [u'garlic flavours', u'penne alfredo pasta']]],
                """

                if eatery_name:
                        self.list_to_exclude = flatten(["food", "service", "cost", "ambience", "place", \
                                "Place", "i", "great", "good", eatery_name.lower().split(), "rs", "delhi",\
                                "india", "indian"])
                        #self.list_to_exclude = ["food", "service", "cost", "ambience", "delhi", "Delhi", 
                        #       "place", "Place", __eatery_name.lower().split()]
                else:
                        self.list_to_exclude = ["food", "i", "service", "cost", "ambience", "delhi", \
                                "Delhi", "place", "Place", "india", "indian"]
                

                self.sentiment_np_time = sentiment_np_time
                self.sentences = sentences
                self.sub_category = sub_category
                new_list, self.clusters, self.result = list(), list(), list()
                

        def run(self):
                self.merged_sentiment_nps = self.merge_similar_elements()
                __sorted = sorted(self.merged_sentiment_nps.keys())
                self.list_to_exclude = flatten(self.list_to_exclude)
                #self.NERs = self.ner()
                self.keys = self.merged_sentiment_nps.keys()
                self.filter_clusters()
                #The noun phrases who were not at all in the self.clusters
                self.without_clusters =  set.symmetric_difference(set(range(0, len(self.keys))), \
                                                                    set(flatten(self.clusters)))
                self.populate_result()

                if self.sub_category == "dishes":
                        self.common_ners = list(set.intersection(set([e[0] for e in self.ner()]), \
                                                        set([e[0] for e in self.custom_ner()])))
                        self.result = self.filter_on_basis_pos_tag()
                        self.result = sorted(self.result, reverse=True, key= lambda x: x.get("total_sentiments"))
                        return self.add_sentiments(self.result)
                
                
                return self.add_sentiments(self.result)


        def add_sentiments(self, __list):
                """
                This takes in a list of dictionaries with sentiments present for each dictionary, 
                and then adds a new key to every dictionary which is the sum of all the sentiments
                """
                __add =  lambda x: x.get("positive") + x.get("negative")+ x.get("neutral") + x.get("super-positive")\
                                    + x.get("super-negative")
                [__dict.update({"total_sentiments": __add(__dict)}) for __dict in __list]
                return __list



        #@print_execution
        def merge_similar_elements(self):
                """
                Result:
                    Merging noun phrases who have exact similar spellings with each other and return a 
                    dictionary in the form
                    u'ice tea': {'positive', 6, 'negative': 5, "neutral": 5, "super-positive": 0, 
                    "super-negative": 10},
                    u'iced tea': {'positive', 2, 'negative', 10, "neutral": 230, "super-positive": 5, 
                    "super-negative": 5},
                }
                """
                
                without_similar_elements = dict()
                for (sentiment, noun_phrases, review_time) in self.sentiment_np_time:
                        for __np in noun_phrases:
                                """
                                if i.get("name") in list(set(self.NERs)):
                                print "This noun_phrase belongs to ner {0}".format(i.get("name"))
                                pass
                                """
                                #if bool(set.intersection(set(__np.split(" ")),  set(self.list_to_exclude))):
                                #       pass    

                                if without_similar_elements.get(__np):
                                        result = without_similar_elements.get(__np)
                                        timeline = result.get("timeline")
                                        timeline.append((sentiment, review_time))
                                        
                                        positive, negative, neutral, super_positive, super_negative = \
                                                result.get("positive"), result.get("negative"),result.get("neutral"), \
                                                result.get("super-positive"), result.get("super-negative")
                                        

                                        new_frequency_negative = (negative, negative+1)[sentiment == "negative"]
                                        new_frequency_positive = (positive, positive+1)[sentiment == "positive"]
                                        new_frequency_neutral = (neutral, neutral+1)[sentiment == "neutral"]
                                        new_frequency_super_positive = (super_positive, super_positive+1)[sentiment == \
                                                "super-positive"]
                                        new_frequency_super_negative = (super_negative, super_negative+1)[sentiment == \
                                                "super-negative"]
                                

                                        without_similar_elements.update(
                                            {__np: 
                                                {"negative": new_frequency_negative, "positive": new_frequency_positive,
                                                    "neutral": new_frequency_neutral, "super-positive": \
                                                            new_frequency_super_positive, 
                                                    "super-negative": new_frequency_super_negative,
                                                    "timeline": timeline,
                                            }})

                
                                else:
                                    without_similar_elements.update(
                                    {__np: 
                                        {"negative": (0, 1)[sentiment=="negative"], "positive": (0, 1)[sentiment=="positive"],
                                            "neutral": (0, 1)[sentiment=="neutral"], 
                                            "super-positive": (0, 1)[sentiment == "super-positive"], 
                                            "super-negative": (0, 1)[sentiment == "super-negative"], 
                                            "timeline": [(sentiment, review_time)],
                                            }})
                
                return without_similar_elements

        #@print_execution
        def filter_clusters(self):
                """
                self.sent_sentiment_nps gave rise to merged_sent_sentiment_nps
                outputs:
                    self.clusters which will have list of lists 
                    with each list having index numbers of the elements who were found to be similar
                """


                X = np.zeros((len(self.keys), len(self.keys)), dtype=np.float)
                for i in xrange(0, len(self.keys)):
                        for j in xrange(0, len(self.keys)):
                                if i == j:
                                        #If calculating for same element
                                        X[i][j] = 0.5
                                        X[j][i] = 0.5
                                    
                                if X[i][j] == 0:
                                        #st = 'Levenshtein.ratio("{1}", "{0}")'.format(self.keys[i], self.keys[j])
                                        #ratio = eval(st)
                                        #ratio = SimilarityMatrices.levenshtein_ratio(self.keys[i], self.keys[j])
                                        ratio = SimilarityMatrices.modified_dice_cofficient(self.keys[i], self.keys[j])
                                        X[i][j] = ratio
                                        X[j][i] = ratio
            
            
                #Making tuples of the indexes for the element in X where the rtion is greater than .76
                #indices = np.where((X > .75) & (X < 1))
                indices = np.where(X > .75)
                new_list = zip(indices[0], indices[1])

                found = False
                for e in new_list:
                        for j in self.clusters:
                                if bool(set.intersection(set(e), set(j))):
                                        j.extend(e)
                                        found = True
                                        break
                        if not found:    
                                self.clusters.append(list(e))
                                found = False
                                
                        found = False
                
                self.clusters = [list(set(element)) for element in self.clusters if len(element)> 2]
                found = False
                new_clusters = list()

                for e in self.clusters:
                        for j in new_clusters:
                                if bool(set.intersection(set(e), set(j))):
                                        j.extend(e)
                                        found = True
                                        break
                        if not found:    
                                new_clusters.append(list(e))
                                found = False
                                
                        found = False

                self.clusters = new_clusters
                #Removing duplicate elements from clusters list
                return 

        #@print_execution
        def populate_result(self):
                """
                without_clusters will have index numbers of the noun phrases for whom no other similar
                noun_phrases were found
                self.result will be populated after execution of this method
                """
                for __int_key in self.without_clusters:
                        new_dict = dict()
                        name = self.keys[__int_key] #name of the noun phrase corresponding to this index number
                        new_dict = self.merged_sentiment_nps[name]
                        new_dict.update({"name": name})
                        new_dict.update({"similar": list()})
                        self.result.append(new_dict)
                
                for cluster_list in self.clusters:
                        __dict = self.maximum_frequency(cluster_list)
                        self.result.append(__dict)
                return

        #@print_execution
        def maximum_frequency(self, cluster_list):
                """
                Returning name with maximum frequency in a cluster, by joining all the frequencies
                cluster_list: [0, 17, 12, 37, 22]
                
                """
                result = list()
                positive, negative, neutral, super_positive, super_negative = int(), int(), int(), int(), int()
                timeline = list()
               
                cluster_names = [self.keys[element] for element in cluster_list]
                whole_cluster_names_n_keys = [self.merged_sentiment_nps.get(self.keys[element]) for element in cluster_list]

                for element in cluster_list:
                        name = self.keys[element]    
                        new_dict = self.merged_sentiment_nps[name]
                        new_dict.update({"name": name})
                        result.append(new_dict)        
                        positive = positive +  self.merged_sentiment_nps[name].get("positive") 
                        negative = negative +  self.merged_sentiment_nps[name].get("negative") 
                        neutral = neutral +  self.merged_sentiment_nps[name].get("neutral") 
                        super_negative = super_negative +  self.merged_sentiment_nps[name].get("super-negative") 
                        super_positive = super_positive +  self.merged_sentiment_nps[name].get("super-positive") 
                        timeline.extend(self.merged_sentiment_nps[name].get("timeline"))

                whole = dict()
                for a in cluster_names:
                        __list = list()
                        for b in cluster_names :
                                __list.append(SimilarityMatrices.modified_dice_cofficient(a, b))
                        whole.update({a: sum(__list)})
                
                name = filter(lambda x: whole[x] == max(whole.values()), whole.keys())[0]

                return {"name": name, "positive": positive, "negative": negative, "neutral": neutral, 
                        "super-negative": super_negative, "super-positive": super_positive, "similar": whole_cluster_names_n_keys,
                        "timeline": timeline}

        #@print_execution
        def custom_ner(self):
                ner = list()
                regexp_grammer = r"NER:{<IN><NN.*><NN.*>?}"
                __parser = nltk.RegexpParser(regexp_grammer)

                hunpos_tagger = HunposTagger(HunPosModelPath, HunPosTagPath)
                for __sentence in self.sentences:
                        try:
                                tagged = hunpos_tagger.tag(nltk.word_tokenize(encoding_helper(__sentence)))
                                tree = __parser.parse(tagged)
                                for subtree in tree.subtrees(filter = lambda t: t.label()=='NER'):
                                        l = " ".join([e[0] for e in subtree.leaves() if e[1] == 'NNP' or e[1] == 'NNS' or e[1] == 'NN'])
                                        ner.append(l.lower())
                        except Exception as e:
                                pass

                result = sorted(Counter(ner).items(), reverse=True, key=lambda x: x[1])
                return result



        #@print_execution
        def ner(self):
                __list = list()
                for sent in self.sentences:
                        tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent.encode("ascii", "xmlcharrefreplace"))))
                        for subtree in tree.subtrees(filter = lambda t: t.label()=='GPE'):
                                __list.append(" ".join([e[0] for e in subtree.leaves()]).lower())
                
                ners = Counter(__list)
                result = sorted(ners.items(), reverse=True, key=lambda x: x[1])
                return result

        def filter_on_basis_pos_tag(self):
                """
                pos tagging of noun phrases will be d
                one, and if the noun phrases contains some adjectives or RB or FW, 
                it will be removed from the total noun_phrases list

                Any Noun phrases when split, if present in self.list_to_exclude will not be included in the final result
                for Example: 
                self.list_to_exclude = ["food", "i", "service", "cost", "ambience", "delhi", "Delhi", "place", "Place"]
                noun_phrase = "great place"
                
                """
                hunpos_tagger = HunposTagger(HunPosModelPath, HunPosTagPath)
                filtered_list = list()
                for __e in self.result:
                        __list = [pos_tag for (np, pos_tag) in hunpos_tagger.tag(nltk.wordpunct_tokenize(__e.get("name").encode("ascii", "ignore")))]
                        if set.intersection(set(__list), set(["FW", "CD", "LS"])):
                                    print "This will be droppped out of total noun phrases %s"%__e.get("name")
                        else:
                            filtered_list.append(__e)


                return filtered_list





if __name__ == "__main__":
        """
        def transform(__object): 
                __l = __object[1] 
                __l.append(__object[0]) 
                return list(set(__l))

        __dict = {} 
        for e in __list:
                __dict.setdefault(min(e), []).append(max(e))
        
        return map(transform, __dict.keys())

        
        
        payload = {'category': 'food',
             'eatery_id': '301489',
              'ner_algorithm': 'stanford_ner',
               'pos_tagging_algorithm': 'hunpos_pos_tagger',
                'total_noun_phrases': 15,
                 'word_tokenization_algorithm': 'punkt_n_treebank'}

        r = requests.post("http://localhost:8000/get_word_cloud", data=payload)
        result = r.json()["result"]
        """
        result = [{u'polarity': 1, u'frequency': 84, u'name': u'main course'},
                {u'polarity': 0, u'frequency': 1, u'name': u'main course order'},
                {u'polarity': 1, u'frequency': 2, u'name': u'other subway outlets'},
                {u'polarity': 1, u'frequency': 2, u'name': u'best subway outlet'},
                {u'polarity': 1, u'frequency': 1, u'name': u'good subway outlet'},
                {u'polarity': 1, u'frequency': 1, u'name': u'other subway'},
                
                {u'polarity': 0, u'frequency': 7, u'name': u'dal makhani'},
                {u'polarity': 0, u'frequency': 5, u'name': u'dal makhni'},
                {u'polarity': 1, u'frequency': 2, u'name': u'love dal makhni'},
                {u'polarity': 1, u'frequency': 2, u'name': u'daal makhni'},
                {u'polarity': 1, u'frequency': 1, u'name': u'daal makhani creamy'},
                {u'polarity': 1, u'frequency': 1, u'name': u'daal punjabi'},
                {u'polarity': 1, u'frequency': 1, u'name': u'paneer lababdaar'},
                {u'polarity': 1, u'frequency': 1, u'name': u"' dal makhani"},
                {u'polarity': 0, u'frequency': 1, u'name': u'dal makhni'},
                {u'polarity': 0, u'frequency': 1, u'name': u'dal makahani'},
                {u'polarity': 0, u'frequency': 1, u'name': u'daal makhani'},


                {u'polarity': 0, u'frequency': 5, u'name': u'paneer lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'murg lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'pannneer lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'paaner lababdar'},
                {u'polarity': 1, u'frequency': 1, u'name': u'panner laabaabdar'},


                {u'polarity': 1, u'frequency': 3, u'name': u'parikrama restaurant'},
                {u'polarity': 1, u'frequency': 3, u'name': u'parikarma restaurant'},
                {u'polarity': 1, u'frequency': 1, u'name': u'parikrama drink'},
                {u'polarity': 1, u'frequency': 1, u'name': u'parikrama hotel'},
                {u'polarity': 1, u'frequency': 1, u'name': u'parikrama coz'},
                
                



                {u'polarity': 0, u'frequency': 60, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 26, u'name': u'barbeque nation'},
        {u'polarity': 0, u'frequency': 20, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 20, u'name': u'bbq nation'},
        {u'polarity': 0, u'frequency': 14, u'name': u'barbeque nation'},
        {u'polarity': 1, u'frequency': 13, u'name': u'good food'},
        {u'polarity': 1, u'frequency': 13, u'name': u'ice cream'},
        {u'polarity': 1, u'frequency': 12, u'name': u'chicken biryani'},
        {u'polarity': 1, u'frequency': 12, u'name': u'barbecue nation'},
        {u'polarity': 1, u'frequency': 10, u'name': u'great food'},
        {u'polarity': 1, u'frequency': 9, u'name': u'chicken tikka'},
        {u'polarity': 1, u'frequency': 9, u'name': u'good place'},
        {u'polarity': 1, u'frequency': 84, u'name': u'main course'},
        {u'polarity': 0, u'frequency': 60, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 26, u'name': u'barbeque nation'},
        {u'polarity': 0, u'frequency': 20, u'name': u'main course'},
        {u'polarity': 1, u'frequency': 20, u'name': u'bbq nation'},
        {u'polarity': 0, u'frequency': 14, u'name': u'barbeque nation'},
        {u'polarity': 1, u'frequency': 13, u'name': u'good food'},
        {u'polarity': 1, u'frequency': 13, u'name': u'ice cream'},
        {u'polarity': 1, u'frequency': 12, u'name': u'chicken biryani'},
        {u'polarity': 1, u'frequency': 12, u'name': u'barbecue nation'},
        {u'polarity': 1, u'frequency': 10, u'name': u'great food'},
        {u'polarity': 1, u'frequency': 9, u'name': u'chicken tikka'},
        {u'polarity': 1, u'frequency': 9, u'name': u'good place'},
        {u'polarity': 1, u'frequency': 9, u'name': u'paneer tikka'},
        {u'polarity': 1, u'frequency': 8, u'name': u'tasty food'},
        {u'polarity': 1, u'frequency': 8, u'name': u'gulab jamun'},
        {u'polarity': 0, u'frequency': 7, u'name': u'gulab jamun'},
        {u'polarity': 0, u'frequency': 7, u'name': u'chicken tikka'},
        {u'polarity': 0, u'frequency': 7, u'name': u'barbecue nation'},
        {u'polarity': 0, u'frequency': 7, u'name': u'bbq nation'},
        {u'polarity': 1, u'frequency': 4, u'name': u'dal makhani'},
        {u'polarity': 1, u'frequency': 4, u'name': u'main course everything'},
        {u'polarity': 1, u'frequency': 3, u'name': u'main course section'},
        {u'polarity': 0, u'frequency': 2, u'name': u'bt main course'},
        {u'polarity': 1, u'frequency': 2, u'name': u'main coursenot'},
        {u'polarity': 1, u'frequency': 2, u'name': u'fabulous main course'}]


        """
        m = HeuristicClustering(result, 'Barbeque Nation')
        for element in m.result:
                print element
        for __dict1 in result:
            for __dict2 in result:
                    #if not __dict1.get("name") != __dict2.get("name"):
                            st = 'Levenshtein.ratio("{1}", "{0}")'.format(__dict1.get("name"), __dict2.get("name"))
                            print eval(st), "\t", __dict1.get("name"), __dict2.get("name")


        test_data = [['non veg buffet', 'veg buffet', False],
        ['mutton seekh', 'mutton seekh kababs', True],
        ['mutton seekh', 'mutton seekh kabab', True],
        ['mutton dum biryani', 'delicious mutton biryani', True],
        ['main course', 'fabulous main course', True],
        ['other bar nations', 'barbeque nation', True],
        ['hot gulab jamun', 'da gulab jamun', True],
        ['main course menu', 'main course', True],
        ['unlimited tasty food', 'unlimited food', True],
        ['coconut brownie', 'chocolate brownie', False],
        ['neat experience', 'bad experience', True],
        ['barbeque nation team', 'barbeque nation buffet', False],
        ['chocolate cake', 'chocolate sauce', False],
        ['great sauce', 'great place', False],
        ['kadhai chicken', 'khatta chicken', False],
        ['kadhai chicken', 'chicken', False],
        ['kadhai chicken', 'peshawari chicken leg', False],
        ['kadhai chicken', 'teriyaki chicken', False],
        ['kadhai chicken', 'fish prawn chicken', False],
        ['kadhai chicken', 'teriyaki chicken', False],
        ['kadhai chicken', 'jerk chicken', False],
        ['kadhai chicken', 'chicken', False],
        ['bbq nation', 'barbque nation', True],
        ['gud mutton seekh kbab', 'mutton seekh kebab', True]] 
        
        def longest_common_string(str1, str2):
                break_strings = lambda __str: flatten([[e[0:i+1] for i in range(0, len(e))] for e in __str.split(" ")])
               
                common_strings = len(list(set.intersection(set(break_strings(str1)), set(break_strings(str2)))))

                min_length = min(sum([len(e) for e in str1.split(" ")]), sum([len(e) for e in str2.split(" ")]))
                max_length = max(sum([len(e) for e in str1.split(" ")]), sum([len(e) for e in str2.split(" ")]))
                if min_length == min([len(e) for e in flatten([str1.split(" "), str2.split(" ")])]):
                        
				return float(common_strings)/max_length
                return float(common_strings)/min_length

      
        for __e in test_data:
                st = 'Levenshtein.ratio("{1}", "{0}")'.format(__e[0], __e[1])
                print __e[0], "\t", __e[1], __e[2], "\t", True if longest_common_string(__e[0], __e[1]) > .75 else False

        result = [[5, 10, 5, 11, 5, 24, 5, 28, 10, 5, 10, 11, 10, 24, 10, 28, 11, 5, 11, 10, 11, 24, 11, 28, 
                24, 5, 24, 10, 24, 11, 24, 28, 28, 5, 28, 10, 28, 11, 28, 24], [6, 9, 6, 13, 6, 14, 6, 23, 
                6, 30, 9, 6, 9, 13, 9, 14, 9, 23, 9, 30, 13, 6, 13, 9, 13, 14, 13, 23, 13, 30, 14, 6, 14, 9, 14, 
                13, 14, 23, 14, 30, 23, 6, 23, 9, 23, 13, 23, 14, 23, 30, 30, 6, 30, 9, 30, 13, 30, 14, 30, 23], 
                [15, 22, 15, 34, 22, 15, 34, 15], [31, 32, 32, 31]]

        __list = [[10, 5], [11, 5], [24, 5], [28, 5], [9, 6], [13, 6], [14, 6], [23, 6], [30, 6], [6, 9], 
                [13, 9], [14, 9], [23, 9], [30, 9], [5, 10], [11, 10], [24, 10], [28, 10], [5, 11], [10, 11], 
                [24, 11], [28, 11], [6, 13], [9, 13], [14, 13], [23, 13], [30, 13], [6, 14], [9, 14], [13, 14], 
                [23, 14], [30, 14], [22, 15], [34, 15], [15, 22], [6, 23], [9, 23], [13, 23], [14, 23], [30, 23], 
                [5, 24], [10, 24], [11, 24], [28, 24], [5, 28], [10, 28], [11, 28], [24, 28], [6, 30], [9, 30], 
                [13, 30], [14, 30], [23, 30], [32, 31], [31, 32], [15, 34]]

        [[u'philadelphia rolls', 'Philadelphia rolls', u'philadelphia', u'veg philadelphia rolls'], [u'amazing gastronomic experience', 'gastronomic experience'], ['nut chocolate', 'chocolate', u'chunky nut chocolate'], ['Yuzu Miso Sauce', 'Japanese miso soup', 'Japanese Yuzu miso sauce', u'yuzu miso sauce', u'japanese yuzu miso sauce'], [u'fried cream spring rolls', u'kylin special fried ice cream spring rolls', 'spring rolls', 'cream spring rolls', 'Fried Ice cream spring rolls', 'ice cream spring rolls', u'fried ice cream spring rolls'], ['Chicken Teriyaki', u'chicken teriyaki', u'chicken tepiniyaki dish', u'chicken teppanyaki'], ['Italian Smooch', u'italian smooch', ':- 1. italian smooch'], ['Spicy Salmon roll', 'salmon rolls', u'spicy salmon rolls', 'salmon maki', u'spicy salmon roll', 'Salmon Roll', u'salmon roll', u'spicy salmon maki'], [u'mango breeze drink', 'Mango breeze', u'mango breeze', 'mango Breeze drink'], ['Japanese Restaurant', u'authentic japanese restaurant', u'japanese restaurant', u'japanese 5 star restaurants', 'Japanese 5 star Restaurants', 'Japanese restaurant']]
        """
        
        
        __list = [
                    ["positive", [u'philadelphia rolls',  u'japanese restaurant', u'japanese 5 star restaurants'
                        'Philadelphia rolls', u'philadelphia', u'chicken tepiniyaki dish']], 

                    ["super-positive",  [u'veg philadelphia rolls', u'authentic japanese restaurant', u'japanese restaurant']], 
                    ["negative", [u'amazing gastronomic experience', u'chicken teppanyaki']], 
                    ["neutral",     ['nut chocolate', 'chocolate', u'chunky nut chocolate']], 
                    ["super-negative",     ['yuzu Miso Sauce', 'japanese miso soup', u'japanese yuzu miso sauce']], 
                    ["positive", [u'fried cream spring rolls', u'kylin special fried ice cream spring rolls', 
                        'spring rolls', 'cream spring rolls', 'fried Ice cream spring rolls']], 
                        
                    ["negative", ['ice cream spring rolls', u'fried ice cream spring rolls']], 
                    ["neutral", ['chicken teriyaki', u'chicken teriyaki', u'chicken teppanyaki',  'mango breeze']], 
                    ["super-negative", [u'mango breeze','sallmon rolls', u'spicy salmon rolls', 'salmon maki', u'spicy salmon roll']], 
                    ["super-positive", ['italian Smooch', u'italian smooch', ':- 1. italian smooch']], 
                    ["positive", ['spicy salmon roll', 'salmon rolls', u'spicy salmon rolls', 'salmon maki']],
                    ["neutral", [u'spicy salmon roll', 'salmon roll', u'salmon roll', u'spicy salmon maki']],
                    ["super-negative", ['philadelphia rolls', u'philadelphia',]], 
                     ["neutral",[u'mango breeze drink', 'mango breeze drink', 'japanese yuzu miso sauce']], 
                    ["neutral", [u'yuzu miso sauce', 'gastronomic experience']], 
                    ["super-positive", ['japanese Restaurant', u'authentic japanese restaurant', u'japanese restaurant']],
                    ["positive", [u'japanese 5 star restaurants', 'japanese 5 star Restaurants']],
                    ["negative", ['japanese restaurant',  u'chicken teriyaki', u'chicken teppanyaki']],
                    ["neutral", ['spicy salmon roll', 'salmon rolls', u'spicy salmon rolls']]]



        def return_food_sentences(eatery_id):
                from sklearn.externals import joblib
                sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
                reviews_list = list()
                for post in reviews.find({"eatery_id": eatery_id}):
                        reviews_list.extend([[sent, post.get("review_time")] for sent in sent_tokenizer.tokenize(post.get("review_text"))])
                

                tags = TAG_CLASSIFIER_LIB.predict([e[0] for e in reviews_list])
                food_sentences = list()
                for (sent, review_time),  tag in zip(reviews_list, tags):
                        if tag == "food":
                                food_sentences.append([sent, review_time])
   
                sub_tags = FOOD_SB_TAG_CLASSIFIER_LIB.predict([e[0] for e in food_sentences])

                dishes_n_drinks = list()

                for (sent, review_time), sub_tag in zip(food_sentences, sub_tags):
                        if sub_tag == "dishes" or sub_tag == "drinks":
                                dishes_n_drinks.append([sent, review_time])
                        
    
  

                sentiments = SENTI_CLASSIFIER_LIB.predict([e[0] for e in dishes_n_drinks])
    
                from topia.termextract import extract
                topia_extractor = extract.TermExtractor()
                noun_phrases = list()
                for (sent, review_time), tag in zip(dishes_n_drinks, sentiments):
                        nouns = topia_extractor(sent)
                        noun_phrases.append([tag, [e[0].lower() for e in nouns], review_time])
                        
                return (filter(lambda x: x[1], noun_phrases), [e[0] for e in dishes_n_drinks])

                


        sentences = [u'do try their Paneer Chilli Pepper starter .',
                 u'even through they have numerous drinks in the menu , on a Friday night ( when i visited the place ) they were serving only specific brands of liquor .',
                  u"so the place is gorgeousambience might have felt better if there wasn't a party goingin case you are looking a pleasant dinner with friends and family you need to check if there is party or event going onit can be quite annoyingapart from that the best appetizer to try is Mustard Fish paparikaits absolutely amazing and deliciously juicyand the prawns are pretty good tooi was pretty disappointed with the pastamy lil sister could cook better pastaso pasta here is a complete no notry going for the pizzas insteadand their appetizers have a decent variety toothe service was pretty good and friendlybut it really got quite irritating cause of the party going on",
                   u'foodi ordered a Quattro Formaggi Pizza and Litchi Caipiroska , i was served my Caipiroska while i was waiting for the Pizza .',
                    u"the Caipiroska was delicious and i couldn't stop myself from praising it .",
                     u'we are pure vegetarians that also jain , we ordered for a Veg Pasta n they served us Pasta which was full of chicken pieces .']
        sentiment_np_time = [
                (u'positive', [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                (u'positive', [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                (u'positive', [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                (u'positive', [u'paneer chilli peppe starter'], u'2014-09-19 06:56:42'),
                (u'positive', [u'paneer chilli peppe ster'], u'2014-09-19 06:56:42'),
                (u'positive', [u'panee chilli peppe starter'], u'2014-09-19 06:56:42'),
                (u'positive', [u'paneer chilli peppe starr'], u'2014-09-19 06:56:42'),
                 
                (u'positive', [u'friday night'], u'2014-09-19 06:56:42'),
                  (u'positive',
                        [u'pizzas insteadand',
                               u'pastamy lil sister',
                                  u'party goingin case',
                                     u'party',
                                        u'mustard fish paparikaits',
                                           u'variety toothe service',
                                              u'pastaso pasta'],
                          u'2014-07-22 01:01:25'),
                   (u'neutral',
                         [u'litchi caipiroska', u'quattro formaggi pizza'],
                           u'2014-05-06 13:06:56'),
                    (u'super-positive', [u'i couldn'], u'2014-05-06 13:06:56'),
                     (u'neutral', [u'chicken pieces', u'veg pasta n'], u'2014-06-20 15:11:42')]

        
        sentiment_np_time, sentences = return_food_sentences("308322")
        print sentiment_np_time
        ins = ProductionHeuristicClustering(sentiment_np_time, "dishes", sentences,)
        i = ins.run()
        for e in ins.run():
            print e
    
        print i[0].keys()
