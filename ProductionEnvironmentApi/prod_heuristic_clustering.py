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
import openpyxl
from nltk.tag.hunpos import HunposTagger
this_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_file_path)
from Text_Processing.PosTaggers import PosTaggerDirPath, HunPosModelPath, HunPosTagPath
from Text_Processing.colored_print import bcolors
from Text_Processing import SentenceTokenizationOnRegexOnInterjections
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")


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
        def __init__(self, sentiment_np_time, sub_category=None, sentences=None, eatery_name=None, places=None, eatery_address=None):
                """ 
                Args:
                    sentiment_nps:
                        [[u'good',[u'paneer chilli pepper starter']], [u'good', []],
                         [u'good', [u'friday night']], [u'good', []],                                   
                         [u'excellent', [u'garlic flavours', u'penne alfredo pasta']]],
                """

                self.list_to_exclude = list()

                print "List of places %s"%places

                if places:
                        places = list(set(flatten(places)))
                        self.list_to_exclude.extend(places)
                

                if eatery_name:
                        #self.list_to_exclude.extend(eatery_name.lower().split())
                        self.list_to_exclude.append(eatery_name.lower())

                if eatery_address:
                        self.list_to_exclude.extend([e.lstrip() for e in eatery_address.lower().split(",")])


                self.list_to_exclude.extend(["i", "drink", "good", "great", "food", "service", "cost", "ambience", "place", "rs", "ok", "r", "taste", "lovers", "lover"])



                print self.list_to_exclude


                self.dropped_nps = list()

                ##stemming noun_phrases with snowball english stemmer 
                #self.sentiment_np_time = [(sentiment, [stemmer.stem(np) for np in nps], review_time) for (sentiment, nps, review_time) in sentiment_np_time]
                self.sentiment_np_time = sentiment_np_time
                self.sentences = sentences
                self.sub_category = sub_category
                new_list, self.clusters, self.result = list(), list(), list()

        def run(self):
                self.merged_sentiment_nps = self.merge_similar_elements()
                __sorted = sorted(self.merged_sentiment_nps.keys())
               
                self.keys = self.merged_sentiment_nps.keys()
                self.filter_clusters()
                self.without_clusters =  set.symmetric_difference(set(range(0, len(self.keys))), \
                                                                    set(flatten(self.clusters)))
                self.populate_result()

                ##only returns noun phrases that have toatal sentiments greater than 1
                __result = self.add_sentiments(self.result)
                __result = self.filter_on_basis_pos_tag(__result)
                
                result = [e for e in __result if e.get("total_sentiments") >1]
                
                
                ##the noun_phrase which have frequency lower than 1
                excluded_nps = [e for e in __result if e.get("total_sentiments") <=1]

                
                print "The np which have been discarded because of low frequency is %s"%(len(__result) - len(result))
               
                return {"nps": result, 
                        "excluded_nps": excluded_nps, #which had total_sentiemnts less than 1 
                        "dropped_nps":  self.dropped_nps }#which were excluded because they matched with places and address"

        def add_sentiments(self, __list):
                """
                This takes in a list of dictionaries with sentiments present for each dictionary, 
                and then adds a new key to every dictionary which is the sum of all the sentiments
                """
                __add =  lambda x: x.get("good") + x.get("poor")+ x.get("average") + x.get("excellent")\
                                    + x.get("terrible")
                [__dict.update({"total_sentiments": __add(__dict)}) for __dict in __list]
                
                __result = sorted(__list, reverse=True, key=lambda x: x.get("total_sentiments"))

                return __result



        #@print_execution
        def merge_similar_elements(self):
                """
                Result:
                    Merging noun phrases who have exact similar spellings with each other and return a 
                    dictionary in the form
                    u'ice tea': {'good', 6, 'poor': 5, "average": 5, "excellent": 0, 
                    "terrible": 10},
                    u'iced tea': {'good', 2, 'poor', 10, "average": 230, "excellent": 5, 
                    "terrible": 5},
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
                                
                                __list = [pos_tag for (np, pos_tag) in nltk.pos_tag(nltk.wordpunct_tokenize(__np.encode("ascii", "ignore")))]
                                if __np in self.list_to_exclude:
                                        print "This will be fucking dropped <<%s>>"%__np
                                        print nltk.pos_tag(nltk.wordpunct_tokenize(__np))
                                        self.dropped_nps.append(__np)
                               
                                elif not set.intersection(set(["NN", "NNS"]), set(__list)):
                                        print "This will be fucking dropped because of no presence of NNS and NN <<%s>>"%__np
                                        print nltk.pos_tag(nltk.wordpunct_tokenize(__np))
                                        self.dropped_nps.append(__np)

                                elif bool(set.intersection(set(__np.split(" ")),  set(self.list_to_exclude))):
                                        print "This will be dropped because of the list to exclude match in np <<%s>>"%__np
                                        self.dropped_nps.append(__np)
                                        
                        
                                elif without_similar_elements.get(__np):
                                        result = without_similar_elements.get(__np)
                                        timeline = result.get("timeline")
                                        timeline.append((sentiment, review_time))
                                        
                                        good, poor, average, excellent, terrible = \
                                                result.get("good"), result.get("poor"),result.get("average"), \
                                                result.get("excellent"), result.get("terrible")
                                        

                                        new_frequency_poor = (poor, poor+1)[sentiment == "poor"]
                                        new_frequency_good = (good, good+1)[sentiment == "good"]
                                        new_frequency_average = (average, average+1)[sentiment == "average"]
                                        new_frequency_excellent = (excellent, excellent+1)[sentiment == \
                                                "excellent"]
                                        new_frequency_terrible = (terrible, terrible+1)[sentiment == \
                                                "terrible"]
                                

                                        without_similar_elements.update(
                                            {__np: 
                                                {"poor": new_frequency_poor, "good": new_frequency_good,
                                                    "average": new_frequency_average, "excellent": \
                                                            new_frequency_excellent, 
                                                    "terrible": new_frequency_terrible,
                                                    "timeline": timeline,
                                            }})

                
                                else:
                                    without_similar_elements.update(
                                    {__np: 
                                        {"poor": (0, 1)[sentiment=="poor"], "good": (0, 1)[sentiment=="good"],
                                            "average": (0, 1)[sentiment=="average"], 
                                            "excellent": (0, 1)[sentiment == "excellent"], 
                                            "terrible": (0, 1)[sentiment == "terrible"], 
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
                good, poor, average, excellent, terrible = int(), int(), int(), int(), int()
                timeline = list()
               
                cluster_names = [self.keys[element] for element in cluster_list]
                whole_cluster_names_n_keys = [self.merged_sentiment_nps.get(self.keys[element]) for element in cluster_list]

                for element in cluster_list:
                        name = self.keys[element]    
                        new_dict = self.merged_sentiment_nps[name]
                        new_dict.update({"name": name})
                        result.append(new_dict)        
                        good = good +  self.merged_sentiment_nps[name].get("good") 
                        poor = poor +  self.merged_sentiment_nps[name].get("poor") 
                        average = average +  self.merged_sentiment_nps[name].get("average") 
                        terrible = terrible +  self.merged_sentiment_nps[name].get("terrible") 
                        excellent = excellent +  self.merged_sentiment_nps[name].get("excellent") 
                        timeline.extend(self.merged_sentiment_nps[name].get("timeline"))

                whole = dict()
                for a in cluster_names:
                        __list = list()
                        for b in cluster_names :
                                __list.append(SimilarityMatrices.modified_dice_cofficient(a, b))
                        whole.update({a: sum(__list)})
                
                name = filter(lambda x: whole[x] == max(whole.values()), whole.keys())[0]

                return {"name": name, "good": good, "poor": poor, "average": average, 
                        "terrible": terrible, "excellent": excellent, "similar": whole_cluster_names_n_keys,
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

        def filter_on_basis_pos_tag(self, result):
                """
                pos tagging of noun phrases will be d
                one, and if the noun phrases contains some adjectives or RB or FW, 
                it will be removed from the total noun_phrases list

                Any Noun phrases when split, if present in self.list_to_exclude will not be included in the final result
                for Example: 
                self.list_to_exclude = ["food", "i", "service", "cost", "ambience", "delhi", "Delhi", "place", "Place"]
                noun_phrase = "great place"
                
                """
                filtered_list = list()
                for __e in self.result:
                       
                       __list = [np for (np, pos_tag) in nltk.pos_tag(nltk.wordpunct_tokenize(__e.get("name").encode("ascii", "ignore"))) if pos_tag not in ["FW", "CD", "LS"] and np != "i"]
                       np = " ".join(__list)
                       if np != "":
                                __e.update({"name": np})
                                filtered_list.append(__e)
                       """
                        __list = [pos_tag for (np, pos_tag) in nltk.pos_tag(nltk.wordpunct_tokenize(__e.get("name").encode("ascii", "ignore")))]
                        
                        
                        if set.intersection(set(__list), set(["FW", "CD", "LS"])):
                                    print "This will be droppped out of total noun phrases %s"%__e.get("name")
                                    self.dropped_nps.append(__e)
                        else:
                            filtered_list.append(__e)
                        """

                return filtered_list





if __name__ == "__main__":
        wb = openpyxl.load_workbook("noun_phrases.xlsx")
        ws = wb.active
        __list = list()
        for row in ws.rows:
                __list.append([cell.value for cell in row if cell.value])

        __sentiment_np_time = [[element[0], element[1: -1], element[-1]]for element in __list]
        places = [[], [u'american'], [u'india'], [u'moscow'], [u'bombay'], [u'moscow'], [], [], [], [u'mumbai'], [u'mexico'], [u'mumbai'], [u'russian', u'moscow'], [u'bombay'], [u'nagaland'], [], [u'american'], [u'mumbai'], [], [], [], [u'mumbai'], [], [u'india'], [], [], [], [], [u'moscow'], [], [], [u'colaba'], [u'india'], [u'south delhi'], [u'india'], [], [u'mumbai'], [], [u'moscow'], [u'china'], [u'britain'], [u'mumbai'], [u'delhi'], [u'mumbai'], [u'india'], [u'noida'], [u'pakistan afghanistan'], [], [], [u'delhi'], [u'brooklyn'], [], [u'moscow'], [u'delhi'], [u'mumbai'], [u'moscow'], [u'moscow']]
        
        clustering_result = ProductionHeuristicClustering(sentiment_np_time = __sentiment_np_time,
                                                                sub_category = "dishes",
                                                                sentences = None,
                                                                eatery_name= "Hauz Khas Socials",
                                                                places = places,
                                                                eatery_address = u'9-A &amp; 12,Hauz Khas Village, New Delhi')
        

        clustering_result.run()

