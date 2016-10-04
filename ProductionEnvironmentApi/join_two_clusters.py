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

class ProductionJoinClusters:
        def __init__(self, nps_dict_list):
                """ 
                Args:
                    sentiment_nps:
                        [[u'positive',[u'paneer chilli pepper starter']], [u'positive', []],
                         [u'positive', [u'friday night']], [u'positive', []],                                   
                         [u'super-positive', [u'garlic flavours', u'penne alfredo pasta']]],
                """

                self.nps_dict_list = nps_dict_list
                new_list, self.clusters, self.result = list(), list(), list()
                

        def run(self):
                self.merged_sentiment_nps = self.merge_similar_elements()
                for key, value in self.merged_sentiment_nps.iteritems():
                    if not value.get("total_sentiments"):
                            print key, value , "\n\n"
                __sorted = sorted(self.merged_sentiment_nps.keys())
                #self.NERs = self.ner()
                self.keys = self.merged_sentiment_nps.keys()
                self.filter_clusters()
                #The noun phrases who were not at all in the self.clusters
                self.without_clusters =  set.symmetric_difference(set(range(0, len(self.keys))), \
                                                                    set(flatten(self.clusters)))
                self.populate_result()
                result = sorted(self.result, reverse= True, key=lambda x: x.get("positive")+x.get("negative") + x.get("neutral")+
                                                                    x.get("super-positive")+ x.get("super-negative"))
                return result



        def merge_similar_elements(self):
                """
                Result:
                    Merging noun phrases who have exact similar spellings with each other and return a
                    dictionary in the form
                        {'name': u'chicken pieces', 'positive': 0, 'timeline': [(u'neutral', u'2014-06-20 15:11:42')],
                        'negative': 0, 'super-positive': 0, 'neutral': 1, 'super-negative': 0, 'similar': []}]
                }
                """

                without_similar_elements = dict()
                for np_dict in self.nps_dict_list:
                        np_name = np_dict.get("name")
                        np_dict.pop("name")


                        if without_similar_elements.get(np_name):
                                old = without_similar_elements.get(np_name)
                                old_timeline  =  old.pop("timeline")
                                new_timeline = np_dict.pop("timeline")

                                new_np_dict = dict()
                                for key in ['positive', 'negative', 'neutral', 'super-positive', 'super-negative', "total_sentiments"]:
                                        __old_frequency, __new_frequency = old.get(key), np_dict.get(key)
                                        new_np_dict.update({key: __old_frequency+__new_frequency})

                                timeline = sorted((old_timeline+new_timeline), key=lambda x: x[1])

                                new_np_dict.update({"timeline": timeline})
                                without_similar_elements.update(
                                            {np_name: new_np_dict})


                        else:
                                without_similar_elements.update({np_name: np_dict })
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
                test_new_list = list()

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
                #Removing duplicate elements from clusters list
                self.clusters = [list(set(element)) for element in self.clusters if len(element)> 2]
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
                total_sentiments, positive, negative, neutral, super_positive, super_negative = int(), int(), int(), int(), int(), int()
                timeline = list()
               
                cluster_names = [self.keys[element] for element in cluster_list]

                for element in cluster_list:
                        name = self.keys[element]    
                        new_dict = self.merged_sentiment_nps[name]
                        new_dict.update({"name": name})
                        result.append(new_dict) 
                        total_sentiments = total_sentiments +  self.merged_sentiment_nps[name].get("total_sentiments") 
                        positive = positive +  self.merged_sentiment_nps[name].get("positive") 
                        negative = negative +  self.merged_sentiment_nps[name].get("negative") 
                        neutral = neutral +  self.merged_sentiment_nps[name].get("neutral") 
                        super_negative = super_negative +  self.merged_sentiment_nps[name].get("super-negative") 
                        super_positive = super_positive +  self.merged_sentiment_nps[name].get("super-positive") 
                        timeline.extend(self.merged_sentiment_nps[name].get("timeline"))
            

                result = sorted(result, reverse= True, key=lambda x: x.get("positive")+x.get("negative") + x.get("neutral")+
                                                                    x.get("super-positive")+ x.get("super-negative"))
                return {"name": result[0].get("name"), "positive": positive, "negative": negative, "neutral": neutral, 
                        "super-negative": super_negative, "super-positive": super_positive, "similar": cluster_names,
                        "timeline": timeline, "total_sentiments": total_sentiments}

        #@print_execution
        def custom_ner(self):
                ner = list()
                regexp_grammer = r"NER:{<IN><NN.*><NN.*>?}"
                __parser = nltk.RegexpParser(regexp_grammer)

                hunpos_tagger = HunposTagger(HunPosModelPath, HunPosTagPath)
                for __sentence in self.sentences:
                        try:
                                tagged = hunpos_tagger.tag(nltk.word_tokenize(__sentence.encode("utf-8")))
                        except Exception as e:    
                                hunpos_tagger = HunposTagger(HunPosModelPath, HunPosTagPath)
                                tagged = hunpos_tagger.tag(nltk.word_tokenize(__sentence.encode("utf-8")))
                        tree = __parser.parse(tagged)
                        for subtree in tree.subtrees(filter = lambda t: t.label()=='NER'):
                                l = " ".join([e[0] for e in subtree.leaves() if e[1] == 'NNP' or e[1] == 'NNS' or e[1] == 'NN'])
                                ner.append(l.lower())

                result = sorted(Counter(ner).items(), reverse=True, key=lambda x: x[1])
                return result





