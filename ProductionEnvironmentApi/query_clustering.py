#!/usr/bin/env python
#-*- coding: utf-8 -*-
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

parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir_path)

from Text_Processing.PosTaggers import PosTaggerDirPath, HunPosModelPath, HunPosTagPath
from Text_Processing.colored_print import bcolors




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

class QueryClustering:
        def __init__(self, noun_phrases, sub_category=None, sentences=None):
                """
            

                """
                self.list_to_exclude = ["food", "i", "service", "cost", "ambience", "delhi", \
                                "Delhi", "place", "Place", "india", "indian"]
                

                self.noun_phrases = noun_phrases
                self.sentences = sentences
                self.sub_category = sub_category
                new_list, self.clusters, self.result = list(), list(), list()
                

        def run(self):
                self.filter_clusters()

                #The noun phrases who were not at all in the self.clusters
                self.without_clusters =  set.symmetric_difference(set(range(0, len(self.noun_phrases))), \
                                                                    set(flatten(self.clusters)))
                self.populate_result()

                if self.sub_category == "dishes":
                        self.common_ners = list(set.intersection(set([e[0] for e in self.ner()]), \
                                                        set([e[0] for e in self.custom_ner()])))
                        self.result = self.filter_on_basis_pos_tag()
               
                print self.result
                return self.result

        def filter_clusters(self):
                """
                self.sent_sentiment_nps gave rise to merged_sent_sentiment_nps
                outputs:
                    self.clusters which will have list of lists 
                    with each list having index numbers of the elements who were found to be similar
                """


                X = np.zeros((len(self.noun_phrases), len(self.noun_phrases)), dtype=np.float)
                for i in xrange(0, len(self.noun_phrases)):
                        for j in xrange(0, len(self.noun_phrases)):
                                if i == j:
                                        #If calculating for same element
                                        X[i][j] = 0.5
                                        X[j][i] = 0.5
                                    
                                if X[i][j] == 0:
                                        #st = 'Levenshtein.ratio("{1}", "{0}")'.format(self.keys[i], self.keys[j])
                                        #ratio = eval(st)
                                        #ratio = SimilarityMatrices.levenshtein_ratio(self.keys[i], self.keys[j])
                                        ratio = SimilarityMatrices.modified_dice_cofficient(self.noun_phrases[i], self.noun_phrases[j])
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
                self.result = [self.noun_phrases[i] for i in list(self.without_clusters)]
                for cluster_list in self.clusters:
                        __maximum_length_np = max([self.noun_phrases[i] for i in cluster_list], key=len)
                        self.result.append(__maximum_length_np)
                return

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
                hunpos_tagger = HunposTagger(HunPosModelPath, HunPosTagPath)
                filtered_list = list()
                def check_np(np):
                        print "Actual NP %s\n"%np
                        try:
                                #__list = hunpos_tagger.tag(nltk.wordpunct_tokenize(np.encode("utf-8")))
                                __list = hunpos_tagger.tag(nltk.wordpunct_tokenize(np))
                                print __list
                        except Exception as e:
                                print "This bullsht string is being ignored %s"%np
                                return None
                        
                
                        if not set.intersection(set(["NNP", "NN", "NNS"]), set([__tag for (token, __tag) in __list])):     
                                return None
                


                        result = [__token for (__token, __tag) in __list if not __tag in ["RB", "CD", "FW"]]
                        print "Stripped off NP %s \n"%" ".join(result)
                        return " ".join(result)
                
                for __e in self.result:
                        filtered_list.append(check_np(__e))
                return filter(None, filtered_list)




if __name__ == "__main__":
        l = [u'cheddar cheese', u'Green Apple Mojito', u'Legends \u2019 Menu', u'portion size', u'chili sauce', u'8 / 10 Red Hot Chili Fried \u2013 Crispy fries', u'\u2018 Summer \u2019', u'\u2019 t', u'Apple Mojito \u2013', u'\u2019', u'salsa dip', u'dip']

        q = QueryClustering(noun_phrases=l, sub_category="dishes", sentences=[u'green Apple Mojito \u2013 The bar at hrc is huge and we decided to order the Green Apple Mojito from the \u2018Summer \u2019 s of the Legends \u2019 Menu .', u'the drink was made from green apples along with Mint .', u'8 / 10 Red Hot Chili Fried \u2013 Crispy fries with sweet and chili sauce .', u'the fries are topped with cheddar cheese .', u'the dish is served with a portion of tangy salsa dip and cheesy dip .', u'the cheesy dip was so - so and didn \u2019 t have any particular flavors .', u'the portion size of this dish was huge and a little more sweet and chili sauce would have been icing on the cake. 8 / 10']
                )
        q.run()
