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
this_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(this_file_path)
from GlobalConfigs import eateries, eateries_results_collection, reviews, reviews_results_collection
from Text_Processing import SentenceTokenizationOnRegexOnInterjections
from Text_Processing.PosTaggers import PosTaggerDirPath, HunPosModelPath, HunPosTagPath
from Text_Processing.colored_print import bcolors
from GlobalConfigs import DEBUG



parent = '/home/kaali2/Programs/Python/MadmachinesNLP01/MadMachinesNLP01/Text_Processing/PrepareClassifiers/InMemoryClassifiers'

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
        def __init__(self, eatery_id,  sentences=None, eatery_name=None, sub_category="dishes"):
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
                

                self.eatery_id = eatery_id
                result = self.return_food_sentences()
                self.sentiment_np_time = result[0]
                self.sentences = result[1]
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
                        self.result = sorted(self.result, reverse=True, key= lambda x: x.get("positive") \
                                                        + x.get("negative")+ x.get("neutral") + x.get("super-positive")+ x.get("super-negative"))
                        return self.result
                return self.result


        def return_food_sentences(self):
                from sklearn.externals import joblib
                sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
                reviews_list = list()
                for post in reviews.find({"eatery_id": self.eatery_id}):
                        reviews_list.extend([[sent, post.get("review_time")] for sent in sent_tokenizer.tokenize(post.get("review_text"))])
                
                tag_classifier = joblib.load("%s/svm_linear_kernel_classifier_tag.lib"%parent)

                tags = tag_classifier.predict([e[0] for e in reviews_list])
                food_sentences = list()
                for (sent, review_time),  tag in zip(reviews_list, tags):
                        if tag == "food":
                                food_sentences.append([sent, review_time])
   
                food_tag_classifier = joblib.load("%s/svm_linear_kernel_classifier_food_sub_tags.lib"%parent)
                sub_tags = food_tag_classifier.predict([e[0] for e in food_sentences])

                dishes_n_drinks = list()

                for (sent, review_time), sub_tag in zip(food_sentences, sub_tags):
                        if sub_tag == "dishes" or sub_tag == "drinks":
                                dishes_n_drinks.append([sent, review_time])
                        
    

                sentiment_classifier = joblib.load("%s/svm_linear_kernel_classifier_sentiment_new_dataset.lib"%parent)
                sentiments = sentiment_classifier.predict([e[0] for e in dishes_n_drinks])
    
                from topia.termextract import extract
                topia_extractor = extract.TermExtractor()
                noun_phrases = list()
                for (sent, review_time), tag in zip(dishes_n_drinks, sentiments):
                        nouns = topia_extractor(sent)
                        noun_phrases.append([tag, [e[0].lower() for e in nouns], review_time])
                        
                
                sentences = [sent for (sent, review_time), tag in zip(dishes_n_drinks, sentiments)]
                
                return (filter(lambda x: x[1], noun_phrases), sentences)



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
                print new_list
                print [(self.keys[a], self.keys[b]) for a, b in new_list]

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
                
                self.clusters = [list(set(element)) for element in self.clusters if len(element)> 2]
                """
                found = False
                new_clusters = list()
                #Removing duplicate elements from clusters list
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
                """
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
                        "super-negative": super_negative, "super-positive": super_positive, "similar": cluster_names,
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
                pos tagging of noun phrases will be done, and if the noun phrases contains some adjectives or RB or FW, 
                it will be removed from the total noun_phrases list

                Any Noun phrases when split, if present in self.list_to_exclude will not be included in the final result
                for Example: 
                self.list_to_exclude = ["food", "i", "service", "cost", "ambience", "delhi", "Delhi", "place", "Place"]
                noun_phrase = "great place"
                
                """
                print "{0} These noun phrases will be removed from the noun phrases {1}".format(bcolors.OKBLUE, bcolors.RESET)
                print "{0} List To Exclude {1}".format(bcolors.OKBLUE, bcolors.RESET)
                print self.list_to_exclude
                print "\n"
                print "{0} Common name entities  {1}".format(bcolors.OKBLUE, bcolors.RESET)
                print self.common_ners
                print "\n"
                hunpos_tagger = HunposTagger(HunPosModelPath, HunPosTagPath)
                filtered_list = list()
                for __e in self.result:
                        try:
                                __list = [pos_tag for (np, pos_tag) in hunpos_tagger.tag(nltk.wordpunct_tokenize(__e.get("name")))]
                                if bool(set.intersection(set(__e.get("name").split(" ")),  set(self.list_to_exclude))):
                                        print __e.get("name")
                                        pass    
                        
                        #elif __e.get("name") in self.common_ners:
                        #        pass
                        
                                elif "RB" == __list[0] or  "CD" in __list or "FW" in __list:
                                        pass
                                else:
                                        filtered_list.append(__e)

                        except Exception:
                                pass

                return filtered_list





if __name__ == "__main__":

        ins = ProductionHeuristicClustering("308322")
        ins.run()
        """

        New incoming (death wings, death wings)
        (94, 94)

        New incoming (death wings chicken, death wings chicken)
        (124, 124)

        New incoming (death wings chicken, chicken death wings)
        (124, 483)
        New incoming (death wings chicken, death chicken wings)
        (124, 762)
        New incoming (chicken wings n, chicken wings n)
        (211, 211)
        New incoming (chicken wings n, chicken wings i)
        (211, 214)
        New incoming (chicken wings n, bbq chicken wings)
        (211, 350)
        New incoming (chicken wings n, chicken wings)
        (211, 359)
        New incoming (chicken wings n, devil chicken wings)
        (211, 628)
        New incoming (chicken wings n, f *ck chicken wings)
        (211, 723)
        New incoming (chicken wings n, death chicken wings)
        (211, 762)
        New incoming (chicken wings i, chicken wings n)
        (214, 211)
        New incoming (chicken wings i, chicken wings i)
        (214, 214)
        New incoming (chicken wings i, bbq chicken wings)
        (214, 350)
        New incoming (chicken wings i, chicken wings)
        (214, 359)
        New incoming (chicken wings i, devil chicken wings)
        (214, 628)
        New incoming (chicken wings i, f *ck chicken wings)
        (214, 723)
        New incoming (chicken wings i, death chicken wings)
        (214, 762)


        (bbq chicken, bbq chicken)
        (305, 305)
        (bbq chicken, bbq chicken wings)
        (305, 350)
        (bbq chicken, chicken)
        (305, 565)


        New incoming (death wings chicken, death wings chicken)
        (124, 124)

        New incoming (death wings chicken, chicken death wings)
        (124, 483)
        cluster it joined [u'chicken death wings', u'death wings chicken']
        cluster it joined [124, 124, 124, 483]

        New incoming (death wings chicken, death chicken wings)
        (124, 762)
        cluster it joined [u'death chicken wings', u'chicken death wings', u'death wings chicken']
        cluster it joined [124, 124, 124, 483, 124, 762]

        New incoming (chicken wings n, chicken wings n)
        (211, 211)

        New incoming (chicken wings n, chicken wings i)
        (211, 214)
        cluster it joined [u'chicken wings n', u'chicken wings i']
        cluster it joined [211, 211, 211, 214]

        New incoming (chicken wings n, bbq chicken wings)
        (211, 350)
        cluster it joined [u'chicken wings n', u'bbq chicken wings', u'chicken wings i']
        cluster it joined [211, 211, 211, 214, 211, 350]

        New incoming (chicken wings n, chicken wings)
        (211, 359)
        cluster it joined [u'chicken wings n', u'bbq chicken wings', u'chicken wings i', u'chicken wings']
        cluster it joined [211, 211, 211, 214, 211, 350, 211, 359]

        New incoming (chicken wings n, devil chicken wings)
        (211, 628)
        cluster it joined [u'chicken wings n', u'devil chicken wings', u'bbq chicken wings', u'chicken wings i', u'chicken wings']
        cluster it joined [211, 211, 211, 214, 211, 350, 211, 359, 211, 628]


        New incoming (chicken wings n, f *ck chicken wings)
        (211, 723)
        cluster it joined [u'chicken wings n', u'chicken wings', u'f *ck chicken wings', u'devil chicken wings', u'chicken wings i', u'bbq chicken wings']
        cluster it joined [211, 211, 211, 214, 211, 350, 211, 359, 211, 628, 211, 723]

        New incoming (chicken wings n, death chicken wings)
        (211, 762)
        cluster it joined [u'death chicken wings', u'chicken death wings', u'death wings chicken', u'chicken wings n']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762]


        New incoming (bbq chicken wings, chicken wings n)
        (350, 211)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211]

        New incoming (bbq chicken wings, chicken wings i)
        (350, 214)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214]

        New incoming (bbq chicken wings, bbq chicken)
        (350, 305)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305]

        New incoming (bbq chicken wings, bbq chicken wings)
        (350, 350)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350]


        New incoming (bbq chicken wings, chicken wings)
        (350, 359)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359]

        New incoming (bbq chicken wings, f *ck chicken wings)
        (350, 723)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723]


        New incoming (chicken wings, chicken wings n)
        (359, 211)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211]


        New incoming (chicken wings, chicken wings i)
        (359, 214)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214]


        New incoming (chicken wings, bbq chicken wings)
        (359, 350)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350]

        New incoming (chicken wings, chicken wings)
        (359, 359)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359]

        New incoming (chicken wings, devil chicken wings)
        (359, 628)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628]

        New incoming (chicken wings, f *ck chicken wings)
        (359, 723)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723]


        New incoming (chicken wings, death chicken wings)
        (359, 762)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762]


        New incoming (chicken death wings, death wings chicken)
        (483, 124)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124]

        
        New incoming (chicken death wings, chicken death wings)
        (483, 483)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483]

        New incoming (chicken death wings, death chicken wings)
        (483, 762)
        cluster it joined [u'chicken death wings', u'chicken wings', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762]


        New incoming (chicken, • chicken)
        (565, 237)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237]

        New incoming (chicken, bbq chicken)
        (565, 305)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305]

        New incoming (chicken, chicken)
        (565, 565)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565]


        New incoming (chicken, chicken bao)
        (565, 693)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'bbq chicken wings', u'chicken bao']


        New incoming (chicken, chicken box)
        (565, 861)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']

        New incoming (devil chicken wings, chicken wings n)
        (628, 211)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211]

        New incoming (devil chicken wings, chicken wings i)
        (628, 214)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214]

        New incoming (devil chicken wings, chicken wings)
        (628, 359)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359]

        New incoming (devil chicken wings, devil chicken wings)
        (628, 628)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628]


        New incoming (chicken bao, chicken)
        (693, 565)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565]

        New incoming (chicken bao, chicken bao)
        (693, 693)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693]

        New incoming (chicken bao, chicken box)
        (693, 861)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861]



        New incoming (f *ck chicken wings, chicken wings n)
        (723, 211)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211]

        New incoming (f *ck chicken wings, chicken wings i)
        (723, 214)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214]

        New incoming (f *ck chicken wings, bbq chicken wings)
        (723, 350)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350]

        New incoming (f *ck chicken wings, chicken wings)
        (723, 359)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359]

        New incoming (f *ck chicken wings, f *ck chicken wings)
        (723, 723)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723]

        New incoming (death chicken wings, death wings chicken)
        (762, 124)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124]

        New incoming (death chicken wings, chicken wings n)
        (762, 211)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211]

        New incoming (death chicken wings, chicken wings i)
        (762, 214)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214]

        New incoming (death chicken wings, chicken wings)
        (762, 359)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214, 762, 359]

        New incoming (death chicken wings, chicken death wings)
        (762, 483)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214, 762, 359, 762, 483]


        New incoming (death chicken wings, death chicken wings)
        (762, 762)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214, 762, 359, 762, 483, 762, 762]


        New incoming (chicken box, chicken)
        (861, 565)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214, 762, 359, 762, 483, 762, 762, 861, 565]


        New incoming (chicken box, chicken bao)
        (861, 693)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214, 762, 359, 762, 483, 762, 762, 861, 565, 861, 693]


        New incoming (chicken box, chicken box)
        (861, 861)
        cluster it joined [u'chicken death wings', u'chicken wings', u'\u2022 chicken', u'f *ck chicken wings', u'bbq chicken', u'chicken wings n', u'devil chicken wings', u'chicken', u'chicken wings i', u'death chicken wings', u'death wings chicken', u'chicken box', u'bbq chicken wings', u'chicken bao']
        cluster it joined [124, 124, 124, 483, 124, 762, 211, 762, 214, 211, 214, 214, 214, 350, 214, 359, 214, 628, 214, 723, 214, 762, 305, 350, 305, 565, 350, 211, 350, 214, 350, 305, 350, 350, 350, 359, 350, 723, 359, 211, 359, 214, 359, 350, 359, 359, 359, 628, 359, 723, 359, 762, 483, 124, 483, 483, 483, 762, 565, 237, 565, 305, 565, 565, 565, 693, 565, 861, 628, 211, 628, 214, 628, 359, 628, 628, 693, 565, 693, 693, 693, 861, 723, 211, 723, 214, 723, 350, 723, 359, 723, 723, 762, 124, 762, 211, 762, 214, 762, 359, 762, 483, 762, 762, 861, 565, 861, 693, 861, 861]


        


        New incoming (wada pao, wada pao)
        (335, 335)
        New incoming (wada pao, vada pao)
        (335, 698)
        cluster it joined [u'vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698]

        New incoming (veg vada pao, veg vada pao)
        (614, 614)

        New incoming (veg vada pao, vada pao)
        (614, 698)
        cluster it joined [u'vada pao', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698]
    
        
        New incoming (vada pav, vada pav)
        (685, 685)

        New incoming (vada pav, vada pao)
        (685, 698)
        cluster it joined [u'vada pao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698]
        
        New incoming (vada pao, wada pao)
        (698, 335)
        cluster it joined [u'vada pao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335]

        New incoming (vada pao, veg vada pao)
        (698, 614)
        cluster it joined [u'vada pao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335, 698, 614]

        New incoming (vada pao, vada pav)
        (698, 685)
        cluster it joined [u'vada pao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335, 698, 614, 698, 685]

        New incoming (vada pao, vada pao)
        (698, 698)
        cluster it joined [u'vada pao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335, 698, 614, 698, 685, 698, 698]

        New incoming (vada pao, vada pao bao)
        (698, 703)
        cluster it joined [u'vada pao', u'vada pao bao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335, 698, 614, 698, 685, 698, 698, 698, 703]
        
        New incoming (vada pao bao, vada pao)
        (703, 698)
        cluster it joined [u'vada pao', u'vada pao bao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335, 698, 614, 698, 685, 698, 698, 698, 703, 703, 698]

        New incoming (vada pao bao, vada pao bao)
        (703, 703)
        cluster it joined [u'vada pao', u'vada pao bao', u'vada pav', u'veg vada pao', u'wada pao']
        cluster it joined [335, 335, 335, 698, 614, 698, 685, 698, 698, 335, 698, 614, 698, 685, 698, 698, 698, 703, 703, 698, 703, 703]


        New incoming (island iced tea, island iced tea)
        (388, 388)
        New incoming (island iced tea, island tea)
        (388, 449)
        cluster it joined [u'island tea', u'island iced tea']
        cluster it joined [388, 388, 388, 449]

        (388, 650)
        cluster it joined [u'island tea', u'island teas', u'island iced tea']
        cluster it joined [388, 388, 388, 449, 388, 650]

        New incoming (island iced tea, island ice tea)
        (388, 651)
        cluster it joined [u'island tea', u'island teas', u'island ice tea', u'island iced tea']
        cluster it joined [388, 388, 388, 449, 388, 650, 388, 651]

        New incoming (island iced tea, long island iced tea)
        (388, 1026)
        cluster it joined [u'long island iced tea', u'island iced tea', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026]


        New incoming (island tea, island iced tea)
        (449, 388)
        cluster it joined [u'island tea', u'long island iced tea', u'island iced tea', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388]
        New incoming (island tea, island tea)
        (449, 449)
        cluster it joined [u'island tea', u'long island iced tea', u'island iced tea', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449]

        New incoming (island tea, island teas)
        (449, 650)
        cluster it joined [u'island tea', u'long island iced tea', u'island iced tea', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650]


        New incoming (island ice tea lovers, island ice tea lovers)
        (613, 613)

        New incoming (island ice tea lovers, island ice tea)
        (613, 651)
        cluster it joined [u'island tea', u'island teas', u'island ice tea', u'island iced tea', u'island ice tea lovers']
        cluster it joined [388, 388, 388, 449, 388, 650, 388, 651, 613, 651]

    New incoming (island teas, island iced tea)
    (650, 388)
    cluster it joined [u'island tea', u'long island iced tea', u'island iced tea', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388]

    New incoming (island teas, island tea)
    (650, 449)
    cluster it joined [u'island tea', u'long island iced tea', u'island iced tea', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449]

    New incoming (island teas, island teas)
    (650, 650)
    cluster it joined [u'island tea', u'long island iced tea', u'island iced tea', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650]




    New incoming (island ice tea, island iced tea)
    (651, 388)
    cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388]

    New incoming (island ice tea, island ice tea lovers)
    (651, 613)
    cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613]

    New incoming (island ice tea, island ice tea)
    (651, 651)
    cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas']

    New incoming (island ice tea, island ice tea tops)
    (651, 773)
    cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773]

    New incoming (island ice tea, long island ice tea)
    (651, 908)
    cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'long island ice tea', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
    cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773, 651, 908]

New incoming (long island ice tea, island ice tea)
(908, 651)
cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'long island ice tea', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773, 651, 908, 773, 651, 773, 773, 908, 651]

        New incoming (long island ice tea, long island ice tea)
        (908, 908)
        cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'long island ice tea', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773, 651, 908, 773, 651, 773, 773, 908, 651, 908, 908]

        New incoming (long island ice tea, longest island ice teas)
        (908, 953)
        cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'long island ice tea', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773, 651, 908, 773, 651, 773, 773, 908, 651, 908, 908, 908, 953]

        New incoming (long island ice tea, long long island iced teas)
        (908, 971)
        cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'long island ice tea', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773, 651, 908, 773, 651, 773, 773, 908, 651, 908, 908, 908, 953, 908, 971]

        New incoming (long island ice tea, long island iced tea)
        (908, 1026)
        cluster it joined [u'island tea', u'long island iced tea', u'island ice tea', u'island iced tea', u'island ice tea lovers', u'island teas', u'long long island iced teas', u'long island ice tea', u'longest long island iced teas', u'longest long island iced tea', u'longest island ice teas', u'island ice tea tops']
        cluster it joined [113, 113, 113, 953, 113, 971, 113, 1015, 113, 1026, 388, 1026, 449, 388, 449, 449, 449, 650, 650, 388, 650, 449, 650, 650, 651, 388, 651, 613, 651, 651, 651, 773, 651, 908, 773, 651, 773, 773, 908, 651, 908, 908, 908, 953, 908, 971, 908, 1026]


        New incoming (parle g biscuits, parle g biscuits)
        (701, 701)
        """
