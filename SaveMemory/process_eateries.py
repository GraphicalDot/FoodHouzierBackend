#!/usr/bin/env pypy
#-*- coding: utf-8 -*-
"""
Author: kaali
Dated: 15 April, 2015
Purpose:
    This file has been written to list all the sub routines that might be helpful in generating result for 
    get_word_cloud api


main_categories = u'cuisine', u'service', u'food', u'menu', u'overall', u'cost', u'place', u'ambience', u'null'])
food_sub_category = {u'dishes', u'null-food', u'overall-food'}


"""
import sys
import time
import os
from sys import path
import itertools
import warnings
import ConfigParser
from sklearn.externals import joblib
from collections import Counter
from db_scripts import MongoScriptsReviews, MongoScriptsDoClusters
from nltk.stem import SnowballStemmer
from sklearn.externals import joblib 
from prod_heuristic_clustering import ProductionHeuristicClustering

#from join_two_clusters import ProductionJoinClusters


this_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.dirname(this_file_path)
print parent_dir_path

sys.path.append(parent_dir_path)
from PreProcessingText import PreProcessText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections


from configs import cd, SolveEncoding, corenlpserver
from configs import SentimentClassifiersPath, TagClassifiersPath,\
                    FoodClassifiersPath, ServiceClassifiersPath,\
                    AmbienceClassifiersPath, CostClassifiersPath 


from configs import SentimentVocabularyFileName, SentimentFeatureFileName,\
                SentimentClassifierFileName

from configs import TagVocabularyFileName, TagFeatureFileName,\
                TagClassifierFileName

from configs import FoodVocabularyFileName, FoodFeatureFileName,\
                FoodClassifierFileName

from configs import ServiceVocabularyFileName, ServiceFeatureFileName,\
                    ServiceClassifierFileName

from configs import CostVocabularyFileName, CostFeatureFileName,\
                    CostClassifierFileName

from configs import AmbienceVocabularyFileName, AmbienceFeatureFileName,\
                    AmbienceClassifierFileName

from configs import eateries, reviews, r_reviews, r_eateries
import blessings
Terminal = blessings.Terminal()

from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

#from elasticsearch_db import ElasticSearchScripts
from topia.termextract import extract  
from simplejson import loads
#from google_places import google_already_present, find_google_places
import time





def prediction(sentences, vocabulary, features, classifier):
        loaded_vectorizer= CountVectorizer(vocabulary=vocabulary)
        sentences_counts = loaded_vectorizer.transform(sentences)
        reduced_features = features.transform(sentences_counts.toarray())
        predictions = classifier.predict(reduced_features)

        return predictions
 

def parallel_prediction(sentence, vocabulary, features, classifier):
        ##No use of this as trained model lready uses joblib parallel
        ##and you can not user nested paralle processing in python
        ##need to shift to celery, :(
        loaded_vectorizer= CountVectorizer(vocabulary=vocabulary)
        sentences_counts = loaded_vectorizer.transform(sentence)
        reduced_features = features.transform(sentences_counts.toarray())
        prediction = classifier.predict(reduced_features)

        return prediction
 

class SentimentClassification(object):
        """
        cant do per eatery because then, All the classifiers must be loaded
        simultaneously, which again puts too much pressure on the memory of the
        server, So its better to load as much as reviews in the memory as
        possible, I guess even of you load 10 million reviews in 240Gb ram,
        there would still be tons of memory left.
        """
        def __init__(self, sentences, path):
                with cd(SentimentClassifiersPath(path)):
                        self.features =  joblib.load(SentimentFeatureFileName)
                        self.vocabulary = joblib.load(SentimentVocabularyFileName)
                        self.classifier= joblib.load(SentimentClassifierFileName)
                self.sentences = sentences
                print Terminal.green("SentimentClassification loaded")
        def run(self):

                print "Doing sequesntial prediction"
                start = time.time()
                sentiments = prediction(self.sentences, self.vocabulary, self.features, self.classifier)
                print "Time took by sequential prediction is %s"%(time.time()- start)
                return sentiments

class TagClassification(object):
        """
        makes sure runs after SentimentClassification
        The input to this class will be in the form 
        return zip(eatery_ids, review_ids, sentences, review_time, sentiments)
        [(eatery_id, review_id, sentences, review_time, sentiment), ...]
        """
        def __init__(self, sentences, path):
                with cd(TagClassifiersPath(path)):
                        self.features =  joblib.load(TagFeatureFileName)
                        self.vocabulary = joblib.load(TagVocabularyFileName)
                        self.classifier = joblib.load(TagClassifierFileName)
                self.sentences = sentences
                print Terminal.green("TagClassification loaded")

        def run(self):
                result = prediction(self.sentences, self.vocabulary, self.features, self.classifier)
                return result
                


class ServiceClassification(object):
        def __init__(self, sentences, path):        
                with cd(ServiceClassifiersPath(path)):
                        self.features =  joblib.load(ServiceFeatureFileName)
                        self.vocabulary = joblib.load(ServiceVocabularyFileName)
                        self.classifier= joblib.load(ServiceClassifierFileName)
                self.sentences = sentences 
                print Terminal.green("ServiceClassification loaded")

        def run(self):
                result = prediction(self.sentences, self.vocabulary, self.features, self.classifier)
                return result
                
            
class FoodClassification(object):
        def __init__(self, sentences, path):        
            
                with cd(FoodClassifiersPath(path)):
                        self.features =  joblib.load(FoodFeatureFileName)
                        self.vocabulary = joblib.load(FoodVocabularyFileName)
                        self.classifier= joblib.load(FoodClassifierFileName)
                self.sentences = sentences
                print Terminal.green("FoodClassification loaded")

        def run(self):
                result = prediction(self.sentences, self.vocabulary, self.features, self.classifier)
                return result
class CostClassification(object):
        def __init__(self, sentences, path):        
                with cd(CostClassifiersPath(path)):
                        self.features =  joblib.load(CostFeatureFileName)
                        self.vocabulary = joblib.load(CostVocabularyFileName)
                        self.classifier= joblib.load(CostClassifierFileName)
                self.sentences = sentences
                print Terminal.green("CostClassification loaded")

        def run(self):
                result = prediction(self.sentences, self.vocabulary, self.features, self.classifier)
                return result

class AmbienceClassification(object):
        def __init__(self, sentences, path):        
                with cd(AmbienceClassifiersPath(path)):
                        self.features =  joblib.load(AmbienceFeatureFileName)
                        self.vocabulary = joblib.load(AmbienceVocabularyFileName)
                        self.classifier= joblib.load(AmbienceClassifierFileName)
                self.sentences = sentences
                print Terminal.green("AmbienceClassification loaded")

        def run(self):
                result = prediction(self.sentences, self.vocabulary, self.features, self.classifier)
                return result







class ClassifyReviews(object):
        def __init__(self, eatery_id_list, path_compiled_classifiers):
                self.np_extractor = extract.TermExtractor() 
                self.eatery_id_list = eatery_id_list #list of ids 
                
                self.review_list = list()
                for eatery_id in self.eatery_id_list:
                        warnings.warn("Fushing whole atery") 
                        MongoScriptsReviews.flush_eatery(eatery_id)
                        MongoScriptsReviews.insert_eatery_into_results_collection(eatery_id)
                        self.review_list.extend([(eatery_id, post.get("review_id"), post.get("review_text"),
                                post.get("review_time")) for post in
                                reviews.find({"eatery_id": eatery_id})])

                print "len of all reviews %s"%len(self.review_list)
                ##review_list at this point will be in the form 
                ##[(review_id, review_text, review_time), ()]

                ##now tokenizing all the reviews 
                self.tokenized_reviews = list() 
                for (eatery_id, review_id, review_text, review_time) in self.review_list:
                            sentences = self.sent_tokenize_review(review_text)
                            self.tokenized_reviews.extend(map(lambda sent: (eatery_id, review_id, sent,
                                review_time), sentences))

                print "len of all sentences %s"%len(self.tokenized_reviews)
                self.path_compiled_classifiers = path_compiled_classifiers
                ##this will return a list of quadruple with each element will
                ##a tokenized sentences form the review whose id is the second
                ##element, tokenized reviews will in the form 
                ##(eatery_id, review_id, sentence, review_time)

        def run(self):
                eatery_ids, review_ids, sentences, review_time = zip(*self.tokenized_reviews)
                
                ##prediction of sentiments 
                sentiment_instance = SentimentClassification(sentences,
                        self.path_compiled_classifiers)
                sentiments = sentiment_instance.run()
                del sentiment_instance
                print Terminal.green("Sentiment Instance Deleted")
                time.sleep(10)

                ##predcting tags
                tag_instance = TagClassification(sentences)
                tags = tag_instance.run()
                del tag_instance
                print Terminal.green("Tag Instance Deleted")
                time.sleep(10)
                
                
                ##combing_whole_list
                ##will be in the form [(eatery_id, review_id, sentence,
                ##review_time, sentiment, tag)]
                sentiment_tag_list = zip(eatery_ids, review_ids, sentences,
                        review_time, sentiments, tags)

                ##now all the lists for food, ambience, service etc has been
                #generated
                self.filter_on_tag(sentiment_tag_list)

                ##going for food sub tag classification 
                f_eatery_ids, f_reviews_ids, f_sentences, f_review_time, f_sentiment, \
                        f_tags = zip(*self.food) 
                f_instance = FoodClassification(f_sentences,
                        self.path_compiled_classifiers)
                f_predictions = f_instance.run()
                del f_instance
                print Terminal.green("Food Instance Deleted")
                time.sleep(10)

                ##get all the noun phrases for the food sentences 
                self.nouns = self.extract_noun_phrases(f_sentences)
                self.all_food = zip(f_reviews_ids, f_sentences, f_tags, f_sentiment,
                        f_predictions, self.nouns, f_review_time)



                ##going for ambience sub tag classification 
                a_eatery_ids, a_reviews_ids, a_sentences, a_review_time, a_sentiment, \
                        a_tags = zip(*self.ambience) 
                a_instance = AmbienceClassification(a_sentences,
                        self.path_compiled_classifiers)
                a_predictions = a_instance.run()
                del a_instance
                self.all_ambience = zip(a_reviews_ids, a_sentences, a_tags,
                        a_sentiment, a_predictions, a_review_time)
                print Terminal.green("Ambience Instance Deleted")
                time.sleep(10)


                ##going for cost sub tag classification 
                c_eatery_ids, c_reviews_ids, c_sentences, c_review_time, c_sentiment, \
                        c_tags = zip(*self.cost) 
                c_instance = CostClassification(c_sentences,
                        self.path_compiled_classifiers)
                c_predictions = c_instance.run()
                del c_instance
                self.all_cost = zip(c_reviews_ids, c_sentences, c_tags,
                        c_sentiment, c_predictions, c_review_time)
                print Terminal.green("Cost Instance Deleted")
                time.sleep(10)

                ##going for service sub tag classification 
                s_eatery_ids, s_reviews_ids, s_sentences, s_review_time, s_sentiment, \
                        s_tags = zip(*self.service) 
                s_instance = ServiceClassification(c_sentences,
                        self.path_compiled_classifiers)
                s_predictions = s_instance.run()
                del s_instance
                self.all_service = zip(s_reviews_ids, s_sentences, s_tags,
                        s_sentiment, s_predictions, s_review_time)
                print Terminal.green("Service Instance Deleted")
                time.sleep(10)


                        

                self.mongo_operations()
        def mongo_operations(self):
                #First of all the reviews for all the eateries must be present
                ##the result database 
                print Terminal.green("Updating reviews")
                for (eatery_id, review_id, review_text, review_time) in\
                        self.review_list:
                        r_reviews.update({"review_id": review_id}, {"$set":
                            {"eatery_id": eatery_id, "review_text":
                                review_text, "review_time": review_time}}, upsert=True, multi=False)

                ##storing all food sentences
                print Terminal.green("Updating food sentences")
                for (review_id, sentence, tag, sentiment, sub_tag, noun,
                        review_time) in self.all_food:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"food_result": [sentence, tag,
                                    sentiment, sub_tag, noun, review_time]}},
                                upsert=False)
                
                print Terminal.green("Updating cost sentences")
                for (review_id, sentence, tag, sentiment, sub_tag, \
                        review_time) in self.all_cost:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"cost_result": [sentence, tag,
                                    sentiment, sub_tag, review_time]}},
                                upsert=False)
                
                print Terminal.green("Updating service sentences")
                for (review_id, sentence, tag, sentiment, sub_tag, \
                        review_time) in self.all_service:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"service_result": [sentence, tag,
                                    sentiment, sub_tag, review_time]}},
                                upsert=False)
                
                print Terminal.green("Updating ambience sentences")
                for (review_id, sentence, tag, sentiment, sub_tag, \
                        review_time) in self.all_ambience:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"ambience_result": [sentence, tag,
                                    sentiment, sub_tag, review_time]}},
                                upsert=False)
                        
                print Terminal.green("Updating overall sentences")
                for (eatery_id, review_time, sentence, review_time, sentiment,\
                        tag) in self.overall:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"overall": [sentence, tag,
                                    sentiment, review_time]}},
                                upsert=False)
                
                print Terminal.green("Updating place sentences")
                for (eatery_id, review_time, sentence, review_time, sentiment,\
                        tag) in self.place:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"place_sentences": [sentence, tag,
                                    sentiment]}},
                                upsert=False)

                for (eatery_id, review_time, sentence, review_time, sentiment,\
                        tag) in self.cuisine:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"cuisine_sentences": [sentence, tag,
                                    sentiment]}},
                                upsert=False)

                for (eatery_id, review_time, sentence, review_time, sentiment,\
                        tag) in self.menu:
                        r_reviews.update({"review_id": review_id},\
                                {"$push": {"menu_result": [sentence, tag,
                                    sentiment, review_time]}},
                                upsert=False)
    
                ##Finding places and storing in mongodb 
                ##TODO
                ##to save time for semo place_name skipped
                for (eatery_id, review_time, sentence, review_time, sentiment,\
                        tag) in self.place:
                        #place_name = self.extract_places(sentence)
                        place_name = []
                        if bool(place_name):
                                r_reviews.update({"review_id": review_id},\
                                {"$push": {"places_result": place_name}},
                                upsert=False)
                        
                ##Finding cuisine and storing in mongodb 
                for (eatery_id, review_time, sentence, review_time, sentiment,\
                        tag) in self.cuisine:
                        cuisine_name = self.extract_cuisines(sentence)
                        if bool(cuisine_name):
                                r_reviews.update({"review_id": review_id},\
                                {"$push": {"cuisine_result": cuisine_name}},
                                upsert=False)
                        
                        
                return 


        def filter_on_tag(self, _list): 
                """
                all lists will be a list of tuoles in the form 
                ##will be in the form [(eatery_id, review_id, sentence,
                ##review_time, sentiment, tag)]
                """
                self.food = filter(lambda _object: _object[5] == "food", _list)
                self.cost = filter(lambda _object: _object[5] == "cost", _list)
                self.ambience = filter(lambda _object: _object[5] == "ambience", _list)
                self.service = filter(lambda _object: _object[5] == "service", _list)
                self.overall = filter(lambda _object: _object[5] == "overall", _list)
                self.place = filter(lambda _object: _object[5] == "place", _list)
                self.cuisine = filter(lambda _object: _object[5] == "cuisine", _list)
                self.menu = filter(lambda _object: _object[5] == "menu", _list)
                return     
            
        def extract_noun_phrases(self, sentences):
                __nouns = list()
                for sent  in sentences:
                            __nouns.append([e[0] for e in self.np_extractor(sent)])

                return __nouns  

        def sent_tokenize_review(self, review_text):
                """
                Tokenize self.reviews tuples of the form (review_id, review) to sentences of the form (review_id, sentence)
                and generates two lists self.review_ids and self.sentences
                """
                
                text = PreProcessText.process(review_text)
                self.sentences = tokenizer.tokenize(text)
                ##now breking sentences on the basis of but
                new_sentences = list()
                for sentence in self.sentences:
                     new_sentences.extend(sentence.split("but"))
                return filter(None, new_sentences)
        
        def extract_places(self, sentence):
                """
                This function filters all the places mentioned in self.places variable
                it generates a list of places mentioned in the self.places wth the help
                of stanford core nlp
                """
                def filter_places(__list):
                        location_list = list()
                        i = 0
                        for __tuple in __list:
                                if __tuple[1] == "LOCATION":
                                        location_list.append([__tuple[0], i])
                                i += 1


                        i = 0
                        try:
                                new_location_list = list()
                                [first_element, i] = location_list.pop(0)
                                new_location_list.append([first_element])
                                for element in location_list:
                                        if i == element[1] -1:
                                                new_location_list[-1].append(element[0])
                                            
                                        else:
                                                new_location_list.append([element[0]])
                                        i = element[1]

                                return list(set([" ".join(element) for element in new_location_list]))
                        except Exception as e:
                                return None


                try:
                            result = loads(corenlpserver.parse(sentence))
                            __result = [(e[0], e[1].get("NamedEntityTag")) for e in result["sentences"][0]["words"]]
                            place_name = filter_places(__result)
                            print "%s in %s"%(place_name, sentence)
                            
                except Exception as e:
                            print e, "__extract_place", sentence
                            place_name = []
                            
                return place_name 
        
        def extract_cuisines(self, sentence):
                """
                This extracts the name of the cuisines fromt he cuisines sentences
                """

		
                cuisine_name = self.np_extractor(sentence)
                result = [np[0] for np in cuisine_name if np[0]]
                print "%s in %s"%(result, sentence)
                return result
                       
        def __update_cuisine_places(self):
                """
                update cuisine and places to the eatery
                """
                MongoScriptsReviews.update_eatery_places_cusines(self.eatery_id, self.places_names, self.cuisine_name)        
                return 
                

class DoClusters(object):
        """
        'eatery_url''eatery_coordinates''eatery_area_or_city''eatery_address'

        Does heuristic clustering for the all reviews
        """
        def __init__(self, eatery_id, category=None, with_sentences=False):
                self.eatery_id = eatery_id
                self.mongo_instance = MongoScriptsDoClusters(self.eatery_id)
                self.eatery_name = self.mongo_instance.eatery_name
                self.category = category
                self.sentiment_tags = ["mixed", "negative", "positive", "neutral"]
                self.food_tags = ["dishes", "null-food", "overall-food"]
                self.ambience_tags = [u'smoking-zone', u'decor', u'ambience-null',\
                 u'ambience-overall', u'in-seating', u'crowd', u'open-area',\
                 u'sports', u'dancefloor', u'music', u'location',  \
                 u'bowling', u'view', u'live-matches', u'romantic']
                self.cost_tags = ["vfm", "expensive", "cheap", "not worth", "cost-null"]
                self.service_tags = [u'management', u'service-charges', u'service-overall', u'service-null', u'waiting-hours', u'presentation', u'booking', u'staff']


        def run(self):
                """
                main_categories = u'cuisine', u'service', u'food', u'menu', u'overall', u'cost', u'place', u'ambience', u'null'])
                food_sub_category = {u'dishes', u'null-food', u'overall-food'}
                Two cases:
                    Case 1:Wither first time the clustering is being run, The old_considered_ids list
                    is empty
                        case food:

                        case ["ambience", "cost", "service"]:
                            instance.fetch_reviews(__category) fetch a list of this kind
                            let say for ambience [["positive", "ambience-null"], ["negative", "decor"], 
                            ["positive", "decor"]...]
                            
                            DoClusters.make_cluster returns a a result of the form 
                                [{"name": "ambience-null", "positive": 1, "negative": 2}, ...]

                            After completing it for all categories updates the old_considered_ids of 
                            the eatery with the reviews

                    Case 2: None of processed_reviews and old_considered_ids are empty
                        calculates the intersection of processed_reviews and old_considered_ids
                        which implies that some review_ids in old_considered_ids are not being 
                        considered for total noun phrases
                        
                        Then fetches the review_ids of reviews who are not being considered for total
                        noun phrases 

                        instance.fetch_reviews(__categoryi, review_ids) fetch a list of this kind
                            let say for ambience [["positive", "ambience-null"], ["negative", "decor"], 
                            ["positive", "decor"]...]

                """

                if self.mongo_instance.if_no_reviews_till_date() == 0:
                        ##This implies that this eatery has no reviews present in the database
                        print "No reviews are present for the eatery_id in reviews colllection\
                                = <<{0}>>".format(self.eatery_id)
                        return 

                #That clustering is running for the first time
                warnings.warn("No clustering of noun phrases has been done yet  for eatery_id\
                                = <<{0}>>".format(self.eatery_id))
                       
                ##update eatery with all the details in eatery reslut collection

                __nps_food = self.mongo_instance.fetch_reviews("food", review_list=None)
                        
                ##sub_tag_dict = {u'dishes': [[u'super-positive', 'sent', [u'paneer chilli pepper starter']],
                ##[u'positive', sent, []],
                ##u'menu-food': [[u'positive', sent, []]], u'null-food': [[u'negative', sent, []],
                ##[u'negative', sent, []],
                sub_tag_dict = self.unmingle_food_sub(__nps_food)


                __result = self.clustering(sub_tag_dict.get("dishes"), "dishes")
                        
                ##this now returns three keys ina dictionary, nps, excluded_nps and dropped nps
                self.mongo_instance.update_food_sub_nps(__result, "dishes")
                        
                #REsult corresponding to the menu-food tag
                __result = self.aggregation(sub_tag_dict.get("overall-food"))
                self.mongo_instance.update_food_sub_nps(__result, "overall-food")
                        


                for __category in ["ambience_result", "service_result", "cost_result"]:
                        __nps = self.mongo_instance.fetch_reviews(__category)
                        __whle_nps = self.make_cluster(__nps, __category)
                                
                        self.mongo_instance.update_nps(__category.replace("_result", ""), __whle_nps)
                        
                        
                __nps = self.mongo_instance.fetch_reviews("overall")
                overall_result = self.__overall(__nps)
                self.mongo_instance.update_nps("overall", overall_result)
                        
                __nps = self.mongo_instance.fetch_reviews("menu_result")
                overall_result = self.__overall(__nps)
                self.mongo_instance.update_nps("menu", overall_result)
                
                """
                ##NOw all the reviews has been classified, tokenized, in short processed, 
                ##time to populate elastic search
                ##instance = NormalizingFactor(self.eatery_id)
                ##instance.run()
                ##ElasticSearchScripts.insert_eatery(self.eatery_id)

                """
                
                return 


        def join_two_clusters(self, __list, sub_category):
                clustering_result = ProductionJoinClusters(__list)
                return clustering_result.run()
                    


        def clustering(self, __sent_sentiment_nps_list, sub_category):
                """
                Args __sent_sentiment_nps_list:
                        [
                        (u'positive', sentence, [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                        (u'positive', sentence, [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                        (u'positive', sentence, [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                         (u'neutral', sentence, [u'chicken pieces', u'veg pasta n'], u'2014-06-20 15:11:42')]  

                Result:
                    [
                    {'name': u'paneer chilli pepper starter', 'positive': 3, 'timeline': 
                    [(u'positive', u'2014-09-19 06:56:42'), (u'positive', u'2014-09-19 06:56:42'), 
                    (u'positive', u'2014-09-19 06:56:42')], 'negative': 0, 'super-positive': 0, 'neutral': 0, 
                    'super-negative': 0, 'similar': []}, 
                    
                    {'name': u'chicken pieces', 'positive': 0, 'timeline': 
                    [(u'neutral', u'2014-06-20 15:11:42')], 'negative': 0, 'super-positive': 0, 
                    'neutral': 1, 'super-negative': 0, 'similar': []}
                    ]
                """
                if not bool(__sent_sentiment_nps_list):
                        return list()

                ##Removing (sentiment, nps) with empyt noun phrases
                __sentiment_np_time = [(sentiment, nps, review_time) for (sentiment, sent, nps, review_time) \
                        in __sent_sentiment_nps_list if nps]
                
                __sentences = [sent for (sentiment, sent, nps, review_time) in __sent_sentiment_nps_list if nps]
                clustering_result = ProductionHeuristicClustering(sentiment_np_time = __sentiment_np_time,
                                                                sub_category = sub_category,
                                                                sentences = __sentences,
                                                                eatery_name= self.eatery_name, 
                                                                places = self.mongo_instance.places_mentioned_for_eatery(), 
                                                                eatery_address = self.mongo_instance.eatery_address)
                return clustering_result.run()



        def aggregation(self, old, new=None):
                """
                __list can be either menu-food, overall-food,
                as these two lists doesnt require clustering but only aggreation of sentiment analysis
                Args:
                    case1:[ 
                        [u'negative', u'food is average .', [], u'2014-09-19 06:56:42'],
                        [u'negative', u"can't say i tasted anything that i haven't had before .", [u'i haven'],
                        u'2014-09-19 06:56:42'],
                        [u'negative', u"however , expect good food and you won't be disappointed .", [],
                        u'2014-09-19 06:56:42'],
                        [u'neutral', u'but still everything came hot ( except for the drinks of course ', [],
                        u'2014-09-19 06:56:42'],
                        ] 
                    
                    case1: output
                        {u'negative': 5, u'neutral': 1, u'positive': 2, u'super-positive': 1,
                        'timeline': [(u'super-positive', u'2014-04-05 12:33:45'), (u'positive', u'2014-05-06 13:06:56'),
                        (u'negative', u'2014-05-25 19:24:26'), (u'negative', u'2014-05-25 19:24:26'),
                        (u'positive', u'2014-06-09 16:28:09'), (u'negative', u'2014-09-19 06:56:42'),
                        (u'negative', u'2014-09-19 06:56:42'), (u'negative', u'2014-09-19 06:56:42'),
                        (u'neutral', u'2014-09-19 06:56:42')]}


                    case2: When aggregation has to be done on old and new noun phrases 
                        old: {"super-positive": 102, "super_negative": 23, "negative": 99, 
                                "positive": 76, "neutral": 32}
                        new as same as case1:
                        new: {"super-positive": 102, "super_negative": 23, "negative": 99, 
                                "positive": 76, "neutral": 32}
                    

                Result:
                    {u'negative': 4, u'neutral': 2}
                """

                ##If overall-food is empty
                if not bool(old):
                        ##returns {'poor': 0, 'good': 0, 'excellent': 0, 'mixed': 0, 'timeline': [], 'average': 0, 'total_sentiments': 0, 'terrible': 0}
                        sentiment_dict = dict()
                        [sentiment_dict.update({key: 0}) for key in self.sentiment_tags]
                        sentiment_dict.update({"timeline": list()})
                        sentiment_dict.update({"total_sentiments": 0})
                        return sentiment_dict

                sentiment_dict = dict()
                if new :
                        [sentiment_dict.update({key: (old.get(key) + new.get(key))}) for key in self.sentiment_tags] 
                        #this statement ensures that we are dealing with case 1
                        sentiment_dict.update({"timeline": sorted((old.get("timeline") + new.get("timeline")), key= lambda x: x[1] )})
                        sentiment_dict.update({"total_sentiments": old.get("total_sentiments")+ new.get("total_sentiments")})
                        return sentiment_dict


                filtered_sentiments = Counter([sentiment for (sentiment, sent, nps, review_time) in old])
                timeline = sorted([(sentiment, review_time) for (sentiment, sent, nps, review_time) in old], key=lambda x: x[1])
               
                def convert(key):
                        if filtered_sentiments.get(key):
                                return {key: filtered_sentiments.get(key) }
                        else:
                                return {key: 0}


                [sentiment_dict.update(__dict) for __dict in map(convert,  self.sentiment_tags)]
                sentiment_dict.update({"timeline": timeline})
                total = sentiment_dict.get("positive") + sentiment_dict.get("negative") + sentiment_dict.get("neutral") \
                                    +sentiment_dict.get("mixed") 

                sentiment_dict.update({"total_sentiments": total})
                return sentiment_dict


        def __overall(self, __list):
                sentiment_dict = dict()
                [sentiment_dict.update({key: 0}) for key in self.sentiment_tags]
                sentiment_dict.update({"timeline": list()})
                sentiment_dict.update({"total_sentiments": 0})
                
                if not __list:
                        return sentiment_dict ##returns empyt dict if there is no sentences belonging to overall

                filtered_sentiments = Counter([sentiment for (sentence, tag, sentiment, review_time) in __list])
                timeline = sorted([(sentiment, review_time) for (sentence, tag, sentiment, review_time) in __list], key=lambda x: x[1])
                def convert(key):
                        if filtered_sentiments.get(key):
                                return {key: filtered_sentiments.get(key) }
                        else:
                                return {key: 0}


                [sentiment_dict.update(__dict) for __dict in map(convert,  self.sentiment_tags)]
                sentiment_dict.update({"timeline": timeline})
                total = sentiment_dict.get("negative")\
                        + sentiment_dict.get("positive")\
                        + sentiment_dict.get("neutral") + sentiment_dict.get("mixed")

                sentiment_dict.update({"total_sentiments": total})
                return sentiment_dict



        def unmingle_food_sub(self, __list):
                """
                __list = [u'the panner chilly was was a must try for the vegetarians from the menu .', 
                u'food', u'positive', u'menu-food',[]],
                [u'and the Penne Alfredo pasta served hot and good with lots of garlic flavours which 
                we absolute love .', u'food',cu'super-positive',u'dishes', [u'garlic flavours', 
                u'penne alfredo pasta']],

                result:
                    {u'dishes': [[u'positive', 'sent', [u'paneer chilli pepper starter'], '2014-09-19 06:56:42'],
                                [u'positive', 'sent', [], '2014-09-19 06:56:42'],
                                [u'positive', sent, [u'friday night'], '2014-09-19 06:56:42'],
                                [u'positive', sent, [], '2014-09-19 06:56:42'],
                                [u'positive', sent, [], '2014-09-19 06:56:42'],
                                [u'super-positive', sent, [u'garlic flavours', u'penne alfredo pasta']]],
                    u'menu-food': [[u'positive', sent, [], u'2014-06-09 16:28:09']],
                    u'null-food': [[u'negative', sent, [], u'2014-06-09 16:28:09'],
                                [u'super-positive', sent, [], '2014-09-19 06:56:42'],
                                [u'super-positive', sent, [], '2014-09-19 06:56:42'],
                                [u'negative', sent, [], '2014-09-19 06:56:42'],
                                }
                """
                __sub_tag_dict = dict()
                for (sent, tag, sentiment, sub_tag, nps, review_time)  in __list:
                        if not __sub_tag_dict.has_key(sub_tag):
                                __sub_tag_dict.update({sub_tag: [[sentiment, sent, nps, review_time]]})
                        
                        else:
                            __old = __sub_tag_dict.get(sub_tag)
                            __old.append([sentiment, sent, nps, review_time])
                            __sub_tag_dict.update({sub_tag: __old})

                return __sub_tag_dict

        def make_cluster(self, __nps, __category):
                """
                args:
                    __nps : [[u'super-positive', u'ambience-overall', u'2014-09-19 06:56:42'],
                            [u'neutral', u'ambience-overall', u'2014-09-19 06:56:42'],
                            [u'positive', u'open-area', u'2014-09-19 06:56:42'],
                            [u'super-positive', u'ambience-overall', u'2014-08-11 12:20:18'],
                            [u'positive', u'decor', u'2014-04-05 12:33:45'],
                            [u'super-positive', u'decor', u'2014-05-06 18:50:17'],
                
                return:
                        [{'name': u'decor', u'positive': 1, "timeline": },
                        {'name': u'ambience-overall', u'neutral': 1, u'super-positive': 2,"timeline"
                                :  [('super-positive','2014-09-19 06:56:42'), ("super-positive": '2014-08-11 12:20:18')]}]
                    
                """
                final_dict = dict()
                nps_dict = self.make_sentences_dict(__nps, __category)
                [final_dict.update({key: self.flattening_dict(key, value)}) for key, value in nps_dict.iteritems()]
                return final_dict

        def adding_new_old_nps(self, __new, __old):
                """
                For lets say ambience category the input will be of the form:
                   __new = { "smoking-zone":
                                {u'total_sentiments': 0, u'positive': 0, u'timeline': [], u'negative': 0, u'super-positive': 0, 
                                                                                            u'neutral': 0, u'super-negative': 0}, 

                            "dancefloor": {u'total_sentiments': 0, u'positive': 0, u'timeline': [], u'negative': 0, 
                                                    u'super-positive': 0, u'neutral': 0, u'super-negative': 0},

                            "open-area": {u'total_sentiments': 1, u'positive': 0, u'timeline': [[u'super-positive', u'2013-04-24 12:08:25']],
                                                        u'negative': 0, u'super-positive': 1, u'neutral': 0, u'super-negative': 0}
                            }
                    __old: same as new

                    Result:
                        Adds both the dictionaries based on the keys!!
                """
                aggregated = dict()

                keys = set.union(set(__new.keys()), set(__old.keys()))
                for key in keys:
                        a = __new.get(key)
                        b = __old.get(key)
                        __keys = set.union(set(a.keys()), set(b.keys()))
                        sentiments = dict()
                        for __key in __keys:
                                sentiments.update({__key: a.get(__key) + b.get(__key)})
                        aggregated.update({key: sentiments})

                return aggregated



        def flattening_dict(self, key, value):
                """
                key: ambience-overall 
                value: 
                    {'sentiment': [u'super-positive',u'neutral', u'super-positive', u'neutral', u'neutral'],
                    'timeline': [(u'super-positive', u'2014-09-19 06:56:42'), (u'neutral', u'2014-09-19 06:56:42'),
                            (u'super-positive', u'2014-08-11 12:20:18'), (u'neutral', u'2014-05-06 13:06:56'),
                            (u'neutral', u'2014-05-06 13:06:56')]},


                Output: 
                    {'neutral': 3, u'super-positive': 2, 
                'timeline': [(u'neutral', u'2014-05-06 13:06:56'), (u'neutral', u'2014-05-06 13:06:56'),
                (u'super-positive', u'2014-08-11 12:20:18'), (u'super-positive', u'2014-09-19 06:56:42'),
                (u'neutral', u'2014-09-19 06:56:42')], "total_sentiments": 10},

                """
                __dict = dict()
                sentiments = Counter(value.get("sentiment"))
                def convert(key):
                        if sentiments.get(key):
                                return {key: sentiments.get(key) }
                        else:
                                return {key: 0}


                [__dict.update(__sentiment_dict) for __sentiment_dict in map(convert, self.sentiment_tags)]

                __dict.update({"timeline": sorted(value.get("timeline"), key=lambda x: x[1] )})
                __dict.update({"total_sentiments": value.get("total_sentiments")})
                return __dict


        def make_sentences_dict(self, noun_phrases, category):
                """
                Input:
                    [[u'super-positive', u'ambience-overall', u'2014-09-19 06:56:42'],
                    [u'neutral', u'ambience-overall', u'2014-09-19 06:56:42'],
                    [u'positive', u'open-area', u'2014-09-19 06:56:42'],
                    [u'super-positive', u'ambience-overall', u'2014-08-11 12:20:18'],
                    [u'positive', u'decor', u'2014-04-05 12:33:45'],
                    [u'super-positive', u'decor', u'2014-05-06 18:50:17'],
                    [u'neutral', u'ambience-overall', u'2014-05-06 13:06:56'],
                    [u'positive', u'decor', u'2014-05-06 13:06:56'],
                    [u'positive', u'music', u'2014-05-06 13:06:56'],
                    [u'neutral', u'ambience-overall', u'2014-05-06 13:06:56']]
                
                Result:

                        {"romantic":  {u'total_sentiments': 0, u'positive': 0, u'timeline': [], u'negative': 0, 
                                        u'super-positive': 0, u'neutral': 0, u'super-negative': 0},

                        "crowd": {u'total_sentiments': 4, u'positive': 1, u'timeline': [[u'negative', u'2013-03-24 02:00:43'], 
                        [u'positive', u'2014-03-27 00:19:55'], [u'negative', u'2014-11-15 15:31:50'], [
                        u'negative', u'2014-11-15 15:31:50']], u'negative': 3, u'super-positive': 0, u'neutral': 0, u'super-negative': 0}
                        }
                """
                sentences_dict = dict()

                for sub_tag in eval("self.{0}_tags".format(category.replace("_result", ""))):
                        print sub_tag
                        for sentiment in self.sentiment_tags:
                            sentences_dict.update({sub_tag: {"sentiment": list(), "timeline": list(), "total_sentiments": 0}})

                for __sentiment, __category, review_time in noun_phrases:
                        try:
                            timeline = sentences_dict.get(__category).get("timeline")
                            timeline.append((__sentiment, review_time))
                            sentiment = sentences_dict.get(__category).get("sentiment")
                            sentiment.append(__sentiment)
                            total_sentiments = sentences_dict.get(__category).get("total_sentiments") +1 

                            sentences_dict.update({
                                __category: {"sentiment": sentiment, "timeline": timeline, "total_sentiments": total_sentiments}})
                        except Exception as e:
                                print e
                                pass

                return sentences_dict


if __name__ == "__main__":
            
            ##To check if __extract_places is working or not            
            ##ins = PerReview('2036121', 'Average quality food, you can give a try to garlic veg chowmien if you are looking for a quick lunch in Sector 44, Gurgaon where not much options are available.','2014-08-08 15:09:17', '302115')
            ##ins.run()
            import optparse
            
            parser = optparse.OptionParser(usage="usage: %prog [options] filename",
                          version="%prog 1.0")
        
        
            parser.add_option("-p", "--path_compiled_classifiers",
                      action="store", # optional because action defaults to "store"
                      dest="path_compiled_classifiers",
                      default=None,
                      help="Path for compiled classifiers",)
        
            (options, args) = parser.parse_args()
            PATH_COMPILED_CLASSIFIERS = options.path_compiled_classifiers
        
            eatery_ids = ["166", "309790", "308322", "834", "1626", "400", "18034053", "6127", "308637", "310078"]
            ins = DoClusters("308322")
            ins.run()
            """
            eatery_ids_one = [ post.get("eatery_id") for post in\
                    eateries.find({"eatery_area_or_city": "Delhi NCR"}) if\
                            reviews.find({"eatery_id": post.get("eatery_id")}).count() >= 1500]
            Instance = ClassifiyReviews(["166"], PATH_COMPILED_CLASSIFIERS)
            Instance.run()
            eatery_ids_one = [ post.get("eatery_id") for post in  eateries.find({"eatery_area_or_city": "Delhi NCR"}) if reviews.find({"eatery_id": post.get("eatery_id")}).count() >= 500]
            j = 5
            for i in eatery_ids_one:
                    __list = eatery_ids_one[j: j+5]
                    print __list
                    Instance = ClassifiyReviews(__list)
                    Instance.run()
                    for eatery_id in __list:
                            ins = DoClusters(eatery_id)
                            ins.run()
                            ElasticSearchScripts.insert_eatery(eatery_id)   
                    j += 5
                    if j == len(eatery_ids_one):
                        break

            """
            eatery_ids_one = [ post.get("eatery_id") for post in eateries.find({"eatery_area_or_city": "Delhi NCR"}) if reviews.find({"eatery_id": post.get("eatery_id")}).count() >= 1000]
            for eatery_id in eatery_ids_one:
                            ins = DoClusters(eatery_id)
                            ins.run()
            """     

            for eatery_id in eatery_ids:
                    print Terminal.green("Clustering %s"%eatery_id)
                    try:
                        ins = DoClusters(eatery_id)
                        ins.run()
                    except Exception as e:
                        print e
                        print eatery_id

            """

