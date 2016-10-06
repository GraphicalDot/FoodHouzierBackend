#!/usr/bin/env python
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
from text_processing_db_scripts import MongoScriptsReviews, MongoScriptsDoClusters
from nltk.stem import SnowballStemmer

#from prod_heuristic_clustering import ProductionHeuristicClustering

#from join_two_clusters import ProductionJoinClusters


this_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.dirname(this_file_path)
print parent_dir_path

sys.path.append(parent_dir_path)
from prod_heuristic_clustering import  ProductionHeuristicClustering
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

from configs import eateries, reviews 

import blessings
Terminal = blessings.Terminal()

from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

#from elasticsearch_db import ElasticSearchScripts
from topia.termextract import extract  
from simplejson import loads
#from google_places import google_already_present, find_google_places







class EachEatery:
        def __init__(self, eatery_id, flush_eatery=False):
                self.eatery_id = eatery_id
                if flush_eatery:
                        ##when you want to process whole data again, No options other than that
                        warnings.warn("Fushing whole atery") 
                        MongoScriptsReviews.flush_eatery(eatery_id)
                return 
        
        def return_non_processed_reviews(self, start_epoch=None, end_epoch=None):
                """
                case1: 
                    Eatery is going to be processed for the first time

                case 2:
                    Eatery was processed earlier but now there are new reviews are to be processed

                case 3: 
                    All the reviews has already been processed, No reviews are left to be processed 
                all_reviews = MongoScriptsReviews.return_all_reviews(self.eatery_id) 
                try:
                        ##case1: all_processed_reviews riases StandardError as there is no key in eatery result for processed_reviews
                        all_processed_reviews = MongoScriptsReviews.get_proccessed_reviews(self.eatery_id)

                except StandardError as e:
                        warnings.warn("Starting processing whole eatery, YAY!!!!")
                reviews_ids = list(set.symmetric_difference(set(all_reviews), set(all_processed_reviews)))
                if reviews_ids:
                        ##case2: returning reviews which are yet to be processed 
                        return MongoScriptsReviews.reviews_with_text(reviews_ids)
                
                else:
                        warnings.warn("{0} No New reviews to be considered for eatery id {1} {2}".format(bcolors.OKBLUE, self.eatery_id, bcolors.RESET))
                        return list() 
                """
                MongoScriptsReviews.insert_eatery_into_results_collection(self.eatery_id)
                """
                if google:
                        google_already_present(eatery_id, google)
                    
                else:
                        find_google_places(eatery_id)
                
                """
                review_ids = MongoScriptsReviews.review_ids_to_be_processed(self.eatery_id)
                if not review_ids:
                        print Terminla.red("No reviews are to be processed")
               
                
                result = MongoScriptsReviews.reviews_with_text(review_ids, self.eatery_id)
                print Terminal.green("Length of the review ids %s for the eatery id is %s"%(len(review_ids), self.eatery_id))
                return result





class PerReview:
        def __init__(self, review_id, review_text, review_time, eatery_id):
                """
                Lowering the review text
                """
                self.review_id, self.review_time, self.eatery_id = review_id, review_time, eatery_id

                """ 
                print review_text
                encoded_text = SolveEncoding.to_unicode_or_bust(review_text)
                print encoded_text
                stemmed_text = PerReview.snowball_stemmer(encoded_text)
                print stemmed_text
                processed_text = PerReview.pre_process_text(review_text)
                """

                self.review_text= review_text.encode("ascii", "ignore")
                self.cuisine_name = list()
                self.places_names = list()
                self.np_extractor = extract.TermExtractor() 
                self.run()

        @staticmethod
        def snowball_stemmer(sentences):
                stemmer = SnowballStemmer("english")
                return [stemmer.stem(sent) for sent in sentences]


        @staticmethod 
        def pre_process_text(sentences):
                return [PreProcessText.process(sent) for sent in sentences]


        def prediction(self, sentences, vocabulary, features, classifier):
                print sentences
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary) 
                
                sentences_counts = loaded_vectorizer.transform(sentences)
                
                reduced_features = features.transform(sentences_counts.toarray())
                         
                predictions = classifier.predict(reduced_features)
                return predictions
 

        def print_execution(func):
                "This decorator dumps out the arguments passed to a function before calling it"
                argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
                fname = func.func_name
                def wrapper(*args,**kwargs):
                        start_time = time.time()
                        print Terminal.green("Now {0} have started executing".format(func.func_name))
                        result = func(*args, **kwargs)
                        print Terminal.green("Total time taken by {0} for execution is --<<{1}>>".format(func.func_name, (time.time() - start_time)))
                        
                        return result
                return wrapper
        
        def get_args(self):
                print self.__dict__
        

        @print_execution
        def run(self):
                print Terminal.green("Now processing review id --<<{0}>>".format(self.review_id))
                self.__sent_tokenize_review() #Tokenize reviews, makes self.reviews_ids, self.sentences
                self.__predict_tags()          #Predict tags, makes self.predict_tags
                self.__predict_sentiment() #makes self.predicted_sentiment

                self.all_sent_tag_sentiment = zip(self.sentences, self.tags, self.sentiments)
                
                self.__filter_on_category() #generates self.food, self.cost, self.ambience, self.service
                

                self.__food_sub_tag_classification()
                self.__service_sub_tag_classification()
                self.__cost_sub_tag_classification()
                self.__ambience_sub_tag_classification()
                self.__extract_places()
                self.__extract_cuisines()
                self.__extract_noun_phrases() #makes self.noun_phrases
                self.__append_time_to_overall()
                self.__append_time_to_menu()
                self.__update_cuisine_places() 
                self.__update_review_result()
                return 

        
        @print_execution
        def __sent_tokenize_review(self):
                """
                Tokenize self.reviews tuples of the form (review_id, review) to sentences of the form (review_id, sentence)
                and generates two lists self.review_ids and self.sentences
                """
                print self.review_text
                
                text = PreProcessText.process(self.review_text)
                self.sentences = tokenizer.tokenize(text)
                ##now breking sentences on the basis of but
                new_sentences = list()
                for sentence in self.sentences:
                     new_sentences.extend(sentence.split("but"))
                self.sentences = filter(None, new_sentences)
                print self.sentences
                #processed_text = PreProcessText.remove_and_replace(text)
                return
        
        @print_execution
        def __predict_tags(self):
                """
                Predict tags of the sentence which were being generated by self.sent_tokenize_reviews
                """
                self.tags = self.prediction(self.sentences, tag_vocabulary,
                                            tag_features, tag_classifier)
                print "Tags for %s review id are %s"%(self.tags,
                                                      self.review_id)
                return

        @print_execution
        def __predict_sentiment(self):
                """
                Predict sentiment of self.c_sentences which were made by filtering self.sentences accoring to 
                the specified category
                """
                self.sentiments = self.prediction(self.sentences, sentiment_vocabulary,
                                            sentiment_features, sentiment_classifier)
                print "Sentiments for %s review id are %s"%(self.sentiments,
                                                      self.review_id)
                return 
     
        @print_execution
        def __filter_on_category(self):
                __filter = lambda tag: [(sent, __tag, sentiment) for \
                                        (sent,__tag, sentiment) in \
                                    self.all_sent_tag_sentiment if __tag ==tag]
                self.food, self.cost, self.ambience, self.service, self.null, \
                    self.overall, self.places, self.cuisine, self.menu = \
                    __filter("food"),  __filter("cost"), __filter("ambience"),\
                    __filter("service"), __filter("null"),  __filter("overall"), __filter("place"), __filter("cuisine"), __filter("menu")
                return 

        @print_execution
        def __food_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                if not bool(self.food):
                        self.all_food = []
                        return 
                self.food_sentences = zip(*self.food)[0]
                self.food_sub_tags = self.prediction(self.food_sentences,
                                                     food_vocabulary,
                                            food_features, food_classifier)
               
                print Terminal.green("Here are the food sentences with predictions")
                for (sent, tag) in zip(self.food_sentences, self.food_sub_tags):
                            print (sent, tag)
                
                self.all_food = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag)\
                        in zip(self.food, self.food_sub_tags)]

                return  

        @print_execution
        def __service_sub_tag_classification(self):
                """
                This deals with the sub classification of service sub tags
		and generates self.all_service with an element in the form 
		(sent, tag, sentiment, sub_tag_service)
                """
                if not bool(self.service):
                        self.all_service = []
                        return 

                self.service_sentences = zip(*self.service)[0]
                self.service_sub_tags = self.prediction(self.service_sentences,
                                                     service_vocabulary,
                                            service_features, service_classifier)
               
                print Terminal.green("Here are the service sentences with predictions")
                for (sent, tag) in zip(self.service_sentences, self.service_sub_tags):
                            print (sent, tag)
                
                self.all_service = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag)\
                        in zip(self.service, self.service_sub_tags)]

                map(lambda __list: __list.append(self.review_time), self.all_service)
                return 

        @print_execution
        def __cost_sub_tag_classification(self):
                """
                This deals with the sub classification of cost sub tags
                
                self.all_cost = [(sent, "cost", sentiment, "cost-overall",), .....]
                """
                if not bool(self.cost):
                        self.all_cost = []
                        return 
                self.cost_sentences = zip(*self.cost)[0]
                self.cost_sub_tags = self.prediction(self.cost_sentences,
                                                     cost_vocabulary,
                                            cost_features, cost_classifier)
               
                print Terminal.green("Here are the cost sentences with predictions")
                for (sent, tag) in zip(self.cost_sentences, self.cost_sub_tags):
                            print (sent, tag)
                
                self.all_cost = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag)\
                                 in zip(self.cost, self.cost_sub_tags)]
                
                map(lambda __list: __list.append(self.review_time), self.all_cost)
                return

        @print_execution
        def __ambience_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                if not bool(self.ambience):
                        self.all_ambience = []
                        return 
                self.ambience_sentences = zip(*self.ambience)[0]
                self.ambience_sub_tags = self.prediction(self.ambience_sentences,
                                                     ambience_vocabulary,
                                            ambience_features, ambience_classifier)
               
                print Terminal.green("Here are the ambience sentences with predictions")
                for (sent, tag) in zip(self.ambience_sentences, self.ambience_sub_tags):
                            print (sent, tag)
                
                self.all_ambience = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag)\
                        in zip(self.ambience, self.ambience_sub_tags)]

               
                map(lambda __list: __list.append(self.review_time), self.all_ambience)
                return 


        @print_execution
        def __append_time_to_overall(self):
                self.overall = [list(e) for e in self.overall]
                map(lambda __list: __list.append(self.review_time), self.overall)
                return  
        
        @print_execution
        def __append_time_to_menu(self):
                self.menu = [list(e) for e in self.menu]
                map(lambda __list: __list.append(self.review_time), self.menu)
                return
            
        @print_execution
        def __extract_places(self):
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


                for (sent, sentiment, tag) in self.places:
                            try:
                                    result = loads(corenlpserver.parse(sent))
                                    __result = [(e[0], e[1].get("NamedEntityTag")) for e in result["sentences"][0]["words"]]
                                    self.places_names.extend(filter_places(__result))
                            
                            except Exception as e:
                                    print e, "__extract_place", self.review_id
                                    pass
                print Terminal.green("Here are the places names for review id %s"%self.review_id)
                print self.places_names
                return
        
        @print_execution
        def __extract_cuisines(self):
                """
                This extracts the name of the cuisines fromt he cuisines sentences
                """

		
                for (sent, tag, sentiment) in self.cuisine:
                        self.cuisine_name.extend(self.np_extractor(sent))
		        		

                self.cuisine_name = [np[0] for np in self.cuisine_name if np[0]]
                print Terminal.green("Here are the cuisine names for review id %s"%self.review_id)
                print self.cuisine_name
                return 

                       


        @print_execution
        def __extract_noun_phrases(self):
                """
                Extarct Noun phrases for the self.c_sentences for each sentence and outputs a list 
                self.sent_sentiment_nps which is of the form 
                [('the only good part was the coke , thankfully it was outsourced ', 
                                            u'positive', [u'good part']), ...]
                """
                __nouns = list()
                for (sent, tag, sentiment, sub_tag) in self.all_food:
                            __nouns.append([e[0] for e in self.np_extractor(sent)])

                self.all_food_with_nps = [[sent, tag, sentiment, sub_tag, nps] for ((sent, tag, sentiment, sub_tag,), nps) in 
                        zip(self.all_food, __nouns)]

                map(lambda __list: __list.append(self.review_time), self.all_food_with_nps)
                print __nouns
                return 

        @print_execution
        def __update_cuisine_places(self):
                """
                update cuisine and places to the eatery
                """
                MongoScriptsReviews.update_eatery_places_cusines(self.eatery_id, self.places_names, self.cuisine_name)        
                return 
                


        
        @print_execution
        def __update_review_result(self):
                MongoScriptsReviews.update_review_result_collection(
                        review_id = self.review_id, 
                        eatery_id = self.eatery_id, 
                        food = self.food,
                        cost = self.cost,
                        ambience = self.ambience,
                        null = self.null,
                        overall = self.overall,
                        service = self.service, 
                        place_sentences = self.places, 
                        cuisine_sentences= self.cuisine,
                        food_result= self.all_food_with_nps, 
                        service_result = self.all_service, 
                        menu_result = self.menu,
                        cost_result = self.all_cost, 
                        ambience_result = self.all_ambience,
                        places_result= self.places_names, 
                        cuisine_result = self.cuisine_name) 
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
                self.ambience_tags = [u'smoking-zone', u'decor', u'ambience-null', u'ambience-overall', u'in-seating', u'crowd', u'open-area', u'dancefloor', u'music', u'location', u'romantic', u'sports', u'live-matches', u'view']
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
                        for sentiment in self.sentiment_tags:
                            sentences_dict.update({sub_tag: {"sentiment": list(), "timeline": list(), "total_sentiments": 0}})


                for __sentiment, __category, review_time in noun_phrases:
                        timeline = sentences_dict.get(__category).get("timeline")
                        timeline.append((__sentiment, review_time))
                        sentiment = sentences_dict.get(__category).get("sentiment")
                        sentiment.append(__sentiment)
                        total_sentiments = sentences_dict.get(__category).get("total_sentiments") +1 

                        sentences_dict.update({
                            __category: {"sentiment": sentiment, "timeline": timeline, "total_sentiments": total_sentiments}})
    
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


            with cd(ServiceClassifiersPath(PATH_COMPILED_CLASSIFIERS)):
                        service_features =  joblib.load(ServiceFeatureFileName)
                        service_vocabulary = joblib.load(ServiceVocabularyFileName)
                        service_classifier= joblib.load(ServiceClassifierFileName)

        
            print Terminal.green("<<%s>> classifiers and vocabulary loaded"%"Service") 
        
            with cd(SentimentClassifiersPath(PATH_COMPILED_CLASSIFIERS)):
                        sentiment_features =  joblib.load(SentimentFeatureFileName)
                        sentiment_vocabulary = joblib.load(SentimentVocabularyFileName)
                        sentiment_classifier= joblib.load(SentimentClassifierFileName)
        
            print Terminal.green("<<%s>> classifiers and vocabulary loaded"%"Sentiment") 


            with cd(TagClassifiersPath(PATH_COMPILED_CLASSIFIERS)):
                        tag_features =  joblib.load(TagFeatureFileName)
                        tag_vocabulary = joblib.load(TagVocabularyFileName)
                        tag_classifier= joblib.load(TagClassifierFileName)
            print Terminal.green("<<%s>> classifiers and vocabulary loaded"%"Tag") 

            with cd(FoodClassifiersPath(PATH_COMPILED_CLASSIFIERS)):
                        food_features =  joblib.load(FoodFeatureFileName)
                        food_vocabulary = joblib.load(FoodVocabularyFileName)
                        food_classifier= joblib.load(FoodClassifierFileName)
            print Terminal.green("<<%s>> classifiers and vocabulary loaded"%"Food") 

            with cd(CostClassifiersPath(PATH_COMPILED_CLASSIFIERS)):
                        cost_features =  joblib.load(CostFeatureFileName)
                        cost_vocabulary = joblib.load(CostVocabularyFileName)
                        cost_classifier= joblib.load(CostClassifierFileName)
            print Terminal.green("<<%s>> classifiers and vocabulary loaded"%"Cost") 

            with cd(AmbienceClassifiersPath(PATH_COMPILED_CLASSIFIERS)):
                        ambience_features =  joblib.load(AmbienceFeatureFileName)
                        ambience_vocabulary = joblib.load(AmbienceVocabularyFileName)
                        ambience_classifier= joblib.load(AmbienceClassifierFileName)
            print Terminal.green("<<%s>> classifiers and vocabulary loaded"%"Ambience") 
            eatery_ids_one = [ post.get("eatery_id") for post in\
                    eateries.find({"eatery_area_or_city": "Delhi NCR"}) if\
                    reviews.find({"eatery_id": post.get("eatery_id")}).count()> 20]
            
            
            for eatery_id in eatery_ids_one:
                    start = time.time()
                    instance = EachEatery(eatery_id)
                    result = instance.return_non_processed_reviews()
            
            
                    for element in result:
                                instance = PerReview(*element)
            
                    instance = DoClusters(eatery_id)
                    instance.run()
                    print time.time() - start
            """
            #ins = DoClusters(eatery_id)
            #ins.run()
            i = 0
            for post in eateries_results_collection.find():
                    eatery_id = post.get("eatery_id")
                    instance = EachEatery(eatery_id)
                    result = instance.return_non_processed_reviews()
                    result = [(e[0], e[1], e[2], eatery_id) for e in result]
                    for element in result:
                            instance = PerReview(element[0], element[1], element[2], element[3])
                            instance.run()
                    ins = DoClusters(eatery_id)
                    ins.run()
                    print "\n\n"
                    print "This is the count %s"%i
                    i += 1

            """

