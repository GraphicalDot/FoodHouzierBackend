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
from prod_heuristic_clustering import ProductionHeuristicClustering

from join_two_clusters import ProductionJoinClusters


this_file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.dirname(this_file_path)
sys.path.append(parent_dir_path)


from sklearn.externals import joblib
from Text_Processing.Sentence_Tokenization.Sentence_Tokenization_Classes import SentenceTokenizationOnRegexOnInterjections
from Text_Processing.NounPhrases.noun_phrases import NounPhrases
from Text_Processing.Word_Tokenization.word_tokenizer import WordTokenize
from Text_Processing.PosTaggers.pos_tagging import PosTaggers
from Text_Processing.MainAlgorithms.paths import path_parent_dir, path_in_memory_classifiers
from Text_Processing.NER.ner import NERs
from Text_Processing.MainAlgorithms.Algorithms_Helpers import get_all_algorithms_result
from Text_Processing.MainAlgorithms.In_Memory_Main_Classification import timeit, cd
from elasticsearch_db import ElasticSearchScripts
from Normalized import NormalizingFactor
from topia.termextract import extract  
from simplejson import loads
from google_places import google_already_present, find_google_places

from connections import sentiment_classifier, tag_classifier, food_sb_classifier, ambience_sb_classifier, service_sb_classifier, \
            cost_sb_classifier, SolveEncoding, bcolors, corenlpserver,  reviews, eateries, eateries_results_collection, reviews_results_collection






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
                google = MongoScriptsReviews.insert_eatery_into_results_collection(self.eatery_id)
                if google:
                        google_already_present(eatery_id, google)
                    
                else:
                        find_google_places(eatery_id)
                
                
                review_ids = MongoScriptsReviews.review_ids_to_be_processed(self.eatery_id)
                if not review_ids:
                        print "No reviews are to be processed"
               
                
                result = MongoScriptsReviews.reviews_with_text(review_ids)
                print review_ids
                return result




class PerReview:
        sent_tokenizer = SentenceTokenizationOnRegexOnInterjections()
        def __init__(self, review_id, review_text, review_time, eatery_id):
                """
                Lowering the review text
                """
                self.review_id, self.review_text, self.review_time, self.eatery_id = review_id, \
                        SolveEncoding.to_unicode_or_bust(review_text.lower().replace("&nbsp;&nbsp;\n", "")), review_time, eatery_id


                print self.review_time, self.review_text, self.review_id, self.eatery_id
                self.cuisine_name = list()
                self.places_names = list()
                self.np_extractor = extract.TermExtractor() 


        def print_execution(func):
                "This decorator dumps out the arguments passed to a function before calling it"
                argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
                fname = func.func_name
                def wrapper(*args,**kwargs):
                        start_time = time.time()
                        print "{0} Now {1} have started executing {2}".format(bcolors.OKBLUE, func.func_name, bcolors.RESET)
                        result = func(*args, **kwargs)
                        print "{0} Total time taken by {1} for execution is --<<{2}>>--{3}\n".format(bcolors.OKGREEN, func.func_name, 
                                (time.time() - start_time), bcolors.RESET)
                        
                        return result
                return wrapper
        
        def get_args(self):
                print self.__dict__
        

        @print_execution
        def run(self):
                print "{0} Now processing review id --<<{1}>>--{2}".format(bcolors.FAIL, \
                                self.review_id, bcolors.RESET)
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
                self.sentences = self.sent_tokenizer.tokenize(self.review_text)
                return
        
        @print_execution
        def __predict_tags(self):
                """
                Predict tags of the sentence which were being generated by self.sent_tokenize_reviews
                """
                self.tags = tag_classifier.predict(self.sentences)
                return

        @print_execution
        def __predict_sentiment(self):
                """
                Predict sentiment of self.c_sentences which were made by filtering self.sentences accoring to 
                the specified category
                """
                self.sentiments = sentiment_classifier.predict(self.sentences)
                return 
        
        
	def __filter_on_category(self):
		"""
		Right now there are 
		[u'cuisine', u'service', u'food', u'overall', u'cost', u'place', u'ambience', u'null']
		main categories for the classification of the sentences

		all that has already been stored in all_sent_tag_sentiment alongwith the tag, sentiment
		of the sentences, 
		this function unzip these categpries and make class variables for these categories

		"""
		__filter = lambda tag: [(sent, __tag, sentiment) for (sent, __tag, sentiment) in \
                                                                                self.all_sent_tag_sentiment if __tag== tag ]
		self.food, self.cost, self.ambience, self.service, self.null, self.overall, self.places, self.cuisine, self.menu = \
                         __filter("food"),  __filter("cost"), __filter("ambience"), __filter("service"),\
			 __filter("null"),  __filter("overall"), __filter("place"), __filter("cuisine"), __filter("menu")

                return 

        @print_execution
        def __food_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                self.food_sub_tags = food_sb_classifier.predict([sent for (sent, tag, sentiment)  in self.food])
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
                self.service_sub_tags = service_sb_classifier.predict([sent for (sent, tag, sentiment) in self.service])
                self.all_service = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag) \
                        in zip(self.service, self.service_sub_tags)]
                
                map(lambda __list: __list.append(self.review_time), self.all_service)
                return 

        @print_execution
        def __cost_sub_tag_classification(self):
                """
                This deals with the sub classification of cost sub tags
                
                self.all_cost = [(sent, "cost", sentiment, "cost-overall",), .....]
                """

                self.cost_sub_tags = cost_sb_classifier.predict([sent for (sent, tag, sentiment) in self.cost])
                self.all_cost = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag) \
                        in zip(self.cost, self.cost_sub_tags)]
                
                map(lambda __list: __list.append(self.review_time), self.all_cost)
                return

        @print_execution
        def __ambience_sub_tag_classification(self):
                """
                This deals with the sub classification of fodd sub tags
                """
                self.ambience_sub_tags = ambience_sb_classifier.predict([sent for (sent, tag, sentiment) in self.ambience])
                self.all_ambience = [[sent, tag, sentiment, sub_tag] for ((sent, tag, sentiment), sub_tag) \
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
                return 
                            


	@print_execution
	def __extract_cuisines(self):
		"""
		This extracts the name of the cuisines fromt he cuisines sentences
		"""

		
                for (sent, tag, sentiment) in self.cuisine:
                        self.cuisine_name.extend(self.np_extractor(sent))
		        		

                self.cuisine_name = [np[0] for np in self.cuisine_name if np[0]]
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
        def __get_review_result(self):
                result = MongoScripts.get_review_result(review_id = self.review_id) 
                return result 

        
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

        @print_execution
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
                self.sentiment_tags = ["good", "poor", "average", "excellent", "terrible", "mixed"]
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
                        print "{0} No reviews are present for the eatery_id in reviews colllection\
                                = <<{1}>> {2}".format(bcolors.OKBLUE, self.eatery_id, bcolors.RESET)
                        return 

                #That clustering is running for the first time
                warnings.warn("{0} No clustering of noun phrases has been done yet  for eatery_id\
                                = <<{1}>>{2}".format(bcolors.FAIL, self.eatery_id, bcolors.RESET))
                       
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
                        (u'positive', [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                        (u'positive', [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                        (u'positive', [u'paneer chilli pepper starter'], u'2014-09-19 06:56:42'),
                         (u'neutral', [u'chicken pieces', u'veg pasta n'], u'2014-06-20 15:11:42')]  

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
                total = sentiment_dict.get("good") + sentiment_dict.get("poor") + sentiment_dict.get("average") + sentiment_dict.get("terrible")\
                                    +sentiment_dict.get("excellent") + sentiment_dict.get("mixed") 

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
                total = sentiment_dict.get("good") + sentiment_dict.get("poor") + sentiment_dict.get("average") + sentiment_dict.get("terrible")\
                                    +sentiment_dict.get("excellent") + sentiment_dict.get("mixed") 

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
            
            """
            eatery_id = "308322"
            instance = EachEatery(eatery_id)
            result = instance.return_non_processed_reviews()
            print result
            result = [(e[0], e[1], e[2], eatery_id) for e in result]
            for element in result:
                            instance = PerReview(element[0], element[1], element[2], element[3])
                            instance.run()
            ins = DoClusters(eatery_id)
            ins.run()

            """
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


