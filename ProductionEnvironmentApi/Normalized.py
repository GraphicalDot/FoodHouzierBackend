#!/usr/bin/env python
from datetime import datetime, timedelta
from compiler.ast import flatten
import time
import pymongo
import os
import sys

parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir_path)    

MOVING_AVEARGE_DAYS = 120

#from GlobalConfigs import eateries, eateries_results_collection, reviews, reviews_results_collection, connection


class NormalizingFactor(object):
        def __init__(self, eatery_id, date_format=None):
                """
                keys = {'total_sentiments': int(), 'positive': int(),  'negative': int(), 'super-positive': int(), 
                                                        'neutral':int(), 'timeline': [[u'negative', u'2014-11-06 12:10:27'], 
                                                            [u'negative', u'2014-11-06 12:10:27']], 'super-negative': int()},

                keys_with_name = {'name': str(), 'total_sentiments': int(), 'positive': int(),  'negative': int(), 'super-positive': int(), 
                                                        'neutral':int(), 'timeline': [[u'negative', u'2014-11-06 12:10:27'], 
                                                            [u'negative', u'2014-11-06 12:10:27']], 'super-negative': int(), "similar": []},

                Args:
                        eatery_result = {"food": { "menu-food": keys,
                                                "dishes": [keys_with_name, keys_with_name, ],
                                                'overall-food': keys  , 
                                                'sub-food': [keys_with_name, keys_with_name], 
                                                'place-food':
                                                }, 
                                    "ambience": {'smoking-zone': {}, 
                                                'decor': keys,  'ambience-null': keys, 'ambience-overall': keys, 'romantic': keys, 'crowd': keys, 
                                                'view': keys, 'open-area': keys, 'dancefloor': keys, 'music': keys, 'sports-props': keys, 
                                                'sports':keys, 'sports-screens': keys, 'in-seating': keys
                                    },
                                    "cost": {'value for money': keys, 'cost-null': keys, 'cheap': keys, 'expensive': keys, 'not worth': keys },
                                    "service": {'management': keys, 'service-overall': keys, 'service-null': keys, 
                                                'waiting-hours': keys, 'presentation': keys, 'booking': keys, 'staff': keys },
                                    }

                Takes in eatery_result and adds normalized constants to specific categories and sub_categories
                """
                self.eatery_id = eatery_id
                self.eatery_result = eateries_results_collection.find_one({"eatery_id": self.eatery_id})

                if not date_format:
                        self.date_format = "%Y-%m-%d"
                else:
                        self.date_format = date_format
                
                start_date = datetime.now() + timedelta(-MOVING_AVEARGE_DAYS)
                self.last_30_date = start_date.strftime(self.date_format)
                
                self.dishes = sorted(self.eatery_result.get("food").get("dishes"), key=lambda x: x.get("total_sentiments"), reverse=True)
                self.sub_food = sorted(self.eatery_result.get("food").get("sub-food"), key=lambda x: x.get("total_sentiments"), reverse=True)
                self.place_food = sorted(self.eatery_result.get("food").get("place-food"), key=lambda x: x.get("total_sentiments"), reverse=True)
                
                self.overall_food = self.eatery_result.get("food").get("overall-food")
                self.menu_food = self.eatery_result.get("food").get("menu_food")
                
                
                self.result_dict = {"food": {}, "ambience": {}, "service": {}, "cost": {}}

                
                        


                ##These category factores insures the weightage of particular category over another
                ##for example a eatery which have great ambience as comaped to food, will have higher ambienece category factor than food
                self.overall_positive_sentiments = sum(flatten([[self.eatery_result[category][key].get("positive")+ self.eatery_result[category][key].get("super-positive") 
                    for key in self.eatery_result[category].keys()] for category in ["service", "cost", "ambience"]])) + \
                            sum([self.eatery_result["food"][key].get("positive") +  self.eatery_result["food"][key].get("super-positive") for key in ["overall-food", "menu-food"]])\
                            + sum(flatten([[_.get("positive") +  _.get("super-positive") for _ in self.eatery_result["food"][key] ] for key in ["dishes", "sub-food", "place-food"]]))


                
                self.food_positives = sum([self.eatery_result["food"][key].get("positive") +  self.eatery_result["food"][key].get("super-positive") for key in ["overall-food", "menu-food"]])\
                        + sum(flatten([[_.get("positive") +  _.get("super-positive") for _ in self.eatery_result["food"][key] ] for key in ["dishes", "sub-food", "place-food"]]))
                
                self.food_overall = sum([self.eatery_result["food"][key].get("total_sentiments") for key in ["overall-food", "menu-food"]])\
                        + sum(flatten([[_.get("total_sentiments")for _ in self.eatery_result["food"][key] ] for key in ["dishes", "sub-food", "place-food"]]))


                


                self.overall_total_sentiments = sum(flatten([[self.eatery_result[category][key].get("total_sentiments") for key in \
                        self.eatery_result[category].keys()] for category in ["service", "cost", "ambience"]])) + \
                            sum([self.eatery_result["food"][key].get("total_sentiments") for key in ["overall-food", "menu-food"]])\
                            + sum(flatten([[_.get("total_sentiments")for _ in self.eatery_result["food"][key] ] for key in ["dishes", "sub-food", "place-food"]]))

                try:
                        self.food_positive_factor = float(self.food_positives)/self.overall_positive_sentiments
                except ZeroDivisionError as e:
                        self.food_positive_factor = 0
               
                try:
                        self.food_overall_factor =  float(self.food_overall)/self.overall_total_sentiments
                except ZeroDivisionError as e:
                        self.food_overall_factor = 0

                try:
                        self.ambience_positive_factor = float(self.per_category_positives("ambience"))/self.overall_positive_sentiments
                except ZeroDivisionError as e:
                        self.ambience_positive_factor = 0

                try:
                        self.ambience_overall_factor = float(self.per_category_positives("ambience"))/self.overall_total_sentiments
                except ZeroDivisionError as e:
                        self.ambience_overall_factor = 0
                
                try:
                        self.cost_positive_factor = float(self.per_category_positives("cost"))/self.overall_positive_sentiments
                except ZeroDivisionError as e:
                        self.cost_positive_factor = 0

                try:
                        self.cost_overall_factor = float(self.per_category_positives("cost"))/self.overall_total_sentiments
                except ZeroDivisionError as e:
                        self.cost_overall_factor = 0
                
                try:
                        self.service_positive_factor = float(self.per_category_positives("service"))/self.overall_positive_sentiments
                except ZeroDivisionError as e:
                        self.service_positive_factor = 0 
                
                try:
                        self.service_overall_factor = float(self.per_category_positives("service"))/self.overall_total_sentiments
                except ZeroDivisionError as e:
                        self.service_overall_factor = 0


                print self.food_overall_factor, self.food_positive_factor
                print self.ambience_overall_factor, self.ambience_positive_factor
                print self.cost_overall_factor, self.cost_positive_factor
                print self.service_overall_factor, self.service_positive_factor


        def run(self):
                if self.dishes:
                        self.result_dict["food"]["dishes"] = self.if_object_a_list(self.dishes, self.food_overall_factor, self.food_positive_factor)                       
                else:
                        self.result_dict["food"]["dishes"] = []
                
                if self.sub_food:
                        self.result_dict["food"]["sub-food"] = self.if_object_a_list(self.sub_food, self.food_overall_factor, self.food_positive_factor)                       
                else:
                        self.result_dict["food"]["sub-food"] = []
                if self.place_food:
                        self.result_dict["food"]["place-food"] = self.if_object_a_list(self.place_food, self.food_overall_factor, self.food_positive_factor)                       
                else:
                        self.result_dict["food"]["place-food"] = []
                
                if self.overall_food:
                        self.result_dict["food"]["overall-food"] = self.if_single_object(self.overall_food, self.food_overall_factor, self.food_positive_factor)                       
                else:
                        self.result_dict["food"]["overall-food"] = {}
                
                if self.menu_food:
                        self.result_dict["food"]["menu-food"] = self.if_single_object(self.menu_food, self.food_overall_factor, self.food_positive_factor)                       
                else:
                        self.result_dict["food"]["menu-food"] = {}
               
                



                self.result_dict["ambience"] = self.if_object(self.eatery_result.get("ambience"), self.ambience_overall_factor, self.ambience_positive_factor)
                self.result_dict["cost"] = self.if_object(self.eatery_result.get("cost"), self.cost_overall_factor, self.cost_positive_factor)
                self.result_dict["service"] = self.if_object(self.eatery_result.get("service"), self.service_overall_factor, self.service_positive_factor)
               

                print self.result_dict
                ##Updating results in the eateries_results_collection with objects with trending_factor and mention_factor
                eateries_results_collection.update({"eatery_id": self.eatery_id}, {"$set": 
                    {"food": self.result_dict["food"],
                        "cost": self.result_dict["cost"],
                        "ambience": self.result_dict["ambience"],
                        "service": self.result_dict["service"],
                                                        }
                    })
                
                return 


        def per_category_positives(self, category):
                """
                """
                return sum(flatten([self.eatery_result[category][key].get("positive")+ self.eatery_result[category][key].get("super-positive") 
                    for key in self.eatery_result[category].keys()]))
        
        def per_category_total_sentiments(self, category):
                """
                """
                return sum(flatten([self.eatery_result[category][key].get("total_sentiments") for key in self.eatery_result[category].keys()])) 
                



        def if_object_a_list(self, __object_list, category_overall_factor, category_positive_factor):
                """
                Calculates trending_factor and mention_factor if __object_list for each oject presents in __object_list
                this is valiid for food sub category "dishes", "sub-food", "place-food", 

                trending_factor = dish.get("positives")/total_sentiments_for_dish * total_sentiments_for_dish/total_sentiments_for_all_dish
                                    *last_30_days_positive_sentiments_for_dish/30 which translates to 
                                    
                                    dish.get("positives")/total_sentiments_for_all_dish *
                                    *last_30_days_positive_sentiments_for_dish/30*category_positive_factor 
                
                mention_factor = total_sentiments_for_dish/total_sentiments_for_all_dish * last_30_days_all_sentiments_for_dish/30*
                                category_factor overall or positive *category_overall_factor
                """
                
                total_positive_sentiments  = sum([__object.get("positive") + __object.get("super-positive") -\
                                            __object.get("negative") - __object.get("super-negative") for __object in __object_list])
                total_sentiments  = sum([__object.get("total_sentiments") for __object in __object_list])

                        
                for __object in __object_list:
                        try:

                                trending_factor = self.trending_sentiment_factor(__object, total_positive_sentiments)*\
                                            self.moving_average_last_30_days(__object, self.last_30_date)*\
                                            category_positive_factor*1000
                        
                        except ZeroDivisionError:
                                trending_factor = 0

                        try:
                                mention_factor = self.mentioned_sentiment_factor(__object, total_sentiments)*\
                                                self.moving_average_last_30_days(__object, self.last_30_date)*\
                                                category_overall_factor*1000
                        except ZeroDivisionError:
                                mention_factor = 0


                        __object.update({"trending_factor": trending_factor, "mention_factor": mention_factor})


                return __object_list

        def if_object(self, __object, category_overall_factor, category_positive_factor):
                """
                Valid for every other category other than "dishes", "sub-food", "place-food", 
                ambience, cost, service
                """
                
                result = {}
                keys = __object.keys()

                ##total_positive_sentiments is the percentage of positive $ super-positive sentiments given to the object 
                total_positive_sentiments  = sum([__object.get(key)["positive"] + __object.get(key)["super-positive"]\
                                                - __object.get(key)["negative"] - __object.get(key)["super-negative"] for key in keys])
                total_sentiments  = sum([__object.get(key).get("total_sentiments") for key in keys])
                
                for name, value_dict in __object.iteritems():
                        try:
                                trending_factor = self.trending_sentiment_factor(value_dict, total_positive_sentiments)*\
                                            self.moving_average_last_30_days(value_dict, self.last_30_date)*\
                                            category_positive_factor*1000
                         
                        except ZeroDivisionError:
                                trending_factor = 0
                        try:
                                mention_factor = self.mentioned_sentiment_factor(value_dict, total_sentiments)*\
                                                self.moving_average_last_30_days(value_dict, self.last_30_date)*\
                                                category_overall_factor*1000  
                        except ZeroDivisionError:
                                mention_factor = 0
                        print trending_factor, mention_factor

                        value_dict.update({"trending_factor": trending_factor, "mention_factor": mention_factor})
                        result[name] = value_dict
                return result

        def if_single_object(self, __object, category_overall_factor, category_positive_factor):
                """
                Valid if the object given to it is a single object like the one stored in 
                overall-food and menu-food
                """
                total_positive_sentiments  = __object.get("positive") + __object.get("super-positive") -\
                                            __object.get("negative") - __object.get("super-negative")

                total_sentiments  = __object.get("total_sentiments")

                        
                try:
                        trending_factor = self.trending_sentiment_factor(__object, total_positive_sentiments)*\
                                            self.moving_average_last_30_days(__object, self.last_30_date)*\
                                            category_positive_factor*1000
                except ZeroDivisionError:
                        trending_factor = 0

                try:
                        mention_factor = self.mentioned_sentiment_factor(__object, total_sentiments)*\
                                                self.moving_average_last_30_days(__object, self.last_30_date)*\
                                                category_overall_factor*1000
                except ZeroDivisionError:
                        mention_factor = 0

                __object.update({"trending_factor": trending_factor, "mention_factor": mention_factor})


                return __object










        def moving_average_last_30_days(self, __object, last_30_date):
                """
                returns the fraction of this __object being mentioned in reviews in last month to the total number of days in the last month
                which is 30
                so lets say the dish cheesecake mentioned 45 times in last month then this factr is 45/30 = 1.5
                """

                ##last_30_day_sentiments gives out the total number of sentiments that comes up in last MOVING_AVEARGE_DAYS days
                ##the idea behind is the dish which is very popular recently will have higher moving average
                ##as compared to the dish that was popular
                last_30_day_sentiments = len([time_stamp.split(" ")[0] for  (sentiment, time_stamp) in __object.get("timeline") \
                                                                if time_stamp.split(" ")[0] >= self.last_30_date])
                average = float(last_30_day_sentiments)/MOVING_AVEARGE_DAYS
                return average

        def trending_sentiment_factor(self, __object, total_positive_sentiments):
                """
                share of this __object positive sentiment in the overall positive sentiments given by the 
                people
                """
                try:
                        ##absolte positivity for the dish , ensured by subtracting negative snetiments for the dish
                        ##from the total positives
                        ratio = float(__object.get("positive") + __object.get("super-positive") - __object.get("negative") \
                                        - __object.get("super-negative"))/total_positive_sentiments
                except Exception as e:
                        ratio =  0.00
                return ratio
        
        def negative_sentiment_factor(self, __object, total_positive_sentiments):
                """
                share of this __object positive sentiment in the overall positive sentiments given by the 
                people
                """
                try:
                        ratio = float(__object.get("negative")+ __object.get("super-negative"))/total_positive_sentiments
                except Exception as e:
                        ratio =  0.00
                return ratio

        def mentioned_sentiment_factor(self, __object, total_sentiments):
                """
                share of this __object total sentiments in the overall sentiments given by the 
                people
                """
                return float(__object.get("total_sentiments"))/total_sentiments

        def per_eatery():

                """
                Gives away the number of days from which the reviews for the this eatery
                started popping out
                """
                pass


if __name__ == "__main__":
        instance = NormalizingFactor(eatery_id="308322")
        instance.run()
        result = instance.result_dict
        for element in result["food"]["dishes"][0: 10]:
                print element.get("name"), [element.get(e) for e in ["super-negative", "negative", "positive", "super-positive","neutral"]], element.get("trending_factor"), element.get("mention_factor")
        
        print "\n\n"                
        for element in result["food"]["dishes"][-10:]:
                print element.get("name"), [element.get(e) for e in ["super-negative", "negative", "positive", "super-positive","neutral"]], element.get("trending_factor"), element.get("mention_factor")

        for name, key in result["ambience"].iteritems():
                print name, [key.get(e) for e in ["super-negative", "negative", "positive", "super-positive","neutral"]], key.get("trending_factor"), key.get("mention_factor")




