#!/usr/bin/env python
"""
Author: kaali
Date: 16 May, 2015
Purpose: Final celery code to be run with tornado
"""
import pymongo
import os
import sys
import warnings
import itertools
import geocoder
import blessings

file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_path)
sys.path.append(parent_dir)

os.chdir(parent_dir)
from configs import reviews, eateries, r_reviews, r_eateries, r_junk_nps, r_clip_eatery
os.chdir(file_path)
Terminal = blessings.Terminal()


class MongoScriptsReviews(object):
        """
        if eatery_id not present in the r_eateries implies processing for the first time
        if eatery_id is present
                if key google is present or its value 0
        """




        @staticmethod
        def insert_eatery_into_results_collection(eatery_id):
                """
                First time when eatery will bei nserted into the r_eateries
                By default, if you are running a eatery, this will flush the eatery and update the result
                again 
               
                try:
                        google = r_eateries.find_one({"eatery_id": eatery_id}).get("google") 
                except Exception as e:
                        print e
                        google = None
                
                print google
                """
                r_eateries.remove({"eatery_id": eatery_id}) 

                eatery = eateries.find_one({"eatery_id": eatery_id}, {"_id": False, "__eatery_id": True, "eatery_id": True, "eatery_name": True, \
                        "eatery_address": True, "location": True,\
                        "eatery_area_or_city": True, "eatery_cost": True,\
                        "pictures": True, "eatery_url": True})

                latitude, longitude = eatery.pop("location")
                latitude, longitude = float(latitude), float(longitude)
                location = [latitude, longitude]
             
                if int(latitude) == 0:
                        g  = geocoder.google(eatery.get("eatery_address"))
                        longitude, latitude = g.geojson.get("geometry").get("coordinates")
                        location = [latitude, longitude]
                        print "eatery_id <<%s>> has not lcoation, founded by google is <<%s>>"%(eatery_id, location)
                        eateries.update({"eatery_id": eatery_id}, {"$set": {"location": location}}, upsert=False)
                eatery.update({"location": location})

                print r_eateries.insert(eatery)
                return 

        @staticmethod
        def reviews_with_text(reviews_ids, eatery_id):
                review_list = list()
                for review_id in reviews_ids:
                        review = reviews.find_one({"review_id": review_id})
                        review_text, review_time = review.get("review_text"), review.get("review_time")
                        
                        if bool(review_text) and review_text != " ":
                                review_list.append((review_id, review_text, review_time, eatery_id))
                print len(review_list)
                return review_list

        @staticmethod
        def update_review_result_collection(**kwargs):
                review_id = kwargs["review_id"]
                eatery_id = kwargs["eatery_id"]
                review = reviews.find_one({"review_id": review_id}, {"_id": False, "review_text": True, \
                        "converted_epoch": True, "review_time": True})    
                kwargs.update({"review_text": review.get("review_text"), "review_time": review.get("review_time")})

                print r_reviews.update({"review_id": review_id}, {"$set": kwargs}, upsert=True, multi=False)
                
                return 



        @staticmethod
        def flush_eatery(eatery_id):
                print r_eateries.remove({"eatery_id": eatery_id})
                print r_reviews.remove({"eatery_id": eatery_id})
                return 


        @staticmethod
        def update_eatery_places_cusines(eatery_id, places, cuisines):
            if places or cuisines:    
                    r_eateries.update({"eatery_id": eatery_id}, {"$push": \
                        {"places": places, "cuisines": cuisines }}, upsert=False)

       
 
        @staticmethod
        def review_ids_to_be_processed(eatery_id):
                total_reviews = [e.get("review_id") for e in reviews.find({"eatery_id": eatery_id}, {"review_id": True, "_id": False})]

                reviews_in_results = [e.get("review_id") for e in r_reviews.find({"eatery_id": eatery_id}, {"review_id": True, "_id": False})]
                review_ids = list(set.symmetric_difference(set(total_reviews), set(reviews_in_results)))
                return review_ids




class MongoScriptsDoClusters(object):
        def __init__(self, eatery_id):
                self.eatery_id = eatery_id
                self.eatery_name = eateries.find_one({"eatery_id": self.eatery_id}).get("eatery_name")
                self.eatery_address = eateries.find_one({"eatery_id": self.eatery_id}).get("eatery_address")




        @staticmethod
        def reviews_with_time(review_list):
                for review_id in review_list:
                    result = reviews.find_one({"review_id": review_id}, {"_id": False, "review_time": True})
                    print "The review_id --<<{0}>>-- with review time --<<{1}>>--".format(review_id, result.get("review_time")), "\n"


        def if_no_reviews_till_date(self):
            """
            If no reviews has been written for this eatery till date, 
            which implies that there is no need to run doclusters 
            """
            return reviews.find({"eatery_id": self.eatery_id}).count()



        def processed_clusters(self):
                """
                This returns all the noun phrases that already have been processed for
                teh eatery id 
                """
                return r_eateries.find_one({"eatery_id": self.eatery_id}, 
                            {"_id": False, noun_phrases: True}).get("noun_phrases")

        def old_considered_ids(self):
                """
                Returns review_ids whose noun_phrases has already been taken into account, 
                which means Clusteirng algorithms has already been done on the noun phrases 
                of these review ids and is stored under noun_phrases key of eatery
                nd these review ids has been stored under old_considered_ids
                """
                try:
                        old_considered_ids = r_eateries.find_one({"eatery_id": self.eatery_id}, {"_id": False, 
                            "old_considered_ids": True}).get("old_considered_ids")
                except Exception as e:
                        print e
                        old_considered_ids = None
                return old_considered_ids
                        

        def places_mentioned_for_eatery(self):
                result = r_eateries.find_one({"eatery_id": self.eatery_id}, {"places": True}).get("places")

                if result:
                    return [place_name for place_name in result if place_name] 

                return []


        def processed_reviews(self):
                return r_eateries.find_one({"eatery_id": self.eatery_id}, {"_id": False, 
                            "processed_reviews": True}).get("processed_reviews")

        def fetch_reviews(self, category, review_list=None):
                if not review_list:
                        review_list = [review.get("review_id") for review in r_reviews.find({"eatery_id": self.eatery_id})]


                if category == "food":
                        food = [r_reviews.find_one({"review_id": review_id})["food_result"] for review_id in  review_list]
                        flatten_food = list(itertools.chain(*food))
                        
                        return flatten_food          
                        
                
                if category == "overall":
                        result = [r_reviews.find_one({"review_id": review_id})[category] for review_id in review_list]
                        result = list(itertools.chain(*result))
                        return result
                    
                if category == "menu_result":
                        result = [r_reviews.find_one({"review_id": review_id})[category] for review_id in review_list]
                        result = list(itertools.chain(*result))
                        return result
       
                    
                result = [r_reviews.find_one({"review_id": review_id})[category] for review_id in review_list]
                
                result = list(itertools.chain(*result))
                #[[u'super-positive', u'ambience-overall'], [u'super-positive', u'ambience-overall'], 
                #[u'neutral', u'ambience-overall']]
                return [[sentiment, sub_tag, review_time] for (sent, tag, sentiment, sub_tag, review_time) in result]


        def update_food_sub_nps(self, np_result, category):
                if category == "dishes":    
                        nps = np_result["nps"]
                        nps = sorted(nps, reverse=True, key= lambda x: x.get("total_sentiments"))
                        excluded_nps = np_result["excluded_nps"]
                        dropped_nps = np_result["dropped_nps"]

                        ##to be inserted in r_clip_eatery
                        
                        try:
                                r_eateries.update({"eatery_id": self.eatery_id}, {"$set": \
                                        {"food.{0}".format(category): nps[0: 25]}}, upsert=False)
                                
                                r_eateries.update({"eatery_id": self.eatery_id}, {"$set": \
                                        {"food.more_{0}".format(category): nps[25:]}}, upsert=False)
                                
                                
                                
                                r_eateries.update({"eatery_id": self.eatery_id}, {"$set": \
                                        {"dropped_nps": dropped_nps}}, upsert=False) 
                                r_junk_nps.update({"eatery_id": self.eatery_id}, \
                                        {"$set": {"excluded_nps": excluded_nps}}, upsert=True) 
                        except Exception as e:
                                print e

                        
                        
                        short_nps = list()
                        for dish in nps[0:3]:
                                dish.pop("similar")
                                dish.pop("timeline")
                                short_nps.append(dish)
                        print "udpating clip eatery with top three dishes data"
                        r_clip_eatery.update({"eatery_id": self.eatery_id}, {"$set": \
                                        {"food.{0}".format(category):short_nps}}, upsert=True)


                        return
            
           
                ##this is meant if category overall-food
                try:    
                        r_eateries.update({"eatery_id": self.eatery_id}, {"$set": {
                        "food.{0}".format(category): np_result}}, upsert=False)
                        
                except Exception as e:
                        raise StandardError(e)
                return 
        
        def fetch_nps_frm_eatery(self, category, sub_category=None):
                if category == "food":
                        if not sub_category:
                                raise StandardError("Food Category shall be provided to fecth nps")
                        return r_eateries.find_one({"eatery_id": self.eatery_id}).get(category).get(sub_category)
                        
                return r_eateries.find_one({"eatery_id": self.eatery_id}).get(category)

        def update_nps(self, category, category_nps):
                """
                Update new noun phrases to the eatery category list
                """
                print "Updating nps for %s"%category
                try:
                    r_eateries.update({"eatery_id": self.eatery_id}, {"$set": {category:
                    category_nps}}, upsert=False)
                except Exception as e:
                    raise StandardError(e)

                
                
                if category == "overall":
                        print "Updating nps for %s in clip_eatery"%category
                        category_nps.pop("timeline")
                        r_clip_eatery.update({"eatery_id": self.eatery_id},
                                {"$set": {category: category_nps}}, upsert=False)
                        return 

                if category == "menu":
                        print "Updating nps for %s in clip_eatery"%category
                        category_nps.pop("timeline")
                        r_clip_eatery.update({"eatery_id": self.eatery_id},
                                {"$set": {category: category_nps}}, upsert=False)
                        return 
                
                
                try:
                        ##this is the dubcategory dict which has the highest total sentiment int he category
                        modifed_category_nps = dict()
                        print "Updating nps for %s in clip_eatery"%category
                        for (key, value) in category_nps.iteritems():
                                    if key.endswith("null"):
                                            pass
                                    else:
                                            modifed_category_nps.update({key: value})

                        __dict = [(key, value) for (key, value) in sorted(modifed_category_nps.iteritems(), reverse=True, key= lambda (k,v): v.get("total_sentiments") )][0]
                        sub_category = __dict[0]
                        sub_category_data = __dict[1]

                        sub_category_data.pop("timeline")
                        r_clip_eatery.update({"eatery_id": self.eatery_id},
                                {"$set": {"%s.%s"%(category,
                                    sub_category):sub_category_data}}, upsert=False)
                        

                except Exception as e:
                    print category
                    print category_nps
                    raise StandardError(e)

                return 








