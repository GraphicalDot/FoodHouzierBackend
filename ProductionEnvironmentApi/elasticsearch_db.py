#!/usr/bin/env python
"""
https://gist.github.com/lightsuner/5df39112b8507d15ede6
https://gist.github.com/lukas-vlcek/5143799
http://192.168.1.5:9200/_cluster/state?pretty&filter_nodes=true&filter_routing_table=true&filter_indices=dishes

Author: kaali
Dated: 9 June, 2015
"""

import requests
import time
import os
import sys
from itertools import ifilter
from compiler.ast import flatten
import ConfigParser
import json
file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir_path = os.path.dirname(file_path)
sys.path.append(parent_dir_path) 


from elasticsearch import RequestError

NUMBER_OF_DOCS = 10
#localhost:9200/test/_analyze?analyzer=whitespace' -d 'this is a test'
#curl -XGET "http://192.168.1.4:9200/food/_analyze?analyzer=custom_analyzer&pretty=1" -d "cheesecake"


os.chdir(parent_dir_path)
from connections import ES_CLIENT, eateries_results_collection, bcolors, eateries, ELASTICSEARCH_IP
os.chdir(file_path)






##These cateories shouldnt contain null values for any category
sentiment_tags =   ["good", "poor", "average", "excellent", "terrible", "mixed"]
food_tags =  ["dishes",  "overall-food"]
ambience_tags = ['smoking-zone', 'decor', 'ambience-overall', 'in-seating', 'crowd', 'open-area', 'dancefloor', 'music', 'location', 'romantic', 'sports', 'live-matches', 'view', "bowling"]
cost_tags =  ["vfm", "expensive", "cheap", "not worth"]
service_tags = [u'management', u'service-charges', u'service-overall', u'service-null', u'waiting-hours', u'presentation', u'booking', u'staff'] 


"""

 address_mappings = {"area":                        
                        {"properties": {
                            "address_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'},
                            "address": {"type": "string", "copy_to": ["address_autocomplete"]},

                            "location": {"type": "string"}}}}




body = {"_source": ["address"],
                        "from": 0,
                        "size": 50,
                                "query": {
                                        "match": {
                                                "address_autocomplete": {
                                                        "query":    "new delhi",
                                                        "analyzer": "standard"
                                                                    }
                                                }
                                        }
                            }




"""






class ElasticSearchScripts(object):
        def __init__(self, renew_indexes=False):
                """
                Will have minimum two indexes
                one for dishes and one for eateries
                FOOD_SUB_TAGS = ['dishes', 'menu-food', 'null-food', 'overall-food', 'place-food', 'sub-food']
                COST_SUB_TAGS = ['cheap', 'cost-null', 'expensive', 'not worth', 'value for money']
                SERV_SUB_TAGS = ['booking', 'management', 'presentation', 'service-null', 'service-overall',\
                'staff', 'waiting-hours']

                AMBI_SUB_TAGS = [u'ambience-null', 'ambience-overall', 'crowd', 'dancefloor', 'decor', \'
                in-seating', 'music', 'open-area', 'romantic', 'smoking-zone', 'sports', 'sports-screens', 'view']

                Index:
                        eatery
                                __type: menu
                                __type: overall
                                __type: eatery
                                __type: cuisine
                        food:
                                _type: dishes
                                _type: overall-food
                        cost:
                                __type:
                                __type:
                                __type:
                                __type:
                        ambience:
                                __type: ambience-overall
                                __type: crowd
                                __type: dancefloor
                                __type: decor
                                __type: in-seating
                                __type: music
                                __type: open-area
                                __type: romantic
                                __type: smoking-zone
                                __type: sports
                                __type: location
                                __type: view
                                __type: live-matches
                        service:
                                __type: management
                                __type: service-overall
                                __type: service-charges
                                __type: serivce-null
                                __type: booking
                                __type: service-null
                                __type: waiting-hours
                                __type: presentation
                                __type: servic-overall
                                __type: staff


                """
                self.settings = {'settings': 
                                    {'analysis': 
                                            {'analyzer': 
                                                    {'custom_analyzer': {
                                                                'filter': ['lowercase', 'asciifolding'],
                                                                'tokenizer': 'ngram_tokenizer',
                                                                'type': 'custom'},
                                                                
                                                        'keyword_analyzer': {
                                                            'filter': ['lowercase', 'asciifolding'],
                                                            'tokenizer': 'keyword',
                                                            'type': 'custom'},
                                                    
                                                        #'phonetic_analyzer': {
                                                         #       'filter': ['lowercase', 'asciifolding', 'standard', 'custom_metaphone'],
                                                          #      'tokenizer': 'whitespace',
                                                           #     'type': 'custom'},
                                                        'shingle_analyzer': {
                                                                'filter': ['lowercase', 'asciifolding', 'shingle_tokenizer'],
                                                                'tokenizer': 'ngram_tokenizer',
                                                                'type': 'custom'},
                                                            
                                                        'standard_analyzer': {
                                                                    'filter': ['lowercase', 'asciifolding'],
                                                                    'tokenizer': 'standard',
                                                                    'type': 'custom'}
                                                        },
                                                                 
                                                    'filter': {
                                                            #'custom_metaphone': {'encoder': 'metaphone',
                                                             #           'replace': False,
                                                              #          'type': 'phonetic'},
                                                            'shingle_tokenizer': {'max_shingle_size': 5,
                                                                                    'min_shingle_size': 2,
                                                                                    'type': 'shingle'}
                                                            },
                                                                                    
                                                    'tokenizer': {
                                                            'limited_tokenizer': {
                                                                        'max_gram': '10',
                                                                        'min_gram': '2',
                                                                        'token_chars': ['letter', 'digit'],
                                                                        'type': 'edgeNGram'},
                                                                                                           
                                                            'ngram_tokenizer': {'max_gram': 100,
                                                                                'min_gram': 2,
                                                                                'token_chars': ['letter', 'digit'],
                                                                                'type': 'edgeNGram'}
                                                            }
                                                    }
                                                    }
                                        }






                self.other_mappings = {'properties': {'eatery_id': {'type': 'string'},
                                                    '__eatery_id': {'type': 'string'},
                                                    'eatery_name': {'type': 'string'},
                                    'location': {'type': 'geo_point',
                                                     "geohash": True,
                                                    "geohash_prefix": True,
                                                    "geohash_precision": 2,
                                        
                                        },
                                                    'mention_factor': {'type': 'double'},
                                                    'poor': {'type': 'long'},
                                                    'average': {'type': 'long'},
                                                    'good': {'type': 'long'},
                                                    'terrible': {'type': 'long'},
                                                    'excellent': {'type': 'long'},
                                                    'timeline': {'type': 'string'},
                                                    'total_sentiments': {'type': 'long'},
                                                    'trending_factor': {'type': 'double'}}}



                self.eatery_mappings = {'eatery':  
                    {"properties":{
                            '__eatery_id': {'index': 'not_analyzed', 'type': 'string'},
                            'eatery_id': {'index': 'not_analyzed', 'type': 'string'},
                            "eatery_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'}, 
                            "address_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'}, 
                            "eatery_cuisine_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'},
                            'eatery_raw': {'analyzer': 'keyword_analyzer', 'type': 'string'},
                            'eatery_shingle': {'analyzer': 'shingle_analyzer', 'type': 'string'},
                            "cuisines":{"type":"string"},
                            "eatery_address":{"copy_to": ["address_autocomplete"], "type":"string"},
                            "eatery_area_or_city":{"type":"string"},
                            "eatery_cuisine":{"type":"string"},
                            "eatery_highlights":{"type":"string"},
                            "eatery_name":{"type":"string"},
                            
                            "eatery_type":{"type":"string", 'copy_to': 'eatery_cuisine_autocomplete'},
                            
                            "eatery_url":{"type":"string"},
                                    'location': {'type': 'geo_point',
                                                     "geohash": True,
                                                    "geohash_prefix": True,
                                                    "geohash_precision": 2,
                                        
                                        },
                            "mention_factor":{"type":"double"},
                            "timeline":{"type":"string"},
                            "trending_factor":{"type":"double"},
                            "eatery_address_split": {"type": "string"},
                            'eatery_name': {'copy_to': ['eatery_shingle', 'eatery_raw', 'eatery_autocomplete'],
                                                        'type': 'string'},
                            'eatery_cuisine_split': {'copy_to': ['eatery_cuisine_autocomplete'],
                                                        'type': 'string'},
                            'eatery_address_split': {'copy_to': ['address_autocomplete'],
                                                        'type': 'string'},
                                
                            }}}


                self.cuisines_mappings = {"cuisines":
                        {"properties": {
                            "cuisines_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'}, 
                            "name": {"type": "string", "copy_to": ["cuisines_autocomplete"]},

                            }}}

                """
                if not ES_CLIENT.indices.exists("food"):
                        self.prep_food_index()
                        
                
                if not ES_CLIENT.indices.exists("ambience"):
                        self.prep_ambience_index()
                
                if not ES_CLIENT.indices.exists("cost"):
                        self.prep_cost_index()
                
                if not ES_CLIENT.indices.exists("service"):
                        self.prep_service_index()
                """
                if renew_indexes:
                        self.prep_food_index()
                        
                        print "{0}Deleting INDEX=<<ambience>> {1}".format(bcolors.FAIL, bcolors.RESET)
                        try:
                                ES_CLIENT.indices.delete(index="ambience")
                        except Exception as e:
                                print e
                        
                        print "{0}Deleting INDEX=<<cost>> {1}".format(bcolors.FAIL, bcolors.RESET)
                        try:
                                ES_CLIENT.indices.delete(index="cost")
                        except Exception as e:
                                print e
                        
                        
                        print "{0}Deleting INDEX=<<service>> {1}".format(bcolors.FAIL, bcolors.RESET)
                        try:
                                ES_CLIENT.indices.delete(index="service")
                        except Exception as e:
                                print e
                        
                        print "{0}Deleting INDEX=<<eatery>> {1}".format(bcolors.FAIL, bcolors.RESET)
                        try:
                                ES_CLIENT.indices.delete(index="eatery")
                        except Exception as e:
                                print e

                        print "{0}Creating INDEX=<<ambience>> {1}".format(bcolors.OKGREEN, bcolors.RESET)
                        self.prep_ambience_index()
                        
                        print "{0}Creating INDEX=<<cost>> {1}".format(bcolors.OKGREEN, bcolors.RESET)
                        self.prep_cost_index()
                        
                        print "{0}Creating INDEX=<<service>> {1}".format(bcolors.OKGREEN, bcolors.RESET)
                        self.prep_service_index()
                        
                        
                        print "{0}Creating INDEX=<<eatery>> {1}".format(bcolors.OKGREEN, bcolors.RESET)
                        self.prep_eatery_index()
                return 

        def prep_eatery_index(self):
                """
                In eatery index there exists four doc_type
                    cuisines
                    eatery
                    menu
                    overall
                """
            
            
                ES_CLIENT.indices.create(index="eatery", body=self.settings)
                ES_CLIENT.indices.put_mapping(index="eatery", doc_type="eatery", body = self.eatery_mappings)
                print "{0}Mappings updated for  {1}  {2}".format(bcolors.OKGREEN, "eatery",  bcolors.RESET)
                for __sub_category in ["overall", "menu"]:
                                ES_CLIENT.indices.put_mapping(index="eatery", doc_type=__sub_category, body = {__sub_category: self.other_mappings })
                                print "{0}Mappings updated for  {1}  {2}".format(bcolors.OKGREEN, __sub_category, bcolors.RESET)
                
                ##Inserting cusines names into cuisines index
                cuisines_list = list()
                ES_CLIENT.indices.put_mapping(index="eatery", doc_type="cuisines", body =self.cuisines_mappings)
                for post in eateries_results_collection.find():
                        try:
                                cuisines = post.get("eatery_cuisine")
                                eatery_cuisine_split = cuisines.split(",")
                                cuisines_list.extend(eatery_cuisine_split)
                                                                 
                        except Exception as e:       
                                print e, "In finding cusines"
                                pass
                __cuisines_list = [e.replace("Quick", "").replace("Bites", "").replace("Cuisines:", "").lstrip()  for e in set(cuisines_list)]

                for cuisine in __cuisines_list:
                                print "Updating cuisine %s"%cuisine
                                ES_CLIENT.index(index="eatery", doc_type="cuisines", body={"name": cuisine})

                
                ES_CLIENT.indices.put_mapping(index="eatery", doc_type="menu", body = {"menu": self.other_mappings })
                ES_CLIENT.indices.put_mapping(index="eatery", doc_type="overall", body = {"overall": self.other_mappings })
                
                return 


        
        def prep_ambience_index(self):
                ES_CLIENT.indices.create(index="ambience")
                #for __sub_category in json.loads(config.get("categories", "ambience_tags")):
                for __sub_category in ambience_tags:
                                ES_CLIENT.indices.put_mapping(index="ambience", doc_type=__sub_category, body = {__sub_category: self.other_mappings })
                                print "{0}Mappings updated for  {1}  {2}".format(bcolors.OKGREEN, __sub_category, bcolors.RESET)
            
        def prep_cost_index(self):
                ES_CLIENT.indices.create(index="cost")
                #for __sub_category in json.loads(config.get("categories", "cost_tags")):
                for __sub_category in cost_tags:
                                ES_CLIENT.indices.put_mapping(index="cost", doc_type=__sub_category, body = {__sub_category: self.other_mappings })
                                print "{0}Mappings updated for {1} {2}".format(bcolors.OKGREEN, __sub_category, bcolors.RESET)
        
            
        def prep_service_index(self):
                ES_CLIENT.indices.create(index="service")
                #for __sub_category in json.loads(config.get("categories", "service_tags")):
                for __sub_category in service_tags:
                                ES_CLIENT.indices.put_mapping(index="service", doc_type=__sub_category, body = {__sub_category: self.other_mappings })
                                print "{0}Mappings updated for {1} {2}".format(bcolors.OKGREEN, __sub_category, bcolors.RESET)

        def prep_food_index(self):
                """
                Two indexes will be created one for the restaurants and one for the dishes
                index:
                        eateries
                            type:
                                eatery
                        dishes:
                                dish
                """
                try:
                        ES_CLIENT.indices.delete(index="food")
                        print "{0} DELETING Index=<<food>>{1}".format(bcolors.OKGREEN, bcolors.RESET)
                except Exception as e:
                        print "{0} {1}".format(bcolors.FAIL, bcolors.RESET)
                        print "{0} Index Food doesnt exists {1}".format(bcolors.FAIL, bcolors.RESET)
                        print e



                print "{0}Settings updated {1}".format(bcolors.OKGREEN, bcolors.RESET)

                ES_CLIENT.indices.create(index="food", body=self.settings)
                __mappings = {'dishes': {'_all': {'enabled': True},
                          'properties': {
                                   # 'dish_phonetic': {'analyzer': 'phonetic_analyzer', 'type': 'string'},
                                    'dish_raw': {'analyzer': 'keyword_analyzer', 'type': 'string'},
                                    'dish_shingle': {'analyzer': 'shingle_analyzer', 'type': 'string'},
                                    "dish_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'}, 
                                        
                                    'eatery_id': {'index': 'not_analyzed', 'type': 'string'},
                                    'eatery_name': {'copy_to': ['eatery_shingle', 'eatery_raw', 'eatery_autocomplete'],
                                                        'type': 'string'},
                                    
                                    "eatery_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'}, 
                                    'eatery_raw': {'analyzer': 'keyword_analyzer', 'type': 'string'},
                                    'eatery_shingle': {'analyzer': 'shingle_analyzer', 'type': 'string'},
                                    'location': {'type': 'geo_point',
                                                     "geohash": True,
                                                    "geohash_prefix": True,
                                                    "geohash_precision": 2,
                                        
                                        },
                                    'mention_factor': {'type': 'double'},
                                    'name': {'copy_to': ['dish_raw', 'dish_shingle', 'dish_phonetic', 'dish_autocomplete'],
                                    'type': 'string'},
                                    'poor': {'type': 'long'},
                                    'average': {'type': 'long'},
                                    'good': {'type': 'long'},
                                    'similar': {'properties': {'name': {'copy_to': ['dish_raw', 'dish_shingle',  'dish_autocomplete'],
                                    'type': 'string'}, 'poor': {'type': 'long'},
                                    'average': {'type': 'long'},
                                    'good': {'type': 'long'},
                                            'terrible': {'type': 'long'},
                                                                                                                          'excellent': {'type': 'long'},
                                                                                                                               'timeline': {'type': 'string'}}},
                                                                            'terrible': {'type': 'long'},
                                                                               'excellent': {'type': 'long'},
                                                                                  'timeline': {'type': 'string'},
                                                                                     'total_sentiment': {'type': 'integer'},
                                                                                        'trending_factor': {'type': 'double'}}}}

                    
                    





                    

                try:
                        ES_CLIENT.indices.put_mapping(index="food", doc_type="dishes", body = __mappings)
                        
                        ##also overall-food mapings should also be there
                        ES_CLIENT.indices.put_mapping(index="food", doc_type="overall-food", body = {"overall-food": self.other_mappings})
                        print "{0}Mappings updated {1}".format(bcolors.OKGREEN, bcolors.RESET)
                
                
                except Exception as e:
                        print "{0}Mappings update Failed with error {1} {2}".format(bcolors.FAIL, e, bcolors.RESET)


                """
                test_doc = {'eatery_name': u"Karim's", u'name': u'chicken korma', u'terrible': 0, u'poor': 0, u'excellent': 1, 
                        u'average': 6, u'timeline': [[u'good', u'2013-01-24 17:34:01'], [u'average', u'2014-07-18 23:49:05'], 
                        [u'average', u'2014-06-05 13:30:14'], [u'excellent', u'2013-07-04 17:03:37'], 
                        [u'average', u'2013-04-18 20:40:35'], [u'average', u'2013-01-18 23:04:17'], 
                        [u'average', u'2013-01-11 14:04:49'], [u'average', u'2012-12-29 21:51:43']], 'eatery_id': '463', 
                        u'similar': [{u'name': u'chicken qorma', u'good': 1, u'poor': 0, u'excellent': 0, u'average': 0, 
                            u'timeline': [[u'good', u'2013-01-24 17:34:01']], u'terrible': 0}, 
                            {u'name': u'chicken korma', u'good': 0, u'poor': 0, u'excellent': 1, u'average': 5, 
                                u'timeline': [[u'average', u'2014-07-18 23:49:05'], [u'average', u'2014-06-05 13:30:14'], 
                                    [u'excellent', u'2013-07-04 17:03:37'], [u'average', u'2013-04-18 20:40:35'], 
                                    [u'average', u'2013-01-18 23:04:17'], [u'average', u'2013-01-11 14:04:49']], u'terrible': 0}, 
                                {u'name': u'i order chicken korma', u'good': 0, u'poor': 0, u'excellent': 0, u'average': 1, 
                                    u'timeline': [[u'average', u'2012-12-29 21:51:43']], u'terrible': 0}], u'good': 1}
                
                
                print "{0}Updating test data {1}".format(bcolors.OKGREEN, bcolors.RESET)
                l = ES_CLIENT.index(index="food", doc_type="dishes", body=test_doc)

                print "{0}Result:\n {1} {2}".format(bcolors.OKGREEN, l, bcolors.RESET)

                __body = {"query" : {
                            "term" : { "_id" : l.get("_id")}
                                }}

                print "{0}Test Doc deleted {1}".format(bcolors.OKGREEN, bcolors.RESET)
                ES_CLIENT.delete_by_query(index="food", doc_type="dishes", body=__body)

                print "{0}_________ Index Dishes ready to be used _______________{1}".format(bcolors.OKGREEN, bcolors.RESET) 
                """
                return 

        @staticmethod
        def insert_eatery(eatery_id):
                """
                This method deasl with updating or inserting th eatery data into 
                Deals only with four categories food, ambience, service, cost
                
                keys that are present in eatery are:
                    [u'eatery_type', u'cuisines', u'processed_reviews', u'__eatery_id', u'cost', u'dropped_nps', u'eatery_longitude_latitude', \
                            u'service', u'menu', u'eatery_id', u'eatery_highlights', u'eatery_address', u'location', u'eatery_cuisine', u'ambience', \
                            u'old_considered_ids', u'eatery_trending', u'eatery_name', u'food', u'overall', u'eatery_known_for', u'eatery_url', u'places', u'eatery_area_or_city', u'_id']
                    

                Need not to store dropped_nps

                SO here it starts:
                    five indexes
                        eatery
                            this has four doctypes , menu, overall, eatery, cuisines
                        food
                            two doctypes, dishes and overallfood
                        cost
                        ambience
                        service



                """

                print "starting inserting eateryid %s"%eatery_id
                delete_body = {
                            "query" : {
                                        "term" : { "eatery_id" : eatery_id }
                                    }
                            }

                #https://people.mozilla.org/~wkahngreene/elastic/guide/reference/api/delete-by-query.html
                try:
                        ES_CLIENT.delete_by_query(index="_all", body=delete_body, consistency="quorum")
                except Exception as e:
                        print "{0}Error occurred while deleting all the entries for the eatery_id {1}{2}".format(\
                                bcolors.FAIL, eatery_id, bcolors.RESET)
                
                
                #Elasticsearch refreshes every index at an interval of 1 second, We need to
                #refresh it for every update
                r = requests.post("http://{0}:9200/_refresh".format(ELASTICSEARCH_IP))
                

                eatery = eateries_results_collection.find_one({"eatery_id": eatery_id})

                __eatery_id = eatery["__eatery_id"]
                eatery_id = eatery["eatery_id"]
                eatery_name = eatery.get("eatery_name")
                food_data = eatery["food"]
                ambience_data = eatery["ambience"]
                cost_data = eatery["cost"]
                service_data = eatery["food"]

                eatery_dict = dict()
                __list = ['eatery_type', "__eatery_id", 'cuisines',  'eatery_longitude_latitude', 'eatery_id', 'eatery_highlights', 'eatery_address', \
                        'eatery_cuisine', 'eatery_trending', 'eatery_name', "eatery_url", 'eatery_known_for', 'eatery_area_or_city',  "location"]        
                
                ##inserting eatery in eatery index and type eatery
                for e in __list:
                        eatery_dict.update({e: eatery.get(e)})

                ##changing long lart for eatery
                try:
                        latitude, longitude = eatery_dict.pop("eatery_longitude_latitude")
                        latitude, longitude = float(latitude), float(longitude)
                except Exception as e:
                        print "eatery_longitude_latitude key not present, tying location"
                        latitude, longitude = eatery_dict.pop("location")
                        latitude, longitude = float(latitude), float(longitude)
                        
                eatery_address = eatery_dict["eatery_address"]
                eatery_address_split = eatery_address.replace(" ", "").split(",")
                
                eatery_dict.update({"location": [longitude, latitude]})
                eatery_dict.update({"eatery_address_split": eatery_address_split})

                try:
                        eatery_cuisine_split = eatery_dict.get("eatery_cuisine").replace(" ", "").split(",")
                        eatery_dict.update({"eatery_cuisine_split": eatery_cuisine_split})
                except:
                        pass

                menu_data = eatery["menu"]
                overall_data = eatery["overall"]

                menu_data.update({"eatery_id": eatery_id, "__eatery_id": __eatery_id, "eatery_name": eatery_dict["eatery_name"], "location": [longitude, latitude], \
                                "eatery_address": eatery_dict["eatery_address"]})
                
                overall_data.update({"eatery_id": eatery_id, "__eatery_id": __eatery_id, "eatery_name": eatery_dict["eatery_name"], "location": [longitude, latitude], \
                                "eatery_address": eatery_dict["eatery_address"]})



                l = ES_CLIENT.index(index="eatery", doc_type="eatery", body=eatery_dict)
                l = ES_CLIENT.index(index="eatery", doc_type="menu", body=menu_data)
                l = ES_CLIENT.index(index="eatery", doc_type="overall", body=overall_data)


                for category in ["food_tags", "cost_tags", "ambience_tags", "service_tags"]:
                        __category = category.replace("_tags", "")
                        data = eatery[__category]
                        for sub_category in eval(category):
                                
                                if sub_category  == "dishes":
                                        sub_data = data[sub_category]
                                        for __dish in sub_data:
                                                __dish.update({"eatery_id": eatery_id})
                                                __dish.update({"__eatery_id": __eatery_id})
                                                __dish.update({"eatery_name": eatery_name})
                                                #__dish.update({"location": {"lat": latitude, "lon": longitude}})
                                                __dish.update({"location": [longitude, latitude]})
                                                __dish.update({"eatery_address": eatery_address})
                                                
                                                l = ES_CLIENT.index(index=__category, doc_type=sub_category, body=__dish)
                                	        print "%s for %s for eatery_id %s for dish_name <<%s>>"%(l, sub_category, eatery_id, __dish.get("name"))
                                else:
					try:
                                                
                                        	sub_data = data[sub_category]
                                        	sub_data.update({"eatery_id": eatery_id})
                                                sub_data.update({"__eatery_id": __eatery_id})
                                                sub_data.update({"eatery_name": eatery_name})
                                                sub_data.update({"eatery_address": eatery_address})
                                                sub_data.update({"location": [longitude, latitude]})

                                        	l = ES_CLIENT.index(index=__category, doc_type=sub_category, body=sub_data)
                                	        print "%s for %s for eatery_id %s"%(l, sub_category, eatery_id)
                                        except Exception as e:
                                                print e
                                                print "Error in Eatery id %s"%eatery_id
                                        	print sub_category
                                        	pass
                print "Eatery id %s done"%eatery_id
                print "Eatery id with __eatery_id %s done\n\n"%__eatery_id
                return 
        def return_dishes(eatery_id):
                """
                Return top 10 dishes filtered on the basis of their total sentiments 
                """
                return 



        @staticmethod
        def process_result(__result):
                result = [l["_source"] for l in __result["hits"]["hits"]]
                return result
       

        @staticmethod
        def dish_suggestions(dish_name):
                
                body = {"_source": ["name"],
                        "from": 0, 
                        "size": 10, 
                                "query": {
                                        "match": {
                                                "dish_autocomplete": {     
                                                        "query":    dish_name,    
                                                        "analyzer": "standard" 
                                                                    }
                                                }
                                        }
                            }

                dish_suggestions = ES_CLIENT.search(index="food", doc_type="dishes", body=body)
                result = ElasticSearchScripts.process_result(dish_suggestions) 
                return result
        
        @staticmethod
        def eatery_suggestions(query):
                body = {"_source": ["eatery_name"],
                        "from": 0, 
                        "size": 5, 
                                "query": {
                                        "match": {
                                                "eatery_autocomplete": {     
                                                        "query":    query,    
                                                        "analyzer": "standard" 
                                                                    }
                                                }
                                        }
                            }

                dish_suggestions = ES_CLIENT.search(index="eatery", doc_type="eatery", body=body)
                result = ElasticSearchScripts.process_result(dish_suggestions) 
                return result

        @staticmethod
        def address_suggestions(query):
                body = {"_source": ["eatery_address", "__eatery_id"],
                        "from": 0, 
                        "size": 5, 
                                "query": {
                                        "match": {
                                                "address_autocomplete": {     
                                                        "query":    query,    
                                                        "analyzer": "standard" 
                                                                    }
                                                }
                                        }
                            }
                
                dish_suggestions = ES_CLIENT.search(index="eatery", doc_type="eatery", body=body)
                return ElasticSearchScripts.process_result(dish_suggestions) 


        @staticmethod
        def cuisines_suggestions(query):
                """
                search in cuisines doc type for cuisines suggestions
                returns only the name of the cuisine
                
                if a usr select particular cuisine eatery_result_on_cuisine must be called which fetched data rom mongodb to 
                shed load from elasticsearch servers
                """
                body = {"_source": ["name"],
                        "from": 0,
                        "size": 5,
                                "query": {
                                        "match": {
                                                "cuisines_autocomplete": {
                                                        "query":    query,
                                                        "analyzer": "standard"
                                                                    }
                                                }
                                        }
                            }


                address_suggestions = ES_CLIENT.search(index="eatery", doc_type="cuisines", body=body)
                return ElasticSearchScripts.process_result(address_suggestions) 
        
        
        
        @staticmethod
        def eatery_on_cuisines(cuisine_name):
                """
                returns __eatery_id's on the basis of the cuisines selected by the user
                """
                body = {"_source": ["__eatery_id", "eatery_cuisine"],
                        "from": 0,
                        "size": 10,
                        "query": {
                                        "match": {
                                                "eatery_cuisine_autocomplete": {
                                                        "query":    cuisine_name,
                                                        "analyzer": "standard"
                                                                    }
                                                }
                                        },
                            }


                eateries_on_cuisines = ES_CLIENT.search(index="eatery", doc_type="eatery", body=body)
                return ElasticSearchScripts.process_result(eateries_on_cuisines) 


        @staticmethod
        def must_eat():
                """
                """

                food_body = {"_source": ["name", "location", "eatery_name", "eatery_id", "good", "poor", "average", "excellent", "terrible", "total_sentiments"],
                        "from": 0, 
                        "size": 10, 
                        "sort": [
                                {"trending_factor" : {"order" : "desc"}}
                                ],
                        "query": {
                                "match_all": {}
                                            },
                        }

                trending_dishes = ES_CLIENT.search(index="food", doc_type="dishes", body=food_body)
                return ElasticSearchScripts.process_result(trending_dishes),  
                        



        @staticmethod
        def get_trending(latitude, longitude):
                """
                        "sort": [
                                {
                                    "_geo_distance" : {
                                    "location" : [longitude, latitude],
                                    "order" : "asc",
                                    "unit" : "km",
                                    "mode" : "min",
                                    "distance_type" : "plane"
                                    }
                                                        },

                                ],
                """

                food_body = {"_source": ["name", "location", "eatery_name", "__eatery_id", "good", "poor", "average", "excellent", "terrible", "total_sentiments"],
                        "from": 0, 
                        "size": 200, 
                        "query": {
                                "match_all": {}
                                            },
                        "filter": {
                                "geo_distance": {
                                        "distance": "20km",
                                        "location": [longitude, latitude],
                                                }     
                                },
                        "sort": [
                                {"trending_factor" : {"order" : "desc"}}
                                ],
                        }
                trending_dishes = ES_CLIENT.search(index="food", doc_type="dishes", body=food_body)
                

                ambience_body = {"_source": ["name", "location", "eatery_name", "__eatery_id", "good", "poor", "average", "excellent", "terrible", "total_sentiments", "eatery_address"],
                        "from": 0, 
                        "size": 20, 
                        "query": {
                                "match_all": {}
                                            },
                        "sort": [
                                {
                                    "_geo_distance" : {
                                    "location" : [longitude, latitude],
                                    "order" : "asc",
                                    "unit" : "km",
                                    "mode" : "min",
                                    "distance_type" : "plane"
                                    }},
                                ],
                        }
                
                trending_ambience = ES_CLIENT.search(index="ambience", doc_type="ambience-overall", body=ambience_body)
                cost_body = {"_source": ["name", "eatery_name", "location", "__eatery_id", "good", "poor", "average", "excellent", "terrible", "total_sentiments", "eatery_address"],
                        "from": 0, 
                        "size": 10, 
                        "sort": [
                                {"total_sentiments" : {"order" : "desc"}}
                                ],
                        "sort": [
                                {
                                    "_geo_distance" : {
                                    "location" : [longitude, latitude],
                                    "order" : "asc",
                                    "unit" : "km",
                                    "mode" : "min",
                                    "distance_type" : "plane"
                                    }},
                                ],
                        }
                
                trending_cost = ES_CLIENT.search(index="cost", doc_type="vfm", body=cost_body)
                
                service_body = {"_source": ["name", "eatery_name", "location", "__eatery_id", "good", "poor", "average", "excellent", "terrible", "total_sentiments", "eatery_address"],
                        "from": 0, 
                        "size": 10, 
                        "query": {
                                "match_all": {}
                                            },
                        "sort": [
                                {
                                    "_geo_distance" : {
                                    "location" : [longitude, latitude],
                                    "order" : "asc",
                                    "unit" : "km",
                                    "mode" : "min",
                                    "distance_type" : "plane"
                                    }},
                                ],
                        }
                trending_service = ES_CLIENT.search(index="service", doc_type="service-overall", body=service_body)
                return {
                        "food": sorted(ElasticSearchScripts.process_result(trending_dishes), key=lambda x: x.get("total_sentiments"), reverse=True)[0:10],  
                        "ambience": sorted(ElasticSearchScripts.process_result(trending_ambience), key=lambda x: x.get("total_sentiments"), reverse=True)[0:2], 
                        "cost": sorted(ElasticSearchScripts.process_result(trending_cost), key=lambda x: x.get("total_sentiments"), reverse=True)[0:2], 
                        "service": sorted(ElasticSearchScripts.process_result(trending_service), key=lambda x: x.get("total_sentiments"), reverse=True)[0:2], 
                        }


        def find_eatery_id(self, eatery_id):

                return 

        def suggest_eatery_name(self, suggest_eatery):
                return 



        def elastic_query_processing(self, __query_dict, number_of_dishes=None, dish_suggestions=None):
                """
                Args:
                    {'ambience': ["decor", "overall-ambience"], 'food': {'dishes': [u'Mango Cowboy', u'mango cyrup', 
                                                                    u'mango ice cream']}, 'cost': [], 'service': []}
                
                Returns:
                    {"dishes": [{"match" either a string or list, "suggestions": []}, {}, {}, {}], 
                    "ambience": [{}, {}],
                    "cost": [{}, {}],
                    "service": [{}, {}]}
                
                """
                result = {"food": {}}
                dish_names = __query_dict.get("food").get("dishes")
                dishes = []
                if dish_names:
                        for dish in dish_names:
                                dishes.append(self.suggest_dish(dish, number_of_dishes, dish_suggestions))
                

                else:
                        dishes.append({"match": "No appropriate dishes could be found related to your query", "suggestions": None})

                result["food"]["dishes"] = dishes
                return result

        @staticmethod
        def get_eatery_match(eatery_name):
                """
                When a user searched for the exact eatery or enters a eatery_name which is not present in the 
                elastic search, For this to be executed successfully
                First: We will searhc for the exact eatery name, if that fails
                First can be searched in the mongodb Itself
            
                Second : We will search search for the fuzzy match for that eatery name
                """
                def find_exact_match(eatery_name):
                        exact_eatery_search_body={"_source": ["eatery_name", "__eatery_id"],
                                    "query":{
                                            "term":{
                                                        "etery_raw":  eatery_name}},

                                    "from": 0,
                                    "size": 5,
                                }

                        __result = ES_CLIENT.search(index="eatery", doc_type="eatery", body=exact_eatery_search_body)
                        __result = ElasticSearchScripts.process_result(__result)
                        return __result

                def find_fuzzy_match(eatery_name):
                        print "Fussy match for %s"%eatery_name
                        fuzzy_search_body = {"_source": ["eatery_name", "__eatery_id"],
                                "query": {
                                    "fuzzy_like_this": {
                                            "fields": ["eatery_raw"],
							"like_text": eatery_name,  
                                                        "fuzziness": 6,
                                                        "max_query_terms": 25,
						        "boost": 1, 
						        "prefix_length": 0, 
						        "ignore_tf": False,  
                                                    }},
                                            
                                "from": 0,
                                "size": 5,
                                }
                        __result = ES_CLIENT.search(index="eatery", doc_type="eatery", body=fuzzy_search_body)
                        __result = ElasticSearchScripts.process_result(__result)
                        return __result


                def find_standard_match(eatery_name):
                        standard_search_body = {"_source": ["eatery_name", "__eatery_id"],
                                "query": {
                                    "match": {
                                            "eatery_name": {
                                                    "query": eatery_name,
                                                    }}},
                                            
                                "from": 0,
                                "size": 5,
                                }
                        __result = ES_CLIENT.search(index="eatery", doc_type="eatery", body=standard_search_body)
                        __result = ElasticSearchScripts.process_result(__result)
                        return __result

                
                result = find_exact_match(eatery_name)
                if result:
                        return result
                else:
                        print "There is no exact result matching the Query %s"%eatery_name
                
                result = find_standard_match(eatery_name)
                if result:
                        return result


                result = find_fuzzy_match(eatery_name)
                
                if result:
                        return result
                
                
                return None

            

            
        @staticmethod
        def get_dish_match(__dish_name, number_of_dishes=None, number_of_suggestions=None):
                """
                        dish_suggestions= {
                                   "query":{
                                            "match":{
                                                    "dish_shingle":  __dish_name}},

                                    "from": 0,
                                   "size": dish_suggestions,
                                    "sort":[
                                            {"total_sentiments": {
                                                    "order" : "desc"}}
                                            ]
			            }
                Args:
                        dish_name must be lower case
                        type(string)

                First it will look for the exact dish_name

                Second it will do the fuzzy search, and find the dish similar considering levenshtein distance algorithm, 

                third it will find the dish_names that sound similar to given dish name on the basis of soundex algorithm
                
                Preference of search:
                        Exact match
                        Fuzzy match
                        Phonetic match
                        Standard match
                """
                print "Dish passed in suggest_dish instance of ElasticSearchScripts is %s"%__dish_name
                if not number_of_dishes:
                        number_of_dishes = 10
                


                def find_exact_match(__dish_name, number_of_dishes):
                        exact_dish_search_body={"_source": ["name", "eatery_name", "__eatery_id", "location", "good", "poor", "average", "excellent", "terrible", "total_sentiments", "eatery_address"],
                                    "query":{
                                            "term":{
                                                        "dish_raw":  __dish_name}},

                                    "from": 0,
                                    "size": number_of_dishes,
                                    "sort": [
                                            {"total_sentiments": {
                                                    "order" : "desc"}}
                                           ]
                                }

                        __result = ES_CLIENT.search(index="food", doc_type="dishes", body=exact_dish_search_body)
                        __result = ElasticSearchScripts.process_result(__result)
                        return __result

                def find_fuzzy_match(__dish_name, number_of_dishes):
                        print "Fussy match for %s"%__dish_name
                        fuzzy_search_body = {"_source": ["name", "eatery_name", "__eatery_id", "location", "good", "poor", "average", "excellent", "terrible", "total_sentiments", "eatery_address"],
                                "query": {
                                    "fuzzy_like_this": {
                                            "fields": ["dish_shingle", "dish_raw"],
							"like_text": __dish_name,  
                                                        "fuzziness": 6,
                                                        "max_query_terms": 25,
						        "boost": 1, 
						        "prefix_length": 0, 
						        "ignore_tf": False,  
                                                    }},
                                            
                                "from": 0,
                                "size": number_of_dishes,
                                }
                        __result = ES_CLIENT.search(index="food", doc_type="dishes", body=fuzzy_search_body)
                        __result = ElasticSearchScripts.process_result(__result)
                        return __result


                def find_standard_match(__dish_name, number_of_dishes):
                        standard_search_body = {"_source": ["name", "eatery_name", "__eatery_id", "location", "good", "poor", "average", "excellent", "terrible", "total_sentiments", "eatery_address"],
                                "query": {
                                    "match": {
                                            "name": {
                                                    "query": __dish_name,
                                                    }}},
                                            
                                "from": 0,
                                "size": number_of_dishes,
                                "sort": [
                                            {"trending_factor": {
                                                    "order" : "desc"}}
                                           ],
                                }
                        __result = ES_CLIENT.search(index="food", doc_type="dishes", body=standard_search_body)
                        __result = ElasticSearchScripts.process_result(__result)
                        return __result

                
                print "FInished Dish passed in suggest_dish instance of ElasticSearchScripts is %s"%__dish_name
                result = find_exact_match(__dish_name, number_of_dishes)
                if result:
                        return result
                else:
                        print "There is no exact result matching the Query %s"%__dish_name
                
                result = find_fuzzy_match(__dish_name, number_of_dishes)
                
                if result:
                        return result
                
                
                return None


if __name__ == "__main__":
            ElasticSearchScripts(renew_indexes=True)
            for post in eateries_results_collection.find():
                    __id = post.get("eatery_id")
            #for __id in [u'310159', u'307852', u'312114', u'18198477', u'18146368', u'17979576', u'306787', u'8234', u'309243', u'311835', u'313058', u'307853', u'17989123', u'17953918', u'18017241', u'307931', u'8893', u'303095', u'9354', u'4412', u'310094', u'9321', u'303092', u'5591', u'301001', u'301442', u'5732', u'5030', u'301131', u'4899', u'308322', u'8369', u'7070', u'305137', u'307360', u'8910', u'309792', u'308463', u'307330', u'306334', u'307454', u'91', u'304027', u'3392', u'2664', u'307146', u'308642', u'311182', u'310846', u'310396', u'8873', u'305681', u'18082196', u'312437', u'313384', u'304676']:
                    ElasticSearchScripts.insert_eatery(__id)

