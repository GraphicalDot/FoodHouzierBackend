#!/usr/bin/env python
"""
https://gist.github.com/lightsuner/5df39112b8507d15ede6
https://gist.github.com/lukas-vlcek/5143799
http://192.168.1.5:9200/_cluster/state?pretty&filter_nodes=true&filter_routing_table=true&filter_indices=dishes

Author: kaali
Dated: 9 June, 2015
"""

import time
import os
import sys
from compiler.ast import flatten
file_name = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_name)
from GlobalConfigs import ES, eateries_results_collection, bcolors
from elasticsearch import Elasticsearch, helpers
from elasticsearch import RequestError

ES_CLIENT = Elasticsearch("localhost:")
EATERY_ONE_DISHES = list()
EATERY_TWO_DISHES = list()
NUMBER_OF_DOCS = 10
#localhost:9200/test/_analyze?analyzer=whitespace' -d 'this is a test'
class ElasticSearchScripts(object):
        def __init__(self):
                """
                Will have minimum two indexes
                one for dishes and one for eateries
                """
                
                pass





        @staticmethod
        def prep_es_indexes():
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
                        ES_CLIENT.indices.delete(index="dishes")
                        print "{0} DELETING index {1}".format(bcolors.OKGREEN, bcolors.RESET)
                except Exception as e:
                        print "{0} {1}".format(bcolors.FAIL, bcolors.RESET)
                        print "{0} Index Dishes doesnt exists {1}".format(bcolors.FAIL, bcolors.RESET)
                        print e

                __settings = {
                                "settings": {
                                        "analysis": {
                                                "analyzer": {
                                                        "phonetic_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "whitespace",
                                                            "filter": ["lowercase", "asciifolding", "standard", "custom_metaphone"],
                                                                    },
                                                        "keyword_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "keyword",
                                                            "filter": ["lowercase", "asciifolding"],
                                                                    },
                                                        "shingle_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "ngram_tokenizer",
                                                            "filter": ["lowercase", "asciifolding", "shingle_tokenizer"],
                                                                    },
                                                        "custom_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "ngram_tokenizer",
                                                            "filter": ["lowercase", "asciifolding"],
                                                                    },
                                                        "custom_analyzer_two": {
                                                            "type": "custom",
                                                            "tokenizer" : "limited_tokenizer",
                                                            "filter": ["lowercase", "asciifolding"],
                                                                    },
                                                        "standard_analyzer": {
                                                                "type": "custom", 
                                                                "tokenizer": "standard",
                                                                "filter": ["lowercase", "asciifolding"],
                                                                }
                                                        },
                                                "tokenizer": {
                                                        "ngram_tokenizer": {
                                                                "type" : "edgeNGram",
                                                                "min_gram" : 2,
                                                                "max_gram" : 100,
                                                                "token_chars": [ "letter", "digit" ]
                                                                },
                                                        "limited_tokenizer": {
                                                                "type" : "edgeNGram",
                                                                "min_gram" : "2",
                                                                "max_gram" : "10",
                                                                "token_chars": [ "letter", "digit" ]
                                                                },
                                                        }, 
                                                "filter": {
                                                            "shingle_tokenizer": {
                                                                "type" : "shingle",
                                                                "min_shingle_size" : 2,
                                                                "max_shingle_size" : 5,
                                                                },

                                                            "custom_metaphone": {
                                                                    "type" : "phonetic",
                                                                    "encoder" : "metaphone",
                                                                    "replace" : False
                                                                    }
                                                            }
                                                }
                                        }}

                print "{0}Settings updated {1}".format(bcolors.OKGREEN, bcolors.RESET)

                ES_CLIENT.indices.create(index="dishes", body=__settings)
                __mappings = {'dish': {
                                 '_all' : {'enabled' : True},
                                'properties': 
                                                {'name': 
                                                        {
                                                            #'analyzer': 'custom_analyzer', 
                                                            'type': 'string', 
                                                            'copy_to': ['dish_raw', 'dish_shingle', "dish_phonetic"],
                                                            },
                                                    
                                                    
                                        'dish_phonetic': {
                                                    'type': 'string', 
                                                    'analyzer': 'phonetic_analyzer',
                                                    },
                                        'dish_shingle': {
                                                    'type': 'string', 
                                                    'analyzer': 'shingle_analyzer',
                                                    },
                                        'dish_raw': {
                                                    'type': 'string', 
                                                    'analyzer': 'keyword_analyzer',
                                                    },

                                        'eatery_shingle': {
                                                    'type': 'string', 
                                                    'analyzer': 'shingle_analyzer',
                                                    },

                                        'eatery_raw': {
                                                    'type': 'string', 
                                                    'analyzer': 'keyword_analyzer',
                                                    },
                                        
                                        'negative': {'type': 'long'},
                                        'neutral': {'type': 'long'},
                                        'positive': {'type': 'long'},
                                        'similar': {
                                                'properties': {'name': 
                                                                    {
                                                                        'type': 'string', 
                                                                        'copy_to': ['dish_raw', 'dish_shingle', "dish_phonetic"],
                                                                    
                                                                    },
                                                            'negative': {
                                                                     'type': 'long'},
                                                            'neutral': {
                                                                    'type': 'long'},
                                                            'positive': {
                                                                    'type': 'long'},
                                                            'super-negative': {
                                                                    'type': 'long'},
                                                            'super-positive': {
                                                                    'type': 'long'},
                                                            'timeline': {
                                                                'type': 'string'}
                                                            }
                                                },
   
                                        'super-negative': {
                                                    'type': 'long'},
                                        'super-positive': {
                                                    'type': 'long'},
                                        'eatery_name': {
                                                            'type': 'string', 
                                                            'copy_to': ['eatery_shingle', "eatery_raw"],
                                                    },
                                        'eatery_id': {
                                                    'type': 'string', 
                                                    'index': 'not_analyzed',
                                                    },
                                        
                                        'total_sentiments': {
                                                    'type': 'integer',
                                                },
                                        'timeline': {
                                            'type': 'string'}}}}
                
                try:
                        ES_CLIENT.indices.put_mapping(index="dishes", doc_type="dish", body = __mappings)
                        print "{0}Mappings updated {1}".format(bcolors.OKGREEN, bcolors.RESET)
                except Exception as e:
                        print "{0}Mappings update Failed with error {1} {2}".format(bcolors.FAIL, e, bcolors.RESET)

                test_doc = {'eatery_name': u"Karim's", u'name': u'chicken korma', u'super-negative': 0, u'negative': 0, u'super-positive': 1, 
                        u'neutral': 6, u'timeline': [[u'positive', u'2013-01-24 17:34:01'], [u'neutral', u'2014-07-18 23:49:05'], 
                        [u'neutral', u'2014-06-05 13:30:14'], [u'super-positive', u'2013-07-04 17:03:37'], 
                        [u'neutral', u'2013-04-18 20:40:35'], [u'neutral', u'2013-01-18 23:04:17'], 
                        [u'neutral', u'2013-01-11 14:04:49'], [u'neutral', u'2012-12-29 21:51:43']], 'eatery_id': '463', 
                        u'similar': [{u'name': u'chicken qorma', u'positive': 1, u'negative': 0, u'super-positive': 0, u'neutral': 0, 
                            u'timeline': [[u'positive', u'2013-01-24 17:34:01']], u'super-negative': 0}, 
                            {u'name': u'chicken korma', u'positive': 0, u'negative': 0, u'super-positive': 1, u'neutral': 5, 
                                u'timeline': [[u'neutral', u'2014-07-18 23:49:05'], [u'neutral', u'2014-06-05 13:30:14'], 
                                    [u'super-positive', u'2013-07-04 17:03:37'], [u'neutral', u'2013-04-18 20:40:35'], 
                                    [u'neutral', u'2013-01-18 23:04:17'], [u'neutral', u'2013-01-11 14:04:49']], u'super-negative': 0}, 
                                {u'name': u'i order chicken korma', u'positive': 0, u'negative': 0, u'super-positive': 0, u'neutral': 1, 
                                    u'timeline': [[u'neutral', u'2012-12-29 21:51:43']], u'super-negative': 0}], u'positive': 1}
                
                
                print "{0}Updating test data {1}".format(bcolors.OKGREEN, bcolors.RESET)
                l = ES_CLIENT.index(index="dishes", doc_type="dish", body=test_doc)

                print "{0}Result:\n {1} {2}".format(bcolors.OKGREEN, l, bcolors.RESET)

                __body = {"query" : {
                            "term" : { "_id" : l.get("_id")}
                                }}

                print "{0}Test Doc deleted {1}".format(bcolors.OKGREEN, bcolors.RESET)
                ES_CLIENT.delete_by_query(index="dishes", doc_type="dish", body=__body)

                
                return 
        
        @staticmethod
        def populate_test_data():
                """
                ##TODO: eatery_address and eatery_coordinates shall be strored in mongodb and also in elstic search
                ##Update very dish with a total_sentiments counter, so that the result from elastic search could be sorted based
                ##on this counter

                Update elastic search with 20 dishes from two restaurants, to run tests 
                eatery_one
                        eatery_name: karim's
                        eatery_id: 463
                        dishes = [u'moti roti', u'20 mins', u'10 mins', u'20 min', u'30 mins', u'20 mins', u'gurda kaleji', 
                        u'gurda kaleji', u'gurda kalegi', u'gurda / kaleji', u'signature dishes', u'signature dish', u'signature dishes', 
                        u'mutton keema', u'mutton kheema', u'mutton keema', u'shahi tukda', u'shahi tukdaa', u'shahi tukda', u'tender meat', 
                        u'roghini naan', u'qeema naan', u'rogan josh']

                        
                eatery_two
                        eatery_name: hauz khas socials
                        eatery_id = 308322
                        dishes_name = [u'steel tiffin box', u'steel tiffins', u'tiffin boxes', u'steel tiffin boxes', 
                        u'steel tiffin box', u'steel tiffin', u'chicken sizzler', u'chicken sizzler', u'peri peri chicken sizzler', 
                        u'grilled chicken sizzler', u'chilli paneer black pepper china box', u'chilli paneer black pepper china box', 
                        u'chilly paneer black pepper china box', u'bombay bachelor sandwich', u'bombay bachelor sandwich', 
                        u'bombay bachelors sandwich', u'\u2022 bombay bachelor sandwich', u'prawn sesame toast', u'prawn sesame', 
                        u'sesame toast', u'sesame prawn toast', u'prawn sesame toast', u'honey chilli potatoes', u'honey chilli potatoes', 
                        u'honey chilli potato', u'chilli potatoes', u'cottage cheese', u'cottage cheese sizzler', u'cottage cheese', 
                        u'toffee sauce', u'cosmo explosion', u'chicken peri peri']
                """
                eatery_dishes_one = eateries_results_collection.find_one({"eatery_id": "463"})["food"]["dishes"][40: 40+NUMBER_OF_DOCS]
                eatery_dishes_two = eateries_results_collection.find_one({"eatery_id": "308322"})["food"]["dishes"][40: 40+NUMBER_OF_DOCS]
                __list = []

                print "\n\n{0}Updating ES with karims {1} documents {2}".format(bcolors.OKGREEN, NUMBER_OF_DOCS, bcolors.RESET)
                for dish in eatery_dishes_one:
                        dish.update({"eatery_name": "karim's", "eatery_id": "463", "_index": "dishes", "_type": "dish"})
                        __list.append(dish)       

                try:
                        __result = helpers.bulk(ES_CLIENT, __list, stats_only=True)
                        print "{0}Updated ES with karims {1} documents with result {2} {3}".format(bcolors.OKGREEN, \
                                                                                NUMBER_OF_DOCS, __result,  bcolors.RESET)
                except Exception as e:
                        print "{0}Update or karim dishes failed with error {1}{2}".format(bcolors.FAIL, e, bcolors.RESET)

                __list = []
                print "\n\n{0}Updating ES with Hauz khas socials {1} documents {2}".format(bcolors.OKGREEN, NUMBER_OF_DOCS, bcolors.RESET)
                for dish in eatery_dishes_two:
                        dish.update({"eatery_name": "Hauz Khas Social", "eatery_id": "308322",  "_index": "dishes", "_type": "dish"})
                        __list.append(dish)       
                try:
                        __result = helpers.bulk(ES_CLIENT, __list, stats_only=True)
                        print "{0}Updated ES with Hauxkhas socials {1} documents with result {2} {3}".format(bcolors.OKGREEN,\
                                NUMBER_OF_DOCS, __result,  bcolors.RESET)
                except Exception as e:
                        print "{0}Update or Hauxkhas socials dishes failed with error {1}{2}".format(bcolors.FAIL, e, bcolors.RESET)


        @staticmethod
        def check_eatery():
                """
                Points to consider
                The difference is simple: filters are cached and don't influence the score, therefore faster than queries.
                As a general rule, filters should be used instead of queries:
                        for binary yes/no searches
                        for queries on exact values
                As a general rule, queries should be used instead of filters:
                        for full text search
                        where the result depends on a relevance score
                more information: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html

                So to search eatery on eatery_id we need exact results
                but on eatery name we need to have query on eatery_name
                
                First:
                    Test for eatery_id, we have implemented analyzer: custom_standard which implies that it takes
                    eatery_id as a string, doesnt convert it to ngrams
                """
                
                ##THis is to make ES wait, so that all the documents that were being indexed, will be available for 
                ##search
                ES_CLIENT.indices.refresh() 
                print "\n\n{0}         Testing for search with eatery_id {1}".format(bcolors.OKGREEN, bcolors.RESET)
                eatery_id = "308322"
                search_body = {"query":
                                    {"match_phrase":
                                                {"eatery_id": eatery_id}}}
                
                filtered_query = {"query": {
                                            "filtered": {
                                                    "filter":   { "term": { "eatery_id": eatery_id }}}}}
                                    
                result = ES_CLIENT.search(index="dishes", doc_type="dish", body=filtered_query)
                if result["hits"]["total"] == NUMBER_OF_DOCS:
                        print "{0}         Search with eatery_id test passed{1}".format(bcolors.OKGREEN, bcolors.RESET)
                else:
                        print "{0}         Search with eatery_id test Failed Miserbly{1}".format(bcolors.FAIL, bcolors.RESET)
                        


                ##testing for exact eatery_name



        def return_dishes(eatery_id, number_of_dishes):
                """
                This will return all the dishes related to the particular eatery_id
                """
                if type(eatery_id) != str:
                        raise StandardError("eatery_id should be a string")


                search_body = {                                                   
                            "query":{ 
                                    "match_phrase":{ 
                                                "eatery_id":  eatery_id}}, 
                            "sort": { "total_sentiments": { "order": "desc" } }, 
                            "from": 0,
                            "size": number_of_dishes, 
                          }

                result = ES_CLIENT.search(index="dishes", doc_type="dish", body=search_body)["hits"]["hits"]
                return [e.get("_source") for e in result]
                






        def update_n_delete(eatery_id, updated_dishes):
                """
                eatery_id
                updated_dishes: New dictionary for dishes with the same datastructues but now have new values due to
                        new reviews update.

                This method deals with deleting the old one and updating the es with the new ones.
                """
                {"query": {
                        "match": {
                                        "title" : "elasticsearch"
                                                }
                            }
            }

                search= ES_CLIENT.search(
                                q='The Query to ES.',
                                index="*logstash-*",
                                    size=10,
                                        search_type="scan",
                                            scroll='5m',
                                            )


        def _change_default_analyzer(__index_name):
                client.indices.close(index=__index_name)
                body = {"index": {
                                "analysis": {
                                        "analyzer": {
                                                "default": {
                                                        "type": "custom_analyzer"}}}}}

                client.indices.put_settings(index=__index_name, body=body)

                client.indices.open(index=__index_name)
                return 


        def initial(self, eatery_id):
                """
                For every eatery stored in mongodb, We have four keys asscoaited with that post
                food, service, cost, ambience

                For these corresponding keys we will have sub keys associated with it, For ex
                food tag has these keys associated with it.
                [u'menu-food', u'overall-food', u'sub-food', u'place-food', u'dishes']
                service:
                return 
                        "mappings": {
                                "name": {
                                        "properties": {
                                                    "name": {
                                                            "type": "string",
                                                            "analyzer": "custom_analyzer",
                                                            },
                                                    "similar": {
                                                            "type": "nested",
                                                             "properties": {
                                                                    "name": {
                                                                            "type": "string", 
                                                                                    "index_analyzer": "custom_analyzer", 
                                                                                    "search_analyzer": "custom_search_analyzer",
                                                                                },
                                                                    'negative': { "type": "int"
                                                                                },
                                                                    'neutral': {"type": "int"
                                                                                },
                                                                    'positive': {"type": "int"
                                                                                },
                                                                    'super-negative': {"type": "int"
                                                                        },
                                                                    'super-positive': {"type": "int"
                                                                                },
                                                                    'timeline': {"type": "string"
                                                                            }
                                                             }
                                                            }
                                                        }
                                            }},

                """
                return 
        @staticmethod
        def upde_dish_for_rest(dish_name, eatery_id, eatery_name):
                """
                This method takes in three arguments dish_name, eatery_name, eatery_id
                """




        def flush(self, index_name):
                return


        @staticmethod
        def auto_complete(__str):
                """
                Returns lists of dishes name else returm empy list
                """
                search_body={"fields": ["name"], 
                        "query": {
                                "prefix": {
                                    "name": __str
                                        }}}


                search_body = {
                            "query" : {
                                "multi_match" : {
                                        "fields" : ["name", "similar"],
                                        "query" : __str,
                                        "type" : "phrase_prefix"
                                                                                        }
                                            }
                            }

                result = ES.search(index='food', doc_type='dishes', body=body)
                return flatten([e["fields"]["name"] for e in result["hits"]["hits"]])

        @staticmethod
        def exact_dish_match(__str):
                search_body = {"query":
                        {"match_phrase": 
                                {"name": __str}
                        }}
                return 

                
        @staticmethod
        def dish_suggestions(__str):
                """
                Case1:
                    if actual resul couldnt be found
                Case2: 
                    To be sent along with the actual result
                return Suggestions for the __str
                """
                search_body =  body = {"fields": ["name", "similar"],
                        "query" : {
                            "multi_match" : {"fields" : ["name", "similar"],
                                            "query" : __str,
                                            "type" : "phrase_prefix"
                                                                                    }
                                        }}

                ##if you dont need score, also filter searchs are very fast to executed
                #and can be cached
                body= { "query" : {
                                    "filtered" : { 
                                                    "query" : {
                                                                        "match_all" : {} 
                                                                                    },
                                                                "filter" : {
                                                                                    "term" : { 
                                                                                                            "name" : "chicken"
                                                                                                                            }
                                                                                                }
                                                                        }
                                        }
                        }

                        
                result = ES.search(index='food', doc_type='dishes', body=body)

if __name__ == "__main__":
        ElasticSearchScripts.prep_es_indexes()
        ElasticSearchScripts.populate_test_data()
        ElasticSearchScripts.check_eatery()

