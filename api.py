#!/usr/bin/env pypy
from datetime import date
import tornado.escape
import tornado.ioloop
import tornado.web
import tornado.autoreload 
import tornado.httpserver
import json
from tornado.log import enable_pretty_logging
from tornado.httpclient import AsyncHTTPClient
from tornado.web import asynchronous
import pymongo
import os
import sys
from functools import update_wrapper
from functools import wraps
import time
import blessings
import optparse
from concurrent.futures import ThreadPoolExecutor
from Sentence_Tokenization import SentenceTokenizationOnRegexOnInterjections
from nltk.stem import SnowballStemmer 
from PreProcessingText import PreProcessText
from sklearn.feature_extraction.text import CountVectorizer 
from cPickle import dump, load, HIGHEST_PROTOCOL
from sklearn.externals import joblib
from topia.termextract import extract  
from textblob import TextBlob
Terminal = blessings.Terminal()

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_path)

from configs import SentimentClassifiersPath, TagClassifiersPath,\
                FoodClassifiersPath, ServiceClassifiersPath,\
                AmbienceClassifiersPath, CostClassifiersPath 


from configs import SentimentVocabularyFileName, SentimentFeatureFileName, SentimentClassifierFileName
from configs import TagVocabularyFileName, TagFeatureFileName, TagClassifierFileName
from configs import FoodVocabularyFileName, FoodFeatureFileName, FoodClassifierFileName
from configs import ServiceVocabularyFileName, ServiceFeatureFileName, ServiceClassifierFileName
from configs import CostVocabularyFileName, CostFeatureFileName, CostClassifierFileName
from configs import AmbienceVocabularyFileName, AmbienceFeatureFileName, AmbienceClassifierFileName
from configs import r_reviews, r_eateries, r_clip_eatery

from configs import cd
import cPickle
from SaveMemory.elasticsearch_db import ElasticSearchScripts
import HTMLParser
html_parser = HTMLParser.HTMLParser()


def cors(f):
        @wraps(f) # to preserve name, docstring, etc.
        def wrapper(self, *args, **kwargs): # **kwargs for compability with functions that use them
                self.set_header("Access-Control-Allow-Origin",  "*")
                self.set_header("Access-Control-Allow-Headers", "content-type, accept")
                self.set_header("Access-Control-Max-Age", 60)
                return f(self, *args, **kwargs)
        return wrapper
                               


def print_execution(func):
        "This decorator dumps out the arguments passed to a function before calling it"
        argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
        fname = func.func_name
        def wrapper(*args,**kwargs):
                start_time = time.time()
                print Terminal.green("Now {0} have started executing".format(func.func_name))
                result = func(*args, **kwargs)
                print Terminal.green("Total time taken by {0} for execution is--<<{1}>>\n".format(func.func_name, (time.time() - start_time)))
                return result
        return wrapper




def prediction(sentences, vocabulary, features, classifier):
                print sentences
                loaded_vectorizer= CountVectorizer(vocabulary=vocabulary) 
                
                sentences_counts = loaded_vectorizer.transform(sentences)
                
                reduced_features = features.transform(sentences_counts.toarray())
                         
                predictions = classifier.predict(reduced_features)
                print predictions
                return predictions



def filter_categories(sentences):
        """
        These sentences will be in the form (sentence, tag, sentiment)
        """
        tags = set([tag for (sentence, tag, sentiment) in sentences])
        result = dict()
        for tag in tags:
                result.update({tag: [e for e in sentences if e[1]==tag]})

        return result



class PostText(tornado.web.RequestHandler):


        @cors
        @print_execution
        @tornado.gen.coroutine
        def post(self):
                print PATH_COMPILED_CLASSIFIERS

                try:
                        text = self.get_argument("text")
                except tornado.web.MissingArgumentError:
                        self.set_status(400, "Missing Arhument")
                        self.write({"messege": "give me the money"})
                        self.finish()
                        return 
                

                if not text:
                        self.set_status(400, "Empty Text")
                        self.write({"success": False, 
                                    "error": True,
                                    "message": "Dont send empty text"})
                        self.finish()
                        return 


                
                tokenized_sentences = sent_tokenizer.tokenize(text)
                #predicting tags
                tags = prediction(tokenized_sentences, tag_vocabulary, tag_features, tag_classifier)
                sentiments = prediction(tokenized_sentences, sentiment_vocabulary, sentiment_features, sentiment_classifier)

                result = filter_categories(zip(tokenized_sentences, tags,
                    sentiments))

                overall_result = list() 



                some_lambda= lambda ((sentence, category, sentiment), sub): (sentence, category, sentiment, sub) 
                other_categories = lambda (sentence, category, sentiment): (sentence, category, sentiment, None)
                
                if result.get("food"):
                        food_sentences = [sentence for (sentence, category, sentiment) in result["food"]]
                        food_subs = prediction(food_sentences, food_vocabulary,
                                food_features, food_classifier)

                        __text = " ".join(food_sentences)
                        tb_nps = TextBlob(__text)
                        topia_nps = [np[0] for np in extractor(__text)]
                        nps = list(set.union(set(tb_nps.noun_phrases), set(topia_nps)))

                        food_result = map(some_lambda, zip(result["food"], food_subs))
                        overall_result.extend(food_result)
        

                if result.get("service"):
                        service_sentences = [sentence for (sentence, category,
                            sentiment) in result["service"]]
                        service_subs = prediction(service_sentences,
                                service_vocabulary, service_features, service_classifier)
                        service_result = map(some_lambda, zip(result["service"], service_subs))
                        overall_result.extend(service_result)

                if result.get("cost"):
                        cost_sentences = [sentence for (sentence, category,
                            sentiment) in result["cost"]]
                        cost_subs = prediction(cost_sentences, cost_vocabulary,
                                cost_features, cost_classifier)
                        cost_result = map(some_lambda, zip(result["cost"], cost_subs))
                        overall_result.extend(cost_result)


                if result.get("ambience"):
                        ambience_sentences = [sentence for (sentence, category,
                            sentiment) in result["ambience"]]
                        ambience_subs = prediction(ambience_sentences,
                                ambience_vocabulary, ambience_features,
                                ambience_classifier)
                        ambience_result = map(some_lambda,
                                zip(result["ambience"], ambience_subs))
                        overall_result.extend(ambience_result)

                if result.get("menu"):
                        menu_result = map(other_categories, result["menu"])
                        overall_result.extend(food_result)

                if result.get("place"):
                        place_result = map(other_categories, result["place"])
                        overall_result.extend(place_result)
                
                if result.get("overall"):
                        verall_result = map(other_categories, result["overall"])
                        overall_result.extend(verall_result)
                
                if result.get("cuisine"):
                        cuisine_result = map(other_categories, result["cuisine"])
                        overall_result.extend(cuisine_result)
                
                if result.get("null"):
                        null_result = map(other_categories, result["null"])
                        overall_result.extend(null_result)
                        
                            

                make_json = lambda (sentence, category, sentiment, sub): {"sentence": sentence, "category":
                                category, "sentiment": sentiment, "sub_category": sub}



                result = map(make_json, overall_result)


                ##removing null sentences
                result = filter(lambda x: x.get("sentence"), result)

                try:
                        noun_phrases = nps
                except Exception as e:
                        noun_phrases = None


                self.set_status(200)
                print result 
                self.write({"success": True, 
                            "error": False,
                            "result": result,
                            "noun_phrases": noun_phrases, 
                            })
                self.finish()
                return


class NearestEateries(tornado.web.RequestHandler):
        @cors
        @print_execution
        #@tornado.gen.coroutine
        @asynchronous
        def post(self):
                """
                Accoriding to the latitude, longitude given to it gives out the 10 restaurants nearby
                """
                
                latitude =  float(self.get_argument("latitude"))
                longitude =  float(self.get_argument("longitude")) 
                

                #result = eateries.find({"eatery_coordinates": {"$near": [lat, long]}}, projection).sort("eatery_total_reviews", -1).limit(10)
                #result = eateries.find({"eatery_coordinates" : SON([("$near", { "$geometry" : SON([("type", "Point"), ("coordinates", [lat, long]), \
                #        ("$maxDistance", range)])})])}, projection).limit(10)


                try:
                        short_eatery_result_collection.index_information()["location_2d"]

                except Exception as e:
                        self.write({"success": False,
			        "error": True,
                                "result": "Location index not present of collection",
			    })
                        self.finish()
                        return 
                        
                projection={"__eatery_id": True, "eatery_name": True, "eatery_address": True, "location": True, "_id": False, "food": True, \
                        "overall": True}
                
                result = short_eatery_result_collection.find({"location": { "$geoWithin": { "$centerSphere": [[latitude, longitude], .5/3963.2] } }}, \
                        projection).sort("overall.total_sentiments", -1).limit(10)
                ##__result  = list(result)

                final_result = list()
                for element in result:
                            sentiments = element.pop("overall")
                            dishes = element.pop("food")
                            element.update({"eatery_details": 
                                {"location": element.pop("location"),
                                    "__eatery_id": element.get("__eatery_id"), 
                                    "eatery_address": element.pop("eatery_address"), 
                                    "eatery_name": element.pop("eatery_name"),
                                    "overall": {"total_sentiments": sentiments.get("total_sentiments")},
                                    "food": dishes,
                                    }})
                                    
                            element.update({"excellent": sentiments.get("excellent"), 
                                    "poor": sentiments.get("poor"), 
                                    "good": sentiments.get("good"), 
                                    "average": sentiments.get("average"), 
                                    "terrible": sentiments.get("terrible"), 
                                    "total_sentiments": sentiments.get("total_sentiments"),    
                                    })

                            final_result.append(element)


                final_result = sorted(final_result, reverse=True, key = lambda x: x.get("total_sentiments"))
                self.write({"success": True,
			"error": False,
                        "result": final_result,
			})
                self.finish()
                return 
                

class TextSearch(tornado.web.RequestHandler):
        @cors
        @print_execution
        @tornado.gen.coroutine
        def post(self):
                """
                This api will be called when a user selects or enter a query in search box
                """
                text = self.get_argument("text")
                __type = self.get_argument("type")


                if __type == "dish":
                        """
                        It might be a possibility that the user enetered the dish which wasnt in autocomplete
                        then we have to search exact dish name of seach on Laveneshtein algo
                        """
                        ##search in ES for dish name 

                        result = list()
                        __result = ElasticSearchScripts.get_dish_match(text)
                        for dish in __result:
                                eatery_id = dish.get("eatery_id")
                                __eatery_details = r_clip_eatery.find_one({"eatery_id": eatery_id})
                                for e in ["eatery_highlights",
                                        "eatery_cuisine", "eatery_trending", "eatery_known_for", "eatery_type", "_id"]:
                                        try:
                                                __eatery_details.pop(e)
                                        except Exception as e:
                                                print e
                                                pass
                                dish.update({"eatery_details": __eatery_details})
                                result.append(dish)

                elif __type == "cuisine":
                        ##gives out the restarant for cuisine name
                        print "searching for cuisine"
                        result = list()
                        __result = ElasticSearchScripts.eatery_on_cuisines(text)
                        print __result
                        for eatery in __result:
                                    __result= short_eatery_result_collection.find_one({"__eatery_id": eatery.get("__eatery_id")}, {"_id": False, "food": True, "ambience": True, \
                                            "cost":True, "service": True, "menu": True, "overall": True, "location": True, "eatery_address": True, "eatery_name": True, "__eatery_id": True})

                                    eatery.update({"eatery_details": __result})
                                    result.append(eatery)

                elif __type == "eatery":
                       
                            ##TODO : Some issue with the restaurant chains for example
                            ##big chills at different locations, DOnt know why ES
                            ##not returning multiple results
                            ##TODO: dont know why dropped nps are still in result.
                            result = r_eateries.find_one({"eatery_name": text},
                                    {"_id": False, "eatery_known_for": False,
                                        "droppped_nps": False,
                                        "eatery_trending": False,
                                        "eatery_highlights": False})
                           
                            print result.get("eatery_id")
                            __result = process_result(result)
                            
                            pictures = result.pop("pictures")
                            result.update({"pictures": pictures[0:2]})
                            result.update(__result)
                            result = [result]

                elif not  __type:
                        print "No type defined"

                else:
                        print __type
                        self.write({"success": False,
			        "error": True,
			        "messege": "Maaf kijiyega, Yeh na ho paayega",
			        })
                        self.finish()
                        return 
                print result
                self.write({"success": True,
			        "error": False,
			        "result": result,
			})
                self.finish()
                return 



class Suggestions(tornado.web.RequestHandler):
        @cors
        @print_execution
        @tornado.gen.coroutine
        def post(self):
                """
                Return:
                        [
                        {u'suggestions': [u'italian food', 'italia salad', 'italian twist', 'italian folks', 'italian attraction'], 'type': u'dish'},
                        {u'suggestions': [{u'eatery_name': u'Slice of Italy'}], u'type': u'eatery'},
                        {u'suggestions': [{u'name': u'Italian'}, {u'name': u'Cuisines:Italian'}], 'type': u'cuisine'}
                        ]
                """
                print self.request.arguments  
                query = self.get_argument("phrase")
                
                dish_suggestions = ElasticSearchScripts.dish_suggestions(query)
                cuisines_suggestions =  ElasticSearchScripts.cuisines_suggestions(query)
                eatery_suggestions = ElasticSearchScripts.eatery_suggestions(query)
                #address_suggestion = ElasticSearchScripts.address_suggestions(query)
                

                if cuisines_suggestions:
                        cuisines_suggestions= [e.get("name") for e in cuisines_suggestions]
                

                if eatery_suggestions:
                        eatery_suggestions= [e.get("eatery_name") for e in eatery_suggestions]

                
                result = {"dish": [{"name": e, "type":"dish"} for e in list(set([e.get("name") for e in
                    dish_suggestions]))] ,
                    "eatery":  [{"name": e, "type": "eatery"} for e in
                        eatery_suggestions],
                    "cuisine": [{"name": e, "type": "cuisine"} for e in cuisines_suggestions], 
                                }
                print result
                self.write(result)
                self.finish()
                return 

def process_result(result):
                number_of_dishes = 20
                dishes = sorted(result["food"]["dishes"], key=lambda x: x.get("total_sentiments"), reverse=True)[0: number_of_dishes]
                overall_food = result["food"]["overall-food"]

                def convert_to_list(_dict):
                        _list = list()
                        for (key, value) in _dict.iteritems():
                                value.update({"name": key})
                                _list.append(value)
                        return _list


                ambience = result["ambience"]
                cost = result["cost"]
                service = result["service"]
                overall = result["overall"]
                menu = result["menu"]

                ##removing timeline
                [value.pop("timeline") for (key, value) in ambience.iteritems()]
                [value.pop("timeline") for (key, value) in cost.iteritems()]
                [value.pop("timeline") for (key, value) in service.iteritems()]
                overall.pop("timeline")
                menu.pop("timeline")
                [element.pop("timeline") for element in dishes]
                [element.pop("similar") for element in dishes]



                result = {"food": dishes,
                            "ambience": convert_to_list(ambience), 
                            "cost": convert_to_list(cost), 
                            "service": convert_to_list(service), 
                            "menu": menu,
                            "overall": overall,
                            "eatery_address": result["eatery_address"],
                            "eatery_name": result["eatery_name"],
                            "__eatery_id": result["__eatery_id"]
                            }
                        
                return result

class GetEatery(tornado.web.RequestHandler):
        @cors
        @print_execution
        @tornado.gen.coroutine
        def post(self):
                """
                """
                        
                __eatery_id =  self.get_argument("__eatery_id", None)
                eatery_name  =  self.get_argument("eatery_name", None)
                if __eatery_id:
                        result = r_eateries.find_one({"__eatery_id": __eatery_id})
                else:
                        result = r_eateries.find_one({"eatery_name": eatery_name})
                        __eatery_id = result.get("__eatery_id")        
                
                if not result:
                        """
                        If the eatery name couldnt found in the mongodb for the popular matches
                        Then we are going to check for demarau levenshetin algorithm for string similarity
                        """

                        self.write({"success": False,
			        "error": True,
                                "result": "Somehow eatery with this eatery is not present in the DB"})
                        self.finish()
                        return 
               

                
                __result = process_result(result)
                __result.update({"images": result["pictures"][0:2]})
                self.write({"success": True,
			"error": False,
                        "result": __result})
                self.finish()


                return 



class Application(tornado.web.Application):
        def __init__(self):
                handlers = [
                    (r"/post_text", PostText),
                    (r"/nearest_eateries", NearestEateries),
                    (r"/suggestions", Suggestions),
                    (r"/geteatery", GetEatery),
                    (r"/textsearch", TextSearch),
                    (r"/images/^(.*)", tornado.web.StaticFileHandler, {"path": "./images"},),
                    (r"/css/(.*)", tornado.web.StaticFileHandler, {"path": "/css"},),
                    (r"/js/(.*)", tornado.web.StaticFileHandler, {"path": "/js"},),]
                settings = dict(cookie_secret="ed3fc328ab47ee27c8f6a72bd5a1b647deb24ab590e7060a803c51c6",)
                tornado.web.Application.__init__(self, handlers, **settings)
                #self.executor = ThreadPoolExecutor(max_workers=60)


def stopTornado():
        tornado.ioloop.IOLoop.instance().stop()


def main():
        http_server = tornado.httpserver.HTTPServer(Application())
        http_server.bind("8000")
        http_server.start()
        #enable_pretty_logging()
        print Terminal.green("Server is started at localhost and running at post 8000")
        tornado.ioloop.IOLoop.instance().start()




if __name__ == "__main__":
        """
        application.listen(8000)
        tornado.autoreload.start()
        tornado.ioloop.IOLoop.instance().start()

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

        sent_tokenizer  = SentenceTokenizationOnRegexOnInterjections()
        print Terminal.green("Sentence Tokenizer has been initialized") 
        extractor = extract.TermExtractor()
        """
        main()





