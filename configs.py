

import pymongo
import os
from os.path import dirname, abspath, exists

base_dir = dirname(dirname(abspath(__file__)))
print base_dir
import platform 





if platform.system() == "Darwin":
        path_jar_files = "/Users/kaali/Programs/Python/ProjectHouzier/stanford-corenlp-python"
else:
        path_jar_files = "/home/kaali/Programs/Python/ProjectHouzier/stanford-corenlp-python"


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

if not exists("%s/CompiledModels"%base_dir):
        os.makedirs("%s/CompiledModels"%base_dir)
        with cd("%s/CompiledModels"%base_dir):
                for _dir in ["SentimentClassifiers", "TagClassifiers",
                             "FoodClassifiers", "ServiceClassifiers",
                             "AmbienceClassifiers", "CostClassifiers"]:
                        print "making %s at path %s"%(_dir, base_dir)
                        os.makedirs(_dir)
                




SentimentClassifiersPath = "%s/CompiledModels/SentimentClassifiers"%base_dir
TagClassifiersPath = "%s/CompiledModels/TagClassifiers"%base_dir
FoodClassifiersPath = "%s/CompiledModels/FoodClassifiers"%base_dir
ServiceClassifiersPath = "%s/CompiledModels/ServiceClassifiers"%base_dir
AmbienceClassifiersPath = "%s/CompiledModels/AmbienceClassifiers"%base_dir
CostClassifiersPath = "%s/CompiledModels/CostClassifiers"%base_dir



SentimentVocabularyFileName = "lk_vectorizer_sentiment.pkl"
SentimentFeatureFileName = "sentiment_features.pkl"
SentimentClassifierFileName = "svmlk_sentiment_classifier.pkl"

TagVocabularyFileName = "lk_vectorizer_tag.pkl"
TagFeatureFileName = "tag_features_pca_selectkbest.pkl"
TagClassifierFileName = "svmlk_tag_classifier.pkl"

FoodVocabularyFileName = "lk_vectorizer_food.pkl"
FoodFeatureFileName = "food_features_pca_selectkbest.pkl"
FoodClassifierFileName = "svmlk_food_classifier.pkl" 

ServiceVocabularyFileName =  "lk_vectorizer_service.pkl"
ServiceFeatureFileName = "service_features_pca_selectkbest.pkl"
ServiceClassifierFileName = "svmlk_service_classifier.pkl"

CostVocabularyFileName = "lk_vectorizer_cost.pkl"
CostFeatureFileName =  "cost_features_pca_selectkbest.pkl"
CostClassifierFileName = "svmlk_cost_classifier.pkl"

AmbienceVocabularyFileName = "lk_vectorizer_ambience.pkl"
AmbienceFeatureFileName = "ambience_features_pca_selectkbest.pkl"
AmbienceClassifierFileName = "svmlk_ambience_classifier.pkl"

reviews_data = dict(
        ip = "localhost",
        port = 27017,
        db = "Reviews",
        eateries= "ZomatoEateries",
        reviews = "ZomatoReviews",
        users = "ZomatoUsers",
)



corenlp_data = dict(
        ip = "localhost",
        port = 3456,
        db = "corenlp",
        sentiment= "sentiment",
        path_jar_files = path_jar_files
)


training_data = dict(
        ip = "localhost",
        port = 27017,
        db  = "training_data",
        sentiment = "training_sentiment_collection",
        food = "training_food_collection",
        service = "training_service_collection",
        ambience = "training_ambience_collection",
        cost = "training_cost_collection",
        tag = "training_tag_collection",
)

results_data = dict(
        ip = "localhost",
        port = 27017,
        db = "results",
        reviews = "reviews",
        eateries = "eateries"
    )


celery = dict(
        celery_redis_broker_ip = "localhost",
        celery_redis_broker_port = 6379,
        celery_redis_broker_db_number = 0,
)

debug = dict(
        all = True,
        results = False,
        execution_time = True,
        print_docs = False,
    )



t_connection = pymongo.MongoClient(training_data["ip"], training_data["port"])
sentiment_collection = t_connection[training_data["db"]][training_data["sentiment"]]
tag_collection = t_connection[training_data["db"]][training_data["tag"]]
food_collection = t_connection[training_data["db"]][training_data["food"]]
service_collection = t_connection[training_data["db"]][training_data["service"]]
cost_collection = t_connection[training_data["db"]][training_data["cost"]]
ambience_collection = t_connection[training_data["db"]][training_data["ambience"]]
corenlp_collection = t_connection[corenlp_data["db"]][corenlp_data["sentiment"]]


import sys
sys.path.append(corenlp_data["path_jar_files"])
import jsonrpc
server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                             jsonrpc.TransportTcpIp(addr=(corenlp_data["ip"],
                                                          corenlp_data["port"]
                                                          )))






