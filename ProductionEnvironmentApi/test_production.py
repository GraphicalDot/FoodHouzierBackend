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

file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_path)
sys.path.append(parent_dir)

os.chdir(parent_dir)
from connections import reviews, eateries, reviews_results_collection, eateries_results_collection, bcolors
os.chdir(file_path)
from prod_heuristic_clustering import ProductionHeuristicClustering




if __name__ == "__main__":
        review_list = eateries_results_collection.find_one({"eatery_id": "6542"}).get("processed_reviews")
        food = [reviews_results_collection.find_one({"review_id": review_id})["food_result"] for review_id in  review_list]
        flatten_food = list(itertools.chain(*food))

        __sub_tag_dict = dict()
        for (sent, tag, sentiment, sub_tag, nps, review_time)  in flatten_food:
                if not __sub_tag_dict.has_key(sub_tag):
                        __sub_tag_dict.update({sub_tag: [[sentiment, sent, nps, review_time]]})

                else:
                    __old = __sub_tag_dict.get(sub_tag)
                    __old.append([sentiment, sent, nps, review_time])
                    __sub_tag_dict.update({sub_tag: __old})


        dishes_sentences = __sub_tag_dict.get("dishes")
        __sentiment_np_time = [(sentiment, nps, review_time) for (sentiment, sent, nps, review_time) in dishes_sentences if nps]
        __sentences = [sent for (sentiment, sent, nps, review_time) in dishes_sentences if nps]

        ins = ProductionHeuristicClustering(__sentiment_np_time, "dishes", __sentences)
        i = ins.run()
        print ins.run()[0:2]


