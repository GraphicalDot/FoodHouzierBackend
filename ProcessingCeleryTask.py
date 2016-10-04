#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import
import celery
from celery import states
from celery.task import Task, subtask
from celery.utils import gen_unique_id, cached_property
from celery.decorators import periodic_task
from datetime import timedelta
from celery.utils.log import get_task_logger
import time
import pymongo
import random
from celery.registry import tasks
import logging
import inspect
from celery import task, group
from sklearn.externals import joblib
import time
import os
import sys
import time
import hashlib
import itertools
from compiler.ast import flatten
from collections import Counter
from itertools import groupby
from operator import itemgetter
from ProductionEnvironmentApi import EachEatery, PerReview, DoClusters


logger = logging.getLogger(__name__)



from __Celery_APP.App import app
from Text_Processing import WordTokenize, PosTaggers, SentenceTokenizationOnRegexOnInterjections, bcolors, NounPhrases,\
                NERs, NpClustering

                
@app.task()
class ReturnResultWorker(celery.Task):
	max_retries=3, 
	acks_late=True
	default_retry_delay = 5
        def run(self, eatery_id):
                eatery_instance = MongoScriptsEateries(eatery_id)
                result = eatery_instance.get_noun_phrases(category, 30) 
        
        def after_return(self, status, retval, task_id, args, kwargs, einfo):
		#exit point of the task whatever is the state
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- with time taken\
                        --<{time}>-- seconds  {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, 
                            time=time.time() -self.start, reset=bcolors.RESET))
		pass

	def on_failure(self, exc, task_id, args, kwargs, einfo):
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- failed fucking\
                        miserably {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, reset=bcolors.RESET))
                logger.info("{0}{1}".format(einfo, bcolors.RESET))
		self.retry(exc=exc)


@app.task()
class CleanResultBackEnd(celery.Task):
	ignore_result = True
	max_retries=3, 
	acks_late=True
	default_retry_delay = 5
        def run(self, id_list):
                """
                To start this worker,
                This worker has a Queue CleanResultBackEndQueue
                celery -A ProcessingCeleryTask  worker -n CleanResultBackEndOne -Q CleanResultBackEndQueue --concurrency=4 --loglevel=info
                It cleans the results which are being stored into the mongodb which is the result backend
                for the celery.

                Variables in Scope:
                    id_list:It is the list of all the ids who are being executed by several celery nodes

                """
                self.start = time.time() 
                connection = pymongo.Connection(MONGO_REVIEWS_IP, MONGO_REVIEWS_PORT)
                celery_collection_bulk = connection.celery.celery_taskmeta.initialize_unordered_bulk_op()
                
                for _id in id_list:
                        celery_collection_bulk.find({'_id': _id}).remove_one()

                try:
                    celery_collection_bulk.execute()
                except BulkWriteError as bwe:
                    print(bwe.details)
                connection.close()
                return 

        def after_return(self, status, retval, task_id, args, kwargs, einfo):
		#exit point of the task whatever is the state
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- with time taken\
                        --<{time}>-- seconds  {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, 
                            time=time.time() -self.start, reset=bcolors.RESET))
                pass

	def on_failure(self, exc, task_id, args, kwargs, einfo):
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- failed fucking\
                        miserably {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, reset=bcolors.RESET))
                logger.info("{0}{1}".format(einfo, bcolors.RESET))
		self.retry(exc=exc)



@app.task()
class DoClustersWorker(celery.Task):
        max_retries=3,
        acks_late=True
        default_retry_delay = 5
        def run(self, eatery_id):
                """
                celery -A ProcessingCeleryTask  worker -n DoClustersWorker -Q DoClustersQueue --concurrency=4 \
                        -P gevent  --loglevel=info --autoreload
                """
                self.start = time.time()
                do_cluster_ins = DoClusters(eatery_id=eatery_id)
                do_cluster_ins.run()
                return 


        def after_return(self, status, retval, task_id, args, kwargs, einfo):
                #exit point of the task whatever is the state
                logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- with time taken\
                        --<{time}>-- seconds  {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__,
                            time=time.time() -self.start, reset=bcolors.RESET))
                pass

        def on_failure(self, exc, task_id, args, kwargs, einfo):
                logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- failed fucking\
                        miserably {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, reset=bcolors.RESET))
                logger.info("{0}{1}".format(einfo, bcolors.RESET))
                self.retry(exc=exc)



@app.task()
class PerReviewWorker(celery.Task):
	max_retries=3, 
        ignore_result=False
	acks_late=True
	default_retry_delay = 5
	def run(self, __list, eatery_id):
                    """
                    celery -A ProcessingCeleryTask  worker -n PerReviewWorker -Q PerReviewQueue --concurrency=4 -P\
                            gevent  --loglevel=info --autoreload
                    """
                    self.start = time.time()
                    review_id = __list[0]
                    review_text = __list[1]
                    review_time = __list[2]
                    per_review_instance = PerReview(review_id, review_text, review_time, eatery_id)
                    per_review_instance.run() 
                    return 
            
        def after_return(self, status, retval, task_id, args, kwargs, einfo):
		#exit point of the task whatever is the state
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- with time taken\
                        --<{time}>-- seconds  {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, 
                            time=time.time() -self.start, reset=bcolors.RESET))
		pass

	def on_failure(self, exc, task_id, args, kwargs, einfo):
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- failed fucking\
                        miserably {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, reset=bcolors.RESET))
                logger.info("{0}{1}".format(einfo, bcolors.RESET))
		self.retry(exc=exc)

        


@app.task()
class EachEateryWorker(celery.Task):
	max_retries=3, 
	acks_late=True
        ignore_result=False
	default_retry_delay = 5
	def run(self, eatery_id):
                start = time.time()
                """
                Start This worker:
                celery yA ProcessingCeleryTask  worker -n EachEateryWorker -Q EachEateryQueue --concurrency=4\
                        -P gevent  --loglevel=info --autoreload
               
                """ 
                ins = EachEatery(eatery_id=eatery_id)
                return ins.return_non_processed_reviews()

@app.task()
class MappingListWorker(celery.Task):
        ignore_result=False
	max_retries=0, 
	acks_late=True
	default_retry_delay = 5
        
        def run(self, __review_list, __eatery_id, __callback):
                """
                celery -A ProcessingCeleryTask  worker -n MappingListWorker -Q MappingListQueue --concurrency=4 -P \
                        gevent  --loglevel=info --autoreload
                """
                self.start = time.time()
                callback = subtask(__callback)
	       
                print __eatery_id
                return group(callback.clone([arg, __eatery_id]) for arg in __review_list)()
        
        def after_return(self, status, retval, task_id, args, kwargs, einfo):
		#exit point of the task whatever is the state
		logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- with time taken\
                        --<{time}>-- seconds  {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, 
                            time=time.time() -self.start, reset=bcolors.RESET))
		pass

	def on_failure(self, exc, task_id, args, kwargs, einfo):
		print args
                logger.info("{color} Ending --<{function_name}--> of task --<{task_name}>-- failed fucking\
                        miserably {reset}".format(color=bcolors.OKBLUE,\
                        function_name=inspect.stack()[0][3], task_name= self.__class__.__name__, reset=bcolors.RESET))
                logger.info("{0}{1}".format(einfo, bcolors.RESET))
		self.retry(exc=exc)



@app.task()
class StartProcessingChainWorker(celery.Task):
        max_retries=3
        acks_late=True
        default_retry_delay = 5
        print "http://docs.celeryproject.org/en/latest/userguide/monitoring.html"

        def run(self, eatery_id_list=None):
                if not eatery_id_list:
                        eatery_id_list = [post.get("eatery_id") for post in eateries.find()]
                        
                self.start = time.time()
                #process_list = eateries_list.s(url, number_of_restaurants, skip, is_eatery)| dmap.s(process_eatery.s())
                for eatery_id in eatery_id_list:
                        process_list = EachEateryWorker.s(eatery_id)| MappingListWorker.s(eatery_id, PerReviewWorker.s())
                        process_list()
                return


        def after_return(self, status, retval, task_id, args, kwargs, einfo):
                #exit point of the task whatever is the state
                pass

        def on_failure(self, exc, task_id, args, kwargs, einfo):
                logger.info("{0}{1}".format(einfo, bcolors.RESET))
                self.retry(exc=exc)


