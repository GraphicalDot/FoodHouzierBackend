#!/usr/bin/env pypy
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
from os.path import dirname, abspath
import sys
import time
import hashlib
import itertools
from compiler.ast import flatten
from collections import Counter
from itertools import groupby
from operator import itemgetter

parentdir= dirname(dirname(abspath(__file__)))
sys.path.append(parentdir)

from SaveMemory.process_eateries import ClassifyReviews

logger = logging.getLogger(__name__)



from houzier_celery_app.App import app

                
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
class ProcessingWorker(celery.Task):
        ignore_result=False
	max_retries=0, 
	acks_late=True
	default_retry_delay = 5
        
        def run(self, __eatery_id, path):
                """
                celery -A ProcessingCeleryTask  worker -n ProcessingWorker -Q ProcessingWorkerQueue --concurrency=4 -P gevent  --loglevel=info --autoreload
                """
                self.start = time.time()
	       
                print __eatery_id
                instance = ClassifyReviews([eatery_id], path)
                instance.run()
                #return group(callback.clone([arg, __eatery_id]) for arg in __review_list)()
        
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





