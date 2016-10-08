#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
#CELERY_DEFAULT_QUEUE = 'default'
#Exchange Type can be specified by [providing type keyword while intializing Exchange with the key word type
#The four options for this type can be
#"direct"
#"topic"
#"fanout"
#"header"
To start a ReviewIdToSentTokenizeOne, This workers returns a list of lists of the form
(id, sentence, predicted_tag, predicted_sentiment)
celery -A ProcessingCeleryTask  worker -n ReviewIdToSentTokenizeOne -Q ReviewIdToSentTokenizeQueue --concurrency=4 --loglevel=info
    
To start a MappingList worker, This worker just executes a parelled exectuion on the result returned by ReviewIdToSentTokenizeQueue 
by mappping each element of the result to each SentTokenizeToNPQueue worker
celery -A ProcessingCeleryTask  worker -n MappingListOne -Q MappingListQueue --concurrency=4 --loglevel=info
    
To start a CleanResultBackEnd worker, This worker takes a liat of parent task, task and its children and removes
their enteries from the result backend which in our case is mongodb, We cannot make our result backend off, because
then the states of taks remain pending, The api doesnt wait for this task to complete, this tasks runs in background
celery -A ProcessingCeleryTask  worker -n CleanResultBackEndOne -Q CleanResultBackEndQueue --concurrency=4 --loglevel=info

To start SentTokenizeToNP worker, This worker does all the heavy lifting, This gets a list of the form 
(id, sentence, predicted_tag, predicted_sentiment)

and returns a list of the form 
    celery -A ProcessingCeleryTask  worker -n SentTokenizeToNP -Q SentTokenizeToNP --concurrency=4 --loglevel=info
"""
from __future__ import absolute_import
import os
from kombu import Exchange, Queue
from celery.schedules import crontab
import sys
import ConfigParser

file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_path)
sys.path.append(parent_dir)
print parent_dir
os.chdir(parent_dir)
from configs import redis_config, celery_backend
os.chdir(file_path)


CELERY_IMPORTS = ("ProcessingCeleryTask", )
BROKER_URL= 'redis://{host}:{port}/{db_number}'.format(host=redis_config["ip"],\
        port=redis_config["port"],  db_number=redis_config["db"])


CELERY_QUEUES = (
		Queue('ProcessingWorkerQueue', Exchange('mapping_list',
                    delivery_mode= 2),  routing_key='ProcessingWorkerQueue.import'),
                    )


#And your routes that will decide which task goes where:
CELERY_ROUTES = {
		'ProcessingCeleryTask.ProcessingWorker': {
				'queue': 'ProcessingWorkerQueue',
				'routing_key': 'ProcessingWorkerQueue.import',
                        },		
		
                        }
#Celery result backend settings, We are using monngoodb to store the results after running the tasks through celery
CELERY_RESULT_BACKEND = 'mongodb'

# mongodb://192.168.1.100:30000/ if the mongodb is hosted on another sevrer or for that matter running on different port or on different url on 
#the same server

CELERY_MONGODB_BACKEND_SETTINGS = {
		'host':  celery_backend["ip"],
		'port':  celery_backend["port"],
		'database': 'celery',
		'taskmeta_collection': 'celery_taskmeta',
			}


CELERYD_POOL_RESTARTS = True
#How many messages to prefetch at a time multiplied by the number of concurrent processes. The default is 4 
#(four messages for each process). The default setting is usually a good choice, however – if you have very 
#long running tasks waiting in the queue and you have to start the workers, note that the first worker to 
#start will receive four times the number of messages initially. Thus the tasks may not be fairly distributed 
#to the workers.
CELERYD_PREFETCH_MULTIPLIER = 1
CELERY_CONCURRENCY = 1
CELERY_ACKS_LATE = True
#CELERY_RESULT_ENGINE_OPTIONS = {'echo': True}
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT=['application/json']
CELERY_ENABLE_UTC = True
#CELERYD_CONCURRENCY = 20
#CELERYD_LOG_FILE="%s/celery.log"%os.path.dirname(os.path.abspath(__file__))
CELERY_DISABLE_RATE_LIMITS = True


#CELERY_ALWAYS_EAGER = True, this is setup for local development and debugging. This setting tells Celery to 
#process all tasks synchronously which is perfect for running our tests and working locally so we don't have 
#to run a separate worker process. You'll obviously want to turn that off in production.
#CELERY_EAGER_PROPAGATES_EXCEPTIONS = True
#CELERY_ALWAYS_EAGER = True

#CELERY_IGNORE_RESULT = True
CELERY_TRACK_STARTED = True

#Added because of the pobable soultion of the problem 
#InconsistencyError: 
#    Cannot route message for exchange 'celery': Table empty or key no longer exists.
#    Probably the key ('_kombu.binding.celery') has been removed from the Redis database.
#While running chunks on Classification task
SEND_TASK_SENT_EVENT = True
#CELERYMON_LOG_FORMAT =  [%(asctime)s: %(levelname)s/%(processName)s] 
