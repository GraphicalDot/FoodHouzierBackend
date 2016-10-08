#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import
import os
from celery import Celery

# instantiate Celery object

# Optional configuration, see the application user guide.

# import celery config file



#: Set default configuration module name
#os.environ.setdefault('CELERY_CONFIG_MODULE', 'celeryconfig')

app = Celery(include=['ProcessingCeleryTask'])
#app.config_from_envvar('CELERY_CONFIG_MODULE')
import celeryconfig
app.config_from_object('celeryconfig')

if __name__ == '__main__':
    app.start()
