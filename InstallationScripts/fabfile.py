#!/usr/bin/env python


from __future__ import with_statement
from fabric.api import task, show, local, settings, prefix, abort, run, cd, env, require, hide, execute
from fabric.contrib.console import confirm
from fabric.network import disconnect_all
from fabric.colors import green as _green, yellow as _yellow, red as _red
from fabric.contrib.files import exists
from fabric.utils import error
import os
import time
from fabric.operations import local as lrun, run



env.use_ssh_config = True
#env.hosts = ["ec2-54-68-29-37.us-west-2.compute.amazonaws.com"] ##For t2 medium
env.hosts = ["ec2-54-186-203-98.us-west-2.compute.amazonaws.com"] ##For m3.large
env.user = "ubuntu"
env.key_filename = "/home/k/Programs/Canworks/new_canworks.pem"
env.warn_only = True

"""
This is the file which remotely makes an ec2 instance for the use of this repository
"""

@task
def basic_setup():
        env.run("sudo apt-get update")
        env.run("sudo apt-get upgrade")
        env.run("sudo apt-get install -y python-pip")
        env.run("sudo apt-get install -y libevent-dev")
        env.run("sudo apt-get install -y python-all-dev")
        env.run("sudo apt-get install -y ipython")
        env.run("sudo apt-get install -y libxml2-dev")
        env.run("sudo apt-get install -y libxslt1-dev")
        env.run("sudo apt-get install -y python-setuptools python-dev build-essential")
        env.run("sudo apt-get install -y libxml2-dev libxslt1-dev lib32z1-dev")
        env.run("sudo apt-get install -y python-lxml")
        #Dependencies for installating sklearn
        env.run("sudo apt-get install -y build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base")
        env.run("sudo apt-get install -y python-matplotlib")
        #Dependencies for installating scipy
        env.run("sudo apt-get install -y liblapack-dev libatlas-dev gfortran")
        env.run("sudo apt-get install -y libatlas-base-dev gfortran build-essential g++ libblas-dev")
        env.run("sudo apt-get install build-essential libssl-dev libffi-dev")
        #Dependicies to install hunpostagger
        env.run("sudo apt-get install -y ocaml-nox")
        env.run("sudo apt-get install -y mercurial")
        env.run("sudo apt-get install libhdf5-dev")
        #env.run("sudo pip install numpy")
        #env.run("sudo  pip install scipy")
        #env.run("sudo pip install scikit-learn")
        #env.run("sudo pip install matplotlib")






@task
def localhost():
    env["user"] = "kaali"
    env.run = lrun
    env.hosts = ['localhost']

@task
def another_localhost():
    env["user"] = "kaali"
    env.run = lrun
    env.hosts = ['192.168.1.2']

@task
def remote():
    env.run = run
    #env.hosts = ['52.66.155.19']
    env.hosts = ['52.66.85.208']
    env.use_ssh_config = True
    env.user = "ubuntu"
    env.key_filename = "/Users/kaali/Downloads/DataProcessingHouzier.pem"
    env.warn_only = True
    env.port = 22



@task 
def install_corenlp_server():
        Directory = raw_input("Please enter the full path of the directory\
                              where you want to install corenlp server, Plase\
                              dont slash afterwards : ")

        with cd(Directory):
                env.run("git clone https://github.com/Wordseer/stanford-corenlp-python.git")
                with cd(stanford-corenlp-python):
                        env.run("wget http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip")
                        env.run("unzip stanford-corenlp-full-2014-08-27.zip")
                        env.run("python setup.py install")
            



        return 


def update_git():
	"""
	This method will be run everytime the git repository is updated on the main machine.This clones the pushed updated 
	repository on the git on the remote server
	"""
	with prefix("cd /home/ubuntu/"):
		run("git pull origin master")



def reboot():
	run("sudo reboot")



def deploy():
	disconnect_all()


