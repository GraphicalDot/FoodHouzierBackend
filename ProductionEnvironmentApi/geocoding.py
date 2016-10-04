#!/usr/bin/env python
import os
import sys

"""
file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_path)
sys.path.append(parent_dir)
os.chdir(parent_dir)    
from connections import reviews, eateries, reviews_results_collection, eateries_results_collection, discarded_nps_collection, bcolors, short_eatery_result_collection
os.chdir(file_path)  

from connections import eateries_results_collection, short_eatery_result_collection, reviews_results_collection, eateries, reviews, discarded_nps_collection
"""
from geopy.geocoders import Nominatim
geolocator = Nominatim()

##address = [[post.get("location"), post.get("eatery_address")] for post in eateries_results_collection.find()]


def geo_decode(latitude, longitude, address):
        address = address.lower()
        if int(latitude) == 0:
                print "latitude i 0"
                """
                try:
                        location = geolocator.geocode(address)
                        latitude, longitude = location.latitude, location.longitude
                except Exception as e:
                        try:
                                __address  = address.split(",")[1: ]
                                location = geolocator.geocode(__address)
                                latitude, longitude = location.latitude, location.longitude
                        except Exception as e:
                                try:
                                        __address  = address.split(",")[2: ]
                                        location = geolocator.geocode(__address)
                                        latitude, longitude = location.latitude, location.longitude
                                except Exception as e:
                                        try:
                                                __address  = address.split(",")[1: ]
                                                location = geolocator.geocode(__address)
                                                latitude, longitude = location.latitude, location.longitude
                                        except :
                                                pass
                """
                for i in range(0, len(address.split(","))):
                                        try:
                                                __address  = ", ".join(address.split(",")[i: ])
                                                print "trying %s"%__address
                                                location = geolocator.geocode(__address)
                                                latitude, longitude = location.latitude, location.longitude
                                                break  
                                        except:
                                                pass



        if int(latitude) == 0:
                    return (latitude, longitude, address, None)
        try:
                location = geolocator.reverse("%s, %s"%(latitude, longitude))
                __area = location.raw.get("address")
        except Exception as e:
                    print e
                    print latitude, longitude, address
                    return (latitude, longitude, address, None)
        if not __area.get("suburb"):
                suburb = address.split(",")[-2]
                print "Trying suburb by %s"%suburb
                __area.update({"suburb": suburb})

        for (key, value) in __area.iteritems():
                __area.update({key: value.lower()})

        return (latitude, longitude, address, __area)        

"""
print geo_decode(0.0, 0.0, "Shop 19, 1st Floor, Leisure Valley Road")
"""

















