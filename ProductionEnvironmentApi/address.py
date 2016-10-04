#!/usr/bin/env python

import os
import sys
import geocoder

"""
file_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_path)                                                                                                                                                                                                                                                     sys.path.append(parent_dir)                                                                                                                                                                                                                                                                 
os.chdir(parent_dir)
from connections import reviews, eateries, reviews_results_collection, eateries_results_collection, discarded_nps_collection, bcolors, short_eatery_result_collection                                                                                 
os.chdir(file_path)     




for address in list(set(result)): 
            g = geocoder.google(address)
                    try:
                                    print address, g.geojson, area_list.append((address, g.geojson.get("geometry").get("coordinates"))), "\n\n"
                                            except Exception as e:        
                                                            print "failed"


"""

address_list = list()
for post in eateries_results_collection.find():
        if post.get("google") != "No entry exists on google":
                address_list.append(post.get("google").get("eatery_address").lower())
		print post.get("google").get("pincode")[1] 
	else:
		address_list.append(post.get("eatery_address"))





result = list()
for (eatery_id, location, address, geo_address) in c:
        """
        Because sometime the address after splitting only has one element in the list 
        like  ['sector 56']
        """
        if geo_address.get("city"):
                city = geo_address.get("city")

        else:
                city = ("", geo_address.get("suburb"))[geo_address.get("suburb") != None] + ", " + ("", geo_address.get("county"))[geo_address.get("county") != None]
		print "city==%s"%(city)
		print "geo_address==%s"%geo_address
		print "address==%s"%address
		print "suburb==%s"%geo_address.get("suburb")
		print "\n"        

        try:
                area =  address.split(",")[-2]
        except Exception as e:
                area = address
        

	__result = area.lstrip() + "," +  city.lstrip()

        
	result.append(__result)



print set(result)





def if_not_location(latitude, longitude, , address):
        if int(latitude) == 0:
                g  = geocoder.google(address)
                location = g.geojson.get("bbox").get("northeast")


	g = geocoder.google(address)
        try:
                location = g.geojson.get("bbox").get("northeast")

        except Exception as e:
                for i in range(0, len(address.split(","))):
                        try:
                                __address  = ", ".join(address.split(",")[i: ])
                                print "trying %s"%__address
                                g = geocoder.google(address)
                                location = g.geojson.get("bbox").get("northeast")
                                break
                        except:
                                    pass

        return location, address




