
sentiment_dict = { "average": {"type":"long"},
                                            "excellent": {"type":"long"},
                                            "good": {"type":"long"},
                                            "mixed": {"type":"long"},
                                            "poor": {"type":"long"},
                                            "terrible": {"type":"long"},
                                            "timeline": {"type":"string"},
                                            "total_sentiments": {"type":"long"}
    }

        {"test":
            {"mappings":
                {"test":
                    {"properties":
                        {"ambience":
                            {"properties":
                                {"ambience-null":
                                    {"properties":
                                            sentiment_dict
                                            },
                                    
                                    "ambience-overall":
                                        {"properties":
                                            sentiment_dict    
                                            },
                                        "crowd":
                                            {"properties":
                                                sentiment_dict,    
                                                },
                                        "dancefloor":
                                            {"properties":
                                                   sentiment_dict,  
                                                    },
                                        "decor":
                                            {"properties":
                                                    sentiment_dict,     
                                                },
                                        "in-seating":{"properties":
                                                sentiment_dict,
                                            },
                                        "live-matches":
                                            {"properties":
                                                sentiment_dict,
                                                },
                                        "location":{
                                            "properties":
                                                sentiment_dict,  
                                            },
                                    "music":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},
                                    "open-area":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},
                                    "romantic":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},
                                    "smoking-zone":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},
                                    "sports":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},
                                    "view":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}}}},
                                
                        "cost":
                            {"properties":
                                {"cheap":
                            {"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},
                            "expensive":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},
                            "not worth":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},
                            "vfm":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}}}},
                                
                        "cuisines":{"type":"string"},
                        "dropped_nps":{"type":"string"},
                        "eatery_address":{"type":"string"},
                        "eatery_area_or_city":{"type":"string"},
                        "eatery_cuisine":{"type":"string"},
                        "eatery_highlights":{"type":"string"},
                        "eatery_id":{"type":"string"},
                        "eatery_known_for":{"type":"string"},
                        "eatery_longitude_latitude":{"type":"string"},
                        "eatery_name":{"type":"string"},
                        "eatery_type":{"type":"string"},
                        "food":
                            {"properties":
                                    {"dishes":{
                                            "properties":
                                            {"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"name":{"type":"string"},"poor":{"type":"long"},"similar":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},
                                                "name":{"type":"string"},
                                                "poor":{"type":"long"},
                                                "terrible":{"type":"long"},
                                                "timeline":{"type":"string"}}},
                                                "terrible":{"type":"long"},"
                                                timeline":{"type":"string"},
                                                "total_sentiments":{"type":"long"}}},
                                            "overall-food":{"properties":{"average":{"type":"long"},
                                                "excellent":{"type":"long"},
                                                "good":{"type":"long"},
                                                "mixed":{"type":"long"},
                                                "poor":{"type":"long"},
                                                "terrible":{"type":"long"},
                                                "timeline":{"type":"string"},
                                                "total_sentiments":{"type":"long"}}}}},
                                        "menu":{"properties":{"average":{"type":"long"},
                                            "excellent":{"type":"long"},
                                            "good":{"type":"long"},
                                            "mixed":{"type":"long"},
                                            "poor":{"type":"long"},
                                            "terrible":{"type":"long"},
                                            "timeline":{"type":"string"},
                                            "total_sentiments":{"type":"long"}}},
                                        "old_considered_ids":{"type":"string"},"overall":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},"places":{"type":"string"},"processed_reviews":{"type":"string"},"service":{"properties":{"booking":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},"management":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},"presentation":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},"serivce-null":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},"servic-overall":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},"service-charges":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"total_sentiments":{"type":"long"}}},"service-null":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},"service-overall":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},"staff":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}},"waiting-hours":{"properties":{"average":{"type":"long"},"excellent":{"type":"long"},"good":{"type":"long"},"mixed":{"type":"long"},"poor":{"type":"long"},"terrible":{"type":"long"},"timeline":{"type":"string"},"total_sentiments":{"type":"long"}}}}}}}}}}
