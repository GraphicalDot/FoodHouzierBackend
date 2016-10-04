 address_mappings = {"area":
                        {"properties": {
                            "address_autocomplete": { 'analyzer': 'custom_analyzer', 'type': 'string'},
                            "address": {"type": "string", "copy_to": ["address_autocomplete"]},

                            "location": {"type": "string"}}}}





body = {"_source": ["address"],
                        "from": 0,
                        "size": 50,
                                "query": {
                                        "match": {
                                                "address_autocomplete": {
                                                        "query":    "new delhi",
                                                        "analyzer": "standard"
                                                                    }
                                                }
                                        }
                            }
