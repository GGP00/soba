from unittest import TestCase
import json, requests
from jsonschema import validate
import socket
import unittest


ipServer = socket.gethostbyname(socket.gethostname())

port = "10000"
URLBASE = "http://127.0.0.1:" + port
URISOBA = "/api/soba/v1/occupants"
URISEBA = "/api/seba/v1/occupants"
URIFIRE = "/api/seba/v1/fire"
stringTemplate = {"type": "string"}
numberTemplate = {"type": "number"}

N = 1

class APITest(TestCase):

    def test_general_gets(self, *args, **kwargs):
        
        print(str('Testing {}').format('GET /api/soba/v1/occupants'))
        template = {
            "type": "object",
            "properties": {
                "occupants": {
                    "type": "array"
                    }
            },
            "required": ["occupants"]
        }

        for i in range(N):
            url = URLBASE + URISOBA
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)
            for o in datajson["occupants"]:
                validate(o, numberTemplate)

        print(str('Testing {}').format('GET /api/soba/v1/occupants/movements'))
        template = {
            "type": "object",
            "properties": {
                "orientation": {
                    "type": "string"
                    },
                "speed": {
                    "type": "number"
                    }
            },
            "required": ["orientation", "speed"]
        }

        template2 = {
            "type": "object"
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/movements"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template2)
            for k, v  in datajson.items():
                validate(k, stringTemplate)
                validate(int(k), numberTemplate)
                validate(v, template)

        print(str('Testing {}').format('GET /api/soba/v1/occupants/positions'))
        template = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number"
                    },
                "y": {
                    "type": "number"
                    }
            },
            "required": ["x", "y"]
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/positions"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            for k, v in datajson.items():
                validate(k, stringTemplate)
                validate(int(k), numberTemplate)
                validate(v, template)


        print(str('Testing {}').format('GET /api/soba/v1/occupants/states'))
        for i in range(N):
            url = URLBASE + URISOBA + "/states"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            for k,v in datajson.items():
                validate(v, stringTemplate)
                validate(k, stringTemplate)
                validate(int(k), numberTemplate)

        print(str('Testing {}').format('GET /api/soba/v1/occupants/{id}'))
        template = {
            "type": "object",
            "properties": {
                "occupant":{
                    "type": "object",
                    "properties": {
                            "state":{
                                "type": "string"
                            },
                            "fov": {
                                "type": "array"
                            },
                            "unique_id":{
                                "type": "string"
                            },
                            "movement": {
                                "type": "object",
                                "properties": {
                                    "orientation":{
                                        "type": "string"
                                    },
                                    "speed":{
                                        "type": "number"
                                    },
                                },
                                "required": ["orientation", "speed"]
                            },
                            "position": {
                                "type": "object",
                                "properties": {
                                    "x":{
                                        "type": "number"
                                    },
                                    "y":{
                                        "type": "number"
                                    }
                                },
                                "required": ["x", "y"]
                            }
                    },
            "required": ["state", "fov", "unique_id", "movement", "position"]
                }
            },
            "required": ["occupant"]
        }

        template2 = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number"
                    },
                "y": {
                    "type": "number"
                }
            },
            "required": ["x", "y"]
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(0)
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)
            validate(int(datajson['occupant']['unique_id']), numberTemplate)
            print(template)
            for p in datajson['occupant']['fov']:
                validate(p, template2)


        print(str('Testing {}').format('GET /api/soba/v1/occupants/{id}/movement'))
        
        template = {
            "type": "object",
            "properties": {
                "movement":{
                    "type": "object",
                    "properties": {
                            "orientation": {
                                "type": "string"
                            },
                            "speed": {
                                "type": "number"
                            }
                    },
                "required": ["orientation", "speed"]
                }
            },
            "required": ["movement"]
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(0) + "/movement"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)

        print(str('Testing {}').format('GET /api/soba/v1/occupants/{id}/position'))
        template = {
            "type": "object",
            "properties": {
                "position":{
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number"
                            },
                        "y": {
                            "type": "number"
                        }
                    },
                    "required": ["x", "y"]
                }
            },
            "required": ["position"]
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(0) + "/position"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)


        print(str('Testing {}').format('GET /api/soba/v1/occupants/{id}/state'))
        template = {
            "type": "object",
            "properties":{
                "state": {
                    "type": "string"
                }
            },
            "required": ["state"]
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(0) + "/state"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)


        print(str('Testing {}').format('GET /api/soba/v1/occupants/{id}/fov'))
        template = {
            "type": "object",
            "properties": {
                "fov": {
                    "type": "array"
                    }
            },
            "required": ["fov"]
        }
        

        template2 = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number"
                    },
                "y": {
                    "type": "number"
                }
            },
            "required": ["x", "y"]
        }

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(0) + "/fov"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)
            for p in datajson['fov']:
                validate(p, template2)



        print(str('Testing {}').format('PUT /api/soba/v1/occupants/{id}'))
        template = {
            "type": "object",
                "properties": {
                    "avatar":{
                        "type": "object",
                        "properties": {
                            "position":{
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "number",
                                    },
                                    "y": {
                                        "type": "number"
                                    }
                                },
                                "required": ["x", "y"]
                            },
                            "id":{
                                "type": "number"
                            }
                    },
                    "required": ["position", "id"]
                }
            },
            "required": ["avatar"]
        }

        dataBody = {"x": 10, "y": 10}

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(0)
            data = requests.put(url, json=dataBody, headers={'Content-Type': "application/json", 'Accept': "application/json"})
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)

        print(str('Testing {}').format('POST /api/soba/v1/occupants/{id}/position'))
        template = {
            "type": "object",
                "properties": {
                    "avatar":{
                        "type": "object",
                        "properties": {
                            "position":{
                                "type": "object",
                                "properties": {
                                    "x": {
                                        "type": "number",
                                    },
                                    "y": {
                                        "type": "number"
                                    }
                                },
                                "required": ["x", "y"]
                            },
                            "id":{
                                "type": "number"
                            }
                    },
                    "required": ["position", "id"]
                }
            },
            "required": ["avatar"]
        }
    
        dataBody = {"x": 11, "y": 11}

        for i in range(N):
            url = URLBASE + URISOBA + "/" + str(100000) + "/position"
            data = requests.post(url, json=dataBody, headers={'Content-Type': "application/json", 'Accept': "application/json"})
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)

        print(str('Testing {}').format('GET /api/seba/v1/occupants/{id}/route/{route_id}'))
        template = {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array"
                    }
            }
        }

        template2 = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number"
                    },
                "y": {
                    "type": "number"
                    }
            },
            "required": ["x", "y"]
        }

        for i in range(N):
            url = URLBASE + URISEBA + "/" + str(100000) + "/route/1"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)
            for m in datajson["positions"]:
                validate(m, template2)


        print(str('Testing {}').format('PUT /api/seba/v1/occupants/{id}'))
        template = {
            "type": "object",
            "properties": {
                "avatar": {
                    "type": "object",
                    "properties":{
                        "position":{
                            "type": "object",
                            "properties":{
                                "x": {
                                    "type": "number"
                                },
                                "y": {
                                    "type": "number"
                                }
                            },
                            "required": ["x", "y"]
                        },
                        "id": {
                            "type": "number"
                        }
                    },
                    "required": ["position", "id"]
                }
            },
            "required": ["avatar"]
        }

        dataBody = {"x": 13, "y": 13}

        for i in range(N):
            url = URLBASE + URISEBA + "/" + str(1)
            data = requests.put(url, json=dataBody, headers={'Content-Type': "application/json", 'Accept': "application/json"})
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)


        print(str('Testing {}').format('GET /api/seba/v1/occupants/{id}/fire'))
        template = {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array"
                    }
            },
            "required": ["positions"]
        }

        template2 = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number"
                    },
                "y": {
                    "type": "number"
                    }
            },
            "required": ["x", "y"]
        }

        for i in range(N):
            url = URLBASE + URISEBA + "/" + str(100000) + "/fire"
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)
            for m in datajson["positions"]:
                validate(m, template2)


        print(str('Testing {}').format('GET /api/seba/v1/fire'))
        template = {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array"
                    }
            },
            "required": ["positions"]
        }

        template2 = {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number"
                    },
                "y": {
                    "type": "number"
                    }
            },
            "required": ["x", "y"]
        }

        for i in range(N):
            url = URLBASE + URIFIRE
            data = requests.get(url)
            datajson = data.json()
            print("Response: ", datajson)
            validate(datajson, template)
            for m in datajson["positions"]:
                validate(m, template2)



if __name__ == "__main__":
      unittest.main()

