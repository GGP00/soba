import json
import soba.launchers.listener as ltnr
from tornado import httpserver
from tornado.ioloop import IOLoop
import tornado.web


"""

    In this file, the API is defined to obtain information about the simulation and control of avatars.
    Specifically, the API provide the next requests:

        /api/v1/seba/positions_fire
            Returns all the position where there is fire as a list of [x, y].

        /api/v1/seba/exit_way_avatar/id&strategy
            Return the best way to be exit of the building in function of a given strategy.
            The way is given as a list of [x, y].

        /api/v1/seba/create_avatar/id&x,y
            Create an avatar on the position (x, y) on the grid.
            
        Where:
            id is a number with the unique_id of an occupant.
            x and y are the two numbers with the grid coordinates.
            strategy is one string of the next options: uncrowded, safest, nearest or lessassigned

"""

# Simulation model
global model
model = None

#Get methods without query
class positions_fire(tornado.web.RequestHandler):
    def get(self):
        global model
        data = model.positions_fire()
        response = json.dumps(data)
        self.write(response)

#Get methods with query
class exit_way_avatar(tornado.web.RequestHandler):
    def get(self, avatar_id, strategy = 1):
        global model
        data = model.exit_way_avatar(avatar_id, strategy)
        response = json.dumps(data)
        self.write(response)

class fire_in_fov(tornado.web.RequestHandler):
    def get(self, avatar_id):
        global model
        data = model.fire_in_pov(avatar_id)
        response = json.dumps(data)
        self.write(response)

#Put methods with query
class create_avatar(tornado.web.RequestHandler):
    def put(self, avatar_id):
        global model
        data = tornado.escape.json_decode(self.request.body)
        x = data["x"]
        y = data["y"]
        pos = (int(x), int(y))
        a = model.create_avatar(avatar_id, pos)
        x, y = a.pos
        data = {'avatar': {'id': a.unique_id, 'position': {'x': x, 'y': y}}}
        response = json.dumps(data)
        self.write(response)

def setModel(modelAux):
    global model
    ltnr.model = modelAux
    model = modelAux
    ltnr.externalHandlers = [
        (r"/api/seba/v1/fire?", positions_fire),
        (r"/api/seba/v1/occupants/([0-9]+)/route/([0-9]+)?", exit_way_avatar),
        (r"/api/seba/v1/occupants/([0-9]+)/fire?", fire_in_fov),
        (r"/api/seba/v1/occupants/([0-9]+)?", create_avatar)
    ]