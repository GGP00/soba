from soba.visualization.drawModelBack import BackEndVisualization
from mesa.visualization.ModularVisualization import ModularServer
import os
import soba
import tornado

"""
In the file visual.py is implemented the execution with visual representation:
	Methods: 
		-run: Execute the simulation with visual representation.

"""

def run(model, parameters, visual):
	"""
	Execute the simulation with visual representation.
		Args:
			model: Model that is simulated.
			visualJS: JS files with the visualization elements that are included in the JavaScript browser visualization template.
			params: Parameters loaded in the models about the agents and anything else.
	"""
	backEndVisualization = BackEndVisualization(int(parameters['width']), int(parameters['height']), 500, 500)
	if visual:
		listAux = [backEndVisualization] + visualJS
	else:
		listAux = [backEndVisualization]
	'''
	method = 'ModularServer(model, listAux, name="Simulation", model_params=dict('
	n=0
	for e in params:
		method = method + 'key'+ str(n) + ' = params['+str(n)+'],'
		n = n +1
	method = "".join(method[:-1])
	print(method)
	method = method + '))'
	print(method)
	server = eval(method)
	'''
	
	

	path = os.path.abspath(soba.__file__)
	path = path.rsplit('/', 1)[0]

	local_handler = (r'/local/(.*)', tornado.web.StaticFileHandler,
                     {"path": path})

	print(ModularServer.handlers)
	ModularServer.handlers = ModularServer.handlers[:-1]
	print(ModularServer.handlers)
	ModularServer.handlers = ModularServer.handlers + [local_handler]
	print(ModularServer.handlers)
	server = ModularServer(model, listAux, name="Simulation", model_params=parameters)
	server.port = 7777
	server.launch()