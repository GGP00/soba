from mesa.visualization.ModularVisualization import VisualizationElement
import collections

class Visualization(VisualizationElement):

	def __init__(self, cellW, cellH,canvas_width=500, canvas_height=500):
		self.grid_width = cellW
		self.grid_height = cellH
		self.canvas_width = canvas_width
		self.canvas_height = canvas_height

		new_element = ("new VisualModule({}, {}, {}, {})"
			.format(self.canvas_width, self.canvas_height,
				self.grid_width, self.grid_height ))

		self.js_code = "elements.push(" + new_element + ");"

	def render(self, model):
		data = collections.defaultdict(list)
		data[1] = list()
		if model.FireControl != False:
			for fire in model.FireControl.fireExpansion:
				x, y = fire.pos
				data[1].append({"x" : x, "y" : y})
		return data