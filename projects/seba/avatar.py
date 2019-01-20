from soba.agents.avatar import Avatar

class EmergencyAvatar(Avatar):
	"""
	This class enables to create avatars that represent virtual occupants, that is, they are
	not controlled by the simulation but by an API Rest. This class inherits from the avatar class of SOBA.

	Attributes:
		Those Inherited from the Avatar class of SOBA.
		alive: Current state of an avatar, live or not.
		life: Number of remaining life points of the avatar.
	
	Methods:
		getExitGate: Obtain the optimal way to evacuate the building according to an evacuation strategy.
		getPosFireFOV: Obtain the positions in the avatar's field of vision where there is fire.
		makeEmergencyAction: Method that is invoked when initiating an emergency to make the decision of response.

	"""
	def __init__(self, unique_id, model, initial_pos, color = 'red', initial_state='walking'):
		super().__init__(unique_id, model, initial_pos, color, initial_state)

		self.alive = True
		self.life = 3
		self.getFOV()

	def getExitGate(self):
		'''
		Obtain the optimal way to evacuate the building according to an evacuation strategy.
			Return: List of positions (x, y)
		'''
		if True:
			if self.exitGateStrategy == 'uncrowded':
				self.pos_to_go = self.model.getNearestGate(self)
				self.model.uncrowdedStr.append(self)
			elif self.exitGateStrategy == 'safest':
				self.pos_to_go = self.model.getSafestGate(self)
			elif self.exitGateStrategy == 'nearest':
				self.pos_to_go = self.model.getNearestGate(self)
			elif self.exitGateStrategy == 'lessassigned':
				self.pos_to_go = self.model.getLessAssignedGate()
			else:
				self.pos_to_go = self.model.getNearestGate(self)
		else:
			self.pos_to_go = self.model.getNearestGate(self)
		pathReturn = super().getWay()
		return pathReturn

	def makeEmergencyAction(self):
		"""
		Method that is invoked when initiating an emergency to make the decision of response.
		"""
		pass

	def getPosFireFOV(self):
		'''
		Check if the position is in my field of vision
			Return: List of positions (x, y)
		'''
		others = []
		print('fov', self.fov)
		for pos in self.fov:
			print(self.model.FireControl.movements)
			if pos in self.model.FireControl.movements:
				others.append(pos)
		return others

	def step():
		pass