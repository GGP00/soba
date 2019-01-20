import random
import sys
from mesa.agent import Agent
import soba.agents.resources.aStar as aStar

class Fire():
    """
    This class enables to create fire object on a position. 
    The objects of this class are controlled by one FireControl object.

    Attributes:
        grade: Intensity level of the fire.
        pos: Fire position.
    """

    def __init__(self, model, pos):
        self.pos = pos
        model.grid.place_agent(self, pos)
        self.grade = 1

class FireControl(Agent):
    """
    This class enables to create agents that control the fire expansión, representing the emergency threat.

    Attributes:
        fireExpansion: Set of Fire objects belonging to this FireControl.
        limitFire: Fire objects that are in the limit to make the expansion.
        expansionRate: Rate of expansion of the threat.
        growthRate: Value of growth in intensity of the fire.

    Methods:
        createFirePos: Create a Fire object in a given position.
        getFirePos: Get a Fire object in a position given.
        expansionFire:  Make the expansion of fire limits.
        growthFire: Make the growth in intensity of the fire.
        step: Method invoked by the Model scheduler in each step. 
    
    """
    def __init__(self, unique_id, model, posInit, expansionRate = 1, growthRate = 1):
        super().__init__(unique_id, model)
        self.model.schedule.add(self)
        self.fireExpansion = []
        self.limitFire = []
        self.expansionRate = expansionRate #m/s
        self.growthRate = growthRate
        self.N = 0
        self.movements = []
        self.costMovement = 0.5*(1/self.expansionRate)*(1/self.model.clock.timeByStep)
        self.costGrowth = 0.5*(1/self.growthRate)*(1/self.model.clock.timeByStep)
        self.createFirePos(posInit)

    def createFirePos(self, pos):
        """
        Create a Fire object in a given position.
            Args:
                pos: Position to put the Fire object as (x, y)
        """
        f = Fire(self.model, pos)
        self.limitFire.append(f)
        self.fireExpansion.append(f)
        self.movements.append(pos)

    def getFirePos(self, pos):
        """
        Get a Fire object in a position given.
            Args: 
                pos: Position to be checked.
            Return: Fire object or False
        """
        for fire in self.fireExpansion:
            if fire.pos == pos:
                return fire
        return False

    def expansionFire(self):
        """
        Make the expansion of fire limits.
        """
        fireExpansionAux = self.limitFire[:]
        n = 0
        for fire in fireExpansionAux:
            n = n +1
            self.limitFire.remove(fire)
            x, y = fire.pos
            posAdj = [(x + 1, y + 1), (x + 1, y), (x - 1, y), (x - 1, y - 1), (x, y + 1), (x, y - 1), (x - 1, y + 1), (x + 1, y - 1)]
            doorsPoss = aStar.doorsPoss
            for pos in posAdj:
                xaux, yaux = pos
                if self.model.xyInGrid((xaux, yaux)):
                    cellPos = fire.pos
                    posAux = pos
                    move = True
                    for wall in self.model.walls:
                        if (cellPos in wall.block1 and posAux in wall.block1) or (cellPos in wall.block2 and posAux in wall.block2) or (cellPos in wall.block3 and posAux in wall.block3):
                            move = False
                    if not move:
                        for doorsPos in doorsPoss:
                            if ((cellPos in doorsPos) and (posAux in doorsPos)):
                                move = True
                    if move:
                        if not (pos in self.movements):
                            self.createFirePos(pos)

    def growthFire(self):
        """
        Make the growth in intensity of the fire.
        """
        for fire in self.fireExpansion:
            fire.grade = fire.grade + 1

    def step(self):
        """
        Method invoked by the Model scheduler in each step.
        """
        if self.costGrowth > 0:
            self.costGrowth = self.costGrowth - 1
        else:
            self.growthFire()
            self.costGrowth = 0.5*(1/self.growthRate)*(1/self.model.clock.timeByStep)
        if self.costMovement > 0:
            self.costMovement = self.costMovement - 1
        else:
            self.expansionFire()
            self.costMovement = 0.5*(1/self.expansionRate)*(1/self.model.clock.timeByStep)