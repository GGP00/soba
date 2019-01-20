from occupant import EmergencyOccupant
from fire import FireControl
import random
import datetime as dt
from soba.models.continuousModel import ContinuousModel
from time import time
import time as sp
from ast import literal_eval as make_tuple
import listener as lstn
from avatar import EmergencyAvatar
from soba.agents.resources.e_greedy_q_learning import State, Qlearning
import numpy as np
import pickle
from copy import deepcopy
import itertools


class SEBAModel(ContinuousModel):
    """
    Base Class to create simulation models  on emergency situations in buildings.
        Attributes:
            Those inherited from the ContinuousModel class.
            adults: List of all adult EmergencyOccupant objects created.
            children: List of all children EmergencyOccupant objects created.
            emergency: Control of the start of the emergency. 
            FireControl: FireControl Object.
            fireTime: Date and time of the start of the emergency
            outDoors: Listing of exit doors of the building
        Methods:
            createOccupants: Create one occupant object of the EmergencyOccupant type.
            createEmergencyAvatar: Create one avatar object of the EmergencyAvatar type.
            isThereFire: Evaluate if there is fire in a position.
            informEmergency: Launches the state of emergency.
            harmOccupant: Damages an occupant with the fire that is in its same position.
            getUncrowdedGate: Get the path to the uncrowded exit door.
            getSafestGate: Get the path to the safest exit door.
            getNearestGate: Get the path to the nearest exit door.
            step: Execution of the scheduler steps.

    """

    def __init__(self, width, height, jsonMap, jsonsOccupants, sebaConfiguration, seed = int(time())):
        lstn.setModel(self)
        super().__init__(width, height, jsonMap, jsonsOccupants, seed = seed, timeByStep = 60)
        self.adults = []
        self.children = []
        self.emergency = False
        self.FireControl = False
        today = dt.date.today()
        self.fireTime = sebaConfiguration.get('hazard') or dt.datetime(today.year, today.month, 1, 8, 30, 0, 0)
        self.outDoors = []
        self.getOutDoors()
        self.familiesJson = sebaConfiguration.get('families')
        self.createOccupants(jsonsOccupants)
        self.uncrowdedStr = []
        self.occupEmerg = []      
        self.itemPos = [item.pos for item in self.generalItems]
        self.complete_exits  = dict()
        self.exit_combinations = []
        self.stack_exits_cells()
        
        # In case 'strategy': 'q_learning' and 'qlearning_files' hasn't been updated -> True, otherwise -> False.
        update_qlearning_files = False
        
        if update_qlearning_files:
            self.create_qlearning_pickles()
        
    def stack_exits_cells (self):
        id = 0
        exits = []
        for i in range(len(self.exits)):
            if i%2==0:
                id += 1
                exits.append(id)
                self.complete_exits.update({id:[self.exits[i], self.exits[i+1]]})
        for i in range(1,len(exits)+1):
              combination = [list(x) for x in itertools.combinations(exits, i)]
              self.exit_combinations.extend(combination)
        
    def getOutDoors(self):
        for poi in self.pois:
            if poi.id == 'out':
                self.outDoors.append(poi)
    
    def set_qlearning_map (self, id_exits):
        '''
        exits: list of id's of each exit to be represented in the ascii map
        '''
        grid = [['.']*self.width for i in range(self.height)]
        
        for door in self.doors:
            x, y = door.pos1
            grid[y][x] = 'd'
            x,y = door.pos2
            grid[y][x] = 'd'
        for x,y in self.itemPos:
            grid[y][x] = '#'
        for k,v in self.complete_exits.items():
            if k in id_exits:
                for x,y in v:
                    grid[y][x]= 'x'
            else:
                for x,y in v:
                    grid[y][x]= 'f'
        for row in grid:
            print(' '.join(map(str, row))) 
        return grid
    
    def create_qlearning_pickles (self):
        
        grid_resources = {'.': [-1, False], 'w': [-1, False], 'f': [-10, False], 'x': [100, True], 'd': [-1, False], 'obstacle':'#', 'walls': self.walls}
        for ids in self.exit_combinations:
            grid = self.set_qlearning_map(ids) 
            qtable = self.create_map_qtable(grid, grid_resources)
            str_ids = ''.join(map(str, ids))
            qtable_file = './auxiliarFiles/qlearning_files/pickles/exits_'+str_ids+'_qtable'
            grid_file = './auxiliarFiles/qlearning_files/grids/exits_'+str_ids+'_grid'
            self.convert_to_pickle(qtable, qtable_file)
            self.convert_to_pickle(grid, grid_file)
        
            
    def convert_to_pickle (self, pickable_object, file_name):
        '''
        Used to convert Q-table from object to pickle
        '''
        outfile = open(file_name,'wb')
        pickle.dump(pickable_object, outfile) 
        outfile.close()

    def create_map_qtable (self, grid, grid_resources):
        start_state = State(grid=grid, agent_pos=(0,0))
        e_greedy_maze = Qlearning(
            start_state = start_state,
            grid_resources = grid_resources)
        n_episode_steps = 100 
        n_episodes = 7 * len(grid) * len(grid[0])
        start = sp.time()
        distributed_positions = [(0,0), (19,16), (0,2), (0,18), (0,16), (0,19), (6,6), (2,19), (2,0), (11,11), (10,12), (12,10), (17,7), (7,17)]
        for x, y in distributed_positions:            
            e_greedy_maze.start_state = State(grid=grid, agent_pos=(x,y))
            e_greedy_maze.learn(n_episodes=n_episodes, n_episode_steps=n_episode_steps)
        end = sp.time()
        print(f'#  create_map_qtable > Time to complete:{end - start: .2f}s = {(end - start)/60:.2f} min = {(end - start)/3600:.2f} hours')
        #e_greedy_maze.visualize_max_quality_action (e_greedy_maze.q_table)
        e_greedy_maze.q_value_ascii_action (e_greedy_maze.q_table, grid)
        
        return e_greedy_maze.q_table

    def infer_optimous_path (self, occupant, exit_ids):
        """
        Extract the optimous path from the q-table with the specified exit number from the position of the given occupant.
        This number also identifies the grid associated to that q-table. If it is not given, it chooses the general grid and
        it's q-table for the mentioned path's inferencing.
                Args:
                    occupant: Occupant instance of the agent who needs the evacuation path.
                    exit_number: list of exit's ids which must be present in the inferecing grid.
                Return: 
                    Inferenced path as list of tuples.
                    It's total reward as int.
        """
        n_episode_steps = 100 
        grid = []
        grid_resources = {'.': [-1, False], 'w': [-1, False], 'f': [-10, False], 'x': [100, True], 'd': [-1, False], 'obstacle':'#', 'walls': self.walls}

        str_ids = ''.join(map(str, exit_ids))
        grid_file = './auxiliarFiles/qlearning_files/grids/exits_'+str_ids+'_grid'
        grid = self.extract_from_pickle(grid_file)
        print(f"occupant.pos {occupant.pos}")
        inference_state = State(grid=grid, agent_pos=occupant.pos)
        e_greedy_maze = Qlearning(
            inference_state = inference_state,
            grid_resources = grid_resources )
        qtable_file = './auxiliarFiles/qlearning_files/pickles/exits_'+str_ids+'_qtable'
        e_greedy_maze.q_table = self.extract_from_pickle(qtable_file)
        path, reward = e_greedy_maze.infer_path(n_episode_steps, inference_state)
        #e_greedy_maze.visualize_inferenced_path(path) 
        return path #, reward
    
    def extract_from_pickle(self,file_name):
        '''
        Used to extract Q-table from pickle to object
        '''
        infile = open(file_name, 'rb')
        pickable_object = pickle.load(infile)
        infile.close()
        return pickable_object

    def createOccupants(self, jsonsOccupants):
        """
        Create one occupant object of the EmergencyOccupant type.
                Args:
                    jsonOccupants: Json of description of the occupants.
        """
        for json in jsonsOccupants:
            for n in range(0, json['N']):
                a = EmergencyOccupant(n, self, json)
                self.occupants.append(a)
                self.adults.append(a)
        if self.familiesJson:
            for f in self.familiesJson:
                nA = 0
                nC = 0
                n = 0
                if f.get('N') and f.get('adult'):
                    n = f.get('N')
                    nA = f.get('adult')
                    nC = n - nA
                elif f.get('adult') and f.get('child'):
                    nA = f.get('adult')
                    nC = f.get('child')
                elif f.get('N') and f.get('child'):
                    nC = f.get('child')
                    n = f.get('N')
                    nA = n - nC
                children = []
                for j in range(0, nC):
                    ch = random.choice(self.adults)
                    while ch.children:
                        ch = random.choice(self.adults)
                    children.append(ch)
                    ch.adult = False
                    self.adults.remove(ch)
                    self.children.append(ch)
                for j in range(0, nA):
                    ad = random.choice(self.adults)
                    while ad.children:
                        ad = random.choice(self.adults)
                    ad.children = children
                    for c in children:
                        c.parents.append(ad)

    def createEmergencyAvatar(self, idAvatar, pos, color = 'red', initial_state = 'walking'):
        """
        Create one avatar object of the EmergencyAvatar type.
            Args:
                idAvatar: Unique ID given to the avatar agent as int.
                pos: Initial position of the avatar as (x, y)
                color: Color assigned to the avatar as String
                initial_state: State of the avatar as String
            Return: EmergencyAvatar object
        """
        unique_id = 100000 + int(idAvatar)
        a = EmergencyAvatar(unique_id, self, pos, color, initial_state)
        self.occupants.append(a)
        return a

    def isThereFire(self, pos):
        """
        Evaluate if there is fire in a position.
            Args:
                pos: Position given as (x, y)
            Return: Boolean
        """
        for fire in self.FireControl:
            if fire.pos == pos:
                return True
        return False

    def informEmergency(self):
        """
        Launches the state of emergency.
        """
        start = sp.time()
        for occupant in self.occupants:
            self.occupEmerg.append(occupant)
            occupant.safe_exits = [*range(1,len(self.complete_exits)+1)]
            print(f"occupant.safe_exits: {occupant.safe_exits}")
            occupant.makeEmergencyAction()


            
        end = sp.time()
        print(f'# 1º model > informEmergency:{end - start: .2f}s = {(end - start)/60:.2f} min = {(end - start)/3600:.2f} hours')

        
        
    def harmOccupant(self, occupant, fire):
        """
        Damages an occupant with the fire that is in its same position.
            Args:
                occupant: EmergencyOccupant object
                fire: 
            Return: Boolean
        """
        if occupant.life > fire.grade:
            occupant.life = occupant.life - fire.grade
        else:
            occupant.life = 0
            occupant.alive = False
            if occupant in self.occupEmerg:
                self.occupEmerg.remove(occupant)
    """
    def getUncrowdedGate(self):
        fewerPeople = 1000000
        doorAux = False
        for door in self.model.outDoors:
            nPeople = 0
            x, y = door.pos
            for xAux in range (-10, 0):
                for yAux in range(-10, 10):
                    if self.model.xyInGrid(x + xAux, y + yAux):
                        items = self.model.grid.get_cell_list_contents((x + xAux, y + yAux))
                        for item in items:
                            if isinstance(item, EmergencyOccupant) and item.inbuilding:
                                nPeople = nPeople + 1
            if fewerPeople > nPeople:
                doorAux = door
                fewerPeople = nPeople
        return doorAux.pos
    """
    def getSafestGate(self, occupant):
        """
        Get the path to the safest exit door.
        """
        longPath = 0
        doorAux = ''
        for door in self.outDoors:
            for fire in self.FireControl.limitFire:
                path = occupant.getWay(door.pos, fire.pos)
                if len(path) > longPath:
                    longPath = len(path)
                    doorAux = door
        return doorAux.pos

    def getNearestGate(self, occupant):
        """
        Get the path to the safest nearest exit door.
        """
        shortPath = 1000000
        doorAux = False
        for door in self.outDoors:
            path = occupant.getWay(occupant.pos, door.pos)
            if shortPath > len(path):
                shortPath = len(path)
                pathReturn = path
                doorAux = door
        return doorAux.pos

    def getUncrowdedGate(self):
        """
        Get the path to the uncrowded exit door.
        """
        doorsN = {}
        for d in self.outDoors:
            doorsN[str(d.pos)] = 0
        for o in self.occupants:
            pos = o.pos_to_go
            doorsN[str(pos)] = doorsN[str(pos)] + 1
        for o in self.uncrowdedStr:
            pos = o.pos_to_go
            n = doorsN[str(pos)]
            naux = 100000
            doorPos = False
            for k, v in doorsN.items():
                if naux > v+1:
                    doorPos = make_tuple(k)
                    naux = v
            if doorPos:
                doorsN[str(doorPos)] = doorsN[str(doorPos)] + 1
                doorsN[str(o.pos_to_go)] = doorsN[str(o.pos_to_go)] - 1
                o.pos_to_go = doorPos
                o.movements = o.getWay()
                o.N = 0
            self.uncrowdedStr.remove(o)

    #API methods
    def positions_fire(self):
        fire = []
        if not self.FireControl:
            return fire
        for f in self.FireControl.fireExpansion:
            x, y = f.pos
            fire.append({"x": x, "y": y})
        data = {"positions": fire}
        return data

    def exit_way_avatar(self, avatar_id, strategy = 0):
        a = self.getOccupantId(int(avatar_id))
        strategies = ['nearest', 'safest', 'uncrowded', 'lessassigned']
        a.exitGateStrategy = strategy
        pos = a.getExitGate()
        positions = []
        for p in pos:
            x, y = p
            positions.append({"x": x, "y": y})
        data = {"positions": positions}
        return data

    def fire_in_pov(self, avatar_id):
        a = self.getOccupantId(int(avatar_id))
        print(a)
        pos = a.getPosFireFOV()
        print(pos)
        positions = []
        for p in pos:
            x, y = p
            positions.append({"x": x, "y": y})
        data = {"positions": positions}
        return data

    def create_avatar(self, idAvatar, pos, color = 'red', initial_state = 'walking'):
        a = self.createEmergencyAvatar(idAvatar, pos, color, initial_state)
        self.occupants.append(a)
        return a
    
    def evaluate_or_terminate (self):
        start = sp.time()
        if self.emergency:
            for occupant in self.occupants:
                if occupant.pos in self.exits or occupant.alive == False:
                    return
                occupant.verify_path(occupant.movements)  
                
                print(f"occupant.safe_exits: {occupant.safe_exits}")

                end = sp.time()
                print(f'# 1º model > evaluate_or_terminate:{end - start: .2f}s = {(end - start)/60:.2f} min = {(end - start)/3600:.2f} hours')

    def step(self):
        """
        Execution of the scheduler steps.
        """
        a = 0
        d = 0
        t = "Normal" 
        if self.emergency:
            t = "Emergency"
        for o in self.occupants:
            if o.alive:
                a = a + 1
            else:
                d = d + 1
        print("Situation: ", t, ", Occupants dead: ", d, ", Occupants alive: ", a)

        if self.emergency and not self.occupEmerg:
            print("Simulation terminated.")
            self.finishSimulation = True
            sp.sleep(1)
        if (self.clock.clock >= self.fireTime) and not self.emergency:
            self.FireControl = FireControl(100000, self, random.choice(self.pois).pos)
            self.informEmergency()
            self.emergency = True
        self.evaluate_or_terminate()
        if self.emergency and self.uncrowdedStr:
            self.getUncrowdedGate()
        super().step()
        if self.emergency:
            for occupant in self.occupants:
                fire = self.FireControl.getFirePos(occupant.pos)
                if fire != False:
                    self.harmOccupant(occupant, fire)
        if self.emergency:
            for occupant in self.occupants:
                if len(occupant.movements):
                    occupant.movements.pop(0)
