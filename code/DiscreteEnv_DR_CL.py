from typing import List, Dict, Union
from dataclasses import dataclass
import pickle
import json
import time
import random
from itertools import chain
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from HybridPPO.hybridppo import HybridPPO
from sb3_contrib import RecurrentPPO
import gym
from gym.spaces import Box, Discrete, Tuple
from gym.utils import seeding
import simpy
from simpy.events import AnyOf, AllOf, Event

from utils_DR import *

num_skipping_stop = 0
num_total_stop = 0


'''
Action： Discrete(3) + Continuous(2)
Action space: Discrete(14)
Holding: 12                  # 12 discrete holding times 0, 10, 20, 30, ...
Skipping: 1
Turning Aroung: 1
Total: 14
'''

NUM_ACTION = 14
SKIPPING_ACTION = NUM_ACTION - 2
TURNING_AROUND_ACTION = NUM_ACTION - 1
HOLDING_INTERVAL = 10
OBSERVATION_DIM = 6


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def action(self, obs):
        return self.action_space.sample()
    
pax_data = {bus_id : [] for bus_id in range(N_BUS)}
# event_buffer = {bus_id: {'ready': False, 'events': []} for bus_id in range(N_BUS)}

class Bus:
    def __init__(self, env, simpy_env, idx, starting_time) -> None:
        self.env = env
        self.simpy_env = simpy_env
        self.idx = idx
        self.capacity = CAPACITY
        self.cur_station = self.env.stations[0]
        self.next_station = self.cur_station
        #self.next_travel_time = self.env.get_travel_time(self.cur_station)
        self.next_travel_time = 0
        self.starting_time = starting_time
        # print(f'Bus {self.idx} is successfully initialized.')
        self.proc = self.simpy_env.process(self.drive())
        self.passengers = []
        self.num_pax = 0
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.action = None
        self.taking_action = False
        self.last_departure_time = starting_time
        self.action_start_time = None


    def drive(self):
        # hold at the terminal before starting to drive
        self.env.departure_times.append(self.starting_time)
        yield self.simpy_env.timeout(self.starting_time)
        self.env.data[self.cur_station.idx].append(self.simpy_env.now)
        pax_data[self.idx].append(self.num_pax)
        with open('env_data5.pkl', 'wb') as f:
            pickle.dump(self.env.data, f)

        # each cycle is a trip from one station to the next
        while True:
            turn_around = False

            # drive till the next station
            yield self.simpy_env.timeout(self.next_travel_time)
            if (self.simpy_env.now > self.starting_time) and (self.next_station.idx == 0):#terminal station?
                if (self.simpy_env.now < self.env.departure_times[-1] + HEADWAY):
                    self.env.departure_times.append(self.env.departure_times[-1] + HEADWAY)
                    yield self.simpy_env.timeout(self.env.departure_times[-2] + HEADWAY - self.simpy_env.now)
            self.env.data[self.next_station.idx].append(self.simpy_env.now)
            pax_data[self.idx].append(self.num_pax)

            # if self.next_station.idx == 0:
            #     print(f'Bus {self.idx} starts at {self.simpy_env.now}')
            
            # request to enter the station
            self.env.ready = True
            self.taking_action = True
            self.env.acting_bus = self.idx
            self.action_start_time = self.simpy_env.now
            self.env.allow_skipping = (not any([pax.alight_station == self.next_station.idx for pax in self.passengers]))
            self.env.allow_turning_around = (self.next_station.idx != 0)

            yield self.simpy_env.timeout(0)

            # high-level and low-level actions
            action = self.env.action
            self.action = action
            
            # action < NUM_ACTION - 2 means holding
            if action < NUM_ACTION - 2:# TODO middle station cannot use holding method? why
                with self.next_station.request() as req:
                    yield req
                    self.station_enter_time = self.simpy_env.now # TODO useless? station_enter_time = self.env.departure_times[-1] + HEADWAY
                #print(f'Bus {self.idx} Arrives At Station {self.next_station.idx}')
                
                    if self.env.allow_skipping:
                        global num_skipping_stop
                        num_skipping_stop = num_skipping_stop + 1
                    global num_total_stop               
                    num_total_stop += 1
                

                    if self.next_station.idx in [N_STATION-1, int(N_STATION//2)-1]:
                        pax_alight = self.alight_all_pax(self.next_station)
                        pax_board = 0
                    else:
                        pax_alight = self.alight_pax(self.next_station)
                        pax_board = self.board_pax(self.next_station)
                    holding_time = max(pax_board * t_b, pax_alight * t_a)
                    holding_time += action * HOLDING_INTERVAL
                    self.station_holding_time = holding_time

                    yield self.simpy_env.timeout(holding_time)
                    self.update_state(action)
                    #print(f'Bus {self.idx} holds at station {self.cur_station.idx} for {holding_time} seconds')
                
            # SKIPPING_ACTION means skipping
            elif action == SKIPPING_ACTION:
                yield self.simpy_env.timeout(0)
                self.alight_pax(self.next_station)
                self.update_state(action)
                #print(f'Bus {self.idx} skips station {self.cur_station.idx}')

            # TURNING_AROUND_ACTION means turning around
            else: 
                turn_around = True

            if turn_around:
                # alight all passengers
                pax_alight = self.alight_all_pax(self.next_station) # then alight all other passengers
                holding_time = pax_alight * t_a
                # wait to alight all passengers
                yield self.simpy_env.timeout(holding_time)

                # turn around
                self.update_state(action)

                # immediately turn around
                yield self.simpy_env.timeout(0)

                #print(f'Bus {self.idx} turns around at station {self.cur_station.idx} to {self.next_station.idx} for {l_action} seconds')
            self.taking_action = False
            self.last_departure_time = self.simpy_env.now

    def alight_all_pax(self, station):
        """
        1, alight all passengers that are on the bus
        2, return the number of passengers that are leaving
        3, update the number of passengers on the bus to 0
        4, update passengers' last_start_time

        Return:
            the number of passengers that are boarding
        """
        passengers_left = []
        for pax in self.passengers:
            if pax.alight_station != station.idx:
                pax.bus = None
                passengers_left.append(pax)
                pax.new_status = 2
                pax.last_arrival_time = self.simpy_env.now
            else:
                pax.new_status = 3
        num_pax = len(self.passengers)
        station.passengers.extend(passengers_left)
        station.passengers.sort(key=lambda x: x.start_time)
        self.passengers = []
        self.num_pax = 0
        return num_pax


    def alight_pax(self, station):
        """
        1, alight the passengers that are supposed to get off
        2, return the number of passengers that are leaving
        3, update the number of passengers on the bus

        Return:
            the number of passengers that are boarding
        """
        pax_alight = 0
        passengers = []
        for pax in self.passengers:
            if pax.alight_station == station.idx:
                pax_alight += 1
                pax.new_status = 3  
            else:
                passengers.append(pax)
        self.passengers = passengers
        self.num_pax = len(passengers)
        return pax_alight

    def board_pax(self, station):
        """
        1, move the passengers to the bus
        2, return the number of passengers that are boarding
        3, update the arrival time of the last bus
        4, remove the passengers that are boarding from the station
        5, update passengers' on station time

        Parameters:
            station: the station that the bus is arriving at
        
        Return:
            the number of passengers that are boarding
        """
        STOP_BOARDING = False
        passengers = []
        n = 0
        for pax in station.passengers:
            if (pax.start_time < self.simpy_env.now) and (pax.bus is None) and (not STOP_BOARDING) and (not pax.status==4):
                pax.bus = self.idx
                self.passengers.append(pax)
                self.num_pax += 1
                n += 1
                pax.new_status = 1
                if self.num_pax == self.capacity:
                    STOP_BOARDING = True
            else:
                passengers.append(pax)
                
        station.last_arrival_time = self.simpy_env.now
        station.passengers = passengers
        self.env.num_board_pax += n
        return n

    def update_state(self, action):
        if action == TURNING_AROUND_ACTION:
            self.cur_station = self.next_station
            self.next_station = self.next_station.get_opposite()
            self.next_travel_time = 0
        else:
            self.cur_station = self.next_station
            self.next_station = self.next_station.get_next()
            self.next_travel_time = self.env.get_travel_time(self.cur_station)

    def get_observation(self):
        # state should include [location, headway, ego, pax, action, action_duration]
        if self.starting_time > self.simpy_env.now:
            location = 0
        else:
            if self.taking_action:
                location = self.next_station.idx * STATION_DIST
            else:
                if self.next_travel_time == 0:
                    location = self.cur_station.idx * STATION_DIST
                else:
                    location = min(1, (self.simpy_env.now - self.last_departure_time) / self.next_travel_time) * STATION_DIST + self.cur_station.idx * STATION_DIST
        location = location % N_STATION
        action_duration = self.simpy_env.now - self.action_start_time if self.taking_action else 0
        return {'location': location,
                'pax': self.num_pax,
                'action': self.action if self.taking_action else None,
                'action_duration': action_duration,}

class Station:
    def __init__(self, 
                 env,
                 simpy_env, 
                 idx,
                 pax_alight,
                 pax_board) -> None:
        self.env = env
        self.simpy_env = simpy_env
        self.resource = simpy.Resource(simpy_env, capacity=1)
        self.idx = idx
        self.last_station = None
        self.next_station = None
        self.opposite_station = None
        self.pax_alight = pax_alight
        self.pax_board = pax_board
        self.passengers = self.generate_pax(self.pax_alight, self.pax_board)
        self.env.passengers.extend(self.passengers)
        self.last_arrival_time = 0

    def set_last(self, station):
        self.last_station = station

    def set_next(self, station):
        self.next_station = station

    def set_opposite(self, station):
        self.opposite_station = station

    def get_last(self):
        return self.last_station

    def get_next(self):
        return self.next_station

    def get_opposite(self):
        return self.opposite_station

    def request(self):
        return self.resource.request()

    def generate_pax(self, pax_alight, pax_board):
        return [Passenger(pax_board[i], pax_alight[i], pax_board[i], pax_board[i]) for i in range((pax_board != np.inf).sum())] 

    def update_pax(self):
        t = self.simpy_env.now
        self.passengers = [pax for pax in self.passengers if t - pax.start_time < MAX_WAITING_TIME and pax.new_status==0]

    def get_num_pax(self):
        t = self.simpy_env.now
        return len([pax for pax in self.passengers if pax.start_time < t])


total_pax_num_sys=[]
total_pax_num_on_bus=[]
total_pax_num_leave=[]
waiting_time_list=[]
on_bus_time_list=[]
indiv_waiting_time_list=[]
env_now_list=[]
status_list=[]
action_list=[]
all_pax_waiting_times = []



class Env(gym.Env):
    def __init__(self, holding_only=False, skipping_only=False, turning_only=False, mode='headway', difficulty_level=1) -> None:
        self.travel_times = TABLE_TRAVEL_TIME
        self.pax_alight = PAX_ALIGHT_TABLE
        self.pax_board = PAX_ARRIVE_TABLE
        self.env = simpy.Environment()
        self.ready = False
        self.action = None
        self.departure_times = []
        self.passengers = []
        self.num_board_pax = 0
        self.acting_bus = None
        self.allow_skipping = False
        self.allow_turning_around = False
        self.data = {station_idx: [] for station_idx in range(N_STATION)}

        self.stations = [Station(self, self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(N_BUS)]
        
        self.last_timestep = 0
        self.acc_waiting_time = 0
        self.acc_on_bus_time = 0

        
        self.mode = mode
        self.holding_only = holding_only
        self.skipping_only = skipping_only
        self.turning_only = turning_only
        
        self.observation_space = Box(low=-1e4, high=1e4, shape=(N_BUS, OBSERVATION_DIM))
        if holding_only:
            self.action_space = Discrete(NUM_ACTION - 2)
        elif skipping_only:
            self.action_space = Discrete(2)
        elif turning_only:
            self.action_space = Discrete(2)
        else:
            self.action_space = Discrete(NUM_ACTION)
            
        self.difficulty_level = difficulty_level
        self.last_difficulty_increase = 0
        self.adjust_for_difficulty()

        self.action_list = []

        # run simulation until the first event
        while not self.ready:
            self.env.step()
        self.ready = False

        # self.action_space = Discrete(NUM_ACTION)
        self.action_mask = [x < SKIPPING_ACTION for x in range(self.action_space.n)]
        
        self.reset()

    def increase_difficulty(self):
        print("Increasing Difficulty")
        self.difficulty_level += 1
        if self.difficulty_level==1:
            print("Currently Difficulty Level 1")
            self.action_mask = [x < SKIPPING_ACTION for x in range(self.action_space.n)]
        elif self.difficulty_level==2:
            print("Currently Difficulty Level 2")
            self.action_mask = [x < 2 for x in range(self.action_space.n)]
        elif self.difficulty_level==3:
            print("Currently Difficulty Level 3")
            self.action_mask = [x < 3 for x in range(self.action_space.n)]
        elif self.difficulty_level==4:
            print("Currently Difficulty Level 4")
            self.action_mask = [x < 4 for x in range(self.action_space.n)]
        elif self.difficulty_level==5:
            print("Currently Difficulty Level 5")
            self.action_mask = [x < 5 for x in range(self.action_space.n)]
        elif self.difficulty_level==6:
            print("Currently Difficulty Level 6")
            self.action_mask = [x < 6 for x in range(self.action_space.n)]
        elif self.difficulty_level==7:
            print("Currently Difficulty Level 7")
            self.action_mask = [x < 7 for x in range(self.action_space.n)]
        elif self.difficulty_level==8:
            print("Currently Difficulty Level 8")
            self.action_mask = [x < 8 for x in range(self.action_space.n)]
        elif self.difficulty_level==9:
            print("Currently Difficulty Level 9")
            self.action_mask = [x < 10 for x in range(self.action_space.n)]
        elif self.difficulty_level==10:
            print("Currently Difficulty Level 10")
            self.action_mask = [x < TURNING_AROUND_ACTION for x in range(self.action_space.n)]
        else:
            print("Currently Difficulty Level 11")
            self.action_mask = [True for x in range(self.action_space.n)]    
    def adjust_for_difficulty(self):
        # if self.difficulty_level==1:
        #     self.holding_only = True
        # elif self.difficulty_level==2:
        #     self.holding_only = False
        #     self.action_space = Discrete(NUM_ACTION-1)
        # else:
        #     self.holding_only = False
        #     self.action_space = Discrete(NUM_ACTION)
        
        if self.difficulty_level==1:
            print("Currently Difficulty Level 1")
            self.action_mask = [x < SKIPPING_ACTION for x in range(self.action_space.n)]
        elif self.difficulty_level==2:
            print("Currently Difficulty Level 2")
            self.action_mask = [x < TURNING_AROUND_ACTION for x in range(self.action_space.n)]
        else:
            print("Currently Difficulty Level 3")
            self.action_mask = [True for x in range(self.action_space.n)]



    def reset(self):
        self.env = simpy.Environment()
        self.passengers = []
        self.stations = [Station(self, self.env, i, self.pax_alight[i], self.pax_board[i]) for i in range(N_STATION)]
        self.arange_stations()
        self.buses = [Bus(self, self.env, i, BUS_SCHEDULE[i]) for i in range(N_BUS)]
        self.ready = False
        #self.action = (0, (0, 0))
        self.action = 0
        self.departure_times = []
        self.data = {station_idx: [] for station_idx in range(N_STATION)}
        self.action_mask = [x < SKIPPING_ACTION for x in range(self.action_space.n)]

        self.acc_waiting_time = 0
        self.acc_on_bus_time = 0
        
        # run simulation until the first event
        while not self.ready:
        # while self.env.now < 3600:
            self.env.step()
            # self.get_reward(mode='waiting_time')
        self.ready = False

        self.last_timestep = self.env.now
        obs = self.get_observation()
        for pax in self.passengers:
            if pax.last_time < self.env.now:
                pax.last_time = self.env.now
        if self.should_increase_difficulty():
            self.difficulty_level += 1
            self.last_difficulty_increase = self.env.now
            self.adjust_for_difficulty()
        return obs
    def should_increase_difficulty(self):
        return self.env.now - self.last_difficulty_increase > 1000
    def arange_stations(self) -> None:
        for index, station in enumerate(self.stations):
            if index == len(self.stations)-1:
                station.set_next(self.stations[0])
            else:
                station.set_next(self.stations[index+1])
            
            if index == 0:
                station.set_last(self.stations[-1])
            else:
                station.set_last(self.stations[index-1])

            station.set_opposite(self.stations[len(self.stations) - index - 1])

    def step(self, action):
        # if self.holding_only:
        #     #action = (0, (action, 0))
        #     action = action
        if self.skipping_only:
            # action = (action, (0, 0))## 12
            action = SKIPPING_ACTION
        # elif self.turning_only:
        #     if action:
        #         action = (2, (0, 0))
        #     else:
        #         action = (0, (0, 0))
        elif self.turning_only:
            if action < SKIPPING_ACTION:
                action = TURNING_AROUND_ACTION
            else:
                action = 0
        self.action = action
        self.action_list.append(self.action)
        with open('action_list5.pkl', 'wb') as f:
            pickle.dump(self.action_list, f)
        while not self.ready:
            self.env.step()
            self.update_pax()
        # print("Environment Step")
        self.ready = False
        
        timestep = self.env.now - self.last_timestep
        self.last_timestep = self.env.now

        obs = self.get_observation()
        rewards = self.get_reward(obs=obs[:, 2].flatten())
        done = self.env.peek() >= HORIZON - 2000
        info = {'timestep': float(timestep)}
        
        return obs, float(rewards), bool(done), info

    def update_pax(self):
        for station in self.stations:
            station.update_pax()

    # def action_masks(self):
    #     return [True, self.allow_skipping, False]

    @staticmethod
    def extract_observations(obs):
        out = np.zeros((len(obs), OBSERVATION_DIM))
        index = 0
        for k, (i, ob) in enumerate(obs):
            action = ob['action']
            action_duration = 0 if ob['action_duration'] is None else ob['action_duration']
            action = 0 if ob['action'] is None else ob['action']
            processed_ob = [ob['ego'], ob['location'], ob['headway'], ob['pax'], action_duration] + [action]
            # processed_ob = [ob[0] if type(ob)==np.ndarray else ob for ob in processed_ob]
            out[k] = processed_ob
            if ob['ego'] == 1:
                index = k
        # print('index: ', index)
        out = np.roll(out, -index, axis=0)
        # out = np.array([out[-1], out[0]])
        return out


    def get_observation(self):
        def getHeadWay(self, loc1, loc2):
            t = int(self.env.now // TRAVEL_TIME_STEP)
            station1 = int(loc1 // STATION_DIST)
            station2 = int(loc2 // STATION_DIST)
            if station2 != station1:
                if station2 > station1:
                    headway = sum([self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)] for station1 in range(station1, station2)])
                else:
                    headway = sum([self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)] for station1 in chain(range(station1, N_STATION), range(0, station2))])
                headway += (loc2 - station2 * STATION_DIST) / STATION_DIST * self.travel_times[station2, int(self.env.now // TRAVEL_TIME_STEP)]
                headway += ((station1 + 1) * STATION_DIST - loc1) / STATION_DIST * self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)]
            else:
                headway = (loc2 - loc1) / STATION_DIST * self.travel_times[station1, int(self.env.now // TRAVEL_TIME_STEP)]
            if type(headway) == np.ndarray:
                return headway[0]
            return headway
            
        obs = {}
        for bus in self.buses:
            ob = bus.get_observation()
            if bus.idx == self.acting_bus:
                # print(self.acting_bus)
                # print(bus.idx)
                ob['ego'] = 1
            else:
                ob['ego'] = 0
            obs[bus.idx] = ob
        obs = sorted(obs.items(), key=lambda x: x[1]['location'])
        for index, (bus, ob) in enumerate(obs):
            ob['headway'] = getHeadWay(self, ob['location'], obs[(index+1)%len(obs)][1]['location'])
        
        obs = self.extract_observations(obs)
        # obs = obs.reshape((-1)) # flatten the observation (96,)
        return obs


    def get_reward(self, obs=None):
        alpha, beta = 1, 1
        waiting_time = 0
        on_bus_time = 0
        n_leave_pax=0
        n_waiting_pax = 0
        n_on_bus_pax = 0
        global all_pax_waiting_times
        for pax in self.passengers:
            status_list.append(pax.status)
            if pax.status in [0, 2] and pax.last_time < self.env.now:
                waiting_time += self.env.now - pax.last_time
                #indiv_waiting_time_list.append(self.env.now - pax.last_time)
                all_pax_waiting_times.append(self.env.now - pax.last_arrival_time)
                if self.env.now - pax.last_arrival_time >= 0.8*HEADWAY: 
                    random_bit = random.choice([0, 1])
                    if random_bit == random.choice([0, 1])==1:
                        n_leave_pax+=1
                        pax.status=4 # leave
                else:
                    n_waiting_pax += 1
            elif pax.status == 1 and pax.last_time < self.env.now:
                on_bus_time += self.env.now - pax.last_time
                n_on_bus_pax += 1
            if pax.last_time < self.env.now and pax.status in [0,1,2,3]:
                pax.last_time = self.env.now
                pax.status = pax.new_status
        # print('total num of pax in the sys: ', len([pax for pax in self.passengers if pax.start_time < self.env.now and pax.status != 2 and pax.status != 4]))
        # print('total num of pax on bus: ', len([pax for pax in self.passengers if pax.status == 1]))    
        # print('total num of pax leave: ', len([pax for pax in self.passengers if pax.start_time < self.env.now  and pax.status == 4])) 

        total_pax_num_sys.append(len([pax for pax in self.passengers if pax.start_time < self.env.now and pax.status != 2 and pax.status != 4]))
        total_pax_num_on_bus.append(len([pax for pax in self.passengers if pax.status == 1]))
        total_pax_num_leave.append(len([pax for pax in self.passengers if pax.start_time < self.env.now  and pax.status == 4]))
        env_now_list.append(self.env.now)
       
        waiting_time_list.append(waiting_time)
        on_bus_time_list.append(on_bus_time)
        
        #print('num of pax leave: ', n_leave_pax)
           
        reward = alpha * waiting_time / max(1, n_waiting_pax) + beta * on_bus_time / max(1, n_on_bus_pax)
        self.acc_waiting_time += waiting_time
        self.acc_on_bus_time += on_bus_time
        if self.mode == 'waiting_time_total':
            return (- waiting_time - on_bus_time) / 1e6
        elif self.mode == 'waiting_time_station':
            return - waiting_time / 1e6
        elif self.mode == 'num_pax':
            reward = self.num_board_pax
            self.num_board_pax = 0
            return reward
        else: 
            reward = -obs.std() ** 2
            return reward

    def get_travel_time(self, station1):
        return self.travel_times[station1.idx, int(self.env.now // TRAVEL_TIME_STEP)]
    
    def action_masks(self):
        return self.action_mask


with open('pax_data_DR_CL2.pkl', 'wb') as f:
    pickle.dump(pax_data, f)
with open('action_list_DR_CL.pkl', 'wb') as f:
    pickle.dump(action_list, f)

@dataclass
class Passenger:
    start_time: float
    alight_station: int
    last_time: float
    last_arrival_time: float
    bus: Bus = None
    status: int = 0 # 0: waiting, 1: on bus, 2: alighted and waiting, 3: alighted, 4: leave
    new_status: int = 0 


if __name__ == '__main__':
    TARGET_AVG_REWARD = -1.04
    def progress_criteria(avg_reward):
        return avg_reward >= TARGET_AVG_REWARD

    # Initialize environment
    env = Env(**{'holding_only': False, 'skipping_only': True, 'turning_only': False, 'mode': 'waiting_time_station'})
    # Initialize agent
    agent = RandomAgent(env.action_space)

    # Number of episodes
    n_episodes = 10
    # Initialize dictionary to store total rewards for each level
    rewards_dict = {i: [] for i in range(n_episodes)}

    # Loop over episodes
    for i_episode in range(n_episodes):

        # Reset state
        observation = env.reset()

        # One episode loop
        for t in range(1000):
            # Agent makes action
            action = agent.action(observation)
            # Apply action to the environment
            observation, reward, done, info = env.step(action)

            # If the episode is done, reset the environment
            if done:
                print(f"Episode finished after {t+1} timesteps")
                break
        obs = env.reset()
        action = 0  
        while env.env.peek() < HORIZON - 15000:
            obs, reward, done, info = env.step(action)  # Execute action
            rewards_dict[i].append(reward)  # Store reward
            # Progress criteria check and level up if criteria met
            if len(rewards_dict[i]) >= 100 and progress_criteria(sum(rewards_dict[i][-100:]) / 100):
                print(f"Progress criteria met at level {i+1}, moving to next level.")
                break  # Progress to the next difficulty level
            # if done:
            #     print(f"Failed to meet progress criteria at level {i+1}, repeating level.")
            #     obs = env.reset()  # Reset environment for the same level
            #     rewards_dict[i] = []  # Clear rewards for the current level

    # Initialize figure for plotting
    plt.figure(figsize=(10, 6))

    # Plot the rewards progress for each level
    for i, level in enumerate(n_episodes):
        plt.plot(rewards_dict[i], label=f"Level {i+1}: {level}")

    # Add title, labels, and legend to the plot
    plt.title("Rewards Progress for Each Level")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    # Display the plot
    plt.show()
                
    # env = Env(**{'holding_only': False, 'skipping_only': True, 'turning_only': False, 'mode': 'waiting_time_station'})
    # env.reset()
    # print(env.mode)
    # action = (0, (0, 0))
    # total_reward = 0
    # cnt = 0
    # print(time.time())
    # while env.env.peek() < HORIZON - 2000:
    #     action = random.randint(0, 1)
    #     obs, rew, done, info = env.step(action)
    #     total_reward += rew
    #     cnt += 1
    #     # if r < 0.1 and env.allow_skipping:
    #     #     action = (1, 0)
    #     # elif r < 0.2:
    #     #     action = (2, 1)
    #     # else:
    #     #     action = policy(obs)
    #     # action = env.action_space.sample()   
    #     # action = (1, (0, 0)) if env.allow_skipping else (0, (0, 0))
    #     # action = (0, (0, 0))
    #     # print(f'Current time: {env.env.now}')
    # print(env.data)
    # pickle.dump(env.data, open('data.pkl', 'wb'))
    # pickle.dump(pax_data, open('pax.pkl', 'wb'))
    with open('data_DR_CL.pkl', 'wb') as f:
        pickle.dump(env.data, f)
    with open('pax_data_DR_CL.pkl', 'wb') as f:
        pickle.dump(pax_data, f)
    with open('action.pkl', 'wb') as f:
        pickle.dump(action_list, f)
    print(time.time())
    print('Total waiting time: ', env.acc_waiting_time)

    print('N_BUS: ',N_BUS)
    print('HEADWAY: ',HEADWAY)
    print('N_STATION: ',N_STATION)
    
    print('Avg passenger waiting time: ', np.mean(all_pax_waiting_times))
    print('Stdev passenger waiting time: ', np.std(all_pax_waiting_times))

    print('Total on bus time: ', env.acc_on_bus_time)
    print('stops allowed to skip: ', num_skipping_stop, ' ', num_total_stop)
    print('Total reward: ', total_reward)
    print('Cnt: ', cnt)
