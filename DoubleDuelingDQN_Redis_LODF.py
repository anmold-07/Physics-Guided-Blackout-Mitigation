import os
import json
import math
import copy
import numpy as np
import tensorflow as tf
from grid2op.Agent import AgentWithConverter

import sys
sys.path.append(os.path.dirname(__file__))

from DoubleDuelingDQNConfig import DoubleDuelingDQNConfig as cfg
from DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from prioritized_replay_buffer import PrioritizedReplayBuffer
from Action_Converter import Action_Converter
from grid2op.dtypes import dt_int, dt_float, dt_bool
import itertools
import csv

# for PTDF 
import numpy as np
import pandapower as pp
import pandapower.converter as pc
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci

# for LODF
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from scipy.sparse import csr_matrix as sparse 
from numpy import ones, diag, eye, r_, arange 

DEFAULT_OUTPUT_NAME = "steps_rewards.csv"

class DoubleDuelingDQN(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 grid_path: str, 
                 init_obs, 
                 powerFlowLimits,
                 name = __name__,
                 is_training = False):
        
        #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        #gpus = tf.config.experimental.list_physical_devices('GPU')
        #if gpus:
        #    try:
        #        # Currently, memory growth needs to be the same across GPUs
        #        for gpu in gpus:
        #            tf.config.experimental.set_memory_growth(gpu, True)
        #    except RuntimeError as e:
        #        print(e)

        # Call parent constructor
        AgentWithConverter.__init__(self, action_space, action_space_converter = Action_Converter)
        self.obs_space = observation_space

        # intiliazing the LODF functions        
        self.grid = pp.from_json(grid_path)
        self.init_obs = init_obs
        self.powerFlowLimits = powerFlowLimits
        _, _, intialgridbranch, _ = self.getLODFPTDFBranch()
        self.branchIndextoBusDict = self.gridInformationandBranchDict(intialgridbranch, self.init_obs, self.powerFlowLimits)
        self.parse_observation_to_grid_time_zero(self.init_obs)                         # to intialize the system

        # loading the custom action space ALL
        self.all_actions = []
        loaded_data = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+'track1_169_actions_numpy.npz') 
        actions = loaded_data['track1_169_actions_numpy']           # each action in this array is a vector action
        for action in actions:
            self.all_actions.append( action_space.from_vect(action) ) # converting each vector action into a grid2op action

        # loading the redispatch actions only
        loaded_data = np.load(os.path.split(os.path.realpath(__file__))[0]+'/'+'track1_51_redis_actions_numpy.npz') 
        self.redispatch_actions = []
        actions_redis = loaded_data["my_array"]
        # print( actions_redis.shape ) # (51, 494)
        for action in actions_redis:
            self.redispatch_actions.append( action_space.from_vect(action) )
            #print( env.action_space.from_vect(action) )
        self.redispatch_actions.pop(0) # removing the do-nothing action

        # Store constructor params
        self.name = name
        self.num_frames = cfg.N_FRAMES   # number of observations required for training
        self.is_training = is_training
        self.batch_size = cfg.BATCH_SIZE
        self.lr = cfg.LR

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []
        self.output_file = DEFAULT_OUTPUT_NAME

        # Declare training vars
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon_1 = 0.0
        self.epsilon_2 = 0.0 # to get an idea of whata epsilon to set from one simulation

        self.observation_size = 22 + 37 + 59 * 8 + 36   
        self.action_size = self.action_space.size()
        if cfg.VERBOSE:
            print ("self.obs_space.size_obs() = {}".format(self.obs_space.size_obs())) ## Attribute error?
            print ("self.observation_size = {}".format(self.observation_size))
            print ("self.action_size = {}".format(self.action_size))

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=cfg.LR_DECAY_STEPS,
                                         learning_rate_decay_rate=cfg.LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def getLODFPTDFBranch(self):
        """
        Building the LODF matrix from grid observations.
        As per the documentation of pandapower, baseMVA = 1MVA a modeling assumption.
        """
        ppc = pc.to_ppc(self.grid)
        #baseMVA, bus, gen, branch, ref, pv, pq, _, gbus, V0, ref_gens = _get_pf_variables_from_ppci(ppc)
        baseMVA, bus, gen, branch, _, _, ref, _, _, _, _, _, _ = _get_pf_variables_from_ppci(ppc)

        PTDFMatrix = self.makePTDF(baseMVA, bus, branch, ref.item())
        LODFMatrix = self.makeLODF(branch, PTDFMatrix)
        return LODFMatrix, PTDFMatrix, branch, baseMVA
    
    def makeLODF(self, branch, PTDFMatrix):
        """
        Builds the line outage distribution factor matrix via the PTDF matrix.
        """    
        nlines, nbuses = PTDFMatrix.shape
        fromBus = branch[:, F_BUS]
        toBus = branch[:, T_BUS]
        Cft = sparse((r_[ones(nlines), -ones(nlines)], (r_[fromBus, toBus], r_[arange(nlines), arange(nlines)])), (nbuses, nlines)) 
        H = PTDFMatrix * Cft
        h = diag(H, 0)
        LODF = H / (ones((nlines, nlines)) - ones((nlines, 1)) * h.T) 
        LODF = LODF - diag(diag(LODF)) - eye(nlines, nlines) 
        
        return LODF
    
    def makePTDF(self, baseMVA, bus, branch, slack):    
        """
        Returns the DC PTDF factor matrix for a given choice of slack.
        Assuming slack is a scalar.
        """
        # works only if slack variable is a scalar for now. I assume a single slack in the implementation.
        slack_bus = slack

        nbuses = bus.shape[0]           # number of buses
        nbranches = branch.shape[0]     # number of branches
        noref = np.arange(1, nbuses)    
        noslack = np.flatnonzero(np.arange(nbuses) != slack_bus) # everything except the slack bus

        #check = makeBdc(bus, branch)
        #print("check within PTDF: ", len(check), check)
        Bbus, Bf, _, _, _ = makeBdc(bus, branch)
        Bbus, Bf = Bbus.todense(), Bf.todense()
        PTDF = np.zeros((nbranches, nbuses))
        PTDF[:, noslack] = np.linalg.solve(Bbus[np.ix_(noslack, noref)].T, Bf[:, noref].T).T

        return PTDF
    
    def gridInformationandBranchDict(self, intialgridbranch, init_obs, powerFlowLimits):
        """
        Power Flow Limits dictionary (necessary, needed only once at the start)
        """    
        # this was creating an issue since the "branch" data structure is adapted to current environement, hence I am using the "intialgridbranch" always to keep track of the initial buses.
        branchIndextoBusDict, count = {}, 0
        result = zip(intialgridbranch[:, F_BUS], intialgridbranch[:, T_BUS])
        for a in result:
            branchIndextoBusDict[count] = (a[0], a[1]) 
            #print(count, a)
            count += 1

        return branchIndextoBusDict

    def parse_observation_to_grid_time_zero(self, obs):
        
        # slack bus assignment in the pandapower model
        assert len(self.grid.ext_grid) == 1 or self.grid.gen.slack.any()
        assert len(obs.gen_type) == len(self.grid.gen) + len(self.grid.ext_grid)

        self.grid.bus.min_vm_pu = 0.9
        self.grid.bus.max_vm_pu = 1.2

        # generators
        if len(self.grid.ext_grid) == 0:
            self.grid.gen.type = obs.gen_type
            self.grid.gen.controllable = obs.gen_redispatchable
            self.grid.gen.min_p_mw = obs.gen_pmin
            self.grid.gen.max_p_mw = obs.gen_pmax
            self.grid.gen.min_q_mvar = -obs.gen_pmax
            self.grid.gen.max_q_mvar = obs.gen_pmax
        else:
            self.grid.gen.type = obs.gen_type[:-1]
            self.grid.gen.controllable = obs.gen_redispatchable[:-1]
            self.grid.gen.min_p_mw = obs.gen_pmin[:-1]
            self.grid.gen.max_p_mw = obs.gen_pmax[:-1]
            self.grid.gen.min_q_mvar = -obs.gen_pmax[:-1]
            self.grid.gen.max_q_mvar = obs.gen_pmax[:-1]

            # ext grid (last gen)
            self.grid.ext_grid.type = obs.gen_type[-1]
            self.grid.ext_grid.min_p_mw = obs.gen_pmin[-1]
            self.grid.ext_grid.max_p_mw = obs.gen_pmax[-1]
            self.grid.ext_grid.min_q_mvar = -obs.gen_pmax[-1]
            self.grid.ext_grid.max_q_mvar = obs.gen_pmax[-1]
            assert obs.gen_redispatchable[-1]

        self.gen_min_p = obs.gen_pmin
        self.gen_max_p = obs.gen_pmax

        self.grid.load["p_mw"] = obs.load_p
        self.grid.load["q_mvar"] = obs.load_q

        # important in order for pp to build the topology associated with the current grid2op observations.
        # in the grid2op observations, the first n_lines elements are the power-grid branches
        # and after them are the transformers.
        self.grid.line["in_service"] = obs.line_status[:len(self.grid.line)]
        self.grid.trafo["in_service"] = obs.line_status[-len(self.grid.trafo):]

        pp.rundcpp(self.grid, init="results") # run dc power flow



    def _init_training(self):
        self.epsilon_1 = cfg.INITIAL_EPSILON
        self.epsilon_2 = cfg.INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(cfg.PER_CAPACITY, cfg.PER_ALPHA)
        self.Qtarget = DoubleDuelingDQN_NN(self.action_size,
                                           self.observation_size,
                                           num_frames = self.num_frames)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = []
        if self.is_training:
            self.frames2 = []

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    def _adaptive_epsilon_decay(self, step):
        ada_div = cfg.DECAY_EPSILON / 10.0
        step_off = step + ada_div
        ada_eps = cfg.INITIAL_EPSILON * -math.log10((step_off + 1) / (cfg.DECAY_EPSILON + ada_div))
        ada_eps_up_clip = min(cfg.INITIAL_EPSILON, ada_eps)
        ada_eps_low_clip = max(cfg.FINAL_EPSILON, ada_eps_up_clip)
        return ada_eps_low_clip

    def _save_hyperparameters(self, logpath, env, steps):
        #r_instance = env.reward_helper.template_reward
        hp = {
            "lr": cfg.LR,
            "lr_decay_steps": cfg.LR_DECAY_STEPS,
            "lr_decay_rate": cfg.LR_DECAY_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "stack_frames": cfg.N_FRAMES,
            "iter": steps,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "per_alpha": cfg.PER_ALPHA,
            "per_beta": cfg.PER_BETA,
            "per_capacity": cfg.PER_CAPACITY,
            "update_freq": cfg.UPDATE_FREQ,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            #"reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)



    ## Agent Interface
    def convert_obs(self, observation):
        tmp_list_vect = ['prod_p','load_p','p_or','a_or','p_ex','a_ex','rho','line_status',
                         'timestep_overflow','time_before_cooldown_line','time_before_cooldown_sub'] # 22 (gen_p) + 37 (load_p) + 59*8 + 36 (time_before_cooldown_sub)

        li_vect =  []
        for el in tmp_list_vect:
            #if el in observation.attr_list_vect:
            v = observation._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)

            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            
            if el =='rho' or el=='line_status':
                pass
            else:
                v_res = v_res/100
            
            li_vect.append(v_res)
        return np.concatenate(li_vect)

    def convert_act(self, action):
        return super().convert_act(action)

    ## Baseline Interface
    def reset(self, observation):
        self._reset_state(observation)
        self._reset_frame_buffer()

    def my_act(self, state, reward, done=False):
        # Register current state to stacking buffer
        self._save_current_frame(state)

        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return 0                          # Do nothing in the DDQN action space

        # Infer with the last num_frames states
        action = self.common_choose_action()
        a = self.all_actions.index(action)
        return a  

    def get_action_by_model(self):
        # this inference process wasn't performing well

        if len(self.frames) < self.num_frames:
            chosen_action = self.action_space({})
            simul_obs, chosen_rwd, simul_has_error, simul_info = self.obs.simulate(chosen_action)
            return chosen_action, chosen_rwd
        
        observation = self.obs

        line_stat_s = copy.deepcopy(observation.line_status)
        cooldown = copy.deepcopy(observation.time_before_cooldown_line)
        can_be_reco = ~line_stat_s & (cooldown == 0)   # if removed (and cooldown), then can be connected
        can_be_removed = line_stat_s & (cooldown == 0) # if connected (and cooldown), then can be removed
        # find all legal actions
        removed_indices = np.where(can_be_removed)[0]
        #print("removed_indices pre: ", removed_indices)  
        removed_indices += 51         # index specific to the structure of the action space
        #print("removed_indices post: ", removed_indices)
        reco_indices = np.where(can_be_reco)[0]

        mask = np.full(self.action_size, False)
        mask[1:51] = True               # Set indices 1 to 50 (0-indexed) to be True - re-dispatch action space to be true
        mask[removed_indices] = True
        mask[0] = True                  # do-nothing is always legal
        #print("mask: ", mask )
        focus_indices = np.where(mask)[0]

        chosen_action_id, q_predictions = self.Qmain.predict_move( np.array(self.frames) )
        focus_values = q_predictions[focus_indices]
        #print("Legal Q-values: ", focus_values )
        top_indices_within_focus = np.argsort(focus_values)[::-1][:5]
        top_actions_legal = focus_indices[top_indices_within_focus].tolist()
        print("top 5 Legal: ", top_actions_legal)

        res = [self.convert_act(act_id) for act_id in tuple(top_actions_legal)]
    
        chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, res)

        action_index = self.all_actions.index(chosen_action) 

        return chosen_action, chosen_rwd, action_index
    
    def get_action_by_model_2(self):

        if len(self.frames) < self.num_frames:
            chosen_action = self.action_space({})
            simul_obs, chosen_rwd, simul_has_error, simul_info = self.obs.simulate(chosen_action)
            return chosen_action, chosen_rwd
        
        observation = self.obs

        line_stat_s = copy.deepcopy(observation.line_status)
        cooldown = copy.deepcopy(observation.time_before_cooldown_line)
        can_be_reco = ~line_stat_s & (cooldown == 0)   # if removed (and cooldown), then can be connected
        can_be_removed = line_stat_s & (cooldown == 0) # if connected (and cooldown), then can be removed
        # find all legal actions
        removed_indices = np.where(can_be_removed)[0]
        #print("removed_indices pre: ", removed_indices)  
        removed_indices += 51         # index specific to the structure of the action space
        #print("removed_indices post: ", removed_indices)
        reco_indices = np.where(can_be_reco)[0]

        mask = np.full(self.action_size, False)
        mask[1:51] = True               # Set indices 1 to 50 (0-indexed) to be True - re-dispatch action space to be true
        mask[removed_indices] = True
        mask[0] = True                  # do-nothing is always legal
        #print("mask: ", mask )
        focus_indices = np.where(mask)[0]

        chosen_action_id, q_predictions = self.Qmain.predict_move( np.array(self.frames) )
        focus_values = q_predictions[focus_indices]
        #print("Legal Q-values: ", focus_values )
        top_indices_within_focus = np.argsort(focus_values)[::-1][:5]
        top_actions_legal = focus_indices[top_indices_within_focus].tolist()
        #print("top 5 Legal: ", top_actions_legal)
        res = [self.convert_act(act_id) for act_id in tuple(top_actions_legal)] # convert to grid2op action

        reconnect = []
        if np.any(can_be_reco):
            reconnect = [ self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0] ]
    
        chosen_action, chosen_rwd = self.choose_best_action_by_simul(observation, res + reconnect)
        action_index = self.all_actions.index(chosen_action) 

        return chosen_action, chosen_rwd, action_index

    def act(self, observation, reward, done):
        self.obs = observation
        transformed_observation = self.convert_obs(observation)
        encoded_act = self.my_act(transformed_observation, reward, done)
        return self.convert_act(encoded_act)

    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    def check_cooldown_legal(self, observation, action):

        lines_impacted, subs_impacted = action.get_topological_impact()
        
        line_need_cooldown = lines_impacted & observation.time_before_cooldown_line
        if np.any(line_need_cooldown):
            return False
        
        sub_need_cooldown = subs_impacted & observation.time_before_cooldown_sub
        if np.any(sub_need_cooldown):
            return False
        
        return True

    def choose_best_action_by_simul_buffer(self, observation, tested_action):
        # choose best LEGAL action based on simulated reward
        best_action = None
        highest_reward = None
        
        if len(tested_action) > 1:
            resulting_rewards = np.full(shape=len(tested_action), fill_value=np.NaN, dtype=dt_float)
            
            for i, action in enumerate(tested_action):
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                resulting_rewards[i] = simul_reward

            reward_idx = int(np.argmax(resulting_rewards))
            best_action = tested_action[reward_idx]
            highest_reward = np.max(resulting_rewards)
            
            if self.check_cooldown_legal(observation, best_action) == False:
                print("This Explored action is illegal! Replacing by do-nothing")
                best_action = self.action_space( {} )

        # only one action to be done
        elif len(tested_action) == 1:
            best_action = tested_action[0]
            simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(best_action)
            highest_reward = simul_reward

        return best_action, highest_reward

    def choose_best_action_by_simul(self, observation, tested_action):
        # choose best LEGAL action based on simulated reward
        best_action = None
        highest_reward = None
        
        if len(tested_action) > 1:
            resulting_rewards = np.full(shape=len(tested_action), fill_value=np.NaN, dtype=dt_float)
            
            for i, action in enumerate(tested_action):
                if self.check_cooldown_legal(observation, action) == False:
                    if cfg.VERBOSE:
                        print("illegal action in func: chooose_best_action_by_simul !!!")
                    continue
                simul_obs, simul_reward, simul_has_error, simul_info = observation.simulate(action)
                resulting_rewards[i] = simul_reward
            
            try:
                # there is a possibility that all of them are illegal actions - All-NaN slice encountered - in that case take do-nothing.
                reward_idx = int(np.nanargmax(resulting_rewards))  # rewards.index(max(rewards))
                highest_reward = np.max(resulting_rewards)
                best_action = tested_action[reward_idx]
            except ValueError:
                print("Implementing Do Nothing since All Illegal")
                best_action = self.action_space( {} )
                simul_obs, highest_reward, simul_has_error, simul_info = observation.simulate(best_action)

        # only one action to be done
        elif len(tested_action) == 1:
            best_action = tested_action[0]
            simul_obs, highest_reward, simul_has_error, simul_info = observation.simulate(best_action)

        return best_action, highest_reward

 
    def parse_observation_to_grid(self, obs):
        """
        Function responsible for parsing the grid2op observations and updating the "backend" information to adapt to real-time grid2op observations.
        """
        self.grid.line["in_service"] = obs.line_status[:len(self.grid.line)]        
        self.grid.trafo["in_service"] = obs.line_status[-len(self.grid.trafo):]

    def findMonitoredLineandBus(self, obs):
        # coming in here, I know the buses assocaited with the lines that is maximally_loaded/monitored. 
        # find it's position in the LODF matrix.
        # input - obs, powerFlowLimits, 

        # the decision for when the agent should kick in determines the "monitored" line when using the LODF matrix to make decisions.
        monitoredLineIndex = np.argmax(obs.rho)            # can always rely on the "rho" provided by grid2op since always considers all the branches in the network. 
        #monitoredMaxValue = obs.rho[monitoredLineIndex]    # and find the maximally loaded line value to quantify to what extent
               
        # the above is not enough, since LODF matrix changes size after line outages. Hence, the above line-index is NOT the true index when querying the LODF matrix. 
        # Hence, finding the buses associated with the "maxLoaded_line_index".
        monitoredFromBus, monitoredToBus = self.branchIndextoBusDict[monitoredLineIndex]

        #print(monitoredMaxValue, np.abs(obs.p_or[monitoredLineIndex]), self.powerFlowLimits[monitoredLineIndex], monitoredLineIndex, monitoredFromBus, monitoredToBus)
        return monitoredLineIndex, monitoredFromBus, monitoredToBus
    
    def lineIDsforLineDisconAlways(self, obs, branch, monitoredLineIndex, monitoredFromBus, monitoredToBus, powerFlowLimits, LODF):

        epsilon = 0 
        finalScreeningIndices = []
        finalScreeningBuses = []
        #print(np.abs(obs.p_or[monitoredLineIndex]), powerFlowLimits[monitoredLineIndex] - epsilon)

        #print("-----------------Checking for Valid Line Indices based on LODF-----------------")

        healthyLinesIndices = list(np.where(obs.line_status==True)[0])
        powerFlowHealthyLines = obs.p_or[healthyLinesIndices]

        if obs.p_or[monitoredLineIndex] > 0: # quick screening of potentially "dropped" lines;
            #print("Positive FLow")
            # use "frombus" and "tobus" in order to find true"index" associated with "maxLoaded_line_index".
            # idea is to use "branch" to do that since the branch data structure returned is, indeed, changing sizes.
            check = list(zip(branch[:, F_BUS], branch[:, T_BUS]))

            # find the index of the maximally loaded line in the gridbranch matrix array to re-use that index later for decision making
            for index, busTuple in enumerate(check):
                if busTuple == (monitoredFromBus, monitoredToBus):
                    trueMonitoredLineIndex = index
                    #print(trueMonitoredLineIndex)
                    break
            firstScreen = LODF[trueMonitoredLineIndex, :]*powerFlowHealthyLines[:] < 0 # candidate list of potentially dropped lines, but still boolean array.
            firstScreenIndices = list(np.where(firstScreen)[0])                            # indices of the lines according to the CURRENT LODF matrix. 

            # to check if the reduction in flow is "enough" to mitigate the line overflow?
            secondScreenIndices = []
            for potential in firstScreenIndices:
                if np.abs(obs.p_or[monitoredLineIndex] + LODF[trueMonitoredLineIndex, potential]*powerFlowHealthyLines[potential]) <= powerFlowLimits[monitoredLineIndex]:
                    secondScreenIndices.append(potential)

            # does dropping the circuit create any new overloads?
            for droppedIndex in secondScreenIndices:
                check = np.abs(powerFlowHealthyLines[:] + LODF[:, droppedIndex]*powerFlowHealthyLines[droppedIndex]) <= powerFlowLimits[healthyLinesIndices]
                if np.all(check == True):
                    finalScreeningIndices.append(droppedIndex)

        else:
            #print("Negative Flow")
            check = list(zip(branch[:, F_BUS], branch[:, T_BUS]))

            # find the index of the maximally loaded line in the gridbranch matrix array to re-use that index later for decision making
            for index, busTuple in enumerate(check):
                if busTuple == (monitoredFromBus, monitoredToBus):
                    trueMonitoredLineIndex = index
                    #print(trueMonitoredLineIndex)
                    break
            firstScreen = LODF[trueMonitoredLineIndex, :]*powerFlowHealthyLines[:] > 0 # candidate list of potentially dropped lines, but still boolean array.
            firstScreenIndices = list(np.where(firstScreen)[0])                            # indices of the lines according to the CURRENT LODF matrix. 

            # to check if the reduction in flow is "enough" to mitigate the line overflow?
            secondScreenIndices = []
            for potential in firstScreenIndices:
                if np.abs(obs.p_or[monitoredLineIndex] + LODF[trueMonitoredLineIndex, potential]*powerFlowHealthyLines[potential]) <= powerFlowLimits[monitoredLineIndex]:
                    secondScreenIndices.append(potential)

            # does dropping the circuit create any new overloads?
            for droppedIndex in secondScreenIndices:
                check = np.abs(powerFlowHealthyLines[:] + LODF[:, droppedIndex]*powerFlowHealthyLines[droppedIndex]) <= powerFlowLimits[healthyLinesIndices]
                if np.all(check == True):
                    finalScreeningIndices.append(droppedIndex)

        # find the bus tuple associated with these indices
        for index in finalScreeningIndices:
            finalScreeningBuses.append( (branch[index, F_BUS], branch[index, T_BUS]) )

        #print("First Screen:  ", firstScreenIndices)
        #print("Second Screen: ", secondScreenIndices)
        #print("Final Screen:  ", finalScreeningIndices, finalScreeningBuses)
        return finalScreeningBuses

    def random_overflow_Action(self, can_be_reco, observation):

        # case-3: reconnection lines
        reconnect = []
        if np.any(can_be_reco):
            reconnect = [ self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0] ]

        # case-4 : LODF actions
        removal = []
        monitoredLineIndex = np.argmax(observation.rho)         # rho doesn't change size so argmax will return correct index
        monitoredMaxValue = observation.rho[monitoredLineIndex] # not really used further
        self.parse_observation_to_grid(observation)
        LODF, _, branch, _  = self.getLODFPTDFBranch()
        # processing observation for decision making;
        monitoredLineIndex, monitoredFromBus, monitoredToBus = self.findMonitoredLineandBus(observation)
        finalScreeningBuses = self.lineIDsforLineDisconAlways(observation, branch, monitoredLineIndex, monitoredFromBus, monitoredToBus, self.powerFlowLimits, LODF)
        finalScreeningBusesSet = set(finalScreeningBuses)

        finalScreeningTrueLineIDs = []
        if len(finalScreeningBuses)>0: # feasible solution exists
            for lineIndex, busTuple in self.branchIndextoBusDict.items():
                if busTuple in finalScreeningBusesSet:
                    finalScreeningTrueLineIDs.append(lineIndex)

        #print("finalScreeningTrueLineIDs: ", finalScreeningTrueLineIDs)
        if len(finalScreeningTrueLineIDs)>0:
            removal = [ self.action_space( {"set_line_status": [(id_, -1)]} ) for id_ in finalScreeningTrueLineIDs ]

        tested_actions = []
        if len(removal)>0 and len(reconnect)>0:
            print("CASE-3 : Feasible LODF and Reconnection ")
            tested_actions = removal + reconnect + self.redispatch_actions  

        elif len(removal)>0:
            print("CASE-4 : Feasible LODF only ")                    
            tested_actions = removal + self.redispatch_actions 

        elif len(reconnect)>0:
            print("CASE-5 : Feasible Reconnection only ")   # choose a random re-dispatch action
            tested_actions = reconnect

        else:
            print("CASE-6 : None ")
            random_index = np.random.randint( 0, len(self.redispatch_actions) )
            chosen_action = self.redispatch_actions [random_index]
            return chosen_action

        chosen_action, rwd_topo = self.choose_best_action_by_simul(observation, tested_actions) # exactly how LODF greedy works
        return chosen_action
            
    def reco_line(self, observation, can_be_reco):
        
        if np.any(can_be_reco):
            res = [ self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0] ]
            chosen_action, rwd = self.choose_best_action_by_simul(observation, res)
            return chosen_action, rwd
        return None, None
                               
    def random_action_redis(self):
        act_id = np.random.randint(0, 51)  # between 0 and 50, both inclusive
        return self.convert_act(act_id)    # grid2op action

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              activateAgentRho,
              logdir = "logs-train"):

        # Loop vars
        num_steps = iterations
        step = 0
        self.epsilon_1 = cfg.INITIAL_EPSILON
        step_for_e2 = 0
        self.epsilon_2 = cfg.INITIAL_EPSILON # not being used in the setting
        alive_steps = 0
        total_reward = 0
        self.done = True

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".h5")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)

        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                new_obs = env.reset()
                self.reset(new_obs)

            # processing heuristics
            observation = self.obs
        
            line_stat_s = copy.deepcopy(observation.line_status)
            cooldown = copy.deepcopy(observation.time_before_cooldown_line)

            can_be_reco = ~line_stat_s & (cooldown == 0)    # if removed (and cooldown), then can be connected
            can_be_removed = line_stat_s & (cooldown == 0)  # if connected (and cooldown), then can be removed

            # find all legal actions
            removed_indices = np.where(can_be_removed)[0]
            #print("removed_indices pre: ", removed_indices)
            removed_indices += 51                           # index specific to the structure of the action space
            #print("removed_indices post: ", removed_indices)
            reco_indices = np.where(can_be_reco)[0]

            mask = np.full(self.action_size, False)
            mask[1:51] = True                               # Set indices 1 to 50 (0-indexed) to be True - re-dispatch action space to be true
            mask[removed_indices] = True
            mask[0] = True                                  # do-nothing is always legal
            #print("mask: ", mask )
            focus_indices = np.where(mask)[0]

            rho = copy.deepcopy(observation.rho)
            max_rho_value = np.max( rho )

            if max_rho_value < activateAgentRho:
                # action selection based on heuristics
                # print(" ----> NoOver")
                action_reco, reco_rwd = self.reco_line(observation, can_be_reco) # if reconnection is possible, JUST DO IT!
                if action_reco is not None:
                    #print("CASE-1: Line Reconnected")
                    act = action_reco
                else:
                    #print("CASE-2: Do Nothing")
                    act = self.action_space( {} )                  # DoNothing
                
            # choose action/action selection based on e-greedy
            elif max_rho_value >= activateAgentRho:

                if max_rho_value >=1.0: # trying to print sparsely to montior progress
                    print("LODF_Redis e-greedy")
                    print( "Current step: ", step)
                    print( "Max Rho: ",  max_rho_value )
                    print( "Step [{}] -- Random [{}]".format(step, self.epsilon_1))
                    if len(reco_indices) > 0:
                        print("legal reconnection possibilities LODF e-greedy: ", reco_indices)

                    print("Current step for epsilon 2: ", step_for_e2)
                    print("Step for epsilon 2 [{}] -- Random [{}]".format(step_for_e2, self.epsilon_2))

                # Choose Action (*valid* action)
                if np.random.rand(1) < self.epsilon_1: 
                    #print("")
                    #print("Action via Prior Knowledge func: train")
                    act = self.random_overflow_Action(can_be_reco, observation) # choosing a random legal grid2op action

                elif len(self.frames) < self.num_frames:
                    act = self.action_space({})       # do nothing grid2op action

                else:
                    #print("")
                    #print("ACTION via Q-function func: train")

                    chosen_action_id, q_predictions = self.Qmain.predict_move( np.array(self.frames) )
                    #print(q_predictions)
                    focus_values = q_predictions[focus_indices]
                    #print("Legal Q-values: ", focus_values )
                    top_indices_within_focus = np.argsort(focus_values)[::-1][:5]        # top-5 Q-values
                    top_actions_legal = focus_indices[top_indices_within_focus].tolist()
                    print("top 5 Legal: ", top_actions_legal)

                    res = [self.convert_act(act_id) for act_id in tuple(top_actions_legal)]

                    reconnect = []
                    if np.any(can_be_reco):
                        reconnect = [ self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0] ]

                    act, highest_reward = self.choose_best_action_by_simul(observation, res + reconnect)
       

            if act is None:
                act = self.action_space( {} ) 

            # Execute Action
            new_obs, reward, self.done, info = env.step(act)

            if self.done and len(info['exception'])!=0:
                reward = -100       # episode over without reaching the end

            new_state = self.convert_obs(new_obs)

            self._save_current_frame(self.state)

            a = self.all_actions.index(act)             # conversion from grid2op action to DDQN output index

            if info["is_illegal"]:
                if cfg.VERBOSE:
                    print ( "$$$$$ illegal selected action {}".format(a) )

            if max_rho_value >= activateAgentRho:
                if cfg.VERBOSE:
                    print("------------------ Actual reward {:.3f},  Actual act:   {},".format(reward, a))

            self._save_next_frame(new_state)

            if max_rho_value >= activateAgentRho:
                # Save to experience buffer only if above rho
                if len(self.frames2) == self.num_frames:
                    if info["is_illegal"]:
                        self.per_buffer.add(np.array(self.frames), 0, reward, np.array(self.frames2), self.done) # since illegal action translates to do nothing
                    else:
                        self.per_buffer.add(np.array(self.frames), a, reward, np.array(self.frames2), self.done)

                # Perform training at given frequency
                if step % cfg.UPDATE_FREQ == 0 and len(self.per_buffer) >= self.batch_size:
                    # Perform training
                    self._batch_train(step)

                    if cfg.UPDATE_TARGET_SOFT_TAU > 0.0:
                        tau = cfg.UPDATE_TARGET_SOFT_TAU
                        # Update target network towards primary network
                        self.Qmain.update_target_soft(self.Qtarget.model, tau)

                # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and step % (cfg.UPDATE_FREQ * cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    self.Qmain.update_target_hard(self.Qtarget.model)

            # Decay chance of random action
            self.epsilon_1 = self._adaptive_epsilon_decay(step)

            self.epsilon_2 = self._adaptive_epsilon_decay(step_for_e2)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                if cfg.VERBOSE:
                    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Survived [{}] steps".format(alive_steps) )
                    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Total reward [{}]".format(total_reward) )
                    print( "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Total Trained Episodes [{}] ".format( len(self.epoch_rewards) ) )
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1

            # Save the network every 1000 iterations
            if (step > 20000 and step < 29000) or (step > 55000):
                if step % 1000 == 0:
                    self.save(str(step) + "_" + modelpath)
            elif (step >= 29000 and step <= 55000) :
                if step % 500 == 0:
                    self.save(str(step) + "_" + modelpath)

            if max_rho_value >= activateAgentRho:
                # Iterate to next loop
                step += 1
                
            if max_rho_value >= 0.96: # this controls the decay of the epsilon_2 which is very imporant for correct exploration. Decrease for quicker decay
                # Iterate to next loop
                step_for_e2 += 1            # can use this step for making a less severe epsilon decay. Also monitor two decays in one go.

            # Make new obs the current obs
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)
        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.epoch_alive)
            writer.writerow(self.epoch_rewards)

    def _batch_train(self, step):
        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, cfg.PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        r_batch = sample_batch[2]
        s2_batch = sample_batch[3]
        d_batch = sample_batch[4]
        w_batch = sample_batch[5]
        idx_batch = sample_batch[6]

        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        input_t = np.reshape(s_batch, (self.batch_size, input_size))
        input_t_1 = np.reshape(s2_batch, (self.batch_size, input_size))

        # Save the graph just the first time
        if step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.Qmain.model.predict(input_t, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.Qmain.model.predict(input_t_1, batch_size=self.batch_size)
        Q2 = self.Qtarget.model.predict(input_t_1, batch_size=self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += cfg.DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.train_on_batch(input_t, Q, w_batch)

        # Update PER buffer
        priorities = self.Qmain.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)               # 

        # Log some useful metrics every even updates
        if step % (cfg.UPDATE_FREQ * 2) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 50:
                    mean_reward_50 = np.mean(self.epoch_rewards[-50:])
                    mean_alive_50 = np.mean(self.epoch_alive[-50:])
                else:
                    mean_reward_50 = mean_reward
                    mean_alive_50 = mean_alive

                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_50", mean_reward_50, step)
                tf.summary.scalar("mean_alive_50", mean_alive_50, step)
                tf.summary.scalar("loss", loss, step)
                tf.summary.scalar("lr", self.Qmain.train_lr, step)
                tf.summary.scalar("epsilon_1", self.epsilon_1, step)
                tf.summary.scalar("epsilon_2", self.epsilon_2, step)
                tf.summary.scalar("Num Episodes", len(self.epoch_rewards), step)
            if cfg.VERBOSE:
                print("loss =", loss)
