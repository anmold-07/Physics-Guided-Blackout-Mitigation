import math
import copy
import itertools
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.dtypes import dt_int, dt_float, dt_bool
from DoubleDuelingDQN_LODF import DoubleDuelingDQN as BackupAgent

class MyAgent(BaseAgent):
    def __init__(self,observation_space, action_space, grid_path: str, init_obs, powerFlowLimits):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.obs_space = observation_space
        self.backupAgent = BackupAgent(observation_space, action_space, grid_path, init_obs, powerFlowLimits, name = "DDDQN")

        self.agent_count = 0

    def load(self, path):
        self.backupAgent.Qmain.load_network(path)

    def check_cooldown_legal(self, observation, action):
        lines_impacted, subs_impacted = action.get_topological_impact()
        line_need_cooldown = lines_impacted & observation.time_before_cooldown_line
        if np.any(line_need_cooldown):
            return False
        sub_need_cooldown = subs_impacted & observation.time_before_cooldown_sub
        if np.any(sub_need_cooldown):
            return False
        return True

    def choose_best_action_by_simul(self, observation, tested_action):
        # choose best action based on simulated reward
        best_action = None
        highest_reward = None

        if len(tested_action) > 1:
            resulting_rewards = np.full(shape=len(tested_action), fill_value=np.NaN, dtype=dt_float)
            for i, action in enumerate(tested_action):
                if self.check_cooldown_legal(observation, action) == False:
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

    def compare_with_model(self, act, rwd):
        if rwd > 0:
            return act
        chosen_action, chosen_rwd, action_index = self.backupAgent.get_action_by_model()
        #if chosen_action is not None and chosen_rwd > rwd:

        self.agent_count += 1
        return chosen_action

    def act(self, observation, reward, done):
        activateAgentRho = 0.95

        state = self.backupAgent.convert_obs(observation)
        self.backupAgent._save_current_frame(state)
        self.backupAgent.obs = observation
        self.obs = observation

        line_stat_s = copy.deepcopy(observation.line_status)
        cooldown = copy.deepcopy(observation.time_before_cooldown_line)

        rho = copy.deepcopy(observation.rho)
        max_value = np.max( rho )

        if max_value < activateAgentRho:
            # action selection based on heuristics
            #print(" ----> NoOver")
            can_be_reco = ~line_stat_s & (cooldown == 0)
            action_reco = self.reco_line(observation, can_be_reco) # if reconnection is possible, JUST DO IT!
            if action_reco is not None:
                #print("CASE-1: Line Reconnected")
                return action_reco
            else:
                #print("CASE-2: Do Nothing")
                return self.action_space( {} )                  # DoNothing

        elif max_value >= activateAgentRho:
            action_topo, rwd_topo = None, -10000 # give any negative number
            chosen_action = self.compare_with_model(action_topo, rwd_topo)          # the random aspect comes in here 
            return chosen_action

    def reco_line(self, observation, can_be_reco):
        
        if np.any(can_be_reco):
            res = [ self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0] ]
            chosen_action, rwd = self.choose_best_action_by_simul(observation, res)
            return chosen_action
        return None