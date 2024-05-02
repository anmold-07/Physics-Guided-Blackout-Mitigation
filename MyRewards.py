import numpy as np
from grid2op.dtypes import dt_float
from grid2op.Reward.BaseReward import BaseReward


RHO = 0.95


# RL-reward
class RLReward(BaseReward):
    def __init__(self,
                 lambdaa=0.0,
                 line_switch_cost=0.0,
                 logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(+1.0)
        self.lambdaa = dt_float(lambdaa)
        self.line_switch_cost = dt_float(line_switch_cost)
        
        print("This is RL reward: ", self.lambdaa, self.line_switch_cost) # just to make sure

    def initialize(self, env):
        pass

    def _get_marginal_cost(self, env):
        gen_activeprod_t = env._gen_activeprod_t
        #print("gen_activeprod_t :", gen_activeprod_t)
        p_t = np.max( env.gen_cost_per_MW[gen_activeprod_t > 0.0] ).astype(dt_float)  
        # price is per MWh be sure to convert the MW (of losses and generation) to MWh before multiplying by the cost 
        return p_t
    
    def _get_redisp_cost(self, env, p_t):
        actual_dispatch = env._actual_dispatch
        #print("actual_dispatch :", actual_dispatch)
        c_redispatching = (
            np.abs(actual_dispatch).sum() * p_t * env.delta_time_seconds / 3600.0
        )
        return c_redispatching
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum( line_cap )             # the same as MarginReward until here
            
            state_obs = env.current_obs
            dict_ = action.as_dict()
            if dict_: # non-empty
                #print("dict_: ", dict_ ) # finding out the type of the action
                #print("rho: ", np.max(state_obs.rho) )
            
                if 'redispatch' in dict_:        # redispatch action taken
                    # compute the marginal cost
                    p_t = self._get_marginal_cost(env)
                    # redispatching amount
                    c_redispatching = self._get_redisp_cost(env, p_t)
                    # "operational cost"
                    c_operations = dt_float( self.lambdaa * c_redispatching )

                elif dict_["set_line_status"]["nb_connected"]:  # reconnect action
                    # penalize this action only if rho above RHO
                    if np.max(state_obs.rho) > RHO:
                        c_operations = dt_float( self.line_switch_cost * 1 ) 
                    else:
                        c_operations = 0

                elif dict_["set_line_status"]["nb_disconnected"]:  # line removal action
                    # "operational cost"
                    c_operations = dt_float( self.line_switch_cost * 1 ) 
                
            else: # do nothing action
                c_operations = 0
  
            # penalize re-dispatch or line switching
            res -= c_operations
            
            ## penality weight on soft overflow
            res = res - 5 * sum( ( ((state_obs.rho - 0.99) * (state_obs.timestep_overflow - 1)) [state_obs.timestep_overflow >= 2] ) )

            ## penalty on hard overflow
            hard_overflow = np.where(state_obs.rho > 2)[0]
            if len(hard_overflow) > 0:
                res = res - dt_float( len(hard_overflow) * 5 )
 
            ## penalty on cascade failure
            cooldown_line = state_obs.time_before_cooldown_line
            cascade_failure_list = np.where(cooldown_line == 12)[0]

            if len(cascade_failure_list) > 0:
                res = res - dt_float( len(cascade_failure_list) * 5 )

            if res < -10.0:
                res = -10.0
        
        elif is_done and has_error: # episode not completed
            res = -60.0
        
        else:
            # illegal action
            res = -10.0

        res = np.interp(res,
                        [dt_float(-60.0), 59.0],
                        [self.reward_min, self.reward_max])
        return res

    def __get_lines_capacity_usage(self, env):
        #print("New Reward")
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        lines_capacity_usage_score = dt_float(1.0) - relative_flow**2
        return lines_capacity_usage_score



# safety margin reward
class MarginReward(BaseReward):
    def __init__(self,
                 logger=None):
        BaseReward.__init__(self, logger=logger)

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        line_cap = self.__get_lines_capacity_usage(env)
        #print("line_cap: ", line_cap)
        res = np.sum(line_cap)
        return res #/env.backend.get_line_flow().shape[0]
            
    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        lines_capacity_usage_score = dt_float(1.0) - relative_flow**2
        return lines_capacity_usage_score
    


# Re-dispatching and Line Switching Cost
class RedisActionCost(BaseReward):
    def __init__(self,
                 logger=None):
        BaseReward.__init__(self, logger=logger)

    def initialize(self, env):
        # TODO compute reward max! 
        return super().initialize(env)
    
    def _get_marginal_cost(self, env):
        gen_activeprod_t = env._gen_activeprod_t
        #print("gen_activeprod_t :", gen_activeprod_t)
        p_t = np.max( env.gen_cost_per_MW[gen_activeprod_t > 0.0] ).astype(dt_float)  
        # price is per MWh be sure to convert the MW (of losses and generation) to MWh before multiplying by the cost 
        return p_t
    
    def _get_redisp_cost(self, env, p_t):
        actual_dispatch = env._actual_dispatch
        #print("actual_dispatch :", actual_dispatch)
        c_redispatching = (
            np.abs(actual_dispatch).sum() * p_t * env.delta_time_seconds / 3600.0
        )
        return c_redispatching
      
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        state_obs = env.current_obs
        dict_ = action.as_dict()
        
        c_redispatch = 0
        
        if dict_: # non-empty
            #print("dict_: ", dict_ ) # finding out the type of the action
            #print("rho: ", np.max(state_obs.rho) )

            if 'redispatch' in dict_:        # redispatch action taken
                # compute the marginal cost
                p_t = self._get_marginal_cost(env)
                # redispatching amount
                c_redispatching = self._get_redisp_cost(env, p_t)
                # "operational cost"
                c_redispatch = dt_float( c_redispatching )
        
        return c_redispatch