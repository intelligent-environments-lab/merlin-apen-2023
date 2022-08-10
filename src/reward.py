from typing import List
import numpy as np
from citylearn.reward_function import RewardFunction

class MultiplicativeReward(RewardFunction):
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        carbon_emission = (np.array(self.carbon_emission).clip(min=0)*self.kwargs['carbon_emission_weight'])**self.kwargs['carbon_emission_exponent']
        electricity_price = (np.array(self.electricity_price).clip(min=0)*self.kwargs['electricity_price_weight'])**self.kwargs['electricity_price_exponent']
        reward = -carbon_emission*electricity_price
        return reward

class AdditiveReward(RewardFunction):
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        carbon_emission = (np.array(self.carbon_emission).clip(min=0)*self.kwargs['carbon_emission_weight'])**self.kwargs['carbon_emission_exponent']
        electricity_price = (np.array(self.electricity_price).clip(min=0)*self.kwargs['electricity_price_weight'])**self.kwargs['electricity_price_exponent']
        reward = -(carbon_emission + electricity_price)
        return reward

class AdditiveSolarPenaltyReward(RewardFunction):
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(electricity_consumption=electricity_consumption, **kwargs)

    def calculate(self) -> List[float]:
        carbon_emission = (np.array(self.carbon_emission)*self.kwargs['carbon_emission_weight'])**self.kwargs['carbon_emission_exponent']
        electricity_price = (np.array(self.electricity_price)*self.kwargs['electricity_price_weight'])**self.kwargs['electricity_price_exponent']
        soc = self.kwargs.get('electrical_storage_soc', np.array([0.0]*self.agent_count))
        reward = -(1.0 + np.sign(electricity_price)*soc)*abs(carbon_emission + electricity_price)
        return reward

class RampingReward(RewardFunction):
    def __init__(self, electricity_consumption: List[float] = None, **kwargs):
        super().__init__(electricity_consumption=electricity_consumption, **kwargs)
        self.previous_electricity_consumption_sum = 0.0

    def calculate(self) -> List[float]:
        electricity_consumption_sum = sum(self.electricity_consumption)
        reward = (abs(electricity_consumption_sum - self.previous_electricity_consumption_sum))**2
        self.previous_electricity_consumption_sum = electricity_consumption_sum
        reward = np.array([-reward for _ in self.electricity_consumption], dtype=float)
        print(reward)
        return reward