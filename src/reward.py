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