from typing import List
from citylearn.agents.rbc import BasicRBC

class FontanaRBC(BasicRBC):
    def __init__(self, *args, capacity: float = None, soc_index: int = None, net_electricity_consumption_index: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity
        self.soc_index = soc_index
        self.net_electricity_consumption_index = net_electricity_consumption_index

class SelfConsumptionFontanaRBC(FontanaRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        soc = observations[self.soc_index]
        net_electricity_consumption = observations[self.net_electricity_consumption_index]

        # discharge when there is net import and SOC is > 25%
        if net_electricity_consumption > 0.0 and soc > 0.25:
            actions = [-2.0/self.capacity for _ in range(self.action_dimension)]
        
        # charge when there is net export
        elif net_electricity_consumption < 0.0:
            actions = [2.0/self.capacity for _ in range(self.action_dimension)]

        else:
            actions = [0.0 for _ in range(self.action_dimension)]

        self.actions = actions
        self.next_time_step()
        return actions

class TOUPeakReductionFontanaRBC(FontanaRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        hour = observations[self.hour_index]
        soc = observations[self.soc_index]

        if 9 <= hour <= 12:
            actions = [2.0/self.capacity for _ in range(self.action_dimension)]

        elif (hour >= 18 or hour < 9) and soc > 0.25:
            actions = [-2.0/self.capacity for _ in range(self.action_dimension)]

        else:
            actions = [0.0 for _ in range(self.action_dimension)]

        self.actions = actions
        self.next_time_step()
        return actions

class TOURateOptimizationFontanaRBC(FontanaRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        hour = observations[self.hour_index]
        soc = observations[self.soc_index]

        if 12 <= hour <= 18 and soc > 0.25:
            actions = [-2.0/self.capacity for _ in range(self.action_dimension)]

        else:
            actions = [2.0/self.capacity for _ in range(self.action_dimension)]

        self.actions = actions
        self.next_time_step()
        return actions