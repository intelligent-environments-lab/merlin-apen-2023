from typing import List
from citylearn.agents.rbc import BasicRBC
from citylearn.agents.sac import SACBasicBatteryRBC

class FontanaRBC(BasicRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SelfConsumptionFontanaRBC(FontanaRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        actions = []

        for n, o, i, d in zip(self.observation_names, observations, self.building_information, self.action_dimension):
            soc = o[n.index('electrical_storage_soc')]
            net_electricity_consumption = o[n.index('net_electricity_consumption')]
            capacity = i['electrical_storage_capacity']

            # discharge when there is net import and SOC is > 25%
            if net_electricity_consumption > 0.0 and soc > 0.25:
                a = [-2.0/capacity for _ in range(d)]
            
            # charge when there is net export
            elif net_electricity_consumption < 0.0:
                a = [2.0/capacity for _ in range(d)]

            else:
                a = [0.0 for _ in range(d)]

            actions.append(a)
        
        self.actions = actions
        self.next_time_step()
        return actions

class TOUPeakReductionFontanaRBC(FontanaRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        actions = []

        for n, o, i, d in zip(self.observation_names, observations, self.building_information, self.action_dimension):
            soc = o[n.index('electrical_storage_soc')]
            hour = o[n.index('hour')]
            capacity = i['electrical_storage_capacity']
        
            if 9 <= hour <= 12:
                a = [2.0/capacity for _ in range(d)]

            elif (hour >= 18 or hour < 9) and soc > 0.25:
                a = [-2.0/capacity for _ in range(d)]

            else:
                a = [0.0 for _ in range(d)]

            actions.append(a)

        self.actions = actions
        self.next_time_step()
        return actions

class TOURateOptimizationFontanaRBC(FontanaRBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select_actions(self, observations: List[float]) -> List[float]:
        actions = []

        for n, o, i, d in zip(self.observation_names, observations, self.building_information, self.action_dimension):
            soc = o[n.index('electrical_storage_soc')]
            hour = o[n.index('hour')]
            capacity = i['electrical_storage_capacity']

            if 12 <= hour <= 18 and soc > 0.25:
                a = [-2.0/capacity for _ in range(d)]

            else:
                a = [2.0/capacity for _ in range(d)]

            actions.append(a)

        self.actions = actions
        self.next_time_step()
        return actions

class SACTOUPeakReductionFontanaRBC(SACBasicBatteryRBC):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbc = TOUPeakReductionFontanaRBC(*args, **kwargs)