from enum import Enum

"""
Optical Flow Types
"""

class BaseOpticalFlow(Enum):
    LUCAS_KANADE = "Lucas-Kanade Flow"
    TVL1 = "TVL1 Flow"
    FARNEBACK = "Farneback Flow"
    DEEPFLOW = "DeepFlow"
