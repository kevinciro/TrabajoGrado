"""
    Class WeightClipper:
    
"""

class WeightClipper(object):
    def __init__(self, min=-1, max=1):

        self.min = min
        self.max = max

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w = w.clamp(self.min, self.max)