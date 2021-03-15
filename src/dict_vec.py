
class DictVector(dict):
    """
    Extension of a dictionary with vector addition and
    scalar multiplication. Values must be numeric.
    """
    def __init__(self, dictionary=None):
        super(DictVector, self).__init__()
        if dictionary is not None:
            for key, value in dictionary.items():
                self[key] = value

    def __add__(self, other):
        result = {}
        keys = set(self.keys()).union(set(other.keys()))
        for key in keys:
            result[key] = self.get(key, 0) + other.get(key, 0)
        return result

    def __sub__(self, other):
        result = self.copy()
        for key in other.keys():
            if key in self:
                result[key] -= other[key]
            else:
                result[key] = -other[key]
        return result

    def __mul__(self, other):
        assert isinstance(other, int), "Only scalar multiplication is defined"
        result = self.copy()
        for key in self.keys():
            result[key] *= other
        return result

    def __rmul__(self, other):
        assert isinstance(other, int), "Only scalar multiplication is defined"
        result = self.copy()
        for key in self.keys():
            result[key] *= other
        return result
