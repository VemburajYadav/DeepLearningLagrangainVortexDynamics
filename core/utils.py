from phi.flow import math


def gaussian_falloff(distance, sigma):
    sq_distance = math.sum(distance ** 2, axis=-1, keepdims=True)
    return math.exp(- sq_distance / sigma ** 2) / math.sqrt(sq_distance)