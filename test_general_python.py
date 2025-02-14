import numpy as np
import pyroomacoustics as pra
import math
a=np.array([1,2,5])
b = np.array([3,2,1])
a = np.maximum(a,b)
roomDims = [7,7,10]
mic_radius = 0.0725
n_mics = 6
height = 1.4
def generate_mic_array_3d(roomDims,mic_radius: float, n_mics: int, height):
    microphones = []
    mic_x = roomDims[0]/2
    mic_y = roomDims[1]/2
    mic_z = height
    angular_difference_degrees = 60
    for i in range(n_mics):
        angle = math.radians(i * angular_difference_degrees)
        x = mic_x + mic_radius * math.cos(angle)
        y = mic_y + mic_radius * math.sin(angle)
        microphones.append([x,y,mic_z])
    microphones = np.array(microphones).transpose()
    print(microphones)
def generate_mic_array(roomDims, mic_radius, n_mics):
    mic_x = roomDims[0] / 2
    mic_y = roomDims[1] / 2
    center = [mic_x, mic_y]
    R = pra.circular_2D_array(center=center, M=n_mics, phi0=0, radius=mic_radius)
    print("2D Mic Array Centered in Room:")
    print(R)
    return R
generate_mic_array_3d(roomDims,mic_radius, n_mics, height)
generate_mic_array(roomDims, mic_radius, n_mics)