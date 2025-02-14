import functools
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

from pathlib import Path
SPEED_OF_SOUND = 343.0
def get_starting_angles(window_size):
    divisor = int(round(2*np.pi/window_size))
    return np.array(list(range(-divisor+1,divisor+2)))*np.pi/divisor
def angular_distance(angle1, angle2):
    """
    Computes the distance in radians betwen angle1 and angle2.
    We assume they are between -pi and pi
    """
    d1 = abs(angle1 - angle2)
    d2 = abs(angle1 - angle2 + 2 * np.pi)
    d3 = abs(angle2 - angle1 + 2 * np.pi)

    return min(d1, d2, d3)
def to_categorical(index,num_classes):
    data = np.zeros((num_classes))
    data[index] = 1
    return data

def shift_mixture(input_data, target_position, mic_radius, sr, inverse=False):
    """
    Shifts the input according to the voice position. This
    lines up the voice samples in the time domain coming from a target_angle
    Args:
        input_data - M x T numpy array or torch tensor
        target_position - The location where the data should be aligned
        mic_radius - In meters. The number of mics is inferred from
            the input_Data
        sr - Sample Rate in samples/sec
        inverse - Whether to align or undo a previous alignment

    Returns: shifted data and a list of the shifts
    """
    # elevation_angle = 0.0 * np.pi / 180
    # target_height = 3.0 * np.tan(elevation_angle)
    # target_position = np.append(target_position, target_height)
    num_channels = input_data.shape[0]

    # Must match exactly the generated or captured data
    mic_array = [[
        mic_radius * np.cos(2 * np.pi / num_channels * i),
        mic_radius * np.sin(2 * np.pi / num_channels * i),
    ] for i in range(num_channels)]

    # Mic 0 is the canonical position
    #print("the target position")
    #print(target_position)
    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    # Check if numpy or torch
    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data, torch.Tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("Unknown input data type: {}".format(type(input_data)))

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(1, num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff = distance - distance_mic0
        shift_time = distance_diff / SPEED_OF_SOUND
        shift_samples = int(round(sr * shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],
                                               -shift_samples)
        shifts.append(shift_samples)

    return input_data, shifts
def plot_area(speakers, window, angle, filename='plot.png'):
    # Convert JSON string to dictionary if speakers is in JSON string format
    if isinstance(speakers, str):
        speakers = json.loads(speakers)
    
    fig, ax = plt.subplots()
    
    # Plot speakers
    for key, value in speakers.items():
        position = value['Position']
        ax.plot(position[0], position[1], 'o', label=value['speaker_id'])
    
    # Plot the angle line
    x_end = np.cos(angle) * 10
    y_end = np.sin(angle) * 10
    ax.plot([0, x_end], [0, y_end], 'r-', label='Angle Line')
    
    # Plot the circular line representing the window
    theta = np.linspace(angle - window/2, angle + window/2, 100)
    radius = 10  # Assume a fixed radius
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    ax.plot(x_circle, y_circle, 'b--', label='Window')
    
    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')
    
    # Adding labels and legend
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.legend()
    
    # Save the plot as an image
    plt.savefig(filename)
    plt.close()

def shift_mixture_old(input_data,target_position,mic_radius,sr,inverse = False):
    num_channels = input_data.shape[0]

    mic_array = [[
        mic_radius * np.cos(2*np.pi / num_channels * i),
        mic_radius * np.sin(2*np.pi / num_channels * i),
    ] for i in range(num_channels)]

    distance_mic0 = np.linalg.norm(mic_array[0] - target_position)
    shifts = [0]

    if isinstance(input_data, np.ndarray):
        shift_fn = np.roll
    elif isinstance(input_data,torch.tensor):
        shift_fn = torch.roll
    else:
        raise TypeError("unknown type")
    
    for channel_idx in range(1,num_channels):
        distance = np.linalg.norm(mic_array[channel_idx] - target_position)
        distance_diff= distance - distance_mic0
        shift_time = distance_diff/340.0
        shift_samples = int(round(sr*shift_time))
        if inverse:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],shift_samples)
        else:
            input_data[channel_idx] = shift_fn(input_data[channel_idx],-shift_samples)
        shifts.append(shift_samples)
    return input_data,shifts
def convert_angular_range(angle: float):
    """Converts an arbitrary angle to the range [-pi pi]"""
    corrected_angle = angle % (2 * np.pi)
    if corrected_angle > np.pi:
        corrected_angle -= (2 * np.pi)

    return corrected_angle
def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__

def check_valid_dir(dir, requires_n_voices=2):
    """Checks that there is at least n voices"""
    if len(list(Path(dir).glob('*_voice00.wav'))) < 1:
        return False

    if requires_n_voices == 2:
        if len(list(Path(dir).glob('*_voice01.wav'))) < 1:
            return False

    if requires_n_voices == 3:
        if len(list(Path(dir).glob('*_voice02.wav'))) < 1:
            return False

    if requires_n_voices == 4:
        if len(list(Path(dir).glob('*_voice03.wav'))) < 1:
            return False

    if len(list(Path(dir).glob('metadata.json'))) < 1:
        return False

    return True