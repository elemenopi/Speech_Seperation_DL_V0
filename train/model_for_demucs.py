import torch.nn as nn
import torch
import torch.nn.functional as F
import math

def rescale_conv(conv, reference):
    # Rescale the standard deviation
    # This ensures the std is close to reference to stabilize training
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    # Rescales all the module layers
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

def center_trim(tensor, reference):
    # Trims the tensor around the center given reference length of trimming
    if hasattr(reference, "size"):  # Is a tensor, take the last dimension size
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2 : -(diff - diff // 2)]
    return tensor

def left_trim(tensor, reference):
    # Same as center but for left
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., 0 : -diff]
    return tensor

def normalize_input(data):
    # Data : batchSize x n_mics x n_samples
    data = (data * 2 ** 15).round() / 2 ** 15  # Normalize data between -1 and 1
    ref = data.mean(1)  # Average across all mics
    means = ref.mean(1).unsqueeze(1).unsqueeze(2)  # Calculate mean across samples
    stds = ref.std(1).unsqueeze(1).unsqueeze(2)   # Calculate std across samples
    data = (data - means) / stds
    return data, means, stds

def unnormalize_input(data, means, stds):
    # Performs broadcasting
    data = (data * stds.unsqueeze(3) + means.unsqueeze(3))
    return data

class CosNetwork(nn.Module):
    def __init__(
        self,
        n_audio_channels: int = 6,
        window_conditioning_size: int = 5,
        kernel_size: int = 8,
        stride: int = 4,
        context: int = 3,
        depth: int = 6,
        channels: int = 64,
        growth: float = 2.0,
        lstm_layers: int = 2,
        rescale: float = 0.1,
    ):
        super().__init__()
        self.n_audio_channels = n_audio_channels
        self.window_conditioning_size = window_conditioning_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context
        self.depth = depth
        self.channels = channels
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        activation = nn.GLU(dim=1)  # GLU on the samples
        in_channels = n_audio_channels

        for index in range(depth):  # For every step of depth, make an encode and decode module
            encode = []
            # conv1
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride)]
            # relu
            encode += [nn.ReLU()]
            # conv2
            encode += [nn.Conv1d(channels, 2 * channels, 1)]
            # activation
            encode += [activation]
            # gc_embed1
            encode += [nn.Conv1d(self.window_conditioning_size, channels, 1)]
            # gc_embed2
            encode += [nn.Conv1d(self.window_conditioning_size, 2 * channels, 1)]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2 * n_audio_channels
            # conv1
            decode += [nn.Conv1d(channels, 2 * channels, context)]
            # activation
            decode += [activation]
            # conv2
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            # gc_embed1
            decode += [nn.Conv1d(self.window_conditioning_size, 2 * channels, 1)]
            # gc_embed2
            decode += [nn.Conv1d(self.window_conditioning_size, out_channels, 1)]
            
            if index > 0:
                # relu
                decode += [nn.ReLU()]
            self.decoder.insert(0, nn.Sequential(*decode))  # Reverse order from encoder
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        self.lstm = nn.LSTM(
            bidirectional=True, num_layers=lstm_layers, hidden_size=channels, input_size=channels
        )
        self.lstm_linear = nn.Linear(2 * channels, channels)
        rescale_module(self, reference=rescale)

    def forward(self, mix: torch.Tensor, angle_conditioning: torch.Tensor):
        x = mix
        saved = [x]

        # Encoder
        for encode in self.encoder:
            x = encode[0](x)  # Conv1d
            embedding = encode[4](angle_conditioning.unsqueeze(2))  # gc_embed1

            x = encode[1](x + embedding)  # ReLU
            x = encode[2](x)  # Conv1d

            embedding2 = encode[5](angle_conditioning.unsqueeze(2))  # gc_embed2
            x = encode[3](x + embedding2)  # Activation (GLU)
            saved.append(x)
        # Bi-directional LSTM at the bottleneck layer
        x = x.permute(2, 0, 1)  # Prepare input for LSTM
        self.lstm.flatten_parameters()  # To improve memory usage
        x = self.lstm(x)[0]
        x = self.lstm_linear(x)
        x = x.permute(1, 2, 0)

        # Source decoder
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip

            x = decode[0](x)  # Conv1d
            embedding = decode[3](angle_conditioning.unsqueeze(2))  # gc_embed1
            x = decode[1](x + embedding)  # Activation (GLU)
            x = decode[2](x)  # ConvTranspose1d
            embedding2 = decode[4](angle_conditioning.unsqueeze(2))  # gc_embed2
            x = x + embedding2
            if len(decode) > 5:
                x = decode[5](x)  # ReLU if present
        # Reformat the output
        x = x.view(x.size(0), 2, self.n_audio_channels, x.size(-1))

        return x

    def loss(self, voice_signals, gt_voice_signals):
        """Simple L1 loss between voice and gt"""
        return F.l1_loss(voice_signals, gt_voice_signals)

    def valid_length(self, length: int) -> int:
        """
        Find the length of the input to the network such that the output's length is
        equal to the given `length`.
        """
        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1

        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        return int(length)

def load_pretrain(model, state_dict):
    """Loads the pretrained keys in state_dict into model"""
    for key in state_dict.keys():
        try:
            _ = model.load_state_dict({key: state_dict[key]}, strict=False)
            print(f"Loaded {key} (shape = {state_dict[key].shape}) from the pretrained model")
        except Exception as e:
            print(f"Failed to load {key}")
            print(e)
