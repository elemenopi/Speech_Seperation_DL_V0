import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

def center_trim(tensor, reference):
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2: -(diff - diff // 2)]
    return tensor

class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim
        )
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x

class CosNetwork(nn.Module):
    def __init__(
        self,
        n_audio_channels: int = 6,
        sources: int = 2,
        channels: int = 64,
        depth: int = 6,
        growth: float = 2.0,
        lstm_layers: int = 2,
        rescale: float = 0.1,
        kernel_size: int = 8,
        stride: int = 4,
        context: int = 3,
        glu: bool = True,
        upsample: bool = False,
    ):
        super().__init__()
        self.n_audio_channels = n_audio_channels
        self.sources = sources
        self.channels = channels
        self.depth = depth
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale
        self.kernel_size = kernel_size
        self.stride = stride
        self.context = context
        self.glu = glu
        self.upsample = upsample

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1

        in_channels = self.n_audio_channels

        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                if upsample:
                    out_channels = channels
                else:
                    out_channels = self.sources * self.n_audio_channels
            decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
            if upsample:
                decode += [nn.Conv1d(channels, out_channels, kernel_size, stride=1)]
            else:
                decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        rescale_module(self, reference=rescale)

    def forward(self, mix):
        x = mix
        saved = [x]

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)
            if self.upsample:
                x = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)

        # LSTM at the bottleneck
        if self.lstm:
            x = self.lstm(x)

        # Decoder
        for decode in self.decoder:
            if self.upsample:
                x = F.interpolate(x, scale_factor=self.stride, mode='linear', align_corners=False)
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        x = x.view(x.size(0), self.sources, self.n_audio_channels, x.size(-1))
        return x

    def loss(self, voice_signals, gt_voice_signals):
        """Simple L1 loss between voice and ground truth"""
        return F.l1_loss(voice_signals, gt_voice_signals)

    def valid_length(self, length: int) -> int:
        for _ in range(self.depth):
            if self.upsample:
                length = math.ceil(length / self.stride) + self.kernel_size - 1
            else:
                length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for _ in range(self.depth):
            if self.upsample:
                length = length * self.stride + self.kernel_size - 1
            else:
                length = (length - 1) * self.stride + self.kernel_size
        return int(length)
