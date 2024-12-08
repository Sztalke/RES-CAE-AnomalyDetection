import torch
import torch.nn as nn
import json
from .base import BaseAE, calculate_conv_output_shape, calculate_deconv_output_shape, calculate_pool_output_shape

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        inter_channels = max(1, in_channels // reduction)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inter_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        out = self.conv(concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_active=False, use_cbam=False, cbam_reduction=16, cbam_kernel=7, dropout=0):
        super(ResidualBlock, self).__init__()
        self.bn_active = bn_active
        self.use_cbam = use_cbam
        self.cbam_reduction = cbam_reduction
        self.cbam_kernel = cbam_kernel

        self.bn1 = nn.InstanceNorm2d(in_channels) if bn_active else nn.Identity()
        self.relu1 = nn.LeakyReLU(inplace=False)
        self.dropout1 = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.bn_active)

        self.bn2 = nn.InstanceNorm2d(out_channels) if bn_active else nn.Identity()
        self.relu2 = nn.LeakyReLU(inplace=False)
        self.dropout2 = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not self.bn_active)
        
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not self.bn_active),
                nn.InstanceNorm2d(out_channels) if bn_active else nn.Identity()
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        if self.use_cbam:
            out = self.cbam(out)

        out += identity
        return out

class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, bn_active=False, use_cbam=False, cbam_reduction=16, cbam_kernel=7, dropout=0):
        super(ResidualUpBlock, self).__init__()
        self.bn_active = bn_active
        self.use_cbam = use_cbam
        self.cbam_reduction = cbam_reduction
        self.cbam_kernel = cbam_kernel

        self.bn1 = nn.InstanceNorm2d(in_channels) if bn_active else nn.Identity()
        self.relu1 = nn.LeakyReLU(inplace=False)
        self.dropout1 = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.convtranspose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.bn_active)

        self.bn2 = nn.InstanceNorm2d(out_channels) if bn_active else nn.Identity()
        self.relu2 = nn.LeakyReLU(inplace=False)
        self.dropout2 = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.convtranspose2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, output_padding=0, bias=not self.bn_active)
        
        self.cbam = CBAM(out_channels) if use_cbam else nn.Identity()

        self.upsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=output_padding, bias=not self.bn_active),
                nn.InstanceNorm2d(out_channels) if bn_active else nn.Identity()
            )

    def forward(self, x):
        identity = self.upsample(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.convtranspose1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.convtranspose2(out)

        if self.use_cbam:
            out = self.cbam(out)

        out += identity
        return out

class ResidualAutoencoder(BaseAE):
    def __init__(self, config, input_shape, device):
        super(ResidualAutoencoder, self).__init__(device)
        self.dropout = config.get("dropout", 0)
        self.encoder, self.encoder_output_shape = self.build_encoder(config['encoder_layers'], input_shape)
        self.decoder = self.build_decoder(config['decoder_layers'], self.encoder_output_shape)

    def build_encoder(self, layers_config, input_shape):
        layers = []
        current_shape = input_shape
        in_channels = layers_config[0]['params']['in_channels']
        
        for layer_cfg in layers_config:
            layer_type = layer_cfg['type']
            params = layer_cfg['params']
            
            # Residual layer
            if layer_type == 'ResidualBlock':
                layer = ResidualBlock(
                    in_channels=in_channels,
                    out_channels=params['out_channels'],
                    kernel_size=params['kernel_size'],
                    stride=params['stride'],
                    padding=params['padding'],
                    bn_active=True,
                    use_cbam=params.get('use_cbam', False),
                    cbam_reduction=params.get('cbam_reduction', 16),
                    cbam_kernel=params.get('cbam_kernel', 7),
                    dropout=self.dropout
                )
                in_channels = params['out_channels']
                current_shape = calculate_conv_output_shape(current_shape, params['kernel_size'], params['stride'], params['padding'])
            # Convolutional layer
            elif layer_type == "Convolutional":
                layer = nn.Sequential(
                    nn.InstanceNorm2d(in_channels) if True else nn.Identity(),
                    nn.LeakyReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=params['out_channels'],
                        kernel_size=params['kernel_size'],
                        stride=params['stride'],
                        padding=params['padding']
                    )
                )
                in_channels = params['out_channels']
                current_shape = calculate_conv_output_shape(current_shape, params['kernel_size'], params['stride'], params['padding'])
            # Pooling layer
            elif layer_type == 'Pooling':
                layer = nn.MaxPool2d(
                    kernel_size=params['kernel_size'],
                    stride=params.get('stride', params['kernel_size']),
                    padding=params.get('padding', 0)
                )
                current_shape = calculate_pool_output_shape(current_shape, params['kernel_size'], params.get('stride', params['kernel_size']), params.get('padding', 0)) 
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            layers.append(layer)
            
        output_shape = (in_channels, *current_shape)
        return nn.Sequential(*layers), output_shape

    def build_decoder(self, layers_config, encoder_output_shape):
        layers = []
        current_shape = encoder_output_shape[1:]
        in_channels = encoder_output_shape[0]
        
        for layer_cfg in layers_config:
            layer_type = layer_cfg['type']
            params = layer_cfg['params']
            
            # Residual layer
            if layer_type == 'ResidualBlock':
                layer = ResidualUpBlock(
                    in_channels=in_channels,
                    out_channels=params['out_channels'],
                    kernel_size=params['kernel_size'],
                    stride=params['stride'],
                    padding=params['padding'],
                    output_padding=params.get('output_padding', 0),
                    bn_active=True,
                    use_cbam=params.get('use_cbam', False),
                    cbam_reduction=params.get('cbam_reduction', 16),
                    cbam_kernel=params.get('cbam_kernel', 7),
                    dropout=self.dropout
                )
                in_channels = params['out_channels']
                current_shape = calculate_deconv_output_shape(current_shape, params['kernel_size'], params['stride'], params['padding'], params.get('output_padding', 0))
            # Convolutional layer
            elif layer_type == "Convolutional":
                layer = nn.Sequential(
                    nn.InstanceNorm2d(in_channels) if True else nn.Identity(),
                    nn.LeakyReLU(inplace=False),
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=params['out_channels'],
                        kernel_size=params['kernel_size'],
                        stride=params['stride'],
                        padding=params['padding']
                    )
                )
                in_channels = params['out_channels']
                current_shape = calculate_conv_output_shape(current_shape, params['kernel_size'], params['stride'], params['padding'])
            # Transopse convolutional layer
            elif layer_type == "TransposeConvolutional":
                layer = nn.Sequential(
                    nn.InstanceNorm2d(in_channels) if True else nn.Identity(),
                    nn.LeakyReLU(inplace=False),
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=params['out_channels'],
                        kernel_size=params['kernel_size'],
                        stride=params['stride'],
                        padding=params['padding'],
                        output_padding=params.get('output_padding', 0),
                    )
                )
                in_channels = params['out_channels']
                current_shape = calculate_deconv_output_shape(current_shape, params['kernel_size'], params['stride'], params['padding'], params.get('output_padding', 0))
            # Upsampling layer
            elif layer_type == 'UpSampling':
                layer = nn.Upsample(
                    scale_factor=params.get('scale_factor', 2),
                    mode=params.get('mode', 'nearest')
                )
                current_shape = [int(dim * params.get('scale_factor', 2)) for dim in current_shape]
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            layers.append(layer)
            
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        return h

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        encoded = self.encode(x)
        # TEST      dodanie szumu gaussowskiego
        # należy eksperymentować z normą szumu- dobrać skalę do bottlenecka
        # encoded = encoded + torch.randn_like(encoded) * 0.1
        # TEST      end
        reconstructed = self.decode(encoded)
        return reconstructed

def load_model_config(filepath):
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config

def initialize_model(config_path, input_shape=(512, 512), device='cuda:0'):
    config = load_model_config(config_path)
    model = ResidualAutoencoder(config, input_shape=input_shape, device=device)
    return model
