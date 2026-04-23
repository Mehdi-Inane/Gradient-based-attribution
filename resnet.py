import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBottleneck(nn.Module):
    """
    Translates the `BottleNeckBlockV2` from the Feldman & Zhang Sonnet code.
    Pre-activation bottleneck block: BatchNorm -> ReLU -> Conv
    """
    def __init__(self, in_channels, channels, stride, use_projection):
        super(PreActBottleneck, self).__init__()
        self.use_projection = use_projection

        # Equivalent to self._proj_conv in Sonnet
        if self.use_projection:
            self.proj_conv = nn.Conv2d(
                in_channels, channels * 4, kernel_size=1, stride=stride, bias=False
            )

        # conv_0 & bn_0
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv0 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)

        # conv_1 & bn_1
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # conv_2 & bn_2
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # i = 0 in Sonnet loop
        net = F.relu(self.bn0(x))
        if self.use_projection:
            shortcut = self.proj_conv(net)
        else:
            shortcut = x
        net = self.conv0(net)

        # i = 1 in Sonnet loop
        net = self.conv1(F.relu(self.bn1(net)))

        # i = 2 in Sonnet loop
        net = self.conv2(F.relu(self.bn2(net)))

        return net + shortcut


class CifarResNet50(nn.Module):
    """
    Translates `CifarResNet50` (which inherits from `ResNetV2`) in the Sonnet code.
    Designed for small inputs (small_input=True).
    """
    def __init__(self, num_classes=100):
        super(CifarResNet50, self).__init__()
        
        # initial_conv for small_input=True (3x3, stride 1)
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # ResNet50 block config
        blocks_per_group = [3, 4, 6, 3]
        channels_per_group = [64, 128, 256, 512]
        strides = [1, 2, 2, 2] # Initial stride is 1 for the first group

        self.groups = nn.Sequential()
        in_channels = 64

        for i in range(len(blocks_per_group)):
            group = nn.Sequential()
            channels = channels_per_group[i]
            num_blocks = blocks_per_group[i]
            group_stride = strides[i]

            for j in range(num_blocks):
                # use_projection=(id_block == 0) and stride logic
                use_proj = (j == 0)
                blk_stride = group_stride if j == 0 else 1
                
                group.add_module(
                    f"block_{j}", 
                    PreActBottleneck(in_channels, channels, blk_stride, use_proj)
                )
                in_channels = channels * 4  # Bottleneck output is always channels * 4

            self.groups.add_module(f"block_group_{i}", group)

        # final_batchnorm
        self.final_batchnorm = nn.BatchNorm2d(in_channels)
        
        # logits
        self.logits = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        net = self.initial_conv(x)
        
        # Note: small_input=True skips the max_pool2d in Sonnet code
        net = self.groups(net)
        
        net = self.final_batchnorm(net)
        net = F.relu(net)
        
        # final_avg_pool: tf.reduce_mean(net, axis=[1, 2])
        net = net.mean(dim=[2, 3]) 
        
        return self.logits(net)

def get_modified_resnet50(num_classes=100):
    """Wrapper to maintain API consistency with our previous step."""
    return CifarResNet50(num_classes=num_classes)