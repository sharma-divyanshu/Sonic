import torch.nn as nn
import torch.nn.functional as FUNCTION


class PPO(nn.Module):
    def __init__(self, input_count, action_count):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(input_count, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, action_count)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = FUNCTION.relu(self.conv1(x))
        x = FUNCTION.relu(self.conv2(x))
        x = FUNCTION.relu(self.conv3(x))
        x = FUNCTION.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x)

