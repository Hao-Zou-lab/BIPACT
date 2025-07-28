#!/usr/bin/env python
import torch.nn as nn

def init_weights(m):
    """ initialize weights of fully connected layer
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# autoencoder with hidden units 20, 2, 20
# Encoder
class Encoder_2(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, 2))
        self.encoder.apply(init_weights)
    def forward(self, x):
        x = self.encoder(x)
        return x
# Decoder
class Decoder_2(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_2, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, num_inputs),
            nn.ReLU())
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x
# Autoencoder
class autoencoder_2(nn.Module):
    def __init__(self, num_inputs):
        super(autoencoder_2, self).__init__()
        self.encoder = Encoder_2(num_inputs)
        self.decoder = Decoder_2(num_inputs)
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return code, x


# autoencoder with hidden units 200, 20, 200
# Encoder
class Encoder_20(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder_20, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 200),
            nn.ReLU(),
            nn.Linear(200, 20))
        self.encoder.apply(init_weights)
    def forward(self, x):
        x = self.encoder(x)
        return x
# Decoder
class Decoder_20(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_20, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(),
            nn.Linear(200, num_inputs),
           	nn.ReLU())
        self.decoder.apply(init_weights)
    def forward(self, x):
        x = self.decoder(x)
        return x
# Autoencoder
class autoencoder_20(nn.Module):
    def __init__(self, num_inputs):
        super(autoencoder_20, self).__init__()
        self.encoder = Encoder_20(num_inputs)
        self.decoder = Decoder_20(num_inputs)
    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return code, x






import torch
import torch.nn as nn
import torch.nn.functional as F

# 初始化权重函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Encoder
class Encoder_20V(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder_20V, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 200)
        self.fc_mu = nn.Linear(200, 20)  # 潜在空间的均值
        self.fc_logvar = nn.Linear(200, 20)  # 潜在空间的对数方差
        
        self.apply(init_weights)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder_20V(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_20V, self).__init__()
        self.fc1 = nn.Linear(20, 200)
        self.fc2 = nn.Linear(200, num_inputs)

        self.apply(init_weights)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))  # 使用 sigmoid 激活函数确保输出在 [0, 1] 范围内

# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, num_inputs):
        super(VAE, self).__init__()
        self.encoder = Encoder_20V(num_inputs)
        self.decoder = Decoder_20V(num_inputs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)  # 这就是隐层表示
        x_recon = self.decoder(z)
        return z, mu, logvar, x_recon  # 返回隐层表示、均值、对数方差和重建结果

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')  # 不再使用 with_logits
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD












def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Encoder_50V(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder_50V, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc_mu = nn.Linear(200, 50)
        self.fc_logvar = nn.Linear(200, 50)
        self.dropout = nn.Dropout(0.1)

        self.apply(init_weights)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder_50V(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_50V, self).__init__()
        self.fc1 = nn.Linear(50, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, num_inputs)
        self.dropout = nn.Dropout(0.1)

        self.apply(init_weights)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        # 注意，Dropout 在最后的全连接层前已经被应用，不在最后一个全连接层后使用
        return torch.sigmoid(self.fc3(h))  # 确保输出在 [0, 1] 范围内

class VAE50(nn.Module):
    def __init__(self, num_inputs):
        super(VAE50, self).__init__()
        self.encoder = Encoder_50V(num_inputs)
        self.decoder = Decoder_50V(num_inputs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return z, mu, logvar, x_recon

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


