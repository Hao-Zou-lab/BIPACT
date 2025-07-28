import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Function

class GradReverse(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -2)

def grad_reverse(x):
    return GradReverse.apply(x)

class Discriminator(nn.Module):
    def __init__(self, h_dim):
        super(Discriminator, self).__init__()
        self.D1 = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = grad_reverse(x)
        yhat = self.D1(x)
        return torch.softmax(yhat, dim=1)

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        return x

def add_noise(x, p=0.2):
    noise = torch.bernoulli(torch.full_like(x, 1-p))
    x_noisy = x * noise
    return x_noisy

def cosine_similarity_loss(codes, cell_types, non_similar_weight_factor=1):
    cosine_loss = 0
    total_weight = 0
    unique_cell_types = torch.unique(cell_types)
    class_weights = {}
    for cell_type in unique_cell_types:
        samples_count = (cell_types == cell_type).sum().item()
        class_weights[cell_type.item()] = 1.0 / samples_count if samples_count > 0 else 0

    for i in range(len(unique_cell_types)):
        type_i = unique_cell_types[i].item()  # 使用item()确保键是标量
        codes_i = codes[cell_types == type_i]
        weight_i = class_weights[type_i]  # 此处应该不再抛出KeyError

        if codes_i.size(0) < 2:
            continue
        
        for j in range(i, len(unique_cell_types)):
            type_j = unique_cell_types[j].item()
            codes_j = codes[cell_types == type_j]
            weight_j = class_weights[type_j]

            if codes_j.size(0) < 1:
                continue

            cos_sim = F.cosine_similarity(codes_i.unsqueeze(1), codes_j.unsqueeze(0), dim=2)
            mean_cos_sim = cos_sim.mean()

            # 调整相似度损失计算，考虑类别权重
            if i == j:
                # 同类之间
                cosine_loss -= mean_cos_sim * weight_i
            else:
                # 非同类之间，增加非同类之间的权重
                cosine_loss += mean_cos_sim * (weight_i + weight_j) / 2 * non_similar_weight_factor

            total_weight += (weight_i + weight_j) / 2 if i != j else weight_i

    return cosine_loss / total_weight if total_weight > 0 else torch.tensor(0.0, device=codes.device)
