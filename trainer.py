import torch
import numpy as np
import pyswarms as ps
from model import WeatherNet

# Mock training data (replace with your CSV loading logic)
X_train = torch.randn(100, 3) 
y_train = torch.randn(100, 1)
model = WeatherNet()

def fitness(particles):
    costs = []
    for p in particles:
        # Map particle position to model weights
        start = 0
        for param in model.parameters():
            size = param.numel()
            param.data = torch.tensor(p[start:start+size].reshape(param.shape)).float()
            start += size
        # MSE Loss
        loss = torch.nn.functional.mse_loss(model(X_train), y_train)
        costs.append(loss.item())
    return np.array(costs)

# Run Optimization
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=model.get_num_params(), 
                                    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9})
best_cost, best_pos = optimizer.optimize(fitness, iters=50)

# Save the trained brain
np.save('best_weights.npy', best_pos)
print("Optimization finished. best_weights.npy saved.")