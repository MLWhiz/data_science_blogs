import torch
import time

start = time.time()
N, D_in, D_out = 64, 1024, 512
x = torch.randn(N, D_in, device="cuda")
y = torch.randn(N, D_out, device="cuda")
model = torch.nn.Linear(D_in, D_out).cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
for to in range(50000):
    y_pred = model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
end = time.time()
print(end - start)
