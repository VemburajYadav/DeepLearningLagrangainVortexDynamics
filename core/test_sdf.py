import numpy as np
import torch
import matplotlib.pyplot as plt

box = [100, 100]

y = np.linspace(0, box[0], 20)
y = np.reshape(y, (y.shape[0], 1))
y = np.tile(y, (1, y.shape[0]))

x = np.linspace(0, box[1], 20)
x = np.reshape(x, (1, x.shape[0]))
x = np.tile(x, (x.shape[1], 1))

y_sdf = np.minimum(y, box[0] - y)
x_sdf = np.minimum(x, box[1] - x)

sdf = np.minimum(y_sdf, x_sdf)

plt.figure()
plt.imshow(sdf)
plt.show()

yc = torch.tensor(y, dtype=torch.float32)
xc = torch.tensor(x, dtype=torch.float32)

# xc = [10.0, 90.0]

yc_pt = torch.tensor(yc, dtype=torch.float32, requires_grad=True)
xc_pt = torch.tensor(xc, dtype=torch.float32, requires_grad=True)

box_pt = torch.tensor(box, dtype=torch.float32)
y_sdf_pt = torch.min(yc_pt, box_pt[1] - yc_pt)
x_sdf_pt = torch.min(xc_pt, box_pt[0] - xc_pt)

sdf_pt = torch.min(y_sdf_pt, x_sdf_pt)
grad_y = torch.autograd.grad(torch.unbind(sdf_pt, dim=-1), yc_pt, retain_graph=True)
grad_x = torch.autograd.grad(torch.unbind(sdf_pt, dim=-1), xc_pt)

x1, y1, x2, y2 = 40, 50, 80, 70


# print(grad_y, grad_x)
# plt.figure()
# plt.plot(sdf[1, :])
# plt.plot(sdf[100, :])
# plt.plot(sdf[200, :])
# plt.plot(sdf[300, :])
# plt.plot(sdf[400, :])
# plt.plot(sdf[500, :])
# plt.show()



