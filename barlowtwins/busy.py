import torch

a = torch.randn(10000, 10000).cuda()
b = torch.randn(10000, 10000).cuda()
while True:
	ans = a.mm(b.t()).cuda()
	del ans