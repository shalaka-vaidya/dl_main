import torch

a = torch.randn(100000, 100000).cuda()
b = torch.randn(100000, 100000).cuda()
while True:
	ans = a.mm(b.t()).cuda()
	del ans