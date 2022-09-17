from Model.Modules.Layers import *



class Lip2WavPrenet(nn.Module):
	def __init__(self, in_dim, sizes):
		super(Lip2WavPrenet, self).__init__()
		in_sizes = [in_dim] + sizes[:-1]
		self.layers = nn.ModuleList(
			[LinearNorm(in_size, out_size, bias=False)
			 for (in_size, out_size) in zip(in_sizes, sizes)])

	def forward(self, x):
		for linear in self.layers:
			x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
		return x
