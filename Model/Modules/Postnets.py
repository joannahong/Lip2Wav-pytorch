from Model.Modules.Layers import *




class Lip2WavPostnet(nn.Module):
	'''Postnet
		- Five 1-d convolution with 512 channels and kernel size 5
	'''

	def __init__(self, hp):
		super(Lip2WavPostnet, self).__init__()
		self.convolutions = nn.ModuleList()

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hp.num_mels, hp.postnet_embedding_dim,
						 kernel_size=hp.postnet_kernel_size, stride=1,
						 padding=int((hp.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='tanh'),
				nn.BatchNorm1d(hp.postnet_embedding_dim))
		)

		for i in range(1, hp.postnet_n_convolutions - 1):
			self.convolutions.append(
				nn.Sequential(
					ConvNorm(hp.postnet_embedding_dim,
							 hp.postnet_embedding_dim,
							 kernel_size=hp.postnet_kernel_size, stride=1,
							 padding=int((hp.postnet_kernel_size - 1) / 2),
							 dilation=1, w_init_gain='tanh'),
					nn.BatchNorm1d(hp.postnet_embedding_dim))
			)

		self.convolutions.append(
			nn.Sequential(
				ConvNorm(hp.postnet_embedding_dim, hp.num_mels,
						 kernel_size=hp.postnet_kernel_size, stride=1,
						 padding=int((hp.postnet_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='linear'),
				nn.BatchNorm1d(hp.num_mels))
			)

	def forward(self, x):
		for i in range(len(self.convolutions) - 1):
			x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
		x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

		return x
