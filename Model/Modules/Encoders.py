from Model.Modules.Layers import *


class Lip2WavEncoder(nn.Module):
	'''Encoder module:
		- Three 1-d convolution banks
		- Bidirectional LSTM
	'''
	def __init__(self, hp):
		super(Lip2WavEncoder, self).__init__()

		convolutions = []
		for _ in range(hp.encoder_n_convolutions):
			conv_layer = nn.Sequential(
				ConvNorm(hp.encoder_embedding_dim,
						 hp.encoder_embedding_dim,
						 kernel_size=hp.encoder_kernel_size, stride=1,
						 padding=int((hp.encoder_kernel_size - 1) / 2),
						 dilation=1, w_init_gain='relu'),
				nn.BatchNorm1d(hp.encoder_embedding_dim))
			convolutions.append(conv_layer)
		self.convolutions = nn.ModuleList(convolutions)

		self.lstm = nn.LSTM(hp.encoder_embedding_dim,
							int(hp.encoder_embedding_dim / 2), 1,
							batch_first=True, bidirectional=True)

	def forward(self, x, input_lengths):
		for conv in self.convolutions:
			x = F.dropout(F.relu(conv(x)), 0.5, self.training)

		x = x.transpose(1, 2)

		# pytorch tensor are not reversible, hence the conversion
		input_lengths = input_lengths.cpu().numpy()
		x = nn.utils.rnn.pack_padded_sequence(
			x, input_lengths, batch_first=True)

		self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x)

		outputs, _ = nn.utils.rnn.pad_packed_sequence(
			outputs, batch_first=True)

		return outputs

	def inference(self, x):
		for conv in self.convolutions:
			x = F.dropout(F.relu(conv(x)), 0.5, self.training)

		x = x.transpose(1, 2)

		self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x)

		return outputs


class Lip2WavEncoder3D(nn.Module):
	"""Encoder module:
        - Three 3-d convolution banks
        - Bidirectional LSTM
    """

	def __init__(self, hp):
		super(Lip2WavEncoder3D, self).__init__()

		self.hp = hp
		self.out_channel = hp.num_init_filters
		self.in_channel = 3
		convolutions = []

		for i in range(hp.encoder_n_convolutions):
			if i == 0:
				conv_layer = nn.Sequential(
					ConvNorm3D(self.in_channel, self.out_channel,
								kernel_size=5, stride=(1, 2, 2),
								# padding=int((hparams.encoder_kernel_size - 1) / 2),
								dilation=1, w_init_gain='relu'),
					ConvNorm3D(self.out_channel, self.out_channel,
							   kernel_size=3, stride=1,
							   # padding=int((hparams.encoder_kernel_size - 1) / 2),
							   dilation=1, w_init_gain='relu', residual=True),
					ConvNorm3D(self.out_channel, self.out_channel,
							   kernel_size=3, stride=1,
							   # padding=int((hparams.encoder_kernel_size - 1) / 2),
							   dilation=1, w_init_gain='relu', residual=True)
				)
				convolutions.append(conv_layer)
			else:
				conv_layer = nn.Sequential(
					ConvNorm3D(self.in_channel, self.out_channel,
							   kernel_size=3, stride=(1, 2, 2),
							   # padding=int((hparams.encoder_kernel_size - 1) / 2),
							   dilation=1, w_init_gain='relu'),
					ConvNorm3D(self.out_channel, self.out_channel,
							   kernel_size=3, stride=1,
							   # padding=int((hparams.encoder_kernel_size - 1) / 2),
							   dilation=1, w_init_gain='relu', residual=True),
					ConvNorm3D(self.out_channel, self.out_channel,
							   kernel_size=3, stride=1,
							   # padding=int((hparams.encoder_kernel_size - 1) / 2),
							   dilation=1, w_init_gain='relu', residual=True)
				)
				convolutions.append(conv_layer)

			if i == hp.encoder_n_convolutions - 1:
				conv_layer = nn.Sequential(
					ConvNorm3D(self.out_channel, self.out_channel,
							   kernel_size=3, stride=(1, 3, 3),
							   # padding=int((hparams.encoder_kernel_size - 1) / 2),
							   dilation=1, w_init_gain='relu'))
				convolutions.append(conv_layer)

			self.in_channel = self.out_channel
			self.out_channel *= 2
		self.convolutions = nn.ModuleList(convolutions)

		self.lstm = nn.LSTM(hp.encoder_embedding_dim,
		                    int(hp.encoder_embedding_dim / 2), 1,
		                    batch_first=True, bidirectional=True)

	def forward(self, x, input_lengths):
		for conv in self.convolutions:
			x = F.dropout(conv(x), 0.5, self.training)
		# for i in range(len(self.convolutions)):
		# 	if i==0 or i==1 or i ==2:
		# 		with torch.no_grad():
		# 			x = F.dropout(self.convolutions[i](x), 0.5, self.training)
		# 	else:
		# 		x = F.dropout(self.convolutions[i](x), 0.5, self.training)

		x = x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()  # [bs x 90 x encoder_embedding_dim]
		print(x.size())
		# pytorch tensor are not reversible, hence the conversion
		input_lengths = input_lengths.cpu().numpy()
		# x = nn.utils.rnn.pack_padded_sequence(
		# 	x, input_lengths, batch_first=True)

		# self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x)
		print('outputs',outputs.size())
		# outputs, _ = nn.utils.rnn.pad_packed_sequence(
		# 	outputs, batch_first=True)
		# print('outputs', outputs.size())

		return outputs

	def inference(self, x):
		for conv in self.convolutions:
			x = F.dropout(conv(x), 0.5, self.training)

		x = x.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()
		# self.lstm.flatten_parameters()
		outputs, _ = self.lstm(x)	#x:B,T,C

		return outputs

