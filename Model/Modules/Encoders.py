from Model.Modules.Layers import *


# class Encoder(nn.Module):
#     '''Encoder module:
#     Landmark Encoder
#     '''
#
#     def __init__(self, hp):
#         super(Encoder, self).__init__()
#
#         self.convolutions = nn.ModuleList()
#         for _ in range(hp.encoder_n_convolutions):
#             conv_layer = nn.Sequential(
#                 ConvNorm(hp.landmark_pad,
#                            hp.landmark_pad,
#                            kernel_size=hp.encoder_kernel_size, stride=1,
#                            padding=int((hp.encoder_kernel_size - 1) / 2), dilation=1,
#                            w_init_gain='relu'),
#                 nn.BatchNorm1d(hp.landmark_pad)
#             )
#             self.convolutions.append(conv_layer)
#
#         # self.dimReduce = LinearNorm(in_dim=3, out_dim=1, bias=True)
#         # self.dim_reduce = LinearNormList(size_seq=[3, 8, 16, 8, 1])
#
#         self.lstm = nn.LSTM(input_size=hp.fmc.landmark_count,
#                          hidden_size=int(hp.fmc.landmark_count/2),
#                          num_layers=1, batch_first=True, bidirectional=True)
#
#     def forward(self, input, input_lengths):
#         """
#
#         :param input: bts * max_len * fmc_count * 3         8, 960, 80, 3
#         :param input_lengths: bts
#         :return: bts * max_len * fmc_count                  8, 960, 80
#         """
#         reduce_input = torch.squeeze(self.dim_reduce(input), dim=3)
#         for conv in self.convolutions:
#             reduce_input = F.dropout(F.relu(conv(reduce_input)), 0.5, self.training)
#         input_lengths = input_lengths.cpu().numpy()
#         t = nn.utils.rnn.pack_padded_sequence(reduce_input, input_lengths, batch_first=True)
#
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(t)
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#
#         return outputs
#
#     def inference(self, input):
#         reduce_input = torch.squeeze(self.dim_reduce(input), dim=3)
#         for conv in self.convolutions:
#             reduce_input = F.dropout(F.relu(conv(reduce_input)), 0.5, self.training)
#         # embedding = embedding.transpose(1, 2)
#
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(reduce_input)
#         return outputs

class Encoder2D(nn.Module):
    """
    Encoder module:
        - Three 3-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hp):
        super(Encoder2D, self).__init__()

        # self.transLayer = TransLayer(hp, channel_first=True)
        out_channel = hp.num_init_filters
        in_channel = 3
        convolutions = []
        for i in range(hp.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm2D(in_channel, out_channel,
                           kernel_size=(5 if i == 0 else 3), stride=(1, 2),
                           # padding=int((hp.encoder_kernel_size - 1) / 2),
                           dilation=1, w_init_gain='relu'),
                ConvNorm2D(out_channel, out_channel,
                           kernel_size=3, stride=1,
                           # padding=int((hp.encoder_kernel_size - 1) / 2),
                           dilation=1, w_init_gain='relu', residual=True),
                ConvNorm2D(out_channel, out_channel,
                           kernel_size=3, stride=1,
                           # padding=int((hp.encoder_kernel_size - 1) / 2),
                           dilation=1, w_init_gain='relu', residual=True)
            )
            convolutions.append(conv_layer)

            if i == hp.encoder_n_convolutions - 1:
                conv_layer = nn.Sequential(
                    ConvNorm2D(out_channel, out_channel,
                               kernel_size=3, stride=(1, 3),
                               # padding=int((hp.encoder_n_convolutions - 1) / 2),
                               dilation=1, w_init_gain='relu'))
                convolutions.append(conv_layer)

            in_channel = out_channel
            out_channel *= 2


        self.convolutions = nn.ModuleList(convolutions)
        self.lstm = nn.LSTM(hp.encoder_embedding_dim, int(hp.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)
        self.si_lstm = nn.LSTM(hp.encoder_embedding_dim, hp.encoder_embedding_dim, 1,
                            batch_first=True, bidirectional=False)

    def forward(self, input):
        """
        Args:
            input: [bts, t_len, fmc_len, 3]                 (8, 90, 80, 3)

        Returns: [bts, t_len, hp.encoder_embedding_dim]     (8, 90, 384)

        """
        # (8, 3, 90, 80)
        input = self.transLayer(input)
        for conv in self.convolutions:
            input = F.dropout(conv(input), 0.5, self.training)
        input = input.permute(0, 2, 1, 3).squeeze(3).contiguous()  # [bs x 90 x encoder_embedding_dim]
        outputs, _ = self.lstm(input)

        return outputs

    def inference(self, input):
        for conv in self.convolutions:
            input = F.dropout(conv(input), 0.5, self.training)
        input = input.permute(0, 2, 1, 3).squeeze(3).contiguous()
        outputs, _ = self.lstm(input)

        return outputs

# class Lip2WavEncoder(nn.Module):
# 	'''Encoder module:
# 		- Three 1-d convolution banks
# 		- Bidirectional LSTM
# 	'''
# 	def __init__(self, hp):
# 		super(Lip2WavEncoder, self).__init__()
#
# 		convolutions = []
# 		for _ in range(hp.encoder_n_convolutions):
# 			conv_layer = nn.Sequential(
# 				ConvNorm(hp.encoder_embedding_dim,
# 						 hp.encoder_embedding_dim,
# 						 kernel_size=hp.encoder_kernel_size, stride=1,
# 						 padding=int((hp.encoder_kernel_size - 1) / 2),
# 						 dilation=1, w_init_gain='relu'),
# 				nn.BatchNorm1d(hp.encoder_embedding_dim))
# 			convolutions.append(conv_layer)
# 		self.convolutions = nn.ModuleList(convolutions)
#
# 		self.lstm = nn.LSTM(hp.encoder_embedding_dim,
# 							int(hp.encoder_embedding_dim / 2), 1,
# 							batch_first=True, bidirectional=True)
#
# 	def forward(self, x, input_lengths):
# 		for conv in self.convolutions:
# 			x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#
# 		x = x.transpose(1, 2)
#
# 		# pytorch tensor are not reversible, hence the conversion
# 		input_lengths = input_lengths.cpu().numpy()
# 		x = nn.utils.rnn.pack_padded_sequence(
# 			x, input_lengths, batch_first=True)
#
# 		self.lstm.flatten_parameters()
# 		outputs, _ = self.lstm(x)
#
# 		outputs, _ = nn.utils.rnn.pad_packed_sequence(
# 			outputs, batch_first=True)
#
# 		return outputs
#
# 	def inference(self, x):
# 		for conv in self.convolutions:
# 			x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#
# 		x = x.transpose(1, 2)
#
# 		self.lstm.flatten_parameters()
# 		outputs, _ = self.lstm(x)
#
# 		return outputs

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

    def forward(self, input):
        for conv in self.convolutions:
            input = F.dropout(conv(input), 0.5, self.training)
        input = input.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()  # [bs x 90 x encoder_embedding_dim]
        outputs, _ = self.lstm(input)

        return outputs

    def inference(self, input):
        for conv in self.convolutions:
            input = F.dropout(conv(input), 0.5, self.training)
        input = input.permute(0, 2, 1, 3, 4).squeeze(4).squeeze(3).contiguous()
        outputs, _ = self.lstm(input)	#x:B,T,C

        return outputs

