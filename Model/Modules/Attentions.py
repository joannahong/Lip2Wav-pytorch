from Model.Modules.Layers import *


class Lip2WavAttention(nn.Module):
	def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
				 attention_location_n_filters, attention_location_kernel_size):
		super(Lip2WavAttention, self).__init__()
		self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
									  bias=False, w_init_gain='tanh')
		self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
									   w_init_gain='tanh')
		self.v = LinearNorm(attention_dim, 1, bias=False)
		self.location_layer = LocationLayer(attention_location_n_filters,
											attention_location_kernel_size,
											attention_dim)
		self.score_mask_value = -float('inf')

	def get_alignment_energies(self, query, processed_memory,
							   attention_weights_cat):
		'''
		PARAMS
		------
		query: decoder output (batch, num_mels * n_frames_per_step)
		processed_memory: processed encoder outputs (B, T_in, attention_dim)
		attention_weights_cat: cumulative and prev. att we;[ights (B, 2, max_time)

		RETURNS
		-------
		alignment (batch, max_time)
		'''

		processed_query = self.query_layer(query.unsqueeze(1))
		processed_attention_weights = self.location_layer(attention_weights_cat)
		energies = self.v(torch.tanh(
			processed_query + processed_attention_weights + processed_memory))

		energies = energies.squeeze(-1)
		return energies

	def forward(self, attention_hidden_state, memory, processed_memory,
				attention_weights_cat, mask):
		'''
		PARAMS
		------
		attention_hidden_state: attention rnn last output
		memory: encoder outputs
		processed_memory: processed encoder outputs
		attention_weights_cat: previous and cummulative attention weights
		mask: binary mask for padded data
		'''
		alignment = self.get_alignment_energies(
			attention_hidden_state, processed_memory, attention_weights_cat)

		if mask is not None:
			alignment.data.masked_fill_(mask, self.score_mask_value)

		attention_weights = F.softmax(alignment, dim=1)
		attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
		attention_context = attention_context.squeeze(1)

		return attention_context, attention_weights
