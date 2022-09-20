from Model.Modules.Layers import *
from Model.Modules.Prenets import Lip2WavPrenet
from Model.Modules.Attentions import Lip2WavAttention
from torch.autograd import Variable



class Decoder(nn.Module):
    def __init__(self, hp, device='cpu', logger=None):
        super(Decoder, self).__init__()
        self.hp = hp
        self.device = device
        self.loss = torch.zeros(1).to(self.device)
        self.logger = logger
        # self.decayer = Decayer(logger=self.logger, annotate="TF")
        # self.decayer.load(params=hp.main_decayer_params)

        self.scheduled_sampling_method = hp.scheduled_sampling_method.split("+")

        self.num_mels = hp.num_mels
        self.n_frames_per_step = hp.n_frames_per_step

        self.encoder_embedding_dim = hp.encoder_embedding_dim
        self.attention_rnn_dim = hp.attention_rnn_dim
        self.decoder_rnn_dim = hp.decoder_rnn_dim
        self.prenet_dim = hp.prenet_dim

        self.max_decoder_steps = hp.max_decoder_steps
        self.gate_threshold = hp.gate_threshold
        self.p_attention_dropout = hp.p_attention_dropout
        self.p_decoder_dropout = hp.p_decoder_dropout


        self.prenet = Lip2WavPrenet(in_dim=hp.num_mels * hp.n_frames_per_step,
                                    sizes=[hp.prenet_dim, hp.prenet_dim])

        self.attention_rnn = nn.LSTMCell(input_size=hp.prenet_dim + hp.encoder_embedding_dim,
                                         hidden_size=hp.attention_rnn_dim)

        self.attention_layer = Lip2WavAttention(attention_rnn_dim=hp.attention_rnn_dim,
                                         embedding_dim=hp.encoder_embedding_dim,
                                         attention_dim=hp.attention_dim,
                                         attention_location_n_filters=hp.attention_location_n_filters,
                                         attention_location_kernel_size=hp.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(input_size=hp.attention_rnn_dim + hp.encoder_embedding_dim,
                                       hidden_size=hp.decoder_rnn_dim,
                                       bias=True)

        self.linear_projection = LinearNorm(in_dim=hp.decoder_rnn_dim + hp.encoder_embedding_dim,
                                            out_dim=hp.num_mels * hp.n_frames_per_step)

        self.gate_layer = LinearNorm(in_dim=hp.decoder_rnn_dim + hp.encoder_embedding_dim, out_dim=1,
                                     bias=True, w_init_gain='sigmoid')

    # def get_go_frame(self, memory):
    #     """
    #     利用 Encoder 的输出获得一个全 0 的 frames
    #     Gets all zeros frames to use as first decoder input
    #     :param memory: decoder outputs
    #     :return: decoder_input
    #         :decoder_input: all zeros frames, Size=[1, Batch, num_mels * n_frames_per_step]
    #     """
    #     B = memory.size(0)  # batch_size
    #     decoder_input = Variable(memory.data.new(B, self.num_mels * self.n_frames_per_step).zero_())
    #     return decoder_input
    #
    # def get_l1_noise(self, shape, coeff=1.0):
    #     noise = torch.from_numpy(np.float32(np.random.uniform(-2, 2, size=shape))).to(self.device) * self.loss * coeff
    #     return noise
    #
    # def get_mse_noise(self, shape, coeff=1.0):
    #     noise = torch.from_numpy(np.float32(np.random.uniform(0, 2, size=shape))).to(self.device) * self.loss * coeff
    #     pn = ((torch.rand(shape) < 0.5) * -2 + 1).to(self.device)
    #     noise = torch.sqrt(noise) * pn
    #     return noise

    def parse_decoder_inputs(self, decoder_inputs):
        """
        Prepares decoder inputs, i.e. mel outputs
        :param decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        :return: inputs
            :inputs: processed decoder inputs
        """

        # (B, num_mels, T_out) -> (B, T_out, num_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        # (B, T_out, num_mels) -> (B, T_out/n_frames_per_step, num_mels*n_frames_per_step)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), int(decoder_inputs.size(1) / self.n_frames_per_step), -1)
        # (B, T_out, num_mels) -> (T_out, B, num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def initialize_decoder_states(self, memory, mask):
        """
        初始化 Attention-RNN 状态、Decoder-RNN 状态、注意力权重、注意力累积权重、注意力上下文、存储内存和存储已处理的内存
        Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        :param memory: Encoder outputs
        :param mask: Mask for padded data if training, expects None for inference
        :return: None
        """

        B = memory.size(0)          # batch_size
        MAX_TIME = memory.size(1)   # max frame_count of landmark

        # 元素全部为 0 的张量
        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_outputs(self, spec_outputs, mask_outputs, alignments):
        """
        Prepares decoder outputs for output
        :param mel_outputs:
        :param gate_outputs: gate output energies
        :param alignments:
        :return: mel_outputs, gate_outputs, alignments
            :mel_outputs:
            :gate_outputs: gate output energies
            :alignments:
        """
        # mel_outputs
        ### (T_out, B, num_mels) -> (B, T_out, num_mels)
        spec_outputs = torch.stack(spec_outputs).transpose(0, 1).contiguous()
        ### decouple frames per step
        spec_outputs = spec_outputs.view(spec_outputs.size(0), -1, self.num_mels)
        ### (B, T_out, num_mels) -> (B, num_mels, T_out)
        spec_outputs = spec_outputs.transpose(1, 2)

        alignments = torch.stack(alignments).transpose(0, 1).contiguous()

        return spec_outputs, alignments

    def decoding(self, decoder_input):
        """
        Decoder step using stored states, attention and memory
        :param decoder_input: previous mel output
        :return:
            :mel_output:                            # [64, 256]
            :gate_output: gate output energies      # [64, 1]
            :attention_weights:                     # [64, 960]
        """

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)
        self.attention_weights_cum += self.attention_weights

        decoder_cell_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_cell_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        # gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, None, self.attention_weights

    def forward(self, memory, target):
        """
        训练阶段模型时使用的 Decoder Forward
        采取了 teacher forcing，即使用前一时间步的 ground truth 预测下一时间步的 mels
        :param memory: Encoder 的输出      torch.Size([64, 960, 478, 3])
        :param target: 采取 teacher forcing 的 Decoder 输入，即作为 ground truth 的 mel-specs
        :return: mel_outputs, gate_outputs, alignments
            :mel_outputs: mel outputs from the decoder
            :gate_outputs: gate outputs from the decoder        bts * 1382/2
            :alignments: sequence of attention weights from the decoder
        """

        ratio = self.decayer()

        start_tag = self.get_go_frame(memory).unsqueeze(0)
        target_input = self.parse_decoder_inputs(target)
        decoder_inputs = torch.cat((start_tag, target_input), dim=0)

        self.initialize_decoder_states(memory=memory, mask=None)
        spec_outputs, alignments = [], []
        last_output = decoder_inputs[0]
        while len(spec_outputs) < decoder_inputs.size(0) - 1:
            if torch.rand(1) <= ratio:
                decoder_input = decoder_inputs[len(spec_outputs)]       # teacher-force
            else:
                decoder_input = last_output.clone().detach()  # auto-regressive

            decoder_input = self.prenet(decoder_input)
            spec_output, mask_output, attention_weights = self.decoding(decoder_input)

            spec_outputs += [spec_output.squeeze(1)]
            alignments += [attention_weights]
            last_output = spec_output

        spec_outputs, alignments = self.parse_decoder_outputs(spec_outputs=spec_outputs, mask_outputs=None, alignments=alignments)

        return spec_outputs, alignments

    def inference(self, memory):
        """
        Decoder inference
        :param memory: Encoder outputs
        :return: mel_outputs, gate_outputs, alignments
            :mel_outputs: mel outputs from the decoder
            :gate_outputs: gate outputs from the decoder
            :alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory=memory, mask=None)

        spec_outputs, alignments = [], []
        while len(spec_outputs) < self.max_decoder_steps:
            decoder_input = self.prenet(decoder_input)

            spec_output, mask_output, attention_weights = self.decoding(decoder_input)
            spec_outputs += [spec_output.squeeze(1)]
            alignments += [attention_weights]

            decoder_input = spec_output
        spec_outputs, alignments = self.parse_decoder_outputs(spec_outputs=spec_outputs, mask_outputs=None, alignments=alignments)

        return spec_outputs, alignments

    def teacher_infer(self, memory, target):

        start_tag = self.get_go_frame(memory).unsqueeze(0)
        target_input = self.parse_decoder_inputs(target)
        decoder_inputs = torch.cat((start_tag, target_input), dim=0)
        self.initialize_decoder_states(memory, mask=None)

        spec_outputs, alignments = [], []
        while len(spec_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(spec_outputs)]
            decoder_input = self.prenet(decoder_input)

            spec_output, mask_output, alignment = self.decoding(decoder_input)
            spec_outputs += [spec_output.squeeze(1)]
            alignments += [alignment]

        spec_outputs, alignments = self.parse_decoder_outputs(spec_outputs, None, alignments)

        return spec_outputs, alignments


class Lip2WavDecoder(nn.Module):
    def __init__(self, hp):
        super(Lip2WavDecoder, self).__init__()
        self.num_mels = hp.num_mels
        self.n_frames_per_step = hp.n_frames_per_step
        self.encoder_embedding_dim = hp.encoder_embedding_dim
        self.attention_rnn_dim = hp.attention_rnn_dim
        self.decoder_rnn_dim = hp.decoder_rnn_dim
        self.prenet_dim = hp.prenet_dim
        self.max_decoder_steps = hp.max_decoder_steps
        self.gate_threshold = hp.gate_threshold
        self.p_attention_dropout = hp.p_attention_dropout
        self.p_decoder_dropout = hp.p_decoder_dropout

        self.prenet = Lip2WavPrenet(
            hp.num_mels * hp.n_frames_per_step,
            [hp.prenet_dim, hp.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hp.prenet_dim + hp.encoder_embedding_dim,
            hp.attention_rnn_dim)

        self.attention_layer = Lip2WavAttention(
            hp.attention_rnn_dim, hp.encoder_embedding_dim,
            hp.attention_dim, hp.attention_location_n_filters,
            hp.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hp.attention_rnn_dim + hp.encoder_embedding_dim,
            hp.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hp.decoder_rnn_dim + hp.encoder_embedding_dim,
            hp.num_mels * hp.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hp.decoder_rnn_dim + hp.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        ''' Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        '''
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.num_mels * self.n_frames_per_step).zero_())
        print(decoder_input)
        print(decoder_input.size())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        ''' Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        '''
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        ''' Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        '''
        # (B, num_mels, T_out) -> (B, T_out, num_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, num_mels) -> (T_out, B, num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        ''' Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        '''
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)

        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, num_mels) -> (B, T_out, num_mels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.num_mels)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        ''' Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        '''
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))

        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)

        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)

        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        ''' Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        print('Encoder outputs', memory.size())
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)
        # print('decoder_input', decoder_input.size())

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))
        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            print('mel_output', mel_output.size())
            print('gate_output', gate_output.size())
            print('attention_weights', attention_weights.size())
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        ''' Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if sum(torch.sigmoid(gate_output.data))/len(gate_output.data) > self.gate_threshold:
                print('Terminated by gate.')
                break
            # elif len(mel_outputs) > 1 and is_end_of_frames(mel_output):
            # 	print('Warning: End with low power.')
            # 	break
            elif len(mel_outputs) == self.max_decoder_steps:
                print('Warning: Reached max decoder steps.')
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments
