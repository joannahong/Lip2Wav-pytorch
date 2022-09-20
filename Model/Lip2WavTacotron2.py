import torch
from torch import nn
# from math import sqrt
from utils import to_var, get_mask_from_lengths
from Model.Modules.Encoders import Lip2WavEncoder3D
from Model.Modules.Decoders import Lip2WavDecoder
from Model.Modules.Postnets import Lip2WavPostnet

class Lip2WavTacotron2(nn.Module):
    def __init__(self, hp):
        super(Lip2WavTacotron2, self).__init__()
        self.hp = hp
        self.num_mels = hp.num_mels
        self.mask_padding = hp.mask_padding
        self.n_frames_per_step = hp.n_frames_per_step
        # self.embedding = nn.Embedding(
        #     hp.n_symbols, hp.symbols_embedding_dim)
        # std = sqrt(2.0 / (hp.n_symbols + hp.symbols_embedding_dim))
        # val = sqrt(3.0) * std
        # self.embedding.weight.data.uniform_(-val, val)
        # self.encoder = Lip2WavEncoder()
        self.encoder = Lip2WavEncoder3D(hp).cuda()
        self.decoder = Lip2WavDecoder(hp).cuda()
        self.postnet = Lip2WavPostnet(hp).cuda()


    def parse_batch(self, batch):
        vid_padded, input_lengths, mel_padded, gate_padded, target_lengths, split_infos = batch
        vid_padded = to_var(vid_padded, self.hp).float()
        input_lengths = to_var(input_lengths, self.hp).float()
        mel_padded = to_var(mel_padded, self.hp).float()
        gate_padded = to_var(gate_padded, self.hp).float()
        target_lengths = to_var(target_lengths, self.hp).float()

        max_len_vid = split_infos[0].data.item()
        mel_padded = to_var(mel_padded, self.hp).float()

        return (
            (vid_padded, input_lengths, mel_padded, max_len_vid, target_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
            mask = mask.expand(self.num_mels, mask.size(0), mask.size(1))  # (80, B, T)
            mask = mask.permute(1, 0, 2)  # (B, 80, T)

            outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            slice = torch.arange(0, mask.size(2), self.n_frames_per_step)
            outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//n_frames_per_step)

        return outputs

    def forward(self, inputs):
        vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
        vid_lengths, output_lengths = vid_lengths.data, output_lengths.data

        embedded_inputs = vid_inputs.type(torch.FloatTensor)
        # print('vid_inputs',vid_inputs)

        encoder_outputs = self.encoder(embedded_inputs.cuda(), vid_lengths.cuda())
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=vid_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs, mode='train'):
        if mode == 'train':
            vid_inputs, vid_lengths, mels, max_len, output_lengths = inputs
        else:
            vid_inputs = inputs
            vid_inputs = to_var(torch.from_numpy(vid_inputs), self.hp).float()
            vid_inputs = vid_inputs.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        # vid_lengths, output_lengths = vid_lengths.data, output_lengths.data
        # embedded_inputs = self.embedding(inputs).transpose(1, 2)

        embedded_inputs = vid_inputs.type(torch.FloatTensor)

        encoder_outputs = self.encoder.inference(embedded_inputs.cuda())

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    # def teacher_infer(self, inputs, mels):
    #     il, _ = torch.sort(torch.LongTensor([len(x) for x in inputs]),
    #                        dim=0, descending=True)
    #     vid_lengths = to_var(il, self.hp)
    #
    #     embedded_inputs = self.embedding(inputs).transpose(1, 2)
    #
    #     encoder_outputs = self.encoder(embedded_inputs, vid_lengths)
    #
    #     mel_outputs, gate_outputs, alignments = self.decoder(
    #         encoder_outputs, mels, memory_lengths=vid_lengths)
    #
    #     mel_outputs_postnet = self.postnet(mel_outputs)
    #     mel_outputs_postnet = mel_outputs + mel_outputs_postnet
    #
    #     return self.parse_output(
    #         [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
