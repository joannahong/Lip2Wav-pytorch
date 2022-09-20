import torch
from torch import nn
from utils import to_var, get_mask_from_lengths
from Model.Modules.Encoders import Encoder2D
from Model.Modules.Decoders import Decoder
from Model.Modules.Postnets import Lip2WavPostnet

class BaseTacotron2(nn.Module):
    def __init__(self, hp):
        super(BaseTacotron2, self).__init__()
        self.hp = hp
        self.num_mels = hp.num_mels
        self.mask_padding = hp.mask_padding
        self.n_frames_per_step = hp.n_frames_per_step

        self.encoder = Encoder2D(hp).cuda()
        self.decoder = Decoder(hp).cuda()
        self.postnet = Lip2WavPostnet(hp).cuda()

    def parse_batch(self, batch):
        Landmark, Spec, input_lengths = batch

        landmark = to_var(Landmark, self.hp).float()
        spec = to_var(Spec, self.hp).float()
        video_length = to_var(input_lengths, self.hp).float()

        return ((landmark, video_length, spec),
                (spec))


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
        landmark, input_length, spec = inputs

        embedded_input = landmark.type(torch.FloatTensor)

        encoder_outputs = self.encoder(embedded_input.to(self.device))
        spec_outputs, alignments = self.decoder(encoder_outputs, spec)

        spec_outputs_postnet = self.postnet(spec_outputs)
        spec_outputs_postnet = spec_outputs + spec_outputs_postnet

        outputs = self.parse_output([spec_outputs, spec_outputs_postnet, alignments])
        return outputs

    def inference(self, inputs, mode='infer'):
        landmark = None
        if mode == 'infer':
            landmark = inputs
            landmark = to_var(torch.from_numpy(landmark), self.hp).float()
            landmark = landmark.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        elif mode == 'eval':
            landmark = inputs[0]

        embedded_inputs = landmark.type(torch.FloatTensor)

        encoder_outputs = self.encoder.inference(embedded_inputs.to(self.device))
        spec_outputs, alignments = self.decoder.inference(encoder_outputs)

        spec_outputs_postnet = self.postnet(spec_outputs)
        spec_outputs_postnet = spec_outputs + spec_outputs_postnet

        outputs = self.parse_output([spec_outputs, spec_outputs_postnet, alignments])
        return outputs