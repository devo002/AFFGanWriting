import torch
from torch import nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_max_len, vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size

    def forward(self, src, tar, src_len, teacher_rate, train=True, eos_id=None):
        """
        Args:
            src: (B, C, H, W)
            tar: (B, T) where tar[:,0] is <GO>. During inference we only use tar[:,0] to seed.
            src_len: tensor/list of input widths per sample
            teacher_rate: float in [0,1]; used only when train=True
            train: bool; if False, uses greedy decoding (argmax) with self-feedback
            eos_id: optional int token id to early-stop beams when all hit EOS
        Returns:
            outputs: (T-1, B, V) — distributions for each step (not including <GO>)
            attns: list length <= T-1 with attention tensors of shape (B, enc_T)
        """
        # Shapes / devices
        device = src.device
        B = src.size(0)

        # (T, B)
        tar = tar.permute(1, 0)

        # Encoder
        enc_out, enc_hidden = self.encoder(src, src_len)   # enc_out: (enc_T, B, F), enc_hidden: (layers, B, H)

        # Pre-allocate outputs
        outputs = torch.zeros(self.output_max_len - 1, B, self.vocab_size, device=device)

        # Seed decoder input with <GO> one-hot
        go_tokens = tar[0].detach()                        # (B,)
        dec_input = self.one_hot(go_tokens, device)        # (B, V)

        hidden = enc_hidden
        attns = []

        # For attention init (shape expected by your decoder code)
        attn_weights = torch.zeros(enc_out.shape[1], enc_out.shape[0], device=device)  # (B, enc_T)

        # Keep track of finished sequences when eos_id is provided (inference only)
        finished = torch.zeros(B, dtype=torch.bool, device=device) if (not train and eos_id is not None) else None

        for t in range(self.output_max_len - 1):
            # Run one decoding step
            step_out, hidden, attn_weights = self.decoder(dec_input, hidden, enc_out, src_len, attn_weights)
            # step_out is expected to be a prob distribution over vocab (B, V)
            outputs[t] = step_out
            attns.append(attn_weights.detach().to('cpu'))

            if train:
                # Scheduled teacher forcing
                use_teacher = random.random() < teacher_rate
                next_input_tokens = tar[t + 1].detach() if use_teacher else step_out.argmax(dim=1)
                dec_input = self.one_hot(next_input_tokens, device)
            else:
                # --- GREEDY: feed argmax back as next input ---
                next_input_tokens = step_out.argmax(dim=1)               # (B,)
                dec_input = self.one_hot(next_input_tokens, device)      # (B, V)

                # Early stop handling (optional)
                if finished is not None:
                    finished |= (next_input_tokens == eos_id)
                    if torch.all(finished):
                        # We can break early; remaining timesteps stay zero in `outputs`
                        break

        return outputs, attns

    def one_hot(self, ids, device):
        """ids: (B,) int64 -> (B, V) float one-hot"""
        eye = torch.eye(self.vocab_size, device=device)
        return eye.index_select(0, ids)
