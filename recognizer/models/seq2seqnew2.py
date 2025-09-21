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

    def forward(
        self,
        src,                # (B, C, H, W)
        tar,                # (B, T) with tar[:,0] == <GO>
        src_len,            # tensor/list/np of input widths per sample
        teacher_rate,       # float [0,1] (only used when train=True)
        train=True,         # False => inference
        eos_id=None,        # optional EOS token id
        beam_size=1         # 1 => greedy; >1 => beam search
    ):
        device = src.device
        B = src.size(0)
        max_steps = self.output_max_len - 1

        # (T, B)
        tar_TB = tar.permute(1, 0)

        # Encode once
        enc_out, enc_hidden = self.encoder(src, src_len)   # enc_out: (enc_T, B, F)

        # ----------------------------
        # Training or Greedy Inference
        # ----------------------------
        if train or beam_size == 1:
            outputs = torch.zeros(max_steps, B, self.vocab_size, device=device)

            # seed with <GO>
            go_tokens = tar_TB[0].detach()                 # (B,)
            dec_input = self.one_hot(go_tokens, device)    # (B, V)

            hidden = enc_hidden
            attns = []
            attn_weights = torch.zeros(enc_out.shape[1], enc_out.shape[0], device=device)  # (B, enc_T)

            for t in range(max_steps):
                step_out, hidden, attn_weights = self.decoder(dec_input, hidden, enc_out, src_len, attn_weights)
                outputs[t] = step_out
                attns.append(attn_weights.detach().to('cpu'))

                if train:
                    use_teacher = random.random() < teacher_rate
                    next_ids = tar_TB[t + 1].detach() if use_teacher else step_out.argmax(dim=1)
                    dec_input = self.one_hot(next_ids, device)
                else:
                    next_ids = step_out.argmax(dim=1)
                    dec_input = self.one_hot(next_ids, device)
                    if eos_id is not None and torch.all(next_ids == eos_id):
                        break

            return outputs, attns

        # ----------------------------
        # Beam Search Inference (beam_size > 1)
        # ----------------------------
        enc_T, _, F = enc_out.shape
        best_outputs = torch.zeros(max_steps, B, self.vocab_size, device=device)

        # We'll also return per-step attention maps across batch (optional).
        # Build them after decoding by padding to the longest produced sequence.
        per_sample_attn_traces = [[] for _ in range(B)]

        # Decode each sample independently (simple and robust with your decoder signature).
        for b in range(B):
            # Slice encoder outputs/hidden for this sample
            enc_out_b = enc_out[:, b:b+1, :]  # (enc_T, 1, F)
            if isinstance(enc_hidden, tuple):  # LSTM
                hidden_b = tuple(h[:, b:b+1, :].contiguous() for h in enc_hidden)  # ((L,1,H), (L,1,H))
            else:                               # GRU
                hidden_b = enc_hidden[:, b:b+1, :].contiguous()                    # (L,1,H)

            # Initial attention weights for this sample: (1, enc_T)
            attn_b = torch.zeros(1, enc_T, device=device)

            # <GO> token from provided targets
            go_id = int(tar[b, 0].item())

            # Beam item structure
            # - logp: cumulative log-prob
            # - tokens: list of token ids including <GO> and new tokens
            # - hidden: decoder hidden state
            # - attn: last attention weights (1, enc_T)
            # - dists: list of per-step distributions (each (V,))
            # - attn_steps: list of per-step attention (each (enc_T,))
            beams = [{
                "logp": 0.0,
                "tokens": [go_id],
                "hidden": hidden_b,
                "attn": attn_b,
                "dists": [],
                "attn_steps": []
            }]

            for t in range(max_steps):
                new_beams = []

                for beam in beams:
                    last_token = beam["tokens"][-1]

                    # If already ended with EOS, just carry forward unchanged
                    if eos_id is not None and last_token == eos_id:
                        new_beams.append(beam)
                        continue

                    # Prepare decoder input one-hot for this hypothesis: (1, V)
                    dec_in = self.one_hot(torch.tensor([last_token], device=device), device)

                    # ---- src_len fix: provide a 1-element **CPU tensor** ----
                    if isinstance(src_len, torch.Tensor):
                        src_len_b = src_len[b:b+1].detach().cpu()
                    elif isinstance(src_len, (list, tuple)):
                        src_len_b = torch.tensor([src_len[b]], dtype=torch.long)
                    else:
                        src_len_b = torch.as_tensor([src_len[b]], dtype=torch.long)

                    # One decoding step for this hypothesis
                    step_out, new_hidden, new_attn = self.decoder(
                        dec_in, beam["hidden"], enc_out_b, src_len_b, beam["attn"]
                    )  # step_out: (1, V), new_attn: (1, enc_T)

                    # Convert to log-probs (guard for numerical safety)
                    log_probs = torch.log(step_out + 1e-12).squeeze(0)  # (V,)

                    # Expand with top-K tokens
                    topk_logp, topk_ids = torch.topk(log_probs, k=beam_size, dim=-1)  # (beam_size,)
                    for k in range(beam_size):
                        nid = int(topk_ids[k].item())
                        nlogp = beam["logp"] + float(topk_logp[k].item())
                        new_beams.append({
                            "logp": nlogp,
                            "tokens": beam["tokens"] + [nid],
                            "hidden": new_hidden,
                            "attn": new_attn,
                            "dists": beam["dists"] + [step_out.squeeze(0)],                # (V,)
                            "attn_steps": beam["attn_steps"] + [new_attn.squeeze(0)]       # (enc_T,)
                        })

                # Keep top beam_size by cumulative log-prob
                new_beams.sort(key=lambda x: x["logp"], reverse=True)
                beams = new_beams[:beam_size]

                # Early stop if all beams ended with EOS
                if eos_id is not None and all(bm["tokens"][-1] == eos_id for bm in beams):
                    break

            # Choose the best beam
            best = max(beams, key=lambda x: x["logp"])
            steps_to_fill = min(len(best["dists"]), max_steps)

            if steps_to_fill > 0:
                d_stack = torch.stack(best["dists"][:steps_to_fill], dim=0)         # (steps, V)
                best_outputs[:steps_to_fill, b, :] = d_stack.to(device)
                per_sample_attn_traces[b] = best["attn_steps"][:steps_to_fill]      # list of (enc_T,)

        # Build `attns` as a list of length = max produced steps, each (B, enc_T) on CPU
        attns = []
        produced_steps = 0
        for t in range(max_steps - 1, -1, -1):
            if best_outputs[t].abs().sum() > 0:
                produced_steps = t + 1
                break
        for t in range(produced_steps):
            # gather per-sample attentions for this step (pad zeros if a sample is shorter)
            step_attn = torch.zeros(B, enc_T)
            for b in range(B):
                if t < len(per_sample_attn_traces[b]):
                    step_attn[b] = per_sample_attn_traces[b][t].detach().cpu()
            attns.append(step_attn)

        return best_outputs, attns

    def one_hot(self, ids, device):
        """ids: (N,) int64 -> (N, V) float one-hot on `device`"""
        eye = torch.eye(self.vocab_size, device=device)
        return eye.index_select(0, ids)
