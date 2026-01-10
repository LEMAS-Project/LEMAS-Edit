import random

import numpy as np
import logging
import argparse, copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from .modules.utils import make_pad_mask

from .modules.embedding import SinePositionalEmbedding, TokenEmbedding
from .modules.transformer import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .codebooks_patterns import DelayedPatternProvider

from argparse import Namespace
from huggingface_hub import PyTorchModelHubMixin


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, prev_tokens=None, filter_value=-float("Inf"), min_tokens_to_keep=1, repetition_penalty=0, num_gen=None,
):
    if prev_tokens:
        prev_tokens = torch.stack(prev_tokens).transpose(1, 0)
    if len(prev_tokens) > 0 and repetition_penalty > 0:
        penalty = (repetition_penalty / 100) * num_gen + 1
        score = torch.gather(logits, dim=1, index=prev_tokens)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(dim=1, index=prev_tokens, src=score)

    if top_k > 0:
        top_k = min(
            max(int(top_k), min_tokens_to_keep), logits.size(-1)
        )
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0, prev_tokens=None, repetition_penalty=1.0, num_gen=None):
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filtering(
        logits,
        top_k=top_k,
        top_p=top_p,
        prev_tokens=prev_tokens or [],
        repetition_penalty=repetition_penalty,
        num_gen=num_gen,
    )
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


class LemasEditCodec(
        nn.Module,
        PyTorchModelHubMixin,
        library_name="lemas_edit_codec",
        repo_url="",
        tags=["text-to-speech"],
    ):
    def __new__(cls, args: Optional[Namespace] = None, config: Optional[Dict] = None, **kwargs) -> "LemasEditCodec":
        if args is not None:
            if config is not None:
                raise ValueError("Cannot provide both `args` and `config`.")
            config = vars(args)
        return super().__new__(cls, args=args, config=config, **kwargs)

    def __init__(self, args: Optional[Namespace] = None, config: Optional[Dict] = None):
        super().__init__()

        if args is None:
            if config is None:
                raise ValueError("Either `args` or `config` must be provided.")
            args = Namespace(**config)

        # Keep field names exactly as in the original autoregressive checkpoints
        # (e.g., nhead, num_decoder_layers, trm_dropout) to avoid any
        # renaming or implicit mapping.
        self.args = copy.copy(args)
        self.pattern = DelayedPatternProvider(n_q=self.args.n_codebooks)
        if not getattr(self.args, "special_first", False):
            self.args.special_first = 0
        if not getattr(self.args, "n_special", False):
            self.args.n_special = 3
        self.args.eos = getattr(self.args, "eos", -1)
        self.eog = nn.Parameter(torch.full((self.args.n_codebooks, 1), self.args.eog, dtype=torch.long), requires_grad=False)  # [K 1]
        if self.args.eos > 0:
            assert self.args.eos != self.args.audio_pad_token and self.args.eos != self.args.empty_token, self.args.eos
            self.eos = nn.Parameter(torch.full((self.args.n_codebooks, 1), self.args.eos, dtype=torch.long), requires_grad=False)  # [K 1]
        if isinstance(self.args.audio_vocab_size, str):
            self.args.audio_vocab_size = eval(self.args.audio_vocab_size)

        self.n_text_tokens = self.args.text_vocab_size

        self.n_audio_tokens = [self.args.audio_vocab_size + self.args.n_special] * self.args.n_codebooks
        assert self.args.audio_vocab_size == self.args.empty_token, self.args.empty_token
        assert self.args.eog == self.args.audio_vocab_size + 1, self.args.eog
        assert self.args.audio_pad_token == self.args.audio_vocab_size + 2, self.args.audio_pad_token

        self.text_embedding = TokenEmbedding(
            dim_model=self.args.d_model,
            vocab_size=self.n_text_tokens,
            dropout=self.args.text_embedding_dropout,
        )

        self.audio_embedding = nn.ModuleList(
            [
                TokenEmbedding(
                    dim_model=self.args.audio_embedding_dim,
                    vocab_size=self.n_audio_tokens[k],
                    dropout=self.args.audio_embedding_dropout,
                )
                for k in range(self.args.n_codebooks)
            ]
        )
        self.mask_embedding = nn.Parameter(
            torch.randn(self.args.max_n_spans, self.args.d_model),
            requires_grad=True,
        )
        self.text_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.text_positional_embedding_dropout,
            scale=False,
            alpha=True,  # learnable scaler, same as original implementation
        )
        self.audio_positional_embedding = SinePositionalEmbedding(
            self.args.d_model,
            dropout=self.args.audio_positional_embedding_dropout,
            scale=False,
            alpha=True,
        )

        if getattr(self.args, "normalize_text_embedding", False):
            self.text_layer_norm = LayerNorm(
                self.args.d_model, eps=self.args.layer_norm_eps
            )
        if getattr(self.args, "normalize_audio_embedding", False):
            self.audio_layer_norm = LayerNorm(
                self.args.d_model, eps=self.args.layer_norm_eps
            )

        # Decoder stack: strictly follow the original autoregressive codec
        # architecture so that state_dict keys match the checkpoint
        # (decoder.layers.* and predict_layer.*).
        dec_layer = TransformerEncoderLayer(
            self.args.d_model,
            self.args.nhead,
            dim_feedforward=self.args.d_model * 4,
            dropout=self.args.trm_dropout,
            batch_first=True,
            norm_first=True,
            layer_norm_cls=LayerNorm,
        )
        self.decoder = TransformerEncoder(
            dec_layer,
            num_layers=self.args.num_decoder_layers,
            norm=LayerNorm(self.args.d_model),
        )

        self.predict_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.args.d_model, self.args.audio_vocab_size // 2),
                    nn.GELU(),
                    nn.Linear(
                        self.args.audio_vocab_size // 2, self.n_audio_tokens[k]
                    ),
                )
                for k in range(self.args.n_codebooks)
            ]
        )

        self.accuracy_metrics = nn.ModuleList(
            [
                MulticlassAccuracy(
                    self.n_audio_tokens[k],
                    top_k=10,
                    average="micro",
                    multidim_average="global",
                    ignore_index=None,
                )
                for k in range(self.args.n_codebooks)
            ]
        )

    def prepare_mask_intervals_new(self, y_lens, n_spans=1):
        mask_intervals = []
        non_mask_intervals = []
        for i, y_len in enumerate(y_lens):
            starts = random.sample(range(1, y_len - 1 - self.args.mask_len_min), n_spans)
            starts = sorted(starts)
            ends = []
            for j, start in enumerate(starts):
                mask_len = min(
                    random.randint(self.args.mask_len_min, self.args.mask_len_max),
                    int(y_len / 1.5),
                )
            ends.append(min(start + mask_len, y_len))
            mask_intervals.append([(s, e) for s, e in zip(starts, ends)])
            non_mask_intervals.append(
                [(ns, ne) for ns, ne in zip([0] + ends, starts + [y_len])]
            )
        return mask_intervals, non_mask_intervals

    def prepare_mask_intervals(self, y_lens):
        mask_intervals = []
        non_mask_intervals = []

        for i, y_len in enumerate(y_lens):
            if self.args.mask_sample_dist == "uniform":
                n_spans = random.choice(range(1, self.args.max_n_spans + 1))
            elif "poisson" in self.args.mask_sample_dist.lower():
                param = float(self.args.mask_sample_dist[len("poisson") :])
                poisson_sample = torch.poisson(torch.tensor([param]))
                n_spans = int(
                    poisson_sample.clamp(1, self.args.max_n_spans).item()
                )

            starts = random.sample(
                range(1, y_len - 1 - self.args.mask_len_min), n_spans
            )
            starts = sorted(starts)

            for j in range(len(starts) - 1, 0, -1):
                if starts[j] - starts[j - 1] < self.args.min_gap:
                    # If elements are too close, delete the later one
                    del starts[j]
            assert (
                len(starts) > 0
            ), f"there is no masked span left, y_len: {y_len}, sampled n_spans: {n_spans}"

            temp_starts = starts + [y_len]
            gaps = [
                temp_starts[j + 1] - temp_starts[j]
                for j in range(len(temp_starts) - 1)
            ]

            ends = []

            for j, (start, gap) in enumerate(zip(starts, gaps)):
                mask_len = random.randint(
                    self.args.mask_len_min, self.args.mask_len_max
                )
                # make sure the masks are not overlapping with each other
                if mask_len > gap - 1:
                    temp_mask_start = 1
                    temp_mask_end = gap - 1
                    mask_len = random.randint(temp_mask_start, temp_mask_end)
                ends.append(start + mask_len)

            mask_intervals.append([(s, e) for s, e in zip(starts, ends)])
            non_mask_intervals.append(
                [(ns, ne) for ns, ne in zip([0] + ends, starts + [y_len])]
            )

        return mask_intervals, non_mask_intervals

    def rearrange(self, y, non_mask_intervals, mask_intervals):
        reduced_eog = getattr(self.args, "reduced_eog", 0)
        rearranged_y = []
        for i in range(len(y)):
            if self.args.eos > 0:
                assert reduced_eog
                cur_y = (
                    [
                        y[i, :, item[0] : item[1]]
                        for item in non_mask_intervals[i][:-1]
                    ]
                    + [
                        torch.cat(
                            [
                                y[
                                    i,
                                    :,
                                    non_mask_intervals[i][-1][0] : non_mask_intervals[
                                        i
                                    ][-1][1],
                                ],
                                self.eos,
                            ],
                            dim=-1,
                        )
                    ]
                    + [
                        torch.cat(
                            [y[i, :, item[0] : item[1]], self.eog], dim=-1
                        )
                        for item in mask_intervals[i]
                    ]
                )
            else:
                if reduced_eog:
                    cur_y = (
                        [
                            y[i, :, item[0] : item[1]]
                            for item in non_mask_intervals[i][:-1]
                        ]
                        + [
                            torch.cat(
                                [
                                    y[
                                        i,
                                        :,
                                        non_mask_intervals[i][-1][0] : non_mask_intervals[
                                            i
                                        ][-1][1],
                                    ],
                                    self.eog,
                                ],
                                dim=-1,
                            )
                        ]
                        + [
                            torch.cat(
                                [y[i, :, item[0] : item[1]], self.eog], dim=-1
                            )
                            for item in mask_intervals[i]
                        ]
                    )
                else:
                    cur_y = [
                        torch.cat(
                            [y[i, :, item[0] : item[1]], self.eog], dim=-1
                        )
                        for item in non_mask_intervals[i]
                    ] + [
                        torch.cat(
                            [y[i, :, item[0] : item[1]], self.eog], dim=-1
                        )
                        for item in mask_intervals[i]
                    ]
            rearranged_y.append(cur_y)
        return rearranged_y

    def shift(self, rearranged_y):
        shifted_y = []
        patterns = []
        for i in range(len(rearranged_y)):
            cur_patterns = [
                self.pattern.get_pattern(cur_y.shape[1])
                for cur_y in rearranged_y[i]
            ]
            out = [
                cur_pattern.build_pattern_sequence(
                    z=cur_y.unsqueeze(0).contiguous(),
                    special_token=self.args.empty_token,
                    keep_only_valid_steps=False,
                )
                for cur_pattern, cur_y in zip(cur_patterns, rearranged_y[i])
            ]
            shifted_y.append(
                [item[0].squeeze(0) for item in out]
            )  # values; later items are indexes and mask
            patterns.append(cur_patterns)
        return shifted_y, patterns

    def insert_mask(self, shifted_y):
        inserted_y = []
        mask_position = []
        mask_value = []
        for i in range(len(shifted_y)):
            num_masks = (len(shifted_y[i]) - 1) // 2
            assert num_masks == (len(shifted_y[i]) - 1) / 2, len(shifted_y[i])
            emb_inds = list(range(self.args.max_n_spans))
            if self.args.shuffle_mask_embedding:
                random.shuffle(emb_inds)
            emb_inds_use = emb_inds[:num_masks]
            emb_inds_use = emb_inds_use + emb_inds_use
            mask_value.append(emb_inds_use)
            cur_inserted_y = []
            cur_mask_position = []
            for j in range(len(shifted_y[i]) - 1):
                cur_inserted_y.append(shifted_y[i][j])
                cur_mask_position.append(
                    sum([item.shape[1] for item in cur_inserted_y])
                )  # each item is [K S]
                cur_inserted_y.append(
                    self.eog
                )  # placeholder; real mask is inserted in embed_y

            cur_inserted_y.append(shifted_y[i][-1])

            inserted_y.append(cur_inserted_y)
            mask_position.append(cur_mask_position)
        return inserted_y, mask_position, mask_value

    def cat_y(self, inserted_y, mask_position, y_lens):
        reduced_eog = getattr(self.args, "reduced_eog", 0)
        cated_y = []
        new_y_lens = []
        for i in range(len(inserted_y)):
            cur_cated_y = torch.cat(inserted_y[i], dim=1)  # [K S]
            cur_cated_y = cur_cated_y.transpose(1, 0)  # [S K]
            cur_cated_y_len = cur_cated_y.shape[0]
            if reduced_eog:
                assert (
                    cur_cated_y_len
                    == y_lens[i]
                    + len(mask_position[i])
                    + (len(mask_position[i]) + 1) * self.args.n_codebooks
                    + (len(mask_position[i]) / 2 + 1)
                ), (
                    f"cur_cated_y_len == {cur_cated_y_len}, but it should be "
                    f"y_lens[i] ({y_lens[i]}) + len(mask_position[i]) ({len(mask_position[i])}) "
                    f"+ (len(mask_position[i]) + 1) * self.args.n_codebooks "
                    f"({(len(mask_position[i]) + 1) * self.args.n_codebooks}) "
                    f"+ (len(mask_position[i])/2 + 1) ({len(mask_position[i])/2 + 1})="
                    f"{y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i])/2 + 1)}"
                )
            else:
                assert (
                    cur_cated_y_len
                    == y_lens[i]
                    + len(mask_position[i])
                    + (len(mask_position[i]) + 1) * self.args.n_codebooks
                    + (len(mask_position[i]) + 1)
                ), (
                    f"cur_cated_y_len == {cur_cated_y_len}, but it should be "
                    f"y_lens[i] ({y_lens[i]}) + len(mask_position[i]) ({len(mask_position[i])}) "
                    f"+ (len(mask_position[i]) + 1) * self.args.n_codebooks "
                    f"({(len(mask_position[i]) + 1) * self.args.n_codebooks}) "
                    f"+ (len(mask_position[i]) + 1) ({len(mask_position[i]) + 1})="
                    f"{y_lens[i] + len(mask_position[i]) + (len(mask_position[i]) + 1) * self.args.n_codebooks + (len(mask_position[i]) + 1)}"
                )

            cated_y.append(cur_cated_y)
            new_y_lens.append(cur_cated_y_len)

        cated_y = torch.nn.utils.rnn.pad_sequence(
            cated_y,
            batch_first=False,
        ).to(inserted_y[0][0].device)  # [S, B, K]
        # Rearrange to [K, S, B] so that dimension 0 matches n_codebooks.
        cated_y = cated_y.permute(2, 0, 1)  # [K, S, B]
        # Keep new_y_lens as a tensor like the original implementation so that
        # downstream code (make_pad_mask, dec_forward, etc.) can call .max().
        new_y_lens = torch.LongTensor(new_y_lens).to(cated_y.device)
        return cated_y, new_y_lens

    def embed_y(self, cated_y, mask_position, mask_value):
        # cated_y: [K, S, B]
        embedded_y = torch.stack(
            [self.audio_embedding[k](cated_y[k]) for k in range(self.args.n_codebooks)],
            dim=0,
        )  # [K, S, B, D]
        assert embedded_y.shape[0] == self.args.n_codebooks, embedded_y.shape
        assert embedded_y.shape[-1] == self.args.d_model, embedded_y.shape
        embedded_y = embedded_y.sum(dim=0)  # [K,S,B,D]->[S,B,D]
        embedded_y = embedded_y.transpose(1, 0)  # [S,B,D]->[B,S,D]
        for i in range(len(embedded_y)):
            if len(mask_position[i]) > 0:
                embedded_y[i, mask_position[i]] = self.mask_embedding[mask_value[i]]
        return embedded_y

    def prepare_input_target(self, y, y_lens):
        # rearrange y
        # assume y shape: [B T K], K is n_codebooks
        assert y.shape[1] == self.args.n_codebooks, y.shape
        # sample mask_intervals
        mask_intervals, non_mask_intervals = self.prepare_mask_intervals(y_lens)

        # rearrange y into masked/non-masked segments
        rearranged_y = self.rearrange(y, non_mask_intervals, mask_intervals)
        targets = rearranged_y  # each element in each sample is of shape [K T]
        assert targets[0][0].shape[0] == self.args.n_codebooks, targets[0][0].shape

        # shift with delayed patterns
        shifted_y, patterns = self.shift(rearranged_y)  # each element [K S]
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks

        # insert mask tokens and record their positions
        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)
        assert inserted_y[0][0].shape[0] == self.args.n_codebooks
        assert inserted_y[0][1].shape == torch.Size(
            (self.args.n_codebooks, 1)
        ), f"mask tensor should be {(self.args.n_codebooks, 1)}, got {inserted_y[0][1].shape}"

        # concat segments and pad across batch
        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens)  # KTB
        assert cated_y.shape[0] == self.args.n_codebooks

        # embed y and add positional encoding
        embedded_y = self.embed_y(cated_y, mask_position, mask_value)  # BTD
        assert embedded_y.shape[1:] == torch.Size(
            (max(new_y_lens), self.args.d_model)
        )

        y_input = self.audio_positional_embedding(embedded_y)

        # make attention mask and padding mask
        y_padding_mask = make_pad_mask(new_y_lens).to(y.device)
        y_attention_mask = torch.triu(
            torch.ones(y_input.shape[1], y_input.shape[1]),
            diagonal=1,
        ).bool().to(y_padding_mask.device)
        return (
            y_input,
            new_y_lens,
            targets,
            y_padding_mask,
            y_attention_mask,
            mask_position,
            patterns,
        )

    def remove_mask(self, logits, mask_position, new_y_lens):
        # logits: [B K S card]
        logits_use = []
        for i in range(len(logits)):
            non_mask_positions = [-1] + mask_position[i] + [new_y_lens[i]]
            non_mask_intervals = [
                [non_mask_positions[i] + 1, non_mask_positions[i + 1]]
                for i in range(len(non_mask_positions) - 1)
            ]
            cur_logits_use = [
                logits[i, :, l:r] for l, r in non_mask_intervals
            ]
            logits_use.append(cur_logits_use)

        return logits_use

    def revert_pattern(self, patterns, logits_use):
        logits_final = []
        logit_masks = []
        for i in range(len(logits_use)):
            cur_logits = [
                item.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
                for item in logits_use[i]
            ]  # [1 card K S]
            cur_logits_final = [
                cur_pattern.revert_pattern_logits(
                    item, 0, keep_only_valid_steps=False
                )
                for cur_pattern, item in zip(patterns[i], cur_logits)
            ]
            cur_logits_final_ret = [
                item[0].permute(0, 2, 3, 1).squeeze(0)
                for item in cur_logits_final
            ]  # [K,T,card]
            logits_final.append(cur_logits_final_ret)
            logit_masks.append([item[2] for item in cur_logits_final])

        return logits_final, logit_masks

    def dec_forward(
        self,
        x_input,
        x_lens,
        x_attention_mask,
        x_padding_mask,
        y_input,
        new_y_lens,
        y_attention_mask,
        y_padding_mask,
        past=None,
        last_3_tokens=False,
    ):
        x_attn_mask = F.pad(
            x_attention_mask,
            (0, new_y_lens.max()),
            value=True,
        )
        y_attn_mask = F.pad(
            y_attention_mask,
            (x_lens.max(), 0),
            value=False,
        )
        xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

        bsz, src_len = x_input.shape[0], x_lens.max() + new_y_lens.max()
        xy_padding_mask = torch.concat([x_padding_mask, y_padding_mask], dim=1)
        _xy_padding_mask = (
            xy_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, self.args.nhead, -1, -1)
            .reshape(bsz * self.args.nhead, 1, src_len)
        )
        if xy_attn_mask.shape != _xy_padding_mask.shape:
            assert (
                xy_attn_mask.ndim + 1 == _xy_padding_mask.ndim
            ), f"xy_attn_mask.shape: {xy_attn_mask.shape}, _xy_padding_mask: {_xy_padding_mask.shape}"
            xy_attn_mask = xy_attn_mask.unsqueeze(0).repeat(
                _xy_padding_mask.shape[0], 1, 1
            )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

        new_attn_mask = torch.zeros_like(xy_attn_mask)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask

        xy_input = torch.cat([x_input, y_input], dim=1)

        if past is None:
            out, _ = self.decoder((xy_input, None), mask=xy_attn_mask)
            return out[:, x_lens.max() :], None
        else:
            if past.ndim > 3:
                if last_3_tokens:
                    xy_input = xy_input[:, -3:]
                    xy_attn_mask = xy_attn_mask[:, -3:]
                else:
                    xy_input = xy_input[:, -1:]
                    xy_attn_mask = xy_attn_mask[:, -1:]

            out, present = self.decoder((xy_input, None), mask=xy_attn_mask, past=past)
            if isinstance(out, tuple):
                out = out[0]

            if out.shape[1] > x_lens.max():
                return out[:, x_lens.max() :], present
            else:
                return out, present

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        mask_interval: torch.Tensor,
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stop_repetition: int = -1,
        kvcache: int = 1,
        silence_tokens: list[int] = [1388, 1898, 131],
        max_rate: int = 5,
        repetition_penalty: float = 1.0,
    ):
        self.RE_GEN = False
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1)  # [1,T,K] -> [1,K,T]
        assert (
            y.shape[0] == 1 and y.shape[1] == self.args.n_codebooks
        ), y.shape
        assert mask_interval.shape == torch.Size(
            (1, mask_interval.shape[1], 2)
        ), mask_interval

        x_attention_mask = torch.triu(
            torch.ones(x.shape[1], x.shape[1]), diagonal=1
        ).bool().to(x.device)
        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)

        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)

        mask_interval = mask_interval[0]
        starts = [item[0].item() for item in mask_interval] + [y_len]
        ends = [0] + [item[1].item() for item in mask_interval]
        mask_intervals = [
            [(item[0].item(), item[1].item()) for item in mask_interval]
        ]
        non_mask_intervals = [[(ns, ne) for ns, ne in zip(ends, starts)]]

        rearranged_y = self.rearrange(y, non_mask_intervals, mask_intervals)
        assert rearranged_y[0][0].shape[0] == self.args.n_codebooks

        shifted_y, patterns = self.shift(rearranged_y)
        assert shifted_y[0][0].shape[0] == self.args.n_codebooks

        inserted_y, mask_position, mask_value = self.insert_mask(shifted_y)
        assert inserted_y[0][0].shape[0] == self.args.n_codebooks

        cated_y, new_y_lens = self.cat_y(inserted_y, mask_position, y_lens)

        num_mask = len(mask_position[0]) // 2
        assert num_mask == len(mask_position[0]) / 2
        cated_y = cated_y[:, : mask_position[0][num_mask] + 2]
        more_mask_value = mask_value[0][num_mask + 1 :]
        new_y_lens[0] = mask_position[0][num_mask] + 2
        mask_position[0] = mask_position[0][: num_mask + 1]

        embedded_y = self.embed_y(
            cated_y, mask_position, [mask_value[0][: num_mask + 1]]
        )

        y_input = self.audio_positional_embedding(embedded_y)

        y_attention_mask = torch.triu(
            torch.ones(y_input.shape[1], y_input.shape[1]), diagonal=1
        ).bool().to(y.device)

        x_padding_mask = torch.full((1, x_lens[0]), False).to(x.device)
        y_padding_mask = torch.full((1, new_y_lens[0]), False).to(y.device)

        codebook_eog = [False] * self.args.n_codebooks
        generated = []
        cur_generated = []
        num_gen = []
        cur_num_gen = 0

        consec_silence_count = 0
        prev_tokens = []

        past = (
            torch.ones(
                [self.args.num_decoder_layers, 2, x.shape[0]],
                device=x.device,
                dtype=torch.float32,
            )
            if kvcache
            else None
        )
        new_masked_span = False

        def sample_helper(
            n_eog,
            logits,
            codebook_eog,
            top_k,
            top_p,
            temperature,
            prev_tokens,
            consec_silence_count,
            stop_repetition,
            silence_tokens,
            cur_num_gen,
        ):
            # Direct port of the original sampling helper used for edit-mode
            # inference, including silence-repetition handling and staged EOG.
            if n_eog == 0:
                logits_adjust = logits
                for jj in range(1, self.args.n_codebooks):
                    logits_adjust[jj][self.args.eog] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                # silence repetition handling
                prev_token = prev_tokens[-1][0] if prev_tokens else None
                if (
                    stop_repetition > 0
                    and prev_token in silence_tokens
                    and consec_silence_count > stop_repetition
                ):
                    if logits_adjust[0, prev_token] < 0:
                        logits_adjust[0, prev_token] = (
                            logits_adjust[0, prev_token]
                            * (consec_silence_count - (stop_repetition - 1))
                        )
                    else:
                        logits_adjust[0, prev_token] = (
                            logits_adjust[0, prev_token]
                            / (consec_silence_count - (stop_repetition - 1))
                        )
                if isinstance(logits_adjust, list):
                    samples_list = []
                    for logit in logits_adjust:
                        cur_sample = topk_sampling(
                            logit.unsqueeze(0),
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            prev_tokens=prev_tokens,
                            repetition_penalty=repetition_penalty,
                            num_gen=cur_num_gen,
                        )
                        samples_list.append(cur_sample)
                    samples = torch.cat(samples_list, dim=0)
                else:
                    samples = topk_sampling(
                        logits_adjust,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        prev_tokens=prev_tokens,
                        repetition_penalty=repetition_penalty,
                        num_gen=cur_num_gen,
                    )
                assert samples.shape == torch.Size(
                    (self.args.n_codebooks, 1)
                ), f"samples.shape: {samples.shape}"
                if cur_num_gen < self.args.n_codebooks - 1:
                    for jj in range(1, self.args.n_codebooks - cur_num_gen):
                        samples[-jj, 0] = self.args.empty_token

                # EOG and length check
                if (
                    samples[0, 0] == self.args.eog
                    or torch.argmax(logits[0], dim=-1) == self.args.eog
                    or y_input.shape[1] > x_lens[0] * max_rate
                ):
                    samples[0, 0] = self.args.eog
                    codebook_eog[0] = True
                    if y_input.shape[1] > x_lens[0] * max_rate:
                        self.RE_GEN = True
                        logging.warning(
                            f"Reaching maximum length, this result may not correct! Set RE_GEN={self.RE_GEN}"
                        )

                # silence repetition bookkeeping
                if samples[0, 0] in silence_tokens and samples[0, 0] == prev_token:
                    consec_silence_count += 1
                else:
                    consec_silence_count = 0

                prev_tokens.append(samples.squeeze())
                prev_tokens = prev_tokens[-100:]
                return (
                    samples,
                    codebook_eog,
                    prev_tokens,
                    consec_silence_count,
                )
            else:
                assert (
                    sum(codebook_eog[i] for i in range(n_eog)) == n_eog
                ), f"codebook_eog: {codebook_eog}, but n_eog: {n_eog}"
                logits_adjust = logits
                for jj in range(n_eog + 1, self.args.n_codebooks):
                    logits_adjust[jj][self.args.eog] = -10000
                    logits_adjust[jj][self.args.empty_token] = -10000
                if isinstance(logits_adjust, list):
                    samples_list = []
                    for logit in logits_adjust:
                        cur_sample = topk_sampling(
                            logit.unsqueeze(0),
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                        )
                        samples_list.append(cur_sample)
                    samples = torch.cat(samples_list, dim=0)
                else:
                    samples = topk_sampling(
                        logits_adjust,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                assert samples.shape == torch.Size(
                    (self.args.n_codebooks, 1)
                ), f"samples.shape: {samples.shape}"
                for jj in range(n_eog):
                    samples[jj, 0] = self.args.empty_token
                samples[n_eog, 0] = self.args.eog
                codebook_eog[n_eog] = True
                return (
                    samples,
                    codebook_eog,
                    prev_tokens,
                    consec_silence_count,
                )

        # Main autoregressive decoding loop (ported from the original edit
        # inference implementation).
        while True:
            y_out, present = self.dec_forward(
                x_input,
                x_lens,
                x_attention_mask,
                x_padding_mask,
                y_input,
                new_y_lens,
                y_attention_mask,
                y_padding_mask,
                past=past,
                last_3_tokens=new_masked_span,
            )
            if new_masked_span:
                new_masked_span = False

            if past is not None:
                past = (
                    torch.cat([past, present.to(past.dtype)], dim=-2)
                    if past.ndim > 3
                    else present.to(past.dtype)
                )

            # Only keep the last step for sampling.
            y_out = y_out[:, -1:]

            logits = torch.stack(
                [self.predict_layer[i](y_out) for i in range(self.args.n_codebooks)],
                dim=1,
            )  # [B K 1 card]
            logits = logits.squeeze(0).squeeze(1)  # [K card]
            assert logits.shape == torch.Size(
                (self.args.n_codebooks, self.n_audio_tokens[0])
            ), f"{logits.shape}"

            n_eog = sum(codebook_eog)
            assert n_eog < self.args.n_codebooks

            # Never generate EOS during speech editing.
            if self.args.eos > 0:
                for jj in range(self.args.n_codebooks):
                    logits[jj][self.args.eos] = -10000.0

            (
                samples,
                codebook_eog,
                prev_tokens,
                consec_silence_count,
            ) = sample_helper(
                n_eog,
                logits,
                codebook_eog,
                top_k,
                top_p,
                temperature,
                prev_tokens,
                consec_silence_count,
                stop_repetition,
                silence_tokens,
                cur_num_gen,
            )

            cur_num_gen += 1
            cur_generated.append(samples.squeeze(-1))  # [K,1] -> [K]

            samples_emb = torch.stack(
                [self.audio_embedding[k](samples[k]) for k in range(self.args.n_codebooks)],
                dim=0,
            )  # [K,1,D]
            samples_emb = samples_emb.sum(dim=0, keepdim=True)  # [1,1,D]

            if sum(codebook_eog) == self.args.n_codebooks:
                # Finished current masked span.
                codebook_eog = [False] * self.args.n_codebooks
                num_gen.append(cur_num_gen)
                cur_num_gen = 0
                generated.append(cur_generated)
                cur_generated = []

                # If there are more masked spans, prepend mask+empty to start next span.
                if len(more_mask_value) > 0:
                    next_mask_ind = more_mask_value.pop(0)
                    mask_emb = self.mask_embedding[next_mask_ind].unsqueeze(0).unsqueeze(0)
                    assert mask_emb.shape == torch.Size(
                        (1, 1, self.args.d_model)
                    ), mask_emb.shape
                    empty_token = torch.LongTensor([self.args.empty_token]).to(y.device)
                    empty_emb = torch.stack(
                        [self.audio_embedding[k](empty_token) for k in range(self.args.n_codebooks)],
                        dim=0,
                    ).sum(dim=0, keepdim=True)  # [1,1,D]
                    assert empty_emb.shape == torch.Size(
                        (1, 1, self.args.d_model)
                    ), empty_emb.shape
                    extra_emb = torch.cat([mask_emb, empty_emb], dim=1)  # [1,2,D]
                    samples_emb = torch.cat(
                        [samples_emb, extra_emb], dim=1
                    )  # [1,3,D]

                    # Reset silence repetition state for new span.
                    consec_silence_count = 0
                    prev_token = None

                    # Force decoder to see the last three tokens for kv-cache alignment.
                    new_masked_span = True
                else:
                    break
            else:
                assert samples_emb.shape == torch.Size(
                    (1, 1, self.args.d_model)
                ), f"samples_emb.shape: {samples_emb.shape}"

            if y_input.shape[1] > x_lens[0] * max_rate:
                break

            # Append newly generated token embeddings and update masks.
            embedded_y = torch.cat([embedded_y, samples_emb], dim=1)
            y_input = self.audio_positional_embedding(embedded_y)
            y_attention_mask = torch.triu(
                torch.ones(y_input.shape[1], y_input.shape[1]),
                diagonal=1,
            ).bool().to(y.device)
            new_y_lens = torch.LongTensor([y_input.shape[1]]).to(y.device)
            y_padding_mask = torch.full(
                (1, new_y_lens[0]), False
            ).to(y.device)

        # If we exited the loop early, make sure current span is accounted for.
        if len(generated) != num_mask:
            generated.append(cur_generated)
            num_gen.append(cur_num_gen)

        # Shift generated spans back and stitch with non-masked regions.
        flatten_gen = []
        for l, orig_span in enumerate(generated):
            span = torch.stack(orig_span, dim=0)  # [T, K]
            span = span.transpose(1, 0)  # [K, T]
            assert span.shape[0] == self.args.n_codebooks, span.shape
            unshifted_span = []
            for j, s in enumerate(span):
                start_from = j
                end_at = -(self.args.n_codebooks - start_from)
                unshifted_span.append(s[start_from:end_at])
            unshifted_span = torch.stack(unshifted_span, dim=0)
            flatten_gen.append(unshifted_span)

        assert len(non_mask_intervals[0]) - 1 == len(
            flatten_gen
        ), f"len(non_mask_intervals[0]): {len(non_mask_intervals[0])}, len(flatten_gen): {len(flatten_gen)}"

        res = []
        for orig_interval, gen in zip(non_mask_intervals[0], flatten_gen):
            res.append(y[0, :, orig_interval[0] : orig_interval[1]])
            res.append(gen)
        res.append(
            y[0, :, non_mask_intervals[0][-1][0] : non_mask_intervals[0][-1][1]]
        )
        res = torch.cat(res, dim=1).unsqueeze(0)  # [1, K, new_T]

        expected_y_len = y_len - sum(
            item[1] - item[0] for item in mask_intervals[0]
        ) + sum(item - self.args.n_codebooks for item in num_gen)
        if res.shape[-1] != expected_y_len:
            print(
                "expected_y_len: ",
                res.shape,
                expected_y_len,
                num_gen,
                mask_intervals,
            )
        if self.args.special_first:
            res = res - int(self.args.n_special)

        return res, num_gen, self.RE_GEN
