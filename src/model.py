from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CityGraph
from graph_utils import torch_sparse_mm

VIEW_NAMES = ("poi", "lu", "source", "destination")
VIEW_INDEX = {name: idx for idx, name in enumerate(VIEW_NAMES)}


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def normalize_view_smooth_type(smooth_type: str) -> str:
    key = str(smooth_type).strip().lower().replace("-", "_")
    aliases = {
        "learnable": "learnable_diffuse",
        "learnable_diffuse": "learnable_diffuse",
        "learnable_diffusion": "learnable_diffuse",
        "diffuse": "learnable_diffuse",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported view_smooth_type for the final model: {smooth_type}")
    return aliases[key]


class ViewNativeDecomposition(nn.Module):
    def __init__(
        self,
        smooth_steps: int = 2,
        *,
        dim: int,
        smooth_type: str = "learnable_diffuse",
        num_views: int = len(VIEW_NAMES),
    ):
        super().__init__()
        if smooth_steps < 1:
            raise ValueError("smooth_steps must be >= 1")
        self.smooth_steps = int(smooth_steps)
        self.smooth_type = normalize_view_smooth_type(smooth_type)
        self.num_views = int(num_views)
        logits = torch.full((self.num_views, self.smooth_steps + 1), -4.0)
        logits[:, -1] = 4.0
        self.diffusion_logits = nn.Parameter(logits)

    def forward(self, view_idx: int, s_norm: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        terms = [h]
        cur = h
        for _ in range(self.smooth_steps):
            cur = torch_sparse_mm(s_norm, cur).to(dtype=h.dtype)
            terms.append(cur)
        weights = torch.softmax(self.diffusion_logits[int(view_idx)], dim=0).to(dtype=h.dtype)
        smooth = sum(w * term for w, term in zip(weights, terms))
        residual = h - smooth
        return smooth, residual


class SparseTractGNNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, gate_init: float = -2.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.message_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(dropout)
        self.message_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor, adjs: Sequence[torch.Tensor]) -> torch.Tensor:
        residual = x
        x_norm = self.ln1(x)
        msg_in = self.message_proj(x_norm)
        msg = torch.zeros_like(msg_in)
        for adj in adjs:
            msg = msg + torch_sparse_mm(adj, msg_in).to(dtype=x_norm.dtype)
        msg = msg / max(1, len(adjs))
        msg = torch.nan_to_num(msg.float(), nan=0.0, posinf=50.0, neginf=-50.0).to(dtype=x_norm.dtype)
        gate = torch.sigmoid(self.message_gate).to(dtype=x_norm.dtype)
        x = residual + self.drop(gate * self.out(msg))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class SparseTractGNN(nn.Module):
    def __init__(self, dim: int, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([SparseTractGNNLayer(dim, dropout=dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, adjs: Sequence[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adjs)
        return x


def normalize_tract_context_type(context_type: str) -> str:
    key = str(context_type).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "identity": "none",
        "no": "none",
        "gcn": "gcn",
        "gnn": "gcn",
        "sparse_gcn": "gcn",
        "sparse_gnn": "gcn",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported tract_context_type for the final model: {context_type}")
    return aliases[key]


def normalize_tract_context_position(position: str) -> str:
    key = str(position).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "post": "post_residual",
        "after_residual": "post_residual",
        "post_residual": "post_residual",
        "after": "post_residual",
    }
    if key not in aliases:
        raise ValueError(f"Unknown tract_context_position: {position}")
    return aliases[key]


def normalize_tract_context_graph(graph: str) -> str:
    key = str(graph).strip().lower().replace("-", "_")
    aliases = {
        "spatial": "spatial",
        "space": "spatial",
    }
    if key not in aliases:
        raise ValueError(f"Unknown tract_context_graph: {graph}")
    return aliases[key]


def normalize_residual_fusion(fusion: str) -> str:
    key = str(fusion).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "no": "none",
        "attn": "attn_gated",
        "attention": "attn_gated",
        "attn_gated": "attn_gated",
        "attention_gated": "attn_gated",
        "resattn": "attn_gated",
        "resattn_gated": "attn_gated",
        "attn_add": "attn_add",
        "attention_add": "attn_add",
        "resattn_add": "attn_add",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported residual_fusion for the final model: {fusion}")
    return aliases[key]


def normalize_motif_consensus_type(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "shared": "shared",
        "motif": "shared",
        "shared_motif": "shared",
        "shared_motif_consensus": "shared",
        "view_mean": "view_mean",
        "mean": "view_mean",
        "no_motif": "view_mean",
        "none": "view_mean",
        "off": "view_mean",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported motif_consensus_type: {value}")
    return aliases[key]


def normalize_profile_residual_decomp(value: str | bool) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "on": "on",
        "true": "on",
        "yes": "on",
        "1": "on",
        "profile_residual": "on",
        "profile_residual_decomp": "on",
        "off": "off",
        "false": "off",
        "no": "off",
        "0": "off",
        "none": "off",
        "raw": "off",
        "raw_views": "off",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported profile_residual_decomp: {value}")
    return aliases[key]


def normalize_enabled_views(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        presets = {
            "all": VIEW_NAMES,
            "full": VIEW_NAMES,
            "none": (),
            "wo_poi": ("lu", "source", "destination"),
            "no_poi": ("lu", "source", "destination"),
            "wo_landuse": ("poi", "source", "destination"),
            "wo_land_use": ("poi", "source", "destination"),
            "wo_lu": ("poi", "source", "destination"),
            "no_landuse": ("poi", "source", "destination"),
            "no_lu": ("poi", "source", "destination"),
            "wo_mobility": ("poi", "lu"),
            "no_mobility": ("poi", "lu"),
            "only_poi": ("poi",),
            "only_landuse": ("lu",),
            "only_land_use": ("lu",),
            "only_lu": ("lu",),
            "only_mobility": ("source", "destination"),
        }
        if key in presets:
            items = presets[key]
        else:
            items = [part.strip().lower().replace("-", "_") for part in value.replace(",", " ").split()]
    else:
        items = [str(part).strip().lower().replace("-", "_") for part in value]

    aliases = {
        "poi": "poi",
        "lu": "lu",
        "landuse": "lu",
        "land_use": "lu",
        "source": "source",
        "src": "source",
        "destination": "destination",
        "dest": "destination",
        "dst": "destination",
        "mobility": "mobility",
    }
    out: list[str] = []
    for item in items:
        canonical = aliases.get(item)
        if canonical == "mobility":
            expanded = ("source", "destination")
        elif canonical is not None:
            expanded = (canonical,)
        else:
            raise ValueError(f"Unsupported view name in enabled_views: {item}")
        for view in expanded:
            if view not in out:
                out.append(view)
    if not out:
        raise ValueError("enabled_views must keep at least one input view")
    return tuple(out)


@dataclass
class UrbanMotifOutput:
    common: torch.Tensor
    z: torch.Tensor
    Q: torch.Tensor
    motif_state: torch.Tensor
    smooth_views: tuple[torch.Tensor, ...]
    residual_views: tuple[torch.Tensor, ...]
    motif_view_attention: torch.Tensor
    contrast_views: tuple[torch.Tensor, ...]
    structure_similarity: torch.Tensor


class UrbanMotifModel(nn.Module):
    def __init__(
        self,
        poi_dim: int,
        lu_dim: int,
        source_dim: int,
        destination_dim: int,
        dim: int = 256,
        num_motifs: int = 64,
        smooth_steps: int = 2,
        temperature: float = 0.2,
        view_smooth_type: str = "learnable_diffuse",
        tract_context_type: str = "gcn",
        tract_context_position: str = "post_residual",
        tract_context_graph: str = "spatial",
        tract_context_layers: int = 2,
        residual_fusion: str = "attn_gated",
        motif_consensus_type: str = "shared",
        profile_residual_decomp: str | bool = "on",
        enabled_views: str | Sequence[str] = "all",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_motifs = int(num_motifs)
        self.temperature = float(temperature)
        self.smooth_steps = int(smooth_steps)
        self.view_smooth_type = normalize_view_smooth_type(view_smooth_type)
        self.tract_context_type = normalize_tract_context_type(tract_context_type)
        self.tract_context_position = normalize_tract_context_position(tract_context_position)
        self.tract_context_graph = normalize_tract_context_graph(tract_context_graph)
        self.residual_fusion = normalize_residual_fusion(residual_fusion)
        self.motif_consensus_type = normalize_motif_consensus_type(motif_consensus_type)
        self.profile_residual_decomp = normalize_profile_residual_decomp(profile_residual_decomp)
        self.enabled_views = normalize_enabled_views(enabled_views)
        self.enabled_view_indices = tuple(VIEW_INDEX[name] for name in self.enabled_views)
        if self.smooth_steps < 1:
            raise ValueError("smooth_steps must be >= 1")
        if tract_context_layers < 0:
            raise ValueError("tract_context_layers must be >= 0")
        if self.tract_context_type == "none":
            self.tract_context_position = "none"

        self.poi_proj = ModalityProjector(poi_dim, dim, dropout)
        self.lu_proj = ModalityProjector(lu_dim, dim, dropout)
        self.source_proj = ModalityProjector(source_dim, dim, dropout)
        self.destination_proj = ModalityProjector(destination_dim, dim, dropout)
        self.decomp = ViewNativeDecomposition(
            self.smooth_steps,
            dim=dim,
            smooth_type=self.view_smooth_type,
            num_views=len(VIEW_NAMES),
        )

        self.motif_keys = nn.Parameter(torch.randn(num_motifs, len(VIEW_NAMES), dim) * 0.02)
        self.view_attn_logits = nn.Parameter(torch.zeros(num_motifs, len(VIEW_NAMES)))
        self.motif_value_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in VIEW_NAMES])
        self.motif_proto_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in VIEW_NAMES])
        self.contrast_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim),
                )
                for _ in VIEW_NAMES
            ]
        )
        self.structure_q1 = nn.Linear(len(VIEW_NAMES) * dim, dim)
        self.structure_q2 = nn.Linear(len(VIEW_NAMES) * dim, dim)
        if self.tract_context_position != "none" and tract_context_layers > 0:
            if self.tract_context_type == "gcn":
                self.tract_ctx = SparseTractGNN(dim=dim, layers=tract_context_layers, dropout=dropout)
            else:
                self.tract_ctx = None
        else:
            self.tract_ctx = None

        self.residual_view_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in VIEW_NAMES])
        self.residual_attn_score = nn.Linear(2 * dim, 1)
        self.gate = nn.Linear(2 * dim, 1)
        for proj in self.residual_view_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self._freeze_unused_parameters_for_config()

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _freeze_parameter(param: nn.Parameter) -> None:
        param.requires_grad_(False)

    def _freeze_unused_parameters_for_config(self) -> None:
        inactive = set(range(len(VIEW_NAMES))) - set(self.enabled_view_indices)
        view_modules = (
            self.poi_proj,
            self.lu_proj,
            self.source_proj,
            self.destination_proj,
        )
        for idx in inactive:
            self._freeze_module(view_modules[idx])
            self._freeze_module(self.motif_value_proj[idx])
            self._freeze_module(self.motif_proto_proj[idx])
            self._freeze_module(self.contrast_projectors[idx])
            self._freeze_module(self.residual_view_proj[idx])

        if self.motif_consensus_type != "shared":
            self._freeze_parameter(self.motif_keys)
            self._freeze_parameter(self.view_attn_logits)
            for module in self.motif_value_proj:
                self._freeze_module(module)
            for module in self.motif_proto_proj:
                self._freeze_module(module)

        if self.profile_residual_decomp == "off":
            self._freeze_module(self.decomp)

        if self.profile_residual_decomp == "off" or self.residual_fusion == "none":
            for module in self.residual_view_proj:
                self._freeze_module(module)
            self._freeze_module(self.residual_attn_score)
            self._freeze_module(self.gate)
        elif self.residual_fusion == "attn_add":
            self._freeze_module(self.gate)

    def project_modalities(self, city: CityGraph) -> tuple[torch.Tensor, ...]:
        hs = (
            self.poi_proj(city.X_poi),
            self.lu_proj(city.X_lu),
            self.source_proj(city.X_source),
            self.destination_proj(city.X_destination),
        )
        return tuple(h if idx in self.enabled_view_indices else torch.zeros_like(h) for idx, h in enumerate(hs))

    def view_norm_adjs(self, city: CityGraph) -> tuple[torch.Tensor, ...]:
        device = city.X_poi.device
        return (
            city.poi_adj_sparse(device=device, normalized=True),
            city.lu_adj_sparse(device=device, normalized=True),
            city.source_adj_sparse(device=device, normalized=True),
            city.destination_adj_sparse(device=device, normalized=True),
        )

    def decompose_views(
        self,
        norm_adjs: Sequence[torch.Tensor],
        hs: Sequence[torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        smooth, residual = [], []
        for view_idx, (s, h) in enumerate(zip(norm_adjs, hs)):
            if self.profile_residual_decomp == "off":
                smooth.append(h)
                residual.append(torch.zeros_like(h))
                continue
            sm, rs = self.decomp(view_idx, s, h)
            smooth.append(sm)
            residual.append(rs)
        return tuple(smooth), tuple(residual)

    def tract_context_adjs(self, city: CityGraph, norm_adjs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if self.tract_context_graph == "spatial":
            return (city.norm_adj_sparse(device=city.X_poi.device),)
        raise ValueError(f"Unknown tract_context_graph: {self.tract_context_graph}")

    def motif_view_attention(self) -> torch.Tensor:
        logits = self.view_attn_logits
        if len(self.enabled_view_indices) != len(VIEW_NAMES):
            mask = torch.full_like(logits, -1e9)
            mask[:, list(self.enabled_view_indices)] = 0.0
            logits = logits + mask
        return torch.softmax(logits, dim=-1)

    def assign_motifs(self, smooth_views: Sequence[torch.Tensor], attn: torch.Tensor) -> torch.Tensor:
        keys = self.motif_keys.to(dtype=smooth_views[0].dtype)
        attn = attn.to(dtype=smooth_views[0].dtype)
        dist = None
        for m, h in enumerate(smooth_views):
            h_norm = F.layer_norm(h, (h.size(-1),))
            k_norm = F.layer_norm(keys[:, m, :], (keys.size(-1),))
            x2 = h_norm.pow(2).sum(dim=-1, keepdim=True)  # [N,1]
            m2 = k_norm.pow(2).sum(dim=-1).view(1, -1)  # [1,K]
            cross = h_norm @ k_norm.t()  # [N,K]
            dm = x2 + m2 - 2.0 * cross
            dm = dm * attn[:, m].view(1, -1)
            dist = dm if dist is None else dist + dm
        return torch.softmax(-dist / self.temperature, dim=-1)

    def motif_node_state(
        self,
        Q: torch.Tensor,
        smooth_views: Sequence[torch.Tensor],
        attn: torch.Tensor,
    ) -> torch.Tensor:
        counts = Q.sum(dim=0).clamp_min(1e-6).unsqueeze(-1)  # [K,1]
        y = None
        for m, h in enumerate(smooth_views):
            x_m = (Q.t() @ h) / counts  # [K,d]
            proto_m = self.motif_keys[:, m, :].to(dtype=h.dtype)
            part = self.motif_value_proj[m](x_m) + self.motif_proto_proj[m](proto_m)
            part = part * attn[:, m].to(dtype=h.dtype).unsqueeze(-1)
            y = part if y is None else y + part
        return y

    def structure_similarity(self, smooth_views: Sequence[torch.Tensor]) -> torch.Tensor:
        z_concat = torch.cat(smooth_views, dim=-1)
        q1 = self.structure_q1(z_concat)
        q2 = self.structure_q2(z_concat)
        logits = (q1 @ q2.t()) / math.sqrt(max(1, q1.size(-1)))
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        return torch.softmax(logits, dim=1).to(dtype=z_concat.dtype)

    def projected_contrast_views(
        self,
        smooth_views: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        out = [
            proj(h)
            for idx, (proj, h) in enumerate(zip(self.contrast_projectors, smooth_views))
            if idx in self.enabled_view_indices
        ]
        return tuple(out)

    def residual_state(self, common: torch.Tensor, residual_views: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.residual_fusion == "none":
            return torch.zeros_like(common)
        projected = torch.stack(
            [
                proj(h)
                for idx, (proj, h) in enumerate(zip(self.residual_view_proj, residual_views))
                if idx in self.enabled_view_indices
            ],
            dim=1,
        )
        common_expanded = common.unsqueeze(1).expand(-1, projected.size(1), -1)
        scores = self.residual_attn_score(torch.cat([common_expanded, projected], dim=-1)).squeeze(-1)
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        weights = torch.softmax(scores, dim=1).to(dtype=projected.dtype)
        return (weights.unsqueeze(-1) * projected).sum(dim=1)

    def common_embedding(
        self,
        Q: torch.Tensor,
        y_ctx: torch.Tensor,
        smooth_views: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if self.motif_consensus_type == "shared":
            return Q @ y_ctx
        active = [h for idx, h in enumerate(smooth_views) if idx in self.enabled_view_indices]
        return torch.stack(active, dim=0).mean(dim=0)

    def forward(self, city: CityGraph) -> UrbanMotifOutput:
        hs = self.project_modalities(city)
        norm_adjs = self.view_norm_adjs(city)
        smooth, residual = self.decompose_views(norm_adjs, hs)
        contrast_views = self.projected_contrast_views(smooth)
        structure_similarity = self.structure_similarity(smooth)

        attn = self.motif_view_attention()
        if self.motif_consensus_type == "shared":
            Q = self.assign_motifs(smooth, attn)
            y_ctx = self.motif_node_state(Q, smooth, attn)
        else:
            Q = torch.full(
                (smooth[0].size(0), self.num_motifs),
                1.0 / max(1, self.num_motifs),
                device=smooth[0].device,
                dtype=smooth[0].dtype,
            )
            y_ctx = torch.zeros(
                (self.num_motifs, self.dim),
                device=smooth[0].device,
                dtype=smooth[0].dtype,
            )
        C = self.common_embedding(Q, y_ctx, smooth)

        R = torch.zeros_like(C) if self.profile_residual_decomp == "off" else self.residual_state(C, residual)
        if self.residual_fusion == "attn_add":
            Z = C + R
        elif self.residual_fusion == "none":
            Z = C
        else:
            alpha = torch.sigmoid(self.gate(torch.cat([C, R], dim=-1)))
            Z = C + alpha * R
        if self.tract_ctx is not None:
            Z = self.tract_ctx(Z, self.tract_context_adjs(city, norm_adjs))

        return UrbanMotifOutput(
            common=C,
            z=Z,
            Q=Q,
            motif_state=y_ctx,
            smooth_views=smooth,
            residual_views=residual,
            motif_view_attention=attn,
            contrast_views=contrast_views,
            structure_similarity=structure_similarity,
        )

def motif_balance_loss(Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = Q.mean(dim=0).clamp_min(eps)
    q = q / q.sum()
    k = q.numel()
    u = torch.full_like(q, 1.0 / k)
    return (q * (q / u).log()).sum()


def _safe_tensor(x: torch.Tensor, clamp_abs: float = 0.0) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if float(clamp_abs) > 0:
        return x.clamp(-float(clamp_abs), float(clamp_abs))
    return x


def _dense_submatrix_from_sparse(
    a: torch.Tensor,
    idx: torch.Tensor,
    *,
    rescale_max: bool = True,
    fill_diagonal: float | None = 1.0,
    clamp_max: float | None = 1.0,
) -> torch.Tensor:
    a = a.coalesce()
    n = int(a.size(0))
    m = int(idx.numel())
    device = idx.device
    dense = torch.zeros(m, m, device=device, dtype=torch.float32)
    if m == 0 or a._nnz() == 0:
        return dense
    mapping = torch.full((n,), -1, device=device, dtype=torch.long)
    mapping[idx] = torch.arange(m, device=device, dtype=torch.long)
    rows, cols = a.indices()[0].long(), a.indices()[1].long()
    mr, mc = mapping[rows], mapping[cols]
    mask = (mr >= 0) & (mc >= 0)
    if mask.any():
        dense[mr[mask], mc[mask]] = a.values()[mask].float()
    if rescale_max and dense.numel() and dense.max() > 0:
        dense = dense / dense.max().clamp_min(1e-6)
    if fill_diagonal is not None:
        dense.fill_diagonal_(float(fill_diagonal))
    dense = dense.clamp_min(0.0)
    if clamp_max is not None:
        dense = dense.clamp_max(float(clamp_max))
    return dense


def graph_soft_ce_reconstruction_loss(
    emb: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.2,
    max_abs_embedding: float = 20.0,
    logit_clip: float = 30.0,
) -> torch.Tensor:
    if emb.size(0) <= 1:
        return emb.sum() * 0.0
    h = F.normalize(_safe_tensor(emb, clamp_abs=max_abs_embedding), dim=-1)
    logits = (h @ h.t()) / max(float(temperature), 1e-6)
    if float(logit_clip) > 0:
        logits = logits.clamp(-float(logit_clip), float(logit_clip))
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    row_mass = target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    target_prob = target / row_mass
    log_prob = F.log_softmax(logits, dim=1)
    return -(target_prob * log_prob).sum(dim=1).mean()


def saware_mora_contrastive_loss(
    z: torch.Tensor,
    structure: torch.Tensor,
    contrast_views: Sequence[torch.Tensor],
    temperature: float = 0.5,
    logit_clip: float = 30.0,
) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.sum() * 0.0
    h = F.normalize(_safe_tensor(z), dim=-1)
    s = torch.nan_to_num(structure.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    eye = torch.eye(h.size(0), device=h.device, dtype=torch.bool)
    weights = torch.where(eye, torch.ones_like(s), (1.0 - s).clamp_min(1e-12))
    log_weights = torch.log(weights)
    total = z.sum() * 0.0
    for view in contrast_views:
        v = F.normalize(_safe_tensor(view), dim=-1)
        sim = (h @ v.t()) / max(float(temperature), 1e-6)
        if float(logit_clip) > 0:
            sim = sim.clamp(-float(logit_clip), float(logit_clip))
        positive = torch.diag(sim)
        z_to_view = -(positive - torch.logsumexp(sim + log_weights, dim=1)).mean()
        view_to_z = -(positive - torch.logsumexp(sim.t() + log_weights, dim=1)).mean()
        total = total + 0.5 * (z_to_view + view_to_z)
    return total / max(1, len(contrast_views))


def graph_positive_contrastive_loss(
    z: torch.Tensor,
    positive: torch.Tensor,
    structure: torch.Tensor,
    *,
    temperature: float = 0.2,
    logit_clip: float = 30.0,
) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.sum() * 0.0
    pos = torch.nan_to_num(positive.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)
    pos = torch.maximum(pos, pos.t())
    pos.fill_diagonal_(0.0)
    pos_mask = pos > 0
    valid = pos_mask.any(dim=1)
    if not bool(valid.any()):
        return z.sum() * 0.0

    h = F.normalize(_safe_tensor(z), dim=-1)
    sim = (h @ h.t()) / max(float(temperature), 1e-6)
    if float(logit_clip) > 0:
        sim = sim.clamp(-float(logit_clip), float(logit_clip))

    s = torch.nan_to_num(structure.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    eye = torch.eye(h.size(0), device=h.device, dtype=torch.bool)
    neg_weight = (1.0 - s).clamp_min(1e-12)
    denom_weight = torch.where(pos_mask, torch.ones_like(pos), neg_weight)
    denom_weight = torch.where(eye, torch.zeros_like(denom_weight), denom_weight).clamp_min(1e-12)
    pos_weight = torch.where(pos_mask, pos.clamp_min(1e-12), torch.zeros_like(pos))

    pos_logits = sim + torch.log(pos_weight.clamp_min(1e-12))
    pos_logits = pos_logits.masked_fill(~pos_mask, -float("inf"))
    denom_logits = sim + torch.log(denom_weight)
    denom_logits = denom_logits.masked_fill(eye, -float("inf"))
    loss = -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(denom_logits, dim=1))
    return loss[valid].mean()


def urbanmotif_loss(
    out: UrbanMotifOutput,
    city: CityGraph,
    sample_size: int = 1024,
    contrast_temperature: float = 0.5,
    poi_weight: float = 0.005,
    lu_weight: float = 0.005,
    mobility_recon_weight: float = 0.015,
    contrast_weight: float = 0.1,
    balance_weight: float = 0.003,
    graph_contrast_weight: float = 0.03,
    graph_contrast_temperature: float = 0.2,
    graph_recon_temperature: float = 0.2,
    max_abs_embedding: float = 20.0,
    logit_clip: float = 30.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = out.z.device
    n = out.z.size(0)
    if sample_size > 0 and n > sample_size:
        idx = torch.randperm(n, device=device)[:sample_size]
    else:
        idx = torch.arange(n, device=device)

    z = out.z.index_select(0, idx)
    contrast_views = [h.index_select(0, idx) for h in out.contrast_views]
    structure = out.structure_similarity.index_select(0, idx).index_select(1, idx)
    poi_adj = city.poi_adj_sparse(device=device, normalized=False)
    lu_adj = city.lu_adj_sparse(device=device, normalized=False)
    source_adj = city.source_adj_sparse(device=device, normalized=False)
    destination_adj = city.destination_adj_sparse(device=device, normalized=False)
    poi_target = _dense_submatrix_from_sparse(poi_adj, idx)
    lu_target = _dense_submatrix_from_sparse(lu_adj, idx)
    source_target = _dense_submatrix_from_sparse(source_adj, idx)
    destination_target = _dense_submatrix_from_sparse(destination_adj, idx)

    recon_fn = lambda emb, target: graph_soft_ce_reconstruction_loss(
        emb,
        target,
        temperature=graph_recon_temperature,
        max_abs_embedding=max_abs_embedding,
        logit_clip=logit_clip,
    )

    poi_loss = recon_fn(z, poi_target)
    lu_loss = recon_fn(z, lu_target)
    source_recon_loss = recon_fn(z, source_target)
    destination_recon_loss = recon_fn(z, destination_target)

    contrast_loss = saware_mora_contrastive_loss(
        z,
        structure,
        contrast_views,
        temperature=contrast_temperature,
        logit_clip=logit_clip,
    )
    bal = motif_balance_loss(out.Q)

    graph_contrast_loss = z.sum() * 0.0
    if float(graph_contrast_weight) > 0:
        spatial_adj = city.spatial_adj_sparse(device=device)
        graph_pos = _dense_submatrix_from_sparse(spatial_adj, idx, fill_diagonal=0.0, clamp_max=1.0)
        graph_pos = graph_pos.clamp_max(1.0)
        graph_contrast_loss = graph_positive_contrastive_loss(
            z,
            graph_pos,
            structure,
            temperature=graph_contrast_temperature,
            logit_clip=logit_clip,
        )

    poi_term = poi_weight * poi_loss
    lu_term = lu_weight * lu_loss
    mobility_recon_loss = source_recon_loss + destination_recon_loss
    mobility_recon_term = mobility_recon_weight * mobility_recon_loss
    contrast_term = contrast_weight * contrast_loss
    balance_term = balance_weight * bal
    graph_contrast_term = graph_contrast_weight * graph_contrast_loss
    total = (
        poi_term
        + lu_term
        + mobility_recon_term
        + contrast_term
        + balance_term
        + graph_contrast_term
    )
    parts = {
        "poi_loss": float(poi_loss.detach().cpu()),
        "lu_loss": float(lu_loss.detach().cpu()),
        "source_recon_loss": float(source_recon_loss.detach().cpu()),
        "destination_recon_loss": float(destination_recon_loss.detach().cpu()),
        "mobility_recon_loss": float(mobility_recon_loss.detach().cpu()),
        "contrast_loss": float(contrast_loss.detach().cpu()),
        "balance_loss": float(bal.detach().cpu()),
        "graph_contrast_loss": float(graph_contrast_loss.detach().cpu()),
        "mobility_recon_effective_weight": float(mobility_recon_weight),
        "poi_weighted_loss": float(poi_term.detach().cpu()),
        "lu_weighted_loss": float(lu_term.detach().cpu()),
        "mobility_recon_weighted_loss": float(mobility_recon_term.detach().cpu()),
        "contrast_weighted_loss": float(contrast_term.detach().cpu()),
        "balance_weighted_loss": float(balance_term.detach().cpu()),
        "graph_contrast_weighted_loss": float(graph_contrast_term.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, parts
