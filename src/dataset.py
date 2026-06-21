from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from graph_utils import torch_sparse_from_saved


def normalize_county_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5)


def load_county_ids_file(path: str | os.PathLike | None) -> List[str]:
    if path is None or str(path).strip() == "":
        return []
    county_path = Path(path)
    if not county_path.exists():
        raise FileNotFoundError(f"county_fips_file does not exist: {county_path}")

    counties: List[str] = []
    seen: set[str] = set()
    with county_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "county_fips" in reader.fieldnames:
            rows = (row.get("county_fips", "") for row in reader)
        else:
            f.seek(0)
            rows = (row[0] if row else "" for row in csv.reader(f))

        for raw in rows:
            for value in str(raw).replace(",", ";").split(";"):
                county = normalize_county_id(value)
                if county and county != "00000" and county not in seen:
                    counties.append(county)
                    seen.add(county)
    return counties


def parse_county_ids(value: str | Sequence[object] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    else:
        raw_items = [str(x) for x in value]
    counties: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        county = normalize_county_id(raw)
        if county and county != "00000" and county not in seen:
            counties.append(county)
            seen.add(county)
    return counties


@dataclass
class CityGraph:
    city_id: str
    tract_ids: List[str]
    num_nodes: int
    A_indices: torch.Tensor
    A_values: torch.Tensor
    S_indices: torch.Tensor
    S_values: torch.Tensor
    X_poi: torch.Tensor
    X_lu: torch.Tensor
    X_source: torch.Tensor
    X_destination: torch.Tensor
    M_indices: Optional[torch.Tensor] = None
    M_values: Optional[torch.Tensor] = None
    P_indices: Optional[torch.Tensor] = None
    P_values: Optional[torch.Tensor] = None
    PN_indices: Optional[torch.Tensor] = None
    PN_values: Optional[torch.Tensor] = None
    L_indices: Optional[torch.Tensor] = None
    L_values: Optional[torch.Tensor] = None
    LN_indices: Optional[torch.Tensor] = None
    LN_values: Optional[torch.Tensor] = None
    SRC_indices: Optional[torch.Tensor] = None
    SRC_values: Optional[torch.Tensor] = None
    SRCN_indices: Optional[torch.Tensor] = None
    SRCN_values: Optional[torch.Tensor] = None
    DST_indices: Optional[torch.Tensor] = None
    DST_values: Optional[torch.Tensor] = None
    DSTN_indices: Optional[torch.Tensor] = None
    DSTN_values: Optional[torch.Tensor] = None
    graph_build_meta: Optional[Dict] = None
    _A_sparse: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _S_sparse: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _M_sparse: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _view_sparse: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def to(self, device: torch.device | str) -> "CityGraph":
        kwargs = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v):
                kwargs[k] = v.to(device, non_blocking=True)
            else:
                kwargs[k] = v
        return CityGraph(**kwargs)

    def pin_memory(self) -> "CityGraph":
        kwargs = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v) and v.device.type == "cpu":
                kwargs[k] = v.pin_memory()
            else:
                kwargs[k] = v
        return CityGraph(**kwargs)

    def spatial_adj_sparse(self, device=None) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.A_indices.device
        key = str(target)
        if key not in self._A_sparse:
            self._A_sparse[key] = torch_sparse_from_saved(
                self.A_indices, self.A_values, (self.num_nodes, self.num_nodes), device=target
            )
        return self._A_sparse[key]

    def norm_adj_sparse(self, device=None) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.S_indices.device
        key = str(target)
        if key not in self._S_sparse:
            self._S_sparse[key] = torch_sparse_from_saved(
                self.S_indices, self.S_values, (self.num_nodes, self.num_nodes), device=target
            )
        return self._S_sparse[key]

    def mobility_flow_sparse(self, device=None) -> torch.Tensor | None:
        if self.M_indices is None or self.M_values is None:
            return None
        target = torch.device(device) if device is not None else self.M_indices.device
        key = str(target)
        if key not in self._M_sparse:
            self._M_sparse[key] = torch_sparse_from_saved(
                self.M_indices, self.M_values, (self.num_nodes, self.num_nodes), device=target
            )
        return self._M_sparse[key]

    def _optional_sparse(self, prefix: str, normalized: bool, device=None) -> torch.Tensor | None:
        if prefix == "poi":
            idx = self.PN_indices if normalized else self.P_indices
            val = self.PN_values if normalized else self.P_values
        elif prefix == "lu":
            idx = self.LN_indices if normalized else self.L_indices
            val = self.LN_values if normalized else self.L_values
        elif prefix == "source":
            idx = self.SRCN_indices if normalized else self.SRC_indices
            val = self.SRCN_values if normalized else self.SRC_values
        elif prefix == "destination":
            idx = self.DSTN_indices if normalized else self.DST_indices
            val = self.DSTN_values if normalized else self.DST_values
        else:
            raise ValueError(prefix)
        if idx is None or val is None:
            raise KeyError(f"Missing {prefix} {'normalized' if normalized else 'raw'} adjacency in processed city file.")
        target = torch.device(device) if device is not None else idx.device
        key = f"{prefix}:{'norm' if normalized else 'raw'}:{target}"
        if key not in self._view_sparse:
            self._view_sparse[key] = torch_sparse_from_saved(idx, val, (self.num_nodes, self.num_nodes), device=target)
        return self._view_sparse[key]

    def view_adj_sparse(self, view: str, device=None, normalized: bool = False) -> torch.Tensor:
        return self._optional_sparse(view, normalized=normalized, device=device)

    def poi_adj_sparse(self, device=None, normalized: bool = False) -> torch.Tensor:
        return self.view_adj_sparse("poi", device=device, normalized=normalized)

    def lu_adj_sparse(self, device=None, normalized: bool = False) -> torch.Tensor:
        return self.view_adj_sparse("lu", device=device, normalized=normalized)

    def source_adj_sparse(self, device=None, normalized: bool = False) -> torch.Tensor:
        return self.view_adj_sparse("source", device=device, normalized=normalized)

    def destination_adj_sparse(self, device=None, normalized: bool = False) -> torch.Tensor:
        return self.view_adj_sparse("destination", device=device, normalized=normalized)


class CityDataset(Dataset):
    def __init__(
        self,
        processed_root: str,
        cache_mode: str = "off",
        county_ids: Sequence[object] | str | None = None,
        county_fips_file: str | os.PathLike | None = None,
    ):
        all_paths = sorted(glob.glob(os.path.join(processed_root, "*.pt")))
        selected_counties = parse_county_ids(county_ids)
        if not selected_counties and county_fips_file:
            selected_counties = load_county_ids_file(county_fips_file)
        if selected_counties:
            wanted = set(selected_counties)
            available = {normalize_county_id(os.path.splitext(os.path.basename(path))[0]) for path in all_paths}
            missing = [county for county in selected_counties if county not in available]
            if missing:
                raise FileNotFoundError(
                    f"Processed root {processed_root} is missing {len(missing)} requested counties, "
                    f"first={missing[:10]}"
                )
            all_paths = [
                path
                for path in all_paths
                if normalize_county_id(os.path.splitext(os.path.basename(path))[0]) in wanted
            ]
        self.paths = all_paths
        self.cache_mode = str(cache_mode)
        if self.cache_mode not in {"off", "cpu"}:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}")
        self._graph_cache: Dict[int, CityGraph] = {}
        self._num_nodes_cache: Dict[int, int] = {}
        if not self.paths:
            raise FileNotFoundError(f"No processed .pt city files found in {processed_root}")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_obj(self, idx: int) -> Dict:
        try:
            return torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(self.paths[idx], map_location="cpu")

    def _build_city_graph(self, obj: Dict, idx: int) -> CityGraph:
        num_nodes = int(obj["num_nodes"])
        tract_ids = [str(tid) for tid in obj["tract_ids"]]
        if len(tract_ids) != num_nodes:
            raise ValueError(
                f"{self.paths[idx]} has {len(tract_ids)} tract IDs but num_nodes={num_nodes}"
            )

        city = CityGraph(
            city_id=obj["city_id"],
            tract_ids=tract_ids,
            num_nodes=num_nodes,
            A_indices=obj["A_indices"],
            A_values=obj["A_values"],
            S_indices=obj["S_indices"],
            S_values=obj["S_values"],
            X_poi=obj["X_poi"].float(),
            X_lu=obj["X_lu"].float(),
            X_source=obj["X_source"].float(),
            X_destination=obj["X_destination"].float(),
            M_indices=obj.get("M_indices"),
            M_values=obj.get("M_values"),
            P_indices=obj.get("P_indices"),
            P_values=obj.get("P_values"),
            PN_indices=obj.get("PN_indices"),
            PN_values=obj.get("PN_values"),
            L_indices=obj.get("L_indices"),
            L_values=obj.get("L_values"),
            LN_indices=obj.get("LN_indices"),
            LN_values=obj.get("LN_values"),
            SRC_indices=obj.get("SRC_indices"),
            SRC_values=obj.get("SRC_values"),
            SRCN_indices=obj.get("SRCN_indices"),
            SRCN_values=obj.get("SRCN_values"),
            DST_indices=obj.get("DST_indices"),
            DST_values=obj.get("DST_values"),
            DSTN_indices=obj.get("DSTN_indices"),
            DSTN_values=obj.get("DSTN_values"),
            graph_build_meta=obj.get("graph_build_meta"),
        )
        self._num_nodes_cache[idx] = int(city.num_nodes)
        return city

    def __getitem__(self, idx: int) -> CityGraph:
        if self.cache_mode == "cpu" and idx in self._graph_cache:
            return self._graph_cache[idx]
        obj = self._load_obj(idx)
        city = self._build_city_graph(obj, idx)
        if self.cache_mode == "cpu":
            self._graph_cache[idx] = city
        return city

    def get_num_nodes(self, idx: int) -> int:
        cached = self._num_nodes_cache.get(idx)
        if cached is not None:
            return int(cached)
        if self.cache_mode == "cpu" and idx in self._graph_cache:
            cached = int(self._graph_cache[idx].num_nodes)
            self._num_nodes_cache[idx] = cached
            return cached
        obj = self._load_obj(idx)
        num_nodes = int(obj["num_nodes"])
        self._num_nodes_cache[idx] = num_nodes
        return num_nodes

    def all_num_nodes(self) -> List[int]:
        return [self.get_num_nodes(i) for i in range(len(self.paths))]


def city_collate(batch: List[CityGraph]) -> List[CityGraph]:
    return batch


def infer_feature_dims(processed_root: str) -> tuple[int, int, int, int]:
    ds = CityDataset(processed_root)
    c = ds[0]
    return c.X_poi.shape[1], c.X_lu.shape[1], c.X_source.shape[1], c.X_destination.shape[1]
