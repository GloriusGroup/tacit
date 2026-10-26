import warnings
from functools import partial
import multiprocessing as mp
import os
import json
import time
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
from bond_match.sampler import weighted_shard_aware_sample
import torch
from functools import partial

from bond_match.dataset.constants import (
    SHARD_SAMPLE_LIMIT,
)
from bond_match.dataset.metadata import (
    SampleMetadata,
    _coerce_sample_metadata,
    _dataset_cache_prefix,
    _trajectory_numpy_to_tensors,
)
from bond_match.dataset.processing import (
    process_reaction_entry,
    process_reaction_entry_v3,
    process_reaction_entry_v3_trajectory,
)

class ReactionDataset(Dataset):
    def __init__(
        self,
        json_path,
        include_product=True,
        limit=None,
        force_recalculate_cache=False,
        extended_features=False,
        data_version="v2_graph",
        use_all_mechanisms=False,
        enable_history=False,
        enable_blockspace_product=True,
        sample_mode="step",
    ):
        self.data = []
        self.extended_features = extended_features
        self.data_version = data_version
        self.use_all_mechanisms = use_all_mechanisms
        self.enable_history = enable_history
        self.sample_mode = sample_mode

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)

        config_str = (
            f"{_dataset_cache_prefix(json_path)}_"
            f"{limit}_{include_product}_{data_version}"
        )
        if extended_features:
            config_str += "_extended"
        if use_all_mechanisms:
            config_str += "_all_mechs"
        if enable_history:
            config_str += "_historyv2"
        if data_version == "v3_blockspace" and include_product:
            config_str += "_blockprod" if enable_blockspace_product else "_rxnprod"
        if sample_mode != "step":
            config_str += f"_{sample_mode}"
        cache_file = os.path.join(cache_dir, f"{config_str}.pkl")

        # Bind all parameters except the entry for multiprocessing.
        if data_version == "v3_blockspace":
            process_entry_fn = (
                process_reaction_entry_v3_trajectory
                if sample_mode == "trajectory"
                else process_reaction_entry_v3
            )
            process_fn = partial(
                process_entry_fn,
                include_product=include_product,
                extended_features=extended_features,
                use_all_mechanisms=use_all_mechanisms,
                enable_blockspace_product=enable_blockspace_product,
            )
            if sample_mode != "trajectory":
                process_fn = partial(process_fn, enable_history=enable_history)
        else:
            process_fn = partial(
                process_reaction_entry,
                include_product=include_product,
                extended_features=extended_features,
                enable_history=enable_history,
            )

        if os.path.exists(cache_file) and not force_recalculate_cache:
            print(f"Loading cached dataset for {os.path.basename(json_path)}...")
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
        else:
            if os.path.exists(json_path):
                # Support JSON arrays and JSONL files with one entry per line.
                if json_path.endswith(".jsonl"):
                    with open(json_path, "r") as f:
                        raw_data = [json.loads(line) for line in f if line.strip()]
                else:
                    with open(json_path, "r") as f:
                        raw_data = json.load(f)

                if limit and limit < len(raw_data):
                    import random

                    sample_indices = sorted(random.sample(range(len(raw_data)), limit))
                else:
                    sample_indices = range(len(raw_data))

                entries = []
                n_failed = 0
                n_too_many_steps = 0
                n_with_catalyst = 0
                for entry_idx in sample_indices:
                    entry = raw_data[entry_idx]
                    if entry.get("success") is False:
                        n_failed += 1
                        continue
                    mechanisms = entry.get("mechanisms", [])
                    if mechanisms and mechanisms[0].get("steps"):
                        num_steps = len(mechanisms[0]["steps"])
                    else:
                        num_steps = entry.get("num_steps", 1)
                    if num_steps > 4:
                        n_too_many_steps += 1
                        continue
                    if entry.get("metadata") and entry["metadata"].get("cat"):
                        n_with_catalyst += 1
                    entries.append(entry)
                del raw_data
                self._n_entries_with_cat = n_with_catalyst
                self._n_entries_total = len(entries)
                print(
                    f"Entries: {len(entries)} kept, {n_failed} failed, {n_too_many_steps} skipped (>4 steps), {n_with_catalyst} with catalyst"
                )

                slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
                max_cpus = int(slurm_cpus) if slurm_cpus else 32
                n_workers = min(16, max_cpus)
                chunksize = max(64, len(entries) // (n_workers * 4))
                print(
                    f"Processing {len(entries)} entries (format: {data_version}, workers: {n_workers}, chunksize: {chunksize})..."
                )
                results = process_map(
                    process_fn,
                    entries,
                    max_workers=n_workers,
                    chunksize=chunksize,
                    desc="Processing",
                )
                for samples in results:
                    self.data.extend(samples)

                print(f"Saving processed data to cache: {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(self.data, f)
            else:
                print(f"Warning: {json_path} not found.")

        actives = sum(1 for d in self.data if "step" in d["type"])
        stables = sum(1 for d in self.data if "product" in d["type"])
        trajectories = sum(1 for d in self.data if d.get("type") == "trajectory")
        n_cat_samples = sum(1 for d in self.data if d.get("has_catalyst", False))
        print(
            f"Total: {len(self.data)}. Actives: {actives}, Stables: {stables}, Trajectories: {trajectories}, With catalyst: {n_cat_samples}"
        )

        n_entries_with_cat = getattr(self, "_n_entries_with_cat", None)
        if n_entries_with_cat is not None and n_entries_with_cat > 0:
            if n_cat_samples == 0:
                raise RuntimeError(
                    f"Dataset has {n_entries_with_cat} entries with catalysts but 0 samples "
                    f"retained has_catalyst=True. Catalysts are being silently dropped!"
                )
            n_entries_total = getattr(self, "_n_entries_total", len(self.data))
            cat_retention = n_cat_samples / len(self.data)
            entry_cat_ratio = n_entries_with_cat / n_entries_total
            print(
                f"Catalyst check: {n_entries_with_cat} entries had catalysts, "
                f"{n_cat_samples}/{len(self.data)} samples ({cat_retention:.1%}) retained catalyst"
            )
            if cat_retention < entry_cat_ratio * 0.5:
                warnings.warn(
                    f"Catalyst retention suspiciously low: only {cat_retention:.1%} of samples "
                    f"have has_catalyst=True, but {entry_cat_ratio:.1%} of entries had catalysts. "
                    f"Check catalyst SMILES parsing!",
                    stacklevel=2,
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return _trajectory_numpy_to_tensors(self.data[idx])

class ShardedReactionDataset(ReactionDataset):
    """Memory-efficient variant of ReactionDataset.

    Processes entries in streaming batches and writes shards directly to disk,
    never holding more than one batch + one shard buffer in memory.
    Supports resuming interrupted builds via periodic checkpoints.
    """

    _BUILD_BATCH_SIZE = 100_000
    _CHECKPOINT_INTERVAL = 10

    def __init__(
        self,
        json_path,
        include_product=True,
        limit=None,
        force_recalculate_cache=False,
        extended_features=False,
        data_version="v2_graph",
        use_all_mechanisms=False,
        enable_history=False,
        shard_size=SHARD_SAMPLE_LIMIT,
        canonical_rank=False,
        enable_blockspace_product=True,
        sample_mode="step",
    ):
        self.shard_size = shard_size
        self.extended_features = extended_features
        self.data_version = data_version
        self.use_all_mechanisms = use_all_mechanisms
        self.enable_history = enable_history
        self.canonical_rank = canonical_rank
        self.sample_mode = sample_mode

        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        config_str = (
            f"{_dataset_cache_prefix(json_path)}_"
            f"{limit}_{include_product}_{data_version}"
        )
        if extended_features:
            config_str += "_extended"
        if use_all_mechanisms:
            config_str += "_all_mechs"
        if enable_history:
            config_str += "_historyv2"
        if canonical_rank:
            config_str += "_canonrank"
        if data_version == "v3_blockspace" and include_product:
            config_str += "_blockprod" if enable_blockspace_product else "_rxnprod"
        if sample_mode != "step":
            config_str += f"_{sample_mode}"
        config_str += f"_shard{shard_size}"
        self._shard_dir = os.path.join(cache_dir, config_str)
        meta_file = os.path.join(self._shard_dir, "meta.pkl")

        self._shard_cache: dict[int, list] = {}
        self._shard_cache_order: list[int] = []
        self._max_cached_shards = 1
        self._index_map: list[int] | None = None

        if os.path.exists(meta_file) and not force_recalculate_cache:
            print(f"Loading sharded dataset for {os.path.basename(json_path)}...")
            with open(meta_file, "rb") as f:
                meta_data = pickle.load(f)
            # Support metadata-only cache files and shard-size metadata files.
            if isinstance(meta_data, dict):
                self.metadata = [
                    _coerce_sample_metadata(m) for m in meta_data["metadata"]
                ]
                self.shard_size = meta_data["shard_size"]
            else:
                self.metadata = [_coerce_sample_metadata(m) for m in meta_data]
                # Detect shard size from existing shard files when metadata lacks it.
                detected = None
                for si in [1, 0]:
                    sp = os.path.join(self._shard_dir, f"shard_{si}.pkl")
                    if os.path.exists(sp):
                        with open(sp, "rb") as f:
                            detected = len(pickle.load(f))
                        break
                if detected:
                    self.shard_size = detected
                    print(f"  Detected shard_size={self.shard_size} from shard files")
                else:
                    print(
                        f"  WARNING: Could not detect shard_size, using config value {self.shard_size}"
                    )
        else:
            self._build_shards_streaming(
                json_path,
                include_product,
                limit,
                extended_features,
                data_version,
                use_all_mechanisms,
                enable_history,
                meta_file,
                canonical_rank=canonical_rank,
                enable_blockspace_product=enable_blockspace_product,
                sample_mode=sample_mode,
            )

        actives = sum(1 for m in self.metadata if "step" in m["type"])
        stables = sum(1 for m in self.metadata if "product" in m["type"])
        trajectories = sum(1 for m in self.metadata if m.get("type") == "trajectory")
        negatives = sum(1 for m in self.metadata if m["type"] == "negative_reaction")
        n_cat_samples = sum(1 for m in self.metadata if m.get("has_catalyst", False))
        print(
            f"Total: {len(self.metadata)}. Actives: {actives}, Stables: {stables}, Trajectories: {trajectories}, Negatives: {negatives}, With catalyst: {n_cat_samples}"
        )

        n_entries_with_cat = getattr(self, "_n_entries_with_cat", None)
        if n_entries_with_cat is not None and n_entries_with_cat > 0:
            if n_cat_samples == 0:
                raise RuntimeError(
                    f"Dataset has {n_entries_with_cat} entries with catalysts but 0 samples "
                    f"retained has_catalyst=True. Catalysts are being silently dropped!"
                )
            n_entries_total = getattr(self, "_n_entries_total", len(self.metadata))
            cat_retention = n_cat_samples / len(self.metadata)
            entry_cat_ratio = n_entries_with_cat / n_entries_total
            print(
                f"Catalyst check: {n_entries_with_cat} entries had catalysts, "
                f"{n_cat_samples}/{len(self.metadata)} samples ({cat_retention:.1%}) retained catalyst"
            )
            if cat_retention < entry_cat_ratio * 0.5:
                warnings.warn(
                    f"Catalyst retention suspiciously low: only {cat_retention:.1%} of samples "
                    f"have has_catalyst=True, but {entry_cat_ratio:.1%} of entries had catalysts. "
                    f"Check catalyst SMILES parsing!",
                    stacklevel=2,
                )

    def _build_shards_streaming(
        self,
        json_path,
        include_product,
        limit,
        extended_features,
        data_version,
        use_all_mechanisms,
        enable_history,
        meta_file,
        canonical_rank=False,
        enable_blockspace_product=True,
        sample_mode="step",
    ):
        """Process entries in batches and stream shards to disk.

        For JSONL inputs, shuffle lightweight file offsets instead of full entry
        dicts, then stream worker results straight into shard writes. This keeps
        peak memory close to one shard buffer plus compact metadata.
        """
        import random

        os.makedirs(self._shard_dir, exist_ok=True)
        checkpoint_file = os.path.join(self._shard_dir, "_build_checkpoint.pkl")
        entry_order_file = os.path.join(self._shard_dir, "_entry_order.pkl")

        if data_version == "v3_blockspace":
            process_entry_fn = (
                process_reaction_entry_v3_trajectory
                if sample_mode == "trajectory"
                else process_reaction_entry_v3
            )
            process_fn = partial(
                process_entry_fn,
                include_product=include_product,
                extended_features=extended_features,
                use_all_mechanisms=use_all_mechanisms,
                canonical_rank=canonical_rank,
                enable_blockspace_product=enable_blockspace_product,
            )
            if sample_mode != "trajectory":
                process_fn = partial(process_fn, enable_history=enable_history)
        else:
            process_fn = partial(
                process_reaction_entry,
                include_product=include_product,
                extended_features=extended_features,
                enable_history=enable_history,
            )

        # Load and filter entries; JSONL inputs are streamed by file offset.
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            self.metadata = []
            return

        def _is_valid_entry(entry):
            if entry.get("success") is False:
                return False
            mechanisms = entry.get("mechanisms", [])
            if mechanisms and mechanisms[0].get("steps"):
                num_steps = len(mechanisms[0]["steps"])
            else:
                num_steps = entry.get("num_steps", 1)
            return num_steps <= 4

        def _dedupe_cached_string(cache: dict[str, str], value: str) -> str:
            if not value:
                return ""
            cached = cache.get(value)
            if cached is None:
                cache[value] = value
                return value
            return cached

        if json_path.endswith(".jsonl"):
            if os.path.exists(entry_order_file):
                load_t0 = time.perf_counter()
                print(f"Loading shuffled JSONL order from {entry_order_file}...")
                with open(entry_order_file, "rb") as f:
                    entries = pickle.load(f)
                print(
                    f"Loaded {len(entries)} shuffled JSONL offsets in {time.perf_counter() - load_t0:.1f}s"
                )
            else:
                scan_t0 = time.perf_counter()
                entries = []
                total_bytes = os.path.getsize(json_path)
                with open(json_path, "rb") as f:
                    with tqdm(
                        total=total_bytes,
                        unit="B",
                        unit_scale=True,
                        desc="Indexing JSONL",
                    ) as pbar:
                        while True:
                            offset = f.tell()
                            raw_line = f.readline()
                            if not raw_line:
                                break
                            pbar.update(len(raw_line))
                            line = raw_line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            if _is_valid_entry(entry):
                                entries.append(offset)
                print(
                    f"Indexed {len(entries)} valid JSONL entries in {time.perf_counter() - scan_t0:.1f}s"
                )
                if limit and limit < len(entries):
                    sample_t0 = time.perf_counter()
                    entries = random.sample(entries, limit)
                    print(
                        f"Sampled {len(entries)} JSONL entries in {time.perf_counter() - sample_t0:.1f}s"
                    )
                shuffle_t0 = time.perf_counter()
                random.shuffle(entries)
                print(
                    f"Shuffled {len(entries)} JSONL entries in {time.perf_counter() - shuffle_t0:.1f}s"
                )
                save_t0 = time.perf_counter()
                with open(entry_order_file, "wb") as f:
                    pickle.dump(entries, f)
                print(
                    f"Saved shuffled JSONL order in {time.perf_counter() - save_t0:.1f}s"
                )
            self._n_entries_with_cat = 0
            self._n_entries_total = len(entries)
        else:
            with open(json_path, "r") as f:
                raw_data = json.load(f)
            if limit and limit < len(raw_data):
                sample_indices = sorted(random.sample(range(len(raw_data)), limit))
            else:
                sample_indices = range(len(raw_data))
            entries = [
                raw_data[i] for i in sample_indices if _is_valid_entry(raw_data[i])
            ]
            del raw_data
            # Shuffle entries before sharding to reduce correlations from source ordering.
            random.shuffle(entries)

            # Count catalyst-containing entries for retention validation.
            n_entries_with_cat = sum(
                1 for e in entries if e.get("metadata") and e["metadata"].get("cat")
            )
            self._n_entries_with_cat = n_entries_with_cat
            self._n_entries_total = len(entries)

        # Resume an interrupted cache build when a checkpoint is available.
        entries_processed = 0
        n_shards = 0
        self.metadata = []
        sample_buffer = []
        type_cache = {"product": "product", "negative_reaction": "negative_reaction"}
        reactant_smi_cache = {"": ""}

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                ckpt = pickle.load(f)
            entries_processed = ckpt["entries_processed"]
            n_shards = ckpt["n_shards"]
            self.metadata = [_coerce_sample_metadata(m) for m in ckpt["metadata"]]
            if "n_entries_with_cat" in ckpt:
                self._n_entries_with_cat = ckpt["n_entries_with_cat"]
            print(
                f"Resuming build from checkpoint: {entries_processed}/{len(entries)} entries, {n_shards} shards written"
            )
            for meta in self.metadata:
                type_cache.setdefault(meta.type, meta.type)
                if meta.reactant_smi:
                    reactant_smi_cache.setdefault(meta.reactant_smi, meta.reactant_smi)

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        max_cpus = int(slurm_cpus) if slurm_cpus else 32
        n_workers = min(16, max_cpus)
        batch_size = self._BUILD_BATCH_SIZE

        print(
            f"Processing {len(entries)} entries (format: {data_version}, workers: {n_workers}, streaming shards of {self.shard_size})..."
        )

        with mp.Pool(processes=n_workers) as pool:
            for batch_start in range(entries_processed, len(entries), batch_size):
                batch_end = min(batch_start + batch_size, len(entries))
                batch_len = batch_end - batch_start
                chunksize = max(64, batch_len // max(1, n_workers * 4))

                if json_path.endswith(".jsonl"):
                    batch_offsets = entries[batch_start:batch_end]

                    def _iter_batch_entries():
                        with open(json_path, "rb") as f:
                            with tqdm(
                                batch_offsets,
                                total=batch_len,
                                desc=f"Reading [{batch_start}:{batch_end}/{len(entries)}]",
                                leave=False,
                            ) as read_pbar:
                                for offset in read_pbar:
                                    f.seek(offset)
                                    entry = json.loads(f.readline())
                                    if (
                                        entry.get("metadata")
                                        and entry["metadata"].get("cat")
                                    ):
                                        self._n_entries_with_cat += 1
                                    yield entry

                    batch_iter = _iter_batch_entries()
                else:
                    batch_iter = iter(entries[batch_start:batch_end])

                results_iter = pool.imap(process_fn, batch_iter, chunksize=chunksize)

                for samples in tqdm(
                    results_iter,
                    total=batch_len,
                    desc=f"Processing [{batch_start}:{batch_end}/{len(entries)}]",
                ):
                    for sample in samples:
                        sample_type = _dedupe_cached_string(
                            type_cache, sample.get("type", "")
                        )
                        reactant_smi = _dedupe_cached_string(
                            reactant_smi_cache, sample.get("reactant_smi", "")
                        )
                        self.metadata.append(
                            SampleMetadata(
                                type=sample_type,
                                has_catalyst=sample.get("has_catalyst", False),
                                step_index=sample.get("step_index", 0),
                                reactant_smi=reactant_smi,
                            )
                        )
                        sample_buffer.append(sample)

                        if len(sample_buffer) >= self.shard_size:
                            self._write_shard(n_shards, sample_buffer[: self.shard_size])
                            sample_buffer = sample_buffer[self.shard_size :]
                            n_shards += 1

                            if n_shards % self._CHECKPOINT_INTERVAL == 0:
                                with open(checkpoint_file, "wb") as f:
                                    pickle.dump(
                                        {
                                            "entries_processed": batch_end,
                                            "n_shards": n_shards,
                                            "metadata": self.metadata,
                                            "n_entries_with_cat": getattr(
                                                self, "_n_entries_with_cat", 0
                                            ),
                                        },
                                        f,
                                    )

                # Persist checkpoint metadata after each processed batch.
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(
                        {
                            "entries_processed": batch_end,
                            "n_shards": n_shards,
                            "metadata": self.metadata,
                            "n_entries_with_cat": getattr(
                                self, "_n_entries_with_cat", 0
                            ),
                        },
                        f,
                    )

        # Write any samples that did not fill a complete shard.
        if sample_buffer:
            self._write_shard(n_shards, sample_buffer)
            n_shards += 1
            sample_buffer = []

        # Save metadata with shard size so cache loading uses the correct value.
        with open(meta_file, "wb") as f:
            pickle.dump({"metadata": self.metadata, "shard_size": self.shard_size}, f)

        # Remove build checkpoint files after a completed build.
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if os.path.exists(entry_order_file):
            os.remove(entry_order_file)

        print(f"Saved {len(self.metadata)} samples across {n_shards} shards")

    def _write_shard(self, shard_idx, samples):
        path = os.path.join(self._shard_dir, f"shard_{shard_idx}.pkl")
        with open(path, "wb") as f:
            pickle.dump(samples, f)

    def _load_shard(self, shard_idx):
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]

        path = os.path.join(self._shard_dir, f"shard_{shard_idx}.pkl")
        with open(path, "rb") as f:
            samples = pickle.load(f)

        if len(self._shard_cache_order) >= self._max_cached_shards:
            evict = self._shard_cache_order.pop(0)
            del self._shard_cache[evict]

        self._shard_cache[shard_idx] = samples
        self._shard_cache_order.append(shard_idx)
        return samples

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self._index_map is not None:
            idx = self._index_map[idx]
        shard_idx = idx // self.shard_size
        offset = idx % self.shard_size
        shard = self._load_shard(shard_idx)
        if offset >= len(shard):
            raise IndexError(
                f"Sample {idx}: shard {shard_idx} has {len(shard)} samples "
                f"but offset is {offset} (shard_size={self.shard_size}, "
                f"total={len(self.metadata)}). Cache may have been built with "
                f"a different shard_size — rebuild with force_recalculate_cache=true"
            )
        return _trajectory_numpy_to_tensors(shard[offset])

    @property
    def n_shards(self):
        if self._index_map is not None:
            return len({i // self.shard_size for i in self._index_map})
        return (len(self.metadata) + self.shard_size - 1) // self.shard_size

    def storage_index(self, idx: int) -> int:
        """Map logical dataset index to the underlying cached sample index."""
        if self._index_map is not None:
            return self._index_map[idx]
        return idx

    def subsample_products(self, retain_fraction: float, seed: int = 42):
        """Retain all non-product samples and only a fraction of product samples.

        This physically shortens the logical dataset, so epochs become faster.
        The underlying shard cache remains unchanged; ``_index_map`` redirects
        logical indices to the original cached sample positions.
        """
        if not (0.0 < retain_fraction <= 1.0):
            raise ValueError(
                f"retain_fraction must be in (0, 1], got {retain_fraction}"
            )
        if retain_fraction >= 1.0:
            return

        import random

        keep_non_product = []
        product_indices = []
        for idx, meta in enumerate(self.metadata):
            if meta.get("type") == "product":
                product_indices.append(idx)
            else:
                keep_non_product.append(idx)

        if not product_indices:
            return

        rng = random.Random(seed)
        keep_products_n = max(1, int(round(len(product_indices) * retain_fraction)))
        keep_products = set(
            rng.sample(product_indices, min(keep_products_n, len(product_indices)))
        )
        keep_indices = sorted(keep_non_product + list(keep_products))

        original_total = len(self.metadata)
        original_products = len(product_indices)
        self._index_map = keep_indices
        self.metadata = [self.metadata[i] for i in keep_indices]

        kept_products = sum(1 for m in self.metadata if m.get("type") == "product")
        print(
            f"Product subsample: kept {kept_products}/{original_products} product samples "
            f"({kept_products / max(1, original_products):.1%}); dataset {len(self.metadata)}/{original_total}"
        )

class ShardAwareSampler(torch.utils.data.Sampler):
    """Weighted sampler that groups indices by shard to avoid disk thrashing.

    Shuffles shard order each epoch, then within each shard samples indices
    according to the provided weights. This gives the same weighted distribution
    as WeightedRandomSampler but with sequential shard access.
    """

    def __init__(
        self,
        dataset: ShardedReactionDataset,
        weights: list[float],
        num_samples: int | None = None,
        generator=None,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = num_samples or len(weights)
        self.generator = generator
        self.seed = seed
        self.epoch = 0
        self.shard_size = dataset.shard_size
        self.n_shards = dataset.n_shards
        self.n_total = len(dataset)

    def __iter__(self):
        generator = self.generator
        if generator is None and self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)

        sampled = weighted_shard_aware_sample(
            self.dataset,
            self.weights,
            self.num_samples,
            generator=generator,
            shard_size=self.shard_size,
        )
        return iter(sampled)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples
