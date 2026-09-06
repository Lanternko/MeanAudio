import hashlib
import logging
import random
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from meanaudio.utils.dist_utils import local_rank
import numpy as np
import glob
import torch.nn.functional as F
log = logging.getLogger()


class ExtractedAudio(Dataset):
    def __init__(
        self,
        tsv_path: Union[str, Path],
        *,
        concat_text_fc: bool,
        npz_dir: Union[str, Path],
        data_dim: dict[str, int],
        text_npz_dir: Optional[Union[str, Path]] = None,
        repa_npz_dir: Optional[Union[str, Path]],   # if passed, repa features (zs) would be returned
        exclude_cls: Optional[bool],
        repa_version: Optional[int],
        gt_cache: Optional[Union[str, Path]] = None,
        require_text_overlay: bool = False,
        multi_cap: bool = False,   # if True, NPZ stores N captions [N, seq_len, dim]; random one picked per __getitem__
        cap_index_fixed: Optional[int] = None,    # reuse a stacked overlay by always taking this caption slot
        cap_index_column: Optional[str] = None,   # reuse a stacked overlay by taking a per-row slot from the TSV
        text_npz_sources: Optional[list] = None,  # multi_cap over slots assembled from several existing overlays
        use_text_attention_mask: bool = True,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.df_list = pd.read_csv(tsv_path, sep='\t').to_dict('records') # id, caption
        self.ids = [str(d['id']) for d in self.df_list]
        if gt_cache is not None:
            with open(gt_cache, 'r') as _f:
                npz_files = [line.strip() for line in _f if line.strip()]
            log.info(f'Loaded NPZ list from gt_cache: {len(npz_files)} files')
        else:
            npz_files = [f'{i}.npz' for i in range(len(self.df_list))]
            log.info(f'Using sequential NPZ indices: {len(npz_files)} files')
        self.npz_files = npz_files
        self.concat_text_fc = concat_text_fc
        self.exclude_cls = exclude_cls
        self.repa_version = repa_version
        self.multi_cap = multi_cap
        self.use_text_attention_mask = use_text_attention_mask
        self.epoch = 0
        self.caption_seed = 14159265
        self.text_npz_dir = Path(text_npz_dir) if text_npz_dir is not None else None
        self.require_text_overlay = require_text_overlay
        if self.require_text_overlay and self.text_npz_dir is None:
            raise ValueError('require_text_overlay=True but text_npz_dir is not configured')
        if self.require_text_overlay and len(self.npz_files) != len(self.df_list):
            raise ValueError(
                f'overlay binding count mismatch: cache={len(self.npz_files)} tsv={len(self.df_list)}'
            )

        # Single-caption arms can reuse a stacked multi-caption overlay instead of
        # re-encoding an identical copy: pick one slot out of [N, seq_len, dim].
        self.cap_indices: Optional[list[int]] = None
        if cap_index_fixed is not None and cap_index_column is not None:
            raise ValueError('cap_index_fixed and cap_index_column are mutually exclusive')
        if (cap_index_fixed is not None or cap_index_column is not None) and self.multi_cap:
            raise ValueError('cap_index_fixed / cap_index_column cannot be combined with multi_cap=True')
        if cap_index_fixed is not None:
            if int(cap_index_fixed) < 0:
                raise ValueError(f'cap_index_fixed must be non-negative, got {cap_index_fixed}')
            self.cap_indices = [int(cap_index_fixed)] * len(self.df_list)
            log.info(f'cap_index_fixed={int(cap_index_fixed)}: reusing one slot of a stacked text overlay')
        elif cap_index_column is not None:
            if cap_index_column not in self.df_list[0]:
                raise ValueError(f'cap_index_column {cap_index_column!r} is not a column of {tsv_path}')
            indices = [int(row[cap_index_column]) for row in self.df_list]
            if min(indices) < 0:
                raise ValueError(f'cap_index_column {cap_index_column!r} holds a negative index')
            self.cap_indices = indices
            log.info(
                f'cap_index_column={cap_index_column!r}: reusing per-row slots of a stacked text overlay '
                f'(range {min(indices)}..{max(indices)})'
            )

        # A rotation arm over a slot set that was never encoded as one stack: assemble
        # the caption pool from overlays that already exist instead of re-encoding a
        # duplicate 225 GB copy. Each source is {dir: <path>, index: <slot|null>};
        # index=null means that directory holds single-caption overlays.
        self.text_sources: Optional[list[tuple[Path, Optional[int]]]] = None
        if text_npz_sources is not None:
            if not self.multi_cap:
                raise ValueError('text_npz_sources requires multi_cap=True')
            if self.cap_indices is not None:
                raise ValueError('text_npz_sources cannot be combined with cap_index_fixed / cap_index_column')
            sources = []
            for spec in text_npz_sources:
                if isinstance(spec, (str, Path)):
                    spec = {'dir': str(spec), 'index': None}
                source_dir = Path(str(spec['dir']))
                if not source_dir.is_dir():
                    raise ValueError(f'text_npz_sources directory does not exist: {source_dir}')
                slot = spec.get('index', None)
                sources.append((source_dir, None if slot is None else int(slot)))
            if len(sources) < 2:
                raise ValueError('text_npz_sources needs at least two caption sources')
            self.text_sources = sources
            log.info(
                'text_npz_sources: rotating over %d captions assembled from %s',
                len(sources),
                ', '.join(f'{d.name}[{i}]' if i is not None else d.name for d, i in sources),
            )

        if self.concat_text_fc:
            log.info(f'We will concat the pooled text_features and text_features_c for text condition')
        if self.multi_cap:
            log.info(f'multi_cap=True: caption randomly sampled per __getitem__')

        # dimension check（使用 npz_files[0] 而非 hardcode 0.npz，相容 gt_cache 不含 0.npz 的情況）
        if not npz_files:
            raise FileNotFoundError(f'No NPZ files found in {npz_dir}')
        sample = np.load(f'{npz_dir}/{npz_files[0]}')
        text_sample = (
            np.load(self.text_npz_dir / npz_files[0])
            if self.text_npz_dir is not None else sample
        )
        self.text_attention_mask_key = None
        if self.use_text_attention_mask:
            for mask_key in ('text_attention_mask', 'attention_mask'):
                if mask_key in text_sample.files:
                    self.text_attention_mask_key = mask_key
                    break
        mean_s = [len(self.df_list)] + list(sample['mean'].shape)
        std_s = [len(self.df_list)] + list(sample['std'].shape)
        # multi_cap: text_features shape is [N, seq_len, dim]; use last two dims for check
        text_features_s = [len(self.df_list)] + list(text_sample['text_features'].shape[-2:])
        text_features_c_s = [len(self.df_list)] + list(text_sample['text_features_c'].shape[-1:])
        if self.concat_text_fc:
            text_features_c_s[-1] = text_features_c_s[-1] + text_features_s[-1]

        log.info(f'Loading {len(npz_files)} npz files from {npz_dir}')
        log.info(f'Loaded mean: {mean_s}.')
        log.info(f'Loaded std: {std_s}.')
        log.info(f'Loaded text features: {text_features_s}.')
        log.info(f'Loaded text features_c: {text_features_c_s}.') 
        if not self.use_text_attention_mask:
            log.info('Text attention masks disabled; reproducing the legacy all-77-token path.')
        elif self.text_attention_mask_key is not None:
            text_attention_mask_s = [len(self.df_list)] + list(text_sample[self.text_attention_mask_key].shape[-1:])
            log.info(f'Loaded text attention masks: {text_attention_mask_s}.')
        else:
            log.info('No text attention mask found in NPZ; treating all text positions as valid.')

        # assert len(npz_files) == len(self.df_list), 'Number mismatch between npz files and tsv items'
        assert mean_s[1] == self.data_dim['latent_seq_len'], \
            f'{mean_s[1]} != {self.data_dim["latent_seq_len"]}'
        assert std_s[1] == self.data_dim['latent_seq_len'], \
            f'{std_s[1]} != {self.data_dim["latent_seq_len"]}'
        assert text_features_s[1] == self.data_dim['text_seq_len'], \
            f'{text_features_s[1]} != {self.data_dim["text_seq_len"]}'
        assert text_features_s[-1] == self.data_dim['text_dim'], \
            f'{text_features_s[-1]} != {self.data_dim["text_dim"]}'
        assert text_features_c_s[-1] == self.data_dim['text_c_dim'], \
            f'{text_features_c_s[-1]} != {self.data_dim["text_c_dim"]}'
    
        self.npz_dir = npz_dir
        if repa_npz_dir != None: 
            self.repa_npz_dir = repa_npz_dir
            sample = np.load(f'{repa_npz_dir}/0.npz')
            repa_npz_files = glob.glob(f"{repa_npz_dir}/*.npz")
            log.info(f'Loading {len(repa_npz_files)} npz representations from {repa_npz_dir}')
            es_s = [len(repa_npz_files)] + list(sample['es'].shape)
            if self.repa_version == 2: 
                es_s[1] = 65  # ad-hoc 8x downsampling for EAT 
            elif self.repa_version == 3: 
                es_s[1] = 1   # we only use cls token for alignment 
            else: 
                if self.exclude_cls: 
                    es_s[1] = es_s[1] - 1

            log.info(f'Loaded es: {es_s}')
            assert len(repa_npz_files) == len(npz_files), 'Number mismatch between repa npz files and latent npz files'
            assert es_s[1] == self.data_dim['repa_seq_len'], \
                f'{es_s[1]} != {self.data_dim["repa_seq_len"]}'
            assert es_s[-1] == self.data_dim['repa_seq_dim'], \
                f'{es_s[-1]} != {self.data_dim["repa_seq_dim"]}'
        else: 
            self.repa_npz_dir = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _true_random_cap_index(self, clip_id: str, n_caps: int) -> int:
        payload = f"k3-true-random-v1\0{self.caption_seed}\0{self.epoch}\0{clip_id}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_caps

    @staticmethod
    def _stored_caption_hashes(value: np.ndarray) -> list[str]:
        """Overlay files store caption_sha256 as a 0-d comma-joined string; accept 1-D too."""
        if value.ndim == 0:
            return str(value.item()).split(',')
        return [str(item) for item in value.tolist()]

    def _composed_caption(self, idx: int):
        """Rotate over captions held in several overlays, one slot taken per source."""
        name = self.npz_files[idx]
        expected_id = str(self.df_list[idx]['id'])
        loaded = []
        shas = []
        for source_dir, slot in self.text_sources:
            data = np.load(source_dir / name)
            if 'clip_id' not in data.files or str(data['clip_id'].item()) != expected_id:
                raise ValueError(
                    f'text overlay clip_id mismatch at index {idx} in {source_dir}: expected {expected_id}'
                )
            stored = self._stored_caption_hashes(data['caption_sha256'])
            if slot is None:
                if len(stored) != 1:
                    raise ValueError(
                        f'{source_dir} holds {len(stored)} stacked captions at index {idx}; a slot index is required'
                    )
                shas.append(stored[0])
            else:
                if slot >= len(stored):
                    raise ValueError(
                        f'slot {slot} out of range for {len(stored)} stacked captions in {source_dir} at index {idx}'
                    )
                shas.append(stored[slot])
            loaded.append(data)
        if self.require_text_overlay:
            row_sha = hashlib.sha256(str(self.df_list[idx]['caption']).encode('utf-8')).hexdigest()
            if row_sha not in shas:
                raise ValueError(
                    f'text overlay caption mismatch at index {idx}: TSV caption is not among the '
                    f'{len(shas)} composed captions'
                )
        cap_idx = self._true_random_cap_index(expected_id, len(self.text_sources))
        data = loaded[cap_idx]
        slot = self.text_sources[cap_idx][1]
        take = (lambda array: array) if slot is None else (lambda array: array[slot])
        text_features = torch.from_numpy(take(data['text_features']))
        text_features_c = torch.from_numpy(take(data['text_features_c']))
        if self.text_attention_mask_key is not None and self.text_attention_mask_key in data.files:
            text_attention_mask = torch.from_numpy(take(data[self.text_attention_mask_key])).bool()
        else:
            text_attention_mask = torch.ones(text_features.shape[0], dtype=torch.bool)
        return text_features, text_features_c, text_attention_mask

    def _check_caption_binding(self, text_np_data, idx: int, cap_idx: Optional[int]) -> None:
        """Guard the caption<->overlay pairing that silently broke the Phase 9 multi-cap runs."""
        if not self.require_text_overlay or 'caption_sha256' not in text_np_data.files:
            return
        stored = self._stored_caption_hashes(text_np_data['caption_sha256'])
        row_sha = hashlib.sha256(str(self.df_list[idx]['caption']).encode('utf-8')).hexdigest()
        if cap_idx is None:
            # multi_cap picks a slot per epoch, so only require membership.
            if row_sha not in stored:
                raise ValueError(
                    f'text overlay caption mismatch at index {idx}: TSV caption is not among the {len(stored)} stacked captions'
                )
            return
        if cap_idx >= len(stored) or stored[cap_idx] != row_sha:
            raise ValueError(
                f'text overlay caption mismatch at index {idx}: slot {cap_idx} of {len(stored)} does not match the TSV caption'
            )

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        # !TODO here we may consider load pre-computed latent mean & std
        raise NotImplementedError('Please manually compute latent stats outside. ')
    
    def __getitem__(self, idx):
        npz_path = f'{self.npz_dir}/{self.npz_files[idx]}'
        np_data = np.load(npz_path)
        text_np_data = (
            np.load(self.text_npz_dir / self.npz_files[idx])
            if self.text_npz_dir is not None else np_data
        )
        if self.require_text_overlay:
            expected_id = str(self.df_list[idx]['id'])
            if 'clip_id' not in np_data.files or str(np_data['clip_id'].item()) != expected_id:
                raise ValueError(f'audio clip_id mismatch at index {idx}: expected {expected_id}')
            if 'clip_id' not in text_np_data.files or str(text_np_data['clip_id'].item()) != expected_id:
                raise ValueError(f'text overlay clip_id mismatch at index {idx}: expected {expected_id}')
        if self.text_sources is not None:
            text_features, text_features_c, text_attention_mask = self._composed_caption(idx)
        elif self.multi_cap:
            # text_features: [N, seq_len, dim], text_features_c: [N, dim]
            n_caps = text_np_data['text_features'].shape[0]
            cap_idx = self._true_random_cap_index(str(self.df_list[idx]['id']), n_caps)
            self._check_caption_binding(text_np_data, idx, None)
            text_features = torch.from_numpy(text_np_data['text_features'][cap_idx])
            text_features_c = torch.from_numpy(text_np_data['text_features_c'][cap_idx])
            if self.text_attention_mask_key is not None and self.text_attention_mask_key in text_np_data.files:
                text_attention_mask = torch.from_numpy(text_np_data[self.text_attention_mask_key][cap_idx]).bool()
            else:
                text_attention_mask = torch.ones(text_features.shape[0], dtype=torch.bool)
        elif self.cap_indices is not None:
            # Stacked overlay reused by a single-caption arm: take exactly one slot.
            stacked = text_np_data['text_features']
            if stacked.ndim != 3:
                raise ValueError(
                    f'cap index selection needs a stacked overlay [N, seq_len, dim], got {stacked.shape} at index {idx}'
                )
            cap_idx = self.cap_indices[idx]
            if cap_idx >= stacked.shape[0]:
                raise ValueError(
                    f'cap index {cap_idx} out of range for {stacked.shape[0]} stacked captions at index {idx}'
                )
            self._check_caption_binding(text_np_data, idx, cap_idx)
            text_features = torch.from_numpy(stacked[cap_idx])
            text_features_c = torch.from_numpy(text_np_data['text_features_c'][cap_idx])
            if self.text_attention_mask_key is not None and self.text_attention_mask_key in text_np_data.files:
                text_attention_mask = torch.from_numpy(text_np_data[self.text_attention_mask_key][cap_idx]).bool()
            else:
                text_attention_mask = torch.ones(text_features.shape[0], dtype=torch.bool)
        else:
            self._check_caption_binding(text_np_data, idx, None if text_np_data['text_features'].ndim == 3 else 0)
            text_features = torch.from_numpy(text_np_data['text_features'])
            text_features_c = torch.from_numpy(text_np_data['text_features_c'])
            if self.text_attention_mask_key is not None and self.text_attention_mask_key in text_np_data.files:
                text_attention_mask = torch.from_numpy(text_np_data[self.text_attention_mask_key]).bool()
            else:
                text_attention_mask = torch.ones(text_features.shape[0], dtype=torch.bool)
        if self.concat_text_fc:
            mask = text_attention_mask.to(dtype=text_features.dtype).unsqueeze(-1)
            text_features_mean = (text_features * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
            text_features_c = torch.cat([text_features_mean, text_features_c], dim=-1)   # [b, d+d_c]

        q_level = int(self.df_list[idx]['q_level']) if 'q_level' in self.df_list[idx] else 9
        out_dict = {
            'id': str(self.df_list[idx]['id']),
            'a_mean': torch.from_numpy(np_data['mean']), 
            'a_std': torch.from_numpy(np_data['std']), 
            'text_features': text_features, 
            'text_features_c': text_features_c,
            'caption': self.df_list[idx]['caption'],
            'q_level': torch.tensor(q_level, dtype=torch.long),
        }
        if self.use_text_attention_mask:
            out_dict['text_attention_mask'] = text_attention_mask
        if self.repa_npz_dir != None: 
            repa_npz_path = f'{self.repa_npz_dir}/{idx}.npz'
            repa_np_data = np.load(repa_npz_path)
            zs =  torch.from_numpy(repa_np_data['es'])   

            if self.repa_version == 1: 
                if self.exclude_cls: 
                    zs = zs[1:,:]
            if self.repa_version == 2: 
                z_cls = zs[0]  # (dim)
                # zs = zs[1:,:].view(64, 8, 768)  
                zs = F.avg_pool2d(zs[1:,:].unsqueeze(0), 
                                  kernel_size=(8, 1), 
                                  stride=(8, 1)).squeeze()  # (64, 768)
                zs = torch.cat((z_cls.unsqueeze(0), zs), dim=0)
            elif self.repa_version == 3:  # cls token
                zs = zs[0].unsqueeze(0)
                
            out_dict['zs'] = zs  #!TODO Here field is WRONG for eat features (should be zs)

        return out_dict

    def __len__(self):
        return len(self.ids)
    

if __name__ == '__main__': 

    from meanaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
    import torch.distributed as distributed
    from datetime import timedelta
    from torch.utils.data.distributed import DistributedSampler


    def distributed_setup():
        distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
        return local_rank, world_size

    distributed_setup()

    tsv_path = '/hpc_stor03/sjtu_home/xiquan.li/TTA/MMAudio/training/audiocaps/train-memmap-t5-clap.tsv'

    data_dim = {'latent_seq_len': 312, 
                'text_seq_len': 77,
                'text_dim': 1024, 
                'text_c_dim': 512}

    dataset = ExtractedAudio(tsv_path=tsv_path,
                                    npz_dir=npz_dir,
                                    data_dim=data_dim)
    loader = DataLoader(dataset,
                        16,
                        num_workers=8,
                        persistent_workers=8,
                        pin_memory=False)
    train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=True)


    for b in loader: 
        print(b['a_mean'].shape)
        break
