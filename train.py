import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import logging
import math
import random
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record

from meanaudio.data.data_setup import setup_training_datasets, setup_val_datasets
from meanaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from meanaudio.runner_flowmatching import RunnerFlowMatching
from meanaudio.runner_meanflow import RunnerMeanFlow
from meanaudio.sample import sample
from meanaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from meanaudio.utils.logger import TensorboardLogger
from meanaudio.utils.synthesize_ema import synthesize_ema
from meanaudio.utils.training_resume import resume_coordinates
import os
import time
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


def _pid_start_time(pid: int):
    try:
        with open(f'/proc/{pid}/stat', encoding='utf-8') as fh:
            return fh.read().split()[21]
    except Exception:
        return None


def _write_pause_ack(request_path: str, run_dir: str, exp_id: str, curr_iter: int) -> None:
    request = Path(request_path)
    checkpoint = Path(run_dir) / f'{exp_id}_ckpt_last.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f'pause checkpoint missing or empty: {checkpoint}')
    payload = {
        'status': 'paused',
        'iteration': curr_iter,
        'checkpoint': str(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'pid': os.getpid(),
        'start_time': _pid_start_time(os.getpid()),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    try:
        req = json.loads(request.read_text(encoding='utf-8'))
        for key in ('run_id', 'job_id', 'request_id'):
            if key in req:
                payload[key] = req[key]
    except Exception:
        pass
    if request.name == 'pause.request.json':
        ack_path = request.with_name('pause.ack.json')
    else:
        ack_path = Path(f'{request_path}.ack.json')
    tmp = ack_path.with_name(f'.{ack_path.name}.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(payload, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, ack_path)


@record
@hydra.main(version_base='1.3.2', config_path='config', config_name='train_config.yaml')
def train(cfg: DictConfig):
    
    # debug setting
    if cfg.get("debug", False): 
        import debugpy
        if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
            debugpy.listen(6665) 
            print(f'Waiting for debugger attach (rank {os.environ["RANK"]})...')
            debugpy.wait_for_client()  

    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # patch data dim
    seq_cfg = CONFIG_16K  # we only support 16k for now
    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len  # update sequence config here

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(cfg.exp_id,
                            run_dir,
                            logging.getLogger(),
                            is_rank0=(local_rank == 0),
                            enable_email=cfg.enable_email and not cfg.debug)

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')

    # number of dataloader workers
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Training configuration: {cfg}')
    cfg.batch_size //= num_gpus
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')  

    # determine time to change max skip
    total_iterations = cfg['num_iterations']

    # setup datasets
    if cfg['text_encoder_name'] == 't5_clap_cat': 
        cfg['concat_text_fc'] = True

    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f'Number of training samples: {len(dataset)}')
    info_if_rank_zero(log, f'Number of training batches: {len(loader)}')

    val_dataset, val_loader, eval_loader = setup_val_datasets(cfg)  # same dataset (val_dataset) but with different dataloader
    info_if_rank_zero(log, f'Number of val samples: {len(val_dataset)}')
    val_cfg = cfg.data.AudioCaps_val_npz  # tsv and memmap dir 

    # compute and set mean and std
    latent_mean, latent_std = torch.load(cfg.data.latent_mean), torch.load(cfg.data.latent_std)

    # construct the trainer
    if not cfg.use_repa: 
        if cfg.use_meanflow: 
            trainer = RunnerMeanFlow(cfg,
                                     log=log,
                                     run_path=run_dir,
                                     for_training=True,
                                     latent_mean=latent_mean,
                                     latent_std=latent_std).enter_train()
        else:
            trainer = RunnerFlowMatching(cfg,
                                         log=log,
                                         run_path=run_dir,
                                         for_training=True,
                                         latent_mean=latent_mean,
                                         latent_std=latent_std).enter_train()
                    
    else: 
        raise NotImplementedError('REPA is not supported yet')
        trainer = RunnerAT_REPA(cfg,
                                log=log,
                                run_path=run_dir,
                                for_training=True,
                                latent_mean=latent_mean,
                                latent_std=latent_std).enter_train()

    eval_rng_clone = trainer.rng.graphsafe_get_state()

    # load previous checkpoint if needed
    if cfg['checkpoint'] is not None:
        curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
        cfg['checkpoint'] = None
        info_if_rank_zero(log, 'Model checkpoint loaded!')
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
            info_if_rank_zero(log, 'Latest checkpoint loaded!')
        else:
            # load previous network weights if needed
            curr_iter = 0
            if cfg['weights'] is not None:
                info_if_rank_zero(log, 'Loading weights from the disk')
                trainer.load_weights(cfg['weights'])
                cfg['weights'] = None
            else: 
                info_if_rank_zero(log, 'No checkpoint or weights found, starting from scratch')

    # determine max epoch
    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch, resume_batch_offset = resume_coordinates(curr_iter, len(loader))
    pause_request_file = cfg.get('pause_request_file')
    paused = False
    info_if_rank_zero(log, f'We will approximately use {total_epoch - current_epoch} epochs.')

    # training loop
    try:
        # Need this to select random bases in different workers
        np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
        while curr_iter < total_iterations:
            # Crucial for randomness!
            sampler.set_epoch(current_epoch)  # guarantee each epoch has different shuffling
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(current_epoch)
            current_epoch += 1
            log.debug(f'Current epoch: {current_epoch}')

            trainer.enter_train()
            trainer.log.data_timer.start()
            for batch_index, data in enumerate(loader):
                if batch_index < resume_batch_offset:
                    continue
                trainer.train_pass(data, curr_iter)

                if (curr_iter + 1) % cfg.val_interval == 0:  
                    # swap into a eval rng state, i.e., use the same seed for every validation pass
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f'Iteration {curr_iter}: validating')
                    total_loss = 0
                    n = 0
                    if cfg.use_repa: 
                        total_diff_loss = 0
                        total_proj_loss = 0 
                    for data in tqdm(val_loader):
                        n += 1
                        if not cfg.use_repa: 
                            mean_loss = trainer.validation_pass(data, curr_iter) 
                            total_loss += mean_loss
                        else: 
                            mean_loss, diff_loss, proj_loss =  trainer.validation_pass(data, curr_iter) 
                            total_loss += mean_loss
                            total_diff_loss += diff_loss
                            total_proj_loss += proj_loss

                    total_loss /= n
                    if cfg.use_repa: 
                        total_diff_loss /= n
                        total_proj_loss /= n
                    if cfg.use_wandb and local_rank == 0: 
                        wandb.log({"val/loss": total_loss})
                        if cfg.use_repa: 
                            wandb.log({"val/diff_loss": total_diff_loss}, step=curr_iter)
                            wandb.log({"val/proj_loss": total_proj_loss}, step=curr_iter)

                    distributed.barrier()
                    trainer.val_integrator.finalize('val', curr_iter, ignore_timer=True)
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)

                if (curr_iter + 1) % cfg.eval_interval == 0:
                    save_eval = (curr_iter + 1) % cfg.save_eval_interval == 0
                    train_rng_snapshot = trainer.rng.graphsafe_get_state()
                    trainer.rng.graphsafe_set_state(eval_rng_clone)
                    info_if_rank_zero(log, f'Iteration {curr_iter}: inference')
                    for data in tqdm(eval_loader):
                        audio_path = trainer.inference_pass(data,
                                                            curr_iter,
                                                            val_cfg,  
                                                            save_eval=save_eval)  # path to audio files generated
                    distributed.barrier()
                    trainer.rng.graphsafe_set_state(train_rng_snapshot)
                    trainer.eval(audio_path, curr_iter, val_cfg)   # av-bench eval

                curr_iter += 1

                if pause_request_file and Path(str(pause_request_file)).exists():
                    paused = True
                    info_if_rank_zero(
                        log,
                        f'Cooperative pause requested after iteration {curr_iter}',
                    )
                    break

                if curr_iter >= total_iterations:
                    break

            resume_batch_offset = 0
            if paused:
                break

    except Exception as e:
        log.error(f'Error occurred at iteration {curr_iter}!')
        log.critical(e.message if hasattr(e, 'message') else str(e))
        raise
    finally:
        if not cfg.debug:
            trainer.save_checkpoint(curr_iter)  # finally will always be called
            trainer.save_weights(curr_iter)

    if paused:
        distributed.barrier()
        if local_rank == 0:
            _write_pause_ack(str(pause_request_file), run_dir, cfg.exp_id, curr_iter)
        distributed.barrier()
        info_if_rank_zero(log, f'Training paused safely at iteration {curr_iter}')
        distributed.destroy_process_group()
        return

    # Inference pass
    del trainer
    torch.cuda.empty_cache()

    # Synthesize EMA
    if local_rank == 0:
        log.info(f'Synthesizing EMA with sigma={cfg.ema.default_output_sigma}')
        ema_sigma = cfg.ema.default_output_sigma
        state_dict = synthesize_ema(cfg, ema_sigma, step=None)
        save_dir = Path(run_dir) / f'{cfg.exp_id}_ema_final.pth'
        torch.save(state_dict, save_dir)
        log.info(f'Synthesized EMA saved to {save_dir}!')
    distributed.barrier()

    log.info(f'Evaluation: {cfg}')  
    # # sample(cfg)  
 
    # clean-up
    log.complete()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    train()
