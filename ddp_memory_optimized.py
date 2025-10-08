# case_study_ddp_optimized.py
"""
Memory-Optimized DDP Training for 8B models on 4x L4 GPUs (24GB each)

KEY OPTIMIZATIONS:
1. 8-bit quantization (compatible with DDP!)
2. Gradient checkpointing (saves 40-50% activation memory)
3. 8-bit AdamW optimizer (reduces optimizer states by 4x)
4. Reduced sequence length and batch size
5. Proper memory cleanup

Usage:
    torchrun --nproc_per_node=4 case_study_ddp_optimized.py
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup,
    logging as transformers_logging,
    BitsAndBytesConfig
)

# Import 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAS_8BIT_OPTIMIZER = True
except ImportError:
    HAS_8BIT_OPTIMIZER = False
    logging.warning("bitsandbytes not found - falling back to regular AdamW")

warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


@dataclass
class ExperimentConfig:
    """Memory-optimized configuration for 4x L4 GPUs"""
    # Model configuration - OPTIMIZED FOR MEMORY
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"  # or your 8B model
    dataset_name: str = "OpenCoder-LLM/opc-sft-stage2"
    dataset_config: str = "educational_instruct"
    
    # CRITICAL: Enable 8-bit quantization (DDP compatible!)
    use_8bit_quantization: bool = True
    use_gradient_checkpointing: bool = True  # NEW: Saves 40-50% memory
    
    # Training configuration - REDUCED for memory
    max_steps: int = 400
    eval_every_n_steps: int = 40
    batch_size_per_gpu: int = 1  # REDUCED from 2
    gradient_accumulation_steps: int = 4  # Effective batch = 1*4*4GPUs = 16
    learning_rate: float = 5e-5
    warmup_steps: int = 50
    max_seq_length: int = 384  # REDUCED from 512
    seconds_per_step: float = 12.0
    
    # Optimizer configuration
    use_8bit_optimizer: bool = True  # NEW: Saves 4x optimizer memory
    
    # Fault injection configuration
    fault_injection_step: int = 200
    fault_type: str = "BIT_FLIP_GRADUAL"
    fault_params: Dict[str, Any] = field(default_factory=lambda: {
        "bit_flip_rate": 1e-6,
        "bit_flip_acceleration": 1.1,
        "bit_flip_affected_layers": 0.3,
    })
    fault_duration: int = 50
    terminal_failure_step: Optional[int] = None
    
    # R-Metric configuration
    r_metric_alert_threshold: float = 0.57
    r_metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda": 0.10,
        "sigma_sq": 0.45,
        "delta_l": 0.70
    })
    
    percentile_alpha: float = 0.3
    signal_scaling_factor: float = 4.0
    
    loss_history_window: int = 10
    gradient_history_window: int = 20
    hardware_event_window: int = 50
    percentile_history_window: int = 100
    
    early_detection_window: int = 50
    timely_detection_window: int = 20
    false_positive_tolerance: int = 3
    
    gpu_cost_per_hour: float = 4.00
    num_gpus: int = 4
    
    output_dir: str = "case_study_results"
    experiment_name: str = f"8b_rmetric_ddp_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    device: str = "cuda"
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.terminal_failure_step is None:
            self.terminal_failure_step = self.fault_injection_step + self.fault_duration
        
        self.output_path = Path(self.output_dir) / self.experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)


# [Keep all monitoring classes the same as original]
class MetricTracker:
    def __init__(self, window_size: int, percentile_window: int = 100):
        self.window_size = window_size
        self.percentile_window = percentile_window
        self.history = deque(maxlen=window_size)
        self.percentile_history = deque(maxlen=percentile_window)
        self.all_values = []
    
    def update(self, value: float):
        if not np.isnan(value) and not np.isinf(value):
            self.history.append(value)
            self.percentile_history.append(value)
            self.all_values.append(value)
    
    def get_percentile_rank(self, value: float) -> float:
        if len(self.percentile_history) < 2:
            return 0.5
        ranks = np.sum(np.array(self.percentile_history) <= value)
        return ranks / len(self.percentile_history)
    
    def get_mean(self) -> float:
        return np.mean(list(self.history)) if self.history else 0.0


class RMetricSignalProcessor:
    def __init__(self, alpha: float = 0.3, scaling: float = 4.0):
        self.alpha = alpha
        self.scaling = scaling
        self.smoothed_value = 0.5
    
    def process(self, percentile_rank: float) -> float:
        ema = self.alpha * percentile_rank + (1 - self.alpha) * self.smoothed_value
        centered_scaled = self.scaling * (ema - 0.5)
        transformed = 1.0 / (1.0 + np.exp(-centered_scaled))
        self.smoothed_value = ema
        return transformed


class ReliabilityMonitor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.loss_tracker = MetricTracker(config.loss_history_window, config.percentile_history_window)
        self.gradient_tracker = MetricTracker(config.gradient_history_window, config.percentile_history_window)
        self.hardware_tracker = MetricTracker(config.hardware_event_window, config.percentile_history_window)
        
        self.lambda_processor = RMetricSignalProcessor(config.percentile_alpha, config.signal_scaling_factor)
        self.sigma_processor = RMetricSignalProcessor(config.percentile_alpha, config.signal_scaling_factor)
        self.delta_l_processor = RMetricSignalProcessor(config.percentile_alpha, config.signal_scaling_factor)
    
    def calculate_lambda(self, hardware_event_count: int, time_window_seconds: float = 60.0) -> float:
        if time_window_seconds == 0:
            return 0.0
        return (hardware_event_count / time_window_seconds) * 3600.0
    
    def calculate_sigma_sq(self, gradients: List[torch.Tensor]) -> float:
        if not gradients:
            return 0.0
        grad_norms = [grad.norm().item() for grad in gradients if grad is not None 
                     and not np.isnan(grad.norm().item()) and not np.isinf(grad.norm().item())]
        return np.var(grad_norms) if len(grad_norms) > 1 else 0.0
    
    def calculate_delta_l(self, current_loss: float) -> float:
        if np.isnan(current_loss) or np.isinf(current_loss):
            return 10.0
        if len(self.loss_tracker.history) < 2:
            self.loss_tracker.update(current_loss)
            return 0.0
        drift = abs(current_loss - self.loss_tracker.get_mean())
        self.loss_tracker.update(current_loss)
        return drift
    
    def calculate_r_metric(self, current_loss: float, gradients: Optional[List[torch.Tensor]], 
                          hardware_event_count: int) -> Dict[str, float]:
        lambda_raw = self.calculate_lambda(hardware_event_count)
        sigma_sq_raw = self.calculate_sigma_sq(gradients) if gradients else 0.0
        delta_l_raw = self.calculate_delta_l(current_loss)
        
        self.hardware_tracker.update(lambda_raw)
        self.gradient_tracker.update(sigma_sq_raw)
        
        lambda_percentile = self.hardware_tracker.get_percentile_rank(lambda_raw)
        sigma_percentile = self.gradient_tracker.get_percentile_rank(sigma_sq_raw)
        delta_l_percentile = self.loss_tracker.get_percentile_rank(delta_l_raw)
        
        lambda_processed = self.lambda_processor.process(lambda_percentile)
        sigma_processed = self.sigma_processor.process(sigma_percentile)
        delta_l_processed = self.delta_l_processor.process(delta_l_percentile)
        
        weights = self.config.r_metric_weights
        r_metric = (weights["lambda"] * lambda_processed + weights["sigma_sq"] * sigma_processed + 
                   weights["delta_l"] * delta_l_processed)
        
        return {
            'r_metric': r_metric, 'lambda_raw': lambda_raw, 'sigma_sq_raw': sigma_sq_raw,
            'delta_l_raw': delta_l_raw, 'lambda_percentile': lambda_percentile,
            'sigma_percentile': sigma_percentile, 'delta_l_percentile': delta_l_percentile,
            'lambda_norm': lambda_processed, 'sigma_sq_norm': sigma_processed,
            'delta_l_norm': delta_l_processed
        }


class FaultInjector:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fault_active = False
        self.fault_start_step = None
        self.fault_state = {}
    
    def should_inject(self, step: int) -> bool:
        return step == self.config.fault_injection_step
    
    def inject_fault(self, optimizer: torch.optim.Optimizer, model: nn.Module, step: int):
        if self.config.fault_type == "BIT_FLIP_GRADUAL":
            self._inject_bit_flip_gradual(model, step)
    
    def _inject_bit_flip_gradual(self, model: nn.Module, step: int):
        if not self.fault_active:
            self.fault_active = True
            self.fault_start_step = step
            logger.warning(f"⚠️  FAULT INJECTED at step {step}: Gradual bit flips")
        
        steps_since = step - self.fault_start_step
        rate = self.config.fault_params['bit_flip_rate'] * (self.config.fault_params['bit_flip_acceleration'] ** steps_since)
        
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad and np.random.random() < rate:
                    flip_mask = torch.rand_like(param) < 0.01
                    param.data[flip_mask] += torch.randn_like(param)[flip_mask] * 0.01
    
    def should_recover(self, step: int) -> bool:
        return self.fault_active and step >= self.fault_start_step + self.config.fault_duration
    
    def recover_fault(self, optimizer: torch.optim.Optimizer):
        self.fault_active = False


class BaselineMonitors:
    def __init__(self, config: ExperimentConfig):
        self.consecutive_loss_increases = 0
        self.grad_norm_history = deque(maxlen=20)
        self.grad_norm_threshold = 100.0
    
    def update_simple_heuristic(self, current_loss: float, prev_loss: float) -> bool:
        if current_loss > prev_loss:
            self.consecutive_loss_increases += 1
        else:
            self.consecutive_loss_increases = 0
        return self.consecutive_loss_increases >= 3
    
    def update_gradient_monitoring(self, gradients: List[torch.Tensor]) -> bool:
        if not gradients:
            return False
        grad_norms = [g.norm().item() for g in gradients if g is not None 
                     and not np.isnan(g.norm().item()) and not np.isinf(g.norm().item())]
        if not grad_norms:
            return False
        max_norm = max(grad_norms)
        self.grad_norm_history.append(max_norm)
        return max_norm > self.grad_norm_threshold


class DDPCaseStudyTrainer:
    """Memory-optimized DDP Trainer"""
    
    def __init__(self, config: ExperimentConfig, rank: int, world_size: int, local_rank: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.is_main_process = (rank == 0)
        
        if self.is_main_process:
            self.setup_logging()
        
        self.setup_model_and_data()
        self.setup_monitoring()
        
        self.results = {
            "config": config.__dict__,
            "metrics": [],
            "alerts": {"r_metric": None, "simple_heuristic": None, "gradient_monitoring": None},
            "alert_counts": {"r_metric": 0, "simple_heuristic": 0, "gradient_monitoring": 0}
        }
    
    def setup_logging(self):
        log_file = self.config.output_path / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info("="*80)
        logger.info(f"MEMORY-OPTIMIZED DDP: {self.config.experiment_name}")
        logger.info(f"World Size: {self.world_size} GPUs")
        logger.info("="*80)
    
    def setup_model_and_data(self):
        """Setup model with 8-bit quantization + gradient checkpointing"""
        if self.is_main_process:
            logger.info(f"Loading model: {self.config.model_name}")
            logger.info("🔧 Optimizations: 8-bit quantization + gradient checkpointing")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # CRITICAL: 8-bit quantization config (DDP compatible!)
        quantization_config = None
        if self.config.use_8bit_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if not self.config.use_8bit_quantization else None,
            use_cache=False,  # Required for gradient checkpointing
            device_map={'': self.local_rank}  # Specific GPU mapping for DDP
        )
        
        # Enable gradient checkpointing (saves 40-50% activation memory)
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_main_process:
                logger.info("✓ Gradient checkpointing enabled")
        
        # Prepare model for k-bit training (required for 8-bit with DDP)
        if self.config.use_8bit_quantization:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False
        )
        
        # Dataset loading
        if self.is_main_process:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split='train'
        ).shuffle(seed=42).select(range(2000))
        
        def tokenize_function(examples):
            texts = [f"Instruction: {inst}\nOutput: {out}" 
                    for inst, out in zip(examples['instruction'], examples['output'])]
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # DDP sampler
        self.train_sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=42
        )
        
        self.train_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.config.batch_size_per_gpu,
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Optimizer: Use 8-bit AdamW if available (saves 4x memory)
        if self.config.use_8bit_optimizer and HAS_8BIT_OPTIMIZER:
            self.optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            if self.is_main_process:
                logger.info("✓ Using 8-bit AdamW optimizer")
        else:
            from torch.optim import AdamW
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )
        
        # Memory report
        if self.is_main_process:
            allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3
            reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3
            logger.info(f"GPU {self.local_rank} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def setup_monitoring(self):
        self.reliability_monitor = ReliabilityMonitor(self.config)
        self.baseline_monitors = BaselineMonitors(self.config)
        self.fault_injector = FaultInjector(self.config)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[torch.Tensor]]:
        """Training step with gradient accumulation"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        outputs = self.model(**batch, labels=batch["input_ids"])
        loss = outputs.loss / self.config.gradient_accumulation_steps
        
        loss.backward()
        
        # Collect gradients
        gradients = []
        for param in self.model.module.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if not np.isnan(grad_norm) and not np.isinf(grad_norm):
                    gradients.append(torch.tensor([grad_norm]))
        
        return loss.item() * self.config.gradient_accumulation_steps, gradients
    
    def evaluate_step(self, step: int, train_loss: float, gradients: List[torch.Tensor]):
        """Evaluation - only main process"""
        if not self.is_main_process:
            return
        
        r_metric_results = self.reliability_monitor.calculate_r_metric(train_loss, gradients, 0)
        
        heuristic_alert = False
        if len(self.results["metrics"]) > 0:
            prev_loss = self.results["metrics"][-1]["train_loss"]
            heuristic_alert = self.baseline_monitors.update_simple_heuristic(train_loss, prev_loss)
        
        grad_alert = self.baseline_monitors.update_gradient_monitoring(gradients)
        r_metric_alert = r_metric_results["r_metric"] > self.config.r_metric_alert_threshold
        
        if r_metric_alert:
            self.results["alert_counts"]["r_metric"] += 1
            if self.results["alerts"]["r_metric"] is None:
                self.results["alerts"]["r_metric"] = step
                logger.warning(f"🚨 R-METRIC ALERT at step {step}")
        
        metrics = {
            "step": step,
            "train_loss": train_loss,
            **r_metric_results,
            "heuristic_alert": heuristic_alert,
            "grad_alert": grad_alert,
            "r_metric_alert": r_metric_alert
        }
        self.results["metrics"].append(metrics)
    
    def train(self):
        """Main training loop with gradient accumulation"""
        if self.is_main_process:
            logger.info("Starting memory-optimized DDP training...")
            progress_bar = tqdm(range(self.config.max_steps), desc="Training")
        
        self.model.train()
        data_iterator = iter(self.train_dataloader)
        accumulated_loss = 0.0
        all_gradients = []
        
        for step in range(self.config.max_steps):
            # Gradient accumulation loop
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    self.train_sampler.set_epoch(step)
                    data_iterator = iter(self.train_dataloader)
                    batch = next(data_iterator)
                
                loss, gradients = self.train_step(batch)
                accumulated_loss += loss
                all_gradients.extend(gradients)
            
            # Gradient step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Fault injection (main process only)
            if self.is_main_process:
                if self.fault_injector.should_inject(step):
                    self.fault_injector.inject_fault(self.optimizer, self.model.module, step)
                if self.fault_injector.should_recover(step):
                    self.fault_injector.recover_fault(self.optimizer)
            
            # Logging
            avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
            if self.is_main_process:
                progress_bar.update(1)
                progress_bar.set_description(f"Step {step} | Loss: {avg_loss:.3f}")
                
                if step % self.config.eval_every_n_steps == 0 and step > 0:
                    self.evaluate_step(step, avg_loss, all_gradients)
            
            # Reset accumulators
            accumulated_loss = 0.0
            all_gradients = []
            
            # Memory cleanup
            if step % 10 == 0:
                torch.cuda.empty_cache()
        
        if self.is_main_process:
            progress_bar.close()
            logger.info("✓ Training completed!")
    
    def save_results(self):
        """Save results"""
        if not self.is_main_process:
            return
        
        df = pd.DataFrame(self.results["metrics"])
        csv_path = self.config.output_path / "metrics.csv"
        df.to_csv(csv_path, index=False)
        
        json_path = self.config.output_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"✓ Results saved to {self.config.output_path}")


def main():
    """Main execution"""
    rank, world_size, local_rank = setup_distributed()
    
    config = ExperimentConfig()
    
    trainer = DDPCaseStudyTrainer(config, rank, world_size, local_rank)
    trainer.train()
    trainer.save_results()
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
