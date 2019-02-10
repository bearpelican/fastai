from .torch_core import *
from .basic_train import Learner,LearnerCallback
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

__all__ = ['DistributedRecorder', 'DistributedTrainer', 'read_metrics', 'setup_distrib']

def make_async(b:Tuple[Tensor,Tensor]):
    return [o.to(o.device, non_blocking=True) for o in b]

class ParallelTrainer(LearnerCallback):
    _order = -20
    def on_train_begin(self, **kwargs): self.learn.model = DataParallel(self.learn.model)
    def on_train_end  (self, **kwargs): self.learn.model = self.learn.model.module

class DDP(DistributedDataParallel):
      # Distributed wrapper. Supports asynchronous evaluation and model saving
    def forward(self, *args, **kwargs):
        # DDP has a sync point on forward. No need to do this for eval. This allows us to have different batch sizes
        if self.training: return super().forward(*args, **kwargs)
        else:             return self.module(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.module.load_state_dict(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()
    
class DistributedTrainer(LearnerCallback):
    _order = -20 #Needs to run before the recorder
    def __init__(self, learn:Learner, cuda_id:int=0, drop_last:bool=False, shuffle:bool=True):
        super().__init__(learn)
        self.cuda_id = cuda_id
        self.train_sampler = None
        self.drop_last,self.shuffle = drop_last, shuffle

    def on_train_begin(self, **kwargs):
        self.learn.model = DDP(self.learn.model, device_ids=[self.cuda_id],
                                                   output_device=self.cuda_id)
        self.train_sampler = LMDistributedSampler(self.learn.data.train_dl.dataset, shuffle=self.shuffle)
        self.learn.data.train_dl = self.learn.data.train_dl.new(shuffle=False, sampler=self.train_sampler, drop_last=self.drop_last)
        self.learn.data.train_dl.add_tfm(make_async)
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.valid_sampler = DistributedSampler(self.learn.data.valid_dl.dataset)
            self.learn.data.valid_dl = self.learn.data.valid_dl.new(shuffle=False, sampler=self.valid_sampler)
            self.learn.data.valid_dl.add_tfm(make_async)
        self.rank = rank_distrib()
        self.learn.recorder.silent = (self.rank != 0)

    def on_epoch_begin(self, epoch, **kwargs): 
        if hasattr(self.train_sampler, 'set_epoch'): self.train_sampler.set_epoch(epoch)

    def on_train_end(self, **kwargs):
        self.learn.model = self.learn.model.module
        self.learn.data.train_dl.remove_tfm(make_async)
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl.remove_tfm(make_async)

class DistributedRecorder(LearnerCallback):
    def __init__(self, learn:Learner, cuda_id:int=0, cache_dir:PathOrStr='tmp'):
        super().__init__(learn)
        self.cuda_id,self.cache_dir = cuda_id,cache_dir

    def on_train_begin(self, **kwargs):
        os.makedirs(self.learn.path/self.cache_dir, exist_ok=True)

    def on_epoch_end(self, **kwargs): self.save_stats()
    def on_train_end(self, **kwargs): self.save_stats()

    def save_stats(self):
        cache_path,recorder = self.learn.path/self.cache_dir,self.learn.recorder
        np.save(cache_path/f'losses_{self.cuda_id}', np.array(recorder.losses))
        stats = np.array([[v] + m for v,m in zip(recorder.val_losses,recorder.metrics)])
        np.save(cache_path/f'metrics_{self.cuda_id}', stats)

def _learner_parallel(learn:Learner):
    "Use nn.DataParallel when training and remove when done"
    learn.callbacks.append(ParallelTrainer(learn))
    return learn

def _learner_distributed(learn:Learner, cuda_id:int, cache_dir:PathOrStr='tmp', drop_last=False, shuffle=True):
    "Put `learn` on distributed training with `cuda_id`."
    learn.callbacks.append(DistributedTrainer(learn, cuda_id, drop_last, shuffle))
    learn.callbacks.append(DistributedRecorder(learn, cuda_id, cache_dir))
    return learn

Learner.to_distributed = _learner_distributed
Learner.to_parallel = _learner_parallel

def read_metrics(cache_path:PathOrStr, n_gpus:int, reduce:bool=True):
    losses,metrics = [],[]
    for i in range(n_gpus):
        losses.append(np.load(cache_path/f'losses_{i}.npy')[None])
        metrics.append(np.load(cache_path/f'metrics_{i}.npy')[None])
    if reduce:
        losses,metrics = np.concatenate(losses,0),np.concatenate(metrics,0)
        return losses.mean(0),metrics.mean(0)
    return losses,metrics

def setup_distrib(gpu:Any=None):
    if gpu is None: return gpu
    gpu = int(gpu)
    torch.cuda.set_device(int(gpu))
    if num_distrib() > 1:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return gpu


class LMDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            np.random.seed(self.epoch)
            indices = torch.arange(len(self.dataset)).tolist()
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
