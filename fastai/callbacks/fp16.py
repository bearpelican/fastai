"Callback support for half precision (fp16) training. Increases training speed."
from ..torch_core import *
from ..callback import *
from ..basic_train import *
from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector

__all__ = ['MixedPrecision']

def get_master(layer_groups:ModuleList, flat_master:bool=False) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_groups = split_bn_bias(layer_groups)
    model_params = [[param for param in lg.parameters() if param.requires_grad] for lg in split_groups]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) !=0 :
                mp = parameters_to_vector([param.data.float() for param in lg])
                mp = torch.nn.Parameter(mp, requires_grad=True)
                if mp.grad is None: mp.grad = mp.new(*mp.size())
                master_params.append([mp])
            else: master_params.append([])
        return model_params, master_params
    else:
        master_params = [[param.clone().float().detach() for param in lg] for lg in model_params]
        for mp in master_params:
            for param in mp: param.requires_grad = True
        return model_params, master_params

def model_g2master_g(model_params:Sequence[Tensor], master_params:Sequence[Tensor], flat_master:bool=False)->None:
    "Copy the `model_params` gradients to `master_params` for the optimizer step."
    if flat_master:
        for model_group,master_group in zip(model_params,master_params):
            if len(master_group) != 0:
                if master_group[0].grad is None: master_group[0].grad = master_group[0].data.new(*master_group[0].data.size())
                master_group[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_group]))
    else:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None: master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else: master.grad = None

def master2model(model_params:Sequence[Tensor], master_params:Sequence[Tensor], flat_master:bool=False)->None:
    "Copy `master_params` to `model_params`."
    if flat_master:
        for model_group,master_group in zip(model_params,master_params):
            if len(model_group) != 0:
                for model, master in zip(model_group, _unflatten_dense_tensors(master_group[0].data, model_group)):
                    model.data.copy_(master)
    else:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, master_group): model.data.copy_(master.data)


class DynamicLossScaler:
    def __init__(self, init_scale=2**10, scale_factor=2., scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if 
            # Pytorch's .sum() creates a one-element tensor of the same type as x 
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class MixedPrecision(LearnerCallback):
    "Callback that handles mixed-precision training."
    def __init__(self, learn:Learner, loss_scale:float=512., flat_master:bool=False):
        super().__init__(learn)
        self.loss_scale,self.flat_master = loss_scale,flat_master
        self.loss_scaler = DynamicLossScaler()
        self.not_min += ['model_params', 'master_params']
        print('Mixed precision')
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."

    def on_train_begin(self, **kwargs:Any)->None:
        "Ensure everything is in half precision mode."
        self.learn.data.train_dl.add_tfm(batch_to_half)
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl.add_tfm(batch_to_half)
        if hasattr(self.learn.data, 'test_dl') and self.learn.data.test_dl is not None:
            self.learn.data.test_dl.add_tfm(batch_to_half)
        #Get a copy of the model params in FP32
        self.model_params, self.master_params = get_master(self.learn.layer_groups, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        opt = self.learn.opt
        mom,wd,beta = opt.mom,opt.wd,opt.beta
        lrs = [lr for lr in self.learn.opt._lr for _ in range(2)]
        opt_params = [{'params': mp, 'lr': lr} for mp,lr in zip(self.master_params, lrs)]
        self.learn.opt.opt = self.learn.opt_func(opt_params)
        opt.mom,opt.wd,opt.beta = mom,wd,beta

    def on_loss_begin(self, last_output:Tensor, **kwargs:Any) -> Tensor:
        "Convert half precision output to FP32 to avoid reduction overflow."
        return to_float(last_output)

    def on_backward_begin(self, last_loss:Rank0Tensor, **kwargs:Any) -> Rank0Tensor:
        "Scale gradients up by `self.loss_scale` to prevent underflow."
        #To avoid gradient underflow, we scale the gradients
        ret_loss = last_loss * self.loss_scaler.loss_scale
#        if torch.isnan(ret_loss): 
#            print(f"You have a `loss_scale` factor that is too high, try to divide it by 2 (current value: {self.loss_scaler.loss_scale}).")
        return ret_loss

    def on_backward_end(self, **kwargs:Any ):
        "Convert the gradients back to FP32 and divide them by the scale."
        params = []
        for group in self.model_params:
            for param in group:
                params.append(param)
        has_overflow = self.loss_scaler.has_overflow(params)
        if has_overflow:
            print(f'fp16 overflow. skipping batch. Scale: {self.loss_scaler.loss_scale}')
            self.learn.model.zero_grad()
            self.learn.opt.zero_grad()
        else:
            model_g2master_g(self.model_params, self.master_params, self.flat_master)
            for group in self.master_params:
                for param in group: 
                    if param.grad is not None: param.grad.div_(self.loss_scaler.loss_scale)
        self.loss_scaler.update_scale(has_overflow)

    def on_step_end(self, **kwargs:Any)->None:
        "Update the params from master to model and zero grad."
        #Zeros the gradients of the model since the optimizer is disconnected.
        self.learn.model.zero_grad()
        #Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)
