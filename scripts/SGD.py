import torch

class ImprovedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.8, weight_decay=5e-4):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(ImprovedSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                p.data.add_(-group['lr'], d_p)