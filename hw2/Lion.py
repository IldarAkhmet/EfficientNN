import torch 
from torch.optim import Optimizer 


class Lion(Optimizer): 
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4):
        if lr < 0.0: 
            raise ValueError(f"learning rate must be more than 0")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None 
        if closure is not None: 
            with torch.enable_grad():
                loss = closure() 

        for group in self.param_groups: 
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None: 
                    continue 

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                momentum = state["momentum_buffer"]

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                update = (
                    beta1 * momentum + (1 - beta1) * grad
                )

                p.add_(update.sign(), alpha=-lr)

                momentum.mul_(beta2).add_(
                    grad, 
                    alpha=1 - beta2,
                )

        return loss




