import torch

class VRAdam(torch.optim.Optimizer):
    """Implements VRAdam algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): maximal learning rate (alpha_0 in paper)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float): term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        alpha1 (float): minimal learning rate control (alpha_1 in paper) (default: 0.0)
        beta3 (float): velocity penalizer (beta_3 in paper) (default: 1.0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, alpha1=0.0, beta3=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha1:
            raise ValueError("Invalid alpha1 value: {}".format(alpha1))
        if not 0.0 <= beta3:
            raise ValueError("Invalid beta3 value: {}".format(beta3))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha1=alpha1, beta3=beta3)
        super(VRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(VRAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            v = []
            m = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    v.append(state['v'])
                    m.append(state['m'])
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            alpha0 = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            alpha1 = group['alpha1']
            beta3 = group['beta3']

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                state_v = v[i]
                state_m = m[i]
                step = state_steps[i]

                step += 1

                # Update biased first moment estimate
                state_v.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                state_m.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Calculate dynamic learning rate
                # Using the norm of the velocity estimate ||v_t||^2
                v_norm_sq = torch.sum(state_v * state_v)
                dynamic_lr = alpha0 / (1 + min(beta3 * v_norm_sq.item(), alpha1))

                # Bias correction for first moment
                v_hat = state_v / (1 - beta1 ** step)

                # Bias correction for second moment
                m_hat = state_m / (1 - beta2 ** step)

                # Denominator for update
                denom = m_hat.sqrt().add_(eps)

                # Update parameters with weight decay
                # θ_t = θ_{t-1} - η_t * ( (1 - η_t * λ) * v̂_t / (√m̂_t + ε) + η_t * λ * θ_{t-1} )
                # Rearranging for implementation:
                # param.data = param.data - dynamic_lr * ( (1 - dynamic_lr * weight_decay) * v_hat / denom + dynamic_lr * weight_decay * param.data )
                # param.data = param.data * (1 - dynamic_lr * dynamic_lr * weight_decay) - dynamic_lr * (1 - dynamic_lr * weight_decay) * v_hat / denom
                # Let's stick closer to the paper's formula structure for clarity:

                # Term 1: (1 - η_t * λ) * v̂_t / (√m̂_t + ε)
                term1 = (1 - dynamic_lr * weight_decay) * v_hat / denom

                # Term 2: η_t * λ * θ_{t-1}
                term2 = dynamic_lr * weight_decay * param.data

                # Full update step
                param.data.add_(-(dynamic_lr * (term1 + term2)))

                # Update state step counter
                state['step'] = step

        return loss