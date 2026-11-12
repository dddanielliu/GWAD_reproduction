import torch
from ....base import SoftLabelAttackBase

"""
Andriushchenko et al.
Square Attack: a Query-Efficient Black-Box Adversarial Attack via Random Search
European Conference on Computer Vision (ECCV), 2020

l_inf version only (Algorithm 1, Algorithm 2 and the vertical stripes initialisation of the paper).
eps is defined in the [0, 1] pixel space (as in the paper). The data loader feeds normalised images,
so the attack works in the normalised space with a per-channel radius eps / std and per-channel
bounds (0 - mean) / std .. (1 - mean) / std. This is the same l_inf ball as in pixel space.
Attack setting used for GWAD (Ali et al., CVPR 2026 Findings): eps in {0.01, 0.05}, p_init = 0.8, single square.
"""

class Square2(SoftLabelAttackBase):
    def __init__(self, device, model, lp='inf', q_budgets=[10000], eps=0.05, p_init=0.8, stop=True,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)):
        super().__init__(device=device,
                         targeted=False,
                         model=model,
                         lp=lp,
                         q_budgets=q_budgets,
                         stop=stop,
                         perturbation='inc')
        self.name = 'square'

        self.eps_pixel = eps
        self.p_init = p_init
        self.min = 0
        self.max = 1
        # normalisation of the CIFAR-10 loader (utilities/load_data.py)
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

        self.x = []             # original input (normalised)
        self.shape = None
        self.epsilon = None     # per-channel radius in the normalised space
        self.lo = None
        self.hi = None
        self.step = 0           # iteration index i of Algorithm 1
        self.loss_best = None   # best margin loss l*
        self.label_idx = 0

    def p_selection(self, i):
        # p is halved at i in {10, 50, 200, 1000, 2000, 4000, 6000, 8000} for N = 10000 iterations;
        # the thresholds are rescaled to the actual number of iterations N
        n = self.query_max
        p = self.p_init
        for t in (10, 50, 200, 1000, 2000, 4000, 6000, 8000):
            if i >= t * n / 10000:
                p /= 2
        return p

    def square_side(self, p):
        # h = closest positive integer to sqrt(p * w^2)
        w = self.shape[2]
        h = int(round((p * w * w) ** 0.5))
        return min(max(h, 1), w)

    def project(self, x_new):
        # projection onto the l_inf ball around x (radius eps per channel) and onto the valid image range
        x_new = torch.max(torch.min(x_new, self.x + self.epsilon), self.x - self.epsilon)
        return torch.max(torch.min(x_new, self.hi), self.lo)

    def init_point(self):
        # vertical stripes of width one, colour of each stripe sampled from {-eps, eps}^c
        c, w = self.shape[1], self.shape[3]
        signs = torch.randint(0, 2, (1, c, 1, w), device=self.x.device).float() * 2 - 1
        delta = signs * self.epsilon
        return self.project(self.x + delta.expand(self.shape))

    def sample_update(self, h):
        # Algorithm 2: random square of side h, each channel set to a constant value rho in {-2eps, 2eps}
        c, hh, ww = self.shape[1], self.shape[2], self.shape[3]
        r = int(torch.randint(0, hh - h + 1, (1,)))
        s = int(torch.randint(0, ww - h + 1, (1,)))
        signs = (torch.randint(0, 2, (1, c, 1, 1), device=self.x.device).float() * 2 - 1)
        rho = signs * 2 * self.epsilon
        delta = torch.zeros(self.shape, device=self.x.device)
        delta[:, :, r:r + h, s:s + h] = rho
        return delta

    def margin_loss(self, output):
        # L = f_y(x) - max_{k != y} f_k(x); the attack minimises L, L < 0 means misclassified
        gt = output[0, self.label_idx]
        other = output[0].clone()
        other[self.label_idx] = -float('inf')
        return (gt - other.max()).detach()

    def attack(self, sm, x_best):
        p = self.p_selection(self.step)
        h = self.square_side(p)
        delta = self.sample_update(h)
        x_new = self.project(x_best + delta)

        # one query per iteration; the margin is computed from the same output (no extra query)
        output = self.inject(x_new)
        loss = self.margin_loss(output)
        if loss < self.loss_best:
            self.loss_best = loss
            x_best = x_new

        self.step += 1
        return sm, x_best

    def stop_criteria(self, x, x_adv):
        # misclassified <=> margin loss < 0; checked without an extra query
        if self.stop and self.loss_best is not None:
            return bool(self.loss_best < 0)
        return False

    def setup(self, x):
        self.shape = x.size()
        self.x = x.detach().clone()
        self.step = 0

        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        self.epsilon = self.eps_pixel / std
        self.lo = (0 - mean) / std
        self.hi = (1 - mean) / std
        # base run() clamps the final example to [min, max]; use the per-channel valid range
        self.min = self.lo
        self.max = self.hi

        x_best = self.init_point()
        output = self.inject(x_best)
        self.loss_best = self.margin_loss(output)

        return x_best

    def core(self, image):
        x_best = self.setup(image)

        # Generate the adversarial samples
        x_adv, queries0, queries1 = self.run(x_best)

        return x_adv, queries0, queries1

    def untarget(self, image, label):
        self.targeted = False
        self.label = label
        self.label_idx = int(label)

        img = image.detach().clone()
        adv, q0, q1 = self.core(img)

        return adv, q0, q1, self.adaptive.query_cnt

    def target(self, image, label, example):
        raise ValueError("targeted attack is not supported yet")
        return None
