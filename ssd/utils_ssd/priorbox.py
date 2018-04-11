from math import sqrt
import numpy as np
import torch


class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

    def __call__(self):
        output = None
        for k, f in enumerate(self.feature_maps):
            mean = None
            f_k = self.image_size / self.steps[k]
            cx, cy = np.meshgrid(np.arange(0, f) + 0.5, np.arange(0, f) + 0.5)
            cy, cx = cy / f_k, cx / f_k
            s_k = self.min_sizes[k] / self.image_size
            s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
            wh = np.tile(s_k, (f, f))
            temp = np.vstack((cx.ravel(), cy.ravel(), wh.ravel(), wh.ravel())).transpose()
            mean = temp if mean is None else np.c_[mean, temp]
            wh = np.tile(s_k_prime, (f, f))
            temp = np.vstack((cx.ravel(), cy.ravel(), wh.ravel(), wh.ravel())).transpose()
            mean = np.c_[mean, temp]
            for ar in self.aspect_ratios[k]:
                w = np.tile(s_k * sqrt(ar), (f, f))
                h = np.tile(s_k / sqrt(ar), (f, f))
                temp = np.vstack((cx.ravel(), cy.ravel(), w.ravel(), h.ravel())).transpose()
                mean = np.c_[mean, temp]
                temp = np.vstack((cx.ravel(), cy.ravel(), h.ravel(), w.ravel())).transpose()
                mean = np.c_[mean, temp]
            output = mean.reshape((-1, 4)) if output is None else np.r_[output, mean.reshape((-1, 4))]
        output = torch.from_numpy(output.astype(np.float32))
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    from ssd import config as cfg

    p = PriorBox(cfg.v2)()
    print(p[100: 110, :])
