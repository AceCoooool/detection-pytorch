from torch.nn import init
from torch import nn


# weight initialization
def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_() if m.bias is not None else None


# Sets the learning rate to the initial LR decayed by 10 at every specified step
def adjust_learning_rate(optimizer, lr, gamma, step):
    lr = lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
