from torch import nn

xavier = nn.init.xavier_uniform_

def weight_init(m):
    try:
        nn.init.xavier_normal_(m.weight)
    except AttributeError:
        pass

    try:
        nn.init.constant_(m.bias, 0)
    except AttributeError:
        pass
