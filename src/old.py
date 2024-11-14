def accuracy_dist(inp, targ, thresh=1):
    inp,targ = flatten_check(inp,targ)
    return ((inp<thresh)==targ.bool()).float().mean()


def my_loss_func_LeCun(dist, target, margin=2, reduction='mean'):
    assert reduction in ['mean', 'none']
    neg_dist = torch.clamp(margin - dist, min=0.0)
    res = torch.where(target.bool(), dist, -neg_dist).pow(2)
    if reduction=='mean':
        res = res.mean()
    return 0.5 * res

def thresh_finder(preds, targs, acc, x0, xf):
    xs = torch.linspace(x0,xf)
    accs = [ acc(preds, targs, thresh=x) for x in xs ]
    plt.plot(xs,accs)