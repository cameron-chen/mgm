import numpy as np
import ot
import torch
import torch.autograd as autograd


def emd(samp1, samp2, method = 'sinkhorn', normalize = False):
    '''Note that the W2 is the squared Euclidean EMD
    EMD with l2 loss should be the sequared root of the W2

    This function can compute emd for at most 2D sample

    Args:
        method: Alternative methods to compute EMD. 'exact' computes the exact
            value of EMD. 'sinkhorn' compute an approximate value for EMD
        normalize: True, the ground distance would be normalized by M.max()

    Notes: 
        if the sample size is larger than 1e4, do not use this method
        within 2 seconds, this algorithm can provide a solution of size 1.2e4
    '''
    batch_size = samp1.shape[0]
    a, b = np.ones((batch_size,))/batch_size, np.ones((batch_size,))/batch_size

    # loss matrix
    M = ot.dist(samp1,samp2, metric='euclidean')
    if normalize:
        M = M/M.max()

    if method == 'exact':
        W2 = ot.emd2(a, b, M)
    elif method == 'sinkhorn':
        normalFactor = M.max()
        M = M/normalFactor
        W2 = ot.sinkhorn2(a,b,M,0.1)
        W2 = W2 * normalFactor

    return W2

def compute_gradient_penalty(D, real_samples, fake_samples, dtype, labels=None):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = dtype(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    real_samples = real_samples.view((real_samples.size(0),-1))
    fake_samples = fake_samples.view((fake_samples.size(0),-1))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # if labels is a tuple, call dtype() on each element in the tuple recurrently
    if type(labels)==tuple:
        labels_change_dtype = []
        for tensor in labels:
            labels_change_dtype.append(dtype(tensor))
        labels=tuple(labels_change_dtype)
    # elif labels is a tensor, call dtype() on labels itself
    elif type(labels)==torch.Tensor:
        labels = dtype(labels)
    # elif there is no labels, do nothing
    elif labels==None:
        pass
    # else, raise an exception
    else:
        raise TypeError("Labels are neithor a tuple nor a tensor!")
    
    if labels == None:
        d_interpolates=D(interpolates)
    else:
        d_interpolates = D(interpolates, labels)
    fake = dtype(real_samples.size(0), 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_gradient_penalty_cond(D, real_samples, fake_samples, dtype, realLabel,fakeLabel):
    """Calculates the gradient penalty loss for WGAN GP in the case where the labels are 
       coutinuous variables
    """
    # Random weight term for interpolation between real and fake samples
    alpha = dtype(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    real_samples = real_samples.view((real_samples.size(0),-1))
    fake_samples = fake_samples.view((fake_samples.size(0),-1))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolatesLabel = (alpha * realLabel + ((1 - alpha) * fakeLabel)).requires_grad_(True)
    interpolatesLabel = dtype(interpolatesLabel)
    
    d_interpolates = D(interpolates, interpolatesLabel)
    fake = dtype(real_samples.size(0), 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=(interpolates, interpolatesLabel),
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gra_s = gradients[0]
    gra_x = gradients[1]
    gra_all = torch.cat((gra_s, gra_x), -1)
    gradient_penalty = ((gra_all.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    

if __name__ == "__main__":
    import numpy as np
    import torch

    samp1= np.array([[1,1],[2,1],[3,1]])
    samp2= np.array([[1,2],[2,3],[3,2]])

    print(emd(samp1, samp2, method='exact'))
