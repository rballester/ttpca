from __future__ import (absolute_import, division,
                        print_function, unicode_literals, )
from future.builtins import range

import numpy as np
import tt
import copy

import ttrecipes as tr


def reduce(t, modes, degree=3, eps=1e-3, verbose=False, return_all=False):
    """
    Given a tensor and a list of modes, consider the space of subtensors obtained
    by fixing these modes. Find a PCA basis for that space and return the coefficients
    for each subtensor

    :param t: a TT
    :param modes: the ensemble modes
    :param degree: how large a PCA basis to take (default is 3)
    :param eps: relative error tolerance. Used while permuting modes. Default is 1e-3
    :param verbose:
    :param return_all: if True, return information about the projection error. Default is False
    :return: ensemble (if not return_all) or (ensemble, preserved, residual) (if return_all):
        - ensemble: a TT of size M1 x ... x Mk x `degree`, where M1, etc. are the sizes of `t` along `modes`
        - preserved: the fraction (between 0 and 1; higher is better) of norm preserved by the embedding
        - residual: the tensor containing the trailing R-degree principal components

    """

    N = t.d
    shape = t.n
    if not hasattr(modes, '__len__'):
        modes = [modes]

    if len(modes) != len(np.unique(modes)):
        raise ValueError('Modes may not be repeated')
    assert 0 <= np.min(modes)
    assert np.max(modes) < N

    if verbose:
        print('Computing PCA ensemble...')

    # Group target modes next to each other
    if verbose:
        print('\tTransposing tensor... 0', end='')
    part1 = np.setdiff1d(np.arange(modes[0]), modes)  # Modes that will lie to the left of the target ones
    part2 = np.setdiff1d(np.arange(modes[0], N), modes)  # Modes that will lie to the right of the target ones
    order = list(part1)+list(modes)+list(part2)
    t = tr.core.transpose(t, order, eps=eps)
    start_pos = order.index(modes[0])  # The first mode (modes[0]) ended up at this position

    # We now project each vector on the subspace spanned by the constant vector
    # This is achieved by subtracting each vector's within-mean
    cores = copy.deepcopy(tt.vector.to_list(t))
    for n in range(N):
        if n < start_pos or n >= start_pos+len(modes):
            cores[n] = np.repeat(np.mean(cores[n], axis=1, keepdims=True), cores[n].shape[1], axis=1)
    withinmeanrep = tt.vector.from_list(cores)
    withinmean = tr.core.means(t, modes=list(range(start_pos))+list(range(start_pos+len(modes), N)))
    t = tt.vector.round(t-withinmeanrep, eps=eps)
    degree -= 1

    withinmean2 = withinmean - tr.core.constant_tt(shape=withinmean.n, fill=tr.core.mean(withinmean))
    norm1 = np.sqrt(np.asscalar(np.sum(withinmean2.full()**2)))

    # Subtract the cross-mean (mean of the collection of vectors)
    cores = copy.deepcopy(tt.vector.to_list(t))
    for n in range(start_pos, start_pos+len(modes)):
        cores[n] = np.repeat(np.mean(cores[n], axis=1, keepdims=True), cores[n].shape[1], axis=1)
    crossmean = tt.vector.from_list(cores)
    t = tt.vector.round(t-crossmean, eps=eps)

    norm2 = tt.vector.norm(t) / np.sqrt(np.prod(shape[np.delete(np.arange(N), modes)]))

    # Orthogonalize to the left and right of these modes
    cores = tt.vector.to_list(t)
    if verbose:
        print('\n\tOrthogonalization...')
    for i in range(start_pos):
        tr.core.left_orthogonalize(cores, i)
    for i in range(N-1, start_pos+len(modes)-1, -1):
        tr.core.right_orthogonalize(cores, i)

    # Cut out the relevant sequence of modes
    cores = cores[start_pos:start_pos+len(modes)]

    # Convert the sequence into a TT (convert ranks to spatial modes)
    cores = [np.eye(cores[0].shape[0])[np.newaxis, :, :]] + cores + [np.eye(cores[-1].shape[2])[:, :, np.newaxis]]
    t = tt.vector.from_list(cores)

    # Put the rank modes together at the very right
    if verbose:
        print('\tPutting rank modes together...')
    t = tr.core.shift_mode(t, 0, t.d-2, eps=eps)
    cores = tt.vector.to_list(t)

    # Merge the two rank dimensions into one
    cores[-2] = np.einsum('iaj,jbk->ikab', cores[-2], cores[-1])
    del cores[-1]
    cores[-1] = np.reshape(cores[-1], [-1, cores[-1].shape[2]*cores[-1].shape[3], 1])
    t = tt.vector.from_list(cores)

    norm = tt.vector.norm(t)
    if norm > 0:
        t *= (norm2 / norm)

    # Compute the truncated PCA of the space spanned by the rank modes
    if verbose:
        print('\tSVD reduction...')
    cores = tt.vector.to_list(t)
    tr.core.orthogonalize(cores, t.d-1)
    cores[-1] = cores[-1][..., 0]
    svd = np.linalg.svd(cores[-1], full_matrices=False)

    if return_all:
        # Compute error
        cores2 = copy.deepcopy(cores)
        cores2[-1] = svd[0][:, degree:].dot(np.diag(svd[1])[degree:])[:, :, np.newaxis]
        error = tt.vector.from_list(cores2)

    cores[-1] = svd[0][:, :degree]
    cores[-1] = cores[-1].dot(np.diag(svd[1][:degree]))
    cores[-1] = cores[-1][:, :, np.newaxis]
    if cores[-1].shape[1] < degree:  # Degenerate case: add a slice of zeros
        cores[-1] = np.concatenate([cores[-1], np.zeros([cores[-1].shape[0], degree-cores[-1].shape[1],
                                                         cores[-1].shape[2]])], axis=1)
    proj = tt.vector.from_list(cores)

    withinmean = tt.vector.from_list(tt.vector.to_list(withinmean) + [np.ones([1, 1, 1])])
    proj = tr.core.concatenate([withinmean, proj], axis=len(modes), eps=eps)

    if verbose:
        print()

    if return_all:
        norm2preserved = tt.vector.norm(proj[[slice(None)]*len(modes) + [slice(1, None)]])
        preserved = np.sqrt((norm1**2 + norm2preserved**2) / (norm1**2 + norm2**2))
        if verbose:
            print('This embedding preserves {:g}% of the original norm'.format(preserved*100))
        return proj, preserved, error
    else:
        return proj
