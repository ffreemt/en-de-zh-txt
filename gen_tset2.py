"""Generate tset from en_v (lang1) and zh_v (lang2)."""
# @title def gen_tset from en_v zh_v

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
# from nptyping import Float, NDArray, Shape
from numpy.typing import NDArray

# def gen_tset(
#     en_v_: np.ndarray, zh_v_: np.ndarray, bsize: int = 100
# ): # -> NDArray[Shape["N, 1"], Float]:


def gen_tset2(
    en_v_: NDArray[np.float64],
    zh_v_: NDArray[np.float64],
    bsize: int = 100,
    p: int = 1,
) -> NDArray[np.float64]:
    """
    Generate tset from en_v (lang1) and zh_v (lang2).

    Returns: triple set tset
    """
    en_v = en_v_[:]
    zh_v = zh_v_[:]

    tset = []
    q, r = divmod(len(zh_v), bsize)
    runs = q + bool(r)

    if p <= 1:
      for idx in tqdm(range(runs)):
          cmat = cosine_similarity(en_v, zh_v[idx * bsize : (idx + 1) * bsize])
          tset_bsize = [
              *zip(
                  range(idx * bsize, (idx + 1) * bsize),
                  cmat.argmax(axis=0),
                  cmat.max(axis=0),
              )
          ]
          tset.extend(tset_bsize)

      return np.array(tset)

    # consider -partition(-cmat, 2), argpartion(-cmat, 2)
    for idx in tqdm(range(q)):
        cmat = cosine_similarity(en_v, zh_v[idx * bsize : (idx + 1) * bsize])
        # tset_bsize = [
        #     *zip(
        #         range(idx * bsize, (idx + 1) * bsize),
        #         cmat.argmax(axis=0),
        #         cmat.max(axis=0),
        #     )
        # ]
        tset_bsize = []
        val = -np.partition(-cmat, 2,axis=0)[:2,:]
        lab = np.argpartition(-cmat, 2, axis=0)[:2,:]

        for jdx in range(bsize):
            tset_bsize.extend([*zip([jdx + idx * bsize ]*2, lab[:, jdx].tolist(), val[:, jdx].tolist())])
        tset.extend(tset_bsize)
    if r:
        # cmat = cosine_similarity(en_v, zh_v[idx * bsize : (idx + 1) * bsize])
        cmat = cosine_similarity(en_v, zh_v[q * bsize : q * bsize + r])
        tset_bsize = []
        val = -np.partition(-cmat, 2, axis=0)[:2,:]
        lab = np.argpartition(-cmat, 2, axis=0)[:2,:]

        for jdx in range(r):
            tset_bsize.extend([*zip([q * bsize + jdx]*2, lab[:, jdx].tolist(), val[:, jdx].tolist())])
        tset.extend(tset_bsize)
        
    return np.array(tset)
