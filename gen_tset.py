"""Generate tset from en_v (lang1) and zh_v (lang2)."""
# @title def gen_tset from en_v zh_v

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nptyping import Float, NDArray, Shape


def gen_tset(
    en_v_: np.ndarray, zh_v_: np.ndarray, bsize: int = 100
) -> NDArray[Shape["int, 1"], Float]:
    """
    Generate tset from en_v (lang1) and zh_v (lang2).

    Returns: triple set tset
    """
    en_v = en_v_[:]
    zh_v = zh_v_[:]

    tset = []
    q, r = divmod(len(zh_v), bsize)
    runs = q + bool(r)

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
