"""Load text with best effort to determine encoding."""
from typing import Union
from pathlib import Path
import re

# import cchardet
# from logzero import logger
import charset_normalizer
from loguru import logger


def load_text(filename: Union[str, Path], splitlines: bool = False) -> str:
    """Load text for given filepath.

    Args
    ----
    filename (str|Path): filename
    splitlines (bool): output list of pars if True, defaul False

    Returns
    ---
    text with blank lines removed or list or pars
    """
    if not Path(filename).is_file():
        _ = Path(filename).resolve().as_posix()
        raise SystemExit(f"{_} does not exist or is not a file")
    try:
        _ = charset_normalizer.detect(Path(filename).read_bytes())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"charset_normalizer exc: {exc}, setting encoding to utf8")
        _ = {"encoding": "utf8"}
    encoding = _["encoding"]

    try:
        cont = Path(filename).read_text(encoding)
    except Exception as exc:
        logger.error(f"read_text exc: {exc}")
        raise

    # replace unicode spaces with normal space " "
    cont = re.sub(r"[\u3000]", " ", cont)

    _ = [elm.strip() for elm in cont.splitlines() if elm.strip()]

    # list of paragraphs with blank lines removed
    if splitlines:
        return _.splitlines()

    # text with blank lines removed
    return _
