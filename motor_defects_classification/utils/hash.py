import checksumdir


def dir_hash(dpath: str) -> str:
    hash_str = checksumdir.dirhash(dpath)
    return hash_str
