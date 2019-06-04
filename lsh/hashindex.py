
class HashIndex:
    '''
    HashIndex that is

    Parameters
    ----------
    hash_tables : int
        the number of hashtables in the index
    '''
    def __init__(self, hash_tables = 128):
        self.hash_tables = hash_tables
        self.hash = [{} for _ in range(hash_tables)]
        self.objects = []

    """Returns union of objects from all buckets found using hash_array.
    Args:
        hash_array: array of hashes, each hash has type int.
    Returns:
        array of hashed of type {"object": <obj>, "hash": <object multihash> }
    """
    def query(self, hash_array):
        ids = set()
        for hash_idx, hash in enumerate(hash_array):
            ids.update(self._get_objects(hash_idx, hash))
        result = []
        for id in ids:
            result.append(self.objects[id])
        return result

    """Adds object to the hash index.
    Args:
        object: object to store in the index
        hash_array: array of hashes, each hash has type int.
    """
    def index(self, object, hash_array):
        id = self._add_and_get_id(object, hash_array)
        for hash_idx, hash in enumerate(hash_array):
            self._add_object(hash_idx, hash, id)

    """Add object to the index and generates new object id.
    Args:
        object: object to store in the index
        hash_array: array of hashes, each hash has type int.
    Returns:
        object id
    """
    def _add_and_get_id(self, object, hash_array):
        id = len(self.objects)
        self.objects.append({"object": object, "hash": hash_array})
        return id

    """Add object to the all hashtables.
    Args:
        hash_idx: hash table number
        hash: hash value
        id: object id
    """
    def _add_object(self, hash_idx, hash, id):
        hashtable = self.hash[hash_idx]
        hash = tuple(hash)
        if hash not in hashtable:
            hashtable[hash] = []
        hashtable[hash].append(id)

    """Gets hashed object.
    Args:
        hash_idx: index of the hash table.
        hash: hash value
    Returns:
        list of hashed objects.
    """
    def _get_objects(self, hash_idx, hash):
        hash = tuple(hash)
        hashtable = self.hash[hash_idx]
        return hashtable.get(hash, [])
