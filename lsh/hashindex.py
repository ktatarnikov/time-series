
class HashIndex:
    def __init__(self, hash_tables = 128):
        self.hash_tables = hash_tables
        self.hash = [{} for _ in range(hash_tables)]
        self.objects = []

    def query_all(self, hash_array):
        ids = set()
        for hash_idx, hash in enumerate(hash_array):
            ids.update(self._get_objects(hash_idx, hash))
        result = []
        for id in ids:
            result.append(self.objects[id])
        return result

    def index(self, object, hash_array):
        id = self._add_and_get_id(object, hash_array)
        for hash_idx, hash in enumerate(hash_array):
            self._add_object(hash_idx, hash, id)

    def _add_and_get_id(self, object, hash_array):
        id = len(self.objects)
        self.objects.append({"object": object, "hash": hash_array})
        return id

    def _add_object(self, hash_idx, hash, id):
        hashtable = self.hash[hash_idx]
        hash = tuple(hash)
        if hash not in hashtable:
            hashtable[hash] = []
        hashtable[hash].append(id)

    def _get_objects(self, hash_idx, hash):
        hash = tuple(hash)
        hashtable = self.hash[hash_idx]
        return hashtable.get(hash, [])
