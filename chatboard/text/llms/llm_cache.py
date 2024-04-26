


from config import DATA_DIR
import json


format = {
    "run_id"
    "input"
    "chunks"
    "completion"
}


class LlmStreamCache:


    def __init__(self, run_name=None, run_id=None, dirname= DATA_DIR/ "llm_cache") -> None:
        self.dirname = dirname
        self.run_name = run_name
        self.run_id = run_id
        self.chunks = []
        # try:
        #     with open(filename, "r") as f:
        #         self.media_lookup= json.load(f)
        # except:
        #     self.media_lookup = {}

    # def __getitem__(self, key):
    #     return self.media_lookup.get(key, None)
    
    # def __setitem__(self, key, value):
    #     self.media_lookup[key] = value
        
    def add(self, chunk):
        self.chunks.append(chunk)
    
    def save(self, chunk):
        with open(self.dirname / f"{self.run_id}.json", "w") as f:
            json.dump(self.chunks, f, indent=4)



    def stream(self, filename):
        stream_test_chunks = json.loads(open(self.dirname / filename, 'r').read())
        for chunk in stream_test_chunks:
            yield chunk


    def list_files(self):
        return [str(f) for f in self.dirname.glob("*.json")]

