import hashlib
from IPython.display import display
from components.image.image import Image
from glob import glob
from config import DATA_DIR
from tqdm.notebook import tqdm
from collections import Counter


def data_loader(folder):
    for file in glob(str(folder / '*.jpg')):
        yield Image.from_file(file)


def count_files(dir_set):
    count = 0
    for directory, label in dir_set:
        count += len(list(glob(str(directory / '*.jpg'))))
    return count

def test_set_loader(dir_set):
    for directory, label in dir_set:
        for img in data_loader(directory):
            yield img, label


VISION_DATA_DIR = DATA_DIR / 'image' / 'vision_test_image'

LANDSCAPE = VISION_DATA_DIR / 'landscape'
MAPS = VISION_DATA_DIR / 'maps'
MULTI_PHOTO = VISION_DATA_DIR / 'multi_photo'
PEOPLE = VISION_DATA_DIR / 'people'
TEXT = VISION_DATA_DIR / 'text'
ILLUSTRATION = VISION_DATA_DIR / 'illustration'
GRAPHIC = VISION_DATA_DIR / 'graphic'




class VisionPromptExperiment:

    def __init__(self, vision_client, prompt, test_set):
        self.test_set = test_set
        self.prompt = prompt
        self.labels = []
        self.predictions = []
        self.results = []
        self.imgs = []
        self.vision_client = vision_client

    def append(self, prediction, label, img):
        self.predictions.append(prediction)
        self.labels.append(label)
        self.results.append(prediction == label)
        self.imgs.append(img)

    def accuracy(self):
        return sum(self.results) / len(self.results)

    def show_wrong(self):
        for i, img in enumerate(self.imgs):
            if not self.results[i]:
                print(f"prediction: {self.predictions[i]}, label: {self.labels[i]}")
                display(img.get_thumbnail())

    def wrong_hist(self):
        counter = Counter()
        for lbl, res in zip(self.labels, self.results):
            if not res:
                counter[lbl] += 1
        return counter
        

    async def run(self):
        # for directory, label in self.test_set: 
            # for img in data_loader(directory):
        total_items = count_files(self.test_set)
        for img, label in tqdm(test_set_loader(self.test_set), total=total_items):
                output = await self.vision_client.complete(self.prompt, img, type(label))
                self.append(output, label, img)

    def __repr__(self) -> str:
        return f"VisionPromptExperiment(\nprompt={self.prompt}\n accuracy={self.accuracy()}\n)"


class VisionPromptTester:

    def __init__(self, vision_client) -> None:
        self.vision_client = vision_client
        self.experiments = {}
    
    async def test_prompts(self, prompts, data_labels):
        exp_out = []
        for prompt in prompts:
            print(prompt)
            exp = VisionPromptExperiment(self.vision_client, prompt, data_labels)
            await exp.run()
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            self.experiments[prompt_hash] = exp
            exp_out.append(exp)
        return exp_out
            
    

