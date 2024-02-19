import os
from PIL import Image
import numpy as np
import io
import pickle

class StreetScene:
    """
    Train/Train001-Train046
    Test/
    fps = 15
    """
    def __init__(self, base_dir="/home/zeyuan/data/street-scene/") -> None:
        self.frame_paths = []
        self.fps = 15
        self.cache_file="streetscene_cache.pkl"
        if os.path.exists(self.cache_file): 
            self.cache = pickle.load(open(self.cache_file, 'rb'))
            print("[StreetScene]Cache Loaded")
        else:
            self.cache = {}
            print("[StreetScene]No Cache Found. Video extension might be slow.")
        for split in ["Train", "Test"]:
            split_base = os.path.join(base_dir, split)
            for segment in sorted(os.listdir(split_base)):
                prefix = os.path.join(split_base, segment)
                for path in sorted(os.listdir(prefix)):
                    if path.endswith(".jpg"):
                        self.frame_paths.append(os.path.join(prefix, path))
        self.frame_paths = self.frame_paths[:54000]
        self.input_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.input_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __len__(self):
        return len(self.frame_paths) // self.fps
    
    def normalize(self, image):
        img_array = np.array(image).astype(np.float32) / 255.0
        normalized_img_array = (img_array - self.input_mean) / self.input_std
        return (normalized_img_array * 255).astype(np.uint8)
        
        
    
    def get_frame(self, i):
        if self.cache and i in self.cache:
            return self.cache[i]
        image = Image.open(self.frame_paths[i]).convert('RGB')
        image.load()
        image = image.resize((400, 224))
        image = image.crop((88, 0, 400-88, 224))
        array = self.normalize(image)
        # buffer = io.BytesIO()
        # image.save(buffer, format='png')
        if self.cache:
            self.cache[i] = array
        return Image.fromarray(array)
    
    def read_frames(self, start=0, num_frames=50):
        assert start + num_frames <= len(self)
        return [self.get_frame(i) for i in range(start, start+num_frames)]
    
    def read_by_time(self, start_time, duration, num_frm):
        start_idx = int(start_time * self.fps)
        end_idx = int((start_time + duration) * self.fps)
        indices = np.linspace(start_idx, end_idx, num=num_frm, dtype=np.int32)
        return [self.get_frame(i) for i in indices]
    
    def save_cache(self):
        pickle.dump(self.cache, open(self.cache_file, 'wb'))
        



if __name__ == "__main__":
    data = StreetScene()
    data.cache = None
    print("total duration:", len(data), "s")
    # from tqdm import tqdm
    # for i in tqdm(range(len(data.frame_paths))):
    #     data.get_frame(i)
    from multiprocessing import Pool, Manager
    from tqdm import tqdm

    # Define a worker function taking the index and frame path
    def worker(idx):
        return idx, data.get_frame(idx)

    def process_frames_in_parallel(data):
        manager = Manager()
        cache_dict = manager.dict()  # Create a managed dictionary to share cache between processes



        num_work = len(data.frame_paths)
        # Create a pool of workers
        with Pool() as pool:
            # Use pool.map to apply 'worker' to each frame index
            # results = pool.map(worker, range(len(data.frame_paths)))
            results_iterator = pool.imap_unordered(worker, range(num_work))
            # Loop over results with progress bar
            for i, frame_data in tqdm(results_iterator, total=num_work, desc="Processing frames"):
                cache_dict[i] = frame_data

        # Convert the managed dictionary back to a regular dictionary
        pickle.dump(dict(cache_dict), open("streetscene_cache.pkl", 'wb'))

    process_frames_in_parallel(data)