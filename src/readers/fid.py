import transformers
import torch
from .archs.fid_model import T5_BASE_PATH, FiDT5

class FiDReader:
    def __init__(self, base_model=T5_BASE_PATH, device='cpu') -> None:
        self.base_model_name = base_model
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(base_model)
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(base_model)
        self.device = device

        self.tokenize = lambda x: self.tokenizer.batch_encode_plus(
            x, max_length=512, truncation=True, 
            padding='max_length', return_tensors='pt')

        self.model = FiDT5(t5.config)
        self.model.load_t5(t5.state_dict())
        self.model.to(self.device)

    def load_model(self, weights_path):
        print("Load tuned FiD-model...")
        self.model.load_state_dict(torch.load(weights_path))
        self.model.to(self.device)