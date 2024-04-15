from dataclasses import dataclass
from typing import Union

@dataclass
class RunConfig:
    base_dir: str = '/home/dzigen/Desktop/ITMO/ВКР/КМУ2024'
    run_name: str = 'reader1'
    run_type: str = 'reader' # 'reader' / 'retriever' / 'join
    reader_type: str = 'fid'
    base_reader_weights: str = 'google/flan-t5-base' # 'google/flan-t5-base' / 'google-t5/t5-base'
    reader_input_format: str = "Context: {} </s> Question: {}"
    tuned_reader_weights: str = '' # 
    retriever_type: str = '' # 'e5' / 'bm25e5' / 'bm25colbert' 
    tuned_retriever_weights: str = '' #
    retriever_frozen: bool = True  # True / False
    eval_only: bool = False
    train_type: str = 'supervised'
    lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 2
    device: str = 'cpu' # 'cuda' / 'cpu'
    reader_gen_ml: int = 64
    retrieved_cands: int = 4 # number of candidates to retriever for reader
    retriever_docs_batch: int = 4 # from stage1 to stage2
    retriever_bm25_cands: int = 256
    dataset: str = 'squad'
    train_size: int = 1000
    eval_size: int = 1000
    base: str = '' # 'squad' / 'nq' / 'triviaqa'
    base_score_compare: str = 'meteor'
    to_save: bool = True
    reader_layers_toupdate: Union[list,str] = 'all' #
    retriever_layers_toupdate: Union[list,str] = 'all' #
    grad_accum_steps: int = 1