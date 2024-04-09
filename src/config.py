from dataclasses import dataclass

@dataclass
class RunConfig:
    base_dir: str = '/home/dzigen/Desktop/ITMO/ВКР/КМУ2024'
    run_name: str = 'reader1'
    run_type: str = 'reader' # 'reader' / 'retriever' / 'join
    reader_type: str = 'fid'
    tuned_reader_weights: str = '' # 
    retriever_type: str = '' # 'e5' / 'bm25e5' / 'bm25colbert' 
    tuned_retriever_weights: str = '' #
    retriever_frozen: bool = True  # True / False
    lr: float = 1e-5
    epochs: int = 10
    batch_size: int = 4
    device: str = 'cpu' # 'cuda' / 'cpu'
    retrieved_cands: int = 4 # number of candidates to retriever for reader
    retriever_docs_batch: int = 4 # from stage1 to stage2
    train_dataset: str = 'squad'
    base: str = '' # 'squad' / 'nq' / 'triviaqa'
    base_score_compare: str = 'meteor'