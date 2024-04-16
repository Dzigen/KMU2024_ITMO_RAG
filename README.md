# KMU2024_ITMO_RAG

Были реализованы следующие RAG-конфигурации:
* Retriever(BM25 + ColBERT) + Reader(FiD);
* Retriever(Single E5) + Reader(FiD);
* Retriever(Dual E5) + Reader(FiD).

Использовалась функция ошибки для совместного обучения Retriever- и Reader- моделей из статьи про EMDR2: https://arxiv.org/pdf/2106.05346.pdf.
