# RAGLens

Ferramenta de inspeção e rastreabilidade para pipelines RAG. Para cada query, expõe quais chunks foram recuperados, seus scores de similaridade semântica, por que cada um foi selecionado e como influenciaram a resposta final do LLM — tornando o comportamento do RAG observável e depurável.

Desenvolvido como projeto de pós-graduação integrado ao [Sabiá Tester](../APP_TESTES).

---

## Arquitetura

```
query + docs
     │
     ▼
RAGDebugger.query()
     │
     ├─ DeepInfra API → embeddings (batch: query + todos os docs)
     ├─ cosine similarity (numpy, sem FAISS)
     ├─ top-k chunks rankeados com scores e explicações
     ├─ DeepInfra API → LLM (resposta com contexto)
     └─ DebugResult (query, chunks, scores, resposta, timestamp)
          │
          ▼
     FastAPI server
          │
          ├─ POST /debug   → executa pipeline e salva no SQLite
          ├─ GET  /history → últimas N queries
          └─ index.html    → frontend com visualização de scores
```

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure a chave de API:

```bash
cp .env.example .env
# edite .env e preencha DEEPINFRA_API_KEY
```

---

## Uso rápido (Python)

```python
import asyncio
from rag_debugger import RAGDebugger, Document

debugger = RAGDebugger(top_k=3)

docs = [
    Document(text="O escore de Apgar avalia o recém-nascido em 5 critérios.", source="protocolo.pdf"),
    Document(text="Hipertensão gestacional ocorre após 20 semanas.", source="ubs_completo.pdf"),
    Document(text="O aleitamento materno exclusivo é recomendado até 6 meses.", source="e-sus.pdf"),
]

result = asyncio.run(debugger.query("Qual o escore usado para avaliar o bebê ao nascer?", docs))
print(result.summary())
```

Saída esperada:

```
Query   : Qual o escore usado para avaliar o bebê ao nascer?
Modelo  : mistralai/Mistral-7B-Instruct-v0.3 | Embedding: intfloat/e5-mistral-7b-instruct
Tempo   : 1243 ms
Chunks  : 3 recuperados de 3 disponíveis

  [1] score=0.921 (92.1%) | protocolo.pdf
      Altíssima similaridade semântica com a query.
      O escore de Apgar avalia o recém-nascido em 5 critérios.
  ...

Resposta: O escore de Apgar é utilizado para avaliar...
```

---

## Servidor

```bash
uvicorn server:app --reload --port 8000
```

Endpoints:

| Método | Rota | Descrição |
|---|---|---|
| POST | `/debug` | Executa pipeline RAG e retorna DebugResult |
| GET | `/history?n=10` | Últimas N queries com resultados |

Exemplo de chamada:

```bash
curl -X POST http://localhost:8000/debug \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qual o peso ideal do recém-nascido?",
    "docs": [
      {"text": "O peso normal ao nascer é entre 2,5 e 4 kg.", "source": "protocolo.pdf"}
    ]
  }'
```

---

## Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `DEEPINFRA_API_KEY` | sim | — | Chave da DeepInfra |
| `EMBEDDING_MODEL` | não | `intfloat/e5-mistral-7b-instruct` | Modelo de embeddings |
| `LLM_MODEL` | não | `mistralai/Mistral-7B-Instruct-v0.3` | Modelo de geração |
| `TOP_K` | não | `3` | Chunks recuperados por query |

---

## Estrutura do projeto

```
RAG_DEBUGGER/
├── rag_debugger.py   # core: RAGDebugger, DebugResult, ChunkResult
├── server.py         # FastAPI + SQLite
├── index.html        # frontend com visualização de scores
├── requirements.txt
├── .env.example
└── README.md
```

---

## Integração com Sabiá Tester

O RAG Debugger foi projetado para substituir ou instrumentar as funções `rag_retrieve` e `inject_rag_context` do `sabia_tester.py`, adicionando visibilidade sem alterar o comportamento do pipeline.
