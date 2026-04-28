# RAGLens

Ferramenta de inspeção e rastreabilidade para pipelines RAG. Para cada query, expõe quais chunks foram recuperados, seus scores de similaridade semântica, por que cada um foi selecionado e como influenciaram a resposta final do LLM — tornando o comportamento do RAG observável e depurável.

---

<img width="964" height="886" alt="image" src="https://github.com/user-attachments/assets/b2140f08-055d-4fa2-97c9-eb94a03416ff" />

---

## Motivação

O RAGLens nasceu de uma necessidade real identificada durante o desenvolvimento da plataforma **PACEF** da DMHealth — uma plataforma de preparação de agentes, prompts, e guardrails voltada à área da saúde.

O PACEF utiliza RAG para fundamentar as respostas dos agentes em documentos clínicos e protocolos institucionais: guias da Atenção Básica, cadernetas de saúde, formulários do e-SUS e diretrizes do Ministério da Saúde. O objetivo é garantir que os agentes respondam com base em evidências concretas, e não em conhecimento genérico do modelo.

O problema surgiu quando o sistema começou a apresentar **respostas imprecisas ou fora de contexto**, e não havia como saber *por quê*: qual documento foi recuperado? com que grau de relevância? o chunk escolhido era realmente o mais adequado? os guardrails estavam funcionando?

Sem visibilidade sobre o pipeline, melhorar a qualidade das respostas era tentativa e erro.

### Por que observabilidade importa na saúde

Em contextos clínicos, um RAG mal calibrado não é apenas um problema técnico — pode significar contexto errado chegando a um agente que orienta profissionais de saúde. O ciclo de desenvolvimento exige:

- **Inspecionar** quais trechos dos documentos estão sendo recuperados para cada pergunta
- **Medir** os scores de similaridade para identificar quando a base de documentos precisa ser revisada
- **Iterar sobre prompts e guardrails** com base em evidências do que o modelo está "vendo"
- **Comparar** o desempenho entre diferentes modelos de embedding e LLMs

O RAGLens resolve isso fornecendo visibilidade completa sobre cada etapa do pipeline, sem alterar seu comportamento.

---

## Arquitetura

```
query + docs
     │
     ▼
RAGDebugger.query()
     │
     ├─ DeepInfra API → embeddings (somente a query; chunks já pré-computados)
     ├─ cosine similarity (numpy, sem FAISS)
     ├─ top-k chunks rankeados com scores e explicações
     ├─ Maritaca API (Sabiá) → resposta fundamentada no contexto
     └─ DebugResult (query, chunks, scores, resposta, timestamp)
          │
          ▼
     FastAPI server
          │
          ├─ POST /debug          → executa pipeline e salva no SQLite
          ├─ GET  /history?n=10   → últimas N queries com resultados
          ├─ POST /ingest/upload  → ingere PDFs via browser
          ├─ POST /ingest/folder  → ingere PDFs de uma pasta no servidor
          └─ index.html           → frontend com visualização de scores
```

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Configure as chaves de API:

```bash
cp .env.example .env
# edite .env e preencha DEEPINFRA_API_KEY e SABIA_API_KEY
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
Modelo  : sabia-3 | Embedding: intfloat/multilingual-e5-large
Tempo   : 1243 ms
Chunks  : 3 recuperados de 3 disponíveis

  [1] score=0.921 (92.1%) | protocolo.pdf
      Altíssima similaridade semântica com a query.
      O escore de Apgar avalia o recém-nascido em 5 critérios.
  ...

Resposta: O escore de Apgar é utilizado para avaliar...
```

Veja `example.py` para um exemplo completo com documentos da atenção básica.

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
| DELETE | `/history` | Limpa o histórico |
| POST | `/ingest/upload` | Ingere PDFs enviados pelo browser |
| POST | `/ingest/folder` | Ingere PDFs de uma pasta no servidor |
| GET | `/documents` | Lista documentos ingeridos |
| DELETE | `/documents` | Limpa a base de chunks |

---

## Variáveis de ambiente

| Variável | Obrigatória | Padrão | Descrição |
|---|---|---|---|
| `DEEPINFRA_API_KEY` | sim | — | Chave da DeepInfra (embeddings) |
| `SABIA_API_KEY` | sim | — | Chave da Maritaca AI (LLM) |
| `EMBEDDING_MODEL` | não | `intfloat/multilingual-e5-large` | Modelo de embeddings |
| `LLM_MODEL` | não | `sabia-3` | Modelo de geração |
| `TOP_K` | não | `3` | Chunks recuperados por query |

---

## Estrutura do projeto

```
RAGLens/
├── rag_debugger.py              # core: RAGDebugger, DebugResult, ChunkResult
├── server.py                    # FastAPI + SQLite
├── index.html                   # frontend com visualização de scores
├── example.py                   # exemplo de uso como biblioteca
├── tests/
│   └── test_rag_debugger.py    # testes unitários (sem chamadas de API)
├── Makefile                     # atalhos: install / run / test / clean
├── requirements.txt
├── .env.example
└── README.md
```

---

## Integração com o PACEF

O RAGLens foi desenvolvido como ferramenta de suporte ao ciclo de qualidade do **PACEF**. Ele permite inspecionar e iterar sobre o pipeline RAG antes que prompts e guardrails sejam promovidos para os agentes em produção — sem alterar o comportamento do sistema principal.

O fluxo típico de uso:

1. Carregar os mesmos documentos que o PACEF utiliza (protocolos, cadernetas, e-SUS)
2. Submeter as perguntas que os agentes receberão no mundo real
3. Analisar quais chunks foram recuperados e com qual score
4. Ajustar o tamanho dos chunks, a base documental ou o prompt do sistema
5. Repetir até que os chunks recuperados sejam consistentemente relevantes

---

## Status e próximos passos

O projeto está funcional e em uso ativo como ferramenta de suporte ao desenvolvimento do PACEF.

Melhorias planejadas:

- **Testes em lote:** executar múltiplas queries de uma vez e comparar os resultados agregados, permitindo avaliar o comportamento do pipeline em cenários variados sem rodar uma query por vez
- **Análise de lacunas pelo histórico:** identificar automaticamente, a partir das queries com baixos scores de similaridade, quais temas recorrentes não estão sendo cobertos pelos documentos atuais — sinalizando onde a base documental precisa ser expandida
