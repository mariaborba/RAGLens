"""
Exemplo de uso do RAGLens como biblioteca Python.
Executa o pipeline RAG completo com documentos embutidos no código.
"""

import asyncio
from rag_debugger import RAGDebugger, Document

DOCS = [
    Document(
        text="O escore de Apgar avalia o recém-nascido em cinco critérios: frequência cardíaca, "
             "esforço respiratório, tônus muscular, irritabilidade reflexa e cor da pele. "
             "Cada critério recebe nota de 0 a 2. Pontuação ≥7 indica boa vitalidade.",
        source="protocolo_neonatal.pdf",
    ),
    Document(
        text="A hipertensão gestacional é definida como pressão arterial ≥140/90 mmHg após "
             "20 semanas de gestação, sem proteinúria. Difere da pré-eclâmpsia pela ausência "
             "de comprometimento de órgãos-alvo.",
        source="ubs_completo.pdf",
    ),
    Document(
        text="O aleitamento materno exclusivo é recomendado pela OMS até os 6 meses de vida. "
             "O leite materno fornece anticorpos, nutrientes e fatores de crescimento "
             "essenciais para o desenvolvimento do recém-nascido.",
        source="caderno_atencao_basica.pdf",
    ),
    Document(
        text="A triagem neonatal (teste do pezinho) deve ser realizada entre o 3º e 5º dia "
             "de vida e detecta doenças como fenilcetonúria, hipotireoidismo congênito, "
             "anemia falciforme, fibrose cística e deficiência de biotinidase.",
        source="protocolo_neonatal.pdf",
    ),
    Document(
        text="A caderneta de saúde da criança registra o crescimento e desenvolvimento "
             "infantil. O perímetro cefálico, peso e altura são monitorados em consultas "
             "periódicas e comparados com as curvas da OMS.",
        source="caderno_atencao_basica.pdf",
    ),
]


async def main():
    debugger = RAGDebugger(top_k=3)

    print("Computando embeddings para os documentos de exemplo...")
    for doc in DOCS:
        debugger.add_chunk(doc)
    await debugger.embed_all_chunks()

    queries = [
        "Qual o escore usado para avaliar o bebê ao nascer?",
        "Quando deve ser feito o teste do pezinho?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        result = await debugger.query(query)
        print(result.summary())


if __name__ == "__main__":
    asyncio.run(main())
