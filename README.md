![image](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/images/PPGI-ufrj-logo.png)

# Trabalho final em Fundamentos de Ciência de Dados - PPGI/UFRJ

## __Professores:__

|     Sergio Serra	  | Jorge Zavaleta         |
|---------------------|------------------------|
|serra@pet-si.ufrrj.br|zavaleta@pet-si.ufrrj.br|


# Separabilidade Linear de Verbos Psicológicos no XLM-RoBERTa-base

Resumo: Este estudo contém o pipeline completo feito para investigar a separabilidade linear de verbos psicológicos do Português utilizando embeddings do modelo Transformer **XLM-RoBERTa-base** e classificadores lineares.

O objetivo do projeto é avaliar se representações vetoriais de modelos de linguagem capturam nuances sintático-semânticas profundas (como papéis temáticos) a ponto de distinguir classes de verbos psicológicos sem o auxílio de anotação sintática prévia.

Autor: Josué David Praciano [Programa de Pós-Graduação em Informática/UFRJ]

Orientador(a): Vivian dos Santos Silva [Programa de Pós-Graduação em Informática/UFRJ]


## Metodologia do Projeto
Inicialmente, foi construído um dataset manualmente com base nas classes de verbos psicológicos propostas por Cançado (1996). Para a estruturação do corpus, os dados foram divididos em cinco categorias:

* Classe 1: verbos do tipo temer
* Classe 2: verbos do tipo preocupar
* Classe 3: verbos do tipo acalmar
* Classe 4: verbos do tipo animar
* Classe Controle: Verbos de transferência

Foram selecionados 10 verbos por classe e extraídas 50 frases para cada um deles, totalizando 2.500 sentenças. Todos os dados foram coletados a partir do [Corpus do Português](https://www.corpusdoportugues.org/).

O restante do projeto foi estruturado em três etapas principais: extração de embeddings, classificação supervisionada e análise visual.

### 1. Extração de Embeddings
Para a extração das características textuais, utilizou-se o modelo de linguagem pré-treinado **XLM-RoBERTa (base)**. 
* **Estratégia:** Os embeddings de cada sentença foram gerados através da técnica de ***Mean Pooling* global**, calculando a média aritmética de todos os tokens válidos (desconsiderando os tokens de *padding*) ao longo das **últimas 4 camadas do Transformer**.
* script disponível em [extracted_features.py](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/src/extracted_features.py)

### 2. Classificação Supervisionada
Com as features consolidadas, o problema foi tratado como uma tarefa de classificação em 5 classes utilizando os algoritmos **SVM (Support Vector Machine) com Kernel Linear** e **Regressão Logística**.
* **Validação:** Os dados foram divididos na proporção 80/20 (treino/teste) usando amostragem estratificada. Ademais, o processo de treinamento contou com uma **Validação Cruzada de 5 folds (*Stratified K-Fold*)** no conjunto de treino. Adicionalmente, calculou-se o *Silhouette Score* (métrica de cosseno) para avaliar o nível de separabilidade prévia do espaço de features.
* script disponível em [models.py](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/src/models.py)

### 3. Análise Visual
Para inspecionar o comportamento geométrico e o agrupamento das classes teóricas, foi desenvolvida uma análise visual bidimensional.
* **Redução de Dimensionalidade:** Primeiro, calculou-se o vetor médio de embedding para cada um dos verbos únicos (comprimindo suas respectivas 50 frases). Em seguida, aplicou-se o algoritmo **t-SNE (*t-Distributed Stochastic Neighbor Embedding*)** com inicialização em PCA para projetar esses vetores em 2 dimensões. Por fim, o resultado foi exportado em gráficos de dispersão vetoriais (PDF/SVG/PNG) com ajuste automático de rótulos para evitar sobreposição de texto.
* script disponível em [visualization.py](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/notebooks/visualization.py)

-----
## Arquivos disponibilizados
* Dataset bruto em txt e tratado em json
* Dataset pré-processado contendo as 4 classes de verbos psicológicos e a classe de controle
* Resultados da validação cruzada e do teste final obtidos via SVM Linear e Regressão Logística

-----
## Imagens disponibilizadas
* Grafo de proveniência do projeto
![image](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/images/pipeline_provenance.png)
* Gráfico 2d dos embeddinggs via t-SNE
![image](https://github.com/Josue-Praciano/verbos-psicol-gicos-separabilidade-linear/blob/main/images/visualizacao_embeddings_tsne_controle.png)

-----
## Autoria:
* Josué David Praciano, Vivian dos Santos Silva
* Contato: josuepraciano@gmail.com
* Página: https://www.linkedin.com/in/josue-praciano/
 
-----
Artigo: [Separabilidade Linear de Verbos Psicológicos no XLM-RoBERTa-base] (LINK)

## Referências:

* CANÇADO, Márcia. Verbos Psicológicos: Análise Descritiva dos Dados do Português Brasileiro. Revista de Estudos da Linguagem, v. 4, n. 1, p. 89-114, 1996.

## Como citar:

* PRACIANO, J. D.; SILVA, V. S. Separabilidade Linear de Verbos Psicológicos no XLM-RoBERTa-base. GitHub, 2026. Disponível em: <https://github.com/SEU_USUARIO/SEU_REPOSITORIO>. Acesso em: 9 ago. 2026.

## Uso de IA Generativa
Este projeto utilizou o modelo **Gemini 1.5 Pro** (Google) de forma complementar para suporte técnico e desenvolvimento, aplicado nas seguintes tarefas:

* **Documentação do Código:** Geração de comentários explicativos nas funções para facilitar a leitura e manutenção do código.
* **Depuração e Rastreabilidade:** Otimização de marcações e instruções de *print* ao longo do código para mapeamento rápido de falhas e rastreabilidade de erros.

> **Nota:** Toda a concepção conceitual da pesquisa, a fundamentação teórica de linguística computacional e a análise crítica dos resultados do modelo XLM-RoBERTa-base foram de autoria inteiramente humana.

