# Stacked Neural Pruners for AV1 Intra-Frame Partition Search: Low-Cost Causal Context and Measured Rate-Distortion Cost

**Abstract** — The intra-frame block-partitioning search of the AV1 encoder evaluates up to ten candidate shapes per node under a recursive rate-distortion criterion, and the native speed presets that shortcut this search form a coarse, discrete ladder. This paper studies two learned pruners deployed inside the reference encoder — a pre-search pruner, internally designated H9a, and a selective pruner of extended partition shapes, internally designated H9d — and asks under which condition their gains compose. The two-dimensional design space is formalized along *when* a pruner acts (before any rate-distortion evaluation, or immediately after the undivided-block candidate) and *what* it removes from the search. All rate-distortion and timing evidence comes from the Common Test Conditions (CTC) Class A1 grid — eight 4K 10-bit sequences, 15 frames, four quantization points — against the libaom v3.10.0 anchor at `cpu-used=0`. Two compositions that consume identical input are shown to have opposite signs: an existing binary post-decision pruner composes with the pre-search pruner at a measured interaction of −1.9 percentage points against the sum of its parts, whereas a selective pruner of extended partition shapes, sharing the identical insertion point and the identical 39-attribute vector with it, adds 1.02 percentage points of time savings at a cost of 0.018 percentage points of BD-rate — about one third of the price of obtaining the same time by loosening the deployed pruner's own decision threshold. Because the two post-decision pruners differ only in which candidate set they remove, and not in what they observe, additivity is shown to depend on the disjointness of the pruned candidate sets at the operating point at which the pruners actually run, and not on the amount or novelty of the information each consumes. No configuration studied dominates the native convolutional pruner on the joint rate-distortion-time frontier; the measured practical value is an operating point between 12.4% and 21.9% time savings, depending on the sequence, inside the gap left by the first two discrete native steps.

**Index Terms** — AV1, intra-frame prediction, block partitioning, learned search pruning, rate-distortion, encoder complexity reduction, model composability.

---

## I. Introdução

A codificação intraquadro do AV1 organiza cada superbloco sob uma busca recursiva de particionamento que, em cada nó da árvore, avalia o custo de taxa-distorção (do inglês *rate-distortion* – RD) de um conjunto de formas candidatas antes de decidir a divisão. A função `av1_rd_pick_partition` percorre, em ordem fixa, o candidato não dividido (`PARTITION_NONE`), a divisão quadrada recursiva, as duas formas retangulares, as quatro formas assimétricas do tipo AB e as duas formas 4-way de proporção 4:1, totalizando dez formas de partição por nó. Esta busca é exaustiva por construção, uma vez que o codificador só sabe qual forma é a melhor depois de codificar efetivamente o bloco sob cada uma delas, e é justamente esse custo que motiva a poda aprendida do espaço de busca: um modelo consultado nos nós da árvore elimina candidatos improváveis antes que o codificador pague o custo de avaliá-los.

O usuário do codificador de referência já dispõe de um controle de velocidade, a escada de *presets* `cpu-used`, mas essa escada é discreta e esparsa. Sob a definição canônica de redução de tempo adotada neste texto — declarada na Seção V, e nunca a alternativa ponderada pelo tempo, uma vez que as duas divergem em até cerca de três pontos percentuais —, a transição de `cpu-used=0` para `cpu-used=1` já salta para 32,59% de redução de tempo, e degrau intermediário algum existe entre os dois pontos. Um podador aprendido, controlado por um limiar contínuo sobre a saída de um modelo, é capaz de preencher exatamente esse vão.

Este artigo apresenta as duas soluções de poda aprendida implantadas em C no codificador de referência que sobreviveram à validação universal contra o botão de velocidade nativo: o podador pré-busca, aqui designado **H9a**, que decide antes de qualquer avaliação de taxa-distorção, e o podador seletivo de partições estendidas, aqui designado **H9d**, que decide imediatamente após a avaliação do `PARTITION_NONE` se vale avaliar as quatro formas AB e as duas formas 4-way. A tese defendida é de natureza composicional: dois podadores aprendidos se somam na medida em que os conjuntos de candidatos que cada um retira da busca são disjuntos **no ponto de operação em que efetivamente rodam** — e não na medida em que dispõem de informação distinta. A prova apresentada é o **contraste de sinal** entre duas composições que consomem entrada idêntica. Um podador pós-decisão binário, empilhado sobre o pré-busca, compõe com interação **negativa**, de −1,9 ponto percentual contra a soma das partes. O H9d, que partilha com ele o mesmo ponto de inserção e o mesmo vetor de trinta e nove atributos, e difere apenas na ação, soma +1,02 ponto percentual de redução de tempo ao custo de +0,018 ponto percentual de taxa BD — cerca de um terço do preço de comprar o mesmo tempo afrouxando o limiar do podador já implantado. O enunciado é, deste modo, prescritivo: a via para novos ganhos é a procura por ações ainda não disputadas na busca, e não por mais informação de entrada.

O cenário experimental deste artigo é a validação universal contra o botão de velocidade nativo do codificador: as oito sequências da Classe A1 das condições comuns de teste (do inglês *Common Test Conditions* – CTC) da *Alliance for Open Media*, em 4K e dez bits, com os quinze quadros que a própria especificação da CTC determina para o modo intraquadro — e não um recorte próprio deste trabalho —, sob os pontos de quantização `cq-level` 20, 32, 43 e 55, contra a âncora libaom v3.10.0 em `cpu-used=0`.

A estrutura deste artigo está organizada como segue. A Seção II formaliza o espaço de projeto em duas dimensões ortogonais. A Seção III apresenta a decomposição medida do custo de busca por família de candidatos, que motiva o alvo do segundo podador. A Seção IV descreve as duas soluções — atributos, topologia, política e a arquitetura que garante a inércia do código desligado. A Seção V apresenta o protocolo de avaliação. As Seções VI e VII apresentam os resultados na CTC e a fronteira bidimensional que estabelece a cláusula do ponto de operação. A Seção VIII discute limitações declaradas, e a Seção IX conclui.

---

## II. Espaço de Projeto: Quando o Podador Age × O Que Ele Poda

As soluções investigadas neste trabalho não constituem níveis sucessivos de sofisticação de uma mesma ideia. Elas ocupam pontos distintos de um plano definido por duas dimensões independentes: *quando* o podador age, o que determina que informação já foi paga; e *o quê* o podador poda, o que determina quais candidatos deixam de ser avaliados. Explicitar este plano é o que torna previsíveis os resultados de composição apresentados nas Seções VI e VII.

A primeira dimensão é o ponto de enganche no fluxo de controle de `av1_rd_pick_partition`. O gancho `av1_prune_partitions_before_search` executa antes de qualquer avaliação de taxa-distorção e caracteriza a poda pré-busca. O gancho `av1_prune_after_none` executa imediatamente após a avaliação de `PARTITION_NONE` e caracteriza a poda pós-decisão, dispondo, então, da taxa, da distorção e do custo RD reais dessa avaliação.

A segunda dimensão é a ação executada. Um podador pode comprometer o nó com `PARTITION_NONE` e encerrar a descida sem recursão; pode restringir a busca apenas à divisão quadrada; pode encerrar a busca de forma binária depois do `PARTITION_NONE`; ou pode agir seletivamente sobre um subconjunto de candidatos, como as partições estendidas.

As duas dimensões são ortogonais, e é justamente por isso que duas soluções podem partilhar o mesmo gancho e o mesmo vetor de atributos e ainda assim produzir ganhos que se somam: o que as distingue é a ação, isto é, o conjunto de candidatos que cada uma retira da busca. Duas soluções cujas ações se sobrepõem, por outro lado, competem pelo mesmo tempo economizável, e a composição rende menos que a soma das partes. Esta distinção entre sobreposição de informação e sobreposição de ação organiza toda a apresentação dos resultados de composição deste artigo.

---

## III. Onde o Tempo Está: Decomposição do Custo de Busca por Família de Candidatos

A viabilidade de qualquer podador seletivo depende de onde o tempo de busca está de fato alocado, e essa distribuição foi medida antes de qualquer experimento de codificação, com a instrumentação nativa de coleta de estatísticas de particionamento do libaom, sobre 875.317 nós de decisão das três sequências reservadas do conjunto de dados de aprendizado de máquina descrito na Seção V, em codificação intraquadro com `cpu-used=0`. Esta seção apresenta, deste modo, uma **distribuição de custo** desse conjunto de dados, e não medição alguma de taxa BD ou de redução de tempo.

O tempo de trabalho local de um nó foi definido como a soma dos candidatos não recursivos, e o denominador dos percentuais a seguir são os **nove candidatos não recursivos** — `PARTITION_NONE`, as duas retangulares, as quatro AB e as duas 4-way —, e não as dez formas do enumerador de tipos de partição. A coluna da divisão quadrada foi excluída por conter a própria recursão.

A Tabela I apresenta a decomposição agregada. O custo se distribui em três blocos de tamanho comparável: `PARTITION_NONE`, as formas retangulares e as partições estendidas — as quatro formas AB somadas às duas formas 4-way. Individualmente, por outro lado, nenhuma forma estendida ultrapassa 7,22% do tempo, o que explica por que este custo passou despercebido em análises que examinavam candidatos isolados.

**Tabela I** — Decomposição do custo de busca local por família de candidato de partição. 875.317 nós de decisão, três sequências reservadas do conjunto de dados de aprendizado de máquina (UVG 4K), codificação intraquadro com `cpu-used=0`. Denominador: nove candidatos não recursivos. Distribuição de custo, e não medição de taxa BD ou de redução de tempo.

| Família de candidato | % do tempo de busca local |
|---|--:|
| `PARTITION_NONE` | 30,1% |
| Retangulares (HORZ + VERT) | 35,6% |
| AB (HORZ_A, HORZ_B, VERT_A, VERT_B) | 20,4% |
| 4-way (HORZ_4, VERT_4) | 13,9% |
| **Estendidas (AB + 4-way)** | **34,3% (28,9%–41,3% entre sequências)** |

> **Procedência.** `results/thesis/M1_objeto_e_formulacao.md` §1.3; artefato `results/benchmark/partstats*/part_timing*.csv` (não versionado); script `src/scripts/benchmark/analyze_partstats.py`.

O custo das partições estendidas concentra-se quase todo nos blocos grandes, o que restringe e, ao mesmo tempo, define o alcance útil de um podador seletivo. Nos blocos de 8×8 amostras estas formas não se aplicam e o custo é nulo. Em 16×16 amostras elas representam 8,7% do tempo local; em 32×32 e 64×64 amostras, respectivamente 50,0% e 51,0%; em 128×128 amostras, 35,3%. É este bloco de custo — grande em agregado, diluído entre seis candidatos individualmente modestos, e concentrado nos tamanhos de bloco maiores — que motiva o alvo do segundo podador apresentado na Seção IV.

A **Figura 1** apresenta esta decomposição por sequência, com o objetivo de tornar visível, num único gráfico, a soma que nenhuma coluna isolada revela.

**Figura 1 — Decomposição do custo de busca por família de candidato.** *Legenda.* A Figura 1 apresenta, no painel superior, a decomposição do tempo de busca local por família de candidato de partição, em barras empilhadas horizontais, uma por sequência reservada do conjunto de dados de aprendizado de máquina (Jockey, RaceNight, RiverBank, do conjunto UVG) e uma para o agregado; e, no painel inferior, o mesmo tempo agregado desagregado nas seis formas estendidas individuais. Os dois painéis, lidos em conjunto, sustentam a afirmação da Seção III: **nenhuma forma estendida isolada ultrapassa 7,22% do tempo de busca local, ainda que a soma das seis atinja 34,3%** — é a diluição entre candidatos individualmente modestos que explica por que este bloco de custo passou despercebido em análises que examinavam candidatos isolados. *Eixos.* Painel superior — horizontal: % do tempo de busca local (0–100); vertical: sequência; séries: `PARTITION_NONE`, retangulares, AB, 4-way. Painel inferior — horizontal: % do tempo de busca local; vertical: forma estendida, ordenada por custo decrescente; cor herdada da família a que cada forma pertence. *Dado de origem.* `results/benchmark/partstats/part_timing_t1.csv`, `results/benchmark/partstats_racenight/part_timing.csv`, `results/benchmark/partstats_riverbank/part_timing.csv`. *Roteiro.* `src/scripts/benchmark/plot_partstats_fig1.py`, que importa as funções `scan` e `pools` de `src/scripts/benchmark/analyze_partstats.py`, de modo que a figura e a Tabela I derivem do mesmo agregador e não possam divergir. O script emite ainda `figura1_dados.csv`, com os valores plotados, o que torna a figura auditável e dispensa a leitura de valores no gráfico. *Idioma e paleta.* A figura é renderizada **em inglês**, com separador decimal em ponto, uma vez que a submissão é em inglês; e a paleta categórica é a sóbria adotada para todas as figuras destes artigos — azul-aço, vinho, violeta e verde-mata, em ordem fixa —, validada nas cinco verificações computáveis de cor (faixa de luminosidade, piso de croma, separação sob deficiência de visão de cores, piso de visão normal e contraste contra a superfície), todas aprovadas. Acompanha uma variante com textura, em que a identidade das quatro famílias sobrevive à impressão em tons de cinza. *Procedência.* `results/thesis/A2_TABELAS_E_FIGURAS.md`, Figura 1.

---

## IV. As Duas Soluções: Atributos, Política e Arquitetura de Inércia

### A. O vetor de atributos

Os dois podadores consomem recortes do mesmo vetor de atributos, decomposto em cinco blocos rotulados A a E. O bloco A, com vinte e quatro atributos, reúne descritores de luminância do bloco e de seu contexto hierárquico — variância, gradientes, perfis de linha e coluna, contraste com o pai e com os irmãos — além do índice de quantização normalizado e da posição do nó dentro da unidade de 64×64. O bloco B, com oito atributos, carrega a vizinhança de particionamento causal, já residente na memória do codificador. O bloco C, com quatro atributos, reúne o passo de dequantização efetivo, a posição no quadro e a profundidade do nó. O bloco E, com três atributos, é o custo de taxa-distorção real da partição `PARTITION_NONE` — a taxa, a distorção e o custo RD, em `log1p` — e só existe depois que o codificador avaliou esta partição, o que o torna estruturalmente indisponível a qualquer podador pré-busca.

O **H9a** consome os blocos A+B+C, trinta e seis atributos, no gancho pré-busca. O **H9d** consome os blocos A+B+C+E, trinta e nove atributos, no gancho pós-decisão — o mesmo vetor e o mesmo ponto de inserção de um podador pós-decisão binário já existente no mesmo codificador, retomado na Seção VII como termo de contraste da evidência de composição.

### B. A política do H9a e a calibração dos limiares

O H9a emite uma distribuição *softmax* de três classes — `P(NONE)`, `P(SPLIT)`, `P(REST)` — por um perceptrone de múltiplas camadas de topologia 36→64→32→3, um por tamanho de bloco de 64, 32 e 16 amostras, executado pela rotina nativa `av1_nn_predict`. A política aplica três ações em cascata, controladas por três limiares τ: se `P(NONE) > τ_none`, o nó é decidido como `PARTITION_NONE` sem recursão; senão, se `P(SPLIT) > τ_split`, a busca recorre apenas nos quatro filhos; senão, se `P(REST) < τ_rest`, os candidatos retangulares — e, por dependência estrutural, as partições estendidas — são desabilitados. A política apenas remove candidatos e jamais força um candidato ilegal.

O vetor de limiares é, literalmente, o ponto de operação da solução, e essa interpretação como grau de confiança foi verificada, e não suposta. Sobre 1.816.393 nós de decisão do conjunto de dados de aprendizado de máquina descrito na Seção V, o erro esperado de calibração da classe predita é de 0,0112, e a precisão real no limiar implantado de 0,90 é de 95,6% para a classe `NONE` e de 96,5% para a classe `SPLIT`. Estas duas grandezas são estatísticas do conjunto de dados, e não medições de codificação; toda taxa BD e toda redução de tempo deste artigo provêm exclusivamente da grade CTC declarada na Seção V.

### C. H9d: alvo, predizibilidade e a moldura de comparação válida

O alvo do H9d foi escolhido por medição, e não por analogia com o H9a: as partições estendidas consomem 34,3% do custo de busca local, com nenhum podador anterior a mirar este conjunto de candidatos. A viabilidade da solução foi decidida fora do codificador. Sobre 792.840 nós de decisão do conjunto de dados de aprendizado de máquina, um perceptrone de múltiplas camadas sobre os trinta e nove atributos obteve área sob a curva característica de operação do receptor (do inglês *receiver operating characteristic* – ROC) de 0,902 no agregado — contra 0,890 usando apenas os trinta e seis atributos pré-busca —, e evita 69,7% das buscas estendidas perdendo 10% dos vencedores verdadeiros, ou evita 50% perdendo apenas 1,1%. Estas grandezas são estatísticas do conjunto de dados, obtidas por simulação e sem codificar quadro algum.

A cota superior do problema — o desligamento total das partições estendidas em codificações reais — foi medida fora da grade CTC e, por decisão de escopo declarada na Seção V, não é reportada neste artigo. O espaço disponível a um podador seletivo é aqui delimitado exclusivamente pela fração de custo de busca da Tabela I e pela predizibilidade offline acima.

A moldura de comparação correta para julgar o H9d não é evidente, e a sua identificação é, em si, uma contribuição metodológica deste trabalho. Comparar o desligamento total das partições estendidas diretamente contra o codificador nativo dupla-conta o tempo que o H9a já colhe sozinho, uma vez que os nós que o H9a compromete com `PARTITION_NONE` jamais alcançariam a avaliação das formas estendidas. Comparar o H9d como substituição direta do H9a, por sua vez, supõe, de forma incorreta, que as duas soluções disputam o mesmo lugar no fluxo de controle. A moldura válida decorre da ordem de execução: o H9d só age no resíduo de nós que o H9a não comprometeu, e por isso a medição correta é **marginal** — H9a somado ao H9d contra o H9a isolado, no mesmo binário — e o teste de mérito é a **não-dominância de Pareto contra a curva de limiares do próprio H9a**, que é o mecanismo de aceleração gratuito de que a solução já dispõe. É esta a moldura empregada nas Seções VI e VII.

### D. Arquitetura de inércia e custo próprio

Todo o código dos dois podadores vive sob uma guarda de compilação de valor padrão zero, `PARTITION_ML_STUDENT`, sem a qual o gancho pós-decisão é uma função vazia. Duas verificações sustentam a atribuição de qualquer diferença de tempo à poda. A paridade C-Python, sobre 588 nós únicos, mediu desvio máximo de 0,0×10⁰ nos trinta e seis atributos do H9a, dentro das tolerâncias fixadas de 2×10⁻⁶ por canal de atributo. A garantia de inércia, com a guarda de compilação desligada, produziu fluxo de bits byte a byte idêntico ao da âncora, confirmado por resumo criptográfico e por decodificação bem-sucedida. Para o H9d especificamente, a exportação dos pesos treinados foi verificada por comparação de ida e volta sobre 192 vetores aleatórios, com diferença máxima de 1,35×10⁻⁷ entre a saída da implementação de treino e a do arranjo gravado no codificador — exata até o erro de ponto flutuante.

O custo computacional próprio do podador implantado, extração de atributos e inferência somadas, foi medido dentro de codificações reais e é de, no máximo, **0,32% do tempo de codificação**. Este número, e não a razão entre o custo de inferência isolada dos dois modelos por chamada, é o que importa: o custo de inferência não é alavanca em direção alguma, e a totalidade do ganho de tempo medido provém das decisões de poda.

---

## V. Protocolo de Avaliação

Todas as medições seguem um protocolo congelado por commit, antes de existir qualquer resultado das soluções aqui defendidas.

Dois conjuntos distintos sustentam este artigo, e a separação entre eles é estrita. O **conjunto de dados de aprendizado de máquina**, empregado no treino dos dois modelos e em todas as estatísticas offline citadas nas Seções III e IV, foi extraído de dezesseis sequências do conjunto UVG em 4K, particionadas por sequência em treino, validação e teste reservado, sem vazamento de conteúdo entre as três parcelas. A **grade de avaliação** é a das condições comuns de teste, descrita a seguir, e é a única fonte de todo valor de taxa BD e de redução de tempo apresentado neste artigo. Nenhum número de taxa BD ou de redução de tempo medido sobre as sequências UVG é reportado aqui, ainda que existam medições nesse conjunto: a decisão de escopo é a de que a eficiência de compressão e o custo de codificação sejam julgados exclusivamente sob o protocolo de avaliação da indústria.

Todo resultado é reportado sob três pilares contra a mesma âncora, o libaom v3.10.0 em `cpu-used=0`: a **taxa BD** sobre PSNR-Y, pelo método de Bjøntegaard; a **redução de tempo**, denotada TS; e a **aceleração**. **A definição de redução de tempo empregada em todo este artigo é a canônica**: calcula-se, para cada ponto de quantização, a fração de tempo poupada `1 − t_configuração/t_âncora`, tira-se a média sobre os quatro pontos e, então, a média sobre as sequências. Uma definição alternativa, ponderada pelo tempo, coexiste na literatura interna deste projeto e diverge da canônica em até cerca de três pontos percentuais, chegando inclusive a reordenar o ranqueamento entre configurações; ela não é utilizada em número algum deste texto.

A resolução da comparação pareada de tempo foi medida diretamente, e não suposta, por cinco repetições intercaladas em execução contínua: o coeficiente de variação mediano do tempo bruto é de 0,28%, com máximo de 0,64%. Isso estabelece uma resolução efetiva, tomada como dois desvios, de aproximadamente **0,46 ponto percentual** para a configuração equilibrada do H9a e de aproximadamente 0,18 ponto percentual para o *preset* nativo `cpu-used=1`. Diferenças de redução de tempo inferiores a esta resolução não são citadas como positivas em ponto algum deste artigo.

A viabilidade do H9a foi decidida por uma cascata de critérios de decisão fixados antes da medição que os resolveria, de modo que o custo caro de integração em C e de codificação real só fosse pago depois que o sinal se provasse fora do codificador. A ablação que atribui o ganho medido ao modelo, e não à política de poda em si — braços de pontuação alternativos sob política idêntica —, foi conduzida sobre o conjunto de dados de aprendizado de máquina, fora da grade CTC, e os seus valores não são reportados neste artigo, pela mesma decisão de escopo declarada acima. Cabe registrar que o critério que exigia dominância direta em taxa BD contra a heurística trivial de variância, à aceleração casada, **não foi atingido na sua forma estrita** naquela ablação, e que este artigo não invoca a atribuição do ganho ao modelo como premissa de resultado algum das Seções VI e VII, que são comparações entre configurações do próprio codificador sob a mesma âncora.

---

## VI. Resultados na CTC

A Tabela II apresenta os pontos de operação medidos nas oito sequências da Classe A1 das condições comuns de teste, todos contra a mesma âncora e na definição canônica de redução de tempo.

**Tabela II** — Resultados na CTC, média sobre as oito sequências da Classe A1. Âncora libaom `cpu-used=0`; taxa BD sobre PSNR-Y; redução de tempo na definição canônica.

| Configuração | Taxa BD | Redução de tempo | Aceleração |
|---|--:|--:|--:|
| H9a equilibrado | +0,568% | 17,72% | 1,223× |
| H9a equilibrado + H9d (PL10, implantado) | +0,586% | 18,74% | 1,238× |
| H9a agressivo | +1,403% | 31,51% | 1,492× |
| libaom `cpu-used=1` | +0,449% | 32,59% | 1,508× |
| libaom `cpu-used=2` | +0,536% | 42,72% | 1,788× |
| libaom `cpu-used=3` | +2,722% | 67,94% | 3,159× |

> **Procedência.** `results/thesis/R2_h9a.md` §2.4 (linhas H9a e presets), `results/thesis/R4_h9d.md` §4.6 (linha H9a+H9d); artefatos `results/benchmark/fase6/bdrate_average.csv` (linhas `ml_balanced`, `ml_aggr`, `native_cpu1`, `native_cpu2`, `native_cpu3`) e `results/benchmark/fase6/raw_results.csv` (configuração `ml_bal_h9d`); scripts `src/scripts/fase6/{encode_ctc.py, report_ctc.py, ctc_h9d.py, ctc_h9d_marginal.py}`.

Nenhuma configuração de aprendizado de máquina desta tabela domina os *presets* nativos nos dois eixos do compromisso simultaneamente: o *preset* `cpu-used=1` entrega mais redução de tempo que o H9a equilibrado a taxa BD menor, e o *preset* `cpu-used=2` faz o mesmo contra o H9a agressivo. **Nenhum ponto de aprendizado de máquina proposto domina a rede convolucional nativa de poda de partição.** O valor prático medido é outro. O ponto equilibrado do H9a opera, conforme a sequência, entre **12,4% e 21,9% de redução de tempo** — de 12,35% na FoodMarket2 a 21,93% na Neon1224 —, e essa é justamente a região que a escada discreta dos *presets* não oferece entre `cpu-used=0` e `cpu-used=1`. Nela o custo é baixo: na sequência NocturneDance, o ponto equilibrado registra +0,15% de taxa BD a 12,6% de redução de tempo. O limiar é, ademais, um controle contínuo por construção, uma vez que é um número real aplicado à saída do modelo, ao passo que o *preset* é um índice inteiro; a densidade da fronteira entre os dois pontos de operação medidos não foi, por outro lado, caracterizada sobre esta grade, e este artigo não a afirma.

A contribuição marginal do H9d sobre o H9a equilibrado é de **+1,02 ponto percentual de redução de tempo ao custo de +0,018 ponto percentual de taxa BD**, o que corresponde a um preço de 0,018 ponto percentual de taxa BD por ponto percentual de tempo economizado. Este preço é diretamente comparável ao único mecanismo alternativo de que o usuário da solução já dispõe: afrouxar o limiar do próprio H9a. O segmento entre o ponto equilibrado e o ponto agressivo desse controle custa 0,063 ponto percentual por ponto percentual. O H9d compra tempo, por conseguinte, por cerca de **um terço** do preço do botão de limiar — aproximadamente 3,5 vezes mais barato, sob o estimador de interpolação por sequência; a média das médias sobre as oito sequências dá 3,38 vezes, e os dois estimadores concordam a cerca de 4%.

O teste de não-dominância de Pareto contra a curva de limiares do H9a, por sequência, confirma o agregado: o H9d é melhor em seis das oito sequências, perdendo levemente apenas nas duas sequências — Crosswalk e Neon1224 — em que o ganho de tempo do H9d é quase nulo, aquelas em que o eixo estendido praticamente não é exercido. Em duas sequências, FoodMarket2 e Tango, a dominância de Pareto é estrita: a taxa BD cai simultaneamente à economia de tempo.

O custo próprio do podador implantado — extração de atributos e inferência somadas — permanece, em toda a campanha, no máximo em **0,32% do tempo de codificação**, o que garante que o ganho medido acima provém integralmente das decisões de poda.

---

## VII. Fronteira Bidimensional e a Cláusula do Ponto de Operação

Um único ponto de operação não é suficiente para caracterizar a aditividade entre os dois podadores. Uma campanha de 96 codificações novas mediu a família completa de configurações do H9d, cruzando duas bases do H9a — o ponto equilibrado e o ponto agressivo — com duas calibrações do H9d — PL10, a implantada, e PL20, mais agressiva —, sobre o mesmo protocolo CTC. A Tabela III apresenta o preço marginal de cada uma das quatro combinações.

**Tabela III** — Preço marginal (pontos percentuais de taxa BD por ponto percentual de redução de tempo) da fronteira bidimensional do H9d, quatro combinações de base do H9a e calibração do H9d.

| Base do H9a | Calibração do H9d | Preço (pp de taxa BD / pp de redução de tempo) |
|---|---|--:|
| Equilibrada | PL10 (implantada) | **0,0179** |
| Equilibrada | PL20 | 0,0399 |
| Agressiva | PL10 | 0,0329 |
| Agressiva | PL20 | 0,0258 |

> **Procedência.** `results/thesis/R4_h9d.md` §4.7; artefato `results/benchmark/fase6/raw_results.csv` (configurações `ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d`, `ml_aggr_h9d_pl20`, 128 linhas no total); script `src/scripts/fase6/ctc_h9d.py`.

A combinação implantada, PL10 sobre a base equilibrada, é a de menor preço das quatro, o que confirma a escolha de projeto feita antes de a campanha existir. O achado que a fronteira expõe, por outro lado, é o que altera o enunciado teórico deste artigo: **o valor marginal do H9d desaba conforme a base do H9a fica agressiva**. Sobre a base equilibrada, o ganho é de +1,02 ponto percentual de redução de tempo. Sobre a base agressiva, ele cai para **+0,17 ponto percentual** — valor abaixo da resolução temporal medida do arranjo, de aproximadamente 0,46 ponto percentual, e que, por isso, **não é citado como positivo**. Por sequência, o ganho supera a resolução temporal em seis das oito sequências sobre a base equilibrada, e em apenas uma das oito sobre a base agressiva.

O mecanismo é o mesmo que explica a aditividade, lido no sentido oposto. Com o limiar de comprometimento com `PARTITION_NONE` afrouxado, o H9a compromete os nós tão cedo que eles nunca alcançam o critério de decisão das partições estendidas. Os dois podadores passam, deste modo, a disputar o mesmo resíduo, e o H9d volta a se comportar como o podador pós-decisão binário de contraste da Seção IV.

A prova de que a não-aditividade entre podadores é sobreposição de ação, e não limite informacional, está justamente neste contraste. O podador pós-decisão binário e o H9d partilham exatamente o mesmo gancho e exatamente o mesmo vetor de trinta e nove atributos, e por isso consomem entrada idêntica; ainda assim, as duas composições têm **sinais opostos**.

A composição do podador binário com o H9a foi medida por decomposição de três pernas em quatro sequências da grade CTC, com o balanço fechado na forma `tempo(pré-busca) + tempo(pós-decisão) + interação = tempo(empilhados)`. Os dois entregam, empilhados, 14,6% de redução de tempo contra 16,5% da soma das partes: a **interação é negativa em −1,9 ponto percentual** na média, e negativa em três das quatro sequências. Cerca de 12% do ganho potencial evapora na sobreposição. O H9d, sobre a base equilibrada e nas oito sequências, faz o contrário: soma +1,02 ponto percentual e compra tempo por um terço do preço do botão de limiar.

Cabe declarar a assimetria entre as duas medições, que não são pareadas: a interação negativa foi medida em quatro sequências sobre a base do H9a nos seus limiares compilados por padrão, ao passo que o marginal do H9d foi medido nas oito sequências sobre o ponto equilibrado implantado. É por esta razão que o argumento repousa sobre o **sinal** da composição, e não sobre a razão entre as duas magnitudes, que uma comparação pareada entre os dois podadores — não executada neste trabalho — seria necessária para estabelecer.

O contraste de sinal seria inexplicável se a informação partilhada fosse, por si só, o limite da composição, uma vez que os dois consomem entrada idêntica. O que os distingue é somente o conjunto de candidatos que cada um retira da busca: um encerra a busca de forma binária sobre o resíduo inteiro, e por isso disputa com o pré-busca os mesmos blocos fáceis; o outro age seletivamente sobre um conjunto de candidatos — as formas AB e 4-way, 34,3% do custo de busca local — que o primeiro jamais visava.

O enunciado correto é, por conseguinte, prescritivo e recebe uma cláusula final, estabelecida por esta própria fronteira: **dois podadores se somam na medida em que os seus conjuntos de candidatos podados são disjuntos no ponto de operação em que efetivamente rodam**. A disjunção de ação não é propriedade absoluta do H9d; é função do ponto de operação sobre o qual o H9a age. O ponto implantado é justamente aquele em que a disjunção é máxima, o que valida a escolha de projeto por uma razão mais estreita do que a simples afirmação de que as ações são disjuntas.

Cabe, por fim, uma nota sobre a posição do H9d na fronteira global de compromisso entre taxa BD e redução de tempo que reúne todos os *presets* nativos. Nessa fronteira mais ampla, **as quatro configurações do H9d são dominadas**. A leitura correta desta dominância requer, por outro lado, uma precisão adicional: a base do H9a, sozinha, já é dominada exatamente pelo mesmo conjunto de configurações que domina o H9a somado ao H9d, uma vez que o ponto de operação do H9a vive em `cpu-used=0`, regime no qual os *presets* nativos a `cpu-used` 1 e 2 entregam mais redução de tempo por menos taxa BD. **O H9d não perde posição alguma na fronteira global** — ele herda a posição do H9a e a melhora marginalmente dentro dela. O quadro em que o H9d é corretamente julgado permanece o desta seção: a contribuição marginal sobre uma base fixa e a não-dominância contra a curva de limiares do próprio H9a — não a fronteira de todos os *presets* e todos os níveis de `cpu-used`.

A **Figura 2** reúne, num único painel, os dois pontos de operação medidos do H9a, os pontos do H9d empilhado sobre cada um deles e os degraus dos *presets* nativos, o que torna visível a cláusula do ponto de operação estabelecida nesta seção.

**Figura 2 — Taxa BD contra redução de tempo: pontos do H9a, pontos do H9d e degraus nativos.** *Legenda.* A Figura 2 apresenta a taxa BD contra a redução de tempo para os dois pontos de operação medidos do H9a — o equilibrado e o agressivo —, para as quatro configurações do H9d empilhadas sobre eles e para os três *presets* nativos, evidenciando que o ganho marginal do H9d encolhe conforme a base do H9a fica agressiva. O segmento que une os dois pontos do H9a é traçado como **interpolação**, e não como curva medida, uma vez que é exatamente esse segmento que define o preço do botão de limiar reportado na Seção VI; a densidade da curva de limiares entre os dois pontos não foi medida sobre esta grade. *Eixos.* Horizontal: redução de tempo (%), definição canônica. Vertical: taxa BD (%). Séries: pontos do H9a, equilibrado e agressivo, unidos por segmento tracejado de interpolação; quatro pontos do H9d (PL10 e PL20 sobre cada base, marcadores distintos); *presets* nativos `cpu-used` 1, 2 e 3 (marcadores quadrados). *Dado de origem.* `results/benchmark/fase6/bdrate_average.csv`, todas as nove configurações — `ml_balanced`, `ml_aggr`, `ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d`, `ml_aggr_h9d_pl20`, `native_cpu1`, `native_cpu2` e `native_cpu3`. *Roteiro.* Ler as colunas `bd_rate` e `ts_pct` do arquivo agregado, sem recálculo, e traçar num único eixo com `matplotlib`; anotar cada ponto do H9d com o ganho marginal em pontos percentuais contra a sua base. *Procedência.* Usa exclusivamente os dados-fonte já indicados nas Tabelas II e III desta seção.

---

## VIII. Discussão e Limitações

Três limitações medidas restringem explicitamente a leitura dos resultados apresentados.

**A ausência do H9d além de `cpu-used=0` na fronteira global.** O H9d não foi codificado empilhado sobre os *presets* nativos `cpu-used` 1, 2 e 3, apenas sobre `cpu-used=0`, que é o regime em que foi medido e implantado. Fechar esta lacuna exigiria cerca de 192 codificações adicionais — duas bases do H9a por três níveis de *preset* por oito sequências por quatro pontos de quantização —, e há razão medida para esperar rendimento baixo: o H9d mostrou-se inerte sobre a base agressiva do H9a, e os *presets* nativos mais rápidos já podam de forma agressiva, o que tende a encolher o resíduo sobre o qual o H9d atua.

**A inércia do H9d sobre a base agressiva.** A Seção VII estabeleceu, por medição direta, que o ganho marginal do H9d colapsa de +1,02 para +0,17 ponto percentual quando a base do H9a passa do ponto equilibrado para o agressivo, valor abaixo da resolução temporal medida do experimento. A aditividade entre os dois podadores não é, por conseguinte, uma propriedade absoluta do mecanismo de ação disjunta; é condicionada ao ponto de operação em que o H9a efetivamente opera, e o texto deste artigo não cita o +0,17 ponto percentual como resultado positivo em nenhuma passagem.

**A resolução temporal medida é intra-execução.** O desvio de 0,28% de coeficiente de variação mediano, que sustenta a resolução de aproximadamente 0,46 ponto percentual empregada em todo este artigo, foi obtido em cinco repetições numa mesma janela contínua de execução, na mesma sequência da grade CTC e no mesmo contêiner, sem reinício. Esta medição não captura deriva entre dias, reinícios de execução ou estados térmicos distintos, e a leitura de diferenças sequência a sequência deve carregar esta ressalva.

**A restrição de escopo do conjunto de avaliação.** Toda taxa BD e toda redução de tempo deste artigo provêm da grade CTC, conforme a decisão declarada na Seção V, ao passo que o conjunto de dados construído sobre as sequências UVG comparece apenas como origem das estatísticas offline das Seções III e IV. Três resultados existentes ficam, por conseguinte, fora deste texto: a cota superior de taxa BD e de aceleração do desligamento total das partições estendidas; a caracterização da densidade da fronteira ao longo da grade de limiares; e a ablação de atribuição contra fontes de pontuação alternativas sob política casada. A consequência para a leitura é dupla: o espaço disponível ao podador seletivo, na Seção IV-C, fica delimitado somente pela fração de custo de busca e pela predizibilidade offline; e a continuidade da fronteira entre os dois pontos de operação medidos do H9a não é afirmada em passagem alguma.

Duas ressalvas adicionais de enquadramento fecham esta seção. Em configuração alguma testada este artigo declara dominância de um ponto de aprendizado de máquina sobre a rede convolucional nativa de poda de partição; o valor prático apresentado é estritamente a granularidade fina de operação que a escada de *presets* não cobre. E o custo de inferência isolada de cada modelo por chamada não é empregado como argumento de vantagem em passagem alguma deste texto — o número que importa, e que sustenta integralmente os resultados apresentados, é o custo total do podador implantado dentro de uma codificação real, de no máximo 0,32% do tempo de codificação.

---

## IX. Conclusão

Este artigo apresentou duas soluções de poda aprendida do espaço de busca de particionamento intraquadro do AV1, implantadas em C no codificador de referência e validadas contra a escada nativa de *presets* de velocidade sob a grade da Classe A1 das condições comuns de teste. Nenhuma das duas domina a rede convolucional nativa na fronteira conjunta de taxa BD e tempo, e o valor prático medido é o ponto de operação que elas oferecem entre 12,4% e 21,9% de redução de tempo, conforme a sequência, região que a escada discreta dos *presets* não cobre entre os seus dois primeiros degraus.

Por outro lado, a contribuição central é composicional. Ao partilhar o mesmo ponto de inserção e o mesmo vetor de trinta e nove atributos de um podador pós-decisão de contraste — que compõe com interação negativa de −1,9 ponto percentual — e ainda assim somar +1,02 ponto percentual sobre a base implantada, o podador seletivo de partições estendidas refuta, por medição direta, a hipótese de que a não-aditividade entre podadores aprendidos decorreria de um limite na informação que compartilham. O que determina se dois podadores se somam ou competem é a disjunção dos conjuntos de candidatos que cada um retira da busca, e essa disjunção foi medida como dependente do ponto de operação em que os dois efetivamente rodam, e não como propriedade absoluta de nenhum dos dois mecanismos.

O enunciado que resulta é, por isso, prescritivo. Diante de um orçamento fixo de engenharia para acrescentar um novo podador a um sistema que já possui um, a pergunta produtiva não é que informação adicional o novo podador poderia consumir, e sim que conjunto de candidatos, ainda não disputado por nenhuma alavanca existente, ele poderia remover da busca — e em que ponto de operação do sistema já implantado essa remoção continua sendo disjunta.

---

## X. Referências

`[completar: confirmar prazo, formato exato e número de páginas no CFP do LASCAS 2027]`

`[completar: citação formal da especificação AV1/AOM (bitstream e enumeração de tipos de partição)]`

`[completar: citação formal do documento de condições comuns de teste da Alliance for Open Media — CWG-G082 AV2 CTC v9, já referenciado internamente em results/thesis/M3_protocolo_avaliacao.md §3.3]`

`[completar: citação do repositório de referência libaom (versão v3.10.0) e de sua licença]`

`[completar: revisão de literatura sobre poda aprendida de particionamento em codificadores de vídeo (AV1/VVC), a incluir conforme exigência de revisores]`

`[completar: citação do método de Bjøntegaard para cálculo de taxa BD]`

---

## Conformidade

> **Apêndice interno de trabalho — NÃO faz parte da submissão.** Esta seção existe para auditoria interna do texto contra as retratações e lacunas registradas na tese de origem, e deve ser removida antes de qualquer envio. Ela é o único ponto deste arquivo em que decisões editoriais e a existência de outros trabalhos derivados da mesma tese são mencionadas.

**(a) Retratações de `A3_RETRATACOES_E_LACUNAS.md` verificadas e respeitadas neste texto.**

- **R8** — Não se afirma, em passagem alguma, que "alavancas de poda não se somam" na forma geral, nem se invoca limite informacional. O texto usa, do início ao fim, o enunciado corrigido: a não-aditividade é sobreposição de ação (Seções II e VII).
- **R9** — A aditividade do H9d não é apresentada como propriedade absoluta da sua ação. O colapso do ganho marginal sobre a base agressiva (+0,17 pp) é declarado explicitamente como abaixo da resolução medida e **não citado como resultado positivo** (Seção VII e Seção VIII).
- **R10** — A razão de aproximadamente cinquenta vezes entre o custo de inferência isolada por chamada dos dois modelos não é citada como vantagem em passagem alguma. O argumento de custo empregado é, em todo o texto, o custo próprio do podador implantado (≤0,32% do tempo de codificação), conforme a Seção IV-D e reiterado na Seção VIII.
- **R11** — Os quinze quadros da grade CTC são declarados, na Seção I (Introdução), como a especificação da própria CTC para o modo intraquadro, e não como recorte próprio deste trabalho.
- **R13** — Não se afirma, em nenhuma passagem, que o contexto de taxa-distorção barato é "necessário e suficiente" para superar o podador nativo. A Seção VI declara explicitamente que nenhuma configuração domina a rede convolucional nativa e delimita o valor prático à granularidade fina.
- **R14** — O H9a e o H9d não são descritos como «destilados» de um modelo substituto em passagem alguma. A Seção IV-B descreve o H9a como treinado diretamente sobre o vetor de trinta e seis atributos, sem qualquer menção ao termo «professor».
- **R23** — A definição canônica de redução de tempo é declarada explicitamente na Seção V, com a divergência de até três pontos percentuais frente à definição alternativa registrada, e as duas definições nunca aparecem misturadas em tabela ou comparação alguma deste texto.
- **R24** — A leitura da dominância do H9d na fronteira global (Seção VII) segue estritamente o enunciado corrigido: a base do H9a já é dominada antes de o H9d ser somado a ela, e o H9d não perde posição alguma, herdando a do H9a e melhorando-a marginalmente.

**(b) Lacunas declaradas explicitamente neste texto.**

- **L1** — A ausência de codificação do H9d empilhado sobre `cpu-used` 1, 2 e 3 na fronteira global, com o custo de fechamento (~192 codificações) e a razão medida para esperar rendimento baixo, é declarada na Seção VIII.
- **L7** — A inércia do H9d sobre a base agressiva do H9a é declarada como limitação, e não como propriedade positiva, tanto na Seção VII quanto na Seção VIII.
- **L8** — A natureza intra-execução da resolução temporal medida (cinco repetições, uma janela contínua, um único contêiner, sem captura de deriva entre dias) é declarada explicitamente na Seção VIII.
- **L11** — Nenhuma figura deste artigo é um artefato de imagem renderizado; ambas as figuras (Figura 1 e Figura 2) são apresentadas como **especificações** — identificador, legenda, eixos, dado de origem e roteiro —, uma vez que nenhuma figura existe ainda produzida no projeto de origem.
- **Restrição de escopo do conjunto de avaliação** (decisão editorial de 2026-08-13, não é lacuna de `A3`) — taxa BD e redução de tempo são reportadas exclusivamente sobre a grade CTC Classe A1; o conjunto de dados construído sobre as sequências UVG comparece apenas como origem de estatísticas e distribuições. Foram removidos deste texto, por esta razão, quatro resultados antes presentes: a cota superior do desligamento total das partições estendidas (+0,89% de taxa BD a 1,431×, e o marginal de 1,293× a +0,798%); a densidade da grade de limiares (vão máximo de 0,15× em vinte e uma vizinhanças); a ablação de atribuição a política casada (razões de 11, 44 e 94 vezes contra a variância); e o veredito de dominação da cota superior do H9d isolado pelo H9a isolado. A Figura 2 foi respecificada para usar somente dados da grade CTC, e a alegação de granularidade contínua entre `cpu-used=0` e `cpu-used=1` foi substituída pela dispersão medida do ponto equilibrado entre sequências (12,4%–21,9%). A restrição está declarada nas Seções V e VIII, e o resumo em inglês foi realinhado a ela.
- **Retratação do contraste de +0,26 ponto percentual** (auditoria de 2026-08-13, registrada em `docs/RESULTADOS_auditoria_artigos_lascas_iscas.md` §4) — o texto afirmava, no resumo e nas Seções I, VII e IX, que o H9d somava «quatro vezes» o que o podador pós-decisão binário somara, 1,02 contra 0,26 ponto percentual. O rastreio do valor de 0,26 mostrou tratar-se do marginal medido numa **única** sequência (Neon1224), sobre a base do H9a nos limiares compilados por padrão, ao passo que o 1,02 é média sobre **oito** sequências e sobre o ponto equilibrado implantado. Na média das quatro sequências em que a decomposição existe, o mesmo marginal vale +3,74 pontos percentuais, e a razão se inverte. A razão entre magnitudes foi, por conseguinte, retirada de todas as quatro passagens e substituída pelo **contraste de sinal** — interação de −1,9 ponto percentual contra marginal de +1,02 —, com a assimetria de base e de cobertura declarada explicitamente na Seção VII. Nenhuma passagem deste artigo cita mais o valor de 0,26 ponto percentual.

**(c) Marcadores `[completar: ...]` pendentes neste arquivo.**

Todos os seis marcadores pendentes estão listados na Seção X (Referências) e correspondem exclusivamente a itens de formatação de submissão e a citações bibliográficas externas ao projeto, que as regras deste trabalho proíbem inventar: o prazo e o formato exato do CFP do LASCAS 2027; a citação formal da especificação AV1/AOM; a citação formal do documento de condições comuns de teste; a citação do repositório libaom v3.10.0; a revisão de literatura sobre poda aprendida de particionamento em codificadores de vídeo; e a citação do método de Bjøntegaard. Nenhum outro marcador de lacuna numérica foi necessário no corpo do artigo, uma vez que todo número citado provém diretamente de um dos onze documentos-fonte autorizados.
