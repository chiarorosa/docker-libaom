# 4 Projeto de atributos e política de poda

Esta seção apresenta o projeto do vetor de atributos que alimenta os podadores
propostos nesta tese e a política de poda que converte a saída do modelo em uma
decisão executável dentro do codificador. São descritos a decomposição do vetor
em cinco blocos, a composição exata de cada conjunto avaliado, o custo de obter
cada bloco, as ações da política e os limiares que as controlam, a descoberta
experimental que reorientou o trabalho do modelo para a política, e a taxonomia
de duas dimensões que organiza a família de podadores. A descrição parte da
fonte única de verdade dos atributos, `src/scripts/partition_model/features.py`,
cujas fórmulas o lado C espelha com paridade verificada.

## 4.1 O vetor de atributos e sua decomposição em cinco blocos

O vetor completo possui quarenta e um atributos, organizados em cinco blocos
rotulados no próprio código que os produz. O bloco **A**, com **vinte e quatro
atributos** (índices 0 a 23), reúne os descritores de luminância do bloco e do
seu contexto hierárquico — variância global e por quadrante, dispersão e
heterogeneidade entre quadrantes, energia de gradiente horizontal e vertical e
sua orientação, perfis de somas de linhas e de colunas, aresta mais forte,
densidade de arestas fortes e nível DC —, o contexto hierárquico do bloco-pai de
dimensão 2n×2n e o contraste com os três blocos-irmãos, acrescidos de dois
descritores que não derivam de luminância: o índice de quantização normalizado e
a posição do nó dentro da unidade de 64 px. O rótulo "bloco de pixels", usado no
código e nos demais documentos, descreve portanto vinte e dois destes vinte e
quatro atributos. O índice de quantização normalizado (`q_norm`, índice 17,
igual a `qindex/255`) pertence conceitualmente ao bloco C e permanece em A por
razão histórica, uma vez que o vetor de pixels foi congelado antes da
introdução dos blocos seguintes e realocá-lo romperia a paridade com o lado C;
a posição na unidade de 64 px (`pos_r`, `pos_c`, índices 22 e 23) é endereço
geométrico dentro do superbloco, que permite ao modelo distinguir os nós de
borda, onde o contexto causal está truncado. Convém ainda separar o contraste
hierárquico do contexto de vizinhança: o contraste com o pai e com os irmãos é
aritmética sobre luminância num suporte espacial maior, e não a partição que
esses blocos escolheram — esta última é o bloco B. O traço comum aos vinte e
quatro não é a natureza do dado, e sim o instante em que está disponível: todos
são computáveis antes de a busca de taxa-distorção do nó começar.
O bloco **B**, com **oito atributos** (24 a 31), carrega a vizinhança de
particionamento causal: disponibilidade dos vizinhos acima e à esquerda, larguras
e alturas em log2 dos blocos já decididos nessas direções, granularidade relativa
da vizinhança e sua anisotropia. O bloco **C**, com **quatro atributos** (32 a
35), reúne quantização e posição: `log1p(dc_q²/256)`, que é o passo de
dequantização efetivo do coeficiente DC e não se confunde com o índice de
quantização normalizado alojado no bloco A; a posição normalizada de linha e de
coluna **no quadro**, distinta da posição na unidade de 64 px; e a profundidade
do nó.

O bloco **D**, com **dois atributos**, e o bloco **E**, com **três atributos**,
completam o vetor. O bloco E é o custo de taxa-distorção real da partição
`PARTITION_NONE` — `log1p` da taxa, da distorção e do custo RD — e só existe
**depois** que o codificador avaliou esta partição para o nó corrente, o que o
torna estruturalmente indisponível a qualquer podador que atue antes da busca. No
vetor completo o bloco E ocupa os índices 38 a 40, e no vetor implantado de
trinta e nove, em que o bloco D é descartado, ele ocupa os índices 36 a 38.

Os conjuntos avaliados são recortes deste vetor. O `pixels24` é o bloco A
isolado, com **vinte e quatro atributos**; o **H9a**, conjunto do podador
pré-busca implantado, é a soma A+B+C, com **trinta e seis atributos**; o **H9c**
é a soma A+B+C+E, com **trinta e nove atributos**; e o **H9d** consome o **mesmo
vetor de trinta e nove atributos** do H9c, no mesmo ponto de enganche, diferindo
dele somente na ação que executa. Existe ainda um recorte de trinta e oito
atributos, o H9b, que acrescenta o bloco D ao H9a e foi avaliado apenas fora do
codificador.

> **Procedência.** `src/scripts/partition_model/features.py`, rótulos do layout
> (196–201), `H9_SUBSETS` (214–220), `NUM_FEATURES_H9A = 36` (204) e
> `NUM_FEATURES_H9C = 39` (382); `docs/SINTESE_resultados_metodologia.md` §2.5;
> `docs/ARQUITETURA_pruner_implantado.md` §3.

## 4.2 O bloco D: a especificação e a implementação divergem

A auditoria de código realizada em 26 de julho de 2026 estabeleceu que o bloco D
**implementado não é o bloco D especificado**. O registro desta divergência é
obrigatório.

A especificação, fixada em `PLANO_H9_contribuicao_tese.md:326`, define
`satd_pred` como o SATD do **resíduo de uma predição intra barata construída a
partir dos vizinhos reconstruídos** — PAETH, ou o melhor entre DC, PAETH e dois
modos direcionais. Define ainda `satd_gain` como a diferença normalizada entre o
SATD da fonte e o SATD desse resíduo. O plano advertia que um resíduo de predição
constante possui a mesma variância do bloco e nada acrescentaria ao bloco A.

A implementação em `features.py:252-260,305-308` calcula `block_satd(block)`, ou
seja, a transformada de Hadamard do **bloco-fonte**, sem predição alguma e sem
tocar em vizinho algum.

O bloco D implementado é, deste modo, estatística exclusivamente da fonte,
correlacionada com a variância e com os gradientes que o bloco A já contém. O seu
resultado nulo na etapa de validação de sinal — 46,9 / 50,5 / 53,4 / 57,8 contra
47,0 / 49,7 / 52,6 / 57,3 do H9a — era esperado e **não informa sobre a hipótese
que o plano formulou**.

A hipótese especificada foi testada posteriormente, sob o nome de bloco D', com
três atributos que escolhem o melhor preditor entre DC, V, H e PAETH e medem a
fração de energia AC removida pela predição a partir dos vizinhos. O resultado
também é negativo no nível de 16 px, justamente o mais numeroso, o que fecha o
domínio de pixels por uma via a mais em vez de reabri-lo. O bloco D permanece,
portanto, excluído de todos os conjuntos implantados.

> **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §3 e §6;
> `docs/PLANO_H9_contribuicao_tese.md:326,330-334`;
> `src/scripts/partition_model/features_intrapred.py`; artefatos
> `results/models/gate_intra_pred.csv` e `_sweep.csv`.

## 4.3 A correção de composição: o H9a contém o `pixels24`

Uma precisão de composição atravessa todo o restante deste texto e corrige três
afirmações anteriores dos documentos do projeto. O conjunto `pixels24` é definido
como `list(range(24))` e o H9a como `list(range(36))`. Ou seja: **o `pixels24` é
literalmente o bloco A do H9a**.

Os dois braços comparados na etapa de validação de sinal e no crivo A5 **não são
conjuntos disjuntos** — um contém o outro. Toda comparação entre `pixels24` e H9a
mede, portanto, o retorno dos **doze atributos adicionais** de vizinhança de
particionamento, quantização e posição **sobre** os vinte e quatro descritores de
luminância. Jamais o retorno de um domínio de informação contra outro.

A consequência é uma releitura da hierarquia medida no crivo A5, em `cost_red`
casado de 25%. A variância isolada atinge `reg_frac` de 0,0573. O `pixels24`,
acrescentando vinte e três descritores de luminância, atinge 0,0121 — um ganho de
4,7×. O `convnext_ce`, acrescentando 28,1 milhões de parâmetros sobre pixels
crus, atinge 0,0207, isto é, 0,6×, o que significa piora. O H9a, acrescentando os
doze atributos de vizinhança, quantização e posição, atinge 0,0036 — um ganho de
3,4×.

A leitura correta não é a de que contexto de taxa-distorção vence pixels. É a de
que descritores manuais compactos vencem uma rede convolucional profunda sobre
pixels crus, e a de que o que separa o conjunto campeão do restante não é
capacidade de modelo, e sim contexto causal de vizinhança.

Registre-se, com o mesmo rigor, que **o podador implantado é majoritariamente um
modelo de pixels**, uma vez que vinte e quatro dos seus trinta e seis atributos
são descritores de luminância. Registre-se também que ele **não contém grandeza
alguma de custo de taxa-distorção**, pois estas existem apenas no bloco E.

> **Procedência.** `docs/RESULTADOS_auditoria_dominio_pixels.md` §2, §2.1 e §7;
> `features.py:204,216`; `docs/RESULTADOS_convnext_regret.md` §2;
> `docs/SINTESE_resultados_metodologia.md` §2.5, convenção de nomenclatura de
> "contexto RD".

## 4.4 O custo de obtenção de cada bloco

O critério que governou a inclusão de cada bloco é o custo de obtê-lo no ponto
exato do fluxo de controle em que o podador é consultado.

Os blocos **B** e **C** são **leitura grátis de estado já residente na memória do
codificador**. Os tamanhos das partições vizinhas vêm de `xd->above_mbmi` e
`xd->left_mbmi`, que a busca nativa já mantém e já consulta. A quantização vem de
`dc_q`. A posição e a profundidade vêm de `mi_row`, `mi_col` e da dimensão do nó.
Nenhum desses doze atributos exige aritmética sobre pixels.

O bloco **A** também não exige nova instrumentação, uma vez que os pixels do
bloco, do bloco-pai e dos irmãos já estão no quadro-fonte, e o pai é obtido por
aritmética de ponteiros a partir das coordenadas do nó. Ele exige, contudo,
**cômputo**: somas inteiras, variâncias, diferenças de gradiente e perfis de
linha e de coluna sobre os n² pixels do bloco. Este custo é linear na área e foi
aceito porque uma única avaliação de candidato na busca RD já percorre a mesma
área com operações muito mais caras.

O bloco **D** exige uma transformada de Hadamard por nó. O bloco D' exigiria,
além dela, uma predição intra por nó — custo barato, mas não nulo, e não
justificado por ganho de sinal algum.

O bloco **E** é o extremo oposto. Ele exige a **avaliação completa de
taxa-distorção da partição `PARTITION_NONE`**, o que significa que o podador que
o consome já pagou este custo e jamais poderá economizá-lo.

> **Procedência.** `docs/ARQUITETURA_pruner_implantado.md` §2 e §3;
> `src/scripts/partition_model/features.py:263-324`;
> `docs/SINTESE_resultados_metodologia.md` §2.5 e §2.8.

## 4.5 A política de poda: ações e limiares

O modelo implantado emite uma distribuição *softmax* de três classes,
`[P(NONE), P(SPLIT), P(REST)]`, por uma rede perceptron de múltiplas camadas de
topologia 36 → 64 → 32 → 3, instanciada uma vez por tamanho de bloco de 64, 32 e
16 px e executada pela função nativa `av1_nn_predict`. O bloco de 8×8 não é
modelado, pois é terminal e a poda nativa já o trata. A política pré-busca aplica
três ações em cascata, controladas por três limiares τ:

• **Compromisso com `PARTITION_NONE`** – Se `P(NONE) > τ_none`, todos os *splits*
são desabilitados e o nó é decidido como `PARTITION_NONE`, o que corta a
subárvore inteira sem recursão e sem avaliação de candidatos. É a alavanca
primária de economia de tempo.

• **Forçar a *split* quadrada** – Se, não satisfeita a condição anterior,
`P(SPLIT) > τ_split`, os candidatos de forma do nó são ignorados e a busca recorre
apenas nos quatro filhos.

• **Desativação das retangulares por P(REST)** – Se, não satisfeitas as duas
condições anteriores, `P(REST) < τ_rest`, a busca entre `PARTITION_NONE` e *split*
é preservada e os candidatos retangulares são desabilitados. Por dependência de
`do_rectangular_split`, esta única chamada desabilita HORZ e VERT e, com eles,
também as partições estendidas AB e 4-way.

Não satisfeita nenhuma das três condições, o nó segue para a busca completa. A
política **apenas remove candidatos e jamais força um candidato ilegal**, de modo
que o fluxo permanece válido e o formato do *bitstream* não muda.

O podador pós-`PARTITION_NONE` da variante H9c aplica uma ação **binária**: se
`P(NONE) > τ`, a busca encerra ali. Isso é seguro justamente porque a partição já
foi avaliada. O H9d, por sua vez, aplica uma ação **seletiva**, decidindo se vale
avaliar as partições estendidas, sem tocar em `PARTITION_NONE`, nas retangulares
ou na *split*.

O vetor de limiares τ é, literalmente, o ponto de operação da solução. Baixar
`τ_none` faz o codificador aceitar a decisão do modelo com menos confiança,
cortar mais subárvores e ceder mais taxa BD. Subir `τ_none` faz o oposto.

O ponto equilibrado, com `τ_none` de 0,95 e `τ_rest` de 0,20, obteve cerca de
0,46% de taxa BD a aproximadamente 26,5% de redução de tempo. O ponto agressivo,
com 0,60 e 0,40, obteve de 1,4% a 2% de taxa BD a 48–57% de redução. São dois
ajustes do mesmo controle, e não dois modelos.

Uma varredura fina do limiar global, com oito valores entre 0,55 e 0,96,
verificou que a fronteira resultante é **densa e contínua**. O maior intervalo de
aceleração entre valores vizinhos, em vinte e uma vizinhanças medidas, é de
0,15×, e a maioria fica em torno de 0,03×, com taxa BD suave e monótona. A
decisão dura por limiar funciona, deste modo, como um controle efetivamente
contínuo do ponto de operação, sem retreino.

> **Procedência.** `docs/ARQUITETURA_pruner_implantado.md` §1, §4, §5 e §6;
> `src/aom/av1/encoder/partition_strategy.c`, `student_prune_partition` e
> `student_get_taus`; `docs/SINTESE_resultados_metodologia.md` §2.7;
> `docs/RESULTADOS_C5_fronteira_tau.md` §2 e §3; script
> `src/scripts/benchmark/c5_fine_tau.py`; dados em
> `results/benchmark/c5_finetau/raw.csv`.

## 4.6 Limiares por nível de tamanho de bloco

Os limiares são mantidos **por nível de tamanho de bloco**, em uma tabela indexada
por 16, 32 e 64 px, e a justificativa é experimental. A decomposição por remoção
de um nível de cada vez, a partir do ponto implantado — `τ_none` de 0,85, 0,80 e
0,80 em 16, 32 e 64 px, `τ_split` de 0,90 e `τ_rest` de 0,20 —, mediu eficiências
radicalmente distintas entre os níveis.

O nível de 64 px é a alavanca agressiva e cara. Desligá-lo na sequência Jockey
derruba a taxa BD de 1,412% para 0,111%, ou seja, remove cerca de 92% do custo de
qualidade ao preço de apenas 0,223× de aceleração. Isso corresponde a 5,84 pontos
percentuais de taxa BD por 1× de aceleração.

O nível de 32 px é o mais eficiente. Ele chega a render 0,391× de aceleração na
sequência RaceNight com uma variação de taxa BD de −0,009 ponto percentual, isto
é, praticamente grátis. O nível de 16 px, por sua vez, contribui pouco em ambos
os eixos.

O ganho, portanto, **não vem de um único nível**, e a dependência do conteúdo é
forte o bastante para que um limiar único por ação seja subótimo.

O mesmo padrão se repetiu na calibração do podador das partições estendidas. Um
limiar global de 0,30 melhorava duas sequências e piorava a terceira em 0,12
ponto percentual de taxa BD. A correção foi a adoção de limiares por nível —
0,091 em 16 px, 0,103 em 32 px e 0,014 em 64 px —, que recuperou a sequência
prejudicada sem abrir mão do ganho nas demais.

> **Procedência.** `docs/RESULTADOS_C2_sweep_niveis.md` §2, §3 e §4, script
> `src/scripts/benchmark/c2_level_sweep.py`, dados em
> `results/benchmark/c2_levels/raw.csv`;
> `docs/SINTESE_resultados_metodologia.md` §5-quater;
> `docs/RESULTADOS_H9d_etapa3_encoder.md`.

## 4.7 A alavanca está no espaço de ações, não no modelo

A análise pré-H7 investigou se existiria, na arquitetura convolucional ou na rede
perceptron, algo não explorado capaz de viabilizar uma redução de busca
substancialmente maior. A resposta obtida é que a maior alavanca **não é o
modelo**. São o **espaço de ações da política de poda** e a **métrica de custo**
usada para escolher o ponto de operação.

Em `cpu-used=0`, cada nó de dimensão igual ou superior a 16 px avalia até **dez
candidatos de forma** — a contagem verificada no código e resolvida na Seção 1.1.
Destes, **oito dos dez** são retangulares, AB ou 4-way, o que corresponde a cerca
de 77% do custo modelado. A política então vigente usava apenas duas das três
saídas do modelo e eliminava somente subárvores, de modo que um nó não podado
pagava os dez candidatos integralmente.

A métrica de simulação, por sua vez, contava **nós** eliminados, e não custo de
busca. Ela era, por construção, cega a qualquer ação em nível de candidato: o
ponto de operação de 8,5% de redução media outra grandeza que não o tempo.

Corrigidos os dois itens — a ação de desativação das retangulares por P(REST) e a
métrica de custo ponderado, somadas aos limiares por nível, ao contexto
hierárquico do bloco A e ao aumento de capacidade da rede —, a simulação passou a
indicar 34,7% de redução de custo, com 0,01% de *split* perdida e 1,5% de
desativações indevidas de retangulares, contra os 9% a 10,5% da política
anterior.

No codificador real, a curva medida vai de 0,25% de taxa BD a 1,03× de aceleração
no ponto conservador até 1,62% a 1,29× no ponto agressivo. A poda de retangulares
contribui no regime seguro, ao passo que o compromisso com `PARTITION_NONE` domina
o regime agressivo. Esta descoberta justifica o peso que esta tese atribui ao
projeto da política, e não apenas ao do modelo.

> **Procedência.** `docs/PREH7_analise_alavancas.md` §1, §2, §3, §5 e §6;
> `src/aom/av1/encoder/partition_search.c:4090,4149,5077`. Ressalva registrada na
> fonte: as tabelas da §4 daquele documento foram geradas sobre luminância
> corrompida e estão superadas pelas da §6.

## 4.8 O espaço de projeto: duas dimensões ortogonais

As três soluções implantadas são frequentemente lidas como três níveis de
sofisticação de uma mesma ideia. Não são: ocupam pontos distintos de um plano de
**duas dimensões independentes**.

A primeira dimensão é **quando** o podador age, ou seja, qual é o seu ponto de
enganche em `av1_rd_pick_partition` e, por conseguinte, que informação já foi
paga. O H9a age em `av1_prune_partitions_before_search`, antes de qualquer
avaliação. O H9c e o H9d agem em `av1_prune_after_none`, depois da avaliação de
`PARTITION_NONE`.

A segunda dimensão é **o que** o podador poda: a cascata de três ações no H9a, o
encerramento binário da busca no H9c e a ação seletiva sobre as partições
estendidas no H9d.

Esta taxonomia prediz quais podadores se somam, e é essa a sua utilidade
metodológica. O H9c e o H9d são idênticos nas duas primeiras coordenadas — mesmo
gancho e mesmo vetor de trinta e nove atributos — e diferem somente na ação. Quem
difere em espécie é o H9a, o único que decide antes de qualquer custo de
taxa-distorção ter sido pago.

Cada posição no plano carrega uma consequência econômica direta.

O H9a possui o maior alcance e a maior cegueira. Um acerto elimina a subárvore
inteira, **incluindo a própria avaliação de `PARTITION_NONE`** — é o único capaz
de economizar aquele custo. Em troca, decide sem ver um único número de
taxa-distorção.

O H9c possui limite superior estruturalmente restrito. Agindo depois da
avaliação, jamais economiza esse custo e só alcança o que vem depois, o que
explica a sua redução de tempo isolada de cerca de 4%: mais informação, menos
alcance.

O H9d compartilha a limitação de alcance do H9c, mas mira um conjunto de
candidatos que nenhum dos outros dois visava e que consome **34,3% do custo de
busca**, com mínimo medido de 28,9%. A taxonomia antecipa, deste modo, que o H9d
se **soma** ao H9a em vez de competir com ele. A medição confirmou: 1,02 ponto
percentual adicional de redução de tempo ao custo de 0,018 ponto percentual de
taxa BD, sobre o H9a já implantado.

Cabe registrar uma fonte recorrente de confusão de nomenclatura. Os rótulos H9a,
H9b e H9c nomeiam **conjuntos de atributos**, ao passo que H9d nomeia uma
**ação**. São dois esquemas de nomeação convivendo na mesma família, o que faz o
H9d parecer o quarto degrau de uma escada de informação quando é, na verdade, uma
coordenada do outro eixo.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §2.8 e §5-quater;
> `docs/RESULTADOS_H9d_etapa2_C.md` §4;
> `docs/RESULTADOS_C1_custo_por_candidato.md`;
> `results/thesis/00_PLANO_capitulos.md` §2.

## 4.9 Fechamento

O projeto de atributos e a política de poda descritos nesta seção só têm valor de
tese sob duas condições. A primeira é que as fórmulas que os definem sejam as
mesmas dos dois lados da fronteira entre o ambiente de treino e o codificador. A
segunda é que o código acrescentado ao libaom não altere em nada o comportamento
do codificador quando desativado.

A seção seguinte apresenta a arquitetura de software que sustenta estas duas
garantias: a paridade bit a bit entre a implementação de referência em Python e a
transliteração em C, e a guarda de compilação que assegura *bitstream*
byte-idêntico à âncora com a solução desligada.
