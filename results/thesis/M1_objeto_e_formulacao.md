# 1. Objeto, escopo e formulação do problema

Esta seção apresenta o objeto de estudo desta tese, a formulação do problema que
dele decorre e os limites de escopo dentro dos quais todos os resultados
apresentados nos capítulos seguintes devem ser lidos. São descritos, nesta ordem,
a decisão de particionamento recursivo do codificador AV1 em predição
intraquadro, a formulação do problema como poda aprendida do espaço de busca, a
decomposição medida do custo de busca por família de candidatos, o espaço de
projeto em que se situam as soluções investigadas e, por fim, o escopo
experimental declarado.

---

## 1.1 O objeto — a decisão de particionamento recursivo

O objeto desta tese é a **decisão de particionamento de blocos do AV1 em
predição intraquadro**, que é um dos maiores custos computacionais do
codificador. A codificação de um quadro é organizada em superblocos, e cada
superbloco é submetido a uma busca recursiva que, em cada nó da árvore de
particionamento, avalia o custo de taxa-distorção (do inglês *rate-distortion* –
RD) de um conjunto de formas de partição candidatas, mantendo a de menor custo e
descendo recursivamente quando a divisão é a escolha vencedora. Esta busca é
exaustiva por construção, uma vez que o codificador só sabe qual forma é a
melhor depois de codificar efetivamente o bloco sob cada uma delas.

O conjunto de formas candidatas por nó é amplo. O codificador avalia
`PARTITION_NONE` (o bloco não dividido), `PARTITION_SPLIT` (a divisão quadrada
em quatro), as duas formas retangulares `PARTITION_HORZ` e `PARTITION_VERT`, as
quatro formas assimétricas do tipo AB (`HORZ_A`, `HORZ_B`, `VERT_A` e `VERT_B`)
e as duas formas 4-way de proporção 4:1 (`HORZ_4` e `VERT_4`). São, ao todo,
**dez formas de partição por nó**.

O espaço de busca cresce, deste modo, combinatoriamente com a profundidade da
árvore. Cada nó que escolhe a divisão quadrada instancia quatro novos nós com o
mesmo leque de candidatos.

A busca está implementada na função recursiva `av1_rd_pick_partition`, que
concentra o fluxo de controle da decisão. Nesta função, os candidatos são
avaliados em ordem fixa: primeiro `PARTITION_NONE`, depois a divisão quadrada,
em seguida as duas retangulares e, por fim, as quatro AB e as duas 4-way. Esta
ordem é relevante para a formulação do problema, pois define exatamente que
informação já foi paga em cada ponto do fluxo, e é sobre ela que se assenta o
espaço de projeto descrito na Seção 1.4.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §1;
> `src/aom/av1/common/enums.h:152–163` (enumeração `PARTITION_TYPE`, com
> `EXT_PARTITION_TYPES` = 10); `src/aom/av1/encoder/partition_search.c:5609`
> (`av1_rd_pick_partition`), `:5829` (NONE), `:5857` (divisão quadrada), `:5889`
> (retangulares), `:5917` (AB) e `:5943` (4-way). *Divergência de registro
> anotada:* a `SINTESE` §1 e o `docs/PLANO_H9_contribuicao_tese.md` referem "até
> 9 formas por nó", enquanto a própria enumeração e o instrumento de tempo de
> `docs/RESULTADOS_C1_custo_por_candidato.md` §1 contabilizam dez candidatos;
> adota-se aqui a contagem verificada no código.

---

## 1.2 A formulação — poda aprendida do espaço de busca

O problema é formulado como **poda aprendida do espaço de busca recursivo de
partições**: um modelo de aprendizado de máquina (do inglês *machine learning* –
ML) é consultado nos nós da árvore e elimina candidatos improváveis antes que o
codificador pague o custo de avaliá-los, com o objetivo de reduzir o tempo de
codificação preservando, tanto quanto possível, a eficiência de compressão. A
poda é, deste modo, uma heurística de restrição do espaço de busca, e não uma
substituição da busca: os candidatos que sobrevivem à poda continuam sendo
avaliados pelo próprio critério de taxa-distorção do codificador, o que preserva
a validade do fluxo de bits gerado.

Duas formas de poda são distinguidas ao longo de todo o texto, uma vez que a
diferença entre elas é estrutural, e não de grau.

**Podar antes da busca** significa decidir no início do nó, quando nenhuma
avaliação de taxa-distorção foi paga. O alcance é máximo, pois um acerto elimina
a subárvore inteira, inclusive a própria avaliação de `PARTITION_NONE`. Em
contrapartida, a decisão é tomada sem que um único número de taxa-distorção
tenha sido observado.

**Podar durante a busca** significa decidir depois de o codificador já ter
avaliado `PARTITION_NONE`, dispondo, então, da taxa, da distorção e do custo RD
reais dessa avaliação. A informação é substancialmente mais rica, ao passo que o
alcance é estruturalmente limitado: o custo do `PARTITION_NONE` já foi pago e
jamais é recuperado.

A decisão de particionamento é, por natureza, uma decisão de taxa-distorção. É
essa constatação que orienta o projeto dos atributos apresentado na Seção 4. O
codificador escolhe a forma de partição que minimiza o custo lagrangiano que
combina distorção e taxa, e não qualquer propriedade fotométrica do bloco.

Por conseguinte, a informação capaz de elevar o limite de desempenho de um
podador aprendido é a informação que a própria decisão utiliza: o contexto de
particionamento da vizinhança já codificada, a força de quantização e a posição
no quadro. Boa parte dela é de custo desprezível, uma vez que já está residente
na memória do codificador no momento da decisão. Esta é a hipótese central que
reorientou a investigação, depois de sucessivas tentativas no domínio de pixels.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §1;
> `docs/PLANO_H9_contribuicao_tese.md` §2 e §3 (hipótese H9 e blocos de
> atributos A–E); `docs/ANDAMENTO_tese.md` §1.1 (arco do resultado negativo do
> domínio de pixels para a virada H9);
> `src/aom/av1/encoder/partition_strategy.c:2293`
> (`av1_prune_partitions_before_search`) e `:2269` (`av1_prune_after_none`).

---

## 1.3 Decomposição do custo de busca por família de candidatos

A viabilidade de qualquer podador depende de onde o tempo de busca está de fato
alocado. Esta distribuição foi medida antes de qualquer experimento de
codificação. Utilizou-se a instrumentação nativa `CONFIG_COLLECT_PARTITION_STATS`
do libaom, que registra os microssegundos gastos em cada candidato de cada nó,
sobre **875.317 nós** de decisão das três sequências do conjunto de teste
reservado, em codificação intraquadro com `cpu-used=0`.

O tempo de trabalho local de um nó foi definido como a soma dos candidatos não
recursivos. A coluna da divisão quadrada foi deliberadamente excluída, uma vez
que o seu temporizador engloba a recursão e contabilizaria o mesmo trabalho
várias vezes. Os percentuais apresentados a seguir têm, deste modo, por
denominador os **nove candidatos não recursivos**: o `PARTITION_NONE`, as duas
retangulares, as quatro AB e as duas 4-way. Não são as dez formas do enum.

Esta distinção deve acompanhar toda citação destes valores, pois as duas
contagens convivem no trabalho. Dez é o número de formas que o codificador pode
escolher e que o modelo prediz. Nove é a base sobre a qual o custo de busca
local é medido.

A decomposição agregada mostra que o custo se distribui em três blocos de
tamanho comparável. O `PARTITION_NONE` consome **30,1%** do tempo de busca
local, as duas formas retangulares consomem **35,6%**, as quatro formas AB
consomem **20,4%** e as duas formas 4-way consomem **13,9%**.

As **partições estendidas — as AB somadas às 4-way — consomem, portanto, 34,3%
do tempo de busca local**, com variação de 28,9% a 41,3% entre as três
sequências. As formas retangulares somadas às estendidas atingem 69,9%.
Individualmente, contudo, nenhuma forma estendida ultrapassa 7,22% do tempo, o
que explica por que este custo passou despercebido em análises que examinavam
candidatos isolados.

O custo das partições estendidas concentra-se quase todo nos blocos grandes, o
que restringe o alcance útil de um podador desta família. Nos blocos de 8×8
amostras estas formas não se aplicam e o custo é nulo. Em 16×16 amostras elas
representam **8,7%** do tempo local; em 32×32 e 64×64 amostras, respectivamente
**50,0%** e **51,0%**; em 128×128 amostras, **35,3%**.

A medição de tempo de parede confirma a decomposição por temporizador. Ao
desligar todas as partições estendidas em codificações reais das três sequências
de teste, obteve-se aceleração média de **1,431×** ao custo de **+0,89%** de
taxa BD. São cerca de 0,03 ponto percentual de taxa BD por 1% de tempo
economizado — o perfil de candidatos caros e raramente decisivos que favorece um
podador seletivo.

> **Procedência.** `docs/RESULTADOS_C1_custo_por_candidato.md` §3 (decomposição
> agregada, por sequência e por tamanho de bloco) e §5 (limitações: compilação
> genérica sem SIMD, 2 quadros por sequência, `cq-level=32`);
> `docs/RESULTADOS_H9d_cota_superior.md` §3 (envelope de desligamento total das
> partições estendidas). Artefatos: `results/benchmark/partstats*/part_timing*.csv`
> (não versionados) e `results/benchmark/{h9d_ub,h9d_marg,h9d_tau}/raw.csv` (não
> versionados). Scripts: `src/scripts/benchmark/analyze_partstats.py`,
> `src/scripts/benchmark/h9d_upper_bound.py`, `src/scripts/benchmark/h9d_tau_curve.py`.

---

## 1.4 O espaço de projeto dos podadores — duas dimensões ortogonais

As soluções investigadas nesta tese são frequentemente lidas como níveis
sucessivos de sofisticação de uma mesma ideia. Esta leitura é incorreta. Elas
ocupam pontos distintos de um plano definido por **duas dimensões
independentes**: *quando* o podador age, o que determina que informação já foi
paga, e *o quê* o podador poda, o que determina quais candidatos deixam de ser
avaliados. Explicitar este plano é o que torna previsíveis os resultados de
composição entre soluções, apresentados no Capítulo de Resultados.

A primeira dimensão é o ponto de enganche no fluxo de controle de
`av1_rd_pick_partition`. O gancho `av1_prune_partitions_before_search` executa
antes de qualquer avaliação e caracteriza a poda pré-busca. O gancho
`av1_prune_after_none` executa imediatamente após a avaliação de
`PARTITION_NONE` e caracteriza a poda pós-NONE.

A segunda dimensão é a ação. Um podador pode comprometer o nó com o
`PARTITION_NONE` e encerrar a descida; pode restringir a busca apenas à divisão
quadrada; pode encerrar a busca de forma binária depois do `PARTITION_NONE`; ou
pode agir seletivamente sobre um subconjunto de candidatos, como as partições
estendidas.

As duas dimensões são ortogonais. É justamente por isso que duas soluções podem
partilhar o mesmo gancho e o mesmo vetor de atributos e ainda assim produzir
ganhos que se somam: o que as distingue é a ação, ou seja, o conjunto de
candidatos que cada uma retira da busca.

Duas soluções cujas ações se sobrepõem, em contrapartida, competem pelo mesmo
tempo economizável, e a composição rende menos que a soma das partes. Esta
distinção
entre sobreposição de informação e sobreposição de ação é uma contribuição
metodológica desta tese, e organiza toda a apresentação dos resultados de
composição.

> **Procedência.** `docs/SINTESE_resultados_metodologia.md` §2.8 (formalização
> das duas dimensões, tabela comparativa e nota de nomenclatura);
> `docs/ANDAMENTO_tese.md` §0.1 (registro da correção da Conclusão 3: a
> não-aditividade é sobreposição de ação, não limite informacional);
> `src/aom/av1/encoder/partition_strategy.c:2269–2290` e `:2293`.

---

## 1.5 Escopo declarado

O escopo experimental foi fixado antes das medições e é mantido sem exceção em
todos os resultados relatados. A investigação restringe-se à **predição
intraquadro** do AV1, no modo *All-Intra*, tendo como âncora o codificador de
referência libaom v3.10.0 configurado com `cpu-used=0`, regime em que a busca de
partição é exaustiva e o dado de referência corresponde à decisão ótima da busca
RD completa. A métrica de qualidade é o **PSNR-Y**, e a métrica de eficiência de
compressão é a taxa BD calculada sobre esta métrica; a métrica de custo é o
tempo de parede, reportado como redução percentual de tempo e como aceleração.

Dois conjuntos experimentais são utilizados, com funções distintas e
deliberadamente não intercambiáveis.

As **dezesseis sequências UVG em resolução 4K** foram particionadas por
sequência, sem vazamento, em dez de treino, três de validação (HoneyBee,
FlowerPan e Lips) e três de teste reservado (Jockey, RaceNight e RiverBank),
fixadas *a priori*. Este conjunto valida e caracteriza a contribuição, provando
que o ganho é atribuível ao modelo.

As **oito sequências da Classe A1 das condições comuns de teste** (do inglês
*Common Test Conditions* – CTC) da AOM, em 4K e 10 bits, com quinze quadros
conforme a especificação, produzem os resultados finais comparáveis à
literatura.

Os ***presets*** **nativos do codificador** são a referência de comparação, uma
vez que constituem o botão de velocidade que o usuário do libaom já possui.
Qualquer heurística proposta precisa justificar-se contra eles.

Sob a definição canônica de redução de tempo adotada neste texto, o *preset*
`cpu-used=1` entrega 32,59% de redução de tempo a +0,449% de taxa BD; o
`cpu-used=2` entrega 42,72% a +0,536%; e o `cpu-used=3` entrega 67,94% a
+2,722%. A escada de *presets* é, deste modo, discreta e esparsa, e a
granularidade fina entre degraus é uma das lacunas que esta tese investiga.

> **Procedência.** `docs/INVENTARIO_solucoes.md` §0 (âncora, definição canônica
> de redução de tempo, partições congeladas e escada de portões) e §1 (valores
> dos presets nativos); `docs/PROTOCOLO_avaliacao.md` (protocolo congelado);
> `docs/DECISOES_escopo.md` (PSNR-Y e quinze quadros como especificação da CTC
> §4.1); `results/thesis/00_PLANO_capitulos.md` §4 (separação funcional entre os
> dois cenários experimentais). Artefato da definição de redução de tempo:
> `results/benchmark/fase6_analysis/ts_definitions.csv`. Script:
> `src/scripts/benchmark/analyze_frontier.py:281`.

---

## 1.6 Síntese e encaminhamento

O objeto está, então, delimitado: a busca recursiva de particionamento
intraquadro do AV1, com dez formas candidatas por nó, cujo custo se distribui em
três blocos de tamanho comparável — 30,1% no `PARTITION_NONE`, 35,6% nas formas
retangulares e 34,3% nas partições estendidas. O problema está formulado como
poda aprendida deste espaço de busca, com duas dimensões de projeto
independentes que organizam todas as soluções investigadas. Por fim, o escopo
experimental está declarado e congelado, o que permite que cada resultado seja
lido contra a mesma régua.

A verificação desta formulação exige, antes de qualquer modelo, que o
comportamento do codificador seja observável nó a nó, e que o conjunto de dados
de treino registre exatamente a informação disponível em cada ponto de enganche
descrito na Seção 1.4. A instrumentação do codificador de referência e a geração
do conjunto de dados que a materializa são apresentadas na próxima seção.

> **Procedência.** Consolidação das notas das Seções 1.1 a 1.5; nenhum valor
> novo é introduzido nesta síntese.
