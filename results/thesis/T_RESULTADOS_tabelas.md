# Tabelas 8 a 21 — Capítulo de Resultados

Este documento materializa as Tabelas 8 a 21 especificadas em
`A2_TABELAS_E_FIGURAS.md`, na forma de produto acabado, pronto para colagem
direta nos documentos `R1` a `R6` do Capítulo de Resultados. Cada tabela traz a
identificação, a legenda redigida no padrão fixo do perfil estilístico da tese,
os valores em Markdown com alinhamento numérico à direita, uma nota de rodapé
com a definição de métrica e as ressalvas de comparabilidade, e a procedência
completa.

Nenhum valor deste documento é estimado. Toda célula provém de um artefato ou de
um documento já auditado do projeto, e a célula sem dado disponível recebe o
marcador `[completar: ...]`, nunca uma aproximação.

Duas convenções valem para todo o documento.

A primeira é de **definição**. A redução de tempo das Tabelas 12, 13, 15, 18, 19
e 20 usa a definição **canônica**, salvo declaração em contrário na própria
tabela: a média sobre os pontos de quantização de `1 − t/t_âncora`, seguida da
média sobre as sequências. Nunca a definição ponderada pelo tempo, que diverge
da canônica em até cerca de três pontos percentuais conforme a configuração. A
Tabela 11 usa uma terceira convenção, própria da campanha do conjunto de teste
reservado, declarada na sua própria nota.

A segunda é de **retratação**. Nenhuma nota interpretativa deste documento
apresenta o modelo substituto convolucional como cota superior do domínio de
pixels, afirma que os pixels saturam na variância, chama o H9c de duas a quatro
vezes mais eficiente que o H9a, ou trata a aditividade do H9d como propriedade
absoluta da sua ação. Assim determinam as retratações R1, R2, R6 e R9 de
`A3_RETRATACOES_E_LACUNAS.md`.

---

## Tabela 8 — Curva de operação do estudante de pixels na sequência Jockey

**Destino.** Resultados §1 (`R1_dominio_pixels.md`, Seção 1.1).

A Tabela 8 apresenta a curva de operação do modelo estudante da era de pixels
(`pixels24`) na sequência Jockey, medida em codificação intraquadro com
`cpu-used=0` e quatro pontos de quantização, e a cota superior por reprodução
das decisões do modelo substituto convolucional (H8), medida sem que uma única
convolução seja executada em C.

| ponto de operação | taxa BD (%) | aceleração (×) |
|---|--:|--:|
| Comprometimento com `PARTITION_NONE` apenas (P0) | 0,249 | 1,033 |
| + poda das formas retangulares (P_rect) | 0,492 | 1,048 |
| + limiares refinados por nível (P_ref) | 0,422 | 1,074 |
| Varredura agressiva — A1 (comprometimento a 80%) | 0,422 | 1,079 |
| Varredura agressiva — A2 (comprometimento a 70%) | 1,230 | 1,176 |
| Varredura agressiva — A3 (comprometimento a 60%, retangular a 40%) | 1,617 | 1,285 |
| Reprodução do H8, ponto conservador | −0,114 | 1,021 |
| Reprodução do H8, ponto agressivo | 0,197 | 1,032 |

*Nota.* Taxa BD sobre PSNR-Y, em porcentagem, contra a âncora libaom
`cpu-used=0`. A aceleração é a razão de tempo de parede sobre a mesma âncora. A
medição cobre duas imagens da sequência Jockey, valor padrão da campanha.

As duas linhas de reprodução do H8 não pagam custo de inferência convolucional.
As probabilidades do modelo substituto são pré-computadas fora do codificador e
reinjetadas pela mesma chamada de poda. A aceleração ali medida é, deste modo,
uma cota superior otimista quanto ao custo de execução. Ela não é comparável a
uma implantação real dos 28,1 milhões de parâmetros do modelo substituto.

> **Procedência.** `results/benchmark/h7h8_real_summary.csv` (linhas
> `P0_oldpolicy`, `P_rect`, `P_ref`, `H8_surrogate`) e
> `results/benchmark/h7h8_aggr/summary.csv` (linhas `A1_none80`, `A2_none70`,
> `A3_none60_rest40`, `H8_surrogate_aggr`). Documentos-fonte:
> `docs/RASTREABILIDADE.md` §5.1; `R1_dominio_pixels.md` §1.1. Scripts:
> `src/scripts/benchmark/h7h8_bench.py`,
> `src/scripts/partition_model/surrogate_replay.py`.

---

## Tabela 9 — Hierarquia no crivo de perda de otimalidade

**Destino.** Resultados §1 (Seção 1.4).

A Tabela 9 apresenta a fração de perda de otimalidade (do inglês *regret*) de
cada subconjunto de atributos avaliado no crivo ponderado por custo de
taxa-distorção real, medida a 25% e a 30% de redução de custo casada, sobre
seis sequências do conjunto de validação e de teste reservado e 792.840 nós de
decisão, ordenada do pior para o melhor subconjunto.

| subconjunto de atributos | fração de perda de otimalidade a 25% de `cost_red` | fração de perda de otimalidade a 30% de `cost_red` |
|---|--:|--:|
| Pontuação aleatória (piso) | [completar: valor a 25% de `cost_red` não registrado em nenhum documento do projeto] | 0,612 |
| Variância | 0,0573 | 0,060 |
| ConvNeXt com alvo de perda de otimalidade | 0,0219 | [completar: valor a 30% de `cost_red` não registrado em nenhum documento do projeto] |
| ConvNeXt com entropia cruzada | 0,0207 | [completar: valor a 30% de `cost_red` não registrado em nenhum documento do projeto] |
| `pixels24` | 0,0121 | 0,015 |
| **H9a** | **0,0036** | **0,006** |

*O que é a métrica.* A fração de perda de otimalidade é uma divisão. Em cima,
o sobrecusto de taxa-distorção das podas feitas. Embaixo, o custo de
taxa-distorção de todos os nós de decisão.

O número de cima, por nó, é o custo de comprometer o nó com `PARTITION_NONE`
menos o custo da subárvore que a busca completa encontraria. Ele nunca é
negativo, uma vez que a subárvore ótima também pode escolher o `PARTITION_NONE`;
no pior caso, empata. Ele vale zero sempre que a decisão correta já era
`PARTITION_NONE`. A soma cobre apenas os nós que a política podou naquele
limiar.

O número de baixo é somado sobre todos os nós de decisão, na montagem do
conjunto de dados, antes de qualquer limiar ser escolhido. Ele é, deste modo,
**fixo**: não muda com o ponto de operação. É essa propriedade que permite ler a
métrica como fronteira. Ela tende a zero quando a poda tende a zero e cresce
junto com a agressividade. Se o denominador fosse apenas o dos nós podados, a
métrica se tornaria um dano médio por poda e permaneceria alta mesmo sob poda
escassa.

A leitura é feita com a redução de custo de busca (`cost_red`) casada entre os
subconjuntos, de modo que todos sejam comparados no mesmo ponto de operação.
Menor é melhor.

*O que esta tabela não diz.* Nada aqui é grandeza medida no codificador, e
confundir os dois planos compromete a leitura.

As colunas são **custo de busca modelado**: número de candidatos de forma
ponderado pela área do bloco, conforme a Seção 3.4.1 do Capítulo de Metodologia.
**Não é tempo.** "25% de `cost_red`" não significa 25% de tempo economizado.

As células, por sua vez, **não são taxa BD** e não convertem em taxa BD. O
denominador soma os três níveis da hierarquia, que recobrem a mesma área da
imagem, ao passo que a codificação real emprega apenas um nível por região. Por
conseguinte, o valor sai sistematicamente menor do que qualquer perda observável
no codificador.

O que a tabela sustenta é a **razão entre subconjuntos no mesmo ponto de
operação**: a 25%, o `pixels24` desperdiça 3,4 vezes o que o H9a desperdiça. O
que ela **não** sustenta é a afirmação de que o `pixels24` custaria 0,0121% de
eficiência de compressão. O custo real do `pixels24` consta da Tabela 8: de
0,422% a 0,492% de taxa BD.

*De onde vêm os números, e o que falta.* Os valores a 25% provêm de
`docs/RESULTADOS_convnext_regret.md` §2, na tabela por `cost_red` redondo (5%,
10%, 15%, 20% e 25%). Os valores a 30% provêm do ranqueamento de
`docs/RESULTADOS_oraculo_regret.md` §3, que não inclui as duas variantes do
ConvNeXt — medidas em campanha posterior e nunca reexecutadas contra esse ponto.
As duas colunas, deste modo, não são comparáveis célula a célula.

*Duas reservas.* A grade de limiares da variância salta de 6,14% para 39,46% de
`cost_red`. Os dois pontos lidos caem dentro desse vão e são interpolados, de
modo que a razão entre a variância e os subconjuntos de pixels é a mais frágil
da tabela.

O modelo substituto convolucional não é apresentado aqui como cota superior do
domínio de pixels, uma vez que perde para o `pixels24` — um perceptrone sobre
vinte e quatro atributos manuais da mesma luminância. A hierarquia mede o treino
realizado, e não o limite da arquitetura.

> **Procedência.** `results/models/oracle_regret/frontier.csv` e
> `results/models/oracle_regret_convnext/frontier.csv` (verificados por
> `Glob`); `docs/RESULTADOS_convnext_regret.md` §2 e §5;
> `docs/RESULTADOS_oraculo_regret.md` §3; `docs/INVENTARIO_solucoes.md` §2.2;
> `R1_dominio_pixels.md` §1.4. Script: `src/scripts/partition_model/oracle_regret.py`
> — definição da perda de otimalidade absoluta na linha 12, acumulação do
> numerador sobre os nós podados nas linhas 258 e 259, acumulação do denominador
> `total_none_rd` sobre todos os nós de decisão nas linhas 119 e 139, e razão
> final na linha 297.

---

## Tabela 10 — Redução de custo de busca por subconjunto de atributos

**Destino.** Resultados §2 (`R2_h9a.md`, Seção 2.1). Corresponde à Tabela 2.1
já redigida no texto-fonte.

A Tabela 10 apresenta a redução de custo de busca, medida na simulação de
oráculo, por subconjunto de atributos, em três níveis de risco casado pela
perda de decisões de divisão, sobre dez sequências de treino e sessenta mil
superblocos.

| subconjunto de atributos | 0,5% | 1% | 2% |
|---|--:|--:|--:|
| Variância | 0,0 | 0,0 | 0,0 |
| `pixels24` (bloco A, 24 descritores de luminância) | 10,1 | 15,3 | 18,9 |
| **H9a** (A + vizinhança + quantização/posição, 36 atributos) | **15,7** | **20,1** | **24,9** |
| H9c (cota superior, custo RD real do `PARTITION_NONE`) | 33,0 | 33,0 | 39,7 |

*Nota.* Os valores são redução de custo de busca (%), medida por simulação de
oráculo. **Não são redução de tempo de parede**, e a simulação superestima o
ganho real de tempo por um fator próximo de cinco.

A seleção segue a **regra de risco casado**. Para cada subconjunto, toma-se, na
varredura de limiares, a linha cujo `split_lost` fica imediatamente abaixo de
cada patamar de risco declarado, e reporta-se o `cost_red` correspondente. O
ponto de 1% de risco do H9a corresponde a um `split_lost` medido de 0,498%, e o
de 2% do `pixels24` a 1,434%.

Registre-se a fonte correta: `gate2_final.csv` reúne agregação distinta, por
limiares fixos de perda de SPLIT, e não sustenta estes valores.

O `pixels24` é o próprio bloco A do H9a. A linha do H9a mede, portanto, o
efeito marginal dos doze atributos adicionais de vizinhança, quantização e
posição sobre os descritores de luminância. Não se trata de comparação entre
conjuntos disjuntos.

> **Procedência.** `results/models/gate2_final_sweep.csv` (verificado por
> `Glob`); `results/models/student_h9a/oracle_sim_*.csv`,
> `results/models/student_h9a/gate4_evidence.txt`. Documentos-fonte:
> `R2_h9a.md` §2.1; `docs/ANDAMENTO_tese.md` §3;
> `docs/RESULTADOS_auditoria_dominio_pixels.md`. Script:
> `src/scripts/partition_model/gate2_signal.py`.

---

## Tabela 11 — Curva do H9a no conjunto de teste reservado

**Destino.** Resultados §2 (Seção 2.3). Corresponde à Tabela 2.2 já redigida
no texto-fonte.

A Tabela 11 apresenta a curva de taxa BD contra redução de tempo do H9a, por
sequência e por ponto de operação, sobre as sequências Jockey, RaceNight e
RiverBank do conjunto de teste reservado, com dez quadros por sequência,
quatro pontos de quantização e execução em uma única linha de execução.

| ponto | Jockey | RaceNight | RiverBank |
|---|---|---|---|
| P0 | 0,19% / 21,6% / 1,28× | 0,27% / 18,1% / 1,22× | 0,003% / 17,4% / 1,21× |
| P_ref | 0,92% / 32,6% / 1,48× | 0,74% / 31,8% / 1,47× | 0,13% / 24,2% / 1,32× |
| A2 | 1,37% / 47,7% / 1,91× | 1,17% / 41,7% / 1,72× | 0,23% / 30,2% / 1,43× |
| A3 | 2,03% / 57,2% / 2,34× | 1,72% / 50,7% / 2,03× | 0,38% / 36,8% / 1,58× |

*Nota.* Cada célula traz taxa BD / redução de tempo / aceleração. A taxa BD é
sobre PSNR-Y, contra a âncora libaom `cpu-used=0`.

A redução de tempo é calculada ponto a ponto como `(1 − 1/aceleração)·100`.
Trata-se de definição própria desta campanha, **distinta** tanto da canônica
quanto da ponderada pelo tempo declaradas na Metodologia §3.5, uma vez que aqui
não há múltiplos pontos de quantização a agregar por sequência antes da média.

A política é a completa: comprometimento com `PARTITION_NONE`, divisão quadrada
forçada e poda das formas retangulares.

Uma ressalva acompanha esta tabela. O critério estrito de dominância sobre a
variância a aceleração casada **não foi atingido** nesta campanha, por ausência
de par casado nas três sequências. Ela não invalida a tabela, uma vez que a
redução de tempo contra a âncora e a superioridade sobre a pontuação aleatória
permanecem como pilares independentes.

> **Procedência.**
> `results/benchmark/h9_test/{Jockey,RaceNight,RiverBank}/{curve_safe,curve_aggr,ablation}/{summary.csv,curve.csv}`
> (verificados por `Glob`). Documentos-fonte: `docs/RESULTADOS_fase5.md` §1 a
> §5; `R2_h9a.md` §2.3; `docs/ANDAMENTO_tese.md` §4.1. Scripts:
> `results/benchmark/{fase5_final,matched_bd}.py`,
> `src/scripts/benchmark/analyze_ablation.py`.

---

## Tabela 12 — Resultados finais na CTC

**Destino.** Resultados §2 (Seção 2.4). Corresponde à Tabela 2.3 já redigida
no texto-fonte.

A Tabela 12 apresenta os dois pontos de operação do H9a contra a âncora e
contra os *presets* nativos, na validação universal sobre as oito sequências
da Classe A1 das condições comuns de teste (do inglês *Common Test
Conditions* – CTC), em 4K, 10 bits e quinze quadros, com a mesma grade de
quantização do restante da tese.

| configuração | taxa BD | redução de tempo | aceleração |
|---|--:|--:|--:|
| H9a equilibrado (P_rect) | +0,568% | 17,72% | 1,223× |
| H9a agressivo (A3) | +1,403% | 31,51% | 1,492× |
| libaom `cpu-used=1` | +0,449% | 32,59% | 1,508× |
| libaom `cpu-used=2` | +0,536% | 42,72% | 1,788× |
| libaom `cpu-used=3` | +2,722% | 67,94% | 3,159× |

*Nota.* Âncora libaom `cpu-used=0`, taxa BD sobre PSNR-Y e **redução de tempo
na definição canônica**.

Sob a definição ponderada pelo tempo, os mesmos artefatos registram 30,42% para
o `cpu-used=1`, em vez de 32,59%, e 34,14% para o ponto agressivo, em vez de
31,51%. A divergência chega a cerca de três pontos percentuais, e é por isso que
as duas definições nunca coexistem nesta tabela.

A rede convolucional nativa de poda intraquadro está ativa nos *presets*
`cpu-used` 1, 2 e 3 e desligada na âncora, o que foi verificado por teste
empírico de identidade byte a byte. A comparação opõe, deste modo,
configurações limpas e disjuntas.

Quanto à leitura: nenhum ponto de aprendizado de máquina desta tabela domina a
rede convolucional nativa na fronteira de taxa BD contra tempo. O valor medido é
outro — a granularidade fina de 12% a 22% de redução de tempo, que a escada
discreta dos *presets* não oferece.

> **Procedência.** `results/benchmark/fase6/{raw_results.csv,
> bdrate_per_seq.csv, bdrate_average.csv, tables.tex}` e
> `results/benchmark/fase6_analysis/ts_definitions.csv`. Documentos-fonte:
> `docs/RESULTADOS_fase6.md` §1 a §3; `R2_h9a.md` §2.4;
> `docs/ANDAMENTO_tese.md` §4.2. Scripts: `src/scripts/fase6/{run_fase6.sh,
> encode_ctc.py, report_ctc.py, analyze_frontier.py}`.

---

## Tabela 13 — Substituição direta do podador nativo pelo H9a

**Destino.** Resultados §2 (Seção 2.5). Corresponde à Tabela 2.4 já redigida
no texto-fonte.

A Tabela 13 apresenta o resultado da substituição direta do podador nativo
pelo H9a, a `cpu-used` fixo — a rede convolucional nativa desligada por
variável de ambiente dedicada e o H9a assumindo como único podador
intraquadro —, sobre as oito sequências da Classe A1 e a mesma grade de
quantização das demais campanhas CTC.

| `cpu-used` | podador | taxa BD | redução de tempo | aceleração |
|:--:|---|--:|--:|--:|
| 1 | Rede convolucional nativa | +0,449% | 32,59% | 1,508× |
| 1 | H9a equilibrado | +0,915% | 40,20% | 1,694× |
| 1 | H9a agressivo | +1,685% | 51,82% | 2,104× |
| 2 | Rede convolucional nativa | +0,536% | 42,72% | 1,788× |
| 2 | H9a equilibrado | +1,030% | 50,05% | 2,046× |
| 2 | H9a agressivo | +1,805% | 60,97% | 2,610× |
| 3 | Rede convolucional nativa | +2,722% | 67,94% | 3,159× |
| 3 | H9a equilibrado | +3,866% | 73,09% | 3,754× |
| 3 | H9a agressivo | +4,347% | 77,30% | 4,465× |

*Nota.* Âncora libaom `cpu-used=0` e redução de tempo na definição canônica. O
mecanismo de substituição foi verificado por identidade de resumo criptográfico
do fluxo de bits em quatro configurações.

No mesmo `cpu-used`, a rede convolucional nativa isolada tem sempre taxa BD e
redução de tempo menores que o H9a. Não há dominância direta em nível algum. Há,
sim, um compromisso: o H9a corta mais tempo a custo de taxa BD desproporcional,
de duas a três vezes o da nativa.

Apenas um dos nove pontos é estritamente dominado — o H9a equilibrado a
`cpu-used=1`, cravado pelo *preset* nativo `cpu-used=2`. O retorno marginal
entre pontos consecutivos, contudo, cai de 116,4 no trecho puramente nativo
para 2,7 no pior trecho. Deste modo, a fronteira de dominância, tomada
isoladamente, superestima quanto o H9a é competitivo.

O H9a não supera a rede convolucional nativa como podador em arranjo algum desta
tabela.

> **Procedência.** `results/benchmark/fase6_swap/{raw_results.csv,
> swap_per_seq.csv, swap_average.csv, swap_tables.tex}`. Documentos-fonte:
> `docs/RESULTADOS_fase6.md` §4 a §4.3; `R2_h9a.md` §2.5;
> `docs/RESULTADOS_microbench_pruner.md` §6. Scripts:
> `src/scripts/fase6/{run_swap.sh, encode_swap.py, report_swap.py}`.

---

## Tabela 14 — O fator de confusão do H9a no H9c, sequência Neon1224

**Destino.** Resultados §3 (`R3_h9c.md`, Seção 3.3).

A Tabela 14 apresenta a quantificação do fator de confusão identificado na
caracterização do H9c: a taxa BD e a redução de tempo do H9c medido junto com
o H9a nos seus limiares compilados por padrão, contra o H9c isolado, com o
H9a explicitamente neutralizado por `τ = 2/2/−1`, na sequência Neon1224 a
`cpu-used=0` e contra a mesma âncora.

| Configuração | Taxa BD | Redução de tempo |
|---|--:|--:|
| H9c a τ=0,95, medido antes (H9a a 0,9 + H9c) | 0,267% | 17,15% |
| H9c a τ=0,95, isolado | 0,037% | **2,96%** |
| H9c a τ=0,90, medido antes (H9a a 0,9 + H9c) | 0,270% | 17,36% |
| H9c a τ=0,90, isolado | 0,037% | **4,23%** |
| H9c a τ=0,60, medido antes (H9a a 0,9 + H9c) | 0,386% | 20,53% |
| H9c a τ=0,60, isolado | 0,100% | **9,31%** |

*Nota.* Redução de tempo na definição canônica e taxa BD sobre PSNR-Y. A única
variável alterada entre as linhas "medido antes" e "isolado" é a neutralização
do H9a.

A origem do fator de confusão é a seguinte. Os primeiros roteiros de teste do
H9c definiam apenas as suas próprias variáveis de ambiente e deixavam o H9a nos
seus limiares compilados por padrão (`tau_none = tau_split = 0,9`). Isso ocorria
porque o H9a roda sob critério puramente geométrico, sem sinalizador de
habilitação, ao contrário do H9c, cuja ativação depende de uma variável
dedicada. Cada linha rotulada como H9c media, portanto, o H9a a 0,9 empilhado
com o H9c, e não o H9c isolado. De 82% a 96% da redução de tempo antes
atribuída ao H9c provinha do H9a.

Esta quantificação é local a uma sequência. A generalização a quatro sequências
está na Tabela 16.

Por decisão editorial, **não se afirma** nesta tabela nem em nenhuma outra desta
tese que o H9c seria de duas a quatro vezes mais eficiente que o H9a. Foi esta
mesma medição que retirou a afirmação.

> **Procedência.** Linhas `h9ciso_*` de `results/benchmark/fase6/raw_results.csv`
> (verificado por `Glob`). Documentos-fonte: `docs/ANDAMENTO_tese.md` §8.1;
> `R3_h9c.md` §3.3; `A3_RETRATACOES_E_LACUNAS.md` R6. Scripts:
> `src/scripts/fase6/{encode_h9c_iso.py, report_bloco7.py}`.

---

## Tabela 15 — Substituição direta do H9c na grade CTC completa

**Destino.** Resultados §3 (Seção 3.4).

A Tabela 15 apresenta o resultado agregado da substituição direta do podador
nativo pelo H9c, em três níveis de *preset* e dois limiares, sobre a grade
CTC completa de oito sequências e quatro pontos de quantização, com o H9a
neutralizado desde o roteiro.

| *Preset* | Podador | Taxa BD (%) | Redução de tempo (%) | Aceleração |
|:--:|---|--:|--:|--:|
| 1 | Rede convolucional nativa | 0,449 | 32,59 | 1,508× |
| 1 | H9c a τ=0,90 | 0,448 | 31,65 | 1,498× |
| 1 | H9c a τ=0,95 | **0,414** | 30,34 | 1,469× |
| 2 | Rede convolucional nativa | 0,536 | 42,72 | 1,788× |
| 2 | H9c a τ=0,90 | 0,539 | 42,07 | 1,787× |
| 2 | H9c a τ=0,95 | **0,516** | 40,73 | 1,746× |
| 3 | Rede convolucional nativa | **2,722** | 67,94 | 3,159× |
| 3 | H9c a τ=0,90 | 3,397 | 70,67 | 3,474× |
| 3 | H9c a τ=0,95 | 3,384 | 70,25 | 3,419× |

*Nota.* Âncora libaom `cpu-used=0` e redução de tempo na definição canônica. A
definição ponderada pelo tempo diverge desta tabela de +2,63 a −2,97 pontos
percentuais conforme a configuração e chega a reordenar o ranqueamento, razão
pela qual as duas não coexistem.

Na média da grade completa, o H9c e a rede convolucional nativa **empatam** nos
*presets* 1 e 2. Nenhuma diferença é estatisticamente significativa em teste
pareado (p = 0,278 e p = 0,291). No *preset* 3, o H9c é significativamente pior
(p = 0,004).

A afirmação defensável na grade completa é, portanto, de paridade, e não de
superioridade. A vantagem própria do H9c aparece decomposta por regime de
quantização, fora desta tabela: no regime de alta qualidade (CQ 20 e 32), onde é
estatisticamente significativa em dois níveis independentes de *preset*.

> **Procedência.**
> `results/benchmark/fase6_swap_h9c/{raw_results,swap_per_seq,swap_average}.csv`
> (verificado por `Glob`). Documentos-fonte: `docs/RESULTADOS_fase6_swap_h9c.md`
> §2 a §6; `R3_h9c.md` §3.4. Scripts:
> `src/scripts/fase6/{encode_swap_h9c.py, report_swap.py}`,
> `src/scripts/benchmark/analyze_frontier.py`.

---

## Tabela 16 — Decomposição de três pernas e interação medida

**Destino.** Resultados §3 (Seção 3.5).

A Tabela 16 apresenta a decomposição aditiva do H9a e do H9c sobre quatro
sequências CTC, com o podador pré-busca isolado, o podador pós-NONE isolado,
os dois empilhados, a soma aritmética das partes isoladas e a interação
medida entre eles, de modo que o balanço feche na forma `tempo(H9a) +
tempo(H9c) + interação = tempo(H9a + H9c)`.

| Sequência | H9a só | H9c só | H9a + H9c | Soma | **Interação** |
|---|--:|--:|--:|--:|--:|
| Neon1224 | 16,8% | 4,2% | 17,1% | 21,0% | **−3,9 pp** |
| PierSeaSide | 10,4% | 5,4% | 12,8% | 15,8% | **−3,0 pp** |
| Tango | 7,2% | 12,3% | 17,1% | 19,4% | **−2,4 pp** |
| TimeLapse | 9,2% | 0,6% | 11,5% | 9,8% | +1,7 pp |
| **Média** | **10,9%** | **5,6%** | **14,6%** | **16,5%** | **−1,9 pp** |

*Nota.* Redução de tempo na definição canônica e âncora libaom `cpu-used=0`.

A interação é negativa em três das quatro sequências e na média. Isso indica que
cerca de 12% do ganho potencial evapora na sobreposição entre o H9a e o H9c.

Esta decomposição cobre **quatro das oito sequências CTC**, uma vez que o
terceiro termo — o H9a nos seus limiares compilados por padrão, sozinho — só foi
medido nelas. A conclusão robusta é, portanto, a direção e a ordem de grandeza
do efeito, e não o valor central exato, dada a dispersão de −3,9 a +1,7 ponto
percentual entre sequências.

A interação negativa não é lida como limite informacional. Ela é sobreposição de
ação entre dois podadores que perguntam, ambos, se o nó pode ser encerrado. A
distinção é confirmada por contraste com o H9d na Tabela 18, que compartilha
informação idêntica à do H9c e ainda assim se soma.

> **Procedência.** Linhas `h9adef`, `h9ciso_tau90`, `ml_balanced` de
> `results/benchmark/fase6/raw_results.csv` (verificado por `Glob`).
> Documentos-fonte: `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §2;
> `docs/ANDAMENTO_tese.md` §8.3; `R3_h9c.md` §3.5. Scripts:
> `src/scripts/fase6/{encode_h9adef.py, report_e3_dec_e2.py}`. *Lacuna:*
> `[completar: a decomposição de três pernas cobre quatro das oito sequências CTC; a extensão às quatro restantes custaria cerca de 12 codificações e 1h30, registrado em A3_RETRATACOES_E_LACUNAS.md L6]`.

---

## Tabela 17 — Critério de predizibilidade offline do H9d, por nível de bloco

**Destino.** Resultados §4 (`R4_h9d.md`, Seção 4.2).

A Tabela 17 apresenta a área sob a curva característica de operação do
receptor (ROC-AUC) do critério de predizibilidade das partições estendidas do
H9d, por tamanho de bloco e por conjunto de atributos, sobre 792.840 nós de
decisão do conjunto reservado (as três sequências de validação e as três de
teste), com os modelos treinados nas dez sequências restantes.

| tamanho de bloco (amostras) | nós avaliados | base de positivos (%) | AUC-ROC (36 atributos) | AUC-ROC (39 atributos) |
|:--:|--:|--:|--:|--:|
| 16 | 572.213 | 9,9 | 0,906 | 0,919 |
| 32 | 172.627 | 13,1 | 0,817 | 0,829 |
| 64 | 48.000 | 3,0 | 0,864 | 0,865 |
| **Agregado** | **792.840** | **10,2** | **0,890** | **0,902** |

*Nota.* O rótulo binário é positivo quando a partição ótima pertence ao conjunto
estendido (`HORZ_A`, `HORZ_B`, `VERT_A`, `VERT_B`, `HORZ_4`, `VERT_4`), e
negativo em qualquer outro caso. A base de positivos é a fração de nós com
rótulo positivo naquele nível, e não coincide com a prevalência de qualquer
forma isolada.

A coluna de 36 atributos usa apenas o vetor pré-busca do H9a. A de 39 acrescenta
o bloco E — o custo de taxa-distorção real do `PARTITION_NONE` —, disponível sem
custo adicional no ponto de inserção pós-NONE. É este o vetor levado ao
codificador.

Estes valores foram lidos das tabelas já publicadas em
`docs/RESULTADOS_H9d_predizibilidade.md` §2 e §2.1. Não existe, no repositório,
CSV estruturado equivalente, apenas o arquivo de execução `run.log`, o que está
registrado como lacuna em `A2_TABELAS_E_FIGURAS.md §3`.

> **Procedência.** `docs/RESULTADOS_H9d_predizibilidade.md` §2 e §2.1;
> `R4_h9d.md` §4.2. Modelos: `results/models/h9d_predictability/students.pt`
> (36 atributos) e `results/models/h9d_predictability_h9c/students.pt` (39
> atributos), verificados por `Glob`. Script:
> `src/scripts/partition_model/h9d_predictability.py` (com `--feature-set h9c`
> para a variante de 39 atributos).

---

## Tabela 18 — Resultado sob protocolo CTC do H9d — contribuição marginal

**Destino.** Resultados §4 (Seção 4.6).

A Tabela 18 apresenta a contribuição marginal do H9d sobre o H9a no ponto
equilibrado implantado, medida nas oito sequências da CTC, e o preço do
único mecanismo alternativo disponível ao usuário da solução implantada — o
botão de limiar do próprio H9a —, para comparação direta.

| configuração | taxa BD (%) | redução de tempo (%) | aceleração | preço (pp de taxa BD por pp de tempo) |
|---|--:|--:|--:|--:|
| H9a equilibrado (base) | +0,568 | 17,72 | 1,223× | — |
| H9a + H9d (calibração PL10, implantado) | +0,586 | 18,74 | 1,238× | **0,0179** |
| Botão de limiar do H9a (segmento P_rect → A3), referência | — | — | — | 0,063 |

*Nota.* Redução de tempo na definição canônica e âncora libaom `cpu-used=0`.

O preço é a razão entre a diferença de taxa BD e a diferença de redução de tempo
contra a base declarada em cada linha. A linha do H9a equilibrado é a própria
base e não tem preço marginal associado.

A terceira linha não é um ponto de operação codificado. Ela é a inclinação do
segmento entre os dois pontos congelados do H9a — o equilibrado e o agressivo —
na mesma grade CTC, tomada como referência do custo do único botão alternativo
de que o usuário já dispõe. O valor de 0,063 pp/pp é o estimador por
interpolação por sequência, ao passo que a média das médias sobre as oito
sequências dá 0,0606 pp/pp. Os dois estimadores concordam a cerca de 4%, mas não
devem ser somados nem misturados na mesma leitura.

O H9d compra tempo por cerca de um terço do preço do botão de limiar, ou seja,
aproximadamente 3,5 vezes mais barato.

> **Procedência.** `results/benchmark/fase6/{bdrate_average.csv,
> bdrate_per_seq.csv, tables.tex}` (linhas `ml_balanced` e `ml_bal_h9d`,
> verificado por `Glob`: `0,5676/17,7238/1,2226` e `0,5858/18,739/1,2378`).
> Documentos-fonte: `docs/RESULTADOS_H9d_CTC.md` §2 e §3; `R4_h9d.md` §4.6.
> Scripts: `src/scripts/fase6/{ctc_h9d.py, report_ctc.py,
> ctc_h9d_marginal.py}`.

---

## Tabela 19 — Fronteira bidimensional do H9d

**Destino.** Resultados §4 (Seção 4.7).

A Tabela 19 apresenta a família completa de configurações do H9d, cruzando
duas bases do H9a — a equilibrada e a agressiva — com duas calibrações do
H9d — PL10 e PL20 —, sob o mesmo protocolo CTC de oito sequências, quatro
pontos de quantização e `cpu-used=0`, com cada valor marginal medido contra a
própria base do H9a correspondente.

| base do H9a | calibração do H9d | taxa BD marginal (pp) | redução de tempo marginal (pp) | preço (pp/pp) |
|---|:--:|--:|--:|--:|
| Equilibrada | PL10 (implantada) | +0,018 | +1,02 | **0,0179** |
| Equilibrada | PL20 | +0,083 | +2,09 | 0,0399 |
| Agressiva | PL10 | +0,006 | +0,17 | 0,0329 |
| Agressiva | PL20 | +0,017 | +0,65 | 0,0258 |

*Nota.* Redução de tempo na definição canônica e âncora libaom `cpu-used=0`. Os
valores marginais são a diferença entre cada configuração empilhada e a sua
própria base, equilibrada ou agressiva, e não contra a âncora nativa
diretamente.

Os quatro pontos foram recalculados nesta auditoria a partir de
`results/benchmark/fase6/bdrate_average.csv`, que já reúne as quatro
configurações agregadas (`ml_bal_h9d`, `ml_bal_h9d_pl20`, `ml_aggr_h9d`,
`ml_aggr_h9d_pl20`). Não foi necessário recalcular a taxa BD a partir do CSV
bruto por codificação, uma vez que a agregação já está disponível e os quatro
valores de preço batem, à quarta casa decimal, os já redigidos em prosa em
`R4_h9d.md` §4.7.

O valor marginal do H9d **desaba** conforme a base do H9a fica agressiva: de
+1,02 pp sobre a base equilibrada para +0,17 pp sobre a agressiva, abaixo da
resolução temporal medida do arranjo (~0,46 pp). Isso não é apresentado como
resultado negativo isolado. É evidência de que a disjunção de ação que sustenta
a soma **depende do ponto de operação**, e não é propriedade absoluta da
alavanca.

Os quatro pontos desta tabela figuram, sem exceção, como **dominados** na
fronteira global consolidada na Tabela 20. Esta dominância não contradiz a
contribuição do H9d, uma vez que a base do H9a sobre a qual ele age já é
dominada pelo mesmo conjunto de configurações. O H9d herda, deste modo, a
posição do H9a e a melhora marginalmente dentro dela, sem perder posição alguma.

> **Procedência.** `results/benchmark/fase6/{raw_results.csv,
> bdrate_average.csv}` (verificado por `Glob`; 128 linhas somadas nas quatro
> configurações, 32 por configuração, oito sequências × quatro CQ).
> Documentos-fonte: `R4_h9d.md` §4.7; `docs/ANDAMENTO_tese.md` §0.1;
> `docs/RESULTADOS_fronteira_pareto_global.md` §4.2. Script:
> `src/scripts/fase6/ctc_h9d.py` (extensão de 96 codificações novas; a
> quarta configuração, `ml_bal_h9d`, já constava da campanha CTC anterior).

---

## Tabela 20 — Fronteira de compromisso global consolidada

**Destino.** Resultados §6 (`R6_analise_integrada.md`, prospectiva).

A Tabela 20 apresenta a fronteira de compromisso entre taxa BD e redução de
tempo de todas as vinte e quatro configurações desta tese avaliadas sob
protocolo CTC nas oito sequências da Classe A1, ordenada por taxa BD
crescente, com a coluna de status marcando as quinze configurações não
dominadas no sentido de Pareto, na execução de 2026-07-29 que resolve a
lacuna L1 sobre a ausência do H9d na versão anterior desta fronteira,
construída sobre apenas três sequências.

| configuração | taxa BD (%) | redução de tempo (%) | aceleração | status na fronteira de Pareto |
|---|--:|--:|--:|:--:|
| H9c τ=0,95 (`cpu-used=0`) | 0,160 | 12,61 | 1,148× | não dominado |
| H9c τ=0,90 (`cpu-used=0`) | 0,172 | 13,59 | 1,162× | não dominado |
| H9c τ=0,95, substituição em `cpu-used=1` | 0,414 | 30,34 | 1,469× | não dominado |
| H9c τ=0,90, substituição em `cpu-used=1` | 0,448 | 31,65 | 1,498× | não dominado |
| libaom `cpu-used=1` (nativo) | 0,449 | 32,59 | 1,508× | não dominado |
| H9c τ=0,95, substituição em `cpu-used=2` | 0,516 | 40,73 | 1,746× | não dominado |
| libaom `cpu-used=2` (nativo) | 0,536 | 42,72 | 1,788× | não dominado |
| H9c τ=0,90, substituição em `cpu-used=2` | 0,539 | 42,07 | 1,787× | dominado |
| H9a equilibrado (`cpu-used=0`) | 0,568 | 17,72 | 1,223× | dominado |
| H9a equilibrado + H9d PL10 (`cpu-used=0`, implantado) | 0,586 | 18,74 | 1,238× | dominado |
| H9c τ=0,45 (`cpu-used=0`) | 0,643 | 21,35 | 1,298× | dominado |
| H9a equilibrado + H9d PL20 (`cpu-used=0`) | 0,651 | 19,81 | 1,255× | dominado |
| H9a equilibrado, substituição em `cpu-used=1` | 0,915 | 40,20 | 1,694× | dominado |
| H9a equilibrado, substituição em `cpu-used=2` | 1,030 | 50,05 | 2,046× | não dominado |
| H9a agressivo (`cpu-used=0`) | 1,403 | 31,51 | 1,492× | dominado |
| H9a agressivo + H9d PL10 (`cpu-used=0`) | 1,409 | 31,68 | 1,494× | dominado |
| H9a agressivo + H9d PL20 (`cpu-used=0`) | 1,420 | 32,16 | 1,505× | dominado |
| H9a agressivo, substituição em `cpu-used=1` | 1,685 | 51,82 | 2,104× | não dominado |
| H9a agressivo, substituição em `cpu-used=2` | 1,805 | 60,97 | 2,610× | não dominado |
| libaom `cpu-used=3` (nativo) | 2,722 | 67,94 | 3,159× | não dominado |
| H9c τ=0,95, substituição em `cpu-used=3` | 3,384 | 70,25 | 3,419× | não dominado |
| H9c τ=0,90, substituição em `cpu-used=3` | 3,397 | 70,67 | 3,474× | não dominado |
| H9a equilibrado, substituição em `cpu-used=3` | 3,866 | 73,09 | 3,754× | não dominado |
| H9a agressivo, substituição em `cpu-used=3` | 4,347 | 77,30 | 4,465× | não dominado |

*Nota.* Âncora libaom `cpu-used=0`, taxa BD sobre PSNR-Y e redução de tempo na
definição canônica. A dominância de Pareto é verificada nas duas dimensões
simultaneamente: menor taxa BD **e** maior redução de tempo. A coluna de status
reproduz o campo `dominated_by` do artefato de origem, vazio nos pontos não
dominados.

As configurações do H9a empilhadas com o H9d figuram nesta fronteira e são, sem
exceção, **dominadas**. A base do H9a sobre a qual o H9d atua já é dominada pelo
mesmo conjunto de configurações que domina o par empilhado. O H9d não perde,
portanto, posição alguma: ele herda a posição do H9a e a melhora marginalmente
dentro dela. Esta dominância **não contradiz** o resultado marginal do H9d, que
é medido como contribuição sobre uma base fixa e contra a curva de limiares do
próprio H9a, e não como posição na fronteira global.

Nenhum ponto de aprendizado de máquina domina a rede convolucional nativa. Os
*presets* `cpu-used=1` e `cpu-used=2` permanecem não dominados. O H9c a τ=0,95
fica colado ao primeiro, com taxa BD e tempo menores nos dois eixos — um empate
técnico, sem dominância em direção alguma.

Uma ressalva de comparabilidade acompanha a fronteira. Ela mistura configurações
medidas em campanhas distintas (`fase6`, `fase6_swap`, `fase6_swap_h9c`), todas
contra a mesma âncora e grade de quantização, mas não na mesma janela contínua
de execução. Diferenças de tempo inferiores à resolução pareada de ~0,46 pp
pedem, deste modo, cautela adicional, e o empate técnico citado acima está nessa
faixa.

> **Procedência.** `results/benchmark/fase6_analysis/pareto_frontier.csv`
> (execução de 2026-07-29, verificada por leitura direta: 24 configurações,
> 15 não dominadas), gerado por `src/scripts/fase6/analyze_frontier.py` a
> partir de `results/benchmark/{fase6, fase6_swap,
> fase6_swap_h9c}/raw_results.csv`. Documentos-fonte:
> `docs/RESULTADOS_fronteira_pareto_global.md` §3 e §4; `A3_RETRATACOES_E_LACUNAS.md`
> R24 e L1.

---

## Tabela 21 — As três conclusões da tese, com o número que as sustenta

**Destino.** Resultados §6, fechamento.

A Tabela 21 apresenta as três conclusões consolidadas desta tese, com o
enunciado vigente, o número central que a sustenta e a seção do Capítulo de
Resultados em que a evidência correspondente é apresentada pela primeira vez.

| # | conclusão | enunciado | número central | seção de origem |
|:--:|---|---|---|---|
| 1 | Nenhuma via de pixels compete com o contexto de taxa-distorção barato | Na hierarquia medida no crivo ponderado por perda de otimalidade, o H9a supera o `pixels24` por 3,4× em fração de perda de otimalidade a 25% de redução de custo casada (0,0036 contra 0,0121), e o `pixels24` supera a variância isolada por 4,7× | Resultados §1.4 (Tabela 9) |
| 2 | O contexto de taxa-distorção barato não supera o podador nativo na média da grade CTC, mas preenche a granularidade fina de baixo regime de aceleração | H9c a τ=0,95 entrega 0,160% de taxa BD a 12,61% de redução de tempo, e a τ=0,90, 0,172% a 13,59%, no vão de 0% a 32,59% que a escada nativa deixa descoberto entre `cpu-used=0` e `cpu-used=1` | Resultados §2.4 e §6 (Tabelas 12 e 20) |
| 3 | Alavancas de poda se somam na medida em que os conjuntos de candidatos que atacam são disjuntos, e não em função da informação que compartilham | O H9d soma +1,02 pp de redução de tempo sobre o H9a com informação idêntica à do H9c, que somara apenas +0,26 pp; a interação medida entre H9a e H9c é negativa em −1,9 pp na média de quatro sequências | Resultados §3.5 e §4.9 (Tabelas 16 e 18) |

*Nota.* Esta tabela não introduz número novo. Ela reúne, sem alteração, valores
já tabulados nas Tabelas 9, 12, 16, 18 e 20.

A Conclusão 3 é apresentada na sua forma corrigida. A não-aditividade das duas
primeiras alavancas de poda **não é um limite informacional**, e sim
sobreposição de ação: os dois podadores perguntam, ambos, se o nó pode ser
encerrado, e por isso caçam os mesmos blocos de conteúdo liso. Assim determina a
retratação R8 de `A3_RETRATACOES_E_LACUNAS.md`.

A disjunção de ação que explica por que o H9d escapa dessa sobreposição **não é
propriedade absoluta da alavanca**. Ela depende do ponto de operação em que os
dois podadores efetivamente rodam, conforme a retratação R9 do mesmo documento,
e desaba quando a base do H9a fica agressiva (Tabela 19).

A Conclusão 2 usa os números da fronteira recomposta em 2026-07-29 (Tabela 20),
e não os valores anteriores de três sequências (0,21% a 0,23% de taxa BD),
superados pela retratação R24.

> **Procedência.** `docs/RESULTADOS_fronteira_pareto_global.md` §4;
> `docs/SINTESE_resultados_metodologia.md` §6; `A3_RETRATACOES_E_LACUNAS.md`
> R8, R9 e R24. Nenhum artefato numérico próprio: tabela de síntese textual.

---

## Registro de lacunas desta entrega

Três células, em uma única tabela, permanecem com o marcador
`[completar: ...]`, todas na Tabela 9: a fração de perda de otimalidade da
pontuação aleatória a 25% de `cost_red` e as frações de perda de otimalidade
das duas variantes do ConvNeXt a 30% de `cost_red`. As três decorrem do mesmo
fato — os documentos
do projeto registram a hierarquia do crivo A5 em grades de `cost_red`
distintas conforme a campanha (5/10/15/20/25% em
`docs/RESULTADOS_convnext_regret.md` §2, e 15/20/30/40/50% em
`docs/RESULTADOS_oraculo_regret.md` §3), e as duas variantes do ConvNeXt só
foram medidas na primeira grade, que não alcança 30%. Fechar esta lacuna
exigiria reexecutar `oracle_regret.py::score_surrogate` sobre os dois
checkpoints do ConvNeXt (`surrogate_real` e `surrogate_regret`) com a grade
estendida até 30%, e não foi feito nesta auditoria, por não envolver
codificação nova nem alterar nenhuma das treze demais tabelas. Todas as
demais tabelas, da 8 à 21, estão completas com dado real e verificado.
