# Tabelas do Capítulo de Metodologia (Tabelas 1 a 7)

> Materialização das sete tabelas especificadas em `A2_TABELAS_E_FIGURAS.md`,
> Seção 1, "Capítulo de Metodologia". Cada bloco traz a identificação, a
> legenda no padrão fixo do perfil estilístico, a tabela em Markdown com os
> valores reais e a procedência. Nenhum valor foi estimado: todo número provém
> de um artefato ou de um documento já redigido deste projeto, e as duas
> agregações que não estavam prontas em nenhum artefato — a partição do
> conjunto de dados por sequência (Tabela 3) e o piso de ruído por configuração
> (Tabela 5) — foram calculadas nesta redação a partir do dado bruto, com o
> comando de reprodução registrado na respectiva procedência.

---

## Tabela 1 — Decomposição do custo de busca local por família de candidato

**Destino.** Capítulo de Metodologia, `M1_objeto_e_formulacao.md`, Seção 1.3.

**Legenda.** A Tabela 1 apresenta a decomposição do tempo de busca local por
família de candidato de partição, medida sobre 875.317 nós de decisão das três
sequências do conjunto de teste reservado — Jockey, RaceNight e RiverBank —,
em codificação intraquadro com `cpu-used=0`, com a coluna da divisão quadrada
excluída por conter a recursão.

| Família de candidato | % agregado do tempo de busca local | Mínimo entre as três sequências | Máximo entre as três sequências |
|---|--:|--:|--:|
| `PARTITION_NONE` | 30,1% | 27,9% | 32,9% |
| Formas retangulares (HORZ + VERT) | 35,6% | 30,8% | 38,2% |
| Formas AB | 20,4% | 17,6% | 25,5% |
| Formas 4-way | 13,9% | 11,3% | 15,8% |
| **Partições estendidas (AB + 4-way)** | **34,3%** | **28,9%** | **41,3%** |

**Nota.** O tempo de trabalho local de um nó é definido como a soma dos
candidatos não recursivos, e o denominador dos percentuais desta tabela é,
portanto, os **nove candidatos não recursivos** — `PARTITION_NONE`, as duas
retangulares, as quatro AB e as duas 4-way —, e não as dez formas do enum
`PARTITION_TYPE`. A coluna `PARTITION_SPLIT` é excluída porque o seu
temporizador engloba a recursão e contabilizaria o mesmo trabalho múltiplas
vezes. Esta distinção entre dez formas de partição e nove candidatos de custo
deve acompanhar toda citação destes percentuais, conforme registrado em
`M1_objeto_e_formulacao.md` Seção 1.3.

**Procedência.** Dado de origem:
`results/benchmark/partstats/part_timing_t1.csv`,
`results/benchmark/partstats_racenight/part_timing.csv` e
`results/benchmark/partstats_riverbank/part_timing.csv` (não versionados).
Script: `src/scripts/benchmark/analyze_partstats.py`. Documento-fonte:
`docs/RESULTADOS_C1_custo_por_candidato.md` §3, reproduzido em
`M1_objeto_e_formulacao.md` §1.3. Nenhum cálculo novo foi feito para esta
tabela: os valores já estavam tabulados no documento-fonte.

---

## Tabela 2 — Decomposição do custo das partições estendidas por tamanho de bloco

**Destino.** Capítulo de Metodologia, `M1_objeto_e_formulacao.md`, Seção 1.3,
imediatamente após a Tabela 1.

**Legenda.** A Tabela 2 apresenta a fração do tempo de busca local consumida
pelas partições estendidas, por tamanho de bloco, sobre o mesmo conjunto de
875.317 nós de decisão da Tabela 1.

| Tamanho de bloco | % do tempo local consumido pelas partições estendidas |
|---|--:|
| 8×8 | 0,0% |
| 16×16 | 8,7% |
| 32×32 | **50,0%** |
| 64×64 | **51,0%** |
| 128×128 | 35,3% |

**Nota.** Em blocos de 8×8 amostras as partições AB e 4-way não se aplicam, e
o percentual é nulo por definição, não por medição de custo zero. O custo das
partições estendidas concentra-se quase inteiramente nos blocos de 32×32 e de
64×64 amostras, onde responde por cerca de metade do tempo local, o que
restringe o alcance útil de um podador desta família aos blocos grandes. O
mesmo denominador de nove candidatos não recursivos da Tabela 1 vale aqui.

**Procedência.** Dado de origem: os mesmos três arquivos da Tabela 1. Script:
`src/scripts/benchmark/analyze_partstats.py`, com a agregação por tamanho de
bloco em vez de agregação global. Documento-fonte:
`docs/RESULTADOS_C1_custo_por_candidato.md` §3 ("Onde o custo se concentra"),
reproduzido em `M1_objeto_e_formulacao.md` §1.3.

---

## Tabela 3 — Composição do conjunto de dados por partição

**Destino.** Capítulo de Metodologia, `M2_instrumentacao_e_dataset.md`,
Seção 2.4.

**Legenda.** A Tabela 3 apresenta a composição do conjunto de dados por
partição de treino, validação e teste reservado, com o número de amostras e o
número de nós de decisão em cada uma, extraídos de dezesseis sequências UVG 4K
sob quatro pontos de quantização, com a partição feita por sequência, sem
vazamento de conteúdo.

| Partição | Sequências | Amostras totais (milhões) | Nós de decisão (milhões) |
|---|--:|--:|--:|
| Treino | 10 | 15,76 | 6,09 |
| Validação | 3 | 5,89 | 2,07 |
| Teste reservado | 3 | 5,32 | 1,91 |
| **Total** | **16** | **26,98** | **10,07** |

**Nota.** Amostra é toda linha registrada da árvore de particionamento visitada,
uma por nó; nó de decisão é a amostra cujo `block_dim` pertence a
{16, 32, 64}, isto é, exclui os blocos 8×8, folhas terminais cujo rótulo é
sempre `PARTITION_NONE`. A partição de sequências segue o congelamento
descrito em `docs/PROTOCOLO_avaliacao.md` §1: treino — Beauty, Bosphorus,
CityAlley, FlowerFocus, FlowerKids, ReadySetGo, ShakeNDry, SunBath, Twilight e
YachtRide; validação — HoneyBee, FlowerPan e Lips; teste reservado — Jockey,
RaceNight e RiverBank.

**Procedência.** Dado de origem: `results/dataset_h9/manifest.csv` (64 linhas,
verificado por `Glob`). A agregação por partição não estava pronta em nenhum
artefato do projeto sob esta forma tabular e foi calculada nesta redação, por
soma das colunas `num_samples` (amostras totais) e `dim16 + dim32 + dim64`
(nós de decisão) do manifesto, agrupadas pela sequência de cada linha segundo
a partição de `docs/PROTOCOLO_avaliacao.md` §1. Os totais obtidos —
15.760.617 amostras e 6.087.229 nós de decisão no treino; 5.892.674 e
2.070.428 na validação; 5.324.375 e 1.908.396 no teste reservado — reproduzem
exatamente os valores já redigidos em prosa em `M2_instrumentacao_e_dataset.md`
§2.4, o que confirma a agregação. Comando de reprodução (PowerShell, a partir
de `manifest.csv`): agrupar por sequência mapeada à partição fixada, somar
`num_samples` e `dim16+dim32+dim64` por grupo. Script de origem das colunas do
manifesto: `src/scripts/partition_dataset/rebuild_manifest_stats.py`.
Documento de suporte: `docs/ZENODO_datasheet.md` (Partes 1 e 2).

---

## Tabela 4 — As duas definições de redução de tempo: divergência por configuração

**Destino.** Capítulo de Metodologia, `M3_protocolo_avaliacao.md`, Seção 3.5.

**Legenda.** A Tabela 4 apresenta a divergência entre as duas definições de
redução de tempo em uso no projeto — a **canônica**, padronizada e adotada
por esta tese, e a **ponderada pelo tempo**, aqui referida como "da síntese"
— para as dezessete configurações da grade completa de substituição direta do
H9a e do H9c, com a diferença em pontos percentuais entre as duas, ordenadas
da maior para a menor magnitude de divergência.

| Configuração | TS% (definição canônica) | TS% (definição ponderada pelo tempo) | Δ (pp) |
|---|--:|--:|--:|
| H9c τ=0,95, cpu-used=2 | 40,73 | 37,76 | −2,97 |
| H9c τ=0,95, cpu-used=1 | 30,34 | 27,53 | −2,81 |
| H9c τ=0,90, cpu-used=2 | 42,07 | 39,30 | −2,76 |
| H9c τ=0,90, cpu-used=1 | 31,65 | 29,05 | −2,60 |
| Preset nativo, cpu-used=2 | 42,72 | 40,37 | −2,36 |
| Preset nativo, cpu-used=1 | 32,59 | 30,42 | −2,17 |
| H9a equilibrado, cpu-used=2 | 50,05 | 48,44 | −1,61 |
| H9a equilibrado, cpu-used=1 | 40,20 | 39,06 | −1,14 |
| H9c τ=0,90, cpu-used=0 | 13,59 | 14,63 | +1,05 |
| H9c τ=0,95, cpu-used=0 | 12,61 | 13,55 | +0,95 |
| Preset nativo, cpu-used=3 | 67,94 | 67,46 | −0,49 |
| H9c τ=0,95, cpu-used=3 | 70,25 | 69,80 | −0,44 |
| H9c τ=0,90, cpu-used=3 | 70,67 | 70,25 | −0,42 |
| H9a agressivo, cpu-used=2 | 60,97 | 60,62 | −0,35 |
| H9a agressivo, cpu-used=1 | 51,82 | 52,13 | +0,30 |
| H9a agressivo, cpu-used=3 | 77,30 | 77,59 | +0,29 |
| H9a equilibrado, cpu-used=3 | 73,09 | 72,96 | −0,13 |

**Nota.** A definição **canônica** calcula, para cada ponto de quantização, a
fração de tempo poupada `1 − t_configuração / t_âncora`, tira a média sobre os
quatro pontos e, então, a média sobre as sequências, de modo que cada ponto de
quantização pesa igualmente; é esta a definição **padronizada** e adotada em
todos os demais números desta tese. A definição **ponderada pelo tempo**
calcula, por sequência, `1 − Σt_configuração / Σt_âncora`, somando os tempos
sobre os quatro pontos antes de dividir, de modo que o resultado é dominado
pelo ponto de quantização mais custoso (`cq=20`). A divergência chega a cerca
de três pontos percentuais, e o sinal da diferença não é constante entre
configurações. As taxas BD não são afetadas por esta escolha, pois dependem
apenas de bytes e de PSNR, valores determinísticos; a ambiguidade é exclusiva
do eixo de tempo. As configurações "cpu-used=N" nomeiam a substituição direta
do podador nativo pelo H9a ou pelo H9c no nível de velocidade N; as linhas de
H9c sem sufixo de `cpu-used` correspondem à medição em `cpu-used=0`, isto é,
sem as demais heurísticas de velocidade dos *presets* nativos.

**Procedência.** Dado de origem:
`results/benchmark/fase6_analysis/ts_definitions.csv`, colunas `config`,
`ts_canonical`, `ts_sintese`, `delta_pp` (verificado por `Glob` e conferido
diretamente; arquivo regenerado em 2026-07-29, conteúdo lido nesta data). O
arquivo contém 24 configurações no total; as dezessete selecionadas para esta
tabela são as previstas pela especificação — H9a equilibrado e H9a agressivo
em `cpu-used` 1/2/3, H9c τ=0,90 e τ=0,95 em `cpu-used=0` e em `cpu-used` 1/2/3,
e os três *presets* nativos —, excluindo sete linhas adicionais
(`h9c_tau45`, `ml_aggr`, `ml_aggr_h9d`, `ml_aggr_h9d_pl20`, `ml_bal_h9d`,
`ml_bal_h9d_pl20`, `ml_balanced`) incorporadas ao artefato pela campanha do
H9d, posterior à especificação original de dezessete linhas. Script:
`src/scripts/fase6/analyze_frontier.py`. Documento-fonte:
`M3_protocolo_avaliacao.md` §3.5.

---

## Tabela 5 — Piso de ruído medido na comparação pareada de tempo

**Destino.** Capítulo de Metodologia, `M3_protocolo_avaliacao.md`, Seção 3.6.

**Legenda.** A Tabela 5 apresenta o piso de ruído medido por repetição
intercalada de três configurações — a âncora, o H9a equilibrado (`ml_balanced`)
e o *preset* nativo `cpu-used=1` — sobre quatro pontos de quantização na
sequência Crosswalk, com cinco repetições em execução contínua, o coeficiente
de variação mediano do tempo bruto por configuração e o desvio da redução de
tempo pareada por configuração.

| Configuração | CV mediano do tempo bruto (%) | Desvio da redução de tempo pareada (±pp) | Resolução efetiva (pp, dois desvios) |
|---|--:|--:|--:|
| Âncora | 0,33% | não aplicável (é a própria referência) | não aplicável |
| `ml_balanced` (H9a equilibrado) | 0,33% | ±0,23 pp | 0,46 pp |
| `native_cpu1` (preset nativo, cpu-used=1) | 0,19% | ±0,09 pp | 0,18 pp |

**Nota.** A "redução de tempo pareada" e a sua resolução só existem para uma
configuração medida contra a âncora; a linha da âncora reporta apenas a
dispersão do seu próprio tempo bruto, uma vez que a redução de tempo da
âncora contra si mesma é nula por construção em toda repetição. A resolução
efetiva é tomada como dois desvios padrão da redução de tempo sobre as cinco
repetições: para o H9a equilibrado, cuja redução medida é de 20,29%, o desvio
de ±0,23 pp estabelece uma resolução de 0,46 pp; para o *preset* nativo
`cpu-used=1`, cuja redução medida é de 32,84%, o desvio de ±0,09 pp estabelece
uma resolução de 0,18 pp. Diferenças de redução de tempo inferiores a esta
resolução não são citadas como positivas em nenhum ponto desta tese. Tomando
as doze células de (configuração × ponto de quantização) em conjunto, o
coeficiente de variação mediano combinado é de 0,28% e o máximo de 0,64%,
valores que reproduzem exatamente os já redigidos em
`docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3.1 e confirmam a agregação por
configuração feita nesta tabela.

**Procedência.** Dado de origem:
`results/benchmark/fase6_repeat/raw_results.csv` (verificado por `Glob`; cinco
repetições por configuração e por ponto de quantização). O desvio da redução
de tempo pareada e a resolução efetiva reproduzem os valores já redigidos em
`docs/RESULTADOS_BLOCO7_E3_DEC_E2.md` §3.2. O coeficiente de variação mediano
**por configuração** não estava tabulado em nenhum artefato — o documento-fonte
reporta apenas o valor agregado das três configurações combinadas — e foi
calculado nesta redação: agrupam-se as cinco repetições por (configuração,
ponto de quantização), calcula-se o desvio padrão amostral e a média do tempo
bruto em cada grupo, obtém-se o coeficiente de variação de cada grupo e toma-se
a mediana dos quatro pontos de quantização por configuração. Os doze valores
de grupo, combinados, reproduzem o 0,28%/0,64% do documento-fonte, o que
verifica o cálculo. Scripts: `src/scripts/fase6/encode_repeat.py --seq
Crosswalk --reps 5` (geração) e `src/scripts/fase6/report_e3_dec_e2.py`
(agregação original). Documento-fonte: `docs/RESULTADOS_BLOCO7_E3_DEC_E2.md`
§3 e §4, reproduzido em `M3_protocolo_avaliacao.md` §3.6.

---

## Tabela 6 — Vetor de atributos por bloco (A–E)

**Destino.** Capítulo de Metodologia, `M4_atributos_e_politica.md`, Seção 4.1.

**Legenda.** A Tabela 6 apresenta a decomposição do vetor de atributos das
soluções da família H9 em cinco blocos, com o número de atributos de cada um,
o momento em que a informação fica disponível no fluxo de controle do
codificador e a solução que o utiliza.

| Bloco | Conteúdo | Nº de atributos | Momento de disponibilidade | Utilizado por |
|---|---|--:|---|---|
| A | Descritores de luminância do bloco e do seu contexto hierárquico: variância global e por quadrante, dispersão e heterogeneidade entre quadrantes, gradiente horizontal e vertical e sua orientação, perfis de linha e de coluna, aresta mais forte, densidade de arestas fortes, nível DC, quantização normalizada, contraste com o bloco-pai de dimensão 2n×2n e com os três blocos-irmãos, posição na unidade de 64 px | 24 | Pré-busca | pixels24; H9a; H9c/H9d |
| B | Vizinhança de particionamento causal: disponibilidade dos vizinhos acima e à esquerda, larguras e alturas em log2 dos blocos já decididos nessas direções, granularidade relativa da vizinhança e sua anisotropia | 8 | Pré-busca | H9a; H9c/H9d |
| C | Quantização e posição: passo de quantização normalizado, posição normalizada de linha e de coluna no quadro, profundidade do nó | 4 | Pré-busca | H9a; H9c/H9d |
| D | Proxy de resíduo intraquadro: SATD de Hadamard do bloco-fonte, sem predição | 2 | Pré-busca | nenhum dos dois (só H9b, avaliado fora do codificador) |
| E | Custo de taxa-distorção real de `PARTITION_NONE`: `log1p` da taxa, da distorção e do custo RD | 3 | Pós-`PARTITION_NONE` | H9c/H9d |

**Nota.** O bloco E só existe depois que o codificador avaliou
`PARTITION_NONE` para o nó corrente, o que o torna estruturalmente
indisponível a qualquer podador que atue antes da busca — é por isso que o
H9a, que age em `av1_prune_partitions_before_search`, nunca o consome. O
bloco D é computado antes da busca, mas foi excluído de todos os conjuntos
implantados: o `pixels24` é o bloco A isolado (24 atributos); o H9a é a soma
A+B+C (36 atributos); o H9c e o H9d consomem a soma A+B+C+E (39 atributos),
diferindo apenas na ação executada sobre o mesmo vetor e o mesmo ponto de
enganche. O vetor completo de 41 atributos (A+B+C+D+E) é utilizado apenas na
etapa de validação de sinal fora do codificador, sob os rótulos H9b (A+B+C+D,
38 atributos) e H9c (A+B+C+E, 39 atributos, sem o D).

**Procedência.** Não há CSV de origem: a tabela é construída por leitura
direta do código-fonte. Script/definição:
`src/scripts/partition_model/features.py`, docstring de layout (linhas
19–41), dicionário `H9_SUBSETS` (linhas 214–220), `NUM_FEATURES_H9A = 36`
(linha 204) e `NUM_FEATURES_H9C = 39` (citado em `M4_atributos_e_politica.md`
linha 49). Documento-fonte: `M4_atributos_e_politica.md` §4.1, §4.2 e §4.4.
Nenhum número é inventado: as contagens de atributos e os índices já constam
de `docs/RESULTADOS_auditoria_dominio_pixels.md` §2.

---

## Tabela 7 — Os quatro desenhos de atribuição

**Destino.** Capítulo de Metodologia, `M6_modelos_e_atribuicao.md`,
Seção 6.6.

**Legenda.** A Tabela 7 apresenta os quatro desenhos experimentais adotados
para separar o mérito do modelo do mérito da política de poda, com a
pergunta que cada um responde, a variável controlada e a variável
manipulada.

| Desenho | Pergunta respondida | Variável controlada | Variável manipulada | Seção de Resultados que o utiliza |
|---|---|---|---|---|
| Ablação de atribuição em três braços | O ganho de tempo medido é atribuível à fonte da pontuação, e não à política de poda que o envolve? | Política de poda, codificador, âncora e sequência | Fonte da probabilidade P(NONE): modelo; limiar de variância; pontuação aleatória | R1 §1.3; R2 §2.6 |
| Substituição direta do podador nativo | O podador aprendido, sozinho, supera a rede convolucional nativa quando ambos operam no mesmo nível de velocidade? | Nível de `cpu-used` e as demais heurísticas de velocidade do *preset* | Podador de partição intraquadro: rede convolucional nativa; podador aprendido | R2 §2.5; R3 §3.4 |
| Decomposição com neutralização de alavanca | Quanto do ganho medido da pilha pertence a cada podador, quando dois coexistem no mesmo binário? | Sequência, âncora e binário de codificação | Podador ativo: cada um isolado, com o outro neutralizado por limiares inatingíveis; a pilha completa | R3 §3.3; R3 §3.5; R4 §4.6 |
| Simulação oráculo | O sinal de um conjunto de atributos ou de um modelo supera o piso trivial da variância, a risco casado, antes de se pagar o custo da codificação real? | Nós de decisão do conjunto de validação e de teste, política replicada, modelo de custo por candidato | Subconjunto de atributos ou arquitetura de modelo avaliada | R1 §1.4; R5 §5.2 |

**Nota.** Os dois primeiros desenhos controlam o invólucro inteiro dentro do
codificador real e produzem, por isso, a evidência mais forte de atribuição;
os dois últimos operam fora do codificador — o terceiro sobre codificações
reais já produzidas, o quarto sobre simulação —, e a simulação oráculo carrega
a ressalva de que a ordenação relativa entre modelos competitivos pode se
inverter ao ser medida no codificador real, conforme registrado em
`M6_modelos_e_atribuicao.md` §6.10.

**Procedência.** Tabela conceitual, sem CSV de origem: elaboração própria a
partir da prosa das Seções 6.6 a 6.10 de `M6_modelos_e_atribuicao.md`. Nenhum
número é introduzido; a tabela é puramente organizativa.
