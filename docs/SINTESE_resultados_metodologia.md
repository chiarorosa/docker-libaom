# Síntese de Metodologia e Resultados — poda aprendida de particionamento intra no AV1

> **DOCUMENTO VIVO.** Última atualização: **2026-07-17**. Há experimentos em
> curso (ver §10); as tabelas marcadas *(parcial)* ou *(em andamento)* serão
> revistas conforme os dados fecharem. Este documento consolida toda a
> investigação para servir de base à escrita do **Capítulo de Metodologia** e do
> **Capítulo de Resultados**. Fontes primárias: `docs/ANDAMENTO_tese.md`,
> `docs/RESULTADOS_fase5.md`, `docs/RESULTADOS_fase6.md`,
> `docs/PLANO_H9_contribuicao_tese.md`, `docs/PROTOCOLO_avaliacao.md`.

---

## 1. Objeto e enquadramento

O objeto da tese é **acelerar a decisão de particionamento em codificação intra
do AV1** por meio de **poda aprendida** do espaço de busca recursivo de
partições (`av1_rd_pick_partition`), preservando a eficiência de compressão. A
decisão de particionamento é um dos maiores custos do codificador: para cada
superbloco, o AV1 avalia recursivamente `PARTITION_NONE`, `SPLIT`, as duas
retangulares (HORZ/VERT), as quatro AB e as duas 4-way — até 9 formas por nó. Um
preditor que elimine formas improváveis **antes** (ou **durante**) a busca reduz
o tempo sem, idealmente, degradar a taxa-distorção (RD).

A investigação produziu **três soluções distintas**, que a tese apresenta e
compara sob diferentes cenários e óticas:

1. **Heurística baseada em CNN (domínio de pixels)** — um modelo convolucional
   (ConvNeXt) desenhado para espelhar a estrutura da árvore de particionamento do
   AV1, treinado sobre a luminância (Y). Serviu de **instrumento de diagnóstico e
   de referência de limite superior**; sua versão implantável foi obtida por
   **destilação** de conhecimento (modelo substituto → modelo estudante MLP).
2. **Heurística baseada em NN pré-busca (H9a)** — um MLP por tamanho de bloco que
   consome **contexto taxa-distorção barato** (vizinhança de particionamento +
   quantização/posição, além dos pixels) e decide, **antes** da busca, se poda o
   nó. É a **contribuição central** da tese.
3. **Heurística baseada em NN pós-NONE (H9c)** — um MLP leve que atua **depois**
   de o codificador ter avaliado `PARTITION_NONE`, usando o custo RD real dessa
   avaliação para decidir se encerra a busca. É um **refinamento complementar**.

O fio condutor metodológico de toda a investigação é a **caracterização honesta
de um limite superior de desempenho informacional**: quanto do particionamento é
de fato predizível, e a partir de que informação.

---

## 2. Metodologia (estrutura do Capítulo de Metodologia)

### 2.1 Instrumentação e geração do conjunto de dados

O codificador de referência (libaom v3.10.0) foi instrumentado sob a guarda de
compilação `LOG_PARTITION_DATA` para registrar, em cada nó de particionamento de
quadros intra, uma estrutura `PartitionSample` com: a decisão RD-ótima (rótulo),
a luminância do bloco, e — a partir da Fase 1 do H9 — o **contexto RD** (blocos
B/C/D/E, §2.5). A extração usa **`cpu-used=0`** (busca RD completa), que fornece
o *ground truth* de particionamento; qualquer preset mais rápido já contém poda
heurística e contaminaria os rótulos.

> **Lição metodológica registrada (memória `partition-dataset-pipeline`):** o
> `aomenc --limit` conta quadros de *entrada*, não de saída; e a extração deve
> rodar sempre dentro do contêiner Docker (libaom e ML/GPU), sendo o Windows
> apenas para edição.

### 2.2 O defeito de luminância nula (lição metodológica central)

Descobriu-se (2026-07-08) que o conjunto de dados armazenava a luminância como
`float32` normalizado em [0,1], enquanto os consumidores em Python assumiam
`uint8` em [0,255]: `features.py` truncava para inteiro (→ **blocos todos
zero**) e o carregador do substituto normalizava de novo por 255 (→ entrada
quase nula). **Toda a cadeia H1–H6 fora treinada sobre luminância em branco.** O
dado bruto estava correto (`round(pkl·255)` = quadro-fonte, `maxdiff=0`); apenas
os consumidores estavam errados. Correção: `data._denorm_uint8` como fonte única
de verdade; caches invalidados; e uma **asserção de luminância real**
(`assert_real_luma`) passou a guardar treino/destilação/simulação.

**Consequência metodológica:** todo resultado anterior à correção foi
**re-medido**; e a lição — *asserir que os dados de treino têm variância não
nula* — é parte da defesa de validade da tese.

### 2.3 Protocolo de avaliação (congelado antes dos resultados)

Para impedir seleção *a posteriori*, o protocolo foi **congelado por commit
assinado** (`docs/PROTOCOLO_avaliacao.md`) antes de qualquer medição de teste:

- **Partição por sequência, sem vazamento:** 10 sequências de **treino**, 3 de
  **validação** (seleção de modelo/limiar), 3 de **teste** (números da tese,
  nunca vistas) — Jockey/RaceNight/RiverBank fixadas *a priori*.
- **Cobertura:** 4 QP (`cq` 20/32/43/55) × ≥5 quadros (amostragem temporal);
  ≥10 quadros no teste para taxa BD estável.
- **Critérios de decisão em cascata (gates):** cada fase tem um portão
  quantitativo; o custo caro (integração em C, benchmark real) só é pago depois
  que o sinal foi provado *offline*. Isto torna a cadeia de decisão auditável.

### 2.4 Métricas — os três pilares

Todo resultado é reportado em **três pilares complementares**, contra a **âncora
libaom original em `cpu-used=0`**:

1. **BD-Rate (PSNR-Y, Bjøntegaard)** — custo em eficiência de compressão (menor é
   melhor).
2. **Redução de tempo TS% = (1 − 1/*speedup*)·100** — economia de tempo de parede.
3. **Speedup** — razão de tempo âncora/configuração.

Métrica auxiliar de compromisso: **TS/BD** (pontos de TS por ponto percentual de
BD-Rate) — quanto tempo se "compra" por unidade de qualidade cedida; maior é mais
eficiente.

### 2.5 Vetor de atributos (blocos A–E)

Fonte única de verdade em `features.py`; o lado C **espelha** exatamente estas
fórmulas (paridade bit-a-bit verificada, §2.6):

- **Bloco A — pixels (0–23):** variância, quadrantes, gradientes H/V, perfis
  linha/coluna, contexto hierárquico (pai 2n×2n, contraste com irmãos), posição.
  O índice 0 (variância) é o *baseline* trivial.
- **Bloco B — vizinhança de particionamento (24–31), grátis:** tamanhos/tipos das
  partições dos blocos acima/esquerda (causais), disponibilidade de bordas,
  granularidade relativa. Dados já residentes → custo de inferência ~zero.
- **Bloco C — quantização/posição (32–35), grátis:** `log(dc_q²)`, qindex, posição
  normalizada.
- **Bloco D — proxy de resíduo intra (SATD de Hadamard), barato:** descartado na
  Fase 3 (não agrega sinal sobre A+B+C).
- **Bloco E — custo RD real do `PARTITION_NONE` (36–38):** `log(none_rate)`,
  `log(none_dist)`, `log(none_rdcost)`. **Só disponível pós-NONE** → é o insumo
  exclusivo do H9c.

Subconjuntos: **H9a = A+B+C (36 atributos)**; **H9c = A+B+C+E (39 atributos)**.

### 2.6 Builds, guarda de compilação e paridade C↔Python

- `libaom_perf_anchor` — `src/aom_baseline` (v3.10.0 **puro**, sem estudante); a
  **âncora** e os presets nativos.
- `libaom_perf` — `src/aom` com `-DPARTITION_ML_STUDENT=1` (estudantes H9a e H9c
  embarcados, controlados por variáveis de ambiente).
- `libaom_ml_check` / `libaom_noop` — validação de paridade e no-op.

**Garantia de fidelidade:** todo código novo vive sob `#if PARTITION_ML_STUDENT`;
com a flag desligada, o bitstream é **byte-idêntico** à âncora (md5 verificado). A
paridade C↔Python das *features* é bit-a-bit; a inferência reusa o
`av1_nn_predict` nativo (nenhum código de inferência novo em C).

### 2.7 Política de poda (as ações e os limiares τ)

O estudante emite `[P(NONE), P(SPLIT), P(REST)]` (softmax de 3 classes). A
política pré-busca (H9a) aplica, em cascata, por limiares τ **por nível de bloco**:
- `P(NONE) > τ_none` → desabilita todos os splits (compromete NONE);
- senão `P(SPLIT) > τ_split` → apenas split quadrada;
- senão `P(REST) < τ_rest` → desabilita retangulares (rect-off).

O H9c, por operar **pós-NONE**, tem ação **binária**: `P(NONE) > τ` → encerra a
busca ali (reusa `av1_disable_all_splits`, seguro pois `PARTITION_NONE` já foi
avaliado). Os τ são o botão de operação que traça a curva TS × BD.

---

## 3. Solução 1 — CNN no domínio de pixels (ConvNeXt): diagnóstico e limite superior

**Arquitetura.** Um modelo convolucional (ConvNeXt) desenhado para **espelhar a
hierarquia da árvore de particionamento** do AV1 (multi-nível), consumindo a
luminância do superbloco. Papel: modelo substituto (*surrogate*) de alta
capacidade.

**Limite superior H8 (surrogate replay).** Para medir *quanto do
particionamento é predizível a partir dos pixels*, as probabilidades do
substituto são gravadas por nó e **re-injetadas** no codificador
(`AV1_STUDENT_PROBS_FILE`), medindo a BD-Rate que se obteria *através do mesmo
gancho de poda* com o teto de qualidade do substituto — sem inferência
convolucional em C. É a **referência de limite superior** do domínio de pixels.

**Destilação.** A versão implantável foi obtida por **destilação de conhecimento**
do substituto (ConvNeXt) para um **modelo estudante MLP** que reusa o
`av1_nn_predict`. A destilação **reduziu o custo computacional** (de convoluções
multi-resolução para um MLP denso por nó), mas também **reduziu os ganhos** — o
estudante herda uma fração do teto do substituto.

**Resultado negativo decisivo (ablação de atribuição).** Sob política casada
(NONE-commit), variando apenas a *fonte do escore* — ML vs variância vs aleatório
— a **variância trivial `exp(−var/1000)` empata ou supera o ML** em *speedup*
casado (ex.: a 1,3×, ML 1,39% BD vs variância 0,76%). Como a variância é **um dos
24 atributos** do próprio modelo, um bom modelo não deveria perder para ela.

> **Leitura (a espinha dorsal da tese):** o domínio de pixels **satura na
> variância**. A afirmação é rigorosa *porque* o modelo de pixels testado não era
> fraco — um ConvNeXt de alta capacidade, com medição de limite superior (H8), e
> ainda assim sem separar-se da variância. Isso transforma "o modelo não
> conseguiu" em "**o sinal não está nos pixels**", e **motiva** a virada para o
> contexto RD (H9a). As convoluções são o **instrumento de diagnóstico e a
> referência de limite superior**, não o mecanismo implantado.

**Ressalvas (para o Capítulo de Resultados):** (i) alto custo computacional da
CNN; (ii) limitação informacional inerente às amostras Y; (iii) a destilação
troca ganho por custo.

---

## 4. Solução 2 — H9a: poda pré-busca por contexto RD barato

**Hipótese e projeto.** Se os pixels saturam, o sinal restante está no
**contexto taxa-distorção**. O H9a acrescenta os blocos B (vizinhança) e C
(quantização/posição) aos 24 pixels — **contexto grátis** (dados já residentes,
custo de inferência ~zero). Decisão pela evidência (Fase 3): como o ganho é
**tabular e não-pixel**, o ConvNeXt (pixel-only) não pode ensiná-lo, e o Gate 2
provou um MLP por tamanho; logo o **estudante implantável é treinado diretamente**
sobre A+B+C (36 atributos), entropia cruzada de rótulo duro, **sem professor**.

**Cadeia de portões (todos PASSARAM):**
- **Gate 2 (sinal *offline*, 10 seqs treino):** no *lever* NONE-commit, redução de
  custo em risco casado — variância ~0; pixels24 10–19%; **H9a 16–25%** (≈ +50%
  relativo sobre pixels). H9c (teto, `none_rdcost`) ~2× H9a.
- **Gate 3 (validação HoneyBee/FlowerPan/Lips):** H9a ~55–58% de redução de custo
  a SPLIT-lost ≤ 1% (oráculo), ~2× os pixels.
- **Gate 4 (integração C):** paridade C↔Python bit-a-bit nos 36 atributos;
  no-op byte-idêntico; testes AllIntra passam.

**Resultado no teste reservado (Fase 5, held-out Jockey/RaceNight/RiverBank, 10
quadros, cpu-used=0)** — média dos 3, curva conservadora:

| ponto | BD-Rate | TS% | speedup |
|---|--:|--:|--:|
| P0 (conservador) | 0,155% | 19,05% | ~1,24× |
| P_rect (equilibrado) | 0,464% | 26,49% | ~1,36× |
| P_ref | 0,595% | 29,53% | ~1,42× |
| H8 (teto do substituto) | 0,12–0,53% | 2,4–3,3% | ~1,03× |

Veredito da Fase 5: **redução de tempo substancial e atribuível** ao contexto RD;
sob política casada, o escore RD do H9a acessa um regime de BD muito mais baixo
que a variância (a variância nem opera em risco baixo). O H8 confirma que o teto
de pixels é quase "de graça" mas minúsculo em TS.

**Validação universal (Fase 6, CTC Classe A1, 8 seqs 4K 10-bit, All-Intra, 15
quadros)** — comparação do praticante: ML@cpu0 vs o **botão de velocidade nativo**
(presets cpu1/2/3). Método de cálculo único, 8 seqs, vs âncora cpu0:

| config | BD-Rate | TS% | speedup | TS/BD |
|---|--:|--:|--:|--:|
| H9a equilibrado (ml_balanced) | 0,568% | 19,26% | 1,242× | 33,9 |
| H9a agressivo (ml_aggr) | 1,403% | 34,14% | 1,530× | 24,3 |
| libaom `cpu-used=1` | 0,449% | 30,42% | 1,440× | **67,8** |
| libaom `cpu-used=2` | 0,536% | 40,37% | 1,682× | **75,3** |
| libaom `cpu-used=3` | 2,722% | 67,46% | 3,082× | 24,8 |

**Leitura honesta:** na fronteira taxa-BD × tempo, o **preset nativo domina** o
H9a a cpu0 — a CNN nativa é mais eficiente (TS/BD ~68–75 vs ~24–34). Isso atinge a
*utilidade prática* (Capítulo de Resultados), não a validação metodológica (o
escore RD do H9a domina a variância sob política casada — mantém-se). O nicho a
favor do ML é o **regime de baixíssimo speedup** que o passo discreto do nativo
(cpu0→cpu1 já salta para ~30% TS) não cobre.

**Comparação de categoria correta — swap (H9a substitui a CNN nativa).** Os
presets cpu1/2/3 misturam a CNN-ML nativa com **dezenas de heurísticas não-ML**;
para isolar o *mérito do pruner*, a cpu fixo desliga-se a CNN nativa
(`AV1_DISABLE_NATIVE_CNN=1`) e coloca-se o H9a como **único** pruner intra. Média
8 seqs (método único, vs âncora cpu0):

| cpu | config | BD-Rate | TS% | speedup |
|:--:|---|--:|--:|--:|
| 1 | CNN nativa | 0,449% | 30,42% | 1,440× |
| 1 | H9a balanced | 0,915% | 39,06% | 1,647× |
| 1 | H9a aggressive | 1,685% | 52,13% | 2,106× |
| 2 | CNN nativa | 0,536% | 40,37% | 1,682× |
| 2 | H9a balanced | 1,030% | 48,44% | 1,950× |
| 3 | CNN nativa | 2,722% | 67,46% | 3,082× |
| 3 | H9a balanced | 3,866% | 72,96% | 3,702× |

O H9a-swap **poda mais** (mais TS) mas a **custo de BD desproporcional** (~2–3× a
BD da nativa no mesmo cpu) → a CNN nativa é mais eficiente. **Contraponto de valor:
custo computacional.** O H9a é um MLP por tamanho (36→64→32→3, ≈13,6 mil
parâmetros no total, forward denso único por nó, contexto B/C residente); a CNN
nativa é convolucional multi-resolução (5 camadas + 4 ramos DNN), ordens de
grandeza mais cara. O H9a **captura parte substancial do ganho a fração do custo
computacional** do pruner de produção — a leitura defensável do swap.

---

## 5. Solução 3 — H9c: refinamento pós-NONE (contexto RD real)

**Motivação e projeto.** O H9c estende o contexto RD do H9a com o **bloco E** — o
custo RD **real** do `PARTITION_NONE`, disponível só *depois* que o codificador
avaliou essa hipótese. Arquitetura leve (MLP 39→64→32→3); o gancho
`av1_prune_after_none` fica dentro de `av1_rd_pick_partition`, logo após
`none_partition_search()`, e decide se encerra a busca (ação binária). Gate por
`AV1_STUDENT_H9C_ENABLE` (desligado por padrão; no-op byte-idêntico).

**Gate 3 (validação, oráculo) PASSOU com folga:** 61,2% de redução de custo a
apenas 0,20% de SPLIT-lost — 5× menos risco que o teto de 1%, superando o H9a
nesse critério. **Gate C (paridade + no-op) PASSOU.**

**Descoberta corretiva — o *confound* do H9a (lição metodológica).** O build
`libaom_perf` roda o estudante **H9a sempre** em quadros intra
(`try_student_prune`, sem flag de habilitação, ao contrário do H9c). Os primeiros
testes do H9c setaram **apenas** o ambiente do H9c, deixando o H9a nos seus
defaults (τ 0,9/0,9) — logo mediram **H9a + H9c empilhados**, não H9c isolado. A
re-execução com H9a neutralizado (τ=2/2/−1, única variável alterada) quantifica
(Neon1224 cpu0, vs âncora):

| config | BD-Rate | TS% |
|---|--:|--:|
| H9a@0,9 sozinho | 0,312% | 17,10% |
| H9a@0,9 + H9c | 0,270% | 17,36% |
| **H9c sozinho** | 0,037% | **4,23%** |

**O H9c isolado poda muito pouco (~3–9% TS);** 82–96% do TS antes atribuído a ele
vinha do H9a. Empilhar H9c sobre H9a **não adiciona TS** (+0,26pp), mas **baixa
levemente o BD** ao mesmo TS (0,312→0,270; TS/BD 54,8→64,3) — indício de que o H9c
*corrige* decisões do H9a via rdcost real (ressalva: 1 seq, delta pequeno, a
confirmar).

**Swap limpo (H9c substitui a CNN nativa; H9a neutralizado desde o script).** Média
8 seqs:

| cpu | config | BD-Rate | TS% | speedup |
|:--:|---|--:|--:|--:|
| 1 | CNN nativa | 0,449% | 30,42% | 1,440× |
| 1 | H9c (τ0,95) | 0,414% | 27,53% | 1,383× |
| 1 | H9c (τ0,90) | 0,448% | 29,05% | 1,415× |
| 2 | CNN nativa | 0,536% | 40,37% | 1,682× |
| 2 | H9c (τ0,95) | 0,516% | 37,76% | 1,613× |
| 2 | H9c (τ0,90) | 0,539% | 39,30% | 1,659× |
| 3 | CNN nativa | 2,722% | 67,46% | 3,082× |
| 3 | H9c (τ0,90) | 3,397% | 70,25% | 3,390× |

**Leitura:** como substituto da CNN nativa, o H9c **empata praticamente com ela a
cpu1 e cpu2** (fronteira quase idêntica; a τ0,95 o H9c tem **BD menor** que a
nativa, com um pouco menos de TS), e é **muito mais competitivo que o H9a era no
swap** (H9a custava ~2–3× o BD). Só a cpu3 o H9c perde. Ou seja, o H9c é um
**pruner de partição intra equivalente à CNN nativa nos presets práticos
(cpu1/2), a custo computacional muito menor** — o cenário em que ele tem mérito
próprio. Como camada extra sobre o H9a a cpu0, não muda a foto.

---

## 6. Análise integrada — fronteira Pareto e as três conclusões

**Fronteira Pareto global (BD × TS, todos os níveis cpu, média 3 seqs
Boxing/FoodMarket2/Tango).** Pontos não-dominados, do menor BD ao maior:
`H9c(cpu0) 0,21%/13,9%` → `H9c 0,23%/15,0%` → `native_cpu1 0,37%/28,6%` →
`H9c_cpu1 0,39%/29,6%` → `native_cpu2 0,41%/38,2% (TS/BD 94, pico)` →
`H9c_cpu2 0,44%/39,1%` → `H9a_bal_cpu2 1,13%/47,1%` → `native_cpu3 2,80%/66,4%` →
`H9c_cpu3 / H9a_cpu3 3,5–4,8%/70–78%`.

**Conclusão 1 — ninguém DOMINA a CNN nativa.** Nenhum ponto ML é estritamente
melhor (mais TS a ≤ BD). O H9c empata a cpu1/2; a nativa mantém o pico de
eficiência (TS/BD 78–94).

**Conclusão 2 — o H9c é dono do extremo de baixo BD que a nativa não alcança.**
Os pontos de menor BD de toda a fronteira (0,21–0,23% BD, 14–15% TS) são do ML a
cpu0. A escada discreta da nativa pula de cpu0 (0% TS) para cpu1 (~30% TS),
deixando **todo o regime 0–30% TS descoberto** — o ML preenche continuamente. O
valor prático é **granularidade fina em baixo speedup**, não superar o pico.

**Conclusão 3 — os *levers* não se somam (limite superior informacional).** H9a
(pixels+contexto), H9c (rdcost pós-NONE) e a CNN nativa exploram o **mesmo sinal
correlacionado** — os "blocos fáceis". Prova direta: H9c sobre H9a = +0,26pp de
TS. É a mesma história de saturação que a Solução 1 estabeleceu no domínio de
pixels, agora confirmada no domínio RD.

**Argumento transversal — custo computacional.** As três soluções ML são MLPs
leves (≈13,6 mil parâmetros, forward denso por nó, contexto residente) frente à
CNN nativa (convolucional multi-resolução, 5 camadas + 4 ramos DNN). Igualar o
SOTA embarcado a fração do custo computacional é a contribuição de engenharia
defensável. *(Pendente: microbenchmark isolado do pruner — μs/bloco ou FLOPs —
para transformar o argumento estrutural em número medido; ver ANDAMENTO §5.)*

---

## 7. Estrutura proposta para os capítulos

### 7.1 Capítulo de Metodologia
1. Objeto, escopo e formulação do problema (§1).
2. Instrumentação e geração do *ground truth* (§2.1); a lição do defeito de
   luminância (§2.2).
3. Protocolo de avaliação congelado: partição sem vazamento, gates em cascata,
   três pilares e métrica de compromisso (§2.3–2.4).
4. Projeto de atributos A–E e a política de poda por τ (§2.5, §2.7).
5. Arquitetura de software: builds, guarda de compilação, paridade C↔Python,
   garantia de no-op (§2.6).
6. As três arquiteturas de modelo (ConvNeXt substituto/destilação; MLP H9a; MLP
   H9c pós-NONE) e a metodologia de atribuição (ablações ml/variância/aleatório;
   swap ML-vs-ML; decomposição com neutralização de *lever*).

### 7.2 Capítulo de Resultados
1. Diagnóstico do domínio de pixels: teto H8 e o resultado negativo da atribuição
   (Solução 1) — *por que* a virada para o contexto RD (§3).
2. H9a: gates, teste reservado (Fase 5), validação universal CTC (Fase 6) vs
   presets nativos, e swap ML-vs-ML (§4).
3. H9c: gate, o *confound* e sua correção, o swap limpo e a decomposição (§5).
4. Análise integrada: fronteira Pareto, as três conclusões, custo computacional
   (§6).
5. Discussão: limite superior informacional; utilidade prática vs validação
   metodológica; o papel de nicho e de substituto-leve das soluções ML.

---

## 8. Tabelas-mestre (referência rápida, vs âncora libaom cpu-used=0)

Método de cálculo **único e consistente** (BD-Rate PSNR-Y Bjøntegaard; TS% de
speedup agregado; média das sequências). Ver §4–§6 para as tabelas por cenário.

- **Fase 5 (teste reservado, 3 seqs):** H9a P_rect ~0,46%/26,5%; P_ref ~0,60%/29,5%.
- **Fase 6 CTC (8 seqs) — praticante:** H9a bal 0,568%/19,3%; H9a aggr 1,403%/34,1%;
  nativo cpu1 0,449%/30,4%; cpu2 0,536%/40,4%; cpu3 2,722%/67,5%.
- **Swap 8 seqs — ML-vs-CNN nativa:** ver §4 (H9a) e §5 (H9c). Resumo: H9c ≈ nativa
  a cpu1/2; H9a poda mais a BD desproporcional.

---

## 9. Ameaças à validade e limitações

- **Oráculo superestima o tempo real (~5×):** os gates *offline* (Gate 2/3) usam
  redução de custo de oráculo; a decisão final é sempre tempo de parede
  (Gate 5/benchmark). As margens *relativas* é que valem *offline*.
- **Ruído de poucos quadros:** Fase 6 usa 15 quadros; deltas de BD < ~0,1% podem
  estar na margem de ruído de conteúdo. Diferenças pequenas (ex.: o indício de
  §5) precisam de confirmação multi-sequência.
- **Grade de τ congelada:** calibrada na validação; no teste pode não cobrir todo
  o espectro de speedup (estender seria *test-tuning*).
- **Dependência de conteúdo:** a eficiência relativa varia por sequência (ex.:
  FoodMarket2 é um *outlier* pró-nativo); médias podem mascarar isso.
- **`--cq-level` vs QP do CTC:** manteve-se a grade da tese (20/32/43/55) por
  consistência com a validação, não os qindex exatos do guia CTC (escolha
  metodológica registrada).

---

## 10. Experimentos em andamento (este documento é VIVO)

| experimento | estado | destino |
|---|---|---|
| Swap H9c — 8 seqs completas | ✅ **concluído** (2026-07-17) | `results/benchmark/fase6_swap_h9c/` |
| Isolação/confound (Neon1224) + H9a@default | ✅ concluído | `fase6/` (`h9ciso_*`, `h9adef`) |
| **Frontier-check combinado** (H9a-conservador + H9c, swap, Tango) | 🔄 **em andamento** | `results/benchmark/fase6_swap_combo/` |
| Microbenchmark isolado do pruner (μs/FLOPs) | ⬜ pendente | — |
| H9c swap: análise de 8 seqs e §5 fechado com todas | ◐ tabelas a consolidar | este doc |

**Próxima atualização deste documento:** ao fechar o *frontier-check* combinado
(§6, Conclusão 3 — testa se H9a-conservador + H9c fura a fronteira da nativa;
prior: não fura, por sinal correlacionado). Cobertura por sequência/config está
inventariada em `docs/ANDAMENTO_tese.md §8.4` e no histórico de commits
`ml-partition-dev`.

---

*Documento gerado a partir de `docs/ANDAMENTO_tese.md`, `RESULTADOS_fase5.md`,
`RESULTADOS_fase6.md`, `PLANO_H9_contribuicao_tese.md`, `PROTOCOLO_avaliacao.md` e
dos CSV em `results/benchmark/{fase6,fase6_swap,fase6_swap_h9c}/`.*
