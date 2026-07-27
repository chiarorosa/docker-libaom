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

A investigação produziu **quatro soluções distintas**, que a tese apresenta e
compara sob diferentes cenários e óticas:

1. **Heurística baseada em CNN (domínio de pixels)** — um modelo convolucional
   (ConvNeXt) desenhado para espelhar a estrutura da árvore de particionamento do
   AV1, treinado sobre a luminância (Y). Serviu de **instrumento de diagnóstico e
   de referência de limite superior**; sua versão implantável **da era pixels (H7)**
   foi obtida por **destilação** de conhecimento (modelo substituto → modelo
   estudante MLP). **Ressalva (2026-07-19):** o pruner **de fato implantado** é o
   H9a do item 2, que **não** é destilado — é treinado diretamente (ver item 2 e
   `ARQUITETURA_pruner_implantado.md`). A destilação vale só para o estudante de
   pixels H7.
2. **Heurística baseada em NN pré-busca (H9a)** — um MLP por tamanho de bloco que
   consome **contexto taxa-distorção barato** (vizinhança de particionamento +
   quantização/posição, além dos pixels) e decide, **antes** da busca, se poda o
   nó. É a **contribuição central** da tese.
3. **Heurística baseada em NN pós-NONE (H9c)** — um MLP leve que atua **depois**
   de o codificador ter avaliado `PARTITION_NONE`, usando o custo RD real dessa
   avaliação para decidir se encerra a busca. É um **refinamento complementar**.
4. **Poda seletiva das partições estendidas (H9d)** — mesmo gancho e **mesmos 39
   atributos** do H9c, mas **ação distinta**: decide se vale avaliar AB e 4-way,
   candidatos que consomem 34,3% do custo de busca. É a **segunda solução positiva
   implantada** e a única cujo ganho **se soma** ao do H9a (+1,02 pp de TS). Ver §2.8
   para o porquê — ação disjunta, não mais informação.

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
- **Bloco D — SATD de Hadamard do bloco-fonte (36–37), barato:** descartado na
  Fase 3 (não agrega sinal sobre A+B+C). **Atenção (26/07):** este bloco foi
  *especificado* como o SATD do **resíduo de uma predição intra a partir dos
  vizinhos** (`PLANO_H9_contribuicao_tese.md:326`), mas *implementado* como o SATD
  do bloco-fonte, sem predição nem vizinho (`features.py:252`). São grandezas
  distintas: a implementada é uma estatística só da fonte, correlacionada com a
  variância e os gradientes que o bloco A já contém — daí o resultado nulo. A
  hipótese especificada nunca foi testada; ver
  `RESULTADOS_auditoria_dominio_pixels.md` e o bloco D' em
  `features_intrapred.py`.
- **Bloco E — custo RD real do `PARTITION_NONE` (36–38):** `log(none_rate)`,
  `log(none_dist)`, `log(none_rdcost)`. **Só disponível pós-NONE** → é o insumo
  exclusivo do H9c.

Subconjuntos: **H9a = A+B+C (36 atributos)**; **H9c = A+B+C+E (39 atributos)**.

> **Convenção de nomenclatura — "contexto RD" (fixada em 2026-07-26).** Ao longo da
> tese, "contexto de taxa-distorção" / "contexto RD" designa o **contexto de decisão
> barato que a busca RD nativa consulta** — blocos B e C (forma das partições vizinhas,
> força de quantização, posição) — e **não** grandezas de custo taxa-distorção. Estas
> últimas existem apenas no bloco E, exclusivo do H9c. Registre-se, para evitar a
> leitura equivocada que já ocorreu em três documentos: **o H9a não contém nenhuma
> grandeza de custo RD, e 24 dos seus 36 atributos são descritores de luma** — ou seja,
> o podador implantado é, majoritariamente, um modelo de pixels. Ver
> `RESULTADOS_auditoria_dominio_pixels.md`.

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

### 2.8 O espaço de projeto dos podadores — duas dimensões ortogonais

As três soluções implantadas (H9a, H9c, H9d) são frequentemente lidas como três níveis de
sofisticação de uma mesma ideia. **Não são.** Elas ocupam pontos distintos de um plano de
duas dimensões independentes: **quando** o podador age — e portanto que informação já foi
paga — e **o que** ele poda. Explicitar esse plano é o que torna previsíveis os resultados
de composição (§6, Conclusão 3).

**Dimensão 1 — o ponto de enganche no fluxo de controle** (`av1_rd_pick_partition`):

```
av1_rd_pick_partition(bsize, ...)
│
├─ av1_prune_partitions_before_search()   ◄── H9a      nada foi avaliado ainda
│
├─ none_partition_search()                    avalia PARTITION_NONE  ← custo pago aqui
│
├─ av1_prune_after_none()                 ◄── H9c e H9d   mesmo gancho
│
├─ split_partition_search()                   SPLIT quadrada
├─ rectangular_partition_search()             HORZ, VERT
└─ portões AB {4,5,6,7} e 4-way {8,9}     ◄── o alvo do H9d
```

**Dimensão 2 — a ação**, isto é, que candidatos deixam de ser avaliados.

| | **H9a** | **H9c** | **H9d** |
|---|---|---|---|
| **Quando age** | pré-busca | pós-NONE | pós-NONE |
| **O que vê** | A+B+C = **36** atributos | + bloco E (taxa/dist/rdcost **reais** do NONE) = **39** | + bloco E, os mesmos **39** |
| **O que decide** | cascata: comprometer NONE → só split quadrada → rect-off | **binária**: encerra a busca aqui | **seletiva**: vale avaliar AB e 4-way? |
| **Custo de informação** | zero — nenhuma avaliação RD paga | uma avaliação de NONE | uma avaliação de NONE |

**O H9c e o H9d são idênticos nas duas primeiras linhas** — mesmo gancho, mesmo vetor de 39
atributos (`RESULTADOS_H9d_etapa2_C.md §4`: `student_h9d_decide` reusa exatamente o
`student_node_features` + contexto pós-NONE do `student_h9c_decide`). Diferem **somente na
ação**. Quem é diferente *em espécie* é o H9a: é o único que decide antes de qualquer custo
de RD ter sido pago.

**Consequência económica de cada posição.**
- **H9a** tem o maior alcance e a maior cegueira: um acerto elimina a subárvore inteira,
  **incluindo a própria avaliação do NONE** — é o único que pode economizar aquele custo.
  Em troca, decide sem ver um único número de RD.
- **H9c** tem teto **estruturalmente limitado**: agindo depois do NONE, jamais economiza o
  NONE, só o que vem depois. Daí podar apenas ~4% de TS isolado (§5). Mais informação,
  menos alcance.
- **H9d** compartilha a limitação de alcance do H9c, mas mira um conjunto de candidatos que
  nenhum dos outros dois visava — e que custa **34,3% do tempo de busca** (`RESULTADOS_C1_custo_por_candidato.md`).

> **Nota de nomenclatura (fonte recorrente de confusão).** `H9a`/`H9b`/`H9c` nomeiam
> **conjuntos de atributos** — qual bloco de informação se acrescenta. `H9d` nomeia uma
> **ação** — qual conjunto de candidatos se poda. São dois esquemas de nomeação convivendo
> na mesma família, o que faz o "H9d" parecer o quarto degrau de uma escada de informação
> quando é, na verdade, uma coordenada do outro eixo. Registrado também em
> `RESULTADOS_modelagem_B3_horz_vert.md:218`.

O H9d, a **segunda solução positiva implantada**, é tratado em **§5-quater**; suas medições
completas estão em `RESULTADOS_H9d_predizibilidade.md`, `RESULTADOS_H9d_cota_superior.md`,
`RESULTADOS_H9d_etapa2_C.md`, `RESULTADOS_H9d_etapa3_encoder.md` e `RESULTADOS_H9d_CTC.md`.

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

**Destilação (era pixels, H7).** A versão implantável **de pixels** foi obtida por
**destilação de conhecimento** do substituto (ConvNeXt) para um **modelo estudante
MLP** que reusa o `av1_nn_predict`. A destilação **reduziu o custo computacional**
(de convoluções multi-resolução para um MLP denso por nó), mas também **reduziu os
ganhos** — o estudante de pixels herda uma fração do teto do substituto.

> **Ressalva (2026-07-19).** Isto descreve o H7. O pruner **implantado** (H9a) **não
> é destilado**: treina-se diretamente sobre os 36 atributos com CE de rótulo duro,
> sem o ConvNeXt no laço. A destilação foi um passo da era pixels, superado; o
> ConvNeXt permanece só como referência de teto (replay H8).

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

> **⚠ Refino (2026-07-20) — a afirmação de "saturação" NÃO está estabelecida; usar a
> hierarquia medida.** A ablação acima é 2 quadros / 1 sequência, e o "ml" testado
> era o estudante **destilado** (macro-F1 0,20), não o ConvNeXt de alta capacidade —
> a defesa "o modelo não era fraco" não se sustenta. O crivo do A5
> (`RESULTADOS_oraculo_regret.md`, 6 seqs, 792 mil nós) mostra o **mesmo** estudante
> de pixels **vencendo** a variância no custo ponderado por *regret* (variância
> 0,060 vs pixels24 0,015 a cost_red 30%). Logo "pixels saturam na variância / o
> sinal não está nos pixels" **é contestável** (a favor: piloto fino de 2 quadros;
> contra: crivo largo offline; e o GNN pixel-only fura o oráculo, `approachB §4`).
> A afirmação **defensável e mais forte** é a hierarquia medida: **variância <
> pixels24 < H9a** no crivo — o contexto RD agrega sinal *além* dos pixels, e os
> pixels *além* da variância. Ver `RESPOSTAS_contra_argumentos_banca.md` (CB-1/2/3).
> A confirmação da ordenação no encoder real (≥10 quadros, ≥2 seqs) é o item E5.

> **⚠⚠ Refino final (2026-07-26) — o ConvNeXt foi medido no crivo e NÃO é teto.**
> O refino acima notava que o "ml" da ablação era o estudante destilado, deixando
> em aberto se o ConvNeXt de alta capacidade se separaria da variância. Agora ele
> foi medido na mesma vara (`RESULTADOS_convnext_regret.md`), e o resultado
> **fecha a questão pelo lado oposto ao esperado**:
> - a hierarquia estende-se para **variância < convnext_regret < convnext_ce <
>   pixels24 < H9a** — o ConvNeXt de 28,1 M de parâmetros sobre pixels crus
>   **perde para o `pixels24`**, um MLP sobre 24 atributos manuais derivados da
>   *mesma* luma (0,0207 contra 0,0121 em `cost_red` 25%, ~1,7×);
> - treiná-lo com o alvo de **regret** (o objetivo correto, nunca antes aplicado a
>   pixels) **piorou-o** em toda a faixa, 1,06× a 3,80×;
> - e **capacidade não é a restrição**: dobrar `fusion_dim` muda a perda de
>   validação em 0,16%.
>
> **Consequência conceitual:** o ConvNeXt **não pode ser reportado como
> "referência de limite superior" do domínio de pixels.** Um modelo que é batido
> por outro com acesso estritamente menor à informação (24 atributos manuais
> extraídos da mesma luma, que ele poderia em princípio representar) não
> estabelece cota superior alguma — o seu desempenho é enunciado sobre o **nosso
> treino**, não sobre os pixels. O que a tese tem é uma **cota inferior**: o
> melhor desempenho observado no domínio de pixels é o do `pixels24`. O teto do
> domínio de pixels permanece **não medido**. O único teto genuíno do arcabouço é
> o **oráculo** (decisão RD-ótima, regret zero), que limita qualquer podador e não
> só os de pixels.

**Ressalvas (para o Capítulo de Resultados):** (i) alto custo computacional da
CNN; (ii) limitação informacional inerente às amostras Y; (iii) a destilação
troca ganho por custo; (iv) **nenhuma via de pixels foi implantada** — quatro
tentativas independentes (substituto por CE, substituto por *regret*, `pixels24`,
GNN estrutural) e nenhuma se aproxima do H9a, que usa contexto RD.

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
BD da nativa no mesmo cpu) → a CNN nativa é mais eficiente. **Sobre custo de
inferência (medido, ver §6 e `docs/RESULTADOS_microbench_pruner.md`):** como
algoritmo de decisão isolado, a **inferência do MLP é ~50× mais barata por
chamada** que a da CNN nativa (~486 ns vs ~24.700 ns) — o modelo aprendido leve
executa sua predição a fração do custo da rede convolucional de produção. A
extração de features (preprocessamento) e a frequência de invocação são de
integração, fora do escopo da comparação de algoritmo isolado.

> **Custo implantado (medido em 2026-07-19, `RESULTADOS_microbench_pruner.md` §6).**
> No codificador, o custo próprio do pruner — inferência **e** extração incluídas —
> é **≤0,32% do tempo de encode** (CNN 0,16–0,21%; H9a 0,26–0,32%). Não é alavanca
> em direção nenhuma: **nenhum resultado de BD×tempo muda**, e o speedup vem das
> **decisões de poda**, não da leveza da inferência. A razão "~50×/chamada"
> caracteriza o algoritmo isolado e **não** deve ser citada como vantagem do
> pruner — a extração não é custo intrínseco da solução (parte é leitura grátis de
> estado do codificador, parte é cópia otimizável), e a soma é imaterial de
> qualquer modo.

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

## 5-bis. Solução 4 — regressão de *regret* (resultado negativo)

Testou-se uma **reformulação**: em vez de classificar o rótulo, **regredir o custo
RD de podar** (*regret*) e podar por custo predito. Alvo reconstruído da árvore
comprometida dos pkls `dataset_h9` (sem re-extração); NN regressora (MLP por
tamanho, cabeça única) sobre as mesmas 36 features H9a. Detalhe completo em
`docs/RESULTADOS_solucao4.md`.

**Resultado — a hipótese é refutada.** No Gate 3 (val, oráculo, SPLIT-lost
casado), o regressor **não alcança a faixa de risco baixo** onde o classificador
opera:

| método (val) | redução de custo @0,5% SPLIT-lost | menor SPLIT-lost |
|---|--:|--:|
| regret naïve | 0,00% | 12,4% (degenerado por zero-inflação) |
| regret balanceado (Huber ponderado) | 0,00% | 11,4% |
| **H9a-classificador** (mesmas features) | **51,73%** | **0,01%** |
| variância | 4,46% | 0,01% |

Mesmo corrigida a zero-inflação (peso 8–46× nos nós de *regret* não-nulo), a
regressão **ranqueia a segurança de poda pior** que o classificador: há ~11% de
nós *true-SPLIT* mapeados a *regret* ≈ 0, indistinguíveis dos seguros.

**Leitura (valor metodológico).** A decisão de poda é fundamentalmente uma
**classificação** (seguro/inseguro); a **magnitude** do *regret*, dominada pelo
conteúdo, não é ranqueável a partir de features pré-busca baratas melhor que a
própria probabilidade da decisão. Isto **afia** a metodologia — dá base empírica a
*por que* a formulação de classificação (Solução 2/H9a) é a correta, em vez de
deixar "prever custo" como alternativa não testada. Não se pagou o Gate 5
(benchmark real): o oráculo, que **superestima** o mérito, já rejeita o método a
risco casado, então o benchmark só confirmaria pior.

---

## 5-ter. Approach B — decisão estruturada (GNN) do quadtree (resultado negativo)

Testou-se se a decisão **conjunta/estruturada** (GNN de *message-passing* sobre a
árvore do superbloco) extrai sinal **além** dos nós independentes (H9a) — a última
alavanca não testada. Ablação controlada `n_layers=0` (MLP) vs `n_layers≥1` (GNN),
mesmas features/dados. Detalhe em `docs/RESULTADOS_approachB.md`.

**Arco e veredito:** no **oráculo**, a estrutura fura o teto (GNN não-causal +28pp;
versão **deployable pixel-only** recupera ~93–100% e supera o H9a em **+20–25pp**).
Mas o **benchmark real** (replay pelo gancho H8, fiel às decisões) refuta: na
fronteira real (Jockey, cada modelo no seu melhor τ), o **H9a domina o GNN por ~2×**
em BD em todo o *sweep* de τ (GNN ~1,5% vs H9a ~0,75–0,94% a TS casado). **O oráculo
inverteu o ranking.** A fronteira do GNN é plana → é a **qualidade das decisões**,
não a calibração; e como o replay é fiel, **C faria as mesmas decisões** (não salva).

**Leitura (contribuição):** a acurácia por-nó e a métrica de custo do oráculo são
**maus proxies** do BD×tempo real — um modelo que **vence** o oráculo pode **perder**
no encoder real (alerta mais forte que "o oráculo superestima"). Confirma que o sinal
**satura no nível do H9a em BD×tempo real** e que nenhuma sofisticação (regressão de
custo §5-bis; estrutura conjunta) o supera. Reordenar candidatos/early-term não foi
perseguido (exige RD por-candidato não instrumentado + novo gancho C, sem gate
offline, contra heurísticas nativas dominantes).

---

## 5-quater. Solução 5 — H9d: poda seletiva das partições estendidas (AB / 4-way)

**O que a distingue — e por que não é "o próximo H9".** Pelo espaço de projeto de §2.8, o
H9d **não** é o quarto degrau de uma escada de informação: é a **outra coordenada**.
Compartilha com o H9c o gancho (`av1_prune_after_none`) e o **mesmo vetor de 39 atributos** —
`student_h9d_decide` reusa exatamente o `student_node_features` + contexto pós-NONE do
`student_h9c_decide`. O que muda é **a ação**: em vez de perguntar *"posso encerrar a busca
aqui?"*, pergunta *"vale avaliar as partições **estendidas** — AB {4,5,6,7} e 4-way {8,9}?"*.
NONE, rect e split ficam **intocados**.

**Por que esse alvo.** O diagnóstico C1/C3, sobre todo o teste congelado, mediu que AB+4-way
consomem **34,3% do custo de busca** (mínimo 28,9% — 3–4× acima do limiar de relevância de
10%). É um pool grande que **nem o H9a nem o H9c jamais visaram especificamente**.

**Etapa 1 — portão de predizibilidade (offline, 792 840 nós held-out).** A pergunta prévia:
*as features sabem dizer que o nó não vai escolher partição estendida?*

| nível | n | base EXT | ROC-AUC (36 feat.) | ROC-AUC (39 feat.) |
|---|--:|--:|--:|--:|
| 16 px | 572 213 | 9,9% | 0,906 | **0,919** |
| 32 px | 172 627 | 13,1% | 0,817 | **0,829** |
| 64 px | 48 000 | 3,0% | 0,864 | 0,865 |
| **agregado** | **792 840** | 10,2% | **0,890** | **0,902** |

Pontos de operação (39 atributos): a **50% de busca evitada**, apenas **1,1%** de vencedores
perdidos; a 10% de perdas, **69,7%** de busca evitada. **Veredito: GO** para as Etapas 2–3.

**Cota superior — quanto existe para ganhar.** Desligando AB/4-way por completo
(`AV1_EXT_PART_OFF=1`, mesmo binário, gate inerte por padrão), 3 seqs de teste, 5 quadros:
**+0,89% de BD por 1,431× de speedup**. E, o que importa, **marginalmente sobre o H9a P_ref**
— pois o H9d só age no resíduo que o H9a deixou passar — ainda restam **1,293× de speedup**
presos em AB/4-way (Jockey 1,367× · RaceNight 1,239× · RiverBank 1,290×). **O pool marginal
é grande; não é ruído.**

**Etapa 2 — integração em C (Gate C PASSOU).** Pesos do MLP de 39 entradas (2 saídas
NÃO-EXT/EXT, por nível 16/32/64) exportados para `partition_student_h9d_weights.h`, com
**round-trip verificado**: sobre 192 vetores aleatórios,
`|softmax(PyTorch) − softmax(layout C)| = 1,35e−07`. A decisão marca
`part_state->h9d_skip_ext`, consumido nos gates AB e 4-way, **reusando o mecanismo já
validado** do `AV1_EXT_PART_OFF`. Superfície de controle por env-var
(`AV1_STUDENT_H9D_ENABLE`, `AV1_STUDENT_H9D_TAU[_16/_32/_64]`), **off por padrão**; H9c e
H9d rodam independentes.

**Etapa 3 — no codificador: o τ global funciona, mas quebra numa sequência.** Sweep de τ
sobre o H9a P_ref (3 seqs de teste, 5 quadros, vs âncora cpu0) mostra a fronteira do H9d
**dentro** da curva do knob de τ do H9a. A 10 quadros, contra subir τ (`H9a A2`) ao mesmo
speedup: **RaceNight domina Pareto** (0,981%/1,771× contra 1,282%/1,757× — mais rápido *e*
−0,30 pp), **Jockey melhor** (−0,22 pp @1,775×), **RiverBank pior** (+0,12 pp). A correção
foi **τ por nível de bloco**:

| variante | Δ BD Jockey | Δ BD RaceNight | Δ BD RiverBank |
|---|--:|--:|--:|
| τ global 0,30 | −0,29 | −0,34 | **+0,12 (pior)** |
| **PL10** (θ por nível @wl≈10%) | −0,02 | **−0,30** | **+0,05 (≈empate)** |
| PL20 (@wl≈20%) | +0,03 | −0,12 | +0,04 |
| PLmix (agressivo-16) | −0,00 | −0,08 | +0,03 |

**Escolhido: PL10** (τ₁₆=0,091 · τ₃₂=0,103 · τ₆₄=0,014) — recupera o RiverBank sem abrir mão
do ganho nas outras duas.

**Validação universal (Fase 6, CTC Classe A1, 8 seqs, 15 quadros, cpu0).** Integridade
verificada primeiro: com o H9d desligado, o codificador reproduz `ml_balanced`
**byte-idêntico** (1 574 775 bytes, PSNR-Y 40,9720 dB).

| config | BD-Rate | TS% | speedup |
|---|--:|--:|--:|
| H9a equilibrado (`ml_balanced`, P_rect) | +0,568% | 17,72 | 1,223× |
| **H9a + H9d (PL10)** | **+0,586%** | **18,74** | **1,238×** |

**A pergunta que importa não é "H9d vs nativo", e sim: partindo do ponto implantado, o H9d
compra tempo mais barato do que simplesmente subir o τ do H9a?** Interpolando o segmento
P_rect→A3 (o knob de τ) no TS que o H9d atingiu:

| mecanismo | preço do tempo |
|---|--:|
| **H9d empilhado (PL10)** | **0,018 pp de BD por pp de TS** |
| knob de τ do H9a (P_rect→A3) | 0,063 pp/pp |

O H9d entrega **+1,02 pp de TS por +0,018 pp de BD** — cerca de **um terço** do que custaria
obter o mesmo tempo afrouxando o limiar do H9a, isto é, **~3,5× mais barato**. Vence o knob
de τ em **6 das 8** sequências, **2 delas por Pareto estrito** (FoodMarket2 0,63%→0,61% com
+1,8 pp de TS; Tango 1,15%→1,14% com +1,6 pp) — BD **cai** enquanto o tempo **melhora**.

> **Leitura (por que esta solução importa para a tese).** O H9d é a **segunda solução
> positiva implantada** e a prova empírica que corrige a Conclusão 3 (§6): ele soma **+1,02
> pp** sobre o H9a — **quatro vezes** o que o H9c somou (+0,26 pp) — usando **informação
> idêntica à do H9c**. Se a não-aditividade dos *levers* fosse um teto **informacional**,
> isso seria impossível. O que a explica é **sobreposição de ação**: H9a e H9c caçam ambos
> os "blocos fáceis"; o H9d caça um conjunto de candidatos **disjunto**. É o resultado que
> torna a Conclusão 3 prescritiva — *procure ações não disputadas, não mais informação*.

**Ressalvas.** (i) A fronteira do H9d na CTC tem **um único ponto** (PL10 × P_rect); as três
configurações restantes (PL20 × P_rect, PL10 × A3, PL20 × A3, ~9 h) são **confirmatórias** e
não foram rodadas — o H9d é um ponto onde o H9a é uma curva. (ii) Os ganhos de TS de
Neon1224 (+0,1 pp) e Crosswalk (+0,4 pp) estão **dentro da resolução temporal medida**
(σ ⇒ ~0,46 pp, `RESULTADOS_BLOCO7_E3_DEC_E2.md`); a média de +1,02 pp está a ~4,4σ e os
BD-rate são exatos, então o Pareto não é afetado. (iii) O τ por nível foi calibrado nas 3
seqs de teste UVG e aplicado sem re-ajuste à CTC — o resultado da CTC é, nesse sentido,
genuinamente held-out.

---

## 6. Análise integrada — fronteira Pareto e as três conclusões

**Fronteira Pareto global (BD × TS, todos os níveis cpu, média 3 seqs
Boxing/FoodMarket2/Tango).** Pontos não-dominados, do menor BD ao maior:
`H9c(cpu0) 0,21%/13,9%` → `H9c 0,23%/15,0%` → `native_cpu1 0,37%/28,6%` →
`H9c_cpu1 0,39%/29,6%` → `native_cpu2 0,41%/38,2% (TS/BD 94, pico)` →
`H9c_cpu2 0,44%/39,1%` → `H9a_bal_cpu2 1,13%/47,1%` → `native_cpu3 2,80%/66,4%` →
`H9c_cpu3 / H9a_cpu3 3,5–4,8%/70–78%`.

> **Nota (26/07).** Esta fronteira é de uma análise anterior, sobre **3 sequências**
> (Boxing/FoodMarket2/Tango); o **H9d não figura nela** por ter sido medido depois, sobre as
> **8 seqs da CTC**. Sua ausência aqui **não** significa dominância — ao contrário, na
> medição própria (§5-quater) ele é Pareto-não-dominado e vence o knob de τ em 6/8 seqs.
> Recompor a fronteira global com o H9d exigiria as 3 configurações confirmatórias ainda não
> rodadas (§5-quater, ressalva i).

**Conclusão 1 — ninguém DOMINA a CNN nativa.** Nenhum ponto ML é estritamente
melhor (mais TS a ≤ BD). O H9c empata a cpu1/2; a nativa mantém o pico de
eficiência (TS/BD 78–94).

**Conclusão 2 — o H9c é dono do extremo de baixo BD que a nativa não alcança.**
Os pontos de menor BD de toda a fronteira (0,21–0,23% BD, 14–15% TS) são do ML a
cpu0. A escada discreta da nativa pula de cpu0 (0% TS) para cpu1 (~30% TS),
deixando **todo o regime 0–30% TS descoberto** — o ML preenche continuamente. O
valor prático é **granularidade fina em baixo speedup**, não superar o pico.

**Conclusão 3 — *levers* que disputam a MESMA ação não se somam.** H9a
(pixels+contexto), H9c (rdcost pós-NONE) e a CNN nativa exploram o **mesmo sinal
correlacionado** — os "blocos fáceis". Prova direta: H9c sobre H9a = +0,26pp de
TS. É a mesma história de saturação que a Solução 1 estabeleceu no domínio de
pixels, agora confirmada no domínio RD.

> **⚠ Correção do enunciado (2026-07-26).** Esta conclusão foi originalmente redigida na
> forma geral — *"os levers não se somam"* — e nessa forma **é refutada pelo H9d**, que
> soma **+1,02 pp** de TS sobre o H9a (quatro vezes o que o H9c somou) usando **informação
> idêntica à do H9c**. A não-aditividade, portanto, **não é um limite informacional**: é
> **sobreposição de ação**.
>
> Pelo espaço de projeto de §2.8: H9a e H9c ocupam pontos diferentes da dimensão *quando*,
> mas a **mesma** posição na dimensão *o quê* — ambos perguntam "este bloco é fácil, posso
> parar?", e por isso caçam os mesmos blocos lisos. O E4 mediu a sobreposição diretamente:
> **64% do TS atribuído ao H9c era, na verdade, o H9a** rodando por baixo nos seus defaults
> (`RESULTADOS_BLOCO7_E1_E4.md §2`). O H9d escapa porque sua ação é **disjunta** — não
> pergunta "posso parar?", pergunta "vale avaliar AB e 4-way?", candidatos que custam 34,3%
> do tempo de busca e que nenhum dos outros dois visava.
>
> **Enunciado correto:** dois podadores se somam na medida em que seus **conjuntos de
> candidatos podados** são disjuntos, independentemente de terem ou não a mesma informação
> de entrada. É uma afirmação mais estreita, mas verdadeira — e **prescritiva**: indica que
> a via para novos ganhos é procurar ações ainda não disputadas, não mais informação.

**Argumento transversal — custo de inferência (MEDIDO).** O microbenchmark de
inferência isolada (`docs/RESULTADOS_microbench_pruner.md`, encode real cpu1)
mede o custo do **algoritmo de decisão** de cada pruner: a **inferência do MLP é
~50× mais barata por chamada** (~486 ns vs ~24.700 ns da CNN) — cerca de uma
ordem e meia de grandeza. As três MLPs (24/36/39 entradas) têm custo quase
idêntico (dominado pelas camadas ocultas `[64,32]`). **Escopo:** compara-se a
inferência (passagem direta) dos modelos; a extração de features do MLP
(preprocessamento, otimizável/cacheável) e a frequência de invocação (por nó vs
por superbloco) são de integração, deliberadamente fora do escopo do algoritmo
isolado. Nota de fidelidade: a passagem direta da CNN funde leitura de pixels e
convolução (não há inferência separável numa conv), então seus ~24.700 ns são a
predição completa pixels→decisão. **Custo implantado (medido em 2026-07-19):** no
codificador, o pruner inteiro — extração e inferência — é **≤0,32% do tempo de
encode**, então nada em BD×tempo se altera e o custo de inferência não é alavanca
em direção nenhuma; perde-se apenas o direito de alegar leveza de inferência como
vantagem. A razão "~50×/chamada" mede o algoritmo isolado, não o pruner implantado
(`RESULTADOS_microbench_pruner.md` §6). Este resultado complementa as Conclusões 1–2
(granularidade fina + paridade de qualidade).

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
4. H9d: o alvo AB/4-way (34,3% do custo), portão de predizibilidade, cota superior,
   τ por nível e a contribuição marginal na CTC — a solução que **se soma** ao H9a
   (§5-quater).
5. Análise integrada: fronteira Pareto, as três conclusões, custo computacional
   (§6).
6. Discussão: limite superior informacional **corrigido** (a não-aditividade é de
   *ação*, não de informação — §2.8); utilidade prática vs validação metodológica;
   o papel de nicho e de substituto-leve das soluções ML.

---

## 8. Tabelas-mestre (referência rápida, vs âncora libaom cpu-used=0)

Método de cálculo **único e consistente** (BD-Rate PSNR-Y Bjøntegaard; TS% de
speedup agregado; média das sequências). Ver §4–§6 para as tabelas por cenário.

- **Fase 5 (teste reservado, 3 seqs):** H9a P_rect ~0,46%/26,5%; P_ref ~0,60%/29,5%.
- **Fase 6 CTC (8 seqs) — praticante:** H9a bal 0,568%/19,3%; H9a aggr 1,403%/34,1%;
  nativo cpu1 0,449%/30,4%; cpu2 0,536%/40,4%; cpu3 2,722%/67,5%.
- **Fase 6 CTC — H9a + H9d (PL10):** 0,586%/18,7% (TS canônico; o par H9a bal na
  mesma definição é 17,7%). Marginal: **+1,02 pp de TS por +0,018 pp de BD**, ~3,5×
  mais barato que o knob de τ (§5-quater).
- **Swap 8 seqs — ML-vs-CNN nativa:** ver §4 (H9a) e §5 (H9c). Resumo: H9c ≈ nativa
  a cpu1/2; H9a poda mais a BD desproporcional.

> **⚠ Atenção ao copiar estes números — há DUAS definições de TS na tese.**
> `analyze_frontier.py:281` computa ambas: a **canônica** (média sobre CQ de
> `1 − t/t_âncora`, depois sobre sequências) e a usada nas linhas de Fase 6 acima
> (`1 − Σ_CQ t / Σ_CQ t_âncora`, ponderada pelo tempo, logo dominada por CQ 20). Elas
> divergem até **~3 pp** (`results/benchmark/fase6_analysis/ts_definitions.csv`): o nativo
> cpu1 é 30,4% numa e **32,6%** na outra. Os documentos mais recentes (H9d, Bloco 7) usam a
> **canônica**. Tabela única já padronizada na canônica: **`INVENTARIO_solucoes.md`**.
> Os BD-rate não são afetados.

---

## 9. Ameaças à validade e limitações

- **Oráculo superestima o tempo real (~5×):** os gates *offline* (Gate 2/3) usam
  redução de custo de oráculo; a decisão final é sempre tempo de parede
  (Gate 5/benchmark). As margens *relativas* é que valem *offline*.
- **Oráculo pode INVERTER a ordenação (ameaça mais forte, 2026-07-19):** não se
  trata só de superestimar a magnitude — o Approach B estabeleceu que a *ordem
  relativa* entre pruners no oráculo pode se inverter no encoder real (GNN ≫ H9a
  offline; H9a ≫ GNN no tempo de parede; `RESULTADOS_approachB.md:106,149`).
  Consequência metodológica: **rejeição no oráculo não implica rejeição no real**,
  logo as margens relativas offline são um indício, não uma prova de ordenação. É
  o motivo pelo qual todo veredito final passa pelo encoder (e pelo qual a decisão
  de pular o Gate 5 da Solução 4 foi reenquadrada como assimetria de custo, não
  implicação lógica — `RESULTADOS_solucao4.md`, correção D2).
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
| **Frontier-check combinado** (H9a-conservador + H9c, swap, Tango) | ✅ **concluído** (2026-07-17) — não fura a fronteira | `results/benchmark/fase6_swap_combo/` |
| Microbenchmark de inferência isolada do pruner | ✅ **concluído** (2026-07-17) — MLP ~50× mais barato/inferência que a CNN; **custo implantado medido em 2026-07-19: pruner inteiro (extração+inferência) ≤0,32% do encode**, sem efeito em BD×tempo, não é alavanca | `docs/RESULTADOS_microbench_pruner.md` §6 |
| **Solução 4 — regressão de *regret*** | ✅ **concluído** (2026-07-17) — **resultado negativo**: regressão ranqueia poda pior que o classificador (Gate 3, mesmas features); Gate 5 pulado por regra de gate | `docs/RESULTADOS_solucao4.md`, `results/models/regret{,_balanced}/` |

**Resultado do frontier-check combinado (Tango, vs âncora cpu0):** o combinado
H9a-conservador + H9c posiciona-se **entre** o H9c-swap e o H9a-swap — mais TS
que o H9c sozinho, mas com **eficiência decrescente** (TS/BD a cpu1: nativa 81,9 >
H9c 77,2 > comb(0,98) 67,9 > comb(0,95) 65,2 > H9a_bal 29,9). **A CNN nativa
permanece no topo da eficiência; o combinado não a domina em nenhum ponto.**
Confirma empiricamente a **Conclusão 3** (§6): empilhar levers correlacionados dá
poda absoluta maior a eficiência marginal pior, sempre dentro da fronteira da
nativa. A porta está fechada — nenhuma combinação de H9a/H9c/τ testada supera a
CNN nativa. Cobertura por sequência/config inventariada em
`docs/ANDAMENTO_tese.md §8.4`.

---

*Documento gerado a partir de `docs/ANDAMENTO_tese.md`, `RESULTADOS_fase5.md`,
`RESULTADOS_fase6.md`, `PLANO_H9_contribuicao_tese.md`, `PROTOCOLO_avaliacao.md` e
dos CSV em `results/benchmark/{fase6,fase6_swap,fase6_swap_h9c}/`.*
