# Plano de contribuição da tese — Poda de particionamento AV1 com contexto de taxa-distorção (H9)

**Versão:** 2026-07-09. Sucede e integra `PLANO_hipoteses_experimentos.md`,
`PREH7_analise_alavancas.md` e `METODOLOGIA_pipeline_ML.md`.

---

## 0. Tese em uma frase

> Demonstra-se que a decisão de particionamento do AV1 em modo All-Intra é
> **saturada, no domínio de pixels, por uma estatística trivial (variância do
> bloco)** — resultado estabelecido por ablação rigorosa — e que a incorporação
> de **contexto de taxa-distorção barato** (vizinhança de particionamento,
> posição e um proxy de resíduo intra) é **necessária e suficiente** para superar
> esse teto, produzindo uma heurística de poda aprendida cujo ganho de tempo é
> **atribuível ao modelo** (supera os baselines de variância e aleatório e
> complementa a heurística nativa do libaom), com compromisso taxa BD/speedup
> mensurável em sequências 4K held-out.

Isto converte o resultado negativo da ablação (a contribuição de ML não se
justificava no domínio de pixels) em uma contribuição positiva e defensável: um
**diagnóstico rigoroso do que a decisão precisa** seguido de uma **solução que
entrega o ganho pela razão certa**.

---

## 1. Base de evidências já estabelecida (o que sustenta este plano)

| Descoberta | Evidência | Consequência para o plano |
|---|---|---|
| Bug da luma em branco corrigido | `round(luma×255)`=fonte, exato; F1 0,12→0,20 | O pipeline e o dado estão íntegros; nada a refazer no dado bruto |
| Pixels saturam na variância | Ablação de atribuição: variância domina o ML em todo speedup casado | O teto é **informacional**; mais capacidade de modelo no domínio de pixels não ajuda |
| O modelo aprende sinal real | ML domina o baseline aleatório | Há o que aprender; o problema é a **entrada**, não o aprendizado |
| Espaço de ações já explorado | Pré-H7: rect-off + NONE-commit + τ por nível | A alavanca restante não é a política, é a **feature** |
| libaom nativo usa contexto RD | `partition_strategy.c:627–636` (vizinhança), `:741–743` (none_rdc) | Features **comprovadas**; baseline SOTA a bater/complementar |

**Por que H9 e não continuar no domínio de pixels:** a variância é uma das 24
entradas do estudante e ainda assim o domina. Um modelo bem ajustado não perde
para a própria feature — logo o ganho remanescente **não está nos pixels**. A
decisão de particionamento é, por construção, uma decisão de taxa-distorção; a
única entrada que pode elevar o teto é a que a própria decisão usa.

---

## 2. Hipótese científica central (H9, refinada)

> **H9.** Existe um conjunto de atributos de contexto de taxa-distorção
> **computável antes (ou no início) da busca RD completa, a custo desprezível
> frente ao que a poda economiza**, tal que um modelo leve treinado sobre ele
> supera o baseline de variância no compromisso taxa BD × speedup, em sequências
> held-out.

Sub-hipóteses, testadas em ordem de custo crescente (falha rápida):

- **H9a (grátis):** o **contexto de vizinhança** de particionamento (tamanhos e
  tipos de partição dos blocos acima/esquerda, disponibilidade de bordas) mais a
  **posição/qindex** já superam a variância. Custo de inferência: zero adicional
  (dados já residentes). É a aposta principal — barata e comprovadamente
  RD-informativa no libaom.
- **H9b (barato):** um **proxy de resíduo intra** (predição DC/SMOOTH rápida +
  SATD/variância do resíduo) adiciona sinal sobre H9a. Custo: uma predição barata
  por nó — a validar contra o que economiza.
- **H9c (referência de teto):** o **rdcost real do PARTITION_NONE** (pós-NONE,
  como o `ml_early_term_after_none` nativo) é o teto de contexto RD; estabelece
  quanto H9a/H9b deixam na mesa. Não é a solução implantada (paga o NONE), mas é
  o teto honesto — o análogo RD do que o H8/ConvNeXt foi para os pixels.

**Critério de refutação:** se H9a **e** H9b, no gate de simulação oráculo, não
superarem a variância por margem além do ruído, H9 é refutada no domínio pré-NONE
e a tese reporta isso honestamente (a contribuição recua para o diagnóstico + a
caracterização do teto via H9c). Este é o compromisso de honestidade da tese.

---

## 3. Projeto dos atributos (ancorado em features nativas comprovadas)

Vetor de entrada do modelo H9 (superconjunto do atual, para inclusão limpa do
baseline de variância):

**Bloco A — pixels (já validado, inclui o baseline):** os 24 atributos atuais
(variância, gradientes, perfis linha/coluna, contexto hierárquico pai/irmãos,
posição). Mantidos para que "ML ⊇ variância" e a comparação seja de superconjunto.

**Bloco B — vizinhança de particionamento (H9a, grátis):** replicando
`partition_strategy.c:627–636`:
- `has_above`, `has_left` (e `has_aboveleft`);
- `mi_size_wide_log2`/`mi_size_high_log2` dos blocos acima e esquerda;
- tipo/gran de partição dos vizinhos (quando disponível no contexto);
- profundidade do bloco atual na árvore (log2 do tamanho) e distância às bordas
  do quadro.

**Bloco C — quantização/posição (H9a, grátis):** `log(dc_q²)`, qindex normalizado,
posição normalizada (row/col) no quadro.

**Bloco D — proxy de resíduo intra (H9b, barato):** variância/SATD do resíduo de
uma predição intra barata (DC e/ou SMOOTH) sobre o bloco. Proxy de "quão caro
será codificar este bloco inteiro" — o sinal que a variância de pixels aproxima
grosseiramente e que a predição refina.

**Bloco E — rdcost do NONE (H9c, só teto):** `log(none_rate)`, `log(none_dist)`,
`log(none_rdcost)`. Registrado para o estudo de teto; não entra no modelo
implantado pré-NONE.

Cada bloco é uma **ablação de feature** independente (§6), então a tese atribui o
ganho ao bloco específico, não a um vetor monolítico.

---

## 4. Dados: enriquecimento e partição rigorosa

**Sequências (16 disponíveis em `src/samples/`):** Beauty, Bosphorus, CityAlley,
FlowerFocus, FlowerKids, FlowerPan, HoneyBee, Jockey, Lips, RaceNight,
ReadySetGo, RiverBank, ShakeNDry, SunBath, Twilight, YachtRide.

**Partição por sequência (sem vazamento):**
- **Treino:** 10 sequências;
- **Validação (seleção de modelo/limiar):** 3 sequências;
- **Teste (números da tese, nunca vistos):** 3 sequências — fixadas a priori e
  jamais usadas em qualquer decisão (Jockey mantido no teste por continuidade).

**Cobertura:** 4 QPs (cq 20/32/43/55) × ≥5 quadros por sequência (amostragem
temporal via `--skip`), cpu-used=0 (ground truth por busca RD completa, como em
`partition-dataset-pipeline`). Mais quadros no conjunto de **teste** (≥10) para
taxa BD estável — endereça a ressalva de ruído de 2 quadros da ablação.

**Re-instrumentação:** estende `PartitionSample` (guardada por
`LOG_PARTITION_DATA`) com os blocos B–E. Custo: campos adicionais por registro; a
luma permanece. O log do NONE (bloco E) exige mover/duplicar a escrita para
depois da avaliação do PARTITION_NONE dentro de `av1_rd_pick_partition` — ponto
já localizado (`none_rdc` existe no fluxo, cf. `:730`).

---

## 5. Execução em fases, com portões de falha-rápida

Cada fase tem um **gate quantitativo**; não se avança sem passar. Isto é o que
torna o plano "inatacável": o custo caro (integração em C, benchmark) só é pago
depois que o sinal foi provado offline.

### Fase 0 — Congelar o baseline e o protocolo (½ dia)
- Fixar as 3 sequências de **teste** e o protocolo de medição (quadros, QPs,
  binários) **antes** de ver qualquer resultado. Registrar em documento assinado
  por commit. Impede *cherry-picking* — objeção clássica de arguidor.
- Deliverable: `docs/PROTOCOLO_avaliacao.md` (congelado).

### Fase 1 — Re-instrumentação e re-extração (1–2 dias de máquina)
- Estender a instrumentação (blocos B–E); validar paridade de features novas
  (harness `check_feature_parity.py` estendido) e integridade (variância não
  nula — `assert_real_luma` + novas asserções para B–E).
- Re-extrair o dataset ampliado (16 seqs). Retomável (memória
  `partition-dataset-pipeline`).
- **Gate 1:** integridade — todos os campos com distribuição sã; paridade
  C↔Python bit-a-bit nas features determinísticas.

### Fase 2 — Gate offline de sinal (o teste barato e decisivo) (1 dia)
- Treinar MLPs por tamanho de bloco sobre subconjuntos de features:
  (i) variância-só [baseline]; (ii) pixels-24 [atual]; (iii) +B+C [H9a];
  (iv) +D [H9b]; (v) +E [H9c, teto].
- Rodar a **simulação oráculo com métrica de custo** (`simulate_pruning.py`) na
  validação, comparando cada conjunto **em risco casado** contra a variância.
- **Gate 2 (DECISIVO):** H9a e/ou H9b superam a variância na simulação por margem
  clara. **Se falhar aqui, para-se** — nenhuma integração em C é feita, e a tese
  reporta o diagnóstico + teto (H9c). Se passar, o restante é engenharia de
  baixo risco (o sinal já está provado).

### Fase 3 — Substituto + destilação sobre features RD (2–3 dias)
- Re-treinar o substituto (ConvNeXt com ramo lateral de features RD, ou um
  modelo tabular mais forte se o convolucional não agregar — a decidir pelo dado)
  e re-destilar o estudante sobre o conjunto vencedor da Fase 2.
- **Gate 3:** macro-F1/SPLIT-recall e simulação oráculo do estudante ≥ variância
  na validação.

### Fase 4 — Integração em C e verificação (1–2 dias)
- Sincronizar as features RD em C (extração de vizinhança/qindex/posição é
  barata e direta; proxy de resíduo reusa predição intra existente).
- Verificações: **paridade C↔Python** das probabilidades; **no-op bit-a-bit**
  (flag desligada = baseline); decodificação válida; `test_libaom`.
- **Gate 4:** paridade verde e no-op idêntico.

### Fase 5 — Benchmark de tese e ablação de atribuição (2 dias de máquina)
- No conjunto de **teste** (held-out, ≥10 quadros): curva taxa BD × speedup do
  estudante H9.
- **Ablação de atribuição** (a defesa central): mesma política, trocando a fonte
  do escore — variância, aleatório, pixels-24, H9. Em **speedup casado**
  (`analyze_ablation.py`), o H9 deve **dominar a variância**.
- **Comparação SOTA:** contra o `intra_cnn_based_part_prune` nativo do libaom
  (ligado por speed feature) — mostrar que H9 é competitivo/complementar e mais
  barato na inferência.
- **Gate 5 (SUCESSO DA TESE):** H9 domina a variância em taxa BD a speedup casado,
  por margem além do ruído, em ≥2 das 3 sequências de teste.

---

## 6. Avaliação e atribuição (o que torna o número inatacável)

**Escada de ablação** (isola a fonte de cada ganho):
1. aleatório (piso);
2. variância (o teto de pixels que precisamos superar);
3. pixels-24 (nosso ML antigo — sabemos que ≈ variância);
4. H9a (+vizinhança/posição) — grátis;
5. H9b (+proxy de resíduo) — barato;
6. H9c (+none_rdcost) — teto.

**Métricas:** taxa BD (Bjøntegaard, PSNR-Y externo, sem depender de build com
estatísticas internas) e speedup de parede, ambos em **speedup casado por
interpolação**; risco RD (SPLIT-perdido, rect-perdido) na simulação.

**Controles anti-objeção:**
- Sequências de teste congeladas a priori (Fase 0) → sem *cherry-picking*.
- Comparação de **superconjunto** (H9 ⊇ variância) → o ganho é atribuível às
  features novas, não a re-tuning.
- Comparação contra o **SOTA nativo** → o resultado situa-se na literatura.
- ≥10 quadros e ≥3 seqs held-out → robustez estatística.
- Contabilidade de custo honesta: o custo de inferência das features RD é medido
  e subtraído; se H9b não se pagar, reporta-se H9a (grátis) como a contribuição.

---

## 7. Registro de riscos e mitigações

| Risco | Prob. | Mitigação |
|---|---|---|
| H9a/H9b não superam a variância | Média | **Gate 2** para antes de qualquer C; tese recua para diagnóstico+teto (ainda publicável) |
| Proxy de resíduo não se paga em tempo | Média | H9a é grátis e independente; proxy é opcional |
| Ganho só aparece com none_rdcost (H9c) | Média | H9c é o teto; se for o único que ganha, a contribuição vira "poda pós-NONE aprendida multi-nível" vs `ml_early_term_after_none` — ainda novel se multi-nível/destilada |
| Ruído de taxa BD | Baixa | ≥10 quadros no teste; 3 seqs; medição repetida |
| Vazamento treino/teste | Baixa | Partição por sequência congelada na Fase 0 |
| Regressão no libaom | Baixa | Tudo guardado por flag; no-op bit-a-bit verificado |

---

## 8. Sobre partir de um "fresh start" no libaom

**Não é necessário e seria contraproducente.** O `src/aom` atual é um superconjunto
do baseline v3.10.0: toda a nossa lógica está sob guardas de compilação
(`LOG_PARTITION_DATA=0`, `PARTITION_ML_STUDENT=0` por padrão) e a compilação padrão
já foi verificada **byte-a-byte idêntica** ao `aom_baseline` (md5 do bitstream).
Clonar do zero descartaria a infraestrutura validada (instrumentação, integração,
harness de paridade, ablação) sem ganho. A re-instrumentação do H9 é **aditiva** e
guardada. Mantém-se `aom_baseline/` intocado como controle cego, como hoje.

---

## 9. Entregáveis e critério de sucesso da tese

**Entregáveis:**
1. Dataset ampliado (16 seqs) com contexto RD, particionado treino/val/teste.
2. Modelo H9 (substituto + estudante destilado) e pesos em C.
3. Curva taxa BD × speedup no teste held-out + tabela de ablação de atribuição.
4. Comparação contra variância, aleatório, pixels-24 e SOTA nativo.
5. Documentos: protocolo congelado, metodologia atualizada, resultados.

**Sucesso (o que a tese afirma):** uma poda de particionamento aprendida que, em
4K All-Intra held-out, entrega **speedup S% a taxa BD B%** com **S/B
estritamente melhor que o baseline de variância a speedup casado** — isto é, um
ganho de tempo **atribuível ao aprendizado de máquina com contexto de
taxa-distorção**, e não a uma heurística trivial. Caso o Gate 2 refute H9 no
domínio pré-NONE, a tese entrega a **caracterização rigorosa do teto informacional
do particionamento** (um resultado científico negativo, porém forte e original),
com o estudo de teto H9c delimitando o que seria possível.

---

## Apêndice A — Projeto detalhado de features (especificação para a Fase 1)

Sequências de teste congeladas: **Jockey, RaceNight, RiverBank**.

Legenda de custo: **livre** (dado já residente, custo de inferência ~0) ·
**barato** (uma operação leve por nó, a pagar-se no benchmark) · **teto**
(disponível só pós-NONE; não entra no modelo implantado, só no estudo de teto).
Ponto de disponibilidade: **pré** (no gancho atual, antes de qualquer busca RD) ·
**pós-NONE** (após avaliar PARTITION_NONE).

### Bloco A — Pixels (24 atuais, mantidos; inclui o baseline de variância)
Já implementados e com paridade C↔Python verificada. Servem para que o modelo
H9 seja **superconjunto** do baseline de variância (comparação limpa). Referência:
`features.py`, `student_node_features`. Nenhuma mudança.

| idx | nome | conteúdo | custo | ponto |
|--:|---|---|---|---|
| 0–17 | bloco-próprio | variância, quadrantes, gradientes h/v, perfis linha/coluna, bordas, DC, qindex | livre | pré |
| 18–23 | contexto hierárquico | var pai, contraste pai/irmãos, posição no superbloco | livre | pré |

### Bloco B — Contexto de vizinhança de particionamento (H9a) — **livre**
Espelha `partition_strategy.c:627–636` (features nativas comprovadas). Fonte:
`xd->above_mbmi`, `xd->left_mbmi` (vizinhos já codificados em ordem raster →
causal e disponível na inferência).

| idx | nome | fórmula / fonte | just. |
|--:|---|---|---|
| 24 | has_above | `!!xd->above_mbmi` | disponibilidade de borda superior |
| 25 | has_left | `!!xd->left_mbmi` | idem esquerda |
| 26 | above_w_log2 | `mi_size_wide_log2[above_bsize]` | largura do vizinho de cima |
| 27 | above_h_log2 | `mi_size_high_log2[above_bsize]` | altura do vizinho de cima |
| 28 | left_w_log2 | `mi_size_wide_log2[left_bsize]` | largura do vizinho da esquerda |
| 29 | left_h_log2 | `mi_size_high_log2[left_bsize]` | altura do vizinho da esquerda |
| 30 | neigh_finer | `(min(above,left area) < área atual)` | vizinhos mais finos ⇒ este tende a dividir |
| 31 | neigh_aniso | `sign(above/left mais largo que alto)` | pista para HORZ vs VERT (rect) |

**Justificativa:** particionamento é espacialmente correlacionado; se os vizinhos
são finos, este bloco tende a dividir — informação **ausente dos pixels** e de
custo zero. É a aposta principal do H9.

### Bloco C — Quantização e posição (H9a) — **livre**
| idx | nome | fórmula / fonte | just. |
|--:|---|---|---|
| 32 | log_dc_q2 | `log1p(dc_q²/256)` (como `:624`) | força de quantização molda a decisão RD |
| 33 | pos_row_norm | `mi_row / mi_rows` | correlação vertical de conteúdo (céu/chão) |
| 34 | pos_col_norm | `mi_col / mi_cols` | correlação horizontal |
| 35 | depth_log2 | `mi_size_wide_log2[bsize]` | profundidade na árvore (prior por nível) |

**Justificativa:** o mesmo bloco decide diferente sob QP diferente; a posição
captura estrutura global barata que a janela local do bloco não vê.

### Bloco D — Proxy de resíduo intra (H9b) — **barato**
| idx | nome | fórmula / fonte | just. |
|--:|---|---|---|
| 36 | satd_pred | SATD (Hadamard) do resíduo de uma predição intra barata a partir dos vizinhos reconstruídos (PAETH ou melhor-de-{DC,PAETH,2 direcionais} por SAD) | proxy de **taxa** de codificar o bloco inteiro |
| 37 | satd_gain | `SATD(fonte) − SATD(resíduo)`, normalizado | quão **predizível** é o bloco a partir dos vizinhos |

**Detalhe crítico (evita repetir a variância):** um resíduo de predição
**constante** (DC) tem a mesma variância do bloco — não agrega. Por isso o proxy
usa **SATD** (energia no domínio da transformada = proxy de taxa) sobre uma
predição **não-constante** (direcional/PAETH). Isto mede predizibilidade, que a
variância não captura. Custo: uma predição + Hadamard por nó — reusa
`aom_satd`/predição intra existentes; a pagar-se no benchmark (se não pagar,
recua-se para H9a, que é grátis).

### Bloco E — rdcost do PARTITION_NONE (H9c) — **teto**, pós-NONE
| idx | nome | fórmula / fonte | just. |
|--:|---|---|---|
| 38 | log_none_rate | `log1p(none_rdc->rate)` | taxa real do NONE |
| 39 | log_none_dist | `log1p(none_rdc->dist)` | distorção real do NONE |
| 40 | log_none_rdcost | `log1p(none_rdc->rdcost)` | custo RD real do NONE |

**Justificativa e papel:** é o sinal RD mais forte (o que `ml_early_term_after_none`
usa, `:741–743`). **Não entra no modelo implantado** (exigiria pagar o NONE em
todo nó); serve como **estudo de teto** — mede quanto H9a/H9b deixam na mesa, o
análogo RD do que o ConvNeXt/H8 foi para os pixels. Registrado no dataset via
escrita do sample **após** a avaliação do PARTITION_NONE em `av1_rd_pick_partition`.

### Resumo dos conjuntos de ablação (Fase 2)
| conjunto | blocos | custo | papel |
|---|---|---|---|
| variância | A[0] só | livre | baseline a superar |
| pixels-24 | A | livre | ML antigo (≈ variância) |
| **H9a** | A+B+C | **livre** | **aposta principal** |
| **H9b** | A+B+C+D | barato | + predizibilidade |
| H9c | A+B+C+D+E | teto | limite superior de contexto RD |

O modelo **implantado** será o melhor entre H9a e H9b que passe o Gate 2 e se
pague no benchmark; H9c fica como teto reportado.
