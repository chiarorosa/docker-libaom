# Solução 4 — NN regressora de *regret* RD para poda intra (design)

> **Documento de design (spec).** Data: 2026-07-17. Branch `ml-partition-dev`.
> Fonte de contexto: `docs/SINTESE_resultados_metodologia.md`,
> `docs/PLANO_H9_contribuicao_tese.md`, `docs/PROTOCOLO_avaliacao.md`,
> `docs/ANDAMENTO_tese.md`. Este spec define **estrutura, interfaces e gates** —
> hiperparâmetros e limiares finais são propostas iniciais, calibradas em cada
> gate (conforme a diretriz de arquitetura da tese).

---

## 1. Descrição breve (a contribuição)

**Rede neural regressora do custo RD de poda (*regret*)** — um MLP por tamanho de
bloco que, em vez de *classificar* o rótulo de partição (NONE/SPLIT/REST),
**prediz um valor contínuo**: quanto de RD se perde ao podar o nó. A poda intra
passa a ser decidida por **custo predito pela NN** (`predicted_regret < τ_regret`),
não por **confiança** de um classificador.

Quatro atributos que a definem: (i) é **NN**; (ii) é **regressão**, não
classificação; (iii) o alvo é **custo RD / *regret***; (iv) aplica-se à **decisão
de particionamento intra** (`av1_rd_pick_partition`).

---

## 2. Motivação e enquadramento na tese

A investigação estabeleceu (Soluções 1–3) que:

1. O domínio de **pixels satura na variância** (Solução 1, resultado negativo
   decisivo).
2. O **contexto RD barato (H9a)** é mais discriminativo que a variância sob
   política casada (Solução 2, contribuição central), mas **não fura a fronteira
   BD×TS da CNN nativa**.
3. Empilhar *levers* (H9a + H9c + τ) **não soma** — **Conclusão 3**, tratada como
   *teto informacional*.

**A hipótese não examinada:** todos os modelos até aqui compartilham três escolhas
nunca variadas — (a) o **alvo é classificação 3-vias**; (b) cada nó decide de forma
**independente**; (c) o ponto de operação é uma **grade τ global fixa**. A Solução 4
ataca a escolha (a) — e, de graça, a (c).

**A tese central da Solução 4:** *a classificação do rótulo nunca foi o objetivo
implantável*. O que governa a fronteira BD×TS é **quanto de RD se perde ao podar**,
uma grandeza **contínua**. Ao **regredir o *regret*** (destilando uma grandeza
pós-NONE em *features* pré-busca) e podar por **custo predito** — em vez de
confiança — a poda passa a acontecer exatamente onde é barata em RD. Esse é o
mecanismo que pode alcançar pontos de **BD mais baixo ao mesmo TS**.

**Barra de contribuição (definida pelo usuário):**
- **Obrigatório (ângulo defensável):** a reformulação regressão-de-*regret* tem
  valor metodológico próprio, independente de furar a fronteira.
- **Desejável (bônus):** furar a fronteira da CNN nativa.
- **Não-negociável:** paradigma **NN**, foco em predição de particionamento intra.

---

## 3. Construção do alvo `regret` (a partir dos dados existentes)

### 3.1 Insumos disponíveis (sem re-extração)

Cada registro `PartitionSample` (4144 B) em `results/dataset_h9/*.pkl` carrega, por
nó quadrado avaliado no *ground truth* (`cpu-used=0`):

- `partition` — rótulo RD-ótimo (PARTITION_TYPE 0..9);
- `none_rdcost`, `none_dist`, `none_rate` — **custo RD real do `PARTITION_NONE`**
  (bloco E; sentinela = 0 quando NONE não foi avaliado);
- `mi_row`, `mi_col`, `bsize`, `block_dim` — posição e tamanho (reconstrução da
  árvore);
- luma + contexto B/C (as 36 *features* A+B+C do H9a).

### 3.2 Reconstrução da árvore e cálculo do *regret*

Agrupar registros por superbloco (origem 64×64: `sb_row = mi_row // 16`,
`sb_col = mi_col // 16`; identidade de quadro via ordem de `sample_id` /
`--frame-offsets`). Seguir a **árvore comprometida** a partir da raiz 64×64 pelos
rótulos: `NONE` → folha; `SPLIT` → quatro filhos quadrados; `HORZ/VERT/AB/4-way`
→ **folha retangular**.

Para cada nó interno *n* em que NONE foi avaliado (`none_rdcost` não-sentinela):

```
regret_rel(n) = ( none_rdcost(n) − RD_subtree(n) ) / RD_subtree(n)

RD_subtree(n) = Σ leaf_RD  sobre as folhas comprometidas sob n
  leaf_RD =
     none_rdcost(folha)      se folha NONE          → EXATO
     recursão                se folha SPLIT
     none_rdcost(folha)      se folha retangular    → LIMITE SUPERIOR (censurado)
```

Propriedades: `regret_rel` é **adimensional** (comparável entre QP/tamanho), ≈0
para nós NONE verdadeiros, e cresce onde podar custa RD.

### 3.3 A censura como nuance metodológica (não um bloqueador)

O `regret` é **exato** no sub-reticulado NONE ↔ SPLIT-quadrado (o *lever* principal
do H9a). Para folhas **retangulares** só temos `none_rdcost` (≥ RD real da
retangular) → `RD_subtree` é superestimado → `regret_rel` **subestimado** = **limite
inferior** (dado **censurado**). Tratamento: cabeça de perda **censurada/unilateral**
para esses nós (§5), e uma variante de treino restrita a árvores sem folha
retangular como controle. A quantificação da fração censurada é parte da defesa de
validade.

### 3.4 Não há vazamento

O alvo usa `none_rdcost` (pós-NONE), mas o **modelo implantado consome só as 36
*features* A+B+C pré-busca**. `none_rdcost` é ingrediente do **alvo de treino**
(offline), nunca entrada do modelo. É **destilação de uma grandeza RD pós-hoc em
*features* pré-busca** — enquadramento limpo e defensável.

---

## 4. Gate 0 — viabilidade PRIMEIRO (barato, dados existentes, sem treino)

Antes de qualquer custo, sobre os pkls de **treino** (10 seqs):

| Verificação | Critério de PASSAGEM (inicial) |
|---|---|
| a. `none_rdcost` populado (não-sentinela) | fração alta dos nós internos |
| b. Árvore reconstruível (todos os nós comprometidos presentes; agrupamento por quadro correto) | reconstrução consistente em amostra de superblocos |
| c. `regret_rel` com variância real e ordenação esperada (NONE≈0 < SPLIT) | separação estatística clara |
| d. Fração de folhas censuradas (retangulares) | quantificada e limitada |

**Se falhar** (dado esparso/censura dominante), a ideia é abortada aqui, a custo
quase zero — coerente com a disciplina de gates da tese. Script: `gate0_regret.py`.

---

## 5. Modelo (NN implantável)

- **Arquitetura:** reusa o **MLP por tamanho de bloco** do H9a (entrada = 36
  *features* A+B+C, já com paridade C↔Python validada), trocando a **cabeça softmax
  3-vias por uma saída de regressão única**.
- **Alvo:** `log1p(regret)` ou `regret_rel` (decisão calibrada no Gate 2).
- **Perda:** Huber; **componente censurada/unilateral** para folhas retangulares
  (alvo é limite inferior).
- **Inferência:** reusa `av1_nn_predict` (nenhum código de inferência novo em C);
  custo ≈ o MLP atual (~486 ns/chamada, ver microbench).
- **Fonte única das *features*:** `features.node_features_h9a` (o C espelha).

---

## 6. Política de poda (o ponto de operação torna-se principiado)

Poda quando `predicted_regret < τ_regret`. Como `τ_regret` está em **unidades de
custo RD**, o mesmo τ **tolera a mesma perda de RD em todo lugar** — um **orçamento
adaptativo ao conteúdo** (absorve a ideia de "τ adaptativo" sem mecanismo extra),
ao contrário de um limiar de **confiança**. Varrer `τ_regret` traça a curva BD×TS.

Ação, por compatibilidade com o gancho existente: `predicted_regret < τ_regret` →
compromete NONE (desabilita splits), reusando o mesmo caminho seguro do H9a. Uma
extensão (opcional) mapeia faixas de `regret` para as ações rect-off/split-only.

---

## 7. Gates (espelham a metodologia congelada; mesma partição de sequências)

Partição **congelada** (`PROTOCOLO_avaliacao.md`): treino 10 · validação
HoneyBee/FlowerPan/Lips · **teste reservado Jockey/RaceNight/RiverBank**.

| Gate | Conteúdo | Critério |
|---|---|---|
| **0** | Viabilidade do sinal (§4) | `regret` computável, com variância, censura limitada |
| **2** | Oráculo *offline* (treino) | `regret`-score ≥ H9a-classificador **e** > variância, redução de custo a SPLIT-lost casado |
| **3** | Oráculo (validação) | idem em HoneyBee/FlowerPan/Lips |
| **4** | Integração C | paridade *features* (já limpa) + cabeça de saída única; **no-op byte-idêntico** (md5) |
| **5** | Benchmark real (teste reservado) | 3 pilares (BD-Rate PSNR-Y / TS% / speedup) + **swap de score sob política casada** `{regret, H9a-classificador, variância, aleatório}` |
| **6** *(opcional, bônus)* | CTC (Classe A1, 8 seqs) | fura a fronteira da CNN nativa? |

**Resultado decisivo (Gate 5):** o **swap de score sob política casada** (mesma
cascata NONE-commit, muda-se só a fonte do escore). **Se o `regret` alcança BD
menor ao mesmo speedup que o classificador H9a → a reformulação está validada** (o
ângulo defensável obrigatório), independentemente da CNN nativa. Se empatar,
**confirma que o teto é independente do objetivo** — também um achado real.

Reuso: `simulate_pruning.py` ganha um modo *regret-score* (Gate 2/3);
`ablation_attrib.py` / `analyze_ablation.py` ganham a fonte `regret` (Gate 5).

---

## 8. Integração C e garantias de fidelidade

- Todo código novo sob `#if PARTITION_ML_STUDENT`, em
  `av1/encoder/partition_strategy.c`.
- Nova variável de ambiente `AV1_STUDENT_TAU_REGRET` (ou `AV1_REGRET_TAU`), lida e
  cacheada; **default (sem set) = no-op byte-idêntico** à âncora (md5 verificado),
  como os *toggles* existentes.
- Cabeça de saída única exportada por `export_weights` para o header
  `partition_student_weights.h` (ou header dedicado, para não sobrescrever o H9a
  implantado — decisão de build no plano).
- Paridade C↔Python das 36 *features* já é bit-a-bit; validar a saída da regressão
  (`check_feature_parity.py` estendido para o escalar de regressão).
- Builds: `libaom_ml_check` (paridade), `libaom_noop`/`libaom_perf_anchor` (no-op),
  `libaom_perf` (benchmark) — conforme `GUIA_builds.md`.

---

## 9. Entregáveis

**Scripts** (`src/scripts/partition_model/`, ver memória `scripts-under-src`):
- `gate0_regret.py` — reconstrução de árvore + viabilidade (Gate 0);
- `build_regret_targets.py` — gera os alvos `regret` por nó a partir dos pkls;
- `train_regret.py` — treino da NN regressora (reusa infra de `distill`/`train_student_h9`);
- modo *regret-score* em `simulate_pruning.py`; fonte `regret` em `ablation_attrib.py`;
- export de pesos (cabeça única).

**C:** caminho *regret-score* em `partition_strategy.c` + env `τ_regret`.

**Docs:** `docs/RESULTADOS_solucao4.md` (resultados); atualização de
`SINTESE_resultados_metodologia.md` (nova seção "Solução 4") e `ANDAMENTO_tese.md`.

---

## 10. Riscos e limitações (honestidade)

- **Bônus de fronteira não garantido:** a saturação da Conclusão 3 pode reaparecer
  (o `regret` e o classificador podem ranquear os mesmos "blocos fáceis"). **O swap
  de score no Gate 5 dá resultado defensável de qualquer forma** (vitória ou
  confirmação do teto).
- **Censura retangular** (§3.3): enviesa o alvo; mitigada por perda unilateral +
  variante de controle. A quantificação é parte da validade.
- **Oráculo superestima o tempo (~5×):** Gates 2/3 valem pelas margens **relativas**;
  o árbitro é sempre o benchmark real (Gate 5).
- **Reconstrução de árvore depende de logging completo:** Gate 0 verifica que todos
  os nós comprometidos estão presentes antes de prosseguir.

---

## 11. Trabalho futuro / teste final da tese — Approach B (NN estruturada)

Documentado aqui como **experimento de fechamento**, **não** implementado nesta
fase: uma **NN estruturada em árvore** (recursiva/GNN) que decide o quadtree do
superbloco **conjuntamente** sobre o contexto RD (B/C) — possivelmente prevendo o
`regret` de forma **estruturada** (fusão A+B). É o **teste direto** de se o "teto"
da Conclusão 3 é informacional ou artefato de nós **independentes**. Fica como a
direção final da tese, a ser aberta após os resultados da Solução 4.

---

## 12. Sequência de execução (visão de plano)

1. **Gate 0** (viabilidade) → decisão go/no-go.
2. `build_regret_targets.py` (alvos) → `train_regret.py` (NN) → **Gate 2/3** (oráculo).
3. Integração C + **Gate 4** (paridade/no-op).
4. **Gate 5** (benchmark teste reservado + swap de score) — resultado defensável.
5. **Gate 6** (opcional CTC) — bônus de fronteira.
6. Documentação (`RESULTADOS_solucao4.md`, síntese, andamento).

O detalhamento tarefa-a-tarefa vai para o plano de implementação (skill
`writing-plans`), após revisão deste spec.
