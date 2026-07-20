# Approach B — decisão estruturada (GNN) do particionamento intra: resultado

> **Resultado metodológico da tese.** Data: 2026-07-18. Branch `ml-partition-dev`.
> Design: `docs/superpowers/specs/2026-07-17-approachB-gnn-estrutural-design.md`.
> Plano: `docs/superpowers/plans/2026-07-17-approachB-gnn-estrutural.md`.
> Artefatos: `results/models/gnn_*/` (bundles + gates de oráculo),
> `results/benchmark/gnn_frontier/frontier_Jockey.csv`,
> `results/benchmark/gnn_replay/gnn_replay_Jockey.csv`.

---

## 1. Pergunta

A **Conclusão 3** da tese (levers correlacionados → teto informacional) foi
estabelecida empilhando *levers* escalares e com modelos que decidem **cada nó de
forma independente** (MLP por tamanho, H9a). Ficou em aberto: o teto é
**informacional**, ou apenas **artefato de decisões independentes**? A Approach B
testa isso com um preditor **estruturado** que decide o quadtree do superbloco
**conjuntamente** — uma GNN de *message-passing* (PyTorch Geometric: SAGE/GAT/GIN)
sobre a árvore, com a **ablação controlada** `n_layers=0` (MLP independente) vs
`n_layers≥1` (com estrutura), tudo o mais idêntico (features, dados, split, treino).

Métrica dupla, na validação congelada (HoneyBee/FlowerPan/Lips): **macro-F1 /
SPLIT-recall por nó** e **redução de custo no oráculo a SPLIT-lost casado** (a
métrica-padrão da tese, comparável a H9a/variância/regret).

---

## 2. Gate 1-B (não-causal): a estrutura fura o teto do oráculo

GNN (`L=2`, SAGE) vs MLP (`L=0`), mesmas 36 features H9a:

| métrica (val) | L0 (MLP) | L2 (GNN) |
|---|--:|--:|
| macro-F1 @64/32/16 | 0,53 / 0,66 / 0,53 | **0,66 / 0,80 / 0,67** |
| SPLIT-recall @64/32/16 | 0,93 / 0,83 / 0,03 | **1,00 / 0,92 / 0,24** |
| oráculo cost_red @0,5/1/2% SPLIT-lost | 42 / 52 / 52 | **70,5 / 73,5 / 77,8** |

A estrutura conjunta extrai **+28pp** no oráculo sobre os nós independentes → a
forma forte da Conclusão 3 (independência era o limite) **não se sustenta** no
oráculo. **Porém** este é um **limite superior não-causal**: o GNN agrega features
de vizinhos **incluindo filhos e irmãos futuros**, que o codificador não tem em
tempo de decisão top-down.

---

## 3. Ablação causal: o ganho evapora sob restrição causal

Arestas **causais** (nó recebe só de já-decididos: pai + irmãos anteriores; sem
filhos/futuros), mesmo modelo/treino:

| oráculo cost_red | L0 | L2 não-causal | L2 causal |
|---|--:|--:|--:|
| @0,5% | 42,2 | 70,5 | 46,5 |
| @1% | 52,0 | 73,5 | 51,1 |
| @2% | 52,0 | 77,8 | 57,2 |

Sob restrição causal, o ganho quase **desaparece** (empata o MLP no ponto de 1%).
Conclusão *provisória* (revista na §4): o teto seria de **causalidade**, não de
independência — o sinal estrutural útil estaria no *futuro* da árvore.

---

## 4. GNN pixel-only deployable: recupera o ganho no oráculo

**Insight corretivo (a ablação §3 foi estrita demais):** as features de um nó são
derivadas de **pixels**, e os pixels do superbloco inteiro **existem antes** da
decisão de particionamento. Logo a agregação *bottom-up* (filho→pai) de features de
pixel **é deployable** — como um **pré-passe por superbloco**, o mesmo padrão de
invocação da CNN nativa (`intra_cnn_based_part_prune`). O não-causal era a
*decisão* do filho (que a GNN nunca usou), não os pixels dele.

Testado com **features pixel-only** (bloco A pixels + C quant/pos = 28; **dropa o
bloco B**, a única entrada dependente de decisão) e arestas completas:

| oráculo cost_red | L0-pixel (MLP) | **L2-pixel (GNN deployable)** | não-causal | H9a |
|---|--:|--:|--:|--:|
| @0,5% | 31,8 | **65,8** | 70,5 | 42 |
| @1% | 37,4 | **70,6** | 73,5 | 52 |
| @2% | 37,4 | **77,8** | 77,8 | 52 |

O GNN pixel-only deployable recupera **~93–100%** do limite superior não-causal e
supera o MLP H9a em **+20–25pp** no oráculo. Atribuição limpa à estrutura: L2-pixel
65,8 vs L0-pixel 31,8 (mesmas 28 features) = **+34pp**. **No oráculo, parecia o
salvamento.**

---

## 5. O árbitro real: benchmark de replay — o ganho NÃO sobrevive

Mede-se o BD-rate × tempo **reais** reinjetando as probs do GNN no encoder pelo
gancho H8 (`AV1_STUDENT_PROBS_FILE`), sem GNN em C. O replay é **fiel às decisões**
(em C, mesmas probs + mesma política → mesmo BD e mesma economia de busca; C só
**soma** custo de inferência → TS real ≤ replay).

**Fronteira real (Jockey, 5 frames, cada modelo no seu MELHOR τ):**

| τ | GNN BD% / TS% | H9a BD% / TS% |
|---|--:|--:|
| 0,99 | 1,50 / 27,6 | **0,88 / 25,1** |
| 0,95 | 1,53 / 29,0 | **0,75 / 27,8** |
| 0,90 | 1,59 / 30,4 | **0,86 / 30,3** |
| 0,80 | 1,56 / 33,3 | **0,94 / 34,1** |

**O H9a domina o GNN por ~2× em BD ao longo de toda a varredura de τ.** O oráculo
**inverteu o ranking** (offline GNN ≫ H9a; real H9a ≫ GNN). A fronteira do GNN é
**plana** (1,50–1,59% em todos os τ): suas probs comprometem um conjunto fixo de
podas "confiantes" que são **caras em RD**; mexer no τ quase não muda o BD → é a
**qualidade das decisões**, não a calibração, o limite. Como o replay é fiel, uma
implantação em C faria as **mesmas** decisões → **a mesma fronteira** (com TS
levemente pior pela inferência). C **não** salva, e não chega perto do nativo
(~0,45%@32,6%).

---

## 6. Raiz do fenômeno e conclusão

**Por que o oráculo (e a acurácia por-nó) mentiram:** a GNN otimiza **acurácia de
classificação por-nó** (CE) e vence no oráculo — mas isso **não** se traduz no
BD×tempo real. **Acurácia por-nó e a métrica de custo do oráculo são maus proxies
do BD×tempo real de um pruner.** É por isso que uma GNN claramente superior offline
fica ~2× pior no encoder real.

> **Correção (2026-07-20).** A versão original desta seção atribuía a derrota real
> do GNN a "poucas podas erradas **caras em RD**". A medição do A5
> (`RESULTADOS_oraculo_regret.md §5`) **não sustenta** essa causa: as podas NONE do
> GNN são baratas por contagem (`split_lost` 0,25%) **e** por *regret* ponderado
> (`reg_frac`≈0, o menor de todos). Logo a falha real do GNN **não está na ação
> NONE-commit**. A causa fica como pergunta aberta (vazamento de vizinhança;
> descasamento cpu0↔cpu1; outra ação da política), não como raiz provada. A
> conclusão de que o oráculo é mau proxy **permanece** — só a explicação do
> mecanismo foi suavizada.

**Conclusão da Approach B.** Um modelo estruturado de **alta capacidade**, dado o
**melhor tiro** (GNN expressiva de biblioteca; versão deployable pixel-only;
fronteira no próprio τ; benchmark real), **ainda perde ~2× para o MLP simples H9a**
em BD×tempo real. Combinada com as Soluções 1–4, a Approach B fecha a investigação
de forma robusta:

> O sinal de particionamento intra **satura no nível do H9a em BD×tempo real**, e
> **nenhuma sofisticação testada** — regressão de custo (Solução 4) ou decisão
> estruturada conjunta (Approach B) — o supera. O teto **não** era artefato de
> independência (a estrutura extrai mais offline); é que **esse ganho offline não se
> traduz em valor real**, e o SOTA nativo permanece à frente.

**Direção não perseguida (decisão registrada):** reordenar candidatos +
early-termination exigiria **RD por-candidato** (logging declinado pelo usuário) +
re-extração + novo gancho em C, **sem gate offline possível**, competindo com as
heurísticas de early-term nativas que já dominam — custo máximo e EV baixo dada a
evidência acumulada. Fica como trabalho futuro condicional, fora do escopo.

---

## 7. Contribuição metodológica (por que este resultado negativo vale)

1. **Refuta a forma forte da Conclusão 3** e a substitui por uma afirmação mais
   precisa: a estrutura conjunta *extrai* mais sinal offline (por-nó e oráculo), mas
   esse sinal é **não-realizável** em BD×tempo real.
2. **Demonstra concretamente que o oráculo pode inverter o ranking** de pruners —
   um alerta metodológico mais forte que "o oráculo superestima a magnitude"
   (Fase 5/H9c): aqui um modelo que **vence** o oráculo **perde** no real.
3. **Decompõe honestamente** de onde vinha cada ganho aparente: não-causal
   (filhos/futuros) → recuperável como pixel deployable no oráculo → mas **refutado**
   no real; e localiza a causa (acurácia ≠ custo RD).
4. Estabelece que a **classificação simples por-nó (H9a)** é a formulação certa, e
   que o **benchmark real é o único árbitro** — reforçando a espinha da tese.

Artefatos e reprodutibilidade: pipeline em `src/scripts/partition_model/`
(`graph_data.py`, `gnn_model.py`, `train_gnn.py`, `gate1_gnn.py`, `gnn_replay.py`)
e `src/scripts/benchmark/{gnn_replay_bench,gnn_frontier_bench}.py`; bundles e CSVs
de gate/fronteira em `results/models/gnn_*` e `results/benchmark/gnn_*`.
