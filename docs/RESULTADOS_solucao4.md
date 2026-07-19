# Solução 4 — NN regressora de *regret* RD para poda intra: resultado (negativo)

> **Resultado metodológico da tese.** Data: 2026-07-17. Branch `ml-partition-dev`.
> Design: `docs/superpowers/specs/2026-07-17-solucao4-regret-regression-design.md`.
> Plano: `docs/superpowers/plans/2026-07-17-solucao4-regret-regression.md`.
> Artefatos: `results/models/regret/` (naïve, `students.pt`; gates `gate0.csv`,
> `gate3_{regret,h9a,var}.csv`), `results/models/regret_balanced/` (balanceado) e
> `results/models/regret/gate3b_regret_r0*.csv`.

---

## 1. Hipótese e enquadramento

As Soluções 1–3 aprenderam o particionamento como **classificação** do rótulo de
partição (NONE/SPLIT/REST) e podaram por **confiança** (limiar τ sobre a
softmax). A Solução 4 testou uma **reformulação**: em vez de classificar o
rótulo, **regredir o custo taxa-distorção de podar** — o *regret* — e podar por
**custo predito** (`regret_pred < τ`). A intuição: a fronteira BD×TS é governada
por *quanto de RD se perde ao podar*, uma grandeza contínua; um preditor desse
custo poderia podar exatamente onde é barato e, assim, acessar um regime de risco
que o classificador não alcança.

**A hipótese não se sustentou.** Um regressor competente de *regret* **ranqueia a
segurança de poda pior** que o classificador H9a sobre **as mesmas features**.
Este documento registra o resultado como achado — negativo, mas informativo: ele
dá base empírica a *por que* a formulação de classificação é a correta para a
decisão de poda.

---

## 2. Método (o que foi construído, sem re-extração)

- **Alvo `regret` reconstruído da árvore comprometida.** Para cada superbloco,
  reconstrói-se o quadtree RD-ótimo a partir de `mi_row/mi_col` + rótulos
  (`regret.py`). Para cada nó de decisão *n*:
  `regret_rel(n) = (none_rdcost(n) − RD_subárvore(n)) / RD_subárvore(n)`, com
  `RD_subárvore` somando `none_rdcost` das folhas comprometidas (exato para
  folhas NONE; limite superior censurado para folhas retangulares). Tudo dos
  pkls `dataset_h9` existentes — **nenhuma re-extração**.
- **NN regressora implantável.** MLP por tamanho de bloco (topologia [64,32] do
  H9a), **cabeça de regressão única**, entrada = as 36 features A+B+C do H9a
  (fonte única, paridade C↔Python já validada), alvo `log1p(regret_rel)`, perda
  Huber (`train_regret.py`).
- **Política de poda.** `P(NONE) = exp(−regret_pred/r0)`, reusando a cascata
  NONE-commit e a métrica de *SPLIT-lost casado* do simulador de oráculo
  (`simulate_pruning.py`, modo `--regret-bundle`), de modo a ser **diretamente
  comparável** ao classificador H9a e à baseline de variância.
- **Cadeia de gates** (mesma partição congelada; validação HoneyBee/FlowerPan/Lips).

---

## 3. Gate 0 — viabilidade do sinal: PASSOU (com ressalva registrada)

Sobre os 10 pkls de treino, por tamanho (`gate0.csv`):

| dim | n | fração exata | desvio do regret | fração de zeros |
|--:|--:|--:|--:|--:|
| 64 | 15.434 | 51,3% | 0,143 | 59,5% |
| 32 | 59.518 | 66,3% | 0,066 | 83,6% |
| 16 | 189.296 | 83,8% | 0,019 | 98,1% |

Os critérios pré-registrados (fração exata ≥ 40%, desvio ≥ 1e-3) passaram. **A
ressalva, registrada já no Gate 0:** o *regret* é **fortemente zero-inflado** e a
inflação cresce com a profundidade (dim16 = 98% de zeros, desvio 0,019). O sinal
treinável concentra-se em 64/32px. Esta ressalva antecipou o modo de falha.

---

## 4. Resultado naïve — colapso no preditor trivial

Treino com Huber sobre `log1p(regret)` (Huber final 0,00184/0,00079/0,00027 por
tamanho — trivialmente baixo, dominado pela massa de zeros).

**Gate 3 (val, lever NONE-commit) — redução de custo a SPLIT-lost casado
{0,5/1/2%}:**

| método | @0,5% | @1% | @2% | menor SPLIT-lost |
|---|--:|--:|--:|--:|
| **regret (naïve)** | **0,00%** | **0,00%** | **0,00%** | **12,4%** |
| H9a-classificador | 51,73% | 51,73% | 62,00% | 0,01% |
| variância | 4,46% | 4,46% | 6,35% | 0,01% |

O regressor naïve é **degenerado**: comete `none_wrong` de 28–49% mesmo no ponto
mais conservador — prevê *regret* ≈ 0 para quase todos os nós, inclusive os que
precisam de SPLIT, e **não pode ser regulado para risco baixo** (piso de SPLIT-lost
12,4%). É a materialização exata da zero-inflação sinalizada no Gate 0.

---

## 5. Correção — perda Huber ponderada (anti-zero-inflação)

Introduziu-se um **Huber ponderado** (`train_regret.py`, flag `--balance`) que
rebalanceia, por tamanho, os nós de *regret* não-nulo com peso
`w = min(n_zero / n_nonzero, cap)` — atacando a raiz do colapso. Pesos e frações
de zero (população exact-only) resultantes:

| dim | fração de zeros | peso não-nulo | Huber |
|--:|--:|--:|--:|
| 64 | 0,89 | 8,21 | 0,00361 |
| 32 | 0,95 | 17,76 | 0,00436 |
| 16 | 0,98 | 45,56 | 0,00196 |

O modelo balanceado **de fato aprende a cauda** (Huber sobe; a curva de operação
muda). A backward-compatibility foi verificada (sem `--balance`, a perda é
idêntica à Huber média).

---

## 6. Resultado balanceado — o balanceamento NÃO resgata o ranqueamento

**Gate 3 (val) do regret balanceado**, varrendo `r0 ∈ {0,05; 0,1; 0,2}`
(`gate3b_regret_r0*.csv`):

| método | @0,5% | @1% | @2% | menor SPLIT-lost |
|---|--:|--:|--:|--:|
| regret balanceado, r0=0,05 | 0,00% | 0,00% | 0,00% | 11,4% |
| regret balanceado, r0=0,10 | 0,00% | 0,00% | 0,00% | 11,4% |
| regret balanceado, r0=0,20 | 0,00% | 0,00% | 0,00% | 11,5% |
| **H9a-classificador** (mesmas 36 features) | **51,73%** | **51,73%** | **62,00%** | **0,01%** |
| variância | 4,46% | 4,46% | 6,35% | 0,01% |

O piso de SPLIT-lost permanece ~11% em **todos** os `r0` → **0% de redução de
custo a risco casado**. Existe uma massa de ~11% de nós *true-SPLIT* que o
regressor mapeia para *regret* ≈ 0, **indistinguíveis dos seguros**. O `r0` só
reescala `P(NONE)`; não corrige um **ranqueamento** ruim.

---

## 7. Conclusão — a decisão de poda é uma classificação, não uma regressão de custo

A hipótese central da Solução 4 **é refutada**: a regressão do *regret* — mesmo
competente (balanceada, aprendendo a cauda) — **ranqueia a segurança de poda pior**
que o classificador de decisão, sobre **entradas idênticas** (as 36 features H9a).

A leitura defensável do porquê:

> A poda é, no fundo, uma **classificação** (seguro/inseguro). A **magnitude** do
> *regret* é dominada pelo conteúdo, com uma cauda de valores difícil de prever a
> partir de features pré-busca baratas; ao otimizar essa magnitude, o regressor
> **desperdiça capacidade** em valores que não sobrevivem à predição, enquanto o
> classificador concentra-se exatamente na **fronteira de decisão** que o
> ranqueamento de poda exige. A probabilidade da decisão (`P(NONE)`) é um sinal de
> prune-safety **estritamente melhor** que o custo predito.

Isto **afia a metodologia da tese**: em vez de deixar "prever custo em vez de
rótulo" como um caminho intuitivo não testado, fica o resultado empírico de que a
formulação de **classificação** (Solução 2, H9a) é a **correta** para a decisão de
particionamento — a alternativa foi construída, treinada com competência, e perdeu.

**Por que não se pagou o Gate 5.** O oráculo *offline* **superestima** o mérito de
um pruner (histórico ~5×, e o precedente H9c: venceu o oráculo e foi refutado no
benchmark real). Como o oráculo já **rejeita** a regressão de *regret* a risco
casado, um benchmark real (integração C + horas de encode) só confirmaria um
resultado pior — não se justifica pagá-lo. As Tasks 6–9 do plano (export C, Gate
5, Gate 6) foram, portanto, **puladas por regra de parada de gate**.

> **⚠ Correção (2026-07-19) — a justificativa acima tem dois defeitos.**
>
> 1. **O precedente H9c invocado está contaminado.** "Venceu o oráculo e foi
>    refutado no benchmark real" descreve o piloto Jockey de 2 quadros, que a
>    §8.1 de `ANDAMENTO_tese.md` prova ser H9a+H9c empilhados, não H9c isolado. O
>    swap limpo (`RESULTADOS_fase6_swap_h9c.md`) mostra o oposto: o H9c **empata**
>    com a CNN nativa na grade CTC e a **supera** em alta taxa. O precedente,
>    portanto, não sustenta "oráculo rejeita ⇒ real rejeita".
> 2. **A implicação lógica não vale.** O argumento pressupõe que rejeição
>    *offline* implica rejeição real. Mas o próprio Approach B estabeleceu que a
>    ordenação offline↔real **pode inverter** (`RESULTADOS_approachB.md:106,149`).
>    Se inverte, **rejeição no oráculo não implica rejeição no encoder** — a regra
>    de parada perde o fundamento lógico.
>
> **Enquadramento correto (não muda a decisão, muda a razão):** pular o Gate 5 foi
> uma decisão sob **assimetria de custo experimental** — o custo de integrar em C
> e rodar horas de encode contra o valor esperado da informação, dado que o sinal
> offline era fraco. É uma escolha de alocação de esforço, defensável como tal, e
> **não** uma implicação de que o real confirmaria a rejeição. A regressão de
> *regret* falhou no oráculo por zero-inflação (§ acima); esse é o motivo
> substantivo de não priorizá-la, independente do argumento do oráculo.

---

## 8. Ameaças à validade

- **Política via `P(NONE)=exp(−regret/r0)`:** o mapeamento é monotônico no *regret*
  predito, logo **preserva o ranqueamento** do modelo; a métrica de SPLIT-lost
  casado é invariante ao mapeamento desde que a varredura cubra a faixa — o que se
  garantiu com três `r0` e um grid amplo de τ. O piso de SPLIT-lost é robusto a
  `r0` → é o ranqueamento, não a escala.
- **Alvo censurado (folhas retangulares):** o treino primário usa `exact_only`;
  a censura foi quantificada no Gate 0 e não altera a conclusão (o piso persiste
  onde o sinal exato é mais rico, 64/32px).
- **Uma única arquitetura de regressão:** testou-se o MLP por tamanho (naïve e
  balanceado). Um modelo *hurdle* (classifica podável × arriscado, depois regride
  a magnitude só nos arriscados) é a correção estatística "certa" para a
  zero-inflação — mas seu estágio-1 é, essencialmente, o **próprio classificador
  H9a**, então seu teto é ~H9a; fica registrado como possível trabalho futuro de
  retorno duvidoso.

---

## 9. Relação com as conclusões da tese

O achado **reforça** a espinha dorsal metodológica: as Soluções 1–3
estabeleceram que o sinal de poda satura (pixels → variância; contexto RD → H9a,
que não fura a fronteira nativa; levers correlacionados não somam). A Solução 4
acrescenta um flanco: **não é só o *sinal* que satura — a *formulação* de
classificação é a que melhor extrai o sinal disponível**; reescrever o alvo como
regressão de custo não abre um regime novo, apenas ranqueia pior. É uma
delimitação honesta do espaço de soluções, não um ganho — e como tal entra no
Capítulo de Resultados como resultado negativo de valor metodológico.
