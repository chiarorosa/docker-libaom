# H9d — predizibilidade offline de AB/4-way (Etapa 1: o seletivo vale?)

**Data:** 2026-07-21
**H9d, Etapa 1 (go/no-go, offline, barata).** Decide se um H9d **seletivo** (podar a busca
AB/4-way só onde não vencem) é viável, ANTES de qualquer código em C. Segue a cota superior
(`RESULTADOS_H9d_cota_superior.md`), que mostrou o H9d **blanket** Pareto-não-dominado mas cru.
A pergunta: as features do nó separam "AB/4-way vence" o suficiente para o seletivo bater o
blanket?
**Script:** `src/scripts/partition_model/h9d_predictability.py` · **Modelo:**
`results/models/h9d_predictability/students.pt`

---

## 1. Método

- **Rótulo binário EXT** por nó: `y = 1 se a partição ótima ∈ {HORZ_A, HORZ_B, VERT_A,
  VERT_B, HORZ_4, VERT_4}` (AB + 4-way), senão 0. É o alvo direto do H9d: "buscar o
  estendido vai valer aqui?".
- **Features:** as **36 do H9a** (`node_features_h9a`, pré-busca) — as mesmas já disponíveis
  no nó em C, garantindo deployabilidade. (Não usa contexto RD pós-NONE; ver §5.)
- **Modelo:** MLP por nível (16/32/64), 36→64→32→2, CE de rótulo duro, sem ponderação —
  mesma receita do H9a/B3, para comparabilidade.
- **Split held-out da tese:** treino nas 10 seqs de treino; avaliação nas **6 held-out**
  (3 validação + 3 teste: HoneyBee/FlowerPan/Lips + Jockey/RaceNight/RiverBank). 792.840 nós
  de decisão held-out.
- **Métrica central — a troca poda×custo.** Regra: "pule a busca AB/4-way se `p_ext < θ`".
  - `search_avoided(θ)` = fração de nós com `p_ext < θ` (busca economizada);
  - `winners_lost(θ)` = fração dos nós **EXT verdadeiros** com `p_ext < θ` (= 1 − recall;
    vencedores reais descartados → custo de BD).
  - Blanket = (avoided 100%, lost 100%); sempre-buscar = (0%, 0%). Um bom seletivo dá
    **alto avoided a baixo lost**.

## 2. Resultado — as features separam EXT (SEPARÁVEL)

| nível | n | base EXT | **ROC-AUC** | PR-AUC | wl 10% → avoided | avoided 50% → wl |
|---|--:|--:|--:|--:|--:|--:|
| 16px | 572.213 | 9,9% | **0,906** | 0,452 | 71,0% | 1,8% |
| 32px | 172.627 | 13,1% | 0,817 | 0,359 | 51,8% | 8,8% |
| 64px | 48.000 | 3,0% | 0,864 | 0,155 | 63,2% | 5,0% |
| **agregado** | **792.840** | 10,2% | **0,890** | 0,425 | **67,1%** | **2,6%** |

Troca poda×custo agregada (held-out):

| fixando | θ | resultado |
|---|--:|---|
| winners_lost ~5% | 0,038 | search_avoided **58,0%** |
| winners_lost ~10% | 0,085 | search_avoided **67,1%** |
| winners_lost ~20% | 0,164 | search_avoided **76,0%** |
| search_avoided ~50% | 0,019 | winners_lost **2,6%** |
| search_avoided ~70% | 0,108 | winners_lost **12,5%** |
| search_avoided ~90% | 0,361 | winners_lost 54,3% |

**Interpretação.** ROC-AUC agregado **0,890** (PR-AUC 0,425 sobre base de 10,2% — muito
acima do acaso 0,10). O seletivo pode **evitar ~67% das buscas AB/4-way perdendo só 10% dos
vencedores** — ou evitar 50% perdendo **2,6%**. O nível 16 é o mais separável (AUC 0,906),
o 32 o menos (0,817), mas todos são fortes.

**Contra o blanket (a versão medida na cota superior):** o blanket perde 100% dos vencedores
para evitar tudo; o seletivo, no mesmo território de busca-economizada, perde uma **pequena
fração**. Isto é uma melhoria de Pareto: recupera a maior parte do custo de BD que o blanket
pagava (0,8–2,2%), preservando quase todo o speedup marginal (~1,29×). Projeta um ponto de
operação **muito mais eficiente** que o blanket — a confirmar no encoder (Etapa 3).

## 2.1 Refinamento — 39 features (com contexto RD pós-NONE) elevam o AUC

O H9d decide **pós-NONE**, logo o `none_rate/dist/rdcost` (o bloco E, a 39ª–37ª features do
H9c) **está** disponível no ponto de enxerto. Reexecutando com `node_features_h9c` (39):

| métrica (held-out) | h9a (36 feats) | **h9c (39 feats)** |
|---|--:|--:|
| ROC-AUC agregado | 0,890 | **0,902** |
| ROC-AUC 16px / 32px / 64px | 0,906 / 0,817 / 0,864 | **0,919 / 0,829** / 0,865 |
| PR-AUC agregado | 0,425 | **0,445** |
| winners_lost 10% → search_avoided | 67,1% | **69,7%** |
| search_avoided 50% → winners_lost | 2,6% | **1,1%** |

Ganho **consistente** (16/32 sobem ~+0,012 AUC; 64 estável), como esperado — o custo/RD do
NONE ajuda a distinguir onde o estendido importa. **O modelo de 39 features é o que segue
para o C (Etapa 2)** — encaixe natural, ganho grátis. Reproduzir: acrescentar
`--feature-set h9c`.

## 3. Veredito — GO para as Etapas 2–3

**A Etapa 1 passa com folga.** As features do H9a carregam sinal forte para a decisão EXT
(AUC 0,89), suficiente para um H9d seletivo dominar o blanket. O H9d deixa de ser
"blanket-promissor" e passa a **candidato a contribuição fechada** — uma segunda solução
positiva, no molde do H9c (podador aprendido pós-NONE, aqui sobre o eixo estendido).

**Próximos passos:**
- **Etapa 2 (C, moderada):** cabeça "h9d" análoga ao `student_h9c_decide` — computa features
  no nó, roda `av1_nn_predict`, seta o gate de AB/4-way (`ab_bsize_thresh`/`part4`) por
  `p_ext < θ`, em vez da env global `AV1_EXT_PART_OFF`. Gate e template (H9c) já existem.
- **Etapa 3 (encoder):** confirmar que H9a+H9d_seletivo domina o blanket e a curva de τ
  (≥10 quadros, 3 seqs de teste). O encode é o árbitro final (A5).

## 4. O que isto significa para a espinha da tese

O C2 mostrou que o custo/risco vive nos blocos grandes; o C1 que AB/4-way são 34% do tempo;
a cota superior que empilhar a poda estendida pós-NONE é Pareto-não-dominado; e agora a
Etapa 1 que **esse pool é previsível** (AUC 0,89). A cadeia fecha: há um segundo eixo de
decisão (estendido) com custo real e sinal aprendível, explorável por um modelo leve — a
mesma tese do H9a, num eixo novo.

## 5. Limitações

- **`search_avoided` é sobre TODOS os nós**, não só os que alcançam o gate AB/4-way (nós
  NONE-ótimos, que o H9a já poda, entram na conta). Superestima um pouco a economia
  *implantável*; a separabilidade (AUC 0,89) é o sinal robusto. A Etapa 3 (encoder) mede a
  economia real.
- **`winners_lost` não é ponderado por regret.** Perder um vencedor EXT marginal custa pouco
  BD; perder um grande custa muito. Uma versão ponderada por ganho-RD (à la crivo A5) tende a
  ser **ainda melhor** (protege os grandes, descarta os marginais) — refinamento para a
  Etapa 3.
- **~~Só as 36 features H9a~~ — RESOLVIDO (§2.1):** a variante de 39 features (com
  `none_rdcost` pós-NONE) foi medida e eleva o AUC (0,890→0,902); é a que segue para o C.
- **Offline ≠ encoder.** Classificação offline forte é sinal verde, não prova de BD×tempo. O
  árbitro é o encode (Etapa 3).

## 6. Reprodução

```bash
/workspace/build/venv-ml/bin/python \
  src/scripts/partition_model/h9d_predictability.py \
  --dataset-dir /workspace/results/dataset_h9
```
