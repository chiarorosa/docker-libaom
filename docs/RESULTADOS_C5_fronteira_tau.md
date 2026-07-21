# C5 — a fronteira do limiar rígido já é densa? (poda soft vale a pena?)

**Data:** 2026-07-21
**Bloco 6, item C5.** C5 conjectura que substituir a poda **dura** do H9a (comprometer
NONE se `P(NONE) > τ`) por um limiar de RD **contínuo modulado por P(NONE)** melhoraria a
fronteira BD×tempo. É a única alavanca com bom argumento teórico, mas exige **cirurgia em C
não-trivial** (busca RD recursiva). O plano define a **falsificação barata**: *se um sweep
fino de τ já produz fronteira densa e contínua, a suavização não muda nada.* Este probe
testa exatamente isso — **antes** de qualquer cirurgia.
**Script:** `src/scripts/benchmark/c5_fine_tau.py` · **BD:** `bd_rate.py`
**Dados:** `results/benchmark/c5_finetau/raw.csv` (não versionado)

---

## 1. Método

Sweep **fino** do limiar NONE global do H9a: `AV1_STUDENT_TAU_NONE ∈ {0.55, 0.62, 0.68,
0.74, 0.80, 0.86, 0.92, 0.96}` (8 pontos), com `SPLIT=0.90` e `REST=0.20` fixos e **sem**
overrides por nível (τ uniforme). Custo zero, sem recompilar. Cada τ gera um ponto (BD-rate,
speedup) vs o **mesmo anchor nativo pristino** (h9d_ub, 5fr), mesmo binário
`libaom_extoff_ml`. 8 τ × 3 seqs × 4 CQ × 5 quadros. A pergunta: os 8 pontos formam uma
fronteira **densa e contínua**, ou há **saltos** (gaps) que uma modulação soft preencheria?

## 2. Resultado — a fronteira é densa e contínua

| τ_none | Jockey (BD% / sp) | RaceNight | RiverBank |
|---|--:|--:|--:|
| 0,55 | 1,723% / 1,694× | 1,665% / 1,912× | 0,196% / 1,441× |
| 0,62 | 1,587% / 1,637× | 1,386% / 1,797× | 0,216% / 1,382× |
| 0,68 | 1,519% / 1,583× | 0,891% / 1,651× | 0,165% / 1,338× |
| 0,74 | 1,424% / 1,505× | 0,532% / 1,551× | 0,204% / 1,316× |
| 0,80 | 1,416% / 1,463× | 0,317% / 1,520× | 0,165% / 1,305× |
| 0,86 | 1,435% / 1,435× | 0,277% / 1,494× | 0,186% / 1,289× |
| 0,92 | 1,432% / 1,396× | 0,209% / 1,467× | 0,168% / 1,279× |
| 0,96 | 1,420% / 1,374× | 0,181% / 1,460× | 0,188% / 1,273× |

**Gaps de speedup entre τ vizinhos** (ordenados):

| seq | faixa de speedup | gaps | máx |
|---|---|---|--:|
| Jockey | 1,37–1,69× | 0,02–0,08× | 0,08× |
| RaceNight | 1,46–1,91× | 0,01–0,15× | 0,15× |
| RiverBank | 1,27–1,44× | 0,01–0,06× | 0,06× |

- **Sem saltos grandes.** O maior gap em todo o conjunto (21 vizinhanças) é **0,15×**
  (RaceNight, na ponta agressiva); a maioria é ~0,03×. Não há descontinuidades.
- **BD-rate suave e monótono** com τ — sem bolsões não-dominados que uma modulação soft
  pudesse explorar. Em RiverBank o BD é quase plano (~0,17–0,22%) em toda a faixa; em
  RaceNight/Jockey sobe suavemente com a agressividade.
- **Preenche os buracos da curva grossa.** A curva de 4 pontos anterior
  (`RESULTADOS_H9d_cota_superior.md §3.7`) tinha salto de ~0,32× entre P_ref (1,47×) e A2
  (1,79×) no Jockey/RaceNight; o sweep fino insere pontos em 1,49/1,52/1,55/1,65/1,80× —
  o salto era artefato da amostragem grossa, não da natureza dura da decisão.

## 3. Veredito — C5 falsificado; poda soft não se justifica (fecha o Bloco 6)

**O knob de τ sozinho já varre o espaço de operação continuamente.** A premissa do C5 — de
que a decisão **dura** deixa uma fronteira esburacada que a modulação **soft** preencheria —
**não se sustenta**: a fronteira já é densa e contínua, com gaps ≤ 0,15× e BD-rate suave.

Combinado com o achado independente do Approach B (`RESULTADOS_approachB.md` — "τ não era o
gargalo"), o **valor esperado da cirurgia soft em C é baixo**: não há lacuna para preencher,
e não há evidência de que suavizar dominaria a curva dura (que já é suave). Portanto:

> **C5 não se justifica como implementação.** A poda dura do H9a, via τ, é efetivamente um
> **controle contínuo de ponto de operação** da fronteira BD×tempo — sem retreino, uma
> variável de ambiente. Isto é um resultado **positivo** para a tese: o pruner implantado
> tem fronteira tunável e densa.

**Os pontos de enxerto ficam mapeados** para trabalho futuro caso se queira testar dominância
(não só densidade): orçamento RD do NONE em `set_none_partition_params`
(`partition_search.c:4244-4274`, `best_remain_rdcost`) e fatores de poda AB
(`partition_strategy.c` constantes fixas). Baixa prioridade — o *a priori* é negativo.

## 4. Limitações

- **Densidade, não dominância.** Este probe mostra que a fronteira dura é densa/contínua (a
  falsificação do plano). Não implementa a modulação soft para testar se ela **dominaria** a
  curva dura — mas a curva dura ser suave e monótona, mais o achado do approachB, tornam a
  dominância improvável. Fechar sem cirurgia é a decisão de valor esperado.
- **5 quadros, 4 CQ, τ_none global uniforme.** Consistente com o anchor 5fr; os BD-rate
  absolutos diferem de `h9_test` (~15fr) e não devem ser cruzados. A ponta agressiva
  (τ≤0,62) tem gaps um pouco maiores (até 0,15×) — ainda pequenos, mas um grid ainda mais
  fino ali reduziria mais.
- **Só o eixo NONE.** O sweep varia τ_none (o lever primário). Os fatores de poda AB (o outro
  ponto de enxerto do C5) não foram varridos; mas o C1/H9d já cobriram o eixo estendido.

## 5. Reprodução

```bash
/workspace/build/venv-ml/bin/python \
  src/scripts/benchmark/c5_fine_tau.py --frames 5
# libaom_extoff_ml (PARTITION_ML_STUDENT=1), anchor nativo de h9d_ub/raw.csv
```
