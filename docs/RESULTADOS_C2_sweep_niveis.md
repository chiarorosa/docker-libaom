# C2 — de onde vem o ganho do H9a, por nível de bloco (16/32/64)?

**Data:** 2026-07-21
**Bloco 6, item C2.** Sweep de variáveis de ambiente (**custo zero, sem recompilar**):
decompõe a contribuição do H9a implantado (P_ref) por nível de bloco, desligando cada
nível e medindo speedup e BD-rate perdidos. Diagnóstico central: se o ganho viesse todo
de um nível, a comparação com a CNN nativa mudaria de figura; e revela onde o H9a é
eficiente vs agressivo.
**Script:** `src/scripts/benchmark/c2_level_sweep.py` · **BD:** `bd_rate.py`
**Dados:** `results/benchmark/c2_levels/raw.csv` (não versionado)

---

## 1. Método — leave-one-out por env-var

O H9a poda em três níveis (16/32/64). Desligar um nível = setar seus três taus para não
disparar nenhuma ação. Semântica (`partition_strategy.c:2142-2148`, cadeia if/elif):

| ação | condição | desligar com |
|---|---|---|
| NONE-commit | `probs[0] > tau_none` | `tau_none = 2.0` (probs ≤ 1) |
| SPLIT-force | `probs[1] > tau_split` | `tau_split = 2.0` |
| REST-off | `probs[2] < tau_rest` | `tau_rest = -1` (**não** 2.0 — 2.0 ativaria sempre) |

**Leave-one-out a partir do P_ref** (ponto de referência implantado: NONE 0,85/0,80/0,80,
SPLIT 0,90, REST 0,20): `P_ref − offL` isola a contribuição **marginal** do nível L. Tudo
vs o **mesmo anchor nativo pristino** (h9d_ub, 5fr), **mesmo binário** `libaom_extoff_ml`.
Sanidade: desligar um nível deixa o encode **mais lento** (menos poda) — confirmado.

## 2. Resultado — BD-rate e speedup vs nativo (5fr)

| seq | P_ref | off16 | off32 | off64 |
|---|--:|--:|--:|--:|
| Jockey | 1,412% / 1,469× | 1,324% / 1,287× | 1,294% / 1,276× | **0,111% / 1,247×** |
| RaceNight | 0,322% / 1,532× | 0,301% / 1,471× | 0,331% / 1,141× | −0,195% / 1,374× |
| RiverBank | 0,160% / 1,292× | 0,142% / 1,266× | 0,097% / 1,109× | 0,055% / 1,205× |

### Contribuição marginal de cada nível (`P_ref − offL`)

| nível | Jockey (Δspeedup \| ΔBD) | RaceNight | RiverBank |
|---|--:|--:|--:|
| 16 | +0,183× \| +0,089 pp | +0,061× \| +0,021 pp | +0,026× \| +0,018 pp |
| 32 | +0,194× \| +0,119 pp | **+0,391× \| −0,009 pp** | +0,184× \| +0,064 pp |
| 64 | +0,223× \| **+1,302 pp** | +0,158× \| +0,517 pp | +0,087× \| +0,105 pp |

### Eficiência (BD-rate em pp por 1× de speedup — menor é melhor)

| nível | Jockey | RaceNight | RiverBank |
|---|--:|--:|--:|
| 16 | 0,49 | 0,34 | 0,69 |
| 32 | 0,61 | ~0 (grátis) | 0,35 |
| 64 | **5,84** | **3,27** | **1,21** |

## 3. Achados

1. **O nível 64 é o lever AGRESSIVO e CARO.** Carrega a maior parte do custo de BD-rate do
   H9a: no Jockey, desligá-lo derruba o BD de 1,41% → 0,11% (some ~92% do custo de
   qualidade) perdendo só 0,223× de speedup. A eficiência é péssima (5,84 pp/× no Jockey,
   3,27 no RaceNight). Faz sentido: um NONE-commit errado num 64×64 descarta uma subárvore
   enorme — muita qualidade em risco por decisão.
2. **O nível 32 é o mais EFICIENTE.** Melhor speedup por pp de BD; no RaceNight rende
   +0,391× **de graça** (BD até melhora 0,009 pp). É o "doce" do H9a.
3. **O nível 16 contribui pouco** — speedup modesto, BD pequeno. O valor do H9a mora em
   32 e 64.
4. **É dependente de conteúdo.** Jockey: o 64 domina (mais speedup e quase todo o BD).
   RaceNight/RiverBank: o 32 dá o maior speedup, e barato. Um τ único por nível é subótimo.
5. **O ganho NÃO vem de um só nível** — espalha-se por 32 e 64 (e um pouco no 16). Logo a
   comparação H9a×CNN **não** é um artefato de "um nível faz tudo"; a resposta ao
   diagnóstico do plano é: distribuído.

## 4. Implicações

- **Retuning por nível pode mover a fronteira.** O H9a usa τ quase uniforme (NONE
  0,85/0,80/0,80). O C2 diz que o **64 deveria ser mais conservador** (τ_none maior, poda
  menos — corta o pior custo de BD) e o **32 pode ser mais agressivo** (τ menor — colhe
  speedup barato). Isso é um ajuste de env-var, sem código — candidato a ganho grátis na
  fronteira BD×tempo.
- **Cruza com o C1 e o H9d.** O custo de AB/4-way (C1) também se concentra em 32/64, e o
  lever agressivo do H9a (nível 64 NONE-commit) é onde a qualidade é mais sensível.
  Convergem: os blocos grandes são o campo de batalha — tanto do custo (C1) quanto do risco
  de qualidade (C2). Um H9d pós-NONE (§`RESULTADOS_H9d_cota_superior.md`) atuaria no resíduo
  desses mesmos blocos grandes.

## 5. Limitações

- **Leave-one-out, não only-one.** Mede a contribuição *marginal* (o que se perde ao
  desligar cada nível, com os outros ativos). Por causa da hierarquia há interação (um
  NONE-commit no 64 remove os 32/16 abaixo), então as contribuições marginais não somam
  exatamente o total. Uma decomposição "só-um-nível" daria a visão complementar.
- **5 quadros, 4 CQ, base P_ref.** Consistente internamente (mesmo anchor/binário); os
  BD-rate absolutos diferem de `h9_test` (~15fr) e não devem ser cruzados. Diagnóstico, não
  fronteira final.
- **Retuning sugerido não foi medido** — a §4 é hipótese acionável (τ_64 mais conservador),
  a confirmar com um sweep dirigido se a tese quiser esse ganho.

## 6. Reprodução

```bash
/workspace/build/venv-ml/bin/python \
  src/scripts/benchmark/c2_level_sweep.py
# usa libaom_extoff_ml (PARTITION_ML_STUDENT=1) e o anchor nativo de h9d_ub/raw.csv
```
