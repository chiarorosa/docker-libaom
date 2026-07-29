# Modelagem B2 — τ adaptativo por qindex (calibração offline, resultado positivo mas pequeno)

**Data:** 2026-07-20
**Sem re-treino.** Usa o estudante implantado `student_h9a`; só muda o limiar τ.
**Split:** validação + teste held-out (48.000 superblocos).
**Script/artefatos:** `src/scripts/partition_model/b2_tau_per_qindex.py`,
`results/models/b2_tau_qindex/`.

---

## 1. Hipótese

O A8 mediu que a mistura de rótulos varia fortemente com o CQ (SPLIT em 16×16 vai
de 7,6% em cq20 a 0,14% em cq55). A grade τ fixa é, então, a priori mal-casada: o
mesmo τ é conservador num regime e agressivo noutro. O lado C **já lê τ do
ambiente** (`partition_strategy.c`), então τ por CQ é implantável **sem
recompilar** — é uma mudança de calibração, não de código.

**Teste (no risco ponderado por *regret* do crivo A5):** comparar, a `cost_red`
casado, duas políticas de NONE-commit do H9a — τ **global** (um só para todos os
CQ) contra τ **estratificado-ótimo** (cada qindex escolhe o seu; enumeração
exaustiva das combinações grade⁴ → fronteira de Pareto).

## 2. O mecanismo é real — o τ necessário difere enormemente por CQ

τ para atingir cada `cost_red`, por qindex isolado:

| qindex | cr 15% | cr 20% | cr 25% | cr 30% | cr 35% |
|---|--:|--:|--:|--:|--:|
| cq20 | **0,64** | — (não alcança 20%) | — | — | — |
| cq32 | 0,95 | 0,91 | 0,86 | 0,79 | 0,69 |
| cq43 | 0,98 | 0,97 | 0,96 | 0,94 | 0,88 |
| cq55 | 0,98 | 0,97 | 0,96 | 0,94 | 0,91 |

O τ ótimo é **monotônico no CQ**: agressivo em baixa qualidade (cq20 τ≈0,6),
conservador em alta (cq55 τ≈0,9). E o **cq20 é intrinsecamente difícil de podar** —
nem no τ mais agressivo alcança 20% de `cost_red` (conteúdo complexo em alta taxa,
poucas podas NONE seguras). Um τ global **é** mal-casado: confirma a hipótese do A8.

A política ótima escolhe, por CQ:

- **cost_red ≈ 30%:** τ = {cq20: 0,60, cq32: 0,75, cq43: 0,80, cq55: 0,90}

## 3. O ganho é real mas pequeno

`reg_frac %` (% de sobrecarga RD) a `cost_red` casado:

| cost_red | τ global | τ estratificado | redução |
|--:|--:|--:|--:|
| 20% | 0,0010 | 0,0005 | **2,0×** |
| 25% | 0,0032 | 0,0012 | **2,7×** |
| 30% | 0,0061 | 0,0028 | **2,2×** |
| 35% | 0,0117 | 0,0071 | 1,6× |

Em **termos relativos**, a estratificação por CQ **corta o regret ~1,6–2,7×** a
`cost_red` casado — é o **primeiro** lever de modelagem/calibração a mover o crivo
(B4 e B1 foram negativos). Mas em **termos absolutos** o ganho é minúsculo: a
sobrecarga RD cai de 0,006% para 0,003% a 30% de `cost_red`.

## 4. A ressalva de magnitude — provavelmente imaterial no encoder

A redução absoluta (≤ 0,005 pp de sobrecarga RD) está **abaixo do piso de ruído do
tempo de parede** do encoder (σ ≈ 1–2% por encode; ver `RESULTADOS_fase6_swap_h9c.md`
§9).

> **Correção (2026-07-29).** O piso de ruído citado acima (σ ≈ 1–2% por encode) é
> uma suposição, não uma medição, e foi retratado pela medição direta do E2: com
> cinco repetições intercaladas na Crosswalk, o coeficiente de variação mediano do
> tempo bruto é de 0,28%, e a resolução do tempo pareado é de aproximadamente 0,46
> ponto percentual, cerca de 4× mais fino do que o piso suposto aqui. A conclusão
> desta seção não muda, pois a sobrecarga absoluta permanece imaterial sob o piso
> medido. Ver `RESULTADOS_BLOCO7_E3_DEC_E2.md` §4.

BD-rate é aproximadamente proporcional à sobrecarga RD, então o efeito
esperado em BD×tempo real é da ordem de ~0,003% — **quase certamente não mensurável**
num encode. Confirmar exigiria encodes (o lado C já lê τ do ambiente), mas o prior
é que não aparece.

**Por que o absoluto é tão pequeno — conecta com o A4.** O estudante é bem calibrado
(`RESULTADOS_calibracao.md`, ECE 0,011): a qualquer τ ele opera em confiança
calibrada, então o regret por unidade de speedup já é baixo em todo CQ. A
estratificação **rebalanceia** onde se poda entre CQs, mas não há muito regret a
recuperar — a boa calibração do H9a já absorve a maior parte do efeito que a grade
τ fixa deixaria na mesa.

## 5. Conclusão

**B2 é um refinamento de calibração defensável, gratuito e a priori correto — mas
de magnitude imaterial.**

- **Positivo:** a hipótese do A8 confirma-se (τ global mal-casado), e a
  estratificação melhora o crivo (~2× menos regret relativo). É o único lever do
  Bloco 5 que não é negativo.
- **Grátis:** o C já lê τ do ambiente; adotar τ por CQ é mudar o driver de encode,
  sem recompilar.
- **Mas pequeno:** o ganho absoluto está sob o ruído do encoder; não move os
  números-título.

**Recomendação:** adotar τ por CQ como **boa prática de calibração** (é mais
defensável que um τ único e custa zero), mas **não** alegar que melhora BD×tempo
sem medir — e o prior é que o efeito é imaterial.

## 6. Limitações

- **Estratificação-ótima é otimista.** O τ por CQ foi escolhido pela envoltória de
  Pareto sobre o mesmo conjunto de avaliação (val+test), então o ganho medido é um
  **limite superior**. Uma calibração honesta ajustaria τ por CQ na validação e
  mediria no teste — o que só reduziria o ganho já pequeno.
- **Escopo NONE-commit**, τ escalar por CQ (mesmo entre níveis dentro de um CQ);
  não se varreu τ por (CQ, nível) conjuntamente.
- **Confirmação real pendente:** o veredito de BD×tempo é do encoder; aqui é só o
  crivo offline.
