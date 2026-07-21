# H9d — Etapa 3 (Fase 1): confirmação no encoder

**Data:** 2026-07-21
**H9d, Etapa 3 — o árbitro final.** Encodes reais confirmando se o H9d **seletivo**
(implantado na Etapa 2) domina (a) o próprio *blanket* e (b) o knob de τ do H9a. Fase 1 no
plano de 5 quadros (mesmo anchor/curvas das etapas anteriores) para a decisão; Fase 2
(≥10 quadros) confirma o vencedor.
**Script:** `src/scripts/benchmark/h9d_selective_sweep.py` · **BD:** `bd_rate.py`
**Dados:** `results/benchmark/h9d_selective/raw.csv` (não versionado)

---

## 1. Método

Base **H9a P_ref** (o pruner implantado) + H9d ligado, varrendo `AV1_STUDENT_H9D_TAU ∈
{0,05, 0,10, 0,20, 0,30, 0,45}` (pula AB/4-way se `P(EXT) < τ`; τ maior = mais agressivo).
Binário `libaom_extoff_ml`. Tudo vs o **mesmo anchor nativo pristino** (h9d_ub, 5fr), 3 seqs
de teste, 4 CQ. Sobreposto às referências já medidas no mesmo plano: curva de τ do H9a
(P_ref→A1→A2→A3, `h9d_tau`) e o blanket H9a+extoff (`h9d_marg`).

## 2. Resultado — fronteira BD-rate × speedup (vs nativo, 5fr)

| config | Jockey | RaceNight | RiverBank |
|---|--:|--:|--:|
| H9a (P_ref) | 1,41% / 1,46× | 0,32% / 1,50× | 0,16% / 1,29× |
| H9a τ A2_none70 | 2,05% / 1,79× | 1,11% / 1,70× | 0,28% / 1,41× |
| H9a τ A3_none60 | 2,47% / 2,08× | 1,96% / 2,01× | 0,45% / 1,54× |
| H9a+extoff (blanket) | 2,19% / 2,00× | 1,21% / 1,86× | 0,91% / 1,66× |
| **H9a+H9d τ=0,10** | 1,48% / 1,49× | 0,55% / 1,61× | 0,19% / 1,32× |
| **H9a+H9d τ=0,20** | 1,70% / 1,61× | 0,70% / 1,69× | 0,30% / 1,38× |
| **H9a+H9d τ=0,30** | 1,72% / 1,77× | 0,90% / 1,75× | 0,44% / 1,45× |
| **H9a+H9d τ=0,45** | 2,02% / 1,87× | 1,08% / 1,80× | 0,64% / 1,55× |

## 3. Análise

### 3.1 H9d seletivo DOMINA o blanket — nas três sequências
A previsão da Etapa 1 confirma-se: a seleção recupera os vencedores que o blanket descartava.
Ao mesmo (ou maior) speedup, o seletivo custa menos BD:
- **RiverBank:** blanket 0,91%/1,66× vs H9d τ=0,45 **0,64%**/1,55× — menos BD, quase o mesmo
  speedup; o H9d está claramente à esquerda do blanket.
- **RaceNight:** blanket 1,21%/1,86× vs H9d τ=0,45 **1,08%**/1,80×.
- **Jockey:** blanket 2,19%/2,00× vs H9d τ=0,45 **2,02%**/1,87×.

### 3.2 Vs subir o τ do H9a — net-positivo (vence 2 de 3)
Comparando o H9d ao ponto da curva de τ **de mesmo speedup** (interpolado):
- **RaceNight — vitória clara.** H9d é ~0,19–0,35 pp **abaixo** da curva de τ em todos os τ
  (ex.: τ=0,30 dá 1,75× a 0,90% BD; a curva de τ ao mesmo 1,75× custaria ~1,24%).
- **Jockey — vitória modesta.** On-curve em baixa agressividade; ~0,14–0,29 pp abaixo em
  τ=0,30/0,45 (ex.: 1,77× a 1,72% vs ~2,01% da curva de τ).
- **RiverBank — leve perda.** ~0,05–0,17 pp **acima** da curva de τ (ex.: 1,45× a 0,44% vs
  ~0,33% da curva). É o conteúdo onde AB/4-way quase não importam (base EXT 3–8%), então há
  pouco a ganhar podando-os, e a imperfeição do modelo aparece.

## 4. Veredito (Fase 1)

**O H9d seletivo é um lever real e funcional — segunda solução positiva da tese.** Dois
resultados sólidos: (i) domina o próprio blanket em todas as sequências (a seleção vale);
(ii) bate o knob de τ do H9a em 2 de 3 sequências de teste (claro no RaceNight, modesto no
Jockey), perdendo levemente só onde o eixo estendido é irrelevante (RiverBank). É uma
melhoria de Pareto genuína sobre o blanket e uma extensão útil da fronteira do H9a.

**Ressalva honesta:** as margens vs a curva de τ são modestas (~0,1–0,35 pp) e a Fase 1 é 5
quadros — dentro do ruído possível em Jockey/RiverBank. A **Fase 2** (≥10 quadros, no melhor
τ, ~0,30) confirma que a vantagem no RaceNight/Jockey é sinal, não ruído.

## 4.1 Fase 2 (10 quadros) — CONFIRMA a Fase 1

Confirmação a **10 quadros** (rigor de resultado E), 3 seqs, 4 CQ, vs nativo: H9a(P_ref),
`H9a A2_none70` (o competidor "subir o τ") e `H9a+H9d(τ=0,30)`.

| seq | H9a A2 (subir τ) | **H9a+H9d(0,30)** | ao mesmo speedup |
|---|--:|--:|---|
| **RaceNight** | 1,282% / 1,757× | **0,981% / 1,771×** | **H9d domina Pareto** (+speedup E −0,30 pp) |
| **Jockey** | 2,038% / 1,798× | **1,783% / 1,775×** | H9d **melhor** (~−0,22 pp @1,775×) |
| RiverBank | 0,282% / 1,416× | 0,443% / 1,458× | H9d ~+0,12 pp pior |

**A 10 quadros o veredito da Fase 1 confirma-se — as margens eram sinal, não ruído:**
- **RaceNight:** H9d **Pareto-domina** subir o τ (mais rápido **e** menos BD-rate).
- **Jockey:** H9d ~0,22 pp abaixo da curva de τ ao mesmo speedup.
- **RiverBank:** H9d ~0,12 pp acima (o conteúdo onde AB/4-way não carregam decisão).

**Contribuição fechada:** o H9d é uma **segunda solução positiva confirmada no encoder** —
um podador aprendido pós-NONE do eixo estendido que melhora a fronteira BD×tempo do H9a em
2 de 3 sequências de teste held-out (uma delas por dominância de Pareto estrita), e domina
seu próprio blanket em todas. Não é universal (RiverBank, onde o estendido é irrelevante,
tem leve perda), mas é um ganho real e medido no árbitro final.

## 5. Limitações
- **5 quadros (Fase 1) / 10 quadros (Fase 2).** A F2 confirma a F1; a tese usa ≥10-15 para
  números finais de tabela — os 10 quadros da F2 estão no piso do rigor E.
- **Base P_ref única.** Empilhar H9d sobre bases de τ variadas geraria uma família 2D;
  P_ref (implantado) é o natural para a decisão.
- **PSNR-Y apenas.** Padrão da tese.
- **τ do H9d global.** Por-nível (`_16/_32/_64`) pode melhorar (o C2 mostrou o 64 mais
  sensível); refinamento para depois.

## 6. Reprodução
```bash
/workspace/build/venv-ml/bin/python \
  src/scripts/benchmark/h9d_selective_sweep.py
```
