# H9d no test set CTC — resultados finais (capítulo de resultados)

**Data:** 2026-07-25
**O que é.** O H9d (podador aprendido pós-NONE do eixo estendido AB/4-way) foi fechado
como segunda solução positiva sobre as 3 sequências de teste (capítulo de **metodologia**,
`RESULTADOS_H9d_etapa3_encoder.md`). Este documento leva a **melhor configuração** — PL10,
o τ por nível baqueado como default no C — ao **test set CTC completo, sob protocolo CTC**,
exatamente como já feito para o H9a (`ml_balanced`/`ml_aggr`). É o número de tabela do
capítulo de resultados.

**Scripts:** `src/scripts/fase6/ctc_h9d.py` (encodes) · `report_ctc.py` (BD/TS) ·
`ctc_h9d_marginal.py` (contribuição marginal)
**Dados:** `results/benchmark/fase6/raw_results.csv` (config `ml_bal_h9d`, 32 linhas;
não versionado)

---

## 1. Método

**Protocolo AOM-CTC All-Intra, Classe A1** (idêntico ao já usado para o H9a): 8 sequências
4K 10-bit, `cpu-used=0`, 15 quadros, CQ ∈ {20, 32, 43, 55}, `--tile-columns=1 --threads=2
--row-mt=0`, `--end-usage=q --passes=1 --kf-max-dist=0 --deltaq-mode=0 --enable-tpl-model=0`.
Anchor = libaom original, `cpu-used=0`. BD-rate/BD-PSNR Bjøntegaard (PCHIP) sobre PSNR-Y;
TS%/speedup médios sobre os QP.

**Configuração medida — `ml_bal_h9d`:** o ponto balanceado implantado do H9a (P_rect:
`TAU_NONE=0,95`, `TAU_SPLIT=0,90`, `TAU_REST=0,20`) **mais** o H9d ligado
(`AV1_STUDENT_H9D_ENABLE=1`, τ não setado → defaults por nível **PL10**:
τ_16=0,091, τ_32=0,103, τ_64=0,014). O H9d é **complemento**, nunca substituto: a política
do H9a é mantida fixa e o H9d é empilhado por cima, de modo que o delta contra as linhas
`ml_balanced` já medidas é a contribuição marginal pura do H9d.

**Binário:** `build/libaom_perf_h9d`, compilado com as mesmas flags de `build/libaom_perf`
(Release, `-DPARTITION_ML_STUDENT=1`, `CONFIG_INTERNAL_STATS=0`, SIMD nativo) para que os
tempos de parede sejam comparáveis entre configurações.

### 1.1 Verificação de integridade (pré-requisito)

Antes de medir, o binário novo foi obrigado a reproduzir a base H9a. Re-encode do ponto de
referência `BoxingPractice cq32` com `libaom_perf_h9d` e **H9d desligado**:

| | obtido | referência (`ml_balanced`) | |
|---|--:|--:|---|
| bytes | 1 574 775 | 1 574 775 | **idêntico** |
| PSNR-Y | 40,9720 dB | 40,9720 dB | **idêntico** |
| tempo | 369,4 s | 370,2 s | Δ 0,2% (ruído) |

O fluxo de bits é **byte-a-byte idêntico** → o código do H9d é inerte quando desligado e a
base H9a no binário novo é a mesma. O marginal medido abaixo é limpo.

## 2. Resultado — tabela CTC (vs anchor libaom `cpu-used=0`)

Média sobre as 8 sequências (`bdrate_average.csv`):

| configuração | BD-rate (%) | BD-PSNR (dB) | TS (%) | speedup |
|---|--:|--:|--:|--:|
| H9a balanceado (P_rect) | +0,57 | −0,019 | 17,7 | 1,22× |
| **H9a + H9d (PL10)** | **+0,59** | −0,020 | **18,7** | **1,24×** |
| H9a agressivo (A3) | +1,40 | −0,049 | 31,5 | 1,49× |
| libaom `cpu-used=1` | +0,45 | −0,017 | 32,6 | 1,51× |
| libaom `cpu-used=2` | +0,54 | −0,020 | 42,7 | 1,79× |
| libaom `cpu-used=3` | +2,72 | −0,094 | 67,9 | 3,16× |

Por sequência, em `results/benchmark/fase6/bdrate_per_seq.csv` e `tables.tex`.

## 3. Contribuição marginal do H9d (a pergunta que importa)

O H9d empilha sobre o H9a; a comparação correta não é "H9d vs nativo", e sim: **partindo do
ponto implantado, o H9d compra tempo mais barato do que simplesmente subir o τ do H9a?**
O knob de τ é representado no CTC pelo segmento P_rect→A3; interpolamos esse segmento no
TS% que o H9d efetivamente atingiu, obtendo o BD que o τ teria custado pelo mesmo tempo.
Δ negativo = H9d melhor.

| sequência | H9a P_rect | H9a+H9d | Δ BD (pp) | Δ TS (pp) | vs knob de τ |
|---|--:|--:|--:|--:|--:|
| BoxingPractice | 0,63% / 19,4% | 0,66% / 20,3% | +0,03 | +0,9 | **−0,037** |
| Crosswalk | 0,50% / 20,2% | 0,53% / 20,5% | +0,03 | +0,4 | +0,018 |
| FoodMarket2 | 0,63% / 12,4% | **0,61%** / 14,1% | **−0,02** | **+1,8** | **−0,117** |
| Neon1224 | 0,64% / 21,9% | 0,68% / 22,0% | +0,05 | +0,1 | +0,041 |
| NocturneDance | 0,15% / 12,6% | 0,17% / 13,8% | +0,01 | +1,2 | **−0,053** |
| PierSeaSide | 0,42% / 16,8% | 0,46% / 17,5% | +0,04 | +0,7 | **−0,040** |
| Tango | 1,15% / 21,3% | **1,14%** / 22,9% | **−0,01** | **+1,6** | **−0,112** |
| TimeLapse | 0,41% / 17,3% | 0,43% / 18,8% | +0,02 | +1,5 | **−0,041** |
| **média** | **0,57% / 17,7%** | **0,59% / 18,7%** | **+0,018** | **+1,02** | **−0,043** |

### 3.1 O H9d compra tempo ~3,5× mais barato que o knob de τ
O preço do tempo, em pp de BD-rate por pp de TS:

| mecanismo | preço |
|---|--:|
| **H9d empilhado (PL10)** | **0,018 pp/pp** |
| knob de τ do H9a (P_rect→A3) | 0,063 pp/pp |

Ou seja: no ponto de operação implantado, o quadro de tempo que o H9d entrega custa cerca de
**um terço** do BD-rate que custaria obtê-lo afrouxando o limiar do H9a. É exatamente a
propriedade que se esperava do lever — ele age sobre um eixo (partição estendida) que o H9a
não toca, em vez de simplesmente ser mais permissivo no mesmo eixo.

### 3.2 Vence o knob de τ em 6 das 8 sequências; duas com dominância de Pareto estrita
Contra o ponto de mesmo TS na curva de τ, o H9d é melhor em **6/8** (média −0,043 pp),
perdendo levemente só em Crosswalk (+0,018) e Neon1224 (+0,041) — as duas em que o Δ TS é
quase nulo (0,4 e 0,1 pp), isto é, onde o eixo estendido praticamente não é exercido e o
modelo quase não tem o que podar. **FoodMarket2 e Tango são dominância de Pareto estrita**:
menos BD-rate **e** mais tempo economizado que a base P_rect.

**Nota sobre ruído:** bytes e PSNR-Y são determinísticos para um dado encoder e entrada, logo
os BD-rates (e os Δ BD) são medidas **exatas**, não estimativas ruidosas — o único componente
com variância entre execuções é o tempo de parede. Os Δ BD negativos de FoodMarket2/Tango,
embora minúsculos, são reais: podar candidatos AB/4-way muda o contexto de vizinhança das
decisões seguintes, e a busca de partição do AV1 não é monotônica nesse sentido.

### 3.3 Magnitude — honestidade sobre o tamanho do efeito
O ganho marginal é **modesto em valor absoluto**: +1,0 pp de TS (17,7% → 18,7%; 1,22× →
1,24×) por +0,018 pp de BD-rate. Duas razões mecânicas, ambas esperadas:
1. **P_rect é conservador.** O H9a já poda bastante *antes* da busca; o resíduo pós-NONE
   sobre o qual o H9d atua é menor do que seria em um ponto base mais frouxo.
2. **PL10 é a calibração segura.** Foi escolhida na Fase 2b justamente por *nunca* perder
   para a curva de τ (robustez), o que a coloca em um ponto de agressividade baixa. Pontos
   PL20/PLmix trocam robustez por mais tempo.

A contribuição científica do H9d, portanto, não é o tamanho do speedup: é **demonstrar que o
eixo de partição estendida é aprendizável e podável a um preço de BD substancialmente menor
que o do limiar existente**, estendendo a fronteira BD×tempo do H9a em uma direção que o
próprio H9a não alcança.

## 4. Veredito

**Confirmado no test set CTC, sob protocolo CTC.** O H9d empilhado ao ponto implantado do H9a
entrega **+1,0 pp de TS por +0,018 pp de BD-rate** — um preço de tempo ~3,5× menor que o do
knob de τ do H9a — vencendo o ponto equivalente da curva de τ em **6 das 8 sequências CTC**,
duas delas por **dominância de Pareto estrita**. O resultado do capítulo de metodologia
(3 sequências, PL10 ≥ curva de τ nas três) **generaliza** para o conjunto CTC completo.

## 5. Limitações

Decisões de escopo (métrica, número de quadros, divergências de flag impostas pelo libaom)
estão registradas em `DECISOES_escopo.md` e **não** são limitações — em particular, os
**15 quadros são exatamente a especificação da CTC §4.1** (`--limit=15`), não um recorte
desta tese. Limitações de fato:

- **Um único ponto de operação do H9d** (PL10 sobre P_rect). Uma família 2D (PL10/PL20 ×
  P_rect/A3) daria a fronteira completa; PL10 sobre P_rect é o par implantado/seguro.
- **Tempo de parede** medido em contêiner compartilhado; Δ TS de ~0,1 pp (Neon1224) está na
  ordem do ruído de medição. Os Δ BD, ao contrário, são exatos (§3.2).
- **Modelo treinado em 8 bits**, aplicado a conteúdo 10-bit; a normalização
  `src >> (bd-8)` em `student_node_features` é a mesma já validada para o H9a/H9c.

## 6. Reprodução
```bash
# integridade (H9d desligado deve reproduzir ml_balanced byte-a-byte)
/workspace/build/venv-ml/bin/python src/scripts/fase6/ctc_h9d.py --integrity
# 8 seqs x 4 CQ, config ml_bal_h9d  (~3,5 h)
/workspace/build/venv-ml/bin/python src/scripts/fase6/ctc_h9d.py
# tabelas + marginal
/workspace/build/venv-ml/bin/python src/scripts/fase6/report_ctc.py
/workspace/build/venv-ml/bin/python src/scripts/fase6/ctc_h9d_marginal.py
```
