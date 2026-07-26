# Bloco 7 — E3 (joelho de τ), decomposição de 3 pernas e E2 (σ medido)

**Data:** 2026-07-26
**104 encodes** em cadeia estritamente serial (~9h54), protocolo AOM-CTC All-Intra,
mesmo anchor da Fase 6. Um `aomenc` por vez, para preservar a comparabilidade dos tempos.

**Scripts:** `encode_h9c_cq20.py --taus 45` (E3) · `encode_h9adef.py` (DEC) ·
`encode_repeat.py` (E2) · `report_e3_dec_e2.py` (análise)
**Dados:** `results/benchmark/fase6/raw_results.csv` e `fase6_repeat/raw_results.csv`

---

## 1. E3 — o joelho está em τ≈60–70, e τ45 **não** é ponto atrativo

Subconjunto casado (Neon1224, PierSeaSide, TimeLapse), onde toda a curva existe:

| τ | BD-rate (%) | TS (%) | preço do degrau (pp BD / pp TS) |
|---|--:|--:|--:|
| 30 | +0,775 | 23,02 | — |
| **45** (novo) | **+0,430** | **18,16** | **0,0711** |
| 60 | +0,298 | 16,92 | **0,1069** |
| 70 | +0,280 | 15,58 | 0,0133 |
| 90 | +0,209 | 13,80 | 0,0397 |
| 95 | +0,187 | 13,27 | 0,0420 |

*Preço do degrau = pp de BD-rate pagos por pp de TS ao afrouxar τ um degrau (isto é, ao
descer na tabela → subir na tabela custa isso).*

**A região plana termina em τ70.** De 95 a 70 o preço fica entre 0,013 e 0,042 pp/pp. A
partir de 60 ele salta para **0,107** — 2,5 a 8 vezes mais caro. O joelho, portanto, está
na fronteira **τ≈60–70**, e não dentro do vão (30,60) como se supunha.

**τ45 não está sobre uma inflexão favorável — está ligeiramente acima da corda.** A reta
que liga τ60 (0,298 / 16,92) a τ30 (0,775 / 23,02) prevê, no TS de 18,16% que τ45 atinge,
um BD de **0,395%**; o medido é **0,430%**. τ45 fica **0,035 pp acima** da interpolação, ou
seja, a curva é levemente convexa ali. Não há ponto de operação escondido a colher.

### 1.1 τ45 nas 8 sequências — e é dominado pela fronteira nativa

| sequência | BD-rate | TS |
|---|--:|--:|
| BoxingPractice | +0,672% | 23,8% |
| Crosswalk | +0,792% | 32,7% |
| FoodMarket2 | +0,501% | 17,1% |
| Neon1224 | +0,426% | 21,3% |
| NocturneDance | +0,328% | 11,2% |
| PierSeaSide | +0,491% | 16,1% |
| Tango | +1,562% | 31,4% |
| TimeLapse | +0,373% | 17,1% |
| **média** | **+0,643%** | **21,4%** |

Contra as referências já medidas: `native_cpu1` entrega **+0,449% a 32,6%** — menos BD-rate
**e** muito mais tempo economizado. **τ45 é estritamente dominado pelo knob nativo.** Isso
**confirma a previsão** registrada no plano ("nenhuma extrapolação de τ em cpu0 alcança a
fronteira nativa; não gastar encodes tentando") — agora por medição, não por extrapolação.

**Conclusão do E3:** a faixa inexplorada foi explorada e **não contém nada**. É um negativo
limpo, que fecha a questão da curva de τ.

## 2. Decomposição de 3 pernas — H9a e H9c são parcialmente **redundantes**

Com `h9adef` (H9a@default sozinho) medido nas mesmas sequências, o balanço fecha:
`TS(H9a) + TS(H9c) + interação = TS(H9a+H9c)`.

| sequência | H9a só | H9c só | H9a+H9c | soma | **interação** |
|---|--:|--:|--:|--:|--:|
| Neon1224 | 16,8% | 4,2% | 17,1% | 21,0% | **−3,9 pp** |
| PierSeaSide | 10,4% | 5,4% | 12,8% | 15,8% | **−3,0 pp** |
| Tango | 7,2% | 12,3% | 17,1% | 19,4% | **−2,4 pp** |
| TimeLapse | 9,2% | 0,6% | 11,5% | 9,8% | +1,7 pp |
| **média** | **10,9%** | **5,6%** | **14,6%** | **16,5%** | **−1,9 pp** |

**Isto refina o E4.** Lá o resultado foi "em média 64% do TS atribuído ao H9c era do H9a".
A decomposição completa mostra que **não é uma partição limpa**: os dois podadores disputam
os mesmos nós. Empilhados entregam 14,6%, contra 16,5% da soma das partes — **~12% do ganho
potencial evapora na sobreposição**.

O Tango é o caso instrutivo: é a única sequência em que o H9c sozinho (12,3%) supera o H9a
sozinho (7,2%), e mesmo ali a interação é negativa. O TimeLapse é a exceção (+1,7 pp de
sinergia), mas com H9c sozinho a 0,6% o número é pequeno e a interpretação, frágil.

**Consequência para a tese:** reforça, com um mecanismo explícito, por que o H9c não é
contribuição autônoma — ele não adiciona um eixo novo, ele repete em parte o que o H9a já
faz. Contrasta com o **H9d**, que age sobre o eixo de partição **estendida**, que o H9a não
toca — e cuja interação, por construção, não pode ser essa.

## 3. E2 — o piso de ruído medido é **~4× menor** do que o inferido

Cinco repetições intercaladas (Crosswalk, 3 configurações × 4 CQ), em execução contínua.

### 3.1 Dispersão do tempo bruto

| configuração | CV mediano | CV máximo |
|---|--:|--:|
| anchor / ml_balanced / native_cpu1 | **0,28%** | **0,64%** |

O plano estimava **σ ≈ 1–2% por encode**, inferido de violações de monotonicidades que
valem por construção. **A medição direta dá 0,28% mediano** — a estimativa anterior era
pessimista por um fator ~4.

### 3.2 O número que importa: σ do TS pareado

| configuração | TS medido | sd (5 repetições) | erro-padrão | resolução (2 sd) |
|---|--:|--:|--:|--:|
| `ml_balanced` | 20,29% | **±0,23 pp** | 0,10 pp | ~0,46 pp |
| `native_cpu1` | 32,84% | **±0,09 pp** | 0,04 pp | ~0,18 pp |

### 3.3 Releitura das tabelas anteriores

Isto **valida o resultado do H9d** e, ao mesmo tempo, obriga a qualificar parte dele:

- O marginal do H9d no CTC foi **+1,02 pp de TS** — **~4,4 σ** acima do ruído. O resultado
  agregado é sólido, e agora com piso de ruído medido, não suposto.
- Mas na tabela por sequência de `RESULTADOS_H9d_CTC.md`, os Δ TS de **Neon1224 (+0,1 pp)**
  e **Crosswalk (+0,4 pp)** estão **dentro** da resolução de ~0,46 pp — não são
  distinguíveis de zero num encode único. Os demais (+0,7 a +1,8 pp) são resolvíveis.
  Os **Δ BD-rate continuam exatos** em todas (bytes e PSNR são determinísticos), então a
  conclusão de Pareto não se altera; o que se qualifica é a leitura dos ganhos de tempo
  sequência a sequência.
- Simetricamente, muitas comparações antes descartadas como "dentro do ruído" pelo piso de
  1–2% **são de fato resolvíveis**. O piso real é bem mais permissivo.

## 4. Limitações

Métrica e número de quadros são decisões de escopo — ver `DECISOES_escopo.md`.

- **O σ do E2 é intra-execução.** Cinco repetições numa janela contínua, mesma sequência,
  mesmo contêiner sem reinício. **Não** captura deriva entre dias, reinícios de contêiner ou
  estados térmicos distintos. Como as campanhas da tese rodaram em janelas contínuas
  análogas, é o σ pertinente para as comparações internas — mas comparar números medidos com
  semanas de intervalo pede cautela adicional.
- **Uma sequência (Crosswalk), três configurações.** A homogeneidade dos CV (0,13–0,64% em
  todas as células) sugere que o valor não é idiossincrático, mas isso não está demonstrado
  em outros conteúdos.
- **A decomposição cobre 4 das 8 sequências**, e a interação positiva do TimeLapse repousa
  sobre um H9c-sozinho de 0,6% — pequeno demais para sustentar interpretação.
- **E3 no subconjunto casado de 3 sequências** para a curva completa (é o único onde todos os
  τ existem); τ45 isolado foi medido nas 8.

## 5. Reprodução
```bash
/workspace/build/venv-ml/bin/python src/scripts/fase6/encode_h9c_cq20.py \
    --cqs 20 32 43 55 --taus 45
/workspace/build/venv-ml/bin/python src/scripts/fase6/encode_h9adef.py \
    --seqs PierSeaSide Tango TimeLapse
/workspace/build/venv-ml/bin/python src/scripts/fase6/encode_repeat.py \
    --seq Crosswalk --reps 5
/workspace/build/venv-ml/bin/python src/scripts/fase6/report_e3_dec_e2.py
```
