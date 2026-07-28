# E5 — ablação de atribuição no codificador, conjunto de validação

**Data:** 2026-07-28
**O que é.** A ablação de atribuição que sustenta a leitura "o ganho é do modelo, não
da política" existia em duas versões insuficientes: a **CB-1** (Jockey, 2 quadros, uma
sequência) e a da **Fase 5** (3 seqs de teste, 10 quadros) — esta última sem comparação
a tempo casado, porque as faixas de speedup de `ml` e `variance` saíram **disjuntas em
3/3**. O E5 refaz a medição no conjunto de **validação**, com o grid da variância
estendido ao extremo conservador que nunca havia sido sondado.

**Scripts:** `src/scripts/benchmark/ablation_attrib.py` (medição) ·
`run_e5_validation.sh` (fila) · `stop_e5_after_lips.sh` (corte de escopo)
**Dados:** `results/benchmark/e5_ablation/{FlowerPan,Lips}/{curve,runs}.csv`
(não versionados)

---

## 1. Método

**Política casada — é isto que torna a atribuição pura.** Os três braços rodam a
**mesma** política no **mesmo** codificador: NONE-commit puro, `TAU_SPLIT=2` (nunca
força split) e `TAU_REST=-1` (poda de retangulares desligada). Varia-se **somente a
fonte de escore**, via `AV1_STUDENT_BASELINE`:

| braço | escore | grid de τ |
|---|---|---|
| `ml` | o estudante H9a implantado (36 atributos) | **congelado**: 0,95 0,90 0,80 0,70 0,60 0,50 |
| `variance` | `P(NONE) = exp(−var/V0)` | **estendido**: 0,999 0,995 0,99 0,97 0,95 0,90 0,80 |
| `random` | hash uniforme (poda a mesma fração ao acaso) | 0,95 0,90 0,80 0,70 |

**Por que só o grid da variância muda.** A não-sobreposição da Fase 5 tem causa
conhecida: o grid foi congelado na calibração da HoneyBee, onde `variance` a τ=0,95
dava 1,34×; em outro conteúdo o mesmo τ já salta para ~1,9×. Estender o grid é
legítimo **na validação**, cujo papel declarado no protocolo é escolher limiares
operacionais, e seria ilegítimo no teste (seria ajustar configuração vendo dado de
teste). O braço do `ml` fica **intocado** justamente para que a proposta não seja a
que se re-ajusta — a mexida beneficia o adversário, não a hipótese.

**Cobertura:** 2 sequências de validação (FlowerPan, Lips), 10 quadros, `cpu-used=0`,
`--threads=1`, CQ {20, 32, 43, 55}, âncora `libaom_perf_anchor` por sequência.
34 pontos de operação, 144 codificações, ~22 h.

**HoneyBee ficou de fora por decisão de escopo** — ver `DECISOES_escopo.md`. É a
sequência em que o grid original foi calibrado, portanto a menos independente das
três. Consequência para a leitura: o portão pré-registrado pedia "≥2 das **3**", e
aqui há **2 sequências**, não 3. Isso é declarado na §4, não contornado.

## 2. FlowerPan — as faixas se sobrepõem, e o `ml` domina

| braço | τ | BD-rate | TS | speedup |
|---|--:|--:|--:|--:|
| `ml` | 0,95 | **−0,015%** | 3,8% | 1,039× |
| `ml` | 0,90 | 0,015% | 5,6% | 1,059× |
| `ml` | 0,80 | 0,015% | 9,0% | 1,099× |
| `ml` | 0,70 | 0,095% | 12,6% | 1,144× |
| `ml` | 0,60 | 0,274% | 16,3% | 1,195× |
| `ml` | 0,50 | 0,638% | 21,2% | 1,269× |
| `variance` | 0,999 | 0,000% | 1,7% | 1,017× |
| `variance` | 0,99 | 0,013% | 2,5% | 1,026× |
| `variance` | 0,97 | 0,437% | 13,0% | 1,150× |
| `variance` | 0,95 | 1,180% | 21,7% | 1,277× |
| `variance` | 0,90 | 3,414% | 40,0% | 1,667× |
| `variance` | 0,80 | 6,610% | 54,0% | 2,173× |
| `random` | 0,95 | 2,372% | 9,0% | 1,099× |
| `random` | 0,90 | 5,181% | 16,3% | 1,194× |
| `random` | 0,80 | 9,642% | 29,5% | 1,418× |
| `random` | 0,70 | 13,225% | 41,1% | 1,697× |

**A tempo casado** (pontos medidos, sem interpolação, pareados pelo speedup mais
próximo):

| speedup | `ml` | `variance` | razão | `random` | razão |
|---|--:|--:|--:|--:|--:|
| ~1,10× | 0,015% (1,099×) | — | — | 2,372% (1,099×) | **158×** |
| ~1,15× | 0,095% (1,144×) | 0,437% (1,150×) | **4,6×** | — | — |
| ~1,19× | 0,274% (1,195×) | — | — | 5,181% (1,194×) | **19×** |
| ~1,27× | 0,638% (1,269×) | 1,180% (1,277×) | **1,85×** | — | — |

O `ml` domina os dois adversários em **todos** os pares casados. O ponto τ=0,95 sai com
BD-rate **negativo** (−0,015%): 3,8% de tempo com ganho marginal de qualidade.

**O grid estendido fez o que se pretendia.** `variance` a τ=0,999 entrega 1,7% de TS a
**0,000%** de BD — a heurística *consegue* operar em regime conservador, e agora isso
está medido em vez de suposto. A comparação ficou mais honesta, não mais favorável.

## 3. Lips — a variância não tem ponto de operação na região implantável

| braço | τ | BD-rate | TS | speedup |
|---|--:|--:|--:|--:|
| `ml` | 0,95 | **−0,073%** | 6,9% | 1,074× |
| `ml` | 0,90 | 0,125% | 9,3% | 1,103× |
| `ml` | 0,80 | 0,190% | 14,0% | 1,163× |
| `ml` | 0,70 | 0,285% | 15,7% | 1,186× |
| `ml` | 0,60 | 0,416% | 19,1% | 1,236× |
| `ml` | 0,50 | 0,521% | 22,1% | 1,283× |
| `variance` | 0,999 | 0,000% | 0,2% | 1,002× |
| `variance` | 0,995 | 0,000% | 0,4% | 1,004× |
| `variance` | 0,99 | 0,019% | 0,6% | 1,006× |
| `variance` | **0,97** | **6,580%** | **71,9%** | **3,563×** |
| `variance` | 0,95 | 7,474% | 78,0% | 4,535× |
| `variance` | 0,90 | 7,903% | 81,2% | 5,309× |
| `variance` | 0,80 | 8,243% | 83,4% | 6,007× |
| `random` | 0,95 | 1,013% | 7,9% | 1,085× |
| `random` | 0,90 | 1,998% | 15,6% | 1,184× |
| `random` | 0,80 | 3,590% | 30,2% | 1,432× |
| `random` | 0,70 | 4,795% | 42,1% | 1,726× |

**Há um precipício entre τ=0,99 e τ=0,97:** o speedup salta de **1,006× para 3,563×**,
e o BD-rate de 0,019% para 6,58%. Não existe ponto intermediário no grid. O `ml` vive
inteiramente dentro desse vão (1,074×–1,283×), de modo que **a comparação a tempo
casado contra a variância não existe nesta sequência** — de novo, mas por um motivo
diferente e mais informativo do que na Fase 5.

**Contra o aleatório, a comparação existe e é limpa:**

| speedup | `ml` | `random` | razão |
|---|--:|--:|--:|
| ~1,08× | −0,073% (1,074×) | 1,013% (1,085×) | — (o `ml` é negativo) |
| ~1,19× | 0,285% (1,186×) | 1,998% (1,184×) | **7,0×** |

**Mecanismo provável.** A Lips é um close-up de rosto: grandes regiões de pele, lisas e
de variância baixa e homogênea. Um limiar sobre a variância é aproximadamente
**bimodal** nesse conteúdo — ou quase nada cruza o limiar, ou quase tudo cruza de uma
vez. É exatamente a falha que se esperaria de um descritor de um único momento
estatístico, e é o contraste com o `ml`, que gradua continuamente ao longo dos seis τ.

## 4. Veredito — honesto sobre o que foi e o que não foi atingido

**O portão pré-registrado NÃO foi atingido na forma estrita.** Ele pedia dominância a
tempo casado sobre a variância em **≥2 de 3** sequências; obteve-se em **1 de 2**
(FlowerPan), porque na Lips não há par casado a comparar e a HoneyBee não foi rodada.

**O que ficou estabelecido, e é substancial:**

1. **A primeira comparação a tempo casado contra a variância na história desta tese**
   — e o `ml` a vence em todos os pares (4,6× e 1,85×). A Fase 5 nunca produziu isto.
2. **Atribuição contra o aleatório: limpa em 2/2**, com margens de 7× a 158×. Sob
   política idêntica, o ganho vem da seleção de nós.
3. **A limitação da Fase 5 está fechada.** Lá, a não-sobreposição tinha uma explicação
   alternativa incômoda: *"o grid foi mal escolhido"*. Sondou-se agora o extremo
   conservador (τ até 0,999) e, na Lips, a variância **continua** sem ponto de operação
   na região implantável. A não-sobreposição é **propriedade do escore**, não do grid —
   e isso é uma afirmação mais forte do que a que se tentava obter.
4. **O `ml` alcança BD-rate negativo em 2/2** (−0,015% e −0,073%), isto é, existe um
   ajuste em que ele economiza 4–7% do tempo sem custo de qualidade. A variância também
   chega a 0,000%, mas por não podar quase nada (0,2–1,7% de TS contra 3,8–6,9%).

**Leitura para a tese.** A afirmação defensável não é "o `ml` domina a variância em
toda parte", e sim: **sob política idêntica, o escore do modelo gradua continuamente a
região implantável de baixo BD, que o escore da variância ou não alcança ou atravessa
de um salto** — e, onde ambos coexistem, o modelo custa 2 a 5 vezes menos BD-rate pelo
mesmo tempo.

## 5. O que falta para decidir o portão

Três pontos de τ na **Lips**, dentro do precipício: **τ = 0,985, 0,98, 0,975**. Se a
variância aterrissar em 1,07×–1,28× ali, há par casado e o portão se decide na forma
estrita; se atravessar o vão de novo, o precipício fica **demonstrado** em vez de
inferido de dois pontos. **Custo ~2,3 h.** Ambos os desfechos são publicáveis, o que
torna o experimento barato em risco. **Não executado** — aguarda decisão.

## 6. Limitações

- **Duas sequências, não três.** A HoneyBee é decisão de escopo (`DECISOES_escopo.md`),
  não omissão de resultado; mas o portão foi redigido para três e isto o deixa
  formalmente indeciso.
- **O precipício da Lips repousa em dois pontos adjacentes** (τ=0,99 e τ=0,97). O
  mecanismo proposto (bimodalidade da variância em pele) é **interpretação**, não
  medição — a §5 é o que o testaria.
- **10 quadros por sequência.** Acima dos 2 da CB-1 e alinhado ao protocolo, mas ainda
  um recorte temporal curto.
- **Os Δ BD são exatos** (bytes e PSNR são determinísticos); o ruído está só no tempo de
  parede, com resolução medida de ~0,46 pp (`RESULTADOS_BLOCO7_E3_DEC_E2.md §3`). Todos
  os Δ TS pareados acima estão muito acima disso.
- **Interrupção e retomada.** A campanha parou em 27/07 20:52 (parada da máquina
  virtual) com a FlowerPan em 15/17 pontos, e foi retomada em 28/07 via `--resume`. Os
  pontos reaproveitados são os mesmos arquivos, não re-medições; a âncora da FlowerPan
  foi medida antes da interrupção e reusada depois. Como tempo de parede é o único
  componente com variância, e a interrupção não caiu no meio de um ponto concluído, o
  efeito é nulo sobre os BD e desprezível sobre os TS — mas fica registrado.

## 7. Reprodução

```bash
# campanha (retomável; --resume reaproveita ancora e pontos concluidos)
docker exec -d av1_bench sh -c \
  'sh /workspace/src/scripts/benchmark/run_e5_validation.sh \
   >> /workspace/results/benchmark/e5_ablation.log 2>&1'

# uma sequencia isolada
/workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/benchmark/ablation_attrib.py \
  --seq /workspace/src/samples/Lips_3840x2160_120fps_420_8bit_YUV_RAW.yuv \
  --frames 10 --cqs 20 32 43 55 --methods ml variance random \
  --tau-none-for ml=0.95,0.90,0.80,0.70,0.60,0.50 \
                 variance=0.999,0.995,0.99,0.97,0.95,0.90,0.80 \
                 random=0.95,0.90,0.80,0.70 \
  --out-dir /workspace/results/benchmark/e5_ablation/Lips --resume

# o preenchimento do precipicio proposto na §5 (nao executado)
#   --methods variance --tau-none-for variance=0.985,0.98,0.975  --resume
```
