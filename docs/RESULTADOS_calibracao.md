# Calibração da softmax do estudante implantado (A4)

**Data:** 2026-07-19
**Estado:** primeira análise de calibração do projeto. Nenhum encode novo.
**Modelo:** `results/models/student_h9a/students.pt` — os **pesos implantados em C**
(os mesmos que `export_weights.py` despeja em `partition_student_weights.h`).
**Split:** teste congelado (Jockey, RaceNight, RiverBank), **1.816.393 nós de decisão**
(`block_dim ∈ {16,32,64}`), held-out — nenhuma dessas sequências entrou no treino.

---

## 1. Por que esta análise existe

O ponto de operação da tese **é** um limiar sobre esta softmax: o pruner em C
comita NONE quando `P(NONE) > τ_none` e força split quando `P(SPLIT) > τ_split`
(`partition_strategy.c:2142-2145`), com os defaults compilados `τ_none = τ_split =
0,90` (`student_get_taus`). Em toda a documentação "τ=0,95" foi lido como "95% de
confiança" **sem que isso tivesse sido medido** — era o contra-argumento de banca
CB-6, sem resposta em disco. Não havia nenhuma análise de calibração no projeto.

Método (reprodução exata da inferência implantada): para cada nó, `node_features_h9a`
→ a rede (que tem a normalização **dobrada na primeira camada**, `distill.py:139-144`,
logo consome features cruas como o codificador) → softmax. Comparado contra o rótulo
verdadeiro colapsado (`student.collapse_label`, o rótulo RDO cpu-used=0). Tudo offline.

**Reprodução (contêiner):**

```bash
venv-ml/bin/python src/scripts/partition_model/calibration.py \
  --out-dir results/models/student_h9a/calibration
```

Saídas: `calibration_report.md` + `ece.csv`, `threshold_precision.csv`, `reliability.csv`.

---

## 2. Resultado — o estudante está bem calibrado

### 2.1 ECE (erro esperado de calibração; 0 = perfeito)

| grupo | n | ECE top | ECE NONE | ECE SPLIT | ECE REST |
|---|--:|--:|--:|--:|--:|
| todos | 1.816.393 | **0,0112** | 0,0206 | 0,0050 | 0,0179 |
| dim16 | 1.308.299 | 0,0168 | 0,0234 | 0,0051 | 0,0229 |
| dim32 | 396.668 | 0,0075 | 0,0194 | 0,0102 | 0,0176 |
| dim64 | 111.426 | 0,0137 | 0,0063 | 0,0227 | 0,0175 |

Um ECE top-label de **1,1%** é bom por qualquer padrão da literatura (redes
profundas não calibradas ficam tipicamente em 5–15%). O modelo **não** precisa de
temperatura nem de Platt/isotônica — sai da destilação com CE de rótulo duro já
calibrado, provavelmente porque é raso (2 camadas ocultas) e treinado sem os
truques que descalibram redes grandes.

### 2.2 Precisão no limiar — o que "τ de confiança" realmente vale

Entre os nós com `P(classe) > τ`, a fração que **de fato** é dessa classe:

| τ | precisão NONE | cobertura NONE | precisão SPLIT | cobertura SPLIT |
|--:|--:|--:|--:|--:|
| 0,50 | 0,859 | 60,1% | 0,792 | 12,8% |
| 0,70 | 0,907 | 48,2% | 0,888 | 8,8% |
| 0,85 | 0,942 | 35,9% | 0,948 | 5,5% |
| **0,90** | **0,956** | 29,9% | **0,965** | 4,1% |
| **0,95** | **0,972** | 21,0% | **0,984** | 2,5% |

**A leitura que responde CB-6:** "τ de confiança" é uma descrição **honesta e até
levemente conservadora** do ponto de operação. Em τ=0,95, a precisão real é 97,2%
(NONE) e 98,4% (SPLIT) — **acima** do nominal. No ponto realmente implantado
(τ=0,90), o pruner comita NONE só quando acerta 95,6% das vezes e força split só
quando acerta 96,5%. A afirmação da tese de que os limiares são pontos de
confiança calibrados agora tem base empírica, não retórica.

### 2.3 Diagrama de confiabilidade (one-vs-rest, classe NONE)

| faixa P(NONE) | n | conf média | freq empírica |
|---|--:|--:|--:|
| 0,0–0,1 | 280.464 | 0,038 | 0,031 |
| 0,3–0,4 | 98.509 | 0,350 | 0,389 |
| 0,5–0,6 | 101.959 | 0,551 | 0,612 |
| 0,6–0,7 | 113.937 | 0,651 | 0,711 |
| 0,8–0,9 | 195.359 | 0,854 | 0,857 |
| 0,9–1,0 | 543.107 | 0,965 | 0,956 |

A curva fica **próxima da diagonal**, com um desvio sistemático suave: no meio da
escala (P∈[0,5;0,7]) o modelo é **sub-confiante** — a frequência real é ~6 pp maior
que a probabilidade predita. Nos extremos (onde as decisões acontecem) o
alinhamento é quase exato. Sub-confiança no meio é o modo de erro benigno: não
gera podas erradas confiantes. (SPLIT tem o mesmo padrão, ver
`calibration_report.md` §3.2.)

---

## 3. Consequências

1. **CB-6 fica respondido, positivamente.** Não é preciso recalibrar nada nem
   qualificar o uso de τ como limiar de confiança. Pelo contrário: o ponto de
   operação é ligeiramente conservador.

2. **Há um pequeno TS potencialmente na mesa — mas exige encodes para confirmar.**
   Como a precisão em τ=0,90 já é 95,6% (NONE) e a de τ=0,95 é 97,2%, o alvo de
   "95% de precisão" é atingido em torno de τ≈0,88. Usar τ=0,95 é, em precisão,
   mais conservador que o rótulo "95%" sugere — pode haver economia de tempo a
   ganhar afrouxando o limiar. Isso conecta-se a **B2** (τ adaptativo) e **não** é
   conclusível offline: a troca BD×tempo tem de ser medida no encoder.

3. **A calibração é uniforme entre níveis**, com ECE por dim entre 0,008 e 0,017.
   Não há um `block_dim` descalibrado que exigisse τ próprio por razão de
   calibração (a motivação de τ por nível em B2 vem da **distribuição de rótulos**,
   A8, não da calibração).

---

## 4. Limitações

- **O rótulo de referência é a decisão RDO cpu-used=0.** A calibração mede o
  acordo da softmax com o oráculo RDO de treino. Se "correto" no encoder cpu1+
  implantado difere dessa decisão cpu0, esta análise não captura a diferença — mas
  o **valor de implantação** já é respondido empiricamente pelo swap
  (`RESULTADOS_fase6_swap_h9c.md`), então as duas peças são complementares.
- **Só o H9a.** O H9c é modelo separado e sua softmax alimenta um gate diferente
  (pós-NONE); `calibration.py` roda sobre ele trocando `--students`, mas não foi
  feito aqui.
- **Split de teste, held-out — mas 3 sequências, uma classe** (A1 4K). Boa medida
  de generalização entre conteúdos, não uma garantia entre resoluções.
- Os nós 8×8 estão fora por construção (folhas terminais, A8); a calibração é dos
  nós de decisão, que é onde o pruner age.
