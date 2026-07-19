# Calibração da softmax do estudante implantado (A4)

**Data:** 2026-07-19  
**Split de teste:** Jockey, RaceNight, RiverBank (1816393 nós de decisão)
**Modelo:** `results/models/student_h9a/students.pt` (pesos implantados em C)  
**Reprodução:** `python src/scripts/partition_model/calibration.py`

## 1. ECE (top-label) e por classe

ECE = erro esperado de calibração; 0 = perfeitamente calibrado. Por classe é one-vs-rest.

| grupo | n | ECE top | ECE NONE | ECE SPLIT | ECE REST |
|---|--:|--:|--:|--:|--:|
| all | 1,816,393 | 0.0112 | 0.0206 | 0.0050 | 0.0179 |
| dim16 | 1,308,299 | 0.0168 | 0.0234 | 0.0051 | 0.0229 |
| dim32 | 396,668 | 0.0075 | 0.0194 | 0.0102 | 0.0176 |
| dim64 | 111,426 | 0.0137 | 0.0063 | 0.0227 | 0.0175 |

## 2. Precisão no limiar — o que "τ de confiança" realmente vale

Entre os nós com P(classe) > τ, a fração que **de fato** é dessa classe (precisão), e a cobertura (fração dos nós além do limiar). Se a precisão em τ=0,95 for < 0,95, o limiar não é 95% de confiança.

### 2.1 P(NONE) > τ  (pooled, todos os níveis)

| τ | precisão | cobertura | cobertura % |
|--:|--:|--:|--:|
| 0.50 | 0.8588 | 1,091,573 | 60.10% |
| 0.60 | 0.8842 | 989,614 | 54.48% |
| 0.70 | 0.9067 | 875,677 | 48.21% |
| 0.80 | 0.9294 | 738,466 | 40.66% |
| 0.85 | 0.9420 | 652,683 | 35.93% |
| 0.90 | 0.9556 | 543,107 | 29.90% |
| 0.95 | 0.9722 | 382,145 | 21.04% |

### 2.2 P(SPLIT) > τ  (pooled, todos os níveis)

| τ | precisão | cobertura | cobertura % |
|--:|--:|--:|--:|
| 0.50 | 0.7915 | 232,819 | 12.82% |
| 0.60 | 0.8391 | 196,338 | 10.81% |
| 0.70 | 0.8876 | 159,668 | 8.79% |
| 0.80 | 0.9298 | 121,756 | 6.70% |
| 0.85 | 0.9476 | 100,248 | 5.52% |
| 0.90 | 0.9652 | 74,727 | 4.11% |
| 0.95 | 0.9841 | 46,042 | 2.53% |

## 3. Diagrama de confiabilidade (one-vs-rest)

Por faixa de P(classe): probabilidade média predita (`conf`) contra frequência empírica da classe (`freq`). Calibrado ⇔ conf ≈ freq.

### 3.1 classe NONE

| faixa P | n | conf | freq |
|---|--:|--:|--:|
| 0.0–0.1 | 280,464 | 0.038 | 0.031 |
| 0.1–0.2 | 140,560 | 0.147 | 0.136 |
| 0.2–0.3 | 108,590 | 0.249 | 0.263 |
| 0.3–0.4 | 98,509 | 0.350 | 0.389 |
| 0.4–0.5 | 96,697 | 0.450 | 0.508 |
| 0.5–0.6 | 101,959 | 0.551 | 0.612 |
| 0.6–0.7 | 113,937 | 0.651 | 0.711 |
| 0.7–0.8 | 137,211 | 0.752 | 0.785 |
| 0.8–0.9 | 195,359 | 0.854 | 0.857 |
| 0.9–1.0 | 543,107 | 0.965 | 0.956 |

### 3.2 classe SPLIT

| faixa P | n | conf | freq |
|---|--:|--:|--:|
| 0.0–0.1 | 1,293,207 | 0.018 | 0.020 |
| 0.1–0.2 | 141,086 | 0.143 | 0.160 |
| 0.2–0.3 | 67,740 | 0.245 | 0.259 |
| 0.3–0.4 | 43,884 | 0.348 | 0.358 |
| 0.4–0.5 | 37,657 | 0.449 | 0.451 |
| 0.5–0.6 | 36,481 | 0.550 | 0.535 |
| 0.6–0.7 | 36,670 | 0.650 | 0.628 |
| 0.7–0.8 | 37,912 | 0.751 | 0.752 |
| 0.8–0.9 | 47,029 | 0.853 | 0.874 |
| 0.9–1.0 | 74,727 | 0.959 | 0.965 |

