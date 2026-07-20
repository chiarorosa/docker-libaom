# Calibração da softmax do estudante implantado (A4)

**Data:** 2026-07-19  
**Split de teste:** Jockey, RaceNight, RiverBank (63963 nós de decisão)
**Modelo:** `results/models/student_h9a/students.pt` (pesos implantados em C)  
**Reprodução:** `python src/scripts/partition_model/calibration.py`

## 1. ECE (top-label) e por classe

ECE = erro esperado de calibração; 0 = perfeitamente calibrado. Por classe é one-vs-rest.

| grupo | n | ECE top | ECE NONE | ECE SPLIT | ECE REST |
|---|--:|--:|--:|--:|--:|
| all | 63,963 | 0.0400 | 0.1736 | 0.0649 | 0.1087 |
| dim16 | 43,963 | 0.0524 | 0.2168 | 0.0949 | 0.1241 |
| dim32 | 15,200 | 0.0147 | 0.1004 | 0.0377 | 0.0650 |
| dim64 | 4,800 | 0.0189 | 0.0645 | 0.1158 | 0.1254 |

## 2. Precisão no limiar — o que "τ de confiança" realmente vale

Entre os nós com P(classe) > τ, a fração que **de fato** é dessa classe (precisão), e a cobertura (fração dos nós além do limiar). Se a precisão em τ=0,95 for < 0,95, o limiar não é 95% de confiança.

### 2.1 P(NONE) > τ  (pooled, todos os níveis)

| τ | precisão | cobertura | cobertura % |
|--:|--:|--:|--:|
| 0.50 | 0.9337 | 36,507 | 57.08% |
| 0.60 | 0.9493 | 32,038 | 50.09% |
| 0.70 | 0.9617 | 27,785 | 43.44% |
| 0.80 | 0.9742 | 23,290 | 36.41% |
| 0.85 | 0.9809 | 20,672 | 32.32% |
| 0.90 | 0.9854 | 17,463 | 27.30% |
| 0.95 | 0.9900 | 12,740 | 19.92% |

### 2.2 P(SPLIT) > τ  (pooled, todos os níveis)

| τ | precisão | cobertura | cobertura % |
|--:|--:|--:|--:|
| 0.50 | 0.5533 | 9,863 | 15.42% |
| 0.60 | 0.6068 | 7,902 | 12.35% |
| 0.70 | 0.6519 | 6,033 | 9.43% |
| 0.80 | 0.7173 | 3,661 | 5.72% |
| 0.85 | 0.7864 | 2,551 | 3.99% |
| 0.90 | 0.8803 | 1,621 | 2.53% |
| 0.95 | 0.9356 | 776 | 1.21% |

## 3. Diagrama de confiabilidade (one-vs-rest)

Por faixa de P(classe): probabilidade média predita (`conf`) contra frequência empírica da classe (`freq`). Calibrado ⇔ conf ≈ freq.

### 3.1 classe NONE

| faixa P | n | conf | freq |
|---|--:|--:|--:|
| 0.0–0.1 | 7,955 | 0.044 | 0.196 |
| 0.1–0.2 | 5,650 | 0.148 | 0.462 |
| 0.2–0.3 | 4,607 | 0.248 | 0.602 |
| 0.3–0.4 | 4,603 | 0.351 | 0.669 |
| 0.4–0.5 | 4,641 | 0.450 | 0.747 |
| 0.5–0.6 | 4,469 | 0.548 | 0.822 |
| 0.6–0.7 | 4,253 | 0.651 | 0.868 |
| 0.7–0.8 | 4,495 | 0.751 | 0.897 |
| 0.8–0.9 | 5,827 | 0.853 | 0.941 |
| 0.9–1.0 | 17,463 | 0.966 | 0.985 |

### 3.2 classe SPLIT

| faixa P | n | conf | freq |
|---|--:|--:|--:|
| 0.0–0.1 | 40,070 | 0.021 | 0.006 |
| 0.1–0.2 | 6,311 | 0.144 | 0.059 |
| 0.2–0.3 | 3,398 | 0.246 | 0.123 |
| 0.3–0.4 | 2,318 | 0.349 | 0.192 |
| 0.4–0.5 | 2,003 | 0.449 | 0.280 |
| 0.5–0.6 | 1,961 | 0.549 | 0.338 |
| 0.6–0.7 | 1,869 | 0.650 | 0.461 |
| 0.7–0.8 | 2,372 | 0.753 | 0.551 |
| 0.8–0.9 | 2,040 | 0.847 | 0.588 |
| 0.9–1.0 | 1,621 | 0.948 | 0.880 |

