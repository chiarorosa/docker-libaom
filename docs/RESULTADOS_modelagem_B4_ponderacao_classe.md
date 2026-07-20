# Modelagem B4 — ponderação de classe por nível (ablação, resultado negativo)

**Data:** 2026-07-20
**Artefato:** `results/models/student_h9a_cw/` (cw = *class-weighted*). Nomeava-se
`student_h9a_otimo` no pedido inicial; renomeado após a ablação, porque o nome
"ótimo" afirmava o que o resultado refuta — o artefato nomeia o **experimento**
(ponderação de classe), não um desfecho.
**Baseline:** `student_h9a` (implantado, treino sem ponderação).
**Split:** validação + teste held-out; modelos treinados nas 10 restantes.

---

## 1. Motivação e a tensão de projeto

O A8 mediu inflação de zeros extrema no SPLIT em 16×16 (`split_recall`≈0,026;
SPLIT é 4,40% dos rótulos ali, e cai a 0,14% em cq55). O B4 propõe corrigir com
**ponderação de classe por nível**. Como `train_student_h9` treina uma MLP por
tamanho de bloco, basta `use_class_weight=True` — a ponderação já é por nível.

Mas há uma **contra-indicação registrada no próprio código** (`distill.py:114-116`):
para um pruner por **limiar de confiança**, a ponderação suprime P(NONE) confiante
e a CE sem peso dá probabilidades mais afiadas e melhor ranqueadas. E o A4
mostrou que o H9a atual (sem peso) é bem calibrado (ECE 0,011). Ou seja: o B4
podia **piorar**. Esta ablação mede, em vez de assumir.

**Reprodução:**
```bash
venv-ml/bin/python src/scripts/partition_model/train_student_h9.py \
  --out-dir results/models/student_h9a_cw --class-weight
venv-ml/bin/python src/scripts/partition_model/compare_students.py     # classificação
venv-ml/bin/python src/scripts/partition_model/calibration.py \
  --students results/models/student_h9a_cw/students.pt ...              # calibração
venv-ml/bin/python src/scripts/partition_model/oracle_regret.py ...     # crivo (inclui H9a_cw)
```

## 2. O que melhorou — classificação (o alvo declarado do B4)

Recall por classe e macro-F1 por nível (30 mil superblocos held-out, mesmos nós
para os dois modelos):

| dim | modelo | rec NONE | rec SPLIT | rec REST | macro-F1 |
|--:|---|--:|--:|--:|--:|
| 16 | H9a | 0,832 | **0,022** | 0,722 | 0,524 |
| 16 | H9a_cw | 0,585 | **0,361** | 0,829 | 0,556 |
| 32 | H9a | 0,547 | 0,827 | 0,548 | 0,639 |
| 32 | H9a_cw | 0,333 | 0,835 | 0,667 | 0,629 |
| 64 | H9a | 0,560 | 0,992 | **0,000** | 0,515 |
| 64 | H9a_cw | 0,681 | 0,925 | **0,497** | 0,556 |

O SPLIT-recall em 16×16 saltou **16×** (0,022 → 0,361) e o modelo passou a prever
REST (era 0 em 64×64). **Como classificador, a ponderação funciona.**

## 3. O que quebrou — calibração (o que importa para o pruner)

Para um pruner por limiar, macro-F1 não é a métrica de implantação; a calibração
da softmax é (é sobre ela que τ opera). Comparado ao baseline do A4:

| | ECE top | ECE NONE | τ=0,90 P(SPLIT) precisão |
|---|--:|--:|--:|
| H9a (A4) | **0,011** | 0,021 | 0,965 |
| H9a_cw | 0,040 (3,6×) | **0,174** (8×) | 0,880 |

O diagrama de confiabilidade do H9a_cw expõe o mecanismo: em P(NONE)=0,5 a
frequência real de NONE é **0,82** — o modelo ficou **sistematicamente
sub-confiante em NONE**, exatamente o que `distill.py:114-116` previa. Consequência
para o pruner: sub-comprometimento de NONE → poda menos.

## 4. O veredito — o crivo ponderado por *regret* (τ-varrido, 792.840 nós)

Esta é a medição decisiva: varrendo τ, o crivo (`oracle_regret.py`) diz a
qualidade **fundamental** de decisão, independente do ponto de τ. A `cost_red`
casado:

| # | solução | reg_frac % ↓ | reg_rel ↓ | split_lost % | cost_red atingível |
|--:|---|--:|--:|--:|---|
| 4 | **H9a (implantado)** | **0,006** | 24,81 | 2,16 | 5–42% |
| 5 | **H9a_cw (B4)** | 0,009 | 24,72 | 1,99 | 2–**35%** |

Empate técnico na qualidade de decisão (H9a marginalmente melhor em `reg_frac`,
H9a_cw marginalmente melhor em `reg_rel` e `split_lost` — tudo dentro do ruído).
Mas o H9a_cw **poda menos**: alcança no máximo 35% de `cost_red` contra 42% do H9a
(o P(NONE) deflacionado limita o teto de poda).

## 5. Conclusão

**O B4 não merece implantação.** Ele troca:
- **ganho:** balanço de classificação (SPLIT-recall 16×, prevê REST);
- **por perda:** calibração arruinada (ECE 8× pior em NONE) e teto de poda menor
  (35% vs 42%), sem nenhum ganho na qualidade de decisão ponderada por custo
  (crivo empatado).

Para um pruner por limiar, o recall de classificação é **irrelevante** ao que o
encoder paga; o que importa é a qualidade calibrada da decisão de poda, e nisso a
ponderação não ajuda e a calibração atrapalha. **A ablação valida empiricamente a
decisão de projeto de treinar sem ponderação** (`distill.py:114-116`) — antes um
argumento assumido, agora medido. O `student_h9a` implantado permanece o correto.

**Contribuição para a tese:** um "conserto óbvio" (ponderar a inflação de zeros
que o A8 mediu) é um **falso positivo** para pruners por limiar — o recall melhora,
mas não se traduz em poda melhor e custa calibração. É justificativa de projeto
medida, não retórica.

## 6. Limitações e o que fica em aberto

- Testou-se a ponderação por frequência inversa (`class_weights_3`, recorte
  [0,1; 10]). Uma ponderação **mais suave** poderia mover o recall com menos dano
  de calibração — não testado; mas como o crivo já empata, é improvável que supere
  o baseline na métrica que importa.
- O **B1** (contexto RD hereditário: `none_rdcost` do pai e irmãos já decididos)
  é a alavanca de modelagem de maior evidência e **não** foi coberta aqui — muda o
  vetor de atributos (e o lado C), então é experimento separado, não um drop-in.
- Nome do artefato: renomeado de `student_h9a_otimo` (aspiracional, do pedido)
  para `student_h9a_cw` — o nome passou a descrever o experimento (ponderação de
  classe), não o desfecho. É um **registro negativo** no inventário.
