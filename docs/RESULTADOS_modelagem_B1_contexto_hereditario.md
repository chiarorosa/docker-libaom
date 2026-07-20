# Modelagem B1 — contexto RD hereditário (ablação, resultado negativo)

**Data:** 2026-07-20
**Artefato:** `results/models/student_h9a_b1/` (42 atributos = h9a 36 + 6 hereditários).
**Baseline:** `student_h9a` (implantado, 36 atributos).
**Split:** validação + teste held-out; modelos treinados nas 10 restantes.
**Avaliação:** `results/models/oracle_regret_b1/` (crivo ponderado por *regret*).

---

## 1. Hipótese e desenho

O contexto RD do **pai** e dos **irmãos já decididos** carrega sinal de
profundidade que o nó isolado não tem — e é **causalmente disponível** no pruner
pré-busca: o pai avalia seu `PARTITION_NONE` antes de recorrer nos filhos, e os
irmãos anteriores na ordem-z são buscados antes do nó corrente. Nenhum modelo
anterior viu esse sinal (o GNN estrutural foi alimentado só com o vetor h9a de 36
dims, cego a RD).

**6 atributos hereditários** (`features.py:node_features_h9a_b1`), anexados aos 36
do h9a → vetor de 42. Deliberadamente **só magnitudes RD medidas**, nunca
rótulos/decisões — para evitar o descasamento treino↔deploy que afundou o H9c:

| idx | atributo |
|--:|---|
| 36 | `has_parent` (1 se dim<64) |
| 37–39 | `log1p` do `none_rdcost`/`rate`/`dist` do **pai** (`dim*2, r//2, c//2`) |
| 40 | `log1p(média none_rdcost dos irmãos anteriores)` (z-index < atual) |
| 41 | nº de irmãos anteriores presentes / 3 |

Tudo zero-fill na raiz (dim=64 não tem pai). Implementação verificada contra a
ordem de busca (ordem-z `[(0,0),(0,1),(1,0),(1,1)]`).

**Reprodução:**
```bash
venv-ml/bin/python src/scripts/partition_model/train_student_h9.py \
  --out-dir results/models/student_h9a_b1 --feature-set h9a_b1
venv-ml/bin/python src/scripts/partition_model/oracle_regret.py \
  --out-dir results/models/oracle_regret_b1
```

## 2. Veredito — o crivo ponderado por *regret* (τ-varrido)

A `cost_red` casado (proxy de speedup), o risco por custo RD:

| @ cost_red 30% | reg_frac % ↓ | reg_rel ↓ | split_lost % |
|---|--:|--:|--:|
| **H9a (implantado)** | **0,006** | **24,81** | 2,16 |
| H9a_b1 (B1) | 0,007 | **38,81** | 1,64 |

E o negativo é **robusto ao longo de toda a fronteira** — a `cost_red` casado, o
H9a_b1 tem `reg_frac` **e** `reg_rel` maiores que o H9a em todos os pontos de τ:

| cost_red ≈ | H9a reg_rel | H9a_b1 reg_rel |
|--:|--:|--:|
| 34% | 35,9 | 59,7 |
| 30% | 28,0 | 40,0 |
| 24% | ~11 | 13,6 |

**O contexto RD hereditário NÃO ajuda — degrada a qualidade de decisão ponderada
por custo.** A τ igual, o B1 poda um pouco *mais* (mais NONE-commits), mas em nós
mais caros em RD — a magnitude RD do pai/irmãos empurra o modelo a comprometer
NONE onde não deveria. O único ganho é no `split_lost` (contagem), a métrica que
já estabelecemos como indefensável (`RESULTADOS_oraculo_regret.md`).

## 3. Conclusão

**O B1 não merece implantação.** É o **segundo** experimento de modelagem negativo
(após o B4): nem a ponderação de classe nem o contexto RD hereditário melhoram o
H9a na métrica que importa. **O estudante implantado (36 atributos, treino sem
peso) mostra-se robusto** — as duas extensões "óbvias" falham.

**Contribuição para a tese:** um sinal que é *informativo e causalmente
disponível* (o custo RD do pai/irmãos) **não é o mesmo que um sinal útil à decisão
de poda**. O modelo o incorpora e fica pior — provavelmente porque a magnitude RD
correlaciona-se com complexidade da região de forma que confunde o compromisso
NONE-vs-split. É justificativa medida para o vetor de 36 atributos, não retórica.

## 4. Limitações e o que fica em aberto

- Testou-se **uma** formulação (magnitudes RD do pai + média dos irmãos
  anteriores). Um sinal negativo desta forma não prova que "hierarquia não tem
  valor" — apenas que estas features, como definidas, não ajudam.
- **Formulações não testadas:** razões (custo do nó relativo ao pai) em vez de
  magnitudes absolutas; a *decisão* do irmão anterior (mas isso reintroduz o
  descasamento treino↔deploy do H9c); um modelo estrutural que agregue o contexto
  em vez de concatená-lo (o GNN tentou isso e falhou no real por outra razão,
  `RESULTADOS_approachB.md`).
- **Orçamento de treino igual** (30 épocas) para 42 vs 36 atributos; a truth-acc
  ficou comparável (0,82/0,75/0,86), então subtreino é improvável como explicação,
  mas não foi varrido.
- Calibração do B1 **não** medida — `calibration.py` usa o vetor h9a de 36; exigiria
  adaptá-lo para 42. Dado que o crivo já é negativo, foi despriorizado.
