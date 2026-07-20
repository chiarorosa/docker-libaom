# Crivo offline ponderado por *regret* — avaliação agregada das soluções (A5)

**Data:** 2026-07-20
**Estado:** contribuição metodológica. Nenhum encode novo.
**Split:** validação + teste held-out (HoneyBee, FlowerPan, Lips, Jockey,
RaceNight, RiverBank), **792.840 nós de decisão**; modelos treinados nas 10
sequências restantes.
**Script:** `src/scripts/partition_model/oracle_regret.py`
**Artefatos:** `results/models/oracle_regret/{report,ranking.csv,frontier.csv}`

---

## 1. O problema metodológico

A tese usa um **oráculo offline** para decidir qual solução merece a implantação
cara em C (os *gates* 2/3), reservando o encoder como árbitro final. Mas o oráculo
vigente (`simulate_pruning.py`) mede o risco por **contagem** —
`true_split_lost_pct`, a fração de nós true-SPLIT comprometidos a NONE. Isso é
indefensável para uma tese de doutorado: trata uma poda errada num bloco liso
(RD-ótimo já era NONE, custo ~0) igual a uma num bloco texturizado (true-SPLIT,
custo alto). Um crivo que pesa todos os erros igualmente não deveria decidir onde
gastar horas de encode.

**Objetivo (não é predizer o encoder):** um crivo de **triagem** defensável —
capaz de eliminar candidatos inferiores antes do encode. O árbitro final continua
sendo o codificador.

## 2. O crivo

Substitui-se a contagem pelo ***regret*** (`regret.py`), o custo RD real de podar:

    regret_abs(n) = none_rdcost(n) − RD_subtree(n)  ≥ 0

zero quando a decisão certa já era NONE; grande quando se corta uma subárvore
cara. Duas correções sobre a primeira tentativa (que produziu artefatos):

- **fronteira, não ponto casado** — reporta-se `regret × cost_red` (análogo a
  BD-rate), evitando comparar a variância no extremo baixo dela contra os modelos
  no alto;
- **normalização** — `reg_frac` = 100·Σregret_abs / RD_total, uma **% de
  sobrecarga RD**, legível e comparável entre conteúdos.

`cost_red` (busca poupada) é o proxy de speedup, já usado na tese. O crivo isola a
ação NONE-commit (a única que o `regret_abs` mede).

**Reprodução (contêiner):**

```bash
venv-ml/bin/python src/scripts/partition_model/oracle_regret.py \
  --out-dir results/models/oracle_regret
```

## 3. Avaliação agregada — todas as soluções na mesma vara

Ranking de triagem a `cost_red = 30%` casado (menor `reg_frac` = melhor). Todas as
soluções propostas pela tese, no mesmo conjunto held-out:

| # | solução | reg_frac % ↓ | reg_rel ↓ | split_lost % (contagem) |
|--:|---|--:|--:|--:|
| 1 | GNN (deployable) | 0,000 | 6,31 | 0,25 |
| 2 | H9c | 0,003 | 6,80 | 0,37 |
| 3 | GNN causal | 0,004 | 21,9 | 2,40 |
| 4 | **H9a (implantado)** | 0,006 | 24,8 | 2,16 |
| 5 | pixels24 | 0,015 | 30,4 | 4,20 |
| 6 | variância | 0,060 | 222 | 13,10 |
| 7 | aleatório | 0,612 | 966 | 18,27 |
| — | regressão-regret | não alcança 30% (34–83%) | | |

Fronteira `reg_frac %` por `cost_red`:

| solução | 15% | 20% | 30% | 40% | 50% |
|---|--:|--:|--:|--:|--:|
| aleatório | 0,285 | 0,395 | 0,612 | 0,847 | 1,123 |
| variância | 0,023 | 0,035 | 0,060 | 0,084 | 0,128 |
| pixels24 | 0,002 | 0,005 | 0,015 | — | — |
| H9a | 0,000 | 0,001 | 0,006 | 0,023 | — |
| H9c | — | 0,000 | 0,003 | 0,015 | — |
| GNN | 0,000 | 0,000 | 0,000 | 0,001 | — |
| GNN causal | 0,000 | 0,001 | 0,004 | 0,016 | — |

## 4. O que o crivo estabelece — e o que não

### 4.1 Triagem por níveis (o resultado positivo)

O crivo **separa inequivocamente** três níveis, por custo RD real e não por
contagem cega:

    aleatório (966) ≫ variância (222) ≫ família aprendida (6–30)   [reg_rel]

E dentro da **família implantável e justa** (causal, sem vantagem de informação),
a ordem é a que a tese prevê:

    H9a (24,8) < pixels24 (30,4) < variância (222) < aleatório (966)

O H9a supera pixels24 e ambos esmagam a variância — **evidência, pela métrica
defensável, de que o contexto RD barato agrega valor de decisão**, agora sem
depender da contagem. A variância ilustra por que a contagem engana: tem
`split_lost` relativamente baixo em regimes onde seu `reg_frac` é alto (corta
poucos true-SPLIT, mas os que corta são caros) — só o *regret* expõe isso.

### 4.2 Os dois primeiros lugares carregam ressalvas

- **GNN (#1)** — o crivo o coloca à frente, mas o encoder o **rebaixa ~2×** em BD
  (§4.3). Um proxy offline não deve ser lido no topo aqui.
- **H9c (#2)** — tem **vantagem de informação**: seu atributo E é o `none_rdcost`,
  a mesma quantidade sobre a qual o *regret* é construído. É o teto, não um
  competidor em pé de igualdade.

A leitura defensável, portanto, é a da família justa (§4.1), não o pódio bruto.

### 4.3 O limite do crivo (o resultado que sustenta a metodologia)

No **único par com chão real limpo** — GNN vs H9a, medido no encoder
(`RESULTADOS_approachB.md §5`, Jockey, replay fiel, ambos competentes) — o crivo
**diverge**: rankeia o GNN à frente, o encoder dá **~2× menos BD ao H9a** em todo
τ. (O outro par documentado, variância vs "ML", tem chão **contaminado**: o "ML"
era o estudante de pixels fraco, macro-F1 0,203, CB-2 — não vale como verdade.)

Conclusão metodológica, medida e não conjecturada: **um crivo offline, mesmo
ponderado por custo, filtra perdedores mas não adjudica o vencedor entre modelos
competitivos.** Essa é a justificativa empírica para o encoder permanecer o
árbitro final — e é mais forte do que a afirmação não medida que a tese tinha.

## 5. Achado que refina a explicação do Approach B

O `RESULTADOS_approachB.md:118-121` atribui a derrota real do GNN a "poucas podas
confiantes **caras em RD**". A medição **não sustenta** essa causa: as podas
NONE do GNN são **baratas por ambos os critérios** — `split_lost` = 0,25% e
`reg_frac` ≈ 0 a 30% de `cost_red` (o menor de todos). Logo a falha real do GNN
**não está na ação NONE-commit** medida aqui.

Sua causa fica como **pergunta aberta** (não testada nesta análise): vazamento de
vizinhança na expressividade do grafo; descasamento entre os rótulos RDO
cpu-used=0 do treino e o custo cpu-used≥1 real; ou dano por outra ação da
política. Registra-se como hipótese a investigar, **não como causa provada** — a
frase original do Approach B deve ser suavizada de acordo.

## 6. Limitações

- **Rótulos cpu-used=0.** O *regret* mede qualidade de decisão contra o ótimo RDO
  cpu0 (a referência de treino), não o custo cpu-used≥1 exato da implantação.
- **Escopo NONE-commit.** O crivo pondera a ação de comprometer NONE; ações de
  forçar-split e desabilitar-retangular não entram no `regret_abs`.
- **Amostragem.** 2.000 superblocos por (sequência, QP) — amostra ampla (792 mil
  nós de decisão), suficiente para o ranking; não é o dataset inteiro.
- **Chão real fino.** Apenas um par (GNN vs H9a) tem verdade real limpa. O crivo é
  validado na sua **consistência interna** (níveis + família justa), não numa
  predição ampla do encoder — que ele explicitamente não faz.
