# Approach B — teste estrutural do teto (GNN diagnóstico → implantável) (design)

> **Documento de design (spec).** Data: 2026-07-17. Branch `ml-partition-dev`.
> Contexto: `docs/SINTESE_resultados_metodologia.md` (Conclusão 3),
> `docs/RESULTADOS_solucao4.md` (classificação > regressão),
> `docs/superpowers/specs/2026-07-17-solucao4-regret-regression-design.md` §11
> (Approach B como trabalho futuro). Define **estrutura, interfaces e gates** —
> hiperparâmetros/limiares são propostas iniciais, calibradas em cada gate.

---

## 1. Descrição breve (a contribuição)

**Um classificador NN estruturado (GNN) sobre o quadtree do superbloco**, que
decide as partições dos nós **conjuntamente** (via *message-passing* pai↔filho↔
irmão) em vez de **independentemente** (como os MLPs por nó do H9a). Serve, no
**Stage 1**, como **instrumento de diagnóstico** para responder a pergunta que a
tese deixou em aberto: o teto de desempenho (Conclusão 3) é **informacional**, ou
apenas um **artefato de decisões independentes por nó**? Só se houver folga passa
ao **Stage 2** (versão implantável causal top-down).

---

## 2. Motivação e enquadramento

A **Conclusão 3** da tese (levers correlacionados = teto informacional) foi
estabelecida **empilhando levers escalares** (H9a + H9c + τ) — **nunca** com um
modelo que decida a árvore **conjuntamente**. E o H9a, embora use contexto **causal**
de vizinhança (bloco B: tamanhos das partições acima/esquerda), decide **cada nó de
forma independente**. Fica aberta a possibilidade de que a estrutura do quadtree
(dependências pai/filho/irmão, não-causais) carregue sinal que os nós independentes
não capturam.

A **Solução 4** acrescentou a lição de que, para a decisão de poda,
**classificação > regressão**. Portanto a Approach B é um **classificador
estruturado** (a versão em árvore do H9a), não um regressor.

**Barra (definida pelo usuário): diagnóstico-primeiro, implantar só se houver
folga.** Segue a disciplina de gates: o custo caro (integração C, benchmark real)
só se paga se o diagnóstico barato mostrar que a estrutura agrega sinal.

---

## 3. Dados e construção do grafo (reuso, sem re-extração)

- **Nós.** Os nós de decisão avaliados (níveis 64/32/16; 8px excluído, como o
  H9a) de cada superbloco, obtidos de `data.iter_superblock_members` sobre os pkls
  `dataset_h9` existentes.
- **Features de nó.** As **36 features H9a** (`features.node_features_h9a`) —
  **idênticas** às do baseline independente, para isolar a contribuição da
  estrutura.
- **Rótulo de nó.** 3 classes N/S/R (`student.collapse_label`), com máscara de
  legalidade por nível (`partition_defs.legality_mask`).
- **Arestas** (derivadas das chaves `(dim, r, c)`, sem estrutura de dados nova):
  - **pai↔filho:** pai de `(dim,r,c)` é `(2·dim, r//2, c//2)`;
  - **irmão↔irmão:** nós que compartilham o mesmo pai.
  Arestas só entre nós **presentes** no superbloco (filhos ausentes são tolerados).
  Grafo minúsculo: ≤ 1+4+16 = 21 nós de decisão por superbloco.

A reconstrução da árvore reusa a lógica de `regret.py` (`_children`, mapeamento por
chave).

---

## 4. Arquitetura GNN e o teste controlado (o ponto crucial)

- **Encoder:** `h_i^0 = MLP(feat_i)` (as 36 features → dim oculta).
- **Message-passing (K rodadas, bidirecional/não-causal):**
  `h_i^{k+1} = Update(h_i^k, Agg_{j∈viz(i)} Msg(h_j^k))`, agregação por média sobre
  vizinhos (pai, filhos, irmãos). **Implementado à mão** (scatter/gather sobre
  listas de arestas) — grafos minúsculos → **sem `torch_geometric`** (nenhuma
  dependência nova no container).
- **Head:** `Linear → 3 logits` por nó + máscara de legalidade aditiva por nível.
- **Perda:** entropia cruzada de rótulo duro (a **mesma** do H9a, para comparação
  justa).

**O teste controlado (a essência do diagnóstico):** o **mesmo modelo** com
**K=0 (sem message-passing)** é, por construção, um **MLP independente por nó** —
o baseline. Comparado com **K>0 (com estrutura)**, *tudo o mais idêntico*
(arquitetura, features, dados, treino, split congelado). **Se K>0 não supera K=0, a
estrutura não agrega → o teto é informacional.** Cross-check externo contra o bundle
H9a existente (`results/models/student_h9a/students.pt`) para validade externa.

---

## 5. Gate 1-B — o marco diagnóstico

Na **validação congelada** (HoneyBee/FlowerPan/Lips), comparar **GNN (K>0) vs
gêmeo independente (K=0)** em duas métricas complementares:

1. **Qualidade de classificação por nó:** macro-F1 e **SPLIT-recall** por tamanho.
2. **Redução de custo no oráculo a SPLIT-lost casado** (a métrica-padrão da tese):
   novo modo `--gnn-bundle` no `simulate_pruning.py` que monta o grafo de cada
   superbloco a partir de `sbs["nodes"]`, roda o GNN e anexa as **probs conjuntas**
   por nó (`prob = softmax(logits)`), alimentando a **mesma** política NONE-commit
   e a **mesma** métrica de SPLIT-lost casado — *apples-to-apples* com H9a,
   variância e regret.

**Critério (calibrado no gate):** Gate 1-B **PASSA** (folga → Stage 2) se K>0
supera K=0 por margem **relevante e consistente** em SPLIT-recall **e** em redução
de custo a risco casado. Caso contrário **FALHA** → **teto informacional
confirmado**, e a Approach B fecha como **caracterização forte** (como a Solução 4),
sem pagar Stage 2.

---

## 6. Stage 2 — implantável, condicional e adiado

**Só se o Gate 1-B mostrar folga.** Um preditor estruturado **causal top-down**
(decide pai→filhos com contexto já decidido, respeitando a ordem do codificador),
implantável em C, com **spec e plano próprios** escritos **após** o Stage 1. Pelos
priors (Conclusão 3 + Solução 4), provavelmente **não é alcançado**; fica
registrado como condicional. **Nada de C, export ou benchmark no Stage 1.**

---

## 7. Cadeia de gates (metodologia)

| Gate | Conteúdo | Critério |
|---|---|---|
| **0-B** | Sanidade de grafo/dados (grafos reconstroem; features/rótulos/arestas alinham; K=0 reproduz um MLP independente) | grafos válidos; K=0 ≈ MLP por nó |
| **1-B** | Ablação **K=0 vs K>0** na validação (macro-F1, SPLIT-recall, redução de custo no oráculo a SPLIT-lost casado) + cross-check vs H9a | PASSA se estrutura agrega margem relevante; senão teto informacional |
| **(2-B+)** | *Condicional ao Stage 2* — integração C causal, paridade, no-op, benchmark real | só se Gate 1-B passar |

Ressalva metodológica herdada: o oráculo **superestima** o mérito (~5× histórico;
precedente H9c/Solução 4). No Stage 1 isso é aceitável porque a comparação é
**relativa** (K>0 vs K=0 sob o **mesmo** oráculo) — o viés cancela.

---

## 8. Entregáveis (Stage 1) e estrutura de arquivos

**Criar** (em `src/scripts/partition_model/`):
- `graph_data.py` — monta listas de nós/arestas do quadtree a partir dos
  superblocos (chaves `(dim,r,c)`); reusa `data.iter_superblock_members`,
  `features.node_features_h9a`, `student.collapse_label`.
- `gnn_model.py` — encoder + message-passing manual (K rodadas, toggle K) + head;
  K=0 = baseline independente.
- `train_gnn.py` — treina K=0 e K>0 com hiperparâmetros idênticos; salva bundles.
- `gate1_gnn.py` — ablação K=0 vs K>0 (macro-F1, SPLIT-recall) + orquestra o oráculo.
- `tests/test_graph.py` — grafo (arestas pai/filho/irmão corretas das chaves; K=0
  reproduz MLP independente; message-passing propaga como esperado num grafo de
  brinquedo).

**Modificar:**
- `simulate_pruning.py` — modo `--gnn-bundle` (`score_with_gnn`), montando o grafo
  de `sbs["nodes"]` e anexando probs conjuntas; reusa `simulate`/`metrics`/`report`.

**Docs:** `docs/RESULTADOS_approachB.md`; atualização de
`SINTESE_resultados_metodologia.md` e `ANDAMENTO_tese.md`.

**Reuso:** `regret.py` (lógica de árvore), `features`, `data`, `partition_defs`,
`student.collapse_label`, harness do `simulate_pruning`. **Sem** dependências novas.

---

## 9. Riscos e limitações (honestidade)

- **Prior forte de saturação** (Conclusão 3 + Solução 4): a estrutura pode não
  bater o independente. **Mitigação de valor:** a ablação **K=0 vs K>0** torna até
  o resultado nulo **nítido e defensável** ("a estrutura não agrega → o teto é
  informacional, não artefato de independência") — é um resultado de tese de
  qualquer forma, e barato.
- **Diagnóstico não-causal ≠ implantável:** o Stage 1 usa contexto do quadtree
  inteiro (não-causal), então é um **limite superior**; um pruner real (Stage 2)
  só tem contexto causal top-down. Por isso Stage 2 é condicional e separado.
- **Filhos ausentes / árvores parciais:** o grafo tolera arestas só entre nós
  presentes; o Gate 0-B verifica a reconstrução.
- **Comparação justa:** K=0 e K>0 devem partilhar features, dados, split e treino;
  qualquer divergência invalidaria a atribuição à estrutura (verificado no Gate 0-B).

---

## 10. Relação com as conclusões da tese

Se o Gate 1-B **falhar** (o caso provável), a Approach B **fecha** o argumento do
teto: as Soluções 1–3 mostraram que o *sinal* satura; a Solução 4 mostrou que a
*formulação de classificação* é a que melhor o extrai; e a Approach B mostra que a
*estrutura conjunta* também não abre regime novo — **o teto é informacional, não
artefato de decisões independentes**. Se **passar**, abre-se o Stage 2 (a única
rota que restaria para furar a fronteira). Em ambos os casos é o **teste de
fechamento** da investigação.
