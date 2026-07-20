# Respostas aos contra-argumentos de banca (preparação de defesa)

**Data:** 2026-07-20
**Propósito:** para cada objeção previsível da banca, a resposta medida e honesta —
concedendo onde a objeção é válida, refutando onde há base. Objeções resolvidas
com dado, não com retórica.

**Estado geral:**

| # | objeção | estado |
|---|---|---|
| CB-1 | espinha = 1 seq + 2 quadros | **respondido** (exposição offline fechada; confirmação real → E5) |
| CB-2 | "talvez o modelo seja ruim, não os pixels" | **concedido e refinado** (a afirmação de saturação não se sustenta) |
| CB-3 | "vocês têm um contraexemplo próprio" (GNN pixel-only) | **respondido** (offline-only + ganho estrutural) |
| CB-4 | "~50× num escopo que exclui o que importa" | respondido (A3, `RESULTADOS_microbench_pruner.md §6`) |
| CB-5 | "por que a Solução 4 não foi ao encoder?" | respondido (D2, `RESULTADOS_solucao4.md`) |
| CB-6 | "τ sobre probabilidades não calibradas" | respondido (A4, `RESULTADOS_calibracao.md`) |
| CB-7 | reprodutibilidade do Approach B | respondido (D10, `torch_geometric` em `requirements.txt`) |

---

## CB-1 — "A espinha dorsal da tese é uma sequência e dois quadros"

**A objeção.** A ablação que sustenta "pixels saturam na variância"
(`METODOLOGIA_pipeline_ML.md §5`, `SINTESE §…`) é Jockey, cpu-used=0, **2 quadros**.
O próprio documento pede confirmação com ≥10 quadros e outra sequência — nunca
executada. Uma afirmação central sobre 6 pontos de uma sequência.

**Resposta.** A base **offline** da hierarquia de sinais deixou de ser 1 seq / 2
quadros: o crivo do A5 (`RESULTADOS_oraculo_regret.md`) compara variância, pixels24
e H9a em **6 sequências held-out / 792.840 nós de decisão**, sob risco ponderado
por custo RD. É uma base larga e estável, não anedótica.

**O que permanece.** A confirmação no **encoder real** da ordenação, com ≥10
quadros e ≥2 sequências, é o item **E5** do plano (Bloco 7). A exposição *offline*
está fechada; a largura *no encoder* fica escopada — e é barata e defensiva.

---

## CB-2 — "Talvez o modelo seja ruim, não os pixels" *(concedido e refinado)*

**A objeção.** A afirmação "o sinal não está nos pixels / pixels saturam na
variância" repousa num estudante de pixels **fraco** (o destilado, macro-F1 0,20).
A leitura alternativa — a destilação/treino degrada o sinal, não os pixels o
limitam — foi levantada no próprio `METODOLOGIA §5` (leitura (b)) e **descartada**
sem ser testada.

**Resposta — a objeção está substancialmente certa, e a medição do A5 a expõe.**
O **mesmo** estudante de pixels (o destilado `student_real`) que **perde** para a
variância na ablação de 2 quadros **vence** a variância no crivo ponderado por
*regret* (6 seqs): `reg_frac` a `cost_red` 30% — variância **0,060** vs pixels24
**0,015** (`RESULTADOS_oraculo_regret.md §3`). É uma inversão offline↔real na dupla
variância×pixels, e o chão real dela é justamente o piloto fino de 2 quadros com o
modelo fraco.

Portanto a afirmação **"pixels saturam na variância" não está estabelecida:**
- a favor dela há só a ablação real de 2 quadros (fraca, uma sequência);
- contra ela há o crivo largo offline (pixels > variância);
- e o próprio A5 estabeleceu que nenhum dos dois é árbitro absoluto (o offline pode
  enganar; o real de 2 quadros é fino).

**A afirmação defensável, que sobrevive à objeção, é outra e mais forte:** na
métrica de custo defensável, sobre 6 sequências, **variância < pixels24 < H9a** —
o contexto RD agrega sinal **além** dos pixels, e os pixels agregam **além** da
variância. Esta é a contribuição real da virada H9, e **não** precisa da alegação
frágil de "saturação".

**Ação documental (aplicada):** a "espinha" de `SINTESE §…` que diz "o sinal não
está nos pixels" recebe nota de refino apontando para a hierarquia medida do crivo.

---

## CB-3 — "Vocês têm um contraexemplo próprio" (o GNN pixel-only)

**A objeção.** `RESULTADOS_approachB.md §4`: a GNN **pixel-only** (blocos A pixels +
C quant, **sem** o bloco B de vizinhança, 28 features) supera o H9a no oráculo em
**+20–25 pp** (65,8 vs 42 @0,5%). Offline, é essencialmente um modelo de pixels
furando o teto que a tese diz existir — uma contradição interna.

**Resposta — dissolve-se em dois pontos, ambos já medidos.**

1. **É offline, e não sobrevive ao encoder.** O mesmo GNN pixel-only perde ~2× em
   BD no replay real (`approachB §5`, Jockey), e o A5 estabeleceu que o oráculo é
   mau proxy do BD×tempo (o GNN é a exceção que prova por que o encoder é o
   árbitro). Furar o teto *offline* não é furá-lo.
2. **O ganho é ESTRUTURAL, não de pixels.** O +34 pp vem de L2-pixel (65,8) vs
   L0-pixel (31,8) com as **exatas mesmas 28 features** (`approachB §4`) — é a
   agregação por *message-passing* que extrai mais, não os pixels carregando mais
   sinal. Não é "pixels vencem o contexto RD"; é "a estrutura do grafo extrai mais
   dos mesmos pixels offline, e isso não se realiza no encoder".

Logo não há contraexemplo real à tese: nem os pixels furam o teto (é estrutura),
nem o furo se realiza (é offline). A tensão entre `approachB §4` e a narrativa de
saturação era aparente.

---

## Consequência transversal

CB-1, CB-2 e CB-3 convergem para a **mesma correção de enquadramento**, que
fortalece a tese em vez de enfraquecê-la:

> Abandonar a afirmação frágil **"o sinal não está nos pixels / pixels saturam na
> variância"** (2 quadros, modelo fraco, contradita pelo crivo e pelo GNN
> offline) e adotar a afirmação medida e larga: **na métrica de custo defensável,
> sobre 6 sequências held-out, variância < pixels24 < H9a — o contexto RD barato
> agrega sinal de decisão além dos pixels.** Essa é a contribuição real, e é
> robusta.

A confirmação no encoder real da ordenação (E5) fecha o resto.
