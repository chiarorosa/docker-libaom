# Andamento da tese — status e próximo passo

**Documento vivo.** Onde a tese está, o que foi decidido, e o próximo passo
concreto. Atualizado em 2026-07-10 (branch `ml-partition-dev`). Índice de
artefatos: `RASTREABILIDADE.md`. Planos: `PLANO_hipoteses_experimentos.md`,
`PLANO_H9_contribuicao_tese.md`. Protocolo congelado: `PROTOCOLO_avaliacao.md`.

---

## 1. Arco da tese (o que estabelecemos, em ordem)

1. **Infraestrutura** — instrumentação C do libaom, extração de dataset, ConvNeXt
   substituto, destilação → estudante MLP embarcado (`av1_nn_predict`), harness de
   benchmark (taxa BD + tempo). *(H1–E, concluído.)*
2. **Exploração H1–H6** — sinal da luminância parecia limitado (teto ~13–18%).
   *(Depois invalidado — ver item 4.)*
3. **Pré-H7** — descoberta de que a alavanca não era o modelo, e sim o **espaço de
   ações da política** (poda de retangulares via P(REST)) e a **métrica** (custo,
   não nós). Implementado.
4. **Bug crítico da luma-em-branco** — o dataset guardava luma `float32 [0,1]` e os
   consumidores assumiam `uint8`, treinando os modelos **sobre imagem em branco**.
   Corrigido; toda a cadeia re-treinada sobre pixels reais. **Invalidou as
   conclusões H1–H6** (eram sobre entrada vazia).
5. **H7/H8 (dado limpo)** — curva de operação real: **~7–29% de speedup a 0,4–1,6%
   de taxa BD**; teto do substituto (H8) ≈ de graça (−0,11% BD).
6. **Ablação de atribuição** — resultado **negativo**: no lever NONE-commit, um
   limiar de variância trivial **empata/supera** o estudante de pixels. Ou seja,
   os pixels saturam na variância; o ganho não era atribuível ao ML.
7. **Pivô H9** — hipótese: **contexto de taxa-distorção barato** supera o teto de
   pixels. Protocolo congelado (Fase 0), instrumentação e re-extração (Fase 1),
   **Gate 2 offline PASSOU** (Fase 2): o contexto RD grátis (H9a) supera pixels
   ~50% relativo no lever NONE-commit em risco casado.

**Tese, em uma frase (estado atual):** demonstra-se que o particionamento
All-Intra satura, no domínio de pixels, numa estatística trivial (variância) — por
ablação rigorosa — e que **contexto de taxa-distorção barato (vizinhança + quant +
posição) é necessário e suficiente para superar esse teto**, produzindo uma poda
aprendida com ganho de tempo atribuível ao ML. Falta confirmar no encoder real
(Gate 5).

### 1.1 Por que o `dataset_h9` existe — de um resultado negativo forte a uma direção positiva

O `dataset_h9` não é "mais um dataset": é o experimento que converte um resultado
negativo numa contribuição positiva. A sequência:

1. **Resultado negativo (ablação de atribuição).** No domínio de pixels, um
   baseline trivial — apenas a variância do bloco — empata ou supera o modelo de
   ML em todo ponto de speedup comparável. Ou seja, o ganho **não era atribuível
   ao aprendizado**: nenhum dos 24 atributos de pixel fazia o modelo bater a
   variância isolada.
2. **Por que esse negativo tem valor.** Ele caracteriza, com rigor, que a
   informação de particionamento **presente nos pixels satura numa estatística
   trivial**. É um resultado negativo forte: mostra, com evidência, que aumentar a
   capacidade do modelo no domínio de pixels não leva a lugar nenhum.
3. **A virada positiva (hipótese H9).** O particionamento do AV1 é, por definição,
   uma decisão de **taxa-distorção (RD)**. Logo, o sinal capaz de elevar o limite é
   o **contexto RD** que a própria decisão usa — e boa parte dele é **barata e já
   está residente na memória do encoder**: tamanho dos blocos vizinhos (acima/
   esquerda), força de quantização (qindex) e posição no quadro. É informação que
   **não está nos pixels**.
4. **`dataset_h9` materializa a hipótese.** Re-extração das 16 sequências
   adicionando os campos B (vizinhança), C (quantização/posição) e E (custo RD real
   do `PARTITION_NONE`, registrado apenas como referência de limite superior).
5. **Evidência positiva até aqui (Gate 2, offline).** MLPs por tamanho de bloco
   sobre o conjunto **H9a** (pixels + vizinhança + quant/posição, tudo de custo
   zero) superam a variância em ~50% relativo, de forma consistente. A confirmação
   definitiva no encoder real (Gate 5) permanece pendente.

### 1.2 O que roda no encoder — o professor (ConvNeXt) nunca foi a heurística

Há dois artefatos com papéis distintos, e é fácil confundi-los:

- **ConvNeXt (o "substituto"/professor).** Modelo convolucional de ~107 MB que
  observa o superbloco 64×64 inteiro. Usado **apenas em tempo de treino** — como
  professor numa **destilação de conhecimento** — e como **referência de limite
  superior** (experimento H8: quanto um modelo convolucional forte alcançaria,
  medido por replay no encoder). É pesado demais para inferência a cada nó de
  particionamento; **nunca foi embarcado no encoder**.
- **Estudante (o que de fato roda).** Uma **MLP pequena por tamanho de bloco**,
  executada pela função nativa `av1_nn_predict` do libaom sobre atributos manuais
  do bloco. É esse artefato que produz o melhor resultado atual — a curva de
  **~7–29% de speedup a 0,4–1,6% de BD-rate** (`h7h8_real`). O ConvNeXt destilou
  conhecimento para ele em treino, mas não participa da inferência.

**Consequência para o enquadramento da tese.** O objetivo original ("aplicar
convoluções para reduzir o tempo de codificação") realizou-se, na prática, como
convoluções no papel de **professor + referência de limite superior**, não como o
mecanismo embarcado. Mais: a ablação de atribuição e o Gate 2 indicam que o **ganho
implantável é tabular** (atributos manuais + contexto RD barato), **não
convolucional**. Por isso a Fase 3 recomendada treina o estudante diretamente sobre
H9a, em vez de reintroduzir o ConvNeXt.

### 1.3 O papel do ConvNeXt na escrita da tese (e a tensão do objetivo declarado)

O trabalho com o ConvNeXt não é descartável — é peça central da narrativa
metodológica, mesmo não estando no pruner embarcado.

**Onde ele entra na escrita:**
- **Metodologia:** a arquitetura do substituto ConvNeXt multinível (luma+qindex,
  cabeças por nível espelhando a hierarquia de particionamento do AV1), a
  destilação de conhecimento substituto → estudante, e a infraestrutura de
  avaliação (instrumentação em C, extração do ground truth em cpu-used=0, harness
  de BD-rate + speedup, simulação oráculo, ablação de atribuição).
- **Resultados/Discussão:** a correção do bug da luma (por que H1–H6 foram
  refeitos), a curva operacional H7/H8, e a ablação de atribuição — o **resultado
  negativo** (a variância trivial empata/supera o ML em speedup comparável).

**Por que o ConvNeXt é indispensável, não opcional:** ele é o que dá
**credibilidade ao resultado negativo**. A afirmação "os pixels saturam na
variância" só é rigorosa porque o modelo de pixels testado **não era fraco**. Se
houvesse apenas uma MLP pequena falhando em superar a variância, caberia a objeção
"o modelo tinha capacidade insuficiente". Ter um modelo convolucional de alta
capacidade, com o experimento de limite superior (H8), e ainda assim **não separar
da variância** em speedup comparável, fecha essa objeção — transforma "o modelo não
conseguiu" em "**o domínio de pixels não contém o sinal**".

**Elo com a solução final:** é esse aprendizado que **motiva e justifica** a virada
para o contexto RD (H9a). Sem o ConvNeXt, a virada seria um palpite; com ele, é uma
conclusão baseada em evidência. A técnica de "medir o teto para saber quanto ainda
resta" também migra para o lado RD: o H9c (`none_rdcost`) é o análogo do que o
ConvNeXt/H8 foi para os pixels.

**Tensão do objetivo declarado (decisão de narrativa a tomar).** Se o objetivo da
tese permanecer "aplicar convoluções para reduzir o tempo de codificação", há uma
tensão: a **solução implantada não é convolucional**. O enquadramento mais fiel aos
dados é: *caracterizar com rigor o que a decisão de particionamento precisa — e o
que ela não precisa (mais capacidade no domínio de pixels) — e entregar um ganho de
tempo atribuível a partir de contexto RD barato*. Nesse enquadramento, as
convoluções são o **instrumento de diagnóstico e a referência de limite superior**,
protagonistas da investigação, não do mecanismo embarcado. Recomenda-se revisar
título/resumo/objetivos com essa informação explícita.

---

## 2. Status por fase

| Fase | Descrição | Estado | Evidência |
|---|---|---|---|
| Infra (H1–E) | instrumentação, dataset, surrogate, destilação, benchmark | ✅ | commits até `71097b1` |
| Pré-H7 | poda de rect + métrica de custo + τ por nível + contexto hierárquico | ✅ | `834639e`, `PREH7_analise_alavancas.md` |
| Bug luma-branco | diagnóstico + correção + guarda | ✅ | `cb9d407`, `63f299c` |
| H7/H8 | curva speedup/BD real + teto do substituto | ✅ | `172e669`, `benchmark/h7h8_real*` |
| Ablação atribuição | ml vs variância vs aleatório (resultado negativo) | ✅ | `2380e91`, `benchmark/ablation_matched.csv` |
| H9 Fase 0 | protocolo congelado (test/val/train, QPs, métricas) | ✅ | `aabbbee`, `PROTOCOLO_avaliacao.md` |
| H9 Fase 1 | instrumentação RD (B/C/E) + re-extração 16 seqs | ✅ | `a17c525`, `a6748c5`, `dataset_h9/` (64 pkl) |
| H9 Fase 2 | Gate 2 offline (sinal do contexto RD) | ✅ **PASSOU** | `8866757`, `models/gate2_final.csv` |
| H9 Fase 3 | estudante tabular direto sobre H9a; Gate 3 (val) | ✅ **PASSOU** | `173aa8f`, `models/student_h9a/` |
| H9 Fase 4 | features B/C em C; paridade C↔Python; no-op byte-idêntico | ✅ **PASSOU** | `b3cd3c1`..`ecf436b`, `models/student_h9a/gate4_evidence.txt` |
| H9 Fase 5 | benchmark no teste held-out + ablação de atribuição | ✅ **CONCLUÍDA** (veredito matizado) | `docs/RESULTADOS_fase5.md`, `benchmark/h9_test/` |

Legenda: ✅ concluído · ⏳ próximo · 🔄 em andamento · ⬜ pendente.

---

## 3. Veredito do Gate 2 (base da decisão atual)

`models/gate2_final.csv` (10 seqs treino, 60k superblocos, ensemble). No lever
**NONE-commit** (relevante para tempo), redução de custo em risco casado de
SPLIT-lost {0,5/1/2}%:

| subset | 0,5% | 1% | 2% |
|---|---:|---:|---:|
| variância | 0 | 0 | 0 |
| pixels24 | 10,1 | 15,3 | 18,9 |
| **H9a (contexto RD grátis)** | **15,7** | **20,1** | **24,9** |
| H9c (teto, none_rdcost) | 33,0 | 33,0 | 39,7 |

**Decisão: cenário (a)** — seguir com o modelo **pré-busca H9a** (pixels +
vizinhança + quant + posição; SATD do bloco D não agrega, descartado). O teto H9c
(`none_rdcost`, pós-NONE) fica documentado como headroom para uma extensão futura
(poda pós-NONE aprendida).

**Ressalva carregada até o fim:** a simulação oráculo superestima o tempo real
(~5×: 35% custo → 7% wall-clock). O que vale é a **margem relativa** de ~50% do
H9a sobre pixels; o árbitro final é o **Gate 5** (benchmark de tempo real no
teste held-out).

---

## 4. Fases 3–4 CONCLUÍDAS e próximo passo (Fase 5)

**O que foi feito (decisão pela evidência).** A questão em aberto — ConvNeXt+ramo
RD vs tabular — foi resolvida pelo dado: como o ganho H9a é **tabular e não-pixel**,
o ConvNeXt pixel-only não pode ensiná-lo, e o Gate 2 já provou a MLP-por-tamanho.
Logo, treinou-se o **estudante implantável diretamente** sobre H9a (36 features =
A pixels + B vizinhança + C quant/pos), CE de rótulo duro, **sem professor**, na
partição congelada. O ConvNeXt (`surrogate_real`) permanece como teto H8 de pixels;
o H9c segue como teto RD. Planos: `docs/superpowers/{specs,plans}/2026-07-11-*`.

**Gate 3 (validação HoneyBee/FlowerPan/Lips), alavanca NONE-commit, risco casado
(SPLIT-lost ≤ 1%), redução de custo do oráculo:**

| modelo | cost_red% @ SL≤1% | menor SPLIT-lost alcançável |
|---|---:|---:|
| variância | ~0 | 22,6% (não opera em risco baixo) |
| pixels24 | ~26–35 | ~0,3% |
| **H9a** | **~55–58** | 0,19% |

O **H9a dobra os pixels e domina a variância** (que nem chega a risco baixo),
reforçando o Gate 2. SPLIT-recall (métrica de segurança) 0,98/0,82 em 64/32px vs
0,73/0,52 dos pixels. **Ressalva mantida:** o oráculo superestima o tempo real
(~5×); vale a margem relativa; o **Gate 5** (encoder real) é o árbitro.
Artefatos: `results/models/student_h9a/` (`students.pt`, `oracle_sim_*.csv`, header
de 36 features exportado — o `student_real` implantado ficou intocado).

**Fase 4 — CONCLUÍDA (integração em C).** As features B/C foram sincronizadas em
`student_node_features` (`partition_strategy.c`), espelhando `node_features_h9a`;
o estudante H9a (36 features) é agora o **implantado**. Gate 4 passou:
- **paridade C↔Python** verde — 36/36 features a `0.0e+00`, probs <1e-3 (588 nós);
- **no-op byte-idêntico** ao `aom_baseline` com a flag desligada (md5 igual, em
  quadro texturizado);
- decode válido + testes AllIntra. Evidência: `models/student_h9a/gate4_evidence.txt`.

Ressalva (do review final): a paridade prova a **aritmética** das features; o
**sourcing** do ctx (vizinhança/quant lidos em runtime) é validado por revisão de
código contra o idioma nativo e, fim-a-fim, pelo benchmark da Fase 5.

**PRÓXIMO PASSO (Fase 5 — benchmark de tese).** No conjunto de **teste** held-out
(Jockey/RaceNight/RiverBank, ≥10 quadros): curva taxa BD × speedup do estudante H9a
com `libaom_perf` (rebuild com o código de 36 features) vs `libaom_perf_anchor`;
**ablação de atribuição** em speedup casado (H9a vs pixels vs variância vs aleatório,
via `ablation_attrib.py`/`analyze_ablation.py`); comparação com o
`intra_cnn_based_part_prune` nativo. Gate 5 (sucesso da tese): H9a domina a variância
em taxa BD a speedup casado, por margem além do ruído, em ≥2 das 3 seqs de teste.

---

## 4.1 Fase 5 — resultados PARCIAIS (em andamento, 2/3 seqs)

**Estado:** benchmark real detached rodando; Jockey e RaceNight concluídos,
RiverBank em execução. `RESULTADOS_fase5.md` (final) ainda não escrito. Resultados
em `results/benchmark/h9_test/<seq>/{curve_safe,curve_aggr,ablation}`. Encoder de
teste `libaom_perf` (36 features, flag on) vs âncora `libaom_perf_anchor` (libaom
cru); cpu-used=0, single-thread, 10 quadros, cq {20,32,43,55}.

**Calibração (validação, HoneyBee) — grid de τ:** com grid alargado
`{0.95..0.50}`, ml e variância **sobrepuseram** e o ml **dominou** (a tempo casado:
@1,5× ml 0,66% vs var 3,53%; @1,75× 1,41% vs 4,92%). Congelou o grid.

**Pilar 1 — redução de tempo (curva H9a vs baseline): FORTE e consistente.**

| seq | melhor ponto barato | ponto agressivo |
|---|---|---|
| Jockey | 0,19% BD @ TS 21,6% (1,28×) | 2,03% BD @ TS 57,2% (2,34×) |
| RaceNight | 0,27% BD @ TS 18,1% (1,22×) | 1,72% BD @ TS 50,7% (2,03×) |

**Pilar 2 — atribuição ao aprendizado (ml vs aleatório): CLARO.** A tempo casado,
o ml domina o aleatório nas duas seqs (ex. RaceNight @1,3×: ml 0,90% vs random
6,01%). O ganho é do modelo, não de poda aleatória.

**Pilar 3 — vs variância (a barra difícil): SEM head-to-head limpo no teste.**
Em Jockey e RaceNight (2/2), as faixas de speedup de ml e variância **NÃO se
sobrepõem**: ml opera em baixo speedup e baixo BD (≤1,46–1,59×, <1,7% BD); a
variância só opera agressivo (≥1,89–2,13×, ≥1,76–3,96% BD). São **regimes
disjuntos** — a variância, mesmo no ajuste mais conservador (τ=0,95), já pula para
alto speedup/alto BD e **não alcança a região implantável de baixo-BD** onde o ml
vive.

- **Leitura honesta:** a dominação limpa "ml vence a variância a tempo casado" só
  se materializou na **validação** (onde o grid sobrepôs). No **teste**, o resultado
  é diferente e mais matizado: ml e variância servem a operating points distintos;
  o ml ocupa a fronteira controlável e barata que a variância grosseira não
  acessa; a variância domina apenas em regime agressivo (que custa BD alto).
- **Causa da não-sobreposição (nota metodológica):** o grid foi congelado na
  calibração do HoneyBee, onde a variância τ=0,95 dava 1,34× (sobrepunha). O
  conteúdo de teste poda mais agressivo sob variância, então τ=0,95 já dá 1,9–2,1×.
  Faltou o lado da variância descer a τ=0,97/0,99. **Não estendido** — seria mexer
  em config vendo dado de teste (viola o congelamento anti-cherry-picking). Fica
  como limitação documentada / trabalho futuro.

**Cruzamento a BD casado (dual do speedup casado).** Como as faixas não sobrepõem
em speedup, testou-se o eixo BD (mesma qualidade → quem economiza mais tempo).
Resultado (`results/benchmark/matched_bd.py`): com o ml **implantado (rect-off)**,
há sobreposição em BD só no **Jockey** (1,76–2,03%), onde o ml ganha — @BD 1,76%:
ml **2,17× (TS 53,4%)** vs variância 1,89× (TS 47%). No **RaceNight** é disjunto até
em BD (variância mínima 3,96% > ml máx 1,72%). Ressalva: usa a política rica do ml
(rect-off) contra a variância sem — é comparação **sistema-vs-sistema**, não
atribuição pura.

**Atribuição LIMPA (política casada) — o argumento mais forte.** O rect-off é ação
**secundária** do ml (3ª saída, P(REST)); a alavanca **primária** é o NONE-commit.
A variância **estruturalmente não faz rect-off** (fixa P(REST)=1). Logo, a
comparação justa usa o **modo primário do ml (NONE-commit puro) vs variância,
política idêntica** — e aí há uma afirmação de atribuição **pura** que dispensa
sobreposição de speedup:

> Sob a **mesma política** (NONE-commit, sem rect-off para ninguém), variando só a
> **fonte de escore**, o escore do ml **alcança pontos de baixo-BD que o da
> variância não alcança**: ml de **0,09–0,10%** BD para cima; variância **nunca
> abaixo de 1,76% (Jockey) / 3,96% (RaceNight)**. Tudo idêntico exceto o escore →
> o ganho é do modelo. O escore RD é **mais discriminativo** (commita NONE só
> quando é o caso, BD ~0%); a variância commita em qualquer bloco liso, inclusive
> nos que deveriam dividir (daí o piso de BD alto).

Isto é matched-*policy* e caracteriza a **região alcançável** — não precisa de
sobreposição de speedup, e vale nas 2 seqs de teste. **O valor central do ml não
depende do rect-off.**

**FULL CONCLUÍDO (3/3 seqs) — 2026-07-13. Ver `docs/RESULTADOS_fase5.md`.**
RiverBank confirmou o padrão: sem sobreposição de speedup (3/3), mas o escore do
ml alcança BD mínimo 0,008% vs 0,75% da variância (razão 94×; Jockey 11×,
RaceNight 44×). Veredito Gate 5: a **forma estrita** ("ml domina variância a
speedup casado em ≥2/3") **não** é atingida no teste por não-sobreposição; mas a
contribuição está sustentada — (i) redução de tempo forte vs baseline (TS ~30–48%
a 0,6–1,4% BD, 3/3); (ii) ml ≫ aleatório (3/3); (iii) atribuição a política casada
(escore do ml alcança baixo-BD inacessível à variância, 3/3); (iv) validação com
dominação direta. Limitações + trabalho futuro (grid da variância a τ=0,97/0,99 na
validação; variância com rect-off; SOTA nativo) em `RESULTADOS_fase5.md` §5.

**Implicação para a escrita:** o Gate 5, como enquadrado ("ml domina variância a
speedup casado em ≥2/3 seqs"), **não** é atingido no teste por não-sobreposição.
Mas a tese afirma com honestidade e rigor: (i) redução de tempo forte vs baseline;
(ii) ganho atribuível ao aprendizado (vs aleatório); (iii) **atribuição a política
casada** — o escore do ml alcança a fronteira implantável de baixo-BD que o da
variância é incapaz de tocar (a barra difícil, agora respondida de forma limpa);
(iv) na validação, dominação direta a speedup casado. RiverBank (3ª seq) pode ou
não alterar o quadro; o veredito final sai quando o full fechar.

---

## 5. Decisões em aberto

- ~~**Arquitetura do substituto H9a** (ConvNeXt+ramo RD vs tabular)~~ — **RESOLVIDO
  na Fase 3**: estudante tabular direto sobre H9a (sem professor); ver §4.
- **Publicação do dataset** — `docs/ZENODO_datasheet.md` pronto; conversão para
  `.npz` uint8 (`pkl_to_npz.py`) **não executada** (backup bruto bin+pkl indo para
  Google Drive primeiro).
- **Extensão pós-NONE (H9c)** — opcional, documentada como headroom; só se o
  Gate 5 do H9a pré-busca ficar aquém.

---

## 6. Riscos vivos

- Gate 5 pode não confirmar a margem do Gate 2 (oráculo superestima) → mitigação:
  a decisão final é sempre o tempo de parede; se a margem não sobreviver, a
  contribuição recai na caracterização do teto informacional + o estudo H9c.
- Custo de inferência das features B/C é ~zero (dados residentes), então não há
  risco de "a poda não se pagar" no regime H9a (grátis).
