# H9d — cota superior do ganho (poda de partição estendida AB+4-way) — **MEDIDO E REJEITADO**

**Data:** 2026-07-20
**Bloco 6, alavanca C3 → modelo H9d.** Mede, com encodes reais nos três de teste
congelados, o **envelope** de ganho de um podador de partição estendida, ANTES de
implementá-lo. Segue o C1 (`RESULTADOS_C1_custo_por_candidato.md`), que estabeleceu que
AB+4-way são 34% do tempo local do nó.

**Veredito (§4/§6):** o teto do H9d é **dominado nos dois eixos pelo H9a já implantado** nas
mesmas sequências e regime (cpu0) — o NONE-commit do H9a já colhe esse pool melhor. **H9d
não se justifica como lever isolado.** Medir antes de construir evitou o investimento em C.
**Script:** `src/scripts/benchmark/h9d_upper_bound.py` · **Análise BD:** `bd_rate.py`
**Dados brutos:** `results/benchmark/h9d_ub/raw.csv` (não versionado)

---

## 1. O que "cota superior" significa aqui

Um H9d **seletivo** decide, por nó, se vale buscar AB/4-way — busca onde provavelmente
vencem, poda o resto. O **blanket disable** (desligar AB/4-way em *todos* os nós) é o
**extremo** desse eixo:

- **maior speedup possível** (não busca nada estendido) → **teto de aceleração**;
- **maior perda de qualidade possível** (perde AB/4-way mesmo quando eram ótimos) →
  **pior caso de BD-rate**.

Qualquer H9d real fica **dentro** deste envelope: menos speedup que o blanket, menos
BD-rate que o blanket. Portanto o blanket entrega as duas fronteiras do problema: quanto
tempo há para ganhar, e quanto custa perder tudo.

## 2. Método — isolamento por env-var no mesmo binário

- **Um único binário** `libaom_extoff` (Release, SIMD nativo, `CONFIG_INTERNAL_STATS=1`).
- A mudança em `partition_search.c` (sítios AB `:4101` e 4-way `:4160`) é **gated por
  env-var** e **inerte por padrão**:
  ```c
  static int ext_off = -1;
  if (ext_off < 0) ext_off = getenv("AV1_EXT_PART_OFF") ? 1 : 0;
  if (ext_off) ab_bsize_thresh = BLOCK_128X128;  // idem part4_bsize_thresh
  ```
  `BLOCK_128X128` faz `bsize > thresh` ser sempre falso → AB/4-way desligados em todo
  tamanho. **Sem a env, é no-op: o braço "nativo" é o encoder real**, byte-idêntico ao
  fonte não modificado. Mesma filosofia opt-in do H9a/H9c (`AV1_STUDENT_TAU_*`) — nenhum
  eixo contamina o outro.
- **Braços:** `native` (env unset) vs `extoff` (`AV1_EXT_PART_OFF=1`), **mesmo binário**,
  única diferença o gate → isolamento limpo do efeito de AB/4-way.
- **Regime:** idêntico ao `encode_ctc.py` — cpu-used=0, All-Intra, `end-usage=q`,
  kf-dist=0, deltaq=0, tpl=0, tiles=1, threads=2. **5 quadros × 4 CQ (20/32/43/55) ×
  3 seqs**. Rate p/ BD-rate = bits totais do fluxo; qualidade = PSNR-Y.

## 3. Resultado — o envelope

| sequência | **BD-rate** (extoff vs nativo) | **speedup** (teto) | tempo nativo→extoff (s) |
|---|--:|--:|--:|
| Jockey | +1,13% | **1,485×** | 636,5 → 428,7 |
| RaceNight | +0,76% | **1,418×** | 759,4 → 535,7 |
| RiverBank | +0,79% | **1,402×** | 733,3 → 523,2 |
| **média** | **+0,89%** | **1,431×** | — |

**Cruza-validação com o C1:** o C1 (offline, decomposição de tempo por candidato) previu
que remover AB+4-way tiraria ~34% do tempo local ≈ ~30% de parede → ~1,43× de speedup. O
encode real dá **1,43×**. As duas medições independentes concordam — o custo medido no
`part_timing` é real e se realiza em tempo de parede.

### A leitura decisiva: densidade de qualidade baixíssima

AB+4-way custam **~43% do tempo de parede** mas carregam apenas **+0,89% de BD-rate** de
qualidade. Isto é, gastam muito e decidem pouco:

> **~0,03 pp de BD-rate por 1% de tempo economizado** (0,89% ÷ ~30%).

É exatamente o perfil que favoreceria um podador: candidatos caros e raramente decisivos.
**Mas — e este é o ponto decisivo — o H9a implantado já poda esse mesmo pool, e melhor.**

## 3.6 Comparação com o H9a implantado — o teto do H9d é DOMINADO

O H9a foi medido nos **mesmos três de teste, mesmo cpu-used=0, mesmo anchor nativo**
(`results/benchmark/h9_test/*/curve_{safe,aggr}/summary.csv`). Colocando os pontos de
operação do H9a lado a lado com o teto do H9d (todos vs nativo, cpu0):

| seq | H9a P_ref (safe) | H9a A1_none80 (aggr) | H9a A2_none70 (aggr+) | **H9d blanket (teto)** |
|---|--:|--:|--:|--:|
| Jockey | 0,919% / 1,483× | 0,929% / **1,503×** | 1,372% / 1,911× | 1,13% / 1,485× |
| RaceNight | 0,736% / 1,467× | 0,729% / **1,484×** | 1,165% / 1,715× | 0,76% / 1,418× |
| RiverBank | 0,129% / 1,319× | 0,19% / 1,328× | 0,226% / **1,434×** | 0,79% / 1,402× |

*(formato: BD-rate% / speedup; frames: H9a ~15, H9d 5 — os números do H9a são os mais
robustos, e são justamente os que dominam.)*

**Em cada sequência existe um ponto de operação do H9a que domina o teto do H9d nos dois
eixos** (mais speedup E menos BD-rate):
- **Jockey**: H9a aggr 1,503×/0,93% > H9d 1,485×/1,13%.
- **RaceNight**: H9a aggr 1,484×/0,729% > H9d 1,418×/0,76%.
- **RiverBank**: H9a A2_none70 1,434×/0,226% > H9d 1,402×/0,79% (mais rápido e ~⅓ do BD).

**Por quê (estrutural):** o NONE-commit do H9a, ao comprometer PARTITION_NONE num nó, **já
pula AB/4-way — e também rect, split e toda a subárvore**. Recupera o pool de 43% de forma
*mais eficiente* que um podador dedicado a AB/4-way, porque corta mais por decisão. O
envelope que o H9d persegue **já está capturado, e superado, pelo modelo em produção.**

## 4. Veredito revisado — H9d não se justifica como lever isolado

A cota superior, tomada sozinha, parecia atrativa (1,43× por 0,89% BD). A comparação com o
H9a **cpu0 nas mesmas sequências** desfaz isso: **o teto do H9d é dominado pelo H9a já
implantado em todas as três de teste.** Consequências:

1. **Como lever isolado, H9d é dispensável.** Seu melhor caso possível (blanket) perde para
   um τ mais agressivo do H9a, que é grátis (variável de ambiente, já implementado).
2. **O único resíduo é o ganho MARGINAL sobre o H9a operante** — nós onde o H9a *não*
   NONE-commita mas AB/4-way ainda são buscados. Esse pool é **menor** que o blanket (parte
   dele já foi cortada pelo H9a), e como o teto isolado já perde, é improvável que o marginal
   supere simplesmente subir o τ. Medir `extoff+H9a` vs `H9a` fecharia a questão a custo
   baixo (mesmo binário, ligar as duas envs), mas o *a priori* agora é negativo.
3. **Se ainda assim perseguido, precisaria de:** seletividade a alta confiança (features H9a
   por nó, à la A4) + política em C (`ab_bsize_thresh`/`part4_bsize_thresh` como função da
   predição, não env global) + confirmação BD×tempo (A5). Investimento alto para um resíduo
   provavelmente pequeno.

## 5. Limitações

- **Blanket ≠ seletivo.** Este é o **envelope**, não o ganho do H9d. O H9d real fica
  dentro; seu ponto exato depende da predizibilidade de AB/4-way (item 4.1), ainda não
  medida. Não é lícito anunciar "1,43×" como ganho do H9d — é o teto.
- **Comparação H9a↔H9d é do mesmo regime.** Ambos cpu-used=0, mesmas 3 seqs, mesmo anchor
  nativo — a dominância de §3.6 é apples-to-apples. (Difere de `RESULTADOS_fase6_swap_h9c.md`,
  que é o swap H9c×CNN em cpu1-3; não confundir.) A única assimetria é o número de quadros
  (H9a ~15, H9d 5) — e favorece o H9a como referência mais robusta, então a dominância é
  conservadora.
- **5 quadros por ponto, 4 CQ (H9d).** Suficiente para um envelope estável (BD-rate
  consistente, 0,76–1,13% nas três seqs), mas não é a rigidez de uma medição de fronteira
  final. É uma sonda de dimensionamento — e como o teto já perde para o H9a, mais quadros só
  reforçariam o veredito.
- **PSNR-Y apenas.** BD-rate sobre PSNR-Y (padrão da tese); não sobre PSNR combinado/SSIM.

## 6. Veredito

**H9d NÃO é uma alavanca justificada — seu teto (blanket AB/4-way off, 1,43× por 0,89% BD)
é dominado nos dois eixos pelo H9a já implantado, nas mesmas três sequências e mesmo
cpu-used=0** (§3.6). O NONE-commit do H9a recupera o mesmo pool de custo de forma mais
eficiente porque corta a subárvore inteira, não só o estendido. O valor da medição foi
**negativo e valioso**: mediu antes de construir, e evitou investir esforço de C num lever
cujo melhor caso possível é inferior a subir o τ do H9a (grátis).

Resíduo possível (ganho marginal de `extoff` **sobre** o H9a operante) fica registrado como
verificação barata opcional, com *a priori* negativo. Este é mais um dado da espinha
metodológica da tese: **o eixo primário (NONE-commit) já colhe o custo dos eixos
secundários** — a caracterização dos três eixos (C1) mais a medição do teto (aqui) fecham a
pergunta "vale um segundo modelo para partição estendida?" com **não**.

## 7. Reprodução

```bash
# build com o gate (uma vez) — a mudanca e inerte sem a env
cmake -S /workspace/src/aom -B /workspace/build/libaom_extoff -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCONFIG_INTERNAL_STATS=1 -DENABLE_TESTS=OFF
cmake --build /workspace/build/libaom_extoff -j"$(nproc)"

# matriz nativo vs extoff (env alternada dentro do script)
/workspace/build/venv-ml/bin/python \
  src/scripts/benchmark/h9d_upper_bound.py --frames 5 \
  --cqs 20 32 43 55 --seqs Jockey RaceNight RiverBank
```
