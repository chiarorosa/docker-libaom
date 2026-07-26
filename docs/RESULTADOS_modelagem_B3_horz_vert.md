# Modelagem B3 — separar HORZ de VERT (existe sinal direcional?)

**Data:** 2026-07-20
**Experimento autocontido** (`b3_horz_vert.py`), fora do pipeline implantado.
**Artefato:** `results/models/student_h9a_4cls/` (MLP 4 classes por nível).
**Split:** treino nas 10 seqs; avaliação held-out (val+test, 6 seqs).

---

## 1. Pergunta

O estudante implantado colapsa 10 tipos de partição em 3 classes (NONE/SPLIT/REST),
descartando a orientação. O A8 mediu que, dentro de REST, a família HORZ vale
53–58% e a VERT 42–47% — quase balanceado, logo pode haver sinal. E o C **já
expõe a ação por direção**: `partition_strategy.c:1260-1261` usa
`prune_rect_part[HORZ]` e `[VERT]` separadamente; a política atual só sabe
desabilitar ambos de uma vez (`av1_disable_rect_partitions`).

**Pergunta decisiva (critério de falsificação do A8):** condicionado a "o rótulo
verdadeiro é retangular", um MLP de 4 classes (NONE/SPLIT/HORZ/VERT) prevê a
DIREÇÃO acima do acaso? Acurácia direcional = `argmax` restrito às colunas
HORZ/VERT, medida só nos nós cujo rótulo verdadeiro é HORZ ou VERT.

Colapso 10→4: NONE={0}, SPLIT={3}, **HORZ={1,4,5,8}**, **VERT={2,6,7,9}**.

**Reprodução:** `python src/scripts/partition_model/b3_horz_vert.py`

## 2. Resultado — o sinal direcional EXISTE

Acurácia direcional condicional, held-out:

| nível | n (HORZ+VERT) | prop. HORZ real | **acurácia direcional** | baseline "sempre HORZ" | acaso |
|---|--:|--:|--:|--:|--:|
| 64px | 2.707 | 53,5% | **72,3%** | 53,5% | 50% |
| 32px | 38.237 | 51,9% | **68,5%** | 51,9% | 50% |
| 16px | 150.965 | 52,6% | **69,1%** | 52,6% | 50% |
| **agregado** | 191.909 | 52,4% | **69,1%** | 52,4% | 50% |

**A falsificação FALHA:** a direção é aprendível — ~69% de acerto, **+16,7 pp
acima** de "chutar sempre a classe majoritária" (52,4%) e muito acima do acaso.
Robusto em todos os níveis. A hipótese do A8 (há orientação a explorar nas
features H9a) confirma-se.

## 3. As ressalvas que o número condicional esconde

O 69% é **condicional a saber que o nó é retangular**. Incondicionalmente, o
modelo é fraco em identificar retangulares — matriz de confusão em 16px
(linhas=verdade):

```
              NONE     SPLIT      HORZ      VERT
HORZ         28226      2325     34069     14727   -> recall HORZ 42,9%
VERT         29511      1978     10384     29745   -> recall VERT 41,5%
```

~35% dos nós verdadeiramente HORZ/VERT são preditos NONE. macro-F1 (4 classes)
fica em 0,45/0,55/0,54 por nível. Ou seja: **o modelo sabe a direção quando sabe
que é retangular, mas frequentemente nem identifica que é retangular.**

## 4. Veredito — o lever mais promissor, mas não é vitória grátis

**Sinal confirmado, valor de implantação ainda não provado.** Diferente do B2
(grátis), o B3 exige trabalho e validação:

1. **É a ação rect-off, secundária.** A alavanca primária é o NONE-commit; o crivo
   ponderado por *regret* (A5) mede aquela, não esta — então o benefício de BD×tempo
   do rect-off direcional **não** está medido aqui.
2. **Precisa de mudança em C.** Um estudante de 4 saídas + política que separe
   `prune_rect_part[HORZ]` de `[VERT]` (`partition_strategy.c:1260-1261`). Não é
   drop-in como o B2.
3. **A precisão sob limiar de confiança não foi medida.** Para desabilitar uma
   direção com segurança, precisa-se da precisão a alta confiança (como o A4 fez
   para NONE) — não feito. 69% incondicional é modesto; o valor real depende da
   precisão no ponto de operação confiante.
4. **Confirmação BD×tempo é do encoder.** Aqui é só classificação offline.

**Conclusão:** o B3 é o **único lever de modelagem com sinal direcional material**
(B4 e B1 negativos; B2 positivo mas imaterial). A falsificação falha — vale
perseguir. Mas cash-in exige: (a) medir precisão-vs-confiança da direção; (b)
implementar a política direcional em C; (c) confirmar em encodes. É a recomendação
para ganhos futuros, não um resultado fechado.

## 5. Limitações

- Estudante de 4 classes treinado do zero, mesmo orçamento (30 épocas), sem
  ponderação — comparável ao H9a de 3 classes, mas não otimizado.
- Acurácia direcional é **incondicional** (não thresholdada); a métrica de
  implantação seria a precisão a alta P(HORZ)/P(VERT).
- O benefício assume que preservar a direção certa no rect-off melhora BD×tempo —
  plausível mas não medido (o rect-off é uma fração do tempo de busca do nó; ver
  a alavanca C3/C1 do plano de ações).

---

## 6. Vale ir em frente? Onde a tese ganha (análise de valor — trabalho futuro)

**Decisão registrada (2026-07-20):** o B3 fica como **trabalho futuro**, possivelmente
reavaliado após os blocos em aberto (engenharia C, encodes, contra-argumentos de
banca). Esta seção documenta *onde* a tese ganharia se for perseguido, para não
reabrir a análise depois.

### 6.1 O mecanismo do ganho

Hoje a decisão sobre retangulares é binária: buscar **todas** (sem economia) ou
desligar **todas** (`av1_disable_rect_partitions` — economia, mas perde a direção
certa se ela era a ótima). O B3 abre um **terceiro eixo de decisão** que nenhuma
solução da tese tocou (todas mexem em NONE-commit e SPLIT-force): desligar **só a
direção errada**, preservando a certa — podando retangulares mais agressivamente
com menos perda de qualidade. O custo de busca retangular não é pequeno: rect+AB+
4-way são ~8/9 do trabalho *dentro* de um nó.

### 6.2 Onde a tese ganha, em ordem de solidez

1. **Completude da caracterização do espaço de ações (ganho quase certo, barato).**
   A tese cobre hoje 2 dos 3 eixos de decisão e **mediu que existe sinal no terceiro
   (69%) sem usá-lo** — lacuna que a banca aponta. Ir em frente (ou mesmo só
   documentar a medição) permite escrever que a tese examina e **ou explora ou
   descarta com medição** todos os três eixos. Vale independentemente da magnitude
   do speedup.
2. **Tradeoff rect mais granular (plausível, modesto).** Se funcionar em encodes,
   adiciona um ponto de operação que a fronteira não tem. Real, mas modesto — REST
   é minoria (22% dos nós em 16×16, 23% em 32×32, 7% em 64×64).
3. **Se falhar no encoder, ainda é ganho metodológico.** Vira mais uma instância
   medida do vão offline↔real, reforçando a espinha que o A5/Approach B construíram.
   B3-adiante é ganha-ganha: ou adiciona um lever, ou adiciona evidência.

### 6.3 Onde a tese NÃO ganha (honestidade)

- **O speedup-título não vem daqui** — vem do NONE-commit (corta subárvores) e do
  SPLIT-force. O B3 opera numa minoria de nós.
- **69% é modesto** — erraria a direção ~31% incondicionalmente; para ser seguro só
  age a alta confiança, o que limita a frequência de disparo.
- **Não está medido em BD×tempo** — o crivo do A5 cobre NONE-commit, não rect-off;
  ir em frente exige estender o avaliador offline para a ação rect **antes** dos
  encodes.

### 6.4 Custo e recomendação

Cash-in exige: (a) medir precisão-vs-confiança da direção (como o A4 para NONE);
(b) estudante de 4 saídas + política direcional em C; (c) encodes. Trabalho
substancial para um lever sobre a minoria retangular.

**Recomendação:** o ganho **mais forte e mais barato** é o (1) — completude. Mesmo
sem chegar aos encodes, a tese já pode registrar: *"o terceiro eixo de decisão
(orientação retangular) carrega sinal aprendível (69%, medido), cuja exploração
exige política direcional em C — deixada como trabalho futuro por operar sobre a
minoria retangular e depender de validação no encoder."* Isso captura a maior parte
do valor de banca a uma fração do custo. Ir até os encodes só se justifica se a
tese quiser **um número de speedup positivo adicional**, e o retorno esperado é
modesto — priorizar os blocos em aberto (CB-1/2/3, Bloco 6 C1) antes.

---

## 7. Etapa 1 pós-NONE (2026-07-26) — hipótese testada e **REFUTADA**; B3 encerrado

**A hipótese.** A §3 identificou que o 69% de acurácia direcional é **condicional a
saber que o nó é retangular**, e que o modelo é ruim exatamente nisso (recall ~42%,
~35% dos HORZ/VERT verdadeiros preditos como NONE). O ponto de enganche natural do
B3 em C seria `av1_prune_after_none`, onde a taxa/distorção/rdcost reais do
`PARTITION_NONE` já foram computados. Hipótese: **esses atributos informam
justamente a distinção que falta** — foi o mecanismo que levou o H9d de ROC-AUC
0,890 (36 atributos) para 0,902 (39).

**Método.** `b3_horz_vert.py --feature-set h9c` (39 atributos) contra um **controle
pareado** `--feature-set h9a` (36) — mesmo código, mesma sessão, mesmos dados, mesmo
split. 1,86 M nós de treino, 191.909 nós retangulares held-out.

| nível | métrica | 36 (controle) | 39 (pós-NONE) | Δ |
|---|---|--:|--:|--:|
| **agregado** | **acurácia direcional** | **69,2%** | **69,5%** | **+0,3 pp** |
| 64px | dir_acc | 72,8% | 71,9% | −0,9 pp |
| 32px | dir_acc | 68,1% | 68,8% | +0,7 pp |
| 16px | dir_acc | 69,4% | 69,7% | +0,3 pp |
| 16px | recall HORZ / VERT | 45,4% / 38,9% | 47,9% / 46,9% | +2,5 / +8,0 pp |
| 32px | recall HORZ / VERT | 26,6% / 19,1% | 30,8% / 25,8% | +4,2 / +6,7 pp |
| 64px | recall HORZ / VERT | 6,1% / 2,3% | 5,5% / 2,5% | ~0 |
| — | macro-F1 @64/32/16 | 0,453 / 0,546 / 0,535 | 0,482 / 0,594 / 0,624 | +0,03 / +0,05 / +0,09 |

### 7.1 O que os números dizem

**A acurácia direcional é plana** (+0,3 pp, mesma baseline de maioria de 52,4%). O
contexto RD pós-NONE **não acrescenta nada** à decisão de *direção*.

**Os atributos pós-NONE ajudam, mas no problema errado.** O macro-F1 sobe de forma
consistente nos três níveis e o recall médio em 16px vai de ~42% para ~47%. É ganho
real — só que o gargalo permanece: **mais da metade dos nós retangulares segue não
identificada**, e em 64px o recall é praticamente nulo (2–6%).

### 7.2 Achado colateral: variância entre execuções

O controle pareado **não reproduz** os números da execução de 20/07 (§2–3): recall
HORZ@16 42,9% → 45,4%; VERT@16 41,5% → 38,9%. Há **±2–3 pp de variância entre
execuções** nesses recalls por direção, vinda de inicialização e embaralhamento.
Consequência: o Δ de +2,5 pp no HORZ está **dentro** do ruído; o de +8,0 pp no VERT
está acima; e o **macro-F1 é o sinal confiável**. Os recalls por direção da §3 não
devem ser lidos com precisão de décimo de ponto.

### 7.3 Veredito — encerrado no portão

O critério primário estabelecido antes da execução (a acurácia condicional subir
materialmente acima dos 69%) **não foi atingido**. O secundário (recall sair de
~42%) melhorou para ~47%, insuficiente para sustentar a política: a ação do B3 é
**desligar uma direção**, e com metade dos nós daquela direção não reconhecidos, ou
se poda pouco — sem speedup — ou se paga RD demais. As duas pontas do compromisso
são ruins simultaneamente.

Some-se o que a §6.3 já registrava: o retangular é minoria do custo do nó, motivo
pelo qual o H9d foi para o eixo **estendido** e não para o retangular.

**O B3 é encerrado como negativo medido**, no mesmo padrão do H9c e do Approach B.
O portão custou ~1h de GPU em vez das ~8h de C + encodes que a Etapa 2 exigiria —
que é exatamente para isso que ele existia.

**Nota de nomenclatura.** O B3 **não** recebe rótulo da família H9. `H9b` já designa
o proxy de resíduo intra (`PLANO_H9_contribuicao_tese.md:68`,
`PROTOCOLO_avaliacao.md:68`, com números medidos em `RASTREABILIDADE.md:220`), e as
letras do H9 estão associadas a levers que entraram na cadeia de contribuição.
Registre-se, porém, que a família usa **dois esquemas de nomeação**: H9a/H9b/H9c
nomeiam *conjuntos de atributos*, enquanto H9d nomeia uma *ação* (o podador de
partição estendida, que usa o conjunto do H9c).

**Reprodução:**
```bash
/workspace/build/venv-ml/bin/python src/scripts/partition_model/b3_horz_vert.py \
    --feature-set h9c --out-dir /workspace/results/models/b3_postnone
/workspace/build/venv-ml/bin/python src/scripts/partition_model/b3_horz_vert.py \
    --feature-set h9a --out-dir /workspace/results/models/b3_control36
```
