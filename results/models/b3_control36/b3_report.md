# B3 -- HORZ vs VERT: existe sinal direcional?

Experimento off-line, autocontido (`b3_horz_vert.py`), nao faz parte do pipeline implantado. Colapso de rotulos 10->4 (NONE/SPLIT/HORZ/VERT); MLP 36->64->32->4 por nivel, CE de rotulo duro, sem ponderacao de classe, treinado nas mesmas 10 sequencias de treino do estudante H9a implantado.


Pergunta decisiva: condicionado a 'o rotulo verdadeiro e retangular' (HORZ ou VERT), o modelo prediz a DIRECAO acima do acaso (50%)? Acuracia direcional condicional = argmax restrito as colunas HORZ/VERT (softmax completo, ignorando NONE/SPLIT), medida apenas nos nos cujo rotulo verdadeiro e HORZ ou VERT.


## Acuracia direcional por nivel (held-out)

| nivel (px) | n (HORZ|VERT) | prop. HORZ real | acuracia direcional | baseline 50% | baseline 'sempre HORZ' |
|---|---|---|---|---|---|
| 64 | 2707 | 53.5% | 72.8% | 50.0% | 53.5% |
| 32 | 38237 | 51.9% | 68.1% | 50.0% | 51.9% |
| 16 | 150965 | 52.6% | 69.4% | 50.0% | 52.6% |
| **agregado** | 191909 | 52.4% | 69.2% | 50.0% | 52.4% |

## Metricas secundarias por nivel


### nivel 64px

n=48000, macro-F1(4cls)=0.4529, recall(HORZ)=6.1%, recall(VERT)=2.3%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE          9869      3826        28        21
SPLIT         1593     29932        10        14
HORZ           279      1078        89         2
VERT           236       976        18        29
```
F1 por classe: NONE=0.7674, SPLIT=0.8887, HORZ=0.1117, VERT=0.0438

### nivel 32px

n=172627, macro-F1(4cls)=0.5458, recall(HORZ)=26.6%, recall(VERT)=19.1%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE         62518      7499      1555      1354
SPLIT         6303     51030      2645      1486
HORZ          7384      5591      5271      1598
VERT          7121      6286      1474      3512
```
F1 por classe: NONE=0.8002, SPLIT=0.7739, HORZ=0.3424, VERT=0.2666

### nivel 16px

n=572213, macro-F1(4cls)=0.5348, recall(HORZ)=45.4%, recall(VERT)=38.9%

Matriz de confusao (linhas=verdade, colunas=predito; ordem ['NONE', 'SPLIT', 'HORZ', 'VERT']):

```
              NONE     SPLIT      HORZ      VERT
NONE        371556      1339     12245      9814
SPLIT         8596      5887      5930      5881
HORZ         28281      1896     36049     13121
VERT         30348      1571     11863     27836
```
F1 por classe: NONE=0.8913, SPLIT=0.3183, HORZ=0.4957, VERT=0.4340

## Veredito

**POSITIVO.** A acuracia direcional condicional supera claramente tanto o acaso (50%) quanto o baseline de 'sempre escolher a classe majoritaria' em todos os niveis -- ha sinal direcional aproveitavel nas features H9a. B3 e viavel e justificaria separar prune_rect_part[HORZ] de prune_rect_part[VERT] em partition_strategy.c:1260-1261.
