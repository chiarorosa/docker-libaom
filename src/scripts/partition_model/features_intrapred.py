#!/usr/bin/env python3
"""Bloco D' -- predizibilidade intra a partir dos vizinhos (2+1 atributos).

MOTIVACAO (auditoria de 2026-07-26, `docs/RESULTADOS_auditoria_dominio_pixels.md`).
O `PLANO_H9_contribuicao_tese.md:326` especificou o bloco D como o SATD do
**residuo de uma predicao intra feita a partir dos vizinhos reconstruidos**, e
advertiu explicitamente: "um residuo de predicao constante (DC) tem a mesma
variancia do bloco -- nao agrega. Por isso o proxy usa SATD sobre uma predicao
NAO-constante (direcional/PAETH). Isto mede predizibilidade, que a variancia nao
captura."

O bloco D implementado em `features.py:252` (`block_satd`) calcula o Hadamard do
**bloco-fonte**, sem predicao e sem vizinho algum. E uma estatistica so da fonte,
fortemente correlacionada com a variancia e os gradientes que o bloco A ja tem --
por isso seu resultado nulo no Gate 2 (H9b ~= H9a) era esperado e NAO informa
sobre a hipotese que o plano formulou. Este modulo implementa a hipotese original.

O QUE E NOVO, EM UMA FRASE. Todo atributo de pixel ja medido na tese (bloco A,
pixels24, variancia, ConvNeXt) le exclusivamente o bloco-fonte. Nenhum le a
**borda a partir da qual o codificador vai de fato extrapolar**. Este bloco le.

APROXIMACAO DELIBERADA (o que torna o portao barato). Os `.pkl` guardam a luma
FONTE do superbloco de 64x64, nao os vizinhos reconstruidos. Usa-se portanto a
luma-fonte das linhas/colunas adjacentes DENTRO do proprio superbloco como
substituta do vizinho reconstruido. Isso implica duas tendencias de sinal oposto,
ambas registradas no doc de resultados:

  - OTIMISTA: fonte e mais limpa que a reconstrucao (que carrega ruido de
    quantizacao). O atributo aqui e um LIMITE SUPERIOR da sua informacao real.
  - CONSERVADORA: so nos com r>0 E c>0 tem os dois vizinhos dentro do 64x64.
    No codificador real, um no na borda superior/esquerda do superbloco tem
    vizinhos vindos do superbloco adjacente. A disponibilidade offline aqui e
    portanto MENOR que em producao. Em 64px ela e ZERO (r=c=0 sempre), logo este
    portao nada diz sobre o nivel de 64.

O portao e um CRIVO, nao um veredito: se falhar como limite superior, a via fecha
sem custo de re-extracao; se passar, justifica-se reinstrumentar o codificador
para ler os vizinhos reconstruidos de verdade.
"""

import numpy as np

NUM_FEATURES_INTRAPRED = 3
INTRAPRED_FEATURE_NAMES = ["pred_avail", "log_satd_resid", "satd_gain"]


def _hadamard_satd_ac(block_i64):
    """Soma dos |coeficientes| de Walsh-Hadamard 2D, menos o termo DC.

    Mesma convencao de `features.block_satd` (DC descartado), para que
    `satd_gain = SATD(fonte) - SATD(residuo)` compare AC com AC. Sem isso, o
    ganho seria dominado pelo offset medio que a predicao remove trivialmente --
    exatamente o efeito "predicao constante" contra o qual o plano advertiu.
    """
    from features import _hadamard  # reusa o cache de matrizes de Sylvester
    b = np.asarray(block_i64, dtype=np.int64)
    n = b.shape[0]
    h = _hadamard(n)
    coeff = h @ b @ h
    return int(np.abs(coeff).sum() - abs(int(coeff[0, 0])))


def _pred_dc(above, left):
    """DC_PRED do AV1 com ambos os vizinhos disponiveis: media das 2n amostras."""
    n = above.size
    s = int(above.sum()) + int(left.sum())
    return np.full((n, n), (s + n) // (2 * n), dtype=np.int64)


def _pred_v(above, n):
    """V_PRED: replica a linha de cima para baixo."""
    return np.broadcast_to(above.reshape(1, n), (n, n)).astype(np.int64)


def _pred_h(left, n):
    """H_PRED: replica a coluna da esquerda para a direita."""
    return np.broadcast_to(left.reshape(n, 1), (n, n)).astype(np.int64)


def _pred_paeth(above, left, topleft):
    """PAETH_PRED, transliterado de `aom_dsp/intrapred.c::paeth_predictor`.

    A ordem dos testes importa e segue a do AV1: esquerda, topo, canto.
    """
    a = above.reshape(1, -1).astype(np.int64)
    l = left.reshape(-1, 1).astype(np.int64)
    tl = np.int64(topleft)
    base = a + l - tl
    p_left = np.abs(base - l)
    p_top = np.abs(base - a)
    p_tl = np.abs(base - tl)
    pred = np.where(
        (p_left <= p_top) & (p_left <= p_tl), np.broadcast_to(l, base.shape),
        np.where(p_top <= p_tl, np.broadcast_to(a, base.shape), tl))
    return pred.astype(np.int64)


def node_features_intrapred(sb_luma, dim, r, c):
    """Bloco D' para o no (dim, r, c) de um superbloco 64x64.

    Escolhe a melhor entre {DC, V, H, PAETH} por SAD (o criterio barato que o
    proprio codificador usa para triagem) e devolve:

      0  pred_avail       1.0 se os dois vizinhos estao dentro do superbloco
      1  log_satd_resid   log1p(SATD_AC do residuo da melhor predicao)
      2  satd_gain        (SATD_AC(fonte) - SATD_AC(residuo)) / (SATD_AC(fonte)+1)

    Quando os vizinhos nao estao disponiveis, devolve (0, 0, 0) -- mesma
    convencao de "ausente e neutro" do bloco B (`has_above`/`has_left`).
    `satd_gain` e adimensional e mede a FRACAO da energia AC que a predicao a
    partir dos vizinhos remove: e essa razao, e nao a magnitude, que carrega
    predizibilidade independente da textura absoluta ja medida pelo bloco A.
    """
    f = np.zeros(NUM_FEATURES_INTRAPRED, dtype=np.float32)
    if r <= 0 or c <= 0:
        return f  # sem vizinho dentro do superbloco; ver docstring do modulo

    sb = np.asarray(sb_luma).astype(np.int64)
    y0, x0 = r * dim, c * dim
    block = sb[y0:y0 + dim, x0:x0 + dim]
    above = sb[y0 - 1, x0:x0 + dim]
    left = sb[y0:y0 + dim, x0 - 1]
    topleft = int(sb[y0 - 1, x0 - 1])

    cands = (_pred_dc(above, left), _pred_v(above, dim), _pred_h(left, dim),
             _pred_paeth(above, left, topleft))
    best = min(cands, key=lambda p: int(np.abs(block - p).sum()))

    satd_src = _hadamard_satd_ac(block)
    satd_res = _hadamard_satd_ac(block - best)
    f[0] = 1.0
    f[1] = np.log1p(max(satd_res, 0))
    f[2] = (satd_src - satd_res) / float(satd_src + 1)
    return f
