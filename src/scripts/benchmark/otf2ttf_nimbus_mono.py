#!/usr/bin/env python3
"""Converte o Nimbus Mono PS (OTF/CFF) para TrueType, uma unica vez.

Motivo: a figura 1 do ICASSP precisa compor os tokens \\texttt do artigo na
MESMA fonte que o spconf.sty usa (NimbusMonL, clone do Courier), mas o
matplotlib, com `pdf.fonttype=42`, declara a fonte embutida como CIDFontType2 e
grava o CFF do OTF. O resultado passa no `pdffonts` com o aviso "Mismatch
between font type and embedded font file" — exatamente a classe de defeito de
embutimento que o PDF eXpress do IEEE reprova. Com a fonte ja em TrueType nao ha
divergencia entre o que o PDF declara e o que ele carrega.

Uso (dentro do conteiner, uma vez por imagem):
    build/venv-ml/bin/python src/scripts/benchmark/otf2ttf_nimbus_mono.py

Requer `apt-get install fonts-urw-base35` (o OTF de origem) e, no venv,
`pip install fonttools cu2qu`. Grava o TTF em ~/.fonts e limpa o cache do
matplotlib; o gerador da figura 1 pede a fonte pelo nome "Nimbus Mono PS TT".
"""
import os
import shutil
import sys

from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

ORIGEM = "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf"
DESTINO = os.path.expanduser("~/.fonts/NimbusMonoPS-TT-Regular.ttf")
FAMILIA = "Nimbus Mono PS TT"   # nome proprio, para nao colidir com o OTF
MAX_ERR = 1.0      # em unidades de em/1000; o padrao do fontmake


def para_quadratica(glyphset, max_err):
    saida = {}
    for nome in glyphset.keys():
        pen_tt = TTGlyphPen(glyphset)
        glyphset[nome].draw(Cu2QuPen(pen_tt, max_err))
        saida[nome] = pen_tt.glyph()
    return saida


def converter(origem, destino):
    fonte = TTFont(origem)
    if fonte.sfntVersion != "OTTO" or "CFF " not in fonte:
        sys.stderr.write("%s nao e OTF/CFF\n" % origem)
        raise SystemExit(1)

    ordem = fonte.getGlyphOrder()
    fonte["loca"] = newTable("loca")
    glyf = fonte["glyf"] = newTable("glyf")
    glyf.glyphOrder = ordem
    glyf.glyphs = para_quadratica(fonte.getGlyphSet(), MAX_ERR)
    del fonte["CFF "]
    glyf.compile(fonte)

    maxp = fonte["maxp"]
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = max(
        (len(g.components) for g in glyf.glyphs.values()
         if hasattr(g, "components")), default=0)
    maxp.compile(fonte)

    fonte["post"].formatType = 2.0
    fonte["post"].extraNames = []
    fonte["post"].mapping = {}
    fonte["post"].glyphOrder = ordem
    fonte["post"].compile(fonte)

    # Renomeia a familia. O OTF de origem continua instalado e tem o MESMO nome;
    # com dois arquivos homonimos o matplotlib escolheria um dos dois sem
    # criterio, e metade das execucoes voltaria a embutir CFF. Com nome proprio,
    # o gerador pede exatamente este arquivo.
    ps_name = FAMILIA.replace(" ", "") + "-Regular"
    for reg in fonte["name"].names:
        novo = FAMILIA if reg.nameID in (1, 4, 16) else (
            ps_name if reg.nameID == 6 else None)
        if novo is None:
            continue
        reg.string = (novo.encode("utf-16-be") if reg.platformID == 3
                      else novo.encode("latin-1"))

    fonte.sfntVersion = "\000\001\000\000"
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    fonte.save(destino)
    print("  gravado: %s" % destino)


def main():
    if not os.path.exists(ORIGEM):
        sys.stderr.write(
            "%s ausente. Rode antes, como root no conteiner:\n"
            "  apt-get install -y --no-install-recommends fonts-urw-base35\n"
            % ORIGEM)
        raise SystemExit(1)
    converter(ORIGEM, DESTINO)
    cache = os.path.expanduser("~/.cache/matplotlib")
    if os.path.isdir(cache):
        shutil.rmtree(cache)
        print("  cache do matplotlib limpo: %s" % cache)


if __name__ == "__main__":
    main()
