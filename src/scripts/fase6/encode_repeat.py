#!/usr/bin/env python3
"""E2 — repeticoes para MEDIR o desvio-padrao do tempo de parede.

Boa parte da tese repousa em diferencas de TS de 1-3 pp (o marginal do H9d e
+1,0 pp). Ate aqui o piso de ruido era apenas INFERIDO, por violacoes de
monotonicidades que valem por construcao (~1-2% por encode). Este script mede.

Desenho: uma sequencia, os quatro CQ, tres configuracoes, cinco repeticoes.
Concentrar as repeticoes numa sequencia (em vez de espalhar tres por duas, como
o plano previa) gasta o mesmo tempo e caracteriza sigma bem melhor -- e sigma
preciso, nao sigma em duas sequencias, e o que blinda um delta de 1 pp.

As tres configuracoes cobrem os dois usos: `anchor` e o denominador de todo TS
da tese; `ml_balanced` e o numerador do tipo que importa para o H9d (ML em
cpu0); `native_cpu1` e o preset nativo com que a proposta e confrontada.

As repeticoes sao INTERCALADAS (para cada repeticao, para cada CQ, as tres
configuracoes em sequencia), espelhando a ordem em que as campanhas reais
rodaram. Rodar cinco anchors seguidos mediria a estabilidade da maquina, nao a
variancia que de fato contamina uma comparacao pareada.

Escreve num CSV proprio (results/benchmark/fase6_repeat/) com o nome da config
sufixado por _rN, para nao poluir o raw_results.csv da Fase 6.

Reproducao:
    /workspace/build/venv-ml/bin/python src/scripts/fase6/encode_repeat.py
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import encode_ctc as base  # noqa: E402

# (nome, env, cpu_used) -- os mesmos pontos ja medidos na Fase 6.
CONFIGS = [
    ("anchor", {}, 0),
    ("ml_balanced", dict(base.TAU_BALANCED), 0),
    ("native_cpu1", {}, 1),
]


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir", default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_repeat")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--seq", default="Crosswalk",
                   help="substring da sequencia (uma so)")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    done = base.load_done(csv_path)

    matches = [f for f in sorted(os.listdir(args.seq_dir))
               if f.endswith(".y4m") and args.seq in f]
    if len(matches) != 1:
        raise SystemExit("--seq deve casar exatamente uma sequencia; casou: "
                         + repr(matches))
    sf = matches[0]
    seq = os.path.join(args.seq_dir, sf)
    name = sf.split("_")[0]
    w, h, fps_num, fps_den, bd = base.parse_y4m(seq)

    total = args.reps * len(args.cqs) * len(CONFIGS)
    print("E2 repeticoes: {} ({}x{}, {}-bit) x {} cqs x {} configs x {} reps "
          "= {} encodes".format(name, w, h, bd, len(args.cqs), len(CONFIGS),
                                args.reps, total), flush=True)

    for rep in range(1, args.reps + 1):
        print("\n########## repeticao {}/{} ##########".format(rep, args.reps),
              flush=True)
        for cq in args.cqs:
            for cname, env, cpu in CONFIGS:
                tag = "{}_r{}".format(cname, rep)
                if (name, tag, cq) in done:
                    print("  cq{:>2} {:<16} [skip, done]".format(cq, tag),
                          flush=True)
                    continue
                enc = args.ml_enc if env else args.anchor_enc
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, tag, cq))
                dt, psnr_y = base.encode(enc, seq, cq, args.frames, bd, cpu,
                                         env, out_obu)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                base.append_row(csv_path, {
                    "seq": name, "config": tag, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                })
                done.add((name, tag, cq))
                print("  cq{:>2} {:<16} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, tag, dt, nbytes, psnr_y), flush=True)

    print("\nE2_REPEAT_ENCODE_DONE  (csv: {})".format(csv_path), flush=True)


if __name__ == "__main__":
    main()
