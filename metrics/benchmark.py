import argparse
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BOOKS_DIR = ROOT / "data" / "books"
SAMPLES_DIR = ROOT / "data" / "samples"
BUILD_DIR = ROOT / "build"
RESULTS_DIR = ROOT / "results"

ALGOS = {
    "LZ77": REPO / "lz77" / "LZ77.c",
    "Huffman": REPO / "huffman" / "huffman.c",
    "LZW": REPO / "lzw" / "lzw.c",
}
COLORS = {"LZ77": "tab:blue", "Huffman": "tab:orange", "LZW": "tab:green"}
MIB = 1 << 20
PHYSICAL_CORES = 4

FULL = dict(threads=[1, 2, 4, 6, 8], sizes=[1, 2, 4, 8, 16, 32], main=32,
            reps_scaling=5, reps_size=3)
QUICK = dict(threads=[1, 2], sizes=[1, 2], main=2,
             reps_scaling=1, reps_size=1)

ENV = os.environ.copy()
TIME_RE = re.compile(r"(?:compute|compress|decompress)=([0-9.]+) s")
EXE_SUFFIX = ".exe" if os.name == "nt" else ""


# ---------------------------------------------------------------- сборка
def find_gcc():
    path = shutil.which("gcc")
    if path:
        return path
    candidates = [
        r"C:\Program Files\CodeBlocks\MinGW\bin\gcc.exe",
        r"C:\Program Files (x86)\CodeBlocks\MinGW\bin\gcc.exe",
        r"C:\MinGW\bin\gcc.exe",
        r"C:\msys64\ucrt64\bin\gcc.exe",
        r"C:\msys64\mingw64\bin\gcc.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    sys.exit("gcc не найден. Добавьте папку MinGW\\bin (например из CodeBlocks) в PATH.")


def build_all():
    gcc = find_gcc()
    ENV["PATH"] = str(Path(gcc).parent) + os.pathsep + ENV["PATH"]
    print(f"gcc: {gcc}")
    BUILD_DIR.mkdir(exist_ok=True)
    for name, src_path in ALGOS.items():
        if not src_path.exists():
            sys.exit(f"Нет файла {src_path}")
        exe = BUILD_DIR / f"{name}{EXE_SUFFIX}"
        r = subprocess.run([gcc, "-O2", "-fopenmp", "-o", str(exe), str(src_path)],
                           capture_output=True, text=True, env=ENV,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.exit(f"Ошибка компиляции {src_path.name}:\n{r.stderr}")
        print(f"собрано: {exe.name}")


# ---------------------------------------------------------------- данные
def make_samples(sizes, main):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    books = sorted(BOOKS_DIR.glob("*.txt"))
    if not books:
        sys.exit(f"Положите .txt файлы с книгами в {BOOKS_DIR}")
    corpus = b"\n".join(b.read_bytes() for b in books)
    print(f"книг: {len(books)}, суммарно {len(corpus) / MIB:.1f} МБ")
    for s in sizes:
        n = s * MIB
        reps = -(-n // len(corpus))
        (SAMPLES_DIR / f"text_{s}MB.txt").write_bytes((corpus * reps)[:n])
    (SAMPLES_DIR / f"random_{main}MB.bin").write_bytes(os.urandom(main * MIB))


# ---------------------------------------------------------------- замеры
def run_prog(exe, args):
    r = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True,
                       env=ENV, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{exe.name} {args} завершилась с кодом {r.returncode}:\n{r.stderr}")
    m = TIME_RE.search(r.stderr)
    if not m:
        raise RuntimeError(f"Не удалось найти время в выводе:\n{r.stderr}")
    return float(m.group(1))


def measure(name, in_path, threads, reps):
    exe = BUILD_DIR / f"{name}{EXE_SUFFIX}"
    tmp_c = BUILD_DIR / "tmp.cmp"
    tmp_d = BUILD_DIR / "tmp.out"
    ct, dt = [], []
    for _ in range(reps):
        ct.append(run_prog(exe, ["c", in_path, tmp_c, threads]))
        dt.append(run_prog(exe, ["d", tmp_c, tmp_d, threads]))
    ratio = tmp_c.stat().st_size / in_path.stat().st_size
    tmp_c.unlink(missing_ok=True)
    tmp_d.unlink(missing_ok=True)
    return ratio, statistics.median(ct), statistics.median(dt)


def run_experiments(cfg, csv_path):
    rows = []

    def add(exp, name, kind, size, threads, path, reps):
        ratio, ct, dt = measure(name, path, threads, reps)
        rows.append(dict(experiment=exp, algo=name, kind=kind, size_mb=size,
                         threads=threads, reps=reps, ratio=ratio, ctime=ct, dtime=dt))
        print(f"  {name:8s} {kind:6s} {size:3d} МБ  потоков={threads}  "
              f"сжатие={ct:.4f} с  распаковка={dt:.4f} с  ratio={ratio:.3f}", flush=True)

    main, tmax = cfg["main"], max(cfg["threads"])
    main_text = SAMPLES_DIR / f"text_{main}MB.txt"

    print("\n[1/3] Масштабирование по потокам")
    for name in ALGOS:
        for t in cfg["threads"]:
            add("scaling", name, "text", main, t, main_text, cfg["reps_scaling"])

    print("\n[2/3] Зависимость от размера файла")
    for name in ALGOS:
        for s in cfg["sizes"]:
            for t in sorted({1, tmax}):
                add("size", name, "text", s, t, SAMPLES_DIR / f"text_{s}MB.txt",
                    cfg["reps_size"])

    print("\n[3/3] Случайные данные")
    for name in ALGOS:
        add("random", name, "random", main, tmax,
            SAMPLES_DIR / f"random_{main}MB.bin", cfg["reps_size"])

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nСырые данные: {csv_path}")


# ---------------------------------------------------------------- отчёт
PAGE = (11.69, 8.27)


def draw_table(ax, df, title, fmt, colnames):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    cells = [[fmt.format(v) for v in row] for row in df.values]
    t = ax.table(cellText=cells, rowLabels=list(df.index), colLabels=colnames,
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dde6f0")


def save_page(pdf, fig, title):
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def plot_threads(ax, tab, ylabel, title, ideal=False, logy=False):
    threads = list(tab.columns)
    for a in tab.index:
        ax.plot(threads, tab.loc[a].values, marker="o", label=a, color=COLORS[a])
    if ideal:
        ax.plot(threads, threads, "k--", label="идеал")
    if PHYSICAL_CORES in threads:
        ax.axvline(PHYSICAL_CORES, color="gray", ls=":", label=f"{PHYSICAL_CORES} физ. ядра")
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(threads)
    ax.set_xlabel("Число потоков")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def build_report(csv_path, pdf_path):
    raw = pd.read_csv(csv_path)
    df = raw.groupby(["algo", "kind", "size_mb", "threads"], as_index=False)[
        ["ratio", "ctime", "dtime"]].mean()
    algos = [a for a in ALGOS if a in set(df.algo)]
    rnd = df[df.kind == "random"]
    main = int(rnd.size_mb.iloc[0])
    text = df[df.kind == "text"]
    A = text[text.size_mb == main]
    threads = [int(t) for t in sorted(A.threads.unique())]
    tmin, tmax = threads[0], threads[-1]
    sizes = [int(s) for s in sorted(text.size_mb.unique())]

    def piv(col):
        return A.pivot(index="algo", columns="threads", values=col).loc[algos, threads]

    ct, dt = piv("ctime"), piv("dtime")
    sc, sd = ct.rdiv(ct[tmin], axis=0), dt.rdiv(dt[tmin], axis=0)
    ec, ed = sc / np.array(threads), sd / np.array(threads)
    tcols = [f"{t} пот." for t in threads]

    R = text[text.threads == tmin].pivot(index="algo", columns="size_mb",
                                         values="ratio").loc[algos, sizes]
    Rt = R.copy()
    Rt["random"] = rnd.groupby("algo").ratio.mean().reindex(algos)
    rcols = [f"текст {s} МБ" for s in sizes] + [f"случайные {main} МБ"]

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=PAGE)
        fig.text(0.5, 0.9, "Сравнение параллельных компрессоров: LZ77, Huffman, LZW",
                 ha="center", fontsize=18, fontweight="bold")
        lines = [
            f"Дата: {datetime.now():%d.%m.%Y %H:%M}",
            f"Система: {platform.system()} {platform.release()}, логических потоков: {os.cpu_count()}"
            f" (физических ядер: {PHYSICAL_CORES})",
            f"Повторов на точку: {raw.reps.min()}–{raw.reps.max()}, берётся медиана",
            "",
            "Что измерено:",
            "  • время сжатия и распаковки: внутреннее вычислительное время, которое печатает программа"
            " (без чтения/записи файлов)",
            "  • ускорение = T(1 поток) / T(N потоков);  эффективность = ускорение / N",
            "  • коэффициент сжатия = размер сжатого файла / размер исходного (меньше = лучше)",
            f"  • масштабирование по потокам: текст {main} МБ, потоки {threads}",
            f"  • зависимость от размера: текст {sizes} МБ, потоки {tmin} и {tmax}",
            f"  • случайные данные {main} МБ: худший случай для сжатия",
            "",
            "Замечания:",
            "  • блок сжатия = 1 МБ, блоки независимы, поэтому параллелизм ограничен числом блоков",
            f"  • после {PHYSICAL_CORES} потоков идёт гиперпоточность, рост ускорения обычно замедляется",
            "  • тексты: конкатенация книг (при нехватке зациклена до нужного размера)",
        ]
        for i, line in enumerate(lines):
            fig.text(0.07, 0.8 - i * 0.04, line, fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        # --- таблицы: время
        fig, axes = plt.subplots(2, 1, figsize=PAGE)
        draw_table(axes[0], ct, f"Время сжатия, с (текст {main} МБ)", "{:.4f}", tcols)
        draw_table(axes[1], dt, f"Время распаковки, с (текст {main} МБ)", "{:.4f}", tcols)
        save_page(pdf, fig, "Таблицы: время работы")

        # --- таблицы: ускорение и эффективность
        fig, axes = plt.subplots(2, 2, figsize=PAGE)
        draw_table(axes[0, 0], sc, "Ускорение: сжатие", "{:.2f}", tcols)
        draw_table(axes[0, 1], sd, "Ускорение: распаковка", "{:.2f}", tcols)
        draw_table(axes[1, 0], ec, "Эффективность: сжатие", "{:.2f}", tcols)
        draw_table(axes[1, 1], ed, "Эффективность: распаковка", "{:.2f}", tcols)
        save_page(pdf, fig, "Таблицы: ускорение и эффективность")

        # --- таблица: коэффициент сжатия
        fig, ax = plt.subplots(1, 1, figsize=PAGE)
        draw_table(ax, Rt, "Коэффициент сжатия (сжатый / исходный, меньше = лучше)",
                   "{:.3f}", rcols)
        save_page(pdf, fig, "Таблица: коэффициент сжатия")

        # --- графики: потоки
        fig, axes = plt.subplots(2, 2, figsize=PAGE)
        plot_threads(axes[0, 0], ct, "Время, с", "Время сжатия", logy=True)
        plot_threads(axes[0, 1], dt, "Время, с", "Время распаковки", logy=True)
        plot_threads(axes[1, 0], sc, "Ускорение", "Ускорение: сжатие", ideal=True)
        plot_threads(axes[1, 1], sd, "Ускорение", "Ускорение: распаковка", ideal=True)
        save_page(pdf, fig, f"Масштабирование по потокам (текст {main} МБ)")

        # --- графики: эффективность и размер
        fig, axes = plt.subplots(2, 2, figsize=PAGE)
        plot_threads(axes[0, 0], ec, "Эффективность", "Эффективность: сжатие")
        plot_threads(axes[0, 1], ed, "Эффективность", "Эффективность: распаковка")
        for ax, col, title in ((axes[1, 0], "ctime", "Время сжатия от размера файла"),
                               (axes[1, 1], "dtime", "Время распаковки от размера файла")):
            for t, ls in ((tmin, "-"), (tmax, "--")):
                S = text[text.threads == t]
                for a in algos:
                    d = S[S.algo == a].sort_values("size_mb")
                    ax.plot(d.size_mb, d[col], ls=ls, marker="o", color=COLORS[a],
                            label=f"{a}, {t} пот.")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels(sizes)
            ax.set_xlabel("Размер файла, МБ")
            ax.set_ylabel("Время, с")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        save_page(pdf, fig, "Эффективность и зависимость от размера")

        # --- графики: сравнение алгоритмов
        fig, axes = plt.subplots(2, 2, figsize=PAGE)
        ax = axes[0, 0]
        T = text[text.threads == tmax]
        for a in algos:
            d = T[T.algo == a].sort_values("size_mb")
            ax.plot(d.size_mb, d.size_mb / d.ctime, marker="o", color=COLORS[a],
                    label=f"{a}, сжатие")
            ax.plot(d.size_mb, d.size_mb / d.dtime, marker="s", ls="--", color=COLORS[a],
                    label=f"{a}, распаковка")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("Размер файла, МБ")
        ax.set_ylabel("Пропускная способность, МБ/с")
        ax.set_title(f"Пропускная способность ({tmax} пот.)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        ax = axes[0, 1]
        for a in algos:
            row = A[(A.algo == a) & (A.threads == tmax)].iloc[0]
            ax.scatter(main / row.ctime, row.ratio, s=90, color=COLORS[a])
            ax.annotate(a, (main / row.ctime, row.ratio), textcoords="offset points",
                        xytext=(6, 6))
        ax.set_xscale("log")
        ax.set_xlabel("Скорость сжатия, МБ/с")
        ax.set_ylabel("Коэффициент сжатия")
        ax.set_title(f"Компромисс скорость / степень сжатия ({tmax} пот.)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for a in algos:
            ax.plot(sizes, R.loc[a].values, marker="o", color=COLORS[a], label=a)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("Размер файла, МБ")
        ax.set_ylabel("Коэффициент сжатия")
        ax.set_title("Коэффициент сжатия от размера (текст)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        x = np.arange(len(algos))
        ax.bar(x - 0.2, Rt[main].values, 0.4, label=f"текст {main} МБ")
        ax.bar(x + 0.2, Rt["random"].values, 0.4, label=f"случайные {main} МБ")
        ax.set_xticks(x)
        ax.set_xticklabels(algos)
        ax.set_ylabel("Коэффициент сжатия")
        ax.set_title("Текст и случайные данные")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        save_page(pdf, fig, "Сравнение алгоритмов")

    print(f"Отчёт: {pdf_path}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="быстрая проверка")
    ap.add_argument("--report-only", action="store_true",
                    help="только пересобрать PDF из готового CSV")
    args = ap.parse_args()

    cfg = QUICK if args.quick else FULL
    suffix = "_quick" if args.quick else ""
    csv_path = RESULTS_DIR / f"results{suffix}.csv"
    pdf_path = RESULTS_DIR / f"report{suffix}.pdf"
    RESULTS_DIR.mkdir(exist_ok=True)

    if not args.report_only:
        build_all()
        make_samples(cfg["sizes"], cfg["main"])
        run_experiments(cfg, csv_path)
    build_report(csv_path, pdf_path)


if __name__ == "__main__":
    main()