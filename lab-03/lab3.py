"""Лабораторная 3, вариант 10: N(1,9), LN(1,9), Exp(rate=2)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N, ALPHA, BINS = 1000, 0.05, 10


def normal_box_muller(rng, size):
    pairs = (size + 1) // 2
    radius = np.sqrt(-2 * np.log(1 - rng.random(pairs)))
    angle = 2 * np.pi * rng.random(pairs)
    z = np.column_stack((radius * np.cos(angle), radius * np.sin(angle))).ravel()[:size]
    return 1 + 3 * z


def lognormal(rng, size):
    return np.exp(normal_box_muller(rng, size))


def exponential(rng, size):
    return -np.log1p(-rng.random(size)) / 2


def distributions():
    return {"Normal": stats.norm(loc=1, scale=3),
            "Lognormal": stats.lognorm(s=3, scale=np.exp(1)),
            "Exponential": stats.expon(scale=0.5)}


GENERATORS = {"Normal": normal_box_muller, "Lognormal": lognormal, "Exponential": exponential}


def make_test(law, n):
    return {"law": law, "n": n, "cuts": law.ppf(np.arange(1, BINS) / BINS),
            "critical_d": float(stats.kstwo.ppf(1 - ALPHA, n)),
            "critical_chi2": float(stats.chi2.ppf(1 - ALPHA, BINS - 1))}


def statistics(sample, test):
    if len(sample) != test["n"] or not np.all(np.isfinite(sample)):
        raise ValueError("Некорректная выборка")
    n = len(sample)
    cdf = test["law"].cdf(np.sort(sample))
    d_plus = float(np.max(np.arange(1, n + 1) / n - cdf))
    d_minus = float(np.max(cdf - np.arange(n) / n))
    d = max(d_plus, d_minus)
    low, high = test["law"].support()
    outside = bool(np.any(sample < low) or np.any(sample > high))
    counts = np.bincount(np.searchsorted(test["cuts"], sample, side="right"), minlength=BINS)
    chi = None if outside else float(np.sum((counts - n / BINS) ** 2 / (n / BINS)))
    return d, d_plus, d_minus, chi, counts


def check(sample, test):
    d, d_plus, d_minus, chi, counts = statistics(sample, test)
    ks_p = float(stats.kstwo.sf(d, len(sample)))
    chi_p = 0.0 if chi is None else float(stats.chi2.sf(chi, BINS - 1))
    return {"kolmogorov": {"d": d, "d_plus": d_plus, "d_minus": d_minus,
                            "p_value": ks_p, "critical": test["critical_d"], "reject_h0": ks_p < ALPHA},
            "pearson": {"statistic": chi, "p_value": chi_p, "counts": counts.tolist(),
                        "critical": test["critical_chi2"], "df": BINS - 1,
                        "reject_h0": chi_p < ALPHA,
                        "outside_support": chi is None}}


def type_i_experiment(generator, test, rng, repetitions):
    rejected_ks = rejected_chi = 0
    history = []
    checkpoints = sorted(set(r for r in (100, 500, 1000, 2000, 5000, repetitions) if r <= repetitions))
    for i in range(1, repetitions + 1):
        d, _, _, chi, _ = statistics(generator(rng, test["n"]), test)
        rejected_ks += d > test["critical_d"]
        rejected_chi += chi is None or chi > test["critical_chi2"]
        if i in checkpoints:
            record = {"repetitions": i}
            for name, count in (("kolmogorov", rejected_ks), ("pearson", rejected_chi)):
                ci = stats.binomtest(count, i).proportion_ci()
                record[name] = {"rejections": int(count), "rate": count / i,
                                "ci_low": float(ci.low), "ci_high": float(ci.high)}
            history.append(record)
    return history


def run(out, seed=2026, repetitions=5000):
    laws = distributions()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(9)]
    samples = {name: generator(rng, N) for (name, generator), rng in zip(GENERATORS.items(), streams)}
    tests = {name: make_test(law, N) for name, law in laws.items()}
    matrix = {name: {null: check(sample, test) for null, test in tests.items()} for name, sample in samples.items()}
    moments = {name: {"sample_mean": float(np.mean(samples[name])), "sample_variance": float(np.var(samples[name], ddof=1)),
                      "true_mean": float(law.mean()), "true_variance": float(law.var())} for name, law in laws.items()}
    errors = {name: type_i_experiment(generator, tests[name], rng, repetitions)
              for (name, generator), rng in zip(GENERATORS.items(), streams[3:6])}
    large_n = {name: type_i_experiment(generator, make_test(laws[name], 10000), rng, min(repetitions, 1000))[-1]
               for (name, generator), rng in zip(GENERATORS.items(), streams[6:])}
    summary = {"variant": 10, "n": N, "alpha": ALPHA, "seed": seed,
               "parameters": {"normal_mean": 1, "normal_variance": 9,
                              "lognormal_log_mean": 1, "lognormal_log_variance": 9, "exponential_rate": 2},
               "moments": moments, "cross_checks": matrix,
               "pearson_cuts": {name: test["cuts"].tolist() for name, test in tests.items()},
               "type_i_error": errors, "type_i_error_n10000": large_n}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["i", *samples])
        writer.writerows((i, *values) for i, values in enumerate(zip(*samples.values()), 1))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for row, (name, law) in enumerate(laws.items()):
        sample = samples[name]
        if name == "Lognormal":
            shown = np.log(sample); x = np.linspace(-10, 12, 400)
            axes[row, 0].hist(shown, bins=30, density=True, alpha=0.6)
            axes[row, 0].plot(x, stats.norm(1, 3).pdf(x), "r-", label="Density of ln(X)")
            axes[row, 0].set(title="Lognormal: logarithmic coordinate", xlabel="ln(x)")
        else:
            x = np.linspace(law.ppf(.001), law.ppf(.999), 400)
            axes[row, 0].hist(sample, bins=30, density=True, alpha=.6)
            axes[row, 0].plot(x, law.pdf(x), "r-", label="Theoretical density")
            axes[row, 0].set(title=name, xlabel="x")
        transformed = np.sort(law.cdf(sample))
        axes[row, 1].step([0, *transformed, 1], [0, *np.arange(1, N + 1) / N, 1], where="post", label="Empirical CDF of F(X)")
        axes[row, 1].plot([0, 1], [0, 1], "r-", label="Uniform CDF")
        axes[row, 1].set(title=f"{name}: probability integral transform", xlabel="F(x)")
        for ax in axes[row]: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / "plots.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (name, history) in zip(axes, errors.items()):
        for criterion in ("kolmogorov", "pearson"):
            ax.plot([h["repetitions"] for h in history], [h[criterion]["rate"] for h in history], "o-", label=criterion)
        ax.axhline(ALPHA, color="red", linestyle="--", label="alpha=0.05")
        ax.set(title=name, xlabel="Experiments", ylabel="Type I rejection rate"); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / "type_i_error.png", dpi=150); plt.close(fig)
    write_report(out, summary)
    return summary


def write_report(out, result):
    lines = ["# Лабораторная работа 3 — вариант 10", "", "## Распределения и моделирование", "",
             "Для каждого закона получено 1000 независимых псевдослучайных реализаций. "
             "Начальные состояния потоков разделены через SeedSequence.spawn; seed записан в summary.json.", "",
             "- N(1,9): точное преобразование Бокса–Мюллера "
             "Z₁=√(−2 ln U₁) cos(2πU₂), Z₂=√(−2 ln U₁) sin(2πU₂), X=1+3Z.",
             "- LN(1,9): X=exp(1+3Z), то есть ln X∼N(1,9). "
             "Параметры 1 и 9 относятся к логарифму, а не к самому X.",
             "- E(2): параметр 2 принят как интенсивность, f(x)=2e^(−2x), x≥0; "
             "X=−ln(1−U)/2. В SciPy это scale=1/2.", "",
             "Используются стандартное среднее и несмещённая дисперсия с делителем n−1.", "",
             "## Оценки и истинные моменты", "", "| Закон | x̄ | E[X] | S² | Var(X) |", "|---|---:|---:|---:|---:|"]
    for name, m in result["moments"].items():
        lines.append(f"| {name} | {m['sample_mean']:.8g} | {m['true_mean']:.8g} | {m['sample_variance']:.8g} | {m['true_variance']:.8g} |")
    lines += ["", "Для нормального E[X]=1, Var(X)=9; для экспоненциального E[X]=1/2, Var(X)=1/4. "
              "Для логнормального E[X]=exp(1+9/2), Var(X)=(exp(9)−1)exp(2+9). "
              "При логарифмической дисперсии 9 редкие огромные наблюдения определяют моменты: "
              "при n=1000 выборочные оценки могут резко отличаться от истинных. "
              "Несмещённость — свойство среднего по всем выборкам, а не точность одной реализации.", "",
              "## Критерии и перекрёстные проверки", "",
              "Для каждого H0 задана своя теоретическая CDF. Параметры известны заранее и не "
              "подгоняются по выборке. D=max(max(i/n−F(xᵢ)), max(F(xᵢ)−(i−1)/n)); "
              "критическое значение и p-value получены по scipy.stats.kstwo для конечного n.", "",
              "Пирсон использует 10 интервалов, границы — теоретические децили H0; "
              "крайние интервалы включают бесконечные хвосты. Вероятности равны 0,1, "
              "ожидаемые частоты 100, df=9. Наблюдение вне носителя H0 означает немедленное отклонение. "
              "В обоих критериях H0 отклоняется при p-value<0,05.", "",
              "| Выборка | H0 | D | p Колмогорова | χ² | p Пирсона | Решения К / П |", "|---|---|---:|---:|---:|---:|---|"]
    for name, hypotheses in result["cross_checks"].items():
        for null, tests in hypotheses.items():
            ks, chi = tests["kolmogorov"], tests["pearson"]
            value = "∞ (вне носителя)" if chi["statistic"] is None else f"{chi['statistic']:.5f}"
            decisions = " / ".join("отклонить" if r["reject_h0"] else "не отклонять" for r in (ks, chi))
            lines.append(f"| {name} | {null} | {ks['d']:.6f} | {ks['p_value']:.6g} | {value} | {chi['p_value']:.6g} | {decisions} |")
    lines += ["", "## Ошибка первого рода", "",
              "Генерируются новые выборки под каждым верным H0, критерии остаются фиксированными. "
              "Для доли ошибочных отклонений рассчитан точный биномиальный 95% интервал. "
              "Рост числа повторений уменьшает ошибку оценки частоты; при росте n "
              "уточняется χ²-аппроксимация. Колмогоров калибруется конечновыборочным распределением. "
              "Результаты эксперимента не обязаны сходиться монотонно.", "",
              "| Закон | n | Повторений | Критерий | Ошибка I рода | 95% интервал |", "|---|---:|---:|---|---:|---|"]
    for name, history in result["type_i_error"].items():
        for h in history:
            for criterion in ("kolmogorov", "pearson"):
                r = h[criterion]
                lines.append(f"| {name} | 1000 | {h['repetitions']} | {criterion} | {r['rate']:.5f} | [{r['ci_low']:.5f}; {r['ci_high']:.5f}] |")
    for name, h in result["type_i_error_n10000"].items():
        for criterion in ("kolmogorov", "pearson"):
            r = h[criterion]
            lines.append(f"| {name} | 10000 | {h['repetitions']} | {criterion} | {r['rate']:.5f} | [{r['ci_low']:.5f}; {r['ci_high']:.5f}] |")
    lines += ["", "## Повторение", "", "```powershell", "python lab-03/lab3.py", "```", "",
              "Рядом с отчётом: samples.csv, summary.json, plots.png и type_i_error.png. "
              "График логнормальной выборки показан в координате ln(x), поэтому не теряет основную массу "
              "наблюдений из-за огромного диапазона; все исходные числа сохранены без усечения.", "",
              "Источники: оригинал условия в lab-03; "
              "[SciPy Lognormal](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.lognorm.html), "
              "[SciPy Exponential](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.expon.html).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.repetitions < 1: parser.error("repetitions должен быть положительным")
    summary = run(args.out, args.seed, args.repetitions)
    for name, moments in summary["moments"].items():
        print(name, moments)
        print("Own H0:", summary["cross_checks"][name][name])
        print("Type I:", summary["type_i_error"][name][-1])
    print("Результаты:", args.out.resolve())


if __name__ == "__main__": main()
