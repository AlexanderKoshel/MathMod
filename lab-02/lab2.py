"""Лабораторная 2, вариант 10: Pois(0.7), Geom(0.2)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N, ALPHA = 1000, 0.05
RATE, P = 0.7, 0.2


def poisson_inverse(rng, size):
    """Обратная функция распределения; библиотеки распределений не генерируют выборку."""
    u = rng.random(size)
    mass = np.exp(-RATE)
    cumulative = mass
    cdf = [cumulative]
    k = 0
    while cumulative < float(np.max(u)):
        k += 1
        mass *= RATE / k
        cumulative += mass
        cdf.append(cumulative)
    return np.searchsorted(cdf, u, side="left")


def geometric_inverse(rng, size, start=1):
    """Число испытаний до успеха (start=1) или неудач (start=0)."""
    return np.floor(np.log1p(-rng.random(size)) / np.log1p(-P)).astype(int) + start


def build_pearson(distribution, support_start, n=N):
    """Объединяет соседние редкие значения, включая весь бесконечный правый хвост."""
    edges, probabilities = [], []
    previous_cdf = 0.0
    k = support_start
    while True:
        current_cdf = float(distribution.cdf(k))
        if n * (current_cdf - previous_cdf) >= 5:
            if n * float(distribution.sf(k)) < 5:
                break
            edges.append(k)
            probabilities.append(current_cdf - previous_cdf)
            previous_cdf = current_cdf
        k += 1
        if k > 100000:
            raise RuntimeError("Не удалось построить категории")
    probabilities.append(1 - previous_cdf)
    probabilities = np.array(probabilities)
    if len(probabilities) < 2 or np.min(n * probabilities) < 5 - 1e-9:
        raise ValueError("Недостаточные ожидаемые частоты")
    return {"edges": np.array(edges), "probabilities": probabilities,
            "support_start": support_start, "n": n}


def pearson(sample, test):
    if len(sample) != test["n"]:
        raise ValueError("Объём выборки отличается от объёма критерия")
    if np.any(sample < test["support_start"]) or np.any(sample != np.floor(sample)):
        return {"statistic": None, "p_value": 0.0, "reject_h0": True,
                "reason": "Наблюдение вне носителя H0: вероятность равна нулю", "counts": None}
    labels = np.searchsorted(test["edges"], sample, side="left")
    counts = np.bincount(labels, minlength=len(test["probabilities"]))
    expected = len(sample) * test["probabilities"]
    statistic = float(np.sum((counts - expected) ** 2 / expected))
    pvalue = float(stats.chi2.sf(statistic, len(expected) - 1))
    return {"statistic": statistic, "p_value": pvalue, "reject_h0": pvalue < ALPHA,
            "reason": "χ² с фиксированными параметрами", "counts": counts.tolist()}


def error_experiment(generate, test, rng, repetitions):
    checkpoints = sorted(set([r for r in (100, 500, 1000, 2000, 5000, repetitions) if r <= repetitions]))
    rejected, history = 0, []
    for trial in range(1, repetitions + 1):
        sample = generate(rng, test["n"])
        rejected += pearson(sample, test)["reject_h0"]
        if trial in checkpoints:
            ci = stats.binomtest(rejected, trial).proportion_ci(confidence_level=0.95)
            history.append({"repetitions": trial, "rejections": rejected,
                            "rate": rejected / trial, "ci_low": float(ci.low), "ci_high": float(ci.high)})
    return history


def run(out, seed=2026, repetitions=5000, geometric_start=1):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    names = ["Poisson", "Geometric"]
    generators = [poisson_inverse, lambda rng, size: geometric_inverse(rng, size, geometric_start)]
    laws = [stats.poisson(RATE), stats.geom(P, loc=geometric_start - 1)]
    starts = [0, geometric_start]
    samples = {name: generator(rng, N) for name, generator, rng in zip(names, generators, streams)}
    tests = {name: build_pearson(law, start) for name, law, start in zip(names, laws, starts)}
    matrix = {sample_name: {null: pearson(sample, test) for null, test in tests.items()}
              for sample_name, sample in samples.items()}
    moments = {name: {"sample_mean": float(np.mean(samples[name])),
                      "sample_variance": float(np.var(samples[name], ddof=1)),
                      "true_mean": float(law.mean()), "true_variance": float(law.var())}
               for name, law in zip(names, laws)}
    experiments = {name: error_experiment(generator, tests[name], rng, repetitions)
                   for name, generator, rng in zip(names, generators, streams[2:])}
    # Сохраняем категории, увеличивая именно n, а не только число повторений.
    large_n = {}
    for name, generator, rng in zip(names, generators, streams[2:]):
        larger_test = dict(tests[name], n=10000)
        large_n[name] = error_experiment(generator, larger_test, rng, repetitions)[-1]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["i", *names])
        writer.writerows((i, int(x), int(y)) for i, (x, y) in enumerate(zip(*samples.values()), 1))
    serial_tests = {name: {"upper_edges": test["edges"].tolist() + ["infinity"],
                           "expected_counts": (N * test["probabilities"]).tolist(),
                           "df": len(test["probabilities"]) - 1,
                           "critical": float(stats.chi2.ppf(1 - ALPHA, len(test["probabilities"]) - 1))}
                    for name, test in tests.items()}
    summary = {"variant": 10, "n": N, "alpha": ALPHA, "seed": seed,
               "geometric_start": geometric_start, "poisson_lambda": RATE, "geometric_p": P,
               "moments": moments, "pearson_tests": serial_tests, "cross_checks": matrix,
               "type_i_error": experiments, "type_i_error_n10000": large_n}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (name, law, start) in enumerate(zip(names, laws, starts)):
        upper = max(int(np.max(samples[name])), int(law.ppf(0.999)))
        x = np.arange(start, upper + 1)
        counts = np.bincount(samples[name], minlength=upper + 1)[start:upper + 1] / N
        axes[row, 0].bar(x, counts, alpha=0.65, label="Empirical")
        axes[row, 0].plot(x, law.pmf(x), "ro-", label="Theoretical PMF")
        axes[row, 0].set(title=name, xlabel="x", ylabel="Probability")
        history = experiments[name]
        axes[row, 1].plot([h["repetitions"] for h in history], [h["rate"] for h in history], "o-")
        axes[row, 1].fill_between([h["repetitions"] for h in history],
                                 [h["ci_low"] for h in history], [h["ci_high"] for h in history], alpha=0.2)
        axes[row, 1].axhline(ALPHA, color="red", label="alpha = 0.05")
        axes[row, 1].set(title=f"{name}: type I error, 95% CI", xlabel="Experiments", ylabel="Rejection rate")
        for ax in axes[row]:
            ax.legend(); ax.grid(alpha=0.2)
    fig.savefig(out / "plots.png", dpi=150)
    plt.close(fig)
    write_report(out, summary)
    return summary


def write_report(out, summary):
    lines = ["# Лабораторная работа 2 — вариант 10", "", "## Постановка и методы", "",
             "Сгенерировано по n = 1000 значений Pois(λ=0,7) и G(p=0,2). α = 0,05. "
             "Пуассон моделируется обращением дискретной функции распределения: "
             "p₀=e^(−λ), pₖ₊₁=pₖ·λ/(k+1), X=min{k:F(k)≥U}.", "",
             f"Геометрическое распределение: носитель {{{summary['geometric_start']},…}}, "
             "X=floor(ln(1−U)/ln(1−p))+start. В условии начало носителя не определено; "
             "по умолчанию start=1 (число испытаний до первого успеха). "
             "Для числа неудач предусмотрено --geometric-start 0.", "",
             "Несмещённые оценки: x̄=Σxᵢ/n, S²=Σ(xᵢ−x̄)²/(n−1). "
             "Для Пуассона E[X]=Var(X)=λ; для выбранной геометрической "
             "E[X]=(1−p)/p+start, Var(X)=(1−p)/p².", "",
             "## Сравнение моментов", "", "| Закон | x̄ | E[X] | S² | Var(X) |",
             "|---|---:|---:|---:|---:|"]
    for name, r in summary["moments"].items():
        lines.append(f"| {name} | {r['sample_mean']:.6f} | {r['true_mean']:.6f} | {r['sample_variance']:.6f} | {r['true_variance']:.6f} |")
    lines += ["", "## Критерии Пирсона и перекрёстная проверка", "",
              "Для каждого H0 границы категорий заданы по теоретическому закону до анализа выборки. "
              "Редкие соседние значения объединены, последний интервал включает весь бесконечный хвост. "
              "Все ожидаемые частоты ≥5. χ²=Σ(Oⱼ−Eⱼ)²/Eⱼ; df=k−1, "
              "параметры заданы, а не оценены по данным. H0 отвергается при p-value<0,05. "
              "Значение вне носителя H0 приводит к немедленному отклонению (вероятность нулевая).", "",
              "| Выборка | H0 | χ² | p-value | Решение |", "|---|---|---:|---:|---|"]
    for sample, nulls in summary["cross_checks"].items():
        for null, result in nulls.items():
            value = "∞ (вне носителя)" if result["statistic"] is None else f"{result['statistic']:.6f}"
            lines.append(f"| {sample} | {null} | {value} | {result['p_value']:.6g} | {'Отклонить H0' if result['reject_h0'] else 'Нет оснований отклонить H0'} |")
    for name, test in summary["pearson_tests"].items():
        lines += ["", f"### Категории H0: {name}", "", f"df={test['df']}; критическое χ²={test['critical']:.6f}.", "",
                  "| Верхняя включённая граница | Ожидаемая частота |", "|---|---:|"]
        lines += [f"| {edge} | {expected:.6f} |" for edge, expected in zip(test["upper_edges"], test["expected_counts"])]
    lines += ["", "## Ошибка первого рода", "",
              "Повторно моделируются независимые выборки по 1000 значений под соответствующим H0; "
              "критерий и границы остаются фиксированными. Доля отклонений оценивает фактический "
              "размер критерия при n=1000. Приведён точный биномиальный 95% интервал Клоппера–Пирсона. "
              "С ростом числа повторений доля сходится к фактическому размеру; близость к 0,05 "
              "зависит также от χ²-аппроксимации. Монотонного приближения не требуется.", "",
              "| H0 | Повторений | Доля отклонений | 95% интервал |", "|---|---:|---:|---|"]
    for name, history in summary["type_i_error"].items():
        lines += [f"| {name} | {h['repetitions']} | {h['rate']:.5f} | [{h['ci_low']:.5f}; {h['ci_high']:.5f}] |" for h in history]
    lines += ["", "### Проверка при увеличении объёма выборки", "",
              "Повторён эксперимент при n=10000 с теми же категориями: это проверяет "
              "асимптотическое χ²-приближение. Конечный эксперимент не доказывает сходимость; "
              "теоретически при фиксированных положительных вероятностях категорий "
              "и n→∞ статистика имеет предельное распределение χ²(k−1).", "",
              "| H0 | n | Повторений | Доля отклонений | 95% интервал |", "|---|---:|---:|---:|---|"]
    for name, h in summary["type_i_error_n10000"].items():
        lines.append(f"| {name} | 10000 | {h['repetitions']} | {h['rate']:.5f} | [{h['ci_low']:.5f}; {h['ci_high']:.5f}] |")
    lines += ["", "Отсутствие отклонения H0 не является доказательством его истинности. "
              "Неверные гипотезы исследуются перекрёстными тестами; их отклонения не являются ошибками первого рода.", "",
              "## Повторение расчёта", "", "```powershell", "python lab-02/lab2.py", "```", "",
              "Файлы samples.csv, summary.json и plots.png находятся рядом с этим отчётом. "
              "Seed и все параметры записаны в summary.json.", "",
              "Источники: оригинал условия в lab-02; "
              "[SciPy Poisson](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.poisson.html), "
              "[SciPy Geometric](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.geom.html).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--geometric-start", type=int, choices=(0, 1), default=1)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions должен быть положительным")
    result = run(args.out, args.seed, args.repetitions, args.geometric_start)
    for name, moment in result["moments"].items():
        print(name, moment)
        print("Type I error:", result["type_i_error"][name][-1])
    print("Результаты:", args.out.resolve())


if __name__ == "__main__":
    main()
