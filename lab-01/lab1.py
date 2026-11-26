"""Лабораторная 1: моделирование БСВ, вариант 10 по умолчанию."""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import fmean, variance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstwo


MODULUS = 2**31
N = 1000
ALPHA = 0.05
BINS = 10
VARIANTS = {
    1: (68921, 48), 2: (79507, 64), 3: (131075, 128),
    4: (78125, 256), 5: (16387, 48), 6: (50653, 64),
    7: (29791, 128), 8: (262147, 256), 9: (24389, 32),
    10: (16807, 64), 11: (32771, 128), 12: (161051, 192),
    13: (65643, 256), 14: (12167, 48), 15: (8195, 64),
    16: (16395, 128),
}


def multiplicative_congruential(seed, multiplier, modulus=MODULUS):
    """a[i] = (beta * a[i-1]) mod M; выдаёт a[1]/M, a[2]/M, ..."""
    if not (0 < seed < modulus and 0 < multiplier < modulus):
        raise ValueError("Начальное состояние и множитель должны быть между 0 и M")
    state = seed
    while True:
        state = (multiplier * state) % modulus
        yield state / modulus


def maclaren_marsaglia(source, selector, table_size):
    """Заполняет таблицу из source; selector выбирает заменяемую ячейку."""
    if table_size < 1:
        raise ValueError("Объём таблицы должен быть положительным")
    table = [next(source) for _ in range(table_size)]
    while True:
        index = int(selector.random() * table_size)
        value = table[index]
        table[index] = next(source)
        yield value


def take(generator, count):
    return [next(generator) for _ in range(count)]


def analyze(sample):
    """Два критерия для H0: U[0,1); параметры распределения заданы заранее."""
    if len(sample) < 2 or any(not 0 <= x < 1 for x in sample):
        raise ValueError("Требуется не менее двух чисел из [0, 1)")
    n = len(sample)
    if n / BINS < 5:
        raise ValueError("Для критерия Пирсона ожидаемые частоты должны быть >= 5")
    ordered = sorted(sample)
    d_plus = max(i / n - x for i, x in enumerate(ordered, 1))
    d_minus = max(x - (i - 1) / n for i, x in enumerate(ordered, 1))
    d = max(d_plus, d_minus)
    d_critical = float(kstwo.ppf(1 - ALPHA, n))
    ks_p = float(kstwo.sf(d, n))
    counts = [0] * BINS
    for x in sample:
        counts[int(x * BINS)] += 1
    expected = n / BINS
    pearson = sum((observed - expected) ** 2 / expected for observed in counts)
    degrees = BINS - 1
    chi_critical = float(chi2.ppf(1 - ALPHA, degrees))
    chi_p = float(chi2.sf(pearson, degrees))
    return {
        "n": n, "mean": fmean(sample), "sample_variance": variance(sample),
        "minimum": min(sample), "maximum": max(sample),
        "kolmogorov": {
            "d_plus": d_plus, "d_minus": d_minus, "d": d,
            "sqrt_n_d": math.sqrt(n) * d,
            "critical_d": d_critical, "p_value": ks_p,
            "reject_h0": ks_p < ALPHA,
        },
        "pearson": {
            "bins": BINS, "counts": counts, "expected_count": expected,
            "degrees_of_freedom": degrees, "statistic": pearson,
            "critical_value": chi_critical, "p_value": chi_p,
            "reject_h0": chi_p < ALPHA,
        },
    }


def decision(reject):
    return "H0 отвергается" if reject else "Нет оснований отвергать H0"


def plot_samples(samples, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (name, sample) in enumerate(samples.items()):
        axes[row, 0].hist(sample, bins=[i / BINS for i in range(BINS + 1)],
                          density=True, edgecolor="black", alpha=0.7)
        axes[row, 0].axhline(1, color="red", label="U[0,1): density = 1")
        axes[row, 0].set(title=f"{name}: histogram", xlabel="x", ylabel="Density")
        ordered = sorted(sample)
        axes[row, 1].step([0, *ordered, 1],
                          [0, *[(i + 1) / len(sample) for i in range(len(sample))], 1],
                          where="post", label="Empirical CDF")
        axes[row, 1].plot([0, 1], [0, 1], color="red", label="F(x) = x")
        axes[row, 1].set(title=f"{name}: CDF", xlabel="x", ylabel="F(x)")
        for axis in axes[row]:
            axis.set_xlim(0, 1)
            axis.legend()
            axis.grid(alpha=0.2)
    fig.savefig(out / "plots.png", dpi=160)
    plt.close(fig)


def save_report(out, parameters, results):
    lines = [
        f"# Лабораторная работа 1. Вариант {parameters['variant']}", "",
        "## Задание и параметры", "",
        "Моделирование базовой случайной величины с распределением U[0,1). "
        "По 1000 реализаций МКМ и метода Макларена–Марсальи; α = 0,05.", "",
        f"a₀ = β = {parameters['a0']}; M = 2³¹ = {MODULUS}; K = {parameters['k']}.",
        f"Второй датчик: Python random.Random (Mersenne Twister), seed = {parameters['selector_seed']}.",
        "МКМ для обоих экспериментов создаётся заново с одинаковыми a₀ и β. "
        "В смешанном генераторе первые K значений заполняют таблицу; далее "
        "они выдаются и заменяются по индексам второго датчика.", "",
        "## Алгоритмы", "",
        "МКМ: aᵢ = (β·aᵢ₋₁) mod M, xᵢ = aᵢ/M; первым выдаётся a₁/M.", "",
        "Макларен–Марсалья: T[j] заполняется K числами МКМ. На каждом шаге "
        "получаем y от второго датчика, j = floor(K·y), выдаём T[j] и заменяем "
        "T[j] следующим числом МКМ. При y ∈ [0,1) индекс находится в 0,…,K−1.", "",
        "## Проверка согласия", "",
        "H0: распределение соответствует U[0,1); H1: распределение отличается. "
        "Для каждого критерия отдельно используется уровень значимости 0,05.", "",
        "Колмогоров: D⁺ = max(i/n − x₍ᵢ₎), D⁻ = max(x₍ᵢ₎ − (i−1)/n), "
        "D = max(D⁺, D⁻), i = 1,…,n. Критическое D и p-value вычисляются "
        "по конечновыборочному распределению scipy.stats.kstwo.", "",
        "Пирсон: [0,1) делится на 10 равных интервалов, левая граница включена, "
        "правая исключена. Ожидаемая частота в каждом — 100. "
        "χ² = Σ(Nⱼ − 100)²/100. Число степеней свободы: 10−1 = 9, "
        "так как параметры равномерного распределения не оценивались по выборке.", "",
        "Правило решения: отвергнуть H0, если p-value < 0,05 "
        "(эквивалентно превышению критического значения).", "",
        "## Результаты", "",
        "| Генератор | Среднее | Несмещённая дисперсия | D | D крит. | p Колмогорова | χ² | χ² крит. | p Пирсона |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        ks, chi = result["kolmogorov"], result["pearson"]
        lines.append(f"| {name} | {result['mean']:.8f} | {result['sample_variance']:.8f} "
                     f"| {ks['d']:.8f} | {ks['critical_d']:.8f} | {ks['p_value']:.6f} "
                     f"| {chi['statistic']:.4f} | {chi['critical_value']:.6f} | {chi['p_value']:.6f} |")
    lines += ["", "Теоретические E[X] = 1/2 и Var[X] = 1/12 ≈ 0,08333333.", "",
              "### Частоты по интервалам", "",
              "| Интервал | МКМ | Макларен–Марсалья | Ожидается |", "|---|---:|---:|---:|"]
    for j in range(BINS):
        lines.append(f"| [{j / BINS:.1f}; {(j + 1) / BINS:.1f}) "
                     f"| {results['MKM']['pearson']['counts'][j]} "
                     f"| {results['MacLaren-Marsaglia']['pearson']['counts'][j]} | 100 |")
    lines += ["", "## Вывод", ""]
    for name, result in results.items():
        lines.append(f"- {name}: Колмогоров — {decision(result['kolmogorov']['reject_h0'])}; "
                     f"Пирсон — {decision(result['pearson']['reject_h0'])}.")
    lines += ["", "Критерии исследуют согласие одномерного распределения. "
              "Отсутствие отклонения H0 не доказывает независимость, идеальную случайность "
              "или превосходство одного датчика. Калибровка критериев предполагает "
              "независимую выборку из непрерывного распределения; для дискретных "
              "псевдослучайных последовательностей это стандартная диагностическая модель.", "",
              "## Файлы результатов", "", "- samples.csv — все 1000 пар значений.",
              "- summary.json — параметры, статистики, частоты и решения без округления для вывода.",
              "- plots.png — гистограммы и эмпирические функции распределения.", "",
              "## Источники", "",
              "- Оригинал: Лабораторная работа 1.doc в этой папке.",
              "- [MacLaren, Marsaglia, Uniform Random Number Generators (1965)](https://www.osti.gov/pages/servlets/purl/4578911).",
              "- [SciPy: kstest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html).",
              "- [SciPy: chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run(variant, selector_seed, out):
    a0, k = VARIANTS[variant]
    samples = {
        "MKM": take(multiplicative_congruential(a0, a0), N),
        "MacLaren-Marsaglia": take(maclaren_marsaglia(
            multiplicative_congruential(a0, a0), random.Random(selector_seed), k), N),
    }
    parameters = {"variant": variant, "a0": a0, "beta": a0, "modulus": MODULUS,
                  "k": k, "n": N, "alpha": ALPHA, "selector_seed": selector_seed}
    results = {name: analyze(sample) for name, sample in samples.items()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["i", *samples])
        writer.writerows((i, x, y) for i, (x, y) in enumerate(zip(*samples.values()), 1))
    (out / "summary.json").write_text(json.dumps(
        {"parameters": parameters, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_samples(samples, out)
    save_report(out, parameters, results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, choices=VARIANTS, default=10)
    parser.add_argument("--selector-seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    results = run(args.variant, args.selector_seed, args.out)
    print(f"Вариант {args.variant}. n = {N}, alpha = {ALPHA}")
    for name, result in results.items():
        ks, chi = result["kolmogorov"], result["pearson"]
        print(f"{name}: D={ks['d']:.6f}, p={ks['p_value']:.6f}; "
              f"chi2={chi['statistic']:.4f}, p={chi['p_value']:.6f}")
        print(f"  Колмогоров: {decision(ks['reject_h0'])}. Пирсон: {decision(chi['reject_h0'])}.")
    print(f"Результаты: {args.out.resolve()}")


if __name__ == "__main__":
    main()
