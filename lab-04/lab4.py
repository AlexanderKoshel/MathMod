"""Лабораторная 4, вариант 10: два интеграла методом Монте-Карло."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import gamma, hyperu
from scipy.stats import norm


def integrand_1(x):
    return np.exp(-x**4) * np.sqrt(1 + x**4)


def weights_1(rng, n):
    """X имеет плотность Коши q(x)=1/[π(1+x²)] на всей прямой."""
    x = np.tan(np.pi * (rng.random(n) - 0.5))
    return np.pi * (1 + x*x) * integrand_1(x)


def annulus_points(rng, n):
    """Равномерное по площади кольцо 1≤r²<4, а не равномерный радиус."""
    radius = np.sqrt(1 + 3 * rng.random(n))
    angle = 2 * np.pi * rng.random(n)
    return radius * np.cos(angle), radius * np.sin(angle)


def weights_2(rng, n):
    x, y = annulus_points(rng, n)
    return 3 * np.pi / (x*x + y**4)


def references():
    half, error_1 = quad(integrand_1, 0, np.inf, epsabs=1e-12, epsrel=1e-12)
    special_1 = float(gamma(.25) * hyperu(.25, 1.75, 1) / 2)
    exact_2 = float(2 * np.pi * (np.sqrt(3) - np.sqrt(5) / 2))
    numerical_2, error_2 = dblquad(
        lambda angle, radius: radius / ((radius * np.cos(angle))**2 + (radius * np.sin(angle))**4),
        1, 2, lambda radius: 0, lambda radius: 2 * np.pi, epsabs=1e-10, epsrel=1e-10)
    if not np.isclose(2 * half, special_1, rtol=1e-10, atol=1e-11):
        raise ArithmeticError("Проверка первого эталона не прошла")
    if not np.isclose(numerical_2, exact_2, rtol=1e-10, atol=1e-10):
        raise ArithmeticError("Проверка второго эталона не прошла")
    return {"I1": {"value": float(2 * half), "quad_error_estimate": float(2 * error_1),
                   "special_function_check": special_1},
            "I2": {"value": exact_2, "quadrature_check": float(numerical_2),
                   "quad_error_estimate": float(error_2)}}


def estimates(weights, sizes, reference):
    if not np.all(np.isfinite(weights)):
        raise ValueError("Получены бесконечные веса")
    sums = np.cumsum(weights, dtype=np.float64)
    squares = np.cumsum(weights * weights, dtype=np.float64)
    z = float(norm.ppf(.975))
    rows = []
    for n in sizes:
        mean = sums[n - 1] / n
        variance = max(0.0, (squares[n - 1] - sums[n - 1]**2 / n) / (n - 1))
        se = np.sqrt(variance / n)
        rows.append({"n": int(n), "estimate": float(mean), "reference": reference,
                     "absolute_error": float(abs(mean - reference)),
                     "relative_error": float(abs(mean - reference) / abs(reference)),
                     "standard_error": float(se), "ci_low": float(mean - z * se),
                     "ci_high": float(mean + z * se)})
    return rows


def run(out, seed=2026, max_n=1000000):
    sizes = sorted(set([n for n in (100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000) if n <= max_n] + [max_n]))
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    refs = references()
    runs = {name: estimates(generator(rng, max_n), sizes, refs[name]["value"])
            for name, generator, rng in zip(("I1", "I2"), (weights_1, weights_2), streams)}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "estimates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["integral", *runs["I1"][0]])
        writer.writeheader()
        for name, rows in runs.items(): writer.writerows(dict(integral=name, **row) for row in rows)
    summary = {"variant": 10, "seed": seed, "max_n": max_n, "confidence_level": .95,
               "references": refs, "estimates": runs}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, (name, rows) in enumerate(runs.items()):
        n = np.array([r["n"] for r in rows])
        means = [r["estimate"] for r in rows]
        axes[row, 0].plot(n, means, "o-", label="Monte Carlo")
        axes[row, 0].fill_between(n, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], alpha=.2, label="Approx. 95% CI")
        axes[row, 0].axhline(refs[name]["value"], color="red", label="Reference")
        axes[row, 0].set_xscale("log"); axes[row, 0].set(title=f"{name}: estimate", xlabel="n", ylabel="Integral")
        axes[row, 1].loglog(n, [r["absolute_error"] for r in rows], "o-", label="Actual absolute error")
        axes[row, 1].loglog(n, [r["standard_error"] for r in rows], "s-", label="Estimated standard error")
        axes[row, 1].loglog(n, rows[-1]["standard_error"] * np.sqrt(n[-1] / n), "--", label="C / sqrt(n)")
        axes[row, 1].set(title=f"{name}: accuracy", xlabel="n", ylabel="Error")
        for ax in axes[row]: ax.legend(); ax.grid(alpha=.2, which="both")
    fig.savefig(out / "convergence.png", dpi=160); plt.close(fig)
    write_report(out, summary)
    return summary


def write_report(out, summary):
    lines = ["# Лабораторная работа 4 — вариант 10", "", "## Постановка", "",
             "Формулы прочитаны из графических представлений встроенных объектов оригинального DOC (строка варианта 10).", "",
             r"$$I_1=\int_{-\infty}^{\infty}e^{-x^4}\sqrt{1+x^4}\,dx,$$", "",
             r"$$I_2=\iint_{1\le x^2+y^2<4}\frac{dx\,dy}{x^2+y^4}.$$", "",
             "## Метод Монте-Карло", "",
             "### Интеграл по всей прямой", "",
             "Берём X=tan(π(U−1/2)), U∼U[0,1), с плотностью Коши "
             "q(x)=1/[π(1+x²)]. Тогда I₁=E[π(1+X²)e^(−X⁴)√(1+X⁴)]. "
             "Оценка — среднее весов. Бесконечный интервал не обрезается; "
             "погрешность усечения границ отсутствует. Веса ограничены и имеют конечную дисперсию.", "",
             "### Интеграл по кольцу", "",
             "Кольцо имеет радиусы 1 и 2 и площадь A=3π. Генерируем R=√(1+3U), "
             "Θ=2πV, где U,V независимы; X=R cos Θ, Y=R sin Θ. "
             "Тогда точка равномерна по площади, и I₂=3π·E[1/(X²+Y⁴)]. "
             "Равномерный выбор R без квадратного корня дал бы другое распределение и смещённый результат. "
             "На кольце знаменатель положителен, интеграл конечен.", "",
             "### Точность", "",
             "Для независимых весов Wᵢ оценка Îₙ=ΣWᵢ/n несмещённа. "
             "S²=Σ(Wᵢ−Îₙ)²/(n−1), SE=√(S²/n). "
             "По центральной предельной теореме приближённый 95% интервал: Îₙ±1,959964·SE. "
             "Стандартная ошибка убывает как O(n^(−1/2)), однако фактическая ошибка одной "
             "траектории не обязана убывать монотонно. Интервал — асимптотическая оценка, не строгая граница ошибки. "
             "Для разных n берутся префиксы одного потока; оценки на графике зависимы.", "",
             "## Эталонные значения", "",
             "I₁ вычислен scipy.integrate.quad с использованием чётности. "
             "Дополнительная проверка после t=x⁴: I₁=Γ(1/4)U(1/4,7/4,1)/2, "
             "где U — вырожденная гипергеометрическая функция Трикоми.", "",
             "Для I₂ в полярных координатах и после t=tan Θ:", "",
             r"$$\int_0^{2\pi}\frac{d\theta}{\cos^2\theta+r^2\sin^4\theta}"
             r"=\frac{2\pi(r+1)}{r\sqrt{2r+1}},$$", "",
             r"$$I_2=2\pi\int_1^2\frac{r+1}{r^2\sqrt{2r+1}}dr"
             r"=2\pi\left(\sqrt3-\frac{\sqrt5}{2}\right).$$", "",
             "Использована первообразная −√(2r+1)/r. Точное выражение независимо сверено "
             "с scipy.integrate.dblquad по радиусу и углу. Оценки ошибки квадратур "
             "сохранены в summary.json и отличаются от статистической ошибки Монте-Карло.", "",
             "| Интеграл | Эталон | Независимая проверка |", "|---|---:|---:|"]
    refs = summary["references"]
    lines += [f"| I₁ | {refs['I1']['value']:.12f} | {refs['I1']['special_function_check']:.12f} |",
              f"| I₂ | {refs['I2']['value']:.12f} | {refs['I2']['quadrature_check']:.12f} |", "",
              "## Результаты", "", "| Интеграл | n | Оценка | Абсолютная ошибка | SE | Приближённый 95% интервал |",
              "|---|---:|---:|---:|---:|---|"]
    for name, rows in summary["estimates"].items():
        for r in rows:
            lines.append(f"| {name} | {r['n']} | {r['estimate']:.9f} | {r['absolute_error']:.8g} | {r['standard_error']:.8g} | [{r['ci_low']:.9f}; {r['ci_high']:.9f}] |")
    lines += ["", "## Итог при максимальном n", ""]
    for name, rows in summary["estimates"].items():
        r = rows[-1]
        lines.append(f"- {name}: Î={r['estimate']:.9f}; абсолютная ошибка {r['absolute_error']:.8g}; относительная ошибка {100*r['relative_error']:.5f}%.")
    lines += ["", "На convergence.png приведены зависимости оценки, фактической абсолютной ошибки "
              "и стандартной ошибки от n. Полная таблица записана в estimates.csv.", "",
              "## Запуск", "", "```powershell", "python lab-04/lab4.py", "```", "",
              f"Seed={summary['seed']}, максимальное n={summary['max_n']}. "
              "Доступны --seed, --max-n и --out. Результаты детерминированы при фиксированном seed.", "",
              "Источники: оригинал «Лабораторная работа 4.doc» в lab-04; "
              "[SciPy quad](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.integrate.quad.html), "
              "[SciPy hyperu](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.special.hyperu.html).", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-n", type=int, default=1000000)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.max_n < 2: parser.error("max-n должен быть не меньше 2")
    result = run(args.out, args.seed, args.max_n)
    for name, rows in result["estimates"].items(): print(name, rows[-1])
    print("Результаты:", args.out.resolve())


if __name__ == "__main__": main()
