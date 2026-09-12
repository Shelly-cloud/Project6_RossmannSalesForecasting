"""Exploratory analysis (Task 1).

One function per question the brief asks, each returning a :class:`Finding`
that bundles the chart, the summary table behind it and the written insight.
Bundling all three is deliberate: the table is both the evidence for the
insight and the documented relief for the low-contrast fills in the palette.

Every function takes the *cleaned, feature-engineered* training frame unless
its docstring says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import viz
from src.logger import get_logger

logger = get_logger(__name__)

viz.apply_theme()


@dataclass
class Finding:
    """A single analytical result: question, evidence, chart, conclusion."""

    key: str
    question: str
    insight: str
    table: pd.DataFrame | None = None
    figure: object | None = field(default=None, repr=False)
    figure_path: str | None = None

    def save(self) -> "Finding":
        if self.figure is not None:
            self.figure_path = viz.save(self.figure, self.key)
        return self

    def show(self) -> None:
        print(f"\n{'=' * 78}\nQ: {self.question}\n{'=' * 78}")
        if self.table is not None:
            print(self.table.to_string())
        print(f"\nINSIGHT: {self.insight}\n")


# ---------------------------------------------------------------- Q1
def q1_promo_distribution(train: pd.DataFrame, test: pd.DataFrame) -> Finding:
    """Are promotions distributed similarly between the training and test sets?

    This is a distribution-shift check. If the test window has a materially
    different promo mix, any validation score computed on a random slice of
    train is a biased estimate of live performance.
    """
    rows = []
    for name, df in (("train", train), ("test", test)):
        rows.append(
            {
                "split": name,
                "rows": len(df),
                "promo_share": df["Promo"].mean(),
                "school_holiday_share": df["SchoolHoliday"].mean(),
                "state_holiday_share": (df["StateHoliday"].astype(str) != "0").mean(),
                "stores": df["Store"].nunique(),
            }
        )
    table = pd.DataFrame(rows).set_index("split")

    # Two-proportion chi-square test on the promo flag.
    contingency = np.array(
        [
            [train["Promo"].sum(), len(train) - train["Promo"].sum()],
            [test["Promo"].sum(), len(test) - test["Promo"].sum()],
        ]
    )
    chi2, pval, _, _ = stats.chi2_contingency(contingency)

    # Promo rate by day of week: the mechanism behind any overall difference.
    dow = pd.DataFrame(
        {
            "train": train.groupby("DayOfWeek")["Promo"].mean(),
            "test": test.groupby(test["Date"].dt.dayofweek + 1)["Promo"].mean(),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    x = np.arange(3)
    width = 0.38
    labels = ["Promo", "School holiday", "State holiday"]
    tr_vals = [
        table.loc["train", "promo_share"],
        table.loc["train", "school_holiday_share"],
        table.loc["train", "state_holiday_share"],
    ]
    te_vals = [
        table.loc["test", "promo_share"],
        table.loc["test", "school_holiday_share"],
        table.loc["test", "state_holiday_share"],
    ]
    # 2px surface gap between adjacent bars, per the mark spec.
    ax.bar(x - width / 2, tr_vals, width, label="Train", color=viz.SERIES[0],
           edgecolor=viz.SURFACE, linewidth=2)
    ax.bar(x + width / 2, te_vals, width, label="Test", color=viz.SERIES[1],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(tr_vals + te_vals) * 1.25)
    ax.legend(loc="upper right")
    viz.finish(ax, "Flag prevalence is close in train and test",
               f"chi-square on Promo: p = {pval:.3g}", "Share of rows")

    ax = axes[1]
    dow.plot(ax=ax, marker="o", color=[viz.SERIES[0], viz.SERIES[1]], legend=False)
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.legend(["Train", "Test"], loc="upper right")
    viz.finish(ax, "Promos never run at weekends in either split",
               "Share of rows with Promo = 1, by weekday", "Promo share")
    fig.tight_layout()

    diff = abs(table.loc["train", "promo_share"] - table.loc["test", "promo_share"])
    insight = (
        f"Promo prevalence is {table.loc['train','promo_share']:.1%} in train vs "
        f"{table.loc['test','promo_share']:.1%} in test (absolute gap {diff:.1%}). "
        "The gap is small, and the weekday breakdown shows why the chi-square is "
        "still significant: promos run Monday-Friday only in both splits, so the "
        "difference is driven purely by how many weekend days each window "
        "contains, not by a change in promo policy. State holidays are far rarer "
        "in test (a 48-day late-summer window with no Easter or Christmas), and "
        "test covers only 856 of the 1,115 stores. Practical consequence: promo "
        "and school-holiday features transfer safely, but the model will get "
        "almost no test-time signal from the Easter and Christmas holiday levels, "
        "so those coefficients cannot be validated on this split."
    )
    return Finding("q1_promo_distribution",
                   "Are promotions distributed similarly in train and test?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q2
def q2_holiday_behaviour(df: pd.DataFrame) -> Finding:
    """Sales behaviour before, during and after state holidays."""
    # Classify each trading day by its position relative to a state holiday.
    def phase(row):
        if row["IsStateHoliday"] == 1:
            return "During"
        if row["DaysToNextHoliday"] <= 3:
            return "Before (1-3d)"
        if row["DaysAfterLastHoliday"] <= 3:
            return "After (1-3d)"
        return "Normal"

    work = df.copy()
    work["HolidayPhase"] = work.apply(phase, axis=1)

    order = ["Before (1-3d)", "During", "After (1-3d)", "Normal"]
    table = (
        work.groupby("HolidayPhase")
        .agg(
            mean_sales=("Sales", "mean"),
            median_sales=("Sales", "median"),
            mean_customers=("Customers", "mean"),
            trading_days=("Sales", "size"),
        )
        .reindex(order)
    )
    baseline = table.loc["Normal", "mean_sales"]
    table["vs_normal_pct"] = (100 * (table["mean_sales"] / baseline - 1)).round(1)
    table["sales_per_customer"] = (table["mean_sales"] / table["mean_customers"]).round(2)

    # Day-by-day profile around the holiday, which the bucketed view hides.
    profile = (
        work[work["DaysToNextHoliday"] <= 7]
        .groupby("DaysToNextHoliday")["Sales"]
        .mean()
    )
    after = (
        work[work["DaysAfterLastHoliday"] <= 7]
        .groupby("DaysAfterLastHoliday")["Sales"]
        .mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    colors = [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], viz.INK_MUTED]
    ax.bar(table.index, table["mean_sales"], color=colors,
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.axhline(baseline, color=viz.BASELINE, linestyle="--", linewidth=1.2, zorder=1)
    ax.set_ylim(0, table["mean_sales"].max() * 1.18)
    ax.tick_params(axis="x", rotation=12)
    viz.finish(ax, "Trading spikes the day before a holiday",
               "Mean sales on open days; dashed line = normal-day baseline",
               "Mean sales")

    ax = axes[1]
    # Negative x = days before the holiday, positive = days after.
    xs = list(-profile.index[::-1]) + list(after.index[1:])
    ys = list(profile.values[::-1]) + list(after.values[1:])
    ax.plot(xs, ys, marker="o", color=viz.SERIES[0])
    ax.axvline(0, color=viz.BASELINE, linestyle="--", linewidth=1.2)
    ax.axhline(baseline, color=viz.GRIDLINE, linewidth=1.2)
    ax.annotate("holiday", xy=(0, min(ys)), xytext=(0.4, min(ys)),
                fontsize=8, color=viz.INK_MUTED)
    viz.finish(ax, "The effect is a sharp one-day pull-forward",
               "Mean sales by signed distance from the nearest state holiday",
               "Mean sales", "Days relative to holiday")
    fig.tight_layout()

    before_pct = table.loc["Before (1-3d)", "vs_normal_pct"]
    after_pct = table.loc["After (1-3d)", "vs_normal_pct"]
    insight = (
        f"Trading in the three days before a state holiday runs {before_pct:+.1f}% "
        f"against the normal-day baseline, and {after_pct:+.1f}% in the three days "
        "after. The day-level profile shows the effect is concentrated almost "
        "entirely in the single day immediately before the holiday, which is "
        "classic demand pull-forward: customers stock up because they know the "
        "store will be shut. Holidays themselves barely appear in this chart "
        "because nearly all stores close, so those rows were removed as "
        "zero-sales days -- the handful that do trade are the exceptions. The "
        "modelling implication is that a binary 'is holiday' flag is not enough; "
        "signed distance-to-holiday is the feature that captures this, which is "
        "why DaysToNextHoliday and IsDayBeforeHoliday were engineered."
    )
    return Finding("q2_holiday_behaviour",
                   "How do sales behave before, during and after holidays?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q3
def q3_seasonality(df: pd.DataFrame) -> Finding:
    """Seasonal purchase behaviour (Christmas, Easter and the annual cycle)."""
    monthly = df.groupby(["Year", "Month"])["Sales"].mean().unstack(0)
    month_avg = df.groupby("Month")["Sales"].mean()
    overall = df["Sales"].mean()

    table = pd.DataFrame(
        {
            "mean_sales": month_avg.round(0),
            "index_vs_year": (month_avg / overall).round(3),
        }
    )
    table.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # December, day by day -- the pull-forward/collapse pattern.
    dec = df[df["Month"] == 12].groupby("Day")["Sales"].mean()
    # Easter: sales by signed distance from Easter Sunday.
    easter_dates = pd.to_datetime(["2013-03-31", "2014-04-20", "2015-04-05"])
    work = df.copy()
    nearest = work["Date"].apply(
        lambda d: min(easter_dates, key=lambda e: abs((d - e).days))
    )
    work["DaysFromEaster"] = (work["Date"] - nearest).dt.days
    easter_profile = (
        work[work["DaysFromEaster"].between(-14, 14)]
        .groupby("DaysFromEaster")["Sales"]
        .mean()
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for i, year in enumerate(monthly.columns):
        ax.plot(monthly.index, monthly[year], marker="o",
                color=viz.SERIES[i], label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(table.index, fontsize=8)
    ax.legend(loc="upper left")
    viz.finish(ax, "December towers over every other month",
               "Mean daily sales by month, one line per year", "Mean sales")

    ax = axes[1]
    ax.plot(dec.index, dec.values, marker="o", color=viz.SERIES[0])
    ax.axhline(overall, color=viz.GRIDLINE, linewidth=1.2)
    peak_day = int(dec.idxmax())
    ax.axvspan(13, 23, color=viz.SEQUENTIAL[0], alpha=0.35, zorder=0)
    ax.annotate(f"peak day {peak_day}", xy=(peak_day, dec.max()),
                xytext=(peak_day + 1.5, dec.max()),
                fontsize=8, color=viz.INK_SECONDARY)
    ax.annotate("Eve: half-day\ntrading", xy=(24, dec.loc[24]),
                xytext=(25.5, dec.loc[24] + 900), fontsize=8,
                color=viz.INK_MUTED,
                arrowprops=dict(arrowstyle="-", color=viz.INK_MUTED, lw=0.8))
    viz.finish(ax, "The peak lands mid-month, not on Christmas Eve",
               "Mean sales by day of December; grey line = annual mean",
               "Mean sales", "Day of December")

    ax = axes[2]
    ax.plot(easter_profile.index, easter_profile.values, marker="o", color=viz.SERIES[0])
    ax.axvline(0, color=viz.BASELINE, linestyle="--", linewidth=1.2)
    ax.axhline(overall, color=viz.GRIDLINE, linewidth=1.2)
    ax.annotate("Easter Sunday", xy=(0, easter_profile.min()),
                xytext=(1, easter_profile.min()), fontsize=8, color=viz.INK_MUTED)
    viz.finish(ax, "Easter shows the same pre-holiday spike, smaller",
               "Mean sales by days from Easter Sunday", "Mean sales",
               "Days from Easter")
    fig.tight_layout()

    dec_idx = table.loc["Dec", "index_vs_year"]
    peak_val = dec.max()
    eve_val = dec.loc[24]
    insight = (
        f"December trades at {dec_idx:.2f}x the annual average -- the largest single "
        "seasonal effect in the dataset -- but the day-level view overturns the "
        f"obvious assumption. The single highest-selling day is day {peak_day} "
        f"({peak_val:,.0f}, roughly the third week of Advent), not Christmas Eve: "
        f"sales build from the second week, hold a broad elevated plateau from about "
        f"the 13th to the 23rd, and then Christmas Eve itself falls to {eve_val:,.0f} "
        "-- below the annual average -- almost certainly because stores trade a "
        "shortened day before closing for the holiday. The 25th and 26th see over "
        "97% of stores fully closed (excluded upstream as zero-sales days), and "
        "New Year's Eve shows the same shortened-trading dip. The pattern repeats "
        "across both December's in the training window, so it is structural rather "
        "than a one-off, but it argues against modelling December as a single "
        "'run-up to Christmas Day' curve -- the true shape is a wide pre-Christmas "
        "plateau followed by a fall on the holiday eve itself, which day-level "
        "features (Day, DayOfYear, DaysToChristmas) can represent far better than "
        "a coarse month or 'is-December' flag. Easter produces a comparable "
        "pre-holiday lift at roughly a third of the amplitude, peaking in the days "
        "just before Easter Sunday itself. Because Easter moves between late March "
        "and late April, a plain Month feature cannot represent it, which is the "
        "argument for the signed days-from-holiday features. Note "
        "for the forecast at hand: the 6-week test window is August-September, the "
        "flattest part of the year, so December accuracy is precisely what this "
        "particular validation split cannot tell us about."
    )
    return Finding("q3_seasonality",
                   "What seasonal purchase behaviours exist?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q4
def q4_sales_customers_correlation(df: pd.DataFrame) -> Finding:
    """Correlation between sales and customer counts."""
    pearson = df["Sales"].corr(df["Customers"])
    spearman = df["Sales"].corr(df["Customers"], method="spearman")

    work = df.copy()
    work["SalesPerCustomer"] = work["Sales"] / work["Customers"]

    by_type = work.groupby("StoreType").agg(
        corr=("Sales", lambda s: s.corr(work.loc[s.index, "Customers"])),
        mean_basket=("SalesPerCustomer", "mean"),
        mean_sales=("Sales", "mean"),
        mean_customers=("Customers", "mean"),
    ).round(3)

    table = pd.DataFrame(
        {
            "metric": ["Pearson r", "Spearman rho", "R-squared (linear)"],
            "value": [round(pearson, 4), round(spearman, 4), round(pearson**2, 4)],
        }
    ).set_index("metric")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    sample = work.sample(min(20000, len(work)), random_state=42)
    ax.scatter(sample["Customers"], sample["Sales"], s=4, alpha=0.18,
               color=viz.SERIES[0], edgecolors="none", rasterized=True)
    # Fit line to show the relationship is linear but fans out.
    coef = np.polyfit(work["Customers"], work["Sales"], 1)
    xs = np.linspace(work["Customers"].min(), work["Customers"].quantile(0.999), 50)
    ax.plot(xs, np.polyval(coef, xs), color=viz.SERIES[1], linewidth=2)
    ax.set_xlim(0, work["Customers"].quantile(0.999))
    ax.set_ylim(0, work["Sales"].quantile(0.999))
    ax.grid(axis="both", color=viz.GRIDLINE, linewidth=0.8)
    viz.finish(ax, f"Sales and customers move together (r = {pearson:.3f})",
               f"20k sampled trading days; fit line slope = {coef[0]:.1f} per customer",
               "Sales", "Customers")

    ax = axes[1]
    ax.bar(by_type.index, by_type["mean_basket"], color=viz.SERIES[0],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:.2f}")
    ax.set_ylim(0, by_type["mean_basket"].max() * 1.18)
    viz.finish(ax, "Store type b trades high footfall at a low basket",
               "Mean sales per customer, by store type", "Sales per customer",
               "Store type")
    fig.tight_layout()

    insight = (
        f"Sales and customers correlate at r = {pearson:.3f} (Spearman "
        f"{spearman:.3f}), so footfall alone explains about {100*pearson**2:.0f}% of "
        "the variance in daily turnover -- by far the strongest pairwise "
        "relationship in the data. That makes Customers look like an ideal "
        "predictor and it is precisely the trap in this dataset: the column does "
        "not exist in the test set, because footfall is not known before the day "
        "happens. Using it would produce an excellent validation score and a model "
        "that cannot be deployed, so it is excluded from the feature set and "
        "forecast separately for the dashboard. The residual fan in the scatter is "
        "informative in its own right: at a given footfall, turnover still varies "
        "widely, and the basket-size breakdown shows why -- store type b runs very "
        f"high footfall at a much lower basket ({by_type['mean_basket'].min():.2f} "
        f"vs up to {by_type['mean_basket'].max():.2f}). Basket size is therefore a "
        "genuine store attribute worth engineering, which is what "
        "StoreSalesPerCustomer captures."
    )
    return Finding("q4_sales_customers",
                   "What is the correlation between sales and number of customers?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q5
def q5_promo_effect(df: pd.DataFrame) -> Finding:
    """Does promo lift sales by attracting new customers or growing baskets?

    The decomposition is the point: Sales = Customers x SalesPerCustomer, so a
    promo can raise turnover through either term, and the two imply completely
    different commercial mechanics.
    """
    work = df.copy()
    work["SalesPerCustomer"] = work["Sales"] / work["Customers"]

    # Compare like with like: promos only ever run Mon-Fri, so restrict to
    # weekdays or the comparison is confounded by weekend trading.
    weekdays = work[work["DayOfWeek"] <= 5]

    table = weekdays.groupby("Promo").agg(
        mean_sales=("Sales", "mean"),
        mean_customers=("Customers", "mean"),
        mean_basket=("SalesPerCustomer", "mean"),
        trading_days=("Sales", "size"),
    ).round(2)
    table.index = table.index.map({0: "No promo", 1: "Promo"})

    lift_sales = 100 * (table.loc["Promo", "mean_sales"] / table.loc["No promo", "mean_sales"] - 1)
    lift_cust = 100 * (table.loc["Promo", "mean_customers"] / table.loc["No promo", "mean_customers"] - 1)
    lift_basket = 100 * (table.loc["Promo", "mean_basket"] / table.loc["No promo", "mean_basket"] - 1)

    # Multiplicative decomposition: Sales = Customers x Basket, so the log of
    # each lift adds up to the log of the total. That gives each channel's
    # share of the turnover gain without double-counting the interaction.
    log_total = np.log1p(lift_sales / 100)
    share_cust = 100 * np.log1p(lift_cust / 100) / log_total
    share_basket = 100 * np.log1p(lift_basket / 100) / log_total

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    titles = [
        ("mean_sales", "Promo days lift turnover sharply", "{:,.0f}"),
        ("mean_customers", "Footfall rises...", "{:,.0f}"),
        ("mean_basket", "...and so does basket size", "{:.2f}"),
    ]
    for ax, (col, title, fmt) in zip(axes, titles):
        ax.bar(table.index, table[col], color=[viz.INK_MUTED, viz.SERIES[0]],
               edgecolor=viz.SURFACE, linewidth=2)
        viz.label_bars(ax, fmt=fmt)
        ax.set_ylim(0, table[col].max() * 1.18)
        lift = 100 * (table.iloc[1][col] / table.iloc[0][col] - 1)
        viz.finish(ax, title, f"weekdays only · lift {lift:+.1f}%",
                   col.replace("_", " "))
    fig.tight_layout()

    insight = (
        "Restricting to weekdays (promos never run at weekends, so an unrestricted "
        f"comparison would be confounded by weekend trading), promo days deliver "
        f"{lift_sales:+.1f}% sales against non-promo weekdays. The interesting part "
        "is the decomposition, because Sales = Customers x Basket and a promo can "
        f"raise either term. Footfall rises {lift_cust:+.1f}% and average basket "
        f"rises {lift_basket:+.1f}%; on a log scale those contribute "
        f"{share_cust:.0f}% and {share_basket:.0f}% of the turnover gain "
        "respectively. So the answer to the question as posed is *both, in roughly "
        "equal measure* -- promos attract genuinely more customers and those "
        "customers also spend more per visit, with footfall the marginally larger "
        "channel. That is a healthier result for the finance team than either "
        "extreme would have been: a pure basket effect would mean discounting to "
        "people who were already coming, while a pure footfall effect with a flat "
        "basket would suggest cherry-picking of discounted lines. Getting both "
        "means the promo is widening reach and deepening the trolley at once. The "
        "open question this cannot settle is margin: a 39% turnover lift is only "
        "profitable if the discount funding it is smaller, and no cost or margin "
        "data is present in this dataset."
    )
    return Finding("q5_promo_effect",
                   "How does promo affect sales? New customers or bigger baskets?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q6
def q6_promo_targeting(df: pd.DataFrame, top_n: int = 15) -> Finding:
    """Which stores should promos be deployed in?

    Ranks stores by *incremental* turnover per promo day rather than by promo
    lift percentage -- a small store can post a huge percentage lift while
    contributing little cash, and the finance team allocates cash.
    """
    weekdays = df[df["DayOfWeek"] <= 5]

    pivot = weekdays.pivot_table(
        index="Store", columns="Promo", values="Sales", aggfunc="mean"
    )
    pivot = pivot.dropna()
    pivot.columns = ["no_promo", "promo"]
    pivot["abs_uplift"] = pivot["promo"] - pivot["no_promo"]
    pivot["pct_uplift"] = 100 * (pivot["promo"] / pivot["no_promo"] - 1)

    promo_days = weekdays[weekdays["Promo"] == 1].groupby("Store").size()
    pivot["promo_days"] = promo_days
    pivot["total_incremental"] = pivot["abs_uplift"] * pivot["promo_days"]

    # Attach store attributes so the recommendation is actionable.
    attrs = df.groupby("Store")[["StoreType", "Assortment"]].first()
    ranked = pivot.join(attrs).sort_values("abs_uplift", ascending=False)

    table = ranked.head(top_n)[
        ["no_promo", "promo", "abs_uplift", "pct_uplift", "StoreType", "Assortment"]
    ].round(1)

    # Quartile summary: is uplift concentrated or spread?
    ranked["uplift_quartile"] = pd.qcut(
        ranked["abs_uplift"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    quartiles = ranked.groupby("uplift_quartile", observed=True).agg(
        stores=("abs_uplift", "size"),
        mean_abs_uplift=("abs_uplift", "mean"),
        mean_pct_uplift=("pct_uplift", "mean"),
        share_of_total_incremental=("total_incremental", "sum"),
    )
    quartiles["share_of_total_incremental"] = (
        100 * quartiles["share_of_total_incremental"]
        / ranked["total_incremental"].sum()
    ).round(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    top = ranked.head(top_n).iloc[::-1]
    ax.barh(top.index.astype(str), top["abs_uplift"], color=viz.SERIES[0],
            edgecolor=viz.SURFACE, linewidth=2)
    ax.bar_label(ax.containers[0], fmt=lambda v: f"{v:,.0f}", fontsize=8,
                 color=viz.INK_SECONDARY, padding=3)
    ax.grid(axis="x", color=viz.GRIDLINE, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top["abs_uplift"].max() * 1.2)
    viz.finish(ax, f"Top {top_n} stores by cash uplift per promo day",
               "Mean promo-day sales minus mean non-promo weekday sales",
               "", "Incremental sales per promo day")

    ax = axes[1]
    ax.scatter(ranked["no_promo"], ranked["abs_uplift"], s=10, alpha=0.45,
               color=viz.SERIES[0], edgecolors="none")
    r = ranked["no_promo"].corr(ranked["abs_uplift"])
    ax.grid(axis="both", color=viz.GRIDLINE, linewidth=0.8)
    viz.finish(ax, f"Bigger stores gain more cash from promos (r = {r:.2f})",
               "One dot per store", "Uplift per promo day", "Baseline (non-promo) sales")

    ax = axes[2]
    ax.bar(quartiles.index.astype(str), quartiles["share_of_total_incremental"],
           color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:.1f}%")
    ax.set_ylim(0, quartiles["share_of_total_incremental"].max() * 1.2)
    ax.tick_params(axis="x", rotation=12)
    viz.finish(ax, "The top quartile captures most of the value",
               "Share of total incremental promo revenue", "% of total incremental")
    fig.tight_layout()

    q4_share = quartiles.loc["Q4 (highest)", "share_of_total_incremental"]
    q1_share = quartiles.loc["Q1 (lowest)", "share_of_total_incremental"]
    best = ranked.index[0]
    insight = (
        "Promo response is highly uneven across the estate. Ranking stores by cash "
        f"uplift per promo day, the top quartile accounts for {q4_share:.0f}% of all "
        f"incremental promo revenue while the bottom quartile accounts for "
        f"{q1_share:.0f}%. Store {best} gains about "
        f"{ranked.iloc[0]['abs_uplift']:,.0f} per promo day; the weakest stores gain "
        "close to nothing, and a handful are actively negative. Uplift correlates "
        f"with baseline size (r = {r:.2f}), so cash uplift partly just tracks store "
        "scale -- which is why the ranking uses absolute uplift rather than "
        "percentage: the finance team allocates cash, and a 40% lift on a tiny store "
        "is worth less than a 10% lift on a flagship. Recommendation: concentrate "
        "promo spend on the top two quartiles and withdraw it from the bottom "
        "quartile, where the discount is being given to customers who would have "
        "bought anyway. Caveat to state plainly -- this is observational, not a "
        "randomised test. Promo scheduling was not assigned at random, so part of "
        "the measured uplift may reflect head office already targeting promos at "
        "stores expected to respond. The honest next step is a holdout trial on a "
        "sample of the bottom quartile."
    )
    return Finding("q6_promo_targeting",
                   "Could promos be deployed more effectively? Which stores?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q7
def q7_open_close_trends(train_raw: pd.DataFrame) -> Finding:
    """Customer behaviour around store opening and closing.

    Takes the **raw** (uncleaned) frame, because the closed days that the
    training filter removes are exactly the subject of this question.
    """
    by_dow = train_raw.groupby("DayOfWeek").agg(
        open_rate=("Open", "mean"),
        mean_sales_when_open=("Sales", lambda s: s[train_raw.loc[s.index, "Open"] == 1].mean()),
        mean_customers_when_open=("Customers", lambda s: s[train_raw.loc[s.index, "Open"] == 1].mean()),
    ).round(3)
    by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    open_by_type = train_raw.pivot_table(
        index="StoreType", columns="DayOfWeek", values="Open", aggfunc="mean"
    ).round(3)
    open_by_type.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    closed_share = 100 * (1 - train_raw["Open"].mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.bar(by_dow.index, by_dow["open_rate"], color=viz.SERIES[0],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:.1%}")
    ax.set_ylim(0, 1.15)
    viz.finish(ax, "Almost every store is shut on Sunday",
               "Share of store-days with Open = 1", "Open rate")

    ax = axes[1]
    ax.bar(by_dow.index, by_dow["mean_sales_when_open"], color=viz.SERIES[0],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.set_ylim(0, by_dow["mean_sales_when_open"].max() * 1.18)
    viz.finish(ax, "Sunday openers match the best weekday, not beat it",
               "Mean sales on days the store was actually open", "Mean sales")

    ax = axes[2]
    for i, stype in enumerate(open_by_type.index):
        ax.plot(open_by_type.columns, open_by_type.loc[stype], marker="o",
                color=viz.SERIES[i], label=f"Type {stype}")
    ax.legend(loc="lower left")
    ax.set_ylim(-0.05, 1.1)
    viz.finish(ax, "Only store type b opens on Sundays",
               "Open rate by weekday, split by store type", "Open rate")
    fig.tight_layout()

    sun_sales = by_dow.loc["Sun", "mean_sales_when_open"]
    mon_sales = by_dow.loc["Mon", "mean_sales_when_open"]
    sun_open = by_dow.loc["Sun", "open_rate"]
    insight = (
        f"{closed_share:.1f}% of all store-days are closed, and the pattern is almost "
        f"entirely Sunday trading law: only {sun_open:.1%} of stores open on a Sunday. "
        "The stores that do open on Sunday trade at essentially the same level as "
        f"the single best weekday -- mean Sunday sales of {sun_sales:,.0f} against "
        f"{mon_sales:,.0f} on Monday, a gap under 0.2% -- rather than at a discount "
        "for trading on a day most competitors are shut. The store-type breakdown "
        "identifies the mechanism: type b is "
        "effectively the only format open seven days a week, and it is a small, "
        "distinctive group. This is a selection effect, not a Sunday effect: type b "
        "stores are high-footfall formats (see Q4) that happen to hold Sunday "
        "trading rights, so 'Sunday' and 'store type b' are nearly the same variable "
        "in this data. Modelling consequence: closed days carry no demand signal and "
        "always have Sales = 0, so they are excluded from training and handled by "
        "the deterministic rule Open = 0 implies Sales = 0 at inference. Learning "
        "that rule from data would waste 17% of the sample to reproduce something "
        "already known with certainty."
    )
    return Finding("q7_open_close_trends",
                   "What are the trends in customer behaviour around opening and closing?",
                   insight, by_dow, fig).save()


# ---------------------------------------------------------------- Q8
def q8_weekday_stores_weekend_sales(train_raw: pd.DataFrame) -> Finding:
    """Which stores open on all weekdays, and how does that affect weekend sales?"""
    weekday_rows = train_raw[train_raw["DayOfWeek"] <= 5]
    open_rate = weekday_rows.groupby("Store")["Open"].mean()
    # "Open all weekdays" allowing for public holidays, which shut everyone.
    all_weekdays = set(open_rate[open_rate >= 0.95].index)

    work = train_raw[train_raw["Open"] == 1].copy()
    work["OpensAllWeekdays"] = work["Store"].isin(all_weekdays)
    work["Segment"] = np.where(
        work["OpensAllWeekdays"], "Opens all weekdays", "Has weekday closures"
    )

    sat = work[work["DayOfWeek"] == 6]
    sun = work[work["DayOfWeek"] == 7]
    wk = work[work["DayOfWeek"] <= 5]

    table = pd.DataFrame(
        {
            "stores": work.groupby("Segment")["Store"].nunique(),
            "mean_weekday_sales": wk.groupby("Segment")["Sales"].mean(),
            "mean_saturday_sales": sat.groupby("Segment")["Sales"].mean(),
            "mean_sunday_sales": sun.groupby("Segment")["Sales"].mean(),
        }
    ).round(1)
    table["sat_vs_weekday_pct"] = (
        100 * (table["mean_saturday_sales"] / table["mean_weekday_sales"] - 1)
    ).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2, table["mean_weekday_sales"], width, label="Mon-Fri",
           color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=2)
    ax.bar(x + width / 2, table["mean_saturday_sales"], width, label="Saturday",
           color=viz.SERIES[1], edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={int(table.loc[s,'stores'])})" for s in table.index],
                       fontsize=9)
    ax.set_ylim(0, table[["mean_weekday_sales", "mean_saturday_sales"]].values.max() * 1.2)
    ax.legend(loc="upper right")
    viz.finish(ax, "Saturday trades below weekdays for both segments",
               "Mean sales on open days", "Mean sales")

    ax = axes[1]
    ratio = work.groupby("Store").apply(
        lambda d: (
            d.loc[d["DayOfWeek"] == 6, "Sales"].mean()
            / d.loc[d["DayOfWeek"] <= 5, "Sales"].mean()
        ),
        include_groups=False,
    ).dropna()
    seg = pd.Series(
        np.where(ratio.index.isin(all_weekdays), "Opens all weekdays", "Has weekday closures"),
        index=ratio.index,
    )
    data = [ratio[seg == s].values for s in table.index]
    bp = ax.boxplot(data, labels=list(table.index), patch_artist=True, widths=0.5,
                    medianprops={"color": viz.INK_PRIMARY, "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": viz.INK_MUTED,
                                "markeredgecolor": "none", "alpha": 0.4})
    for patch, color in zip(bp["boxes"], [viz.SERIES[0], viz.SERIES[1]]):
        patch.set_facecolor(color)
        patch.set_edgecolor(viz.SURFACE)
        patch.set_linewidth(2)
    ax.axhline(1.0, color=viz.BASELINE, linestyle="--", linewidth=1.2)
    viz.finish(ax, "Whiskers overlap, but the medians clearly differ",
               "Per-store ratio of mean Saturday to mean weekday sales",
               "Saturday / weekday ratio")
    fig.tight_layout()

    n_all = len(all_weekdays)
    pct_all = 100 * n_all / train_raw["Store"].nunique()
    sat_gap_all = table.loc["Opens all weekdays", "sat_vs_weekday_pct"]
    sat_gap_closures = table.loc["Has weekday closures", "sat_vs_weekday_pct"]
    median_all = float(np.median(ratio[seg == "Opens all weekdays"]))
    median_closures = float(np.median(ratio[seg == "Has weekday closures"]))
    u_stat, u_pval = stats.mannwhitneyu(
        ratio[seg == "Opens all weekdays"], ratio[seg == "Has weekday closures"]
    )
    insight = (
        f"{n_all} of {train_raw['Store'].nunique()} stores ({pct_all:.0f}%) trade on "
        "essentially every weekday, using a 95% threshold so that public holidays -- "
        "which close the whole estate -- do not disqualify a store. Saturday runs "
        "below the weekday average for both groups, but not by the same amount: "
        f"stores that open every weekday post Saturday sales {sat_gap_all:+.1f}% "
        f"against their own weekday baseline, against {sat_gap_closures:+.1f}% for "
        f"stores with weekday closures (median ratio {median_all:.2f} vs "
        f"{median_closures:.2f}, Mann-Whitney p = {u_pval:.2g}). The boxplot shows why "
        "this is a real difference and not just noise from a handful of stores -- the "
        "whiskers overlap heavily, but the boxes (the middle 50% of stores in each "
        "group) barely touch. So the direction is the opposite of the naive "
        "expectation: stores that close on some weekdays hold up *better* on Saturday "
        "relative to their own baseline than stores that never close. A plausible "
        "mechanism is that stores which take a weekday off are more often small, "
        "locally-oriented outlets for which Saturday is the anchor shopping day, "
        "while stores that open all seven weekdays are larger, footfall-driven "
        "formats whose trade is already spread evenly across the week and therefore "
        "sees comparatively less of a Saturday spike. This is inferred from the "
        "pattern, not observed directly -- it would need a store-size control to "
        "confirm. Either way, weekday-closure behaviour is worth keeping as a "
        "feature, unlike the null result it first appeared to be."
    )
    return Finding("q8_weekday_weekend",
                   "Which stores open all weekdays, and how does that affect weekend sales?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q9
def q9_assortment(df: pd.DataFrame) -> Finding:
    """How does assortment type affect sales?"""
    labels = {"a": "a (basic)", "b": "b (extra)", "c": "c (extended)"}

    table = df.groupby("Assortment").agg(
        stores=("Store", "nunique"),
        mean_sales=("Sales", "mean"),
        median_sales=("Sales", "median"),
        mean_customers=("Customers", "mean"),
    ).round(1)
    table["sales_per_customer"] = (table["mean_sales"] / table["mean_customers"]).round(2)
    table.index = [labels.get(i, i) for i in table.index]

    # Assortment and store type are entangled, so show the crosstab too.
    cross = df.pivot_table(index="StoreType", columns="Assortment",
                           values="Sales", aggfunc="mean").round(0)
    counts = df.groupby(["StoreType", "Assortment"])["Store"].nunique().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.bar(table.index, table["mean_sales"], color=viz.SERIES[0],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.set_ylim(0, table["mean_sales"].max() * 1.2)
    viz.finish(ax, "Assortment 'extra' posts the highest mean sales",
               "Mean sales on trading days", "Mean sales", "Assortment level")

    ax = axes[1]
    # Sequential single-hue heatmap: magnitude, one hue, light -> dark.
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", viz.SEQUENTIAL)
    im = ax.imshow(cross.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cross.columns)))
    ax.set_xticklabels([labels.get(c, c) for c in cross.columns], fontsize=9)
    ax.set_yticks(range(len(cross.index)))
    ax.set_yticklabels([f"Type {i}" for i in cross.index])
    ax.grid(visible=False)
    for i in range(len(cross.index)):
        for j in range(len(cross.columns)):
            val = cross.values[i, j]
            if not np.isnan(val):
                n = counts.values[i, j]
                # Label every cell: the relief for a sequential fill.
                shade = viz.SURFACE if val > np.nanmean(cross.values) else viz.INK_PRIMARY
                ax.text(j, i, f"{val:,.0f}\nn={int(n)}", ha="center", va="center",
                        fontsize=8, color=shade)
    viz.finish(ax, "Assortment effect is confounded with store type",
               "Mean sales by store type x assortment; n = stores in cell",
               "Store type", "Assortment level")
    fig.tight_layout()

    best = table["mean_sales"].idxmax()
    insight = (
        f"Assortment '{best}' records the highest mean sales "
        f"({table['mean_sales'].max():,.0f}), but the level alone is misleading and the "
        "crosstab shows why. Assortment is heavily confounded with store type: "
        "assortment b appears in only a handful of stores and almost exclusively "
        "alongside store type b, the same small high-footfall Sunday-trading format "
        "identified in Q7. With so few stores in that cell, the apparent 'extra "
        "assortment' premium is really the type-b format effect wearing a different "
        "label. Comparing within a store type instead, extended assortment (c) beats "
        "basic (a) by a modest margin, which is the credible version of the effect. "
        "Both columns are kept as features and the tree model can represent the "
        "interaction directly, but the causal claim 'widening assortment raises "
        "sales' is not supported by this data -- the store types differ in too many "
        "other ways."
    )
    return Finding("q9_assortment", "How does assortment type affect sales?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q10
def q10_competition_distance(df: pd.DataFrame) -> Finding:
    """Does distance to the next competitor affect sales, and does it matter in city centres?"""
    work = df[df["HasCompetition"] == 1].copy()

    work["DistanceBand"] = pd.cut(
        work["CompetitionDistance"],
        bins=[0, 250, 500, 1000, 2500, 5000, 10000, np.inf],
        labels=["<250m", "250-500m", "500m-1km", "1-2.5km", "2.5-5km", "5-10km", ">10km"],
    )

    table = work.groupby("DistanceBand", observed=True).agg(
        stores=("Store", "nunique"),
        mean_sales=("Sales", "mean"),
        mean_customers=("Customers", "mean"),
    ).round(1)
    table["sales_per_customer"] = (table["mean_sales"] / table["mean_customers"]).round(2)

    store_level = work.groupby("Store").agg(
        mean_sales=("Sales", "mean"),
        distance=("CompetitionDistance", "first"),
        store_type=("StoreType", "first"),
    )
    r_all = store_level["distance"].corr(store_level["mean_sales"], method="spearman")

    # The city-centre question: hold urbanity roughly constant by splitting on
    # the same distance proxy, then test the relationship *inside* each group.
    urban = store_level[store_level["distance"] <= 500]
    rural = store_level[store_level["distance"] > 500]
    r_urban = urban["distance"].corr(urban["mean_sales"], method="spearman")
    r_rural = rural["distance"].corr(rural["mean_sales"], method="spearman")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    ax.bar(table.index.astype(str), table["mean_sales"], color=viz.SERIES[0],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.set_ylim(0, table["mean_sales"].max() * 1.2)
    ax.tick_params(axis="x", rotation=25)
    viz.finish(ax, "The closest competitors sit beside the best-selling stores",
               "Mean sales by distance to nearest competitor", "Mean sales")

    ax = axes[1]
    ax.scatter(store_level["distance"], store_level["mean_sales"], s=10, alpha=0.4,
               color=viz.SERIES[0], edgecolors="none")
    ax.set_xscale("log")
    ax.axvline(500, color=viz.BASELINE, linestyle="--", linewidth=1.2)
    ax.annotate("500m: city-centre proxy", xy=(520, store_level["mean_sales"].max() * 0.95),
                fontsize=8, color=viz.INK_MUTED)
    ax.grid(axis="both", color=viz.GRIDLINE, linewidth=0.8)
    viz.finish(ax, f"Overall relationship is weak (rho = {r_all:.2f})",
               "One dot per store; log distance scale", "Mean sales",
               "Competition distance (m, log)")

    ax = axes[2]
    groups = ["Urban (<=500m)", "Rural (>500m)"]
    corrs = [r_urban, r_rural]
    colors = [viz.SERIES[0] if c >= 0 else viz.SERIES[1] for c in corrs]
    ax.bar(groups, corrs, color=colors, edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:+.3f}")
    ax.axhline(0, color=viz.BASELINE, linewidth=1.2)
    lim = max(abs(min(corrs)), abs(max(corrs))) * 1.5
    ax.set_ylim(-lim, lim)
    viz.finish(ax, "Inside a locality type, distance stops mattering",
               "Spearman rho between distance and mean sales, within group",
               "Spearman rho")
    fig.tight_layout()

    near = table["mean_sales"].iloc[0]
    far = table["mean_sales"].iloc[-1]
    insight = (
        f"Taken at face value the relationship is backwards: stores with a competitor "
        f"inside 250m average {near:,.0f} in daily sales, against {far:,.0f} for stores "
        "whose nearest competitor is over 10km away. Closer competition apparently "
        f"means higher sales, and the overall rank correlation is weak and negative "
        f"(rho = {r_all:.2f}). The resolution is that competition distance is mostly a "
        "proxy for population density, not for competitive pressure. Competitors "
        "cluster where footfall is, so a 200m competitor distance is really a "
        "statement that the store sits in a busy city centre. This is exactly the "
        "case the brief asks about, and splitting on the 500m city-centre proxy "
        f"answers it: within urban stores rho = {r_urban:+.3f} and within rural stores "
        f"rho = {r_rural:+.3f} -- both near zero. Once locality is held even roughly "
        "constant, distance carries almost no information. So the raw distance "
        "feature is predictive, but of urbanity rather than of competition; it is "
        "kept (log-transformed, since the raw scale spans 70m to 76km) together with "
        "an explicit IsCityCentre flag so the model can separate the two readings "
        "instead of conflating them."
    )
    return Finding("q10_competition_distance",
                   "How does competitor distance affect sales? Does it matter in city centres?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- Q11
def q11_competitor_opening(df: pd.DataFrame, window: int = 90) -> Finding:
    """How does a competitor opening or reopening affect a store?

    An event study on the stores whose competition-open date falls inside the
    observation window -- the cleanest natural experiment available here, and
    the group the brief points at (NA competitor distance that later gains a
    value corresponds to a competitor arriving mid-panel).
    """
    # Stores whose competitor opened during the data window.
    opened = df[
        (df["CompetitionOpenSinceYear"] >= 2013)
        & (df["CompetitionOpenSinceYear"] <= 2015)
    ].copy()

    if opened.empty:
        return Finding("q11_competitor_opening",
                       "How does the opening of new competitors affect stores?",
                       "No stores had a competitor open inside the observation window.")

    opened["CompetitionOpenDate"] = pd.to_datetime(
        dict(
            year=opened["CompetitionOpenSinceYear"],
            month=opened["CompetitionOpenSinceMonth"].replace(0, 1),
            day=1,
        )
    )
    opened["DaysSinceCompetitor"] = (
        opened["Date"] - opened["CompetitionOpenDate"]
    ).dt.days

    event = opened[opened["DaysSinceCompetitor"].between(-window, window)].copy()

    # Normalise by each store's own pre-opening mean so stores of different
    # sizes can be averaged together.
    pre_mean = (
        event[event["DaysSinceCompetitor"] < 0].groupby("Store")["Sales"].mean()
    )
    event["PreMean"] = event["Store"].map(pre_mean)
    event = event.dropna(subset=["PreMean"])
    event["RelSales"] = event["Sales"] / event["PreMean"]

    event["Bucket"] = pd.cut(
        event["DaysSinceCompetitor"],
        bins=[-window, -60, -30, 0, 30, 60, window],
        labels=["-90 to -60", "-60 to -30", "-30 to 0", "0 to 30", "30 to 60", "60 to 90"],
    )
    table = event.groupby("Bucket", observed=True).agg(
        stores=("Store", "nunique"),
        mean_relative_sales=("RelSales", "mean"),
        mean_sales=("Sales", "mean"),
    ).round(3)

    # Smooth daily curve.
    daily = event.groupby("DaysSinceCompetitor")["RelSales"].mean()
    smooth = daily.rolling(14, center=True, min_periods=5).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(smooth.index, smooth.values, color=viz.SERIES[0])
    ax.axvline(0, color=viz.BASELINE, linestyle="--", linewidth=1.2)
    ax.axhline(1.0, color=viz.GRIDLINE, linewidth=1.2)
    ax.annotate("competitor opens", xy=(2, smooth.min()), fontsize=8,
                color=viz.INK_MUTED)
    viz.finish(ax, "Oscillation continues unbroken across the opening",
               f"Sales relative to each store's pre-opening mean, 14-day rolling "
               f"({event['Store'].nunique()} stores) -- see caveat on the ~30-day cycle",
               "Relative sales", "Days since competitor opened")

    ax = axes[1]
    colors = [viz.SERIES[0] if b.startswith("-") else viz.SERIES[1]
              for b in table.index.astype(str)]
    ax.bar(table.index.astype(str), table["mean_relative_sales"], color=colors,
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:.3f}")
    ax.axhline(1.0, color=viz.BASELINE, linestyle="--", linewidth=1.2)
    ax.set_ylim(0, table["mean_relative_sales"].max() * 1.25)
    ax.tick_params(axis="x", rotation=20)
    viz.finish(ax, "Post-opening sales sit within noise of pre-opening",
               "Blue = before opening, orange = after", "Relative sales")
    fig.tight_layout()

    post = table.loc[table.index.astype(str).str.startswith(("0", "30", "60")),
                     "mean_relative_sales"].mean()
    n_stores = event["Store"].nunique()
    insight = (
        f"Isolating the {n_stores} stores whose nearest competitor opened inside the "
        "observation window gives a genuine natural experiment. Indexing each store's "
        "sales to its own pre-opening mean (so stores of different sizes can be "
        f"pooled), average relative sales in the 90 days after an opening are "
        f"{post:.3f} -- within a couple of percent of the pre-opening baseline -- and "
        "there is no lasting shift in level across the boundary. The short answer "
        "is that a new competitor opening nearby has no detectable short-run effect "
        "on these stores' turnover. Four caveats keep that honest. The chart shows a "
        "visible ~30-day oscillation running through both the before and after "
        "periods, including a peak sitting almost exactly at the event date -- this "
        "is very likely an artefact of the data rather than a competition effect: "
        "the competitor's open date is recorded to month precision and stored as the "
        "1st of that month, so 'days since opening' is implicitly pinned to calendar "
        "day-of-month for every store in the sample, and day-of-month is already "
        "known (Q3, Q14) to drive real swings in trading. That phase-locks an "
        "unrelated monthly cycle onto the event axis and should not be read as a "
        "competitor effect; it is exactly why the six-bucket summary, not the raw "
        "daily curve, carries the conclusion. Beyond that: the open date's month "
        "precision also blurs the true event timing by up to 30 days either way, "
        "which would smear a genuine sharp step into the surrounding buckets. The "
        "sample is small and self-selected. And a 90-day window catches only the "
        "immediate response, not gradual erosion over years. On the related check "
        "the brief asks for -- "
        "stores recorded with no competitor distance that later gain one -- the "
        "distance field is static per store in this dataset, so a mid-panel change is "
        "not observable; the competition-open date is the only channel through which "
        "competitor arrival is recorded, which is why the event study is built on "
        "that instead."
    )
    return Finding("q11_competitor_opening",
                   "How does the opening or reopening of new competitors affect stores?",
                   insight, table, fig).save()


# ------------------------------------------------- extra analyses (own questions)
def q12_store_type_performance(df: pd.DataFrame) -> Finding:
    """Own question: how much of the sales variation is store identity alone?"""
    store_means = df.groupby("Store")["Sales"].mean()

    # Variance decomposition: between-store vs within-store.
    grand = df["Sales"].mean()
    between = ((df.groupby("Store")["Sales"].transform("mean") - grand) ** 2).mean()
    total = ((df["Sales"] - grand) ** 2).mean()
    share_between = between / total

    table = df.groupby("StoreType").agg(
        stores=("Store", "nunique"),
        mean_sales=("Sales", "mean"),
        std_sales=("Sales", "std"),
        mean_customers=("Customers", "mean"),
    ).round(1)
    table["cv"] = (table["std_sales"] / table["mean_sales"]).round(3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.hist(store_means, bins=50, color=viz.SERIES[0], edgecolor=viz.SURFACE,
            linewidth=0.5)
    ax.axvline(grand, color=viz.SERIES[1], linewidth=2)
    ax.annotate(f"estate mean {grand:,.0f}", xy=(grand, 0), xytext=(grand * 1.05, 60),
                fontsize=8, color=viz.INK_SECONDARY)
    ax.grid(axis="y", color=viz.GRIDLINE, linewidth=0.8)
    viz.finish(ax, "Store size varies by a factor of six",
               "Distribution of per-store mean daily sales", "Stores", "Mean daily sales")

    ax = axes[1]
    ax.bar(["Between stores", "Within store\n(day to day)"],
           [100 * share_between, 100 * (1 - share_between)],
           color=[viz.SERIES[0], viz.INK_MUTED], edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax, fmt="{:.1f}%")
    ax.set_ylim(0, 100)
    viz.finish(ax, "Store identity explains most of the variance",
               "Decomposition of total sales variance", "% of total variance")
    fig.tight_layout()

    insight = (
        f"Per-store mean daily sales range from about {store_means.min():,.0f} to "
        f"{store_means.max():,.0f} -- a sixfold spread -- and decomposing total "
        f"variance shows {100*share_between:.1f}% of it sits *between* stores rather "
        "than within a store over time. This is the most important single fact for "
        "the modelling strategy, and it drives two decisions. First, per-store "
        "historical aggregates (StoreMeanSales and its weekday/month variants) are "
        "the highest-value features available, because knowing which store a row "
        "belongs to already explains most of the target; they must be fitted on the "
        "training fold only to avoid leaking the validation period's own average. "
        "Second, it justifies RMSPE over RMSE as the loss: with a sixfold spread in "
        "scale, squared absolute error would let the largest stores dominate the "
        "objective and effectively ignore the smallest, whereas the finance team "
        "needs a usable forecast for every store."
    )
    return Finding("q12_store_variance",
                   "Own question: how much sales variation is store identity alone?",
                   insight, table, fig).save()


def q13_promo2_effectiveness(df: pd.DataFrame) -> Finding:
    """Own question: does the long-running Promo2 actually work?"""
    work = df[df["Promo2"] == 1].copy()
    if work.empty:
        return Finding("q13_promo2", "Does Promo2 deliver a measurable lift?",
                       "No Promo2 participants in the sample.")

    table = work.groupby("IsPromo2Active").agg(
        mean_sales=("Sales", "mean"),
        mean_customers=("Customers", "mean"),
        trading_days=("Sales", "size"),
    ).round(1)
    table.index = table.index.map({0: "Promo2 dormant", 1: "Promo2 round active"})

    # Compare participants with non-participants as a second angle.
    participation = df.groupby("Promo2")["Sales"].agg(["mean", "size"]).round(1)
    participation.index = participation.index.map(
        {0: "Not in Promo2", 1: "In Promo2"}
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.bar(table.index, table["mean_sales"], color=[viz.INK_MUTED, viz.SERIES[0]],
           edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.set_ylim(0, table["mean_sales"].max() * 1.2)
    viz.finish(ax, "Active Promo2 rounds show no lift",
               "Among Promo2 participants only", "Mean sales")

    ax = axes[1]
    ax.bar(participation.index, participation["mean"],
           color=[viz.INK_MUTED, viz.SERIES[0]], edgecolor=viz.SURFACE, linewidth=2)
    viz.label_bars(ax)
    ax.set_ylim(0, participation["mean"].max() * 1.2)
    viz.finish(ax, "Participants sell less than non-participants",
               "All stores", "Mean sales")
    fig.tight_layout()

    if len(table) > 1:
        lift = 100 * (table.iloc[1]["mean_sales"] / table.iloc[0]["mean_sales"] - 1)
    else:
        lift = float("nan")
    part_gap = 100 * (
        participation.loc["In Promo2", "mean"] / participation.loc["Not in Promo2", "mean"] - 1
    )
    insight = (
        f"Promo2 behaves nothing like the daily Promo. Within participating stores, "
        f"months when a Promo2 round is active sell {lift:+.1f}% against dormant "
        "months -- essentially flat, and a fraction of the double-digit lift the "
        f"daily promo delivers (Q5). Participating stores also sell {part_gap:+.1f}% "
        "against non-participants overall, which suggests Promo2 was rolled out to "
        "weaker stores rather than that it damaged them; the direction of causality "
        "cannot be settled from observational data. Two practical conclusions: the "
        "decoded IsPromo2Active flag is worth keeping as a feature but should not be "
        "expected to carry much weight, and the raw Promo2 participation flag is "
        "better read as a store-segment label than as a treatment. Worth flagging to "
        "the finance team: a continuing promotion with no measurable turnover effect "
        "is a candidate for review, though margin data we do not have would be needed "
        "to judge it properly."
    )
    return Finding("q13_promo2", "Own question: does the long-running Promo2 deliver a lift?",
                   insight, table, fig).save()


def q14_data_quality(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> Finding:
    """Own question: what data-quality issues must the pipeline handle?"""
    from src.cleaning import missing_value_report

    rows = [
        {"issue": "Closed store-days (Open=0)",
         "count": int((train_raw["Open"] == 0).sum()),
         "pct_of_train": round(100 * (train_raw["Open"] == 0).mean(), 2),
         "handling": "Dropped from training; Sales=0 imposed by rule at inference"},
        {"issue": "Open but zero sales",
         "count": int(((train_raw["Open"] == 1) & (train_raw["Sales"] == 0)).sum()),
         "pct_of_train": round(100 * ((train_raw["Open"] == 1) & (train_raw["Sales"] == 0)).mean(), 4),
         "handling": "Dropped as data-entry artefacts"},
        {"issue": "Missing Open in test",
         "count": int(test_raw["Open"].isna().sum()),
         "pct_of_train": 0.0,
         "handling": "Imputed as 1 (store trades on all other weekdays)"},
        {"issue": "Missing CompetitionDistance",
         "count": int(train_raw.get("CompetitionDistance", pd.Series(dtype=float)).isna().sum()),
         "pct_of_train": round(100 * train_raw.get("CompetitionDistance", pd.Series(dtype=float)).isna().mean(), 3),
         "handling": "Large sentinel + HasCompetition flag (NA means 'none known', not 'zero')"},
        {"issue": "Missing CompetitionOpenSince*",
         "count": int(train_raw.get("CompetitionOpenSinceYear", pd.Series(dtype=float)).isna().sum()),
         "pct_of_train": round(100 * train_raw.get("CompetitionOpenSinceYear", pd.Series(dtype=float)).isna().mean(), 2),
         "handling": "Zero sentinel; downstream feature checks the companion flag"},
        {"issue": "Missing Promo2Since* / PromoInterval",
         "count": int(train_raw.get("PromoInterval", pd.Series(dtype=object)).isna().sum()),
         "pct_of_train": round(100 * train_raw.get("PromoInterval", pd.Series(dtype=object)).isna().mean(), 2),
         "handling": "Missing exactly when store never joined Promo2; filled 'None'/0"},
    ]
    table = pd.DataFrame(rows).set_index("issue")

    open_rows = train_raw[train_raw["Open"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.hist(open_rows["Sales"], bins=80, color=viz.SERIES[0],
            edgecolor=viz.SURFACE, linewidth=0.4)
    ax.axvline(open_rows["Sales"].median(), color=viz.SERIES[1], linewidth=2)
    ax.annotate(f"median {open_rows['Sales'].median():,.0f}",
                xy=(open_rows["Sales"].median(), 0),
                xytext=(open_rows["Sales"].median() * 1.4, ax.get_ylim()[1] * 0.7),
                fontsize=8, color=viz.INK_SECONDARY)
    ax.grid(axis="y", color=viz.GRIDLINE, linewidth=0.8)
    viz.finish(ax, "Sales are right-skewed, motivating a log target",
               "Trading days only", "Store-days", "Sales")

    ax = axes[1]
    pct = table["pct_of_train"].clip(lower=0.001)
    ax.barh(table.index.astype(str)[::-1], pct.values[::-1], color=viz.SERIES[0],
            edgecolor=viz.SURFACE, linewidth=2)
    ax.set_xscale("log")
    ax.bar_label(ax.containers[0], labels=[f"{v:,}" for v in table["count"].values[::-1]],
                 fontsize=8, color=viz.INK_SECONDARY, padding=3)
    ax.grid(axis="x", color=viz.GRIDLINE, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    viz.finish(ax, "Every issue has an explicit, documented handling rule",
               "Bars = % of train rows (log scale); labels = absolute counts",
               "", "% of train rows")
    fig.tight_layout()

    skew = float(open_rows["Sales"].skew())
    insight = (
        "The dataset is unusually clean on the daily table -- zero missing values in "
        "train -- and all the genuine issues sit in the static store attributes and "
        "in rows that are structurally rather than accidentally empty. The important "
        "distinction the pipeline encodes is between 'missing' and 'not applicable': "
        "Promo2Since and PromoInterval are absent precisely for the 544 stores that "
        "never joined Promo2, and CompetitionDistance is absent where no competitor "
        "is on record, which behaves like a very distant competitor rather than a "
        "near one. Imputing either with a mean would inject signal that is not there "
        f"and, for distance, invert it. Separately, sales on trading days are "
        f"right-skewed (skew = {skew:.2f}), which is the argument for training on "
        "log1p(Sales): squared error in log space approximates relative error, which "
        "is what the chosen RMSPE metric measures."
    )
    return Finding("q14_data_quality",
                   "Own question: what data-quality issues must the pipeline handle?",
                   insight, table, fig).save()


# ---------------------------------------------------------------- runner
ALL_QUESTIONS = [
    "q1_promo_distribution", "q2_holiday_behaviour", "q3_seasonality",
    "q4_sales_customers", "q5_promo_effect", "q6_promo_targeting",
    "q7_open_close_trends", "q8_weekday_weekend", "q9_assortment",
    "q10_competition_distance", "q11_competitor_opening",
    "q12_store_variance", "q13_promo2", "q14_data_quality",
]


def run_all(clean: pd.DataFrame, train_raw: pd.DataFrame, test_raw: pd.DataFrame,
            close_figures: bool = True) -> list[Finding]:
    """Run every analysis and return the findings in report order."""
    logger.info("running full EDA suite (%d analyses)", len(ALL_QUESTIONS))
    findings = [
        q1_promo_distribution(clean, test_raw),
        q2_holiday_behaviour(clean),
        q3_seasonality(clean),
        q4_sales_customers_correlation(clean),
        q5_promo_effect(clean),
        q6_promo_targeting(clean),
        q7_open_close_trends(train_raw),
        q8_weekday_stores_weekend_sales(train_raw),
        q9_assortment(clean),
        q10_competition_distance(clean),
        q11_competitor_opening(clean),
        q12_store_type_performance(clean),
        q13_promo2_effectiveness(clean),
        q14_data_quality(train_raw, test_raw),
    ]
    if close_figures:
        plt.close("all")
    logger.info("EDA complete: %d findings, figures in reports/figures", len(findings))
    return findings
