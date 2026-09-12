# Task 1 - Exploration of Customer Purchasing Behaviour

Rossmann Pharmaceuticals sales forecasting | NextHikes IT Solutions

Dataset: 1,017,209 raw training rows across 1,115 stores, 2013-01-01 to 2015-07-31. After removing closed and zero-sales days, 844,338 trading rows remain.

Each section states the question, the evidence table, the chart and the conclusion drawn. Charts are in `reports/figures/`.

---

## 1. Are promotions distributed similarly in train and test?

![q1_promo_distribution](figures/q1_promo_distribution.png)

```
         rows  promo_share  school_holiday_share  state_holiday_share  stores
split                                                                        
train  844338     0.446356              0.193578             0.001078    1115
test    41088     0.395833              0.443487             0.004381     856
```

**Insight.** Promo prevalence is 44.6% in train vs 39.6% in test (absolute gap 5.1%). The gap is small, and the weekday breakdown shows why the chi-square is still significant: promos run Monday-Friday only in both splits, so the difference is driven purely by how many weekend days each window contains, not by a change in promo policy. State holidays are far rarer in test (a 48-day late-summer window with no Easter or Christmas), and test covers only 856 of the 1,115 stores. Practical consequence: promo and school-holiday features transfer safely, but the model will get almost no test-time signal from the Easter and Christmas holiday levels, so those coefficients cannot be validated on this split.

---

## 2. How do sales behave before, during and after holidays?

![q2_holiday_behaviour](figures/q2_holiday_behaviour.png)

```
                mean_sales  median_sales  mean_customers  trading_days  vs_normal_pct  sales_per_customer
HolidayPhase                                                                                             
Before (1-3d)  7743.080994        7145.0      829.926453         95204           13.3                9.33
During         8808.631868        7649.5     1366.769231           910           28.9                6.44
After (1-3d)   7045.440299        6502.0      777.275605         74672            3.1                9.06
Normal         6832.279270        6258.0      750.862510        673552            0.0                9.10
```

**Insight.** Trading in the three days before a state holiday runs +13.3% against the normal-day baseline, and +3.1% in the three days after. The day-level profile shows the effect is concentrated almost entirely in the single day immediately before the holiday, which is classic demand pull-forward: customers stock up because they know the store will be shut. Holidays themselves barely appear in this chart because nearly all stores close, so those rows were removed as zero-sales days -- the handful that do trade are the exceptions. The modelling implication is that a binary 'is holiday' flag is not enough; signed distance-to-holiday is the feature that captures this, which is why DaysToNextHoliday and IsDayBeforeHoliday were engineered.

---

## 3. What seasonal purchase behaviours exist?

![q3_seasonality](figures/q3_seasonality.png)

```
     mean_sales  index_vs_year
Jan      6564.0          0.944
Feb      6589.0          0.947
Mar      6977.0          1.003
Apr      7047.0          1.013
May      7107.0          1.022
Jun      7001.0          1.007
Jul      6954.0          1.000
Aug      6649.0          0.956
Sep      6547.0          0.941
Oct      6603.0          0.949
Nov      7189.0          1.033
Dec      8609.0          1.238
```

**Insight.** December trades at 1.24x the annual average -- the largest single seasonal effect in the dataset -- but the day-level view overturns the obvious assumption. The single highest-selling day is day 15 (12,970, roughly the third week of Advent), not Christmas Eve: sales build from the second week, hold a broad elevated plateau from about the 13th to the 23rd, and then Christmas Eve itself falls to 4,803 -- below the annual average -- almost certainly because stores trade a shortened day before closing for the holiday. The 25th and 26th see over 97% of stores fully closed (excluded upstream as zero-sales days), and New Year's Eve shows the same shortened-trading dip. The pattern repeats across both December's in the training window, so it is structural rather than a one-off, but it argues against modelling December as a single 'run-up to Christmas Day' curve -- the true shape is a wide pre-Christmas plateau followed by a fall on the holiday eve itself, which day-level features (Day, DayOfYear, DaysToChristmas) can represent far better than a coarse month or 'is-December' flag. Easter produces a comparable pre-holiday lift at roughly a third of the amplitude, peaking in the days just before Easter Sunday itself. Because Easter moves between late March and late April, a plain Month feature cannot represent it, which is the argument for the signed days-from-holiday features. Note for the forecast at hand: the 6-week test window is August-September, the flattest part of the year, so December accuracy is precisely what this particular validation split cannot tell us about.

---

## 4. What is the correlation between sales and number of customers?

![q4_sales_customers](figures/q4_sales_customers.png)

```
                     value
metric                    
Pearson r           0.8236
Spearman rho        0.8318
R-squared (linear)  0.6782
```

**Insight.** Sales and customers correlate at r = 0.824 (Spearman 0.832), so footfall alone explains about 68% of the variance in daily turnover -- by far the strongest pairwise relationship in the data. That makes Customers look like an ideal predictor and it is precisely the trap in this dataset: the column does not exist in the test set, because footfall is not known before the day happens. Using it would produce an excellent validation score and a model that cannot be deployed, so it is excluded from the feature set and forecast separately for the dashboard. The residual fan in the scatter is informative in its own right: at a given footfall, turnover still varies widely, and the basket-size breakdown shows why -- store type b runs very high footfall at a much lower basket (5.13 vs up to 11.28). Basket size is therefore a genuine store attribute worth engineering, which is what StoreSalesPerCustomer captures.

---

## 5. How does promo affect sales? New customers or bigger baskets?

![q5_promo_effect](figures/q5_promo_effect.png)

```
          mean_sales  mean_customers  mean_basket  trading_days
Promo                                                          
No promo     5928.70          705.07         8.83        319818
Promo        8228.74          844.48        10.18        376875
```

**Insight.** Restricting to weekdays (promos never run at weekends, so an unrestricted comparison would be confounded by weekend trading), promo days deliver +38.8% sales against non-promo weekdays. The interesting part is the decomposition, because Sales = Customers x Basket and a promo can raise either term. Footfall rises +19.8% and average basket rises +15.3%; on a log scale those contribute 55% and 43% of the turnover gain respectively. So the answer to the question as posed is *both, in roughly equal measure* -- promos attract genuinely more customers and those customers also spend more per visit, with footfall the marginally larger channel. That is a healthier result for the finance team than either extreme would have been: a pure basket effect would mean discounting to people who were already coming, while a pure footfall effect with a flat basket would suggest cherry-picking of discounted lines. Getting both means the promo is widening reach and deepening the trolley at once. The open question this cannot settle is margin: a 39% turnover lift is only profitable if the discount funding it is smaller, and no cost or margin data is present in this dataset.

---

## 6. Could promos be deployed more effectively? Which stores?

![q6_promo_targeting](figures/q6_promo_targeting.png)

```
       no_promo    promo  abs_uplift  pct_uplift StoreType Assortment
Store                                                                
261      9732.8  16948.9      7216.1        74.1         d          c
335     11885.5  18175.5      6290.1        52.9         b          a
1027    10341.1  16190.6      5849.5        56.6         a          c
544     12038.5  17460.4      5421.8        45.0         a          a
380     11661.1  17057.1      5396.0        46.3         a          a
234      8431.1  13800.6      5369.5        63.7         d          a
368      7052.5  12326.8      5274.3        74.8         d          c
831     10608.1  15700.9      5092.8        48.0         a          a
1112     7859.4  12847.2      4987.8        63.5         c          c
336     11732.4  16716.2      4983.9        42.5         a          a
963      9490.3  14341.4      4851.1        51.1         a          c
768     10791.7  15598.7      4807.0        44.5         a          c
1014    10975.7  15772.6      4796.9        43.7         a          c
552      5881.4  10565.7      4684.3        79.6         a          a
545      8678.8  13320.3      4641.4        53.5         a          c
```

**Insight.** Promo response is highly uneven across the estate. Ranking stores by cash uplift per promo day, the top quartile accounts for 38% of all incremental promo revenue while the bottom quartile accounts for 14%. Store 261 gains about 7,216 per promo day; the weakest stores gain close to nothing, and a handful are actively negative. Uplift correlates with baseline size (r = 0.58), so cash uplift partly just tracks store scale -- which is why the ranking uses absolute uplift rather than percentage: the finance team allocates cash, and a 40% lift on a tiny store is worth less than a 10% lift on a flagship. Recommendation: concentrate promo spend on the top two quartiles and withdraw it from the bottom quartile, where the discount is being given to customers who would have bought anyway. Caveat to state plainly -- this is observational, not a randomised test. Promo scheduling was not assigned at random, so part of the measured uplift may reflect head office already targeting promos at stores expected to respond. The honest next step is a holdout trial on a sample of the bottom quartile.

---

## 7. What are the trends in customer behaviour around opening and closing?

![q7_open_close_trends](figures/q7_open_close_trends.png)

```
     open_rate  mean_sales_when_open  mean_customers_when_open
Mon      0.950              8216.073                   855.445
Tue      0.988              7088.114                   769.987
Wed      0.974              6728.123                   740.599
Thu      0.923              6767.310                   755.570
Fri      0.951              7072.677                   781.772
Sat      0.995              5874.840                   660.178
Sun      0.025              8224.724                  1441.532
```

**Insight.** 17.0% of all store-days are closed, and the pattern is almost entirely Sunday trading law: only 2.5% of stores open on a Sunday. The stores that do open on Sunday trade at essentially the same level as the single best weekday -- mean Sunday sales of 8,225 against 8,216 on Monday, a gap under 0.2% -- rather than at a discount for trading on a day most competitors are shut. The store-type breakdown identifies the mechanism: type b is effectively the only format open seven days a week, and it is a small, distinctive group. This is a selection effect, not a Sunday effect: type b stores are high-footfall formats (see Q4) that happen to hold Sunday trading rights, so 'Sunday' and 'store type b' are nearly the same variable in this data. Modelling consequence: closed days carry no demand signal and always have Sales = 0, so they are excluded from training and handled by the deterministic rule Open = 0 implies Sales = 0 at inference. Learning that rule from data would waste 17% of the sample to reproduce something already known with certainty.

---

## 8. Which stores open all weekdays, and how does that affect weekend sales?

![q8_weekday_weekend](figures/q8_weekday_weekend.png)

```
                      stores  mean_weekday_sales  mean_saturday_sales  mean_sunday_sales  sat_vs_weekday_pct
Segment                                                                                                     
Has weekday closures     215              6690.8               6019.1             4763.4               -10.0
Opens all weekdays       900              7270.2               5845.2             8332.0               -19.6
```

**Insight.** 900 of 1115 stores (81%) trade on essentially every weekday, using a 95% threshold so that public holidays -- which close the whole estate -- do not disqualify a store. Saturday runs below the weekday average for both groups, but not by the same amount: stores that open every weekday post Saturday sales -19.6% against their own weekday baseline, against -10.0% for stores with weekday closures (median ratio 0.80 vs 0.92, Mann-Whitney p = 1.7e-09). The boxplot shows why this is a real difference and not just noise from a handful of stores -- the whiskers overlap heavily, but the boxes (the middle 50% of stores in each group) barely touch. So the direction is the opposite of the naive expectation: stores that close on some weekdays hold up *better* on Saturday relative to their own baseline than stores that never close. A plausible mechanism is that stores which take a weekday off are more often small, locally-oriented outlets for which Saturday is the anchor shopping day, while stores that open all seven weekdays are larger, footfall-driven formats whose trade is already spread evenly across the week and therefore sees comparatively less of a Saturday spike. This is inferred from the pattern, not observed directly -- it would need a store-size control to confirm. Either way, weekday-closure behaviour is worth keeping as a feature, unlike the null result it first appeared to be.

---

## 9. How does assortment type affect sales?

![q9_assortment](figures/q9_assortment.png)

```
              stores  mean_sales  median_sales  mean_customers  sales_per_customer
a (basic)        593      6621.5        6082.0           748.0                8.85
b (extra)          9      8642.5        8088.0          2067.6                4.18
c (extended)     513      7300.8        6675.0           752.2                9.71
```

**Insight.** Assortment 'b (extra)' records the highest mean sales (8,642), but the level alone is misleading and the crosstab shows why. Assortment is heavily confounded with store type: assortment b appears in only a handful of stores and almost exclusively alongside store type b, the same small high-footfall Sunday-trading format identified in Q7. With so few stores in that cell, the apparent 'extra assortment' premium is really the type-b format effect wearing a different label. Comparing within a store type instead, extended assortment (c) beats basic (a) by a modest margin, which is the credible version of the effect. Both columns are kept as features and the tree model can represent the interaction directly, but the causal claim 'widening assortment raises sales' is not supported by this data -- the store types differ in too many other ways.

---

## 10. How does competitor distance affect sales? Does it matter in city centres?

![q10_competition_distance](figures/q10_competition_distance.png)

```
              stores  mean_sales  mean_customers  sales_per_customer
DistanceBand                                                        
<250m            122      7999.8          1054.6                7.59
250-500m          98      7115.1           859.2                8.28
500m-1km         110      6676.7           835.4                7.99
1-2.5km          247      6871.4           756.2                9.09
2.5-5km          186      6790.7           705.6                9.62
5-10km           162      6742.6           629.9               10.70
>10km            187      6825.0           660.0               10.34
```

**Insight.** Taken at face value the relationship is backwards: stores with a competitor inside 250m average 8,000 in daily sales, against 6,825 for stores whose nearest competitor is over 10km away. Closer competition apparently means higher sales, and the overall rank correlation is weak and negative (rho = -0.03). The resolution is that competition distance is mostly a proxy for population density, not for competitive pressure. Competitors cluster where footfall is, so a 200m competitor distance is really a statement that the store sits in a busy city centre. This is exactly the case the brief asks about, and splitting on the 500m city-centre proxy answers it: within urban stores rho = -0.112 and within rural stores rho = +0.077 -- both near zero. Once locality is held even roughly constant, distance carries almost no information. So the raw distance feature is predictive, but of urbanity rather than of competition; it is kept (log-transformed, since the raw scale spans 70m to 76km) together with an explicit IsCityCentre flag so the model can separate the two readings instead of conflating them.

---

## 11. How does the opening or reopening of new competitors affect stores?

![q11_competitor_opening](figures/q11_competitor_opening.png)

```
            stores  mean_relative_sales  mean_sales
Bucket                                             
-90 to -60     176                0.997    7503.358
-60 to -30     186                0.995    7487.119
-30 to 0       186                1.010    7561.681
0 to 30        181                0.976    7309.058
30 to 60       172                0.951    7081.270
60 to 90       166                0.955    7014.244
```

**Insight.** Isolating the 190 stores whose nearest competitor opened inside the observation window gives a genuine natural experiment. Indexing each store's sales to its own pre-opening mean (so stores of different sizes can be pooled), average relative sales in the 90 days after an opening are 0.961 -- within a couple of percent of the pre-opening baseline -- and there is no lasting shift in level across the boundary. The short answer is that a new competitor opening nearby has no detectable short-run effect on these stores' turnover. Four caveats keep that honest. The chart shows a visible ~30-day oscillation running through both the before and after periods, including a peak sitting almost exactly at the event date -- this is very likely an artefact of the data rather than a competition effect: the competitor's open date is recorded to month precision and stored as the 1st of that month, so 'days since opening' is implicitly pinned to calendar day-of-month for every store in the sample, and day-of-month is already known (Q3, Q14) to drive real swings in trading. That phase-locks an unrelated monthly cycle onto the event axis and should not be read as a competitor effect; it is exactly why the six-bucket summary, not the raw daily curve, carries the conclusion. Beyond that: the open date's month precision also blurs the true event timing by up to 30 days either way, which would smear a genuine sharp step into the surrounding buckets. The sample is small and self-selected. And a 90-day window catches only the immediate response, not gradual erosion over years. On the related check the brief asks for -- stores recorded with no competitor distance that later gain one -- the distance field is static per store in this dataset, so a mid-panel change is not observable; the competition-open date is the only channel through which competitor arrival is recorded, which is why the event study is built on that instead.

---

## 12. Own question: how much sales variation is store identity alone?

![q12_store_variance](figures/q12_store_variance.png)

```
           stores  mean_sales  std_sales  mean_customers     cv
StoreType                                                      
a             602      6925.7     3277.4           795.4  0.473
b              17     10233.4     5155.7          2022.2  0.504
c             148      6933.1     2897.0           815.5  0.418
d             348      6822.3     2556.4           606.4  0.375
```

**Insight.** Per-store mean daily sales range from about 2,704 to 21,757 -- a sixfold spread -- and decomposing total variance shows 60.2% of it sits *between* stores rather than within a store over time. This is the most important single fact for the modelling strategy, and it drives two decisions. First, per-store historical aggregates (StoreMeanSales and its weekday/month variants) are the highest-value features available, because knowing which store a row belongs to already explains most of the target; they must be fitted on the training fold only to avoid leaking the validation period's own average. Second, it justifies RMSPE over RMSE as the loss: with a sixfold spread in scale, squared absolute error would let the largest stores dominate the objective and effectively ignore the smallest, whereas the finance team needs a usable forecast for every store.

---

## 13. Own question: does the long-running Promo2 deliver a lift?

![q13_promo2](figures/q13_promo2.png)

```
                     mean_sales  mean_customers  trading_days
IsPromo2Active                                               
Promo2 dormant           6596.7           685.5        295233
Promo2 round active      6470.4           671.9        125813
```

**Insight.** Promo2 behaves nothing like the daily Promo. Within participating stores, months when a Promo2 round is active sell -1.9% against dormant months -- essentially flat, and a fraction of the double-digit lift the daily promo delivers (Q5). Participating stores also sell -10.8% against non-participants overall, which suggests Promo2 was rolled out to weaker stores rather than that it damaged them; the direction of causality cannot be settled from observational data. Two practical conclusions: the decoded IsPromo2Active flag is worth keeping as a feature but should not be expected to carry much weight, and the raw Promo2 participation flag is better read as a store-segment label than as a treatment. Worth flagging to the finance team: a continuing promotion with no measurable turnover effect is a candidate for review, though margin data we do not have would be needed to judge it properly.

---

## 14. Own question: what data-quality issues must the pipeline handle?

![q14_data_quality](figures/q14_data_quality.png)

```
                                       count  pct_of_train                                                                  handling
issue                                                                                                                               
Closed store-days (Open=0)            172817       16.9900               Dropped from training; Sales=0 imposed by rule at inference
Open but zero sales                       54        0.0053                                           Dropped as data-entry artefacts
Missing Open in test                      11        0.0000                         Imputed as 1 (store trades on all other weekdays)
Missing CompetitionDistance             2642        0.2600  Large sentinel + HasCompetition flag (NA means 'none known', not 'zero')
Missing CompetitionOpenSince*         323348       31.7900               Zero sentinel; downstream feature checks the companion flag
Missing Promo2Since* / PromoInterval  508031       49.9400           Missing exactly when store never joined Promo2; filled 'None'/0
```

**Insight.** The dataset is unusually clean on the daily table -- zero missing values in train -- and all the genuine issues sit in the static store attributes and in rows that are structurally rather than accidentally empty. The important distinction the pipeline encodes is between 'missing' and 'not applicable': Promo2Since and PromoInterval are absent precisely for the 544 stores that never joined Promo2, and CompetitionDistance is absent where no competitor is on record, which behaves like a very distant competitor rather than a near one. Imputing either with a mean would inject signal that is not there and, for distance, invert it. Separately, sales on trading days are right-skewed (skew = 1.59), which is the argument for training on log1p(Sales): squared error in log space approximates relative error, which is what the chosen RMSPE metric measures.

---
