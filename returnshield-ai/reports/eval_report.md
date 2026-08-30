# ReturnShield test-set evaluation

Generated: `2026-08-30T11:57:45.378992+00:00`

## Scope and methodology

- Evaluation rows: **445** returns from the untouched `test` customer split.
- Train rows: **2,095** returns from the `train` customer split.
- Train customers: **2,800**; test customers: **600**; overlap: **0**.
- The Random Forest was fit only on train rows. `confirmed_abuse_label` was not used as a feature and test labels were used only after scoring.
- The reported model score is the configured rule/ML blend with rule weight **0.50**.
- Classification metrics use a score threshold of **0.50**. Review policies treat Medium and High bands as manual reviews.

## Overall discrimination

| precision | recall | f1 | pr_auc | false_positive_rate | true_negatives | false_positives | false_negatives | true_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.5714 | 0.7273 | 1.0000 | 0.0000 | 326 | 0 | 51 | 68 |

## Top-5% review capacity

The test cases were ranked by blended risk score and only the top **5%** were selected for review.

| capacity_percent | review_count | precision | recall | abuse_cases_selected | abuse_cases_total |
| --- | --- | --- | --- | --- | --- |
| 5.0000 | 23 | 1.0000 | 0.1933 | 23 | 119 |

## Calibration by category

| category | cases | mean_predicted_risk | observed_abuse_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| apparel | 92 | 0.1881 | 0.2500 | -0.0619 |
| beauty | 91 | 0.1829 | 0.2418 | -0.0589 |
| electronics | 94 | 0.1725 | 0.2447 | -0.0722 |
| home | 75 | 0.2048 | 0.2667 | -0.0619 |
| sports | 93 | 0.2383 | 0.3333 | -0.0950 |

## Calibration by payment type

| payment_type | cases | mean_predicted_risk | observed_abuse_rate | calibration_gap |
| --- | --- | --- | --- | --- |
| card | 230 | 0.1871 | 0.2522 | -0.0651 |
| cash_on_delivery | 31 | 0.1997 | 0.2903 | -0.0906 |
| upi | 129 | 0.1956 | 0.2636 | -0.0679 |
| wallet | 55 | 0.2402 | 0.3273 | -0.0870 |

## Expected financial loss

Expected loss uses:

```text
(false_positives * fp_cost) + (false_negatives * fn_cost) +
(manual_reviews * review_cost)
```

Configured costs: FP **150.00**, FN **1200.00**, review **40.00**, wrong-swap refund **1500.00**. False-negative wrong-item swaps use the wrong-swap refund instead of the generic FN cost.

| policy | expected_loss | savings_vs_approve_all | savings_vs_rule_only | false_positives | false_negatives | manual_reviews |
| --- | --- | --- | --- | --- | --- | --- |
| Approve-all baseline | 160200.0000 | 0.0000 | -90480.0000 | 0 | 119 | 0 |
| Rule-only baseline | 69720.0000 | 90480.0000 | 0.0000 | 0 | 56 | 63 |
| Blended model | 4760.0000 | 155440.0000 | 64960.0000 | 0 | 0 | 119 |

The blended model's expected loss is **4760.00**, versus **160200.00** for approve-all and **69720.00** for rule-only.
