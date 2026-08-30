# ReturnShield AI 60-second demo script

## Demo setup

- Open the **ReturnShield Dashboard** workflow.
- Keep the **Return queue** and **Case detail** views available in the sidebar.
- The two cases below are both from the untouched `test` split:
  - Legitimate: `RET-0000008` / `ORD-0009171`
  - Abusive: `RET-0000037` / `ORD-0015920`

## Timed narration and actions

### 0–7 seconds — Start in the queue

**Action:** Open **Return queue** and point to the ranked table.

**Say:** “ReturnShield ranks every incoming return by blended abuse risk and shows the recommended next action. I’ll compare one clean return with one clear item-swap case.”

### 7–19 seconds — Legitimate return

**Action:** Select `RET-0000008`, then switch to **Case detail**.

**Say:** “This test-set case is a size issue for `SKU-00363`. The item weighs 358.6 grams versus 345.1 expected, the serial matches, and inspection verified the return. The score is **0.008**, Low risk, so the recommended action is **approve**.”

**Action:** Scroll to **SHAP top reasons**.

**Say:** “The SHAP explanation reinforces the low-risk decision: the serial matches, the return is not a defect or wrong-item claim, and the small 3.9% weight difference lowers predicted abuse risk.”

### 19–38 seconds — Abusive return

**Action:** Select `RET-0000037` in the case selector.

**Say:** “Now compare this test-set case. The customer claims the wrong item was received, but the returned serial does not match the shipment. The returned item is 196.7 grams instead of 518.9, a **62.1%** mismatch, and inspection also recorded a serial mismatch.”

**Action:** Point to the risk metrics and SHAP list.

**Say:** “The score jumps to **0.825**, High risk, with a recommended action of **manual review**. SHAP highlights the serial mismatch, the large weight mismatch, and the wrong-item claim as the main risk drivers.”

### 38–52 seconds — Recommended action

**Action:** Point to **Recommended action** and **Recommended verification**.

**Say:** “The reviewer should not approve this refund automatically. The recommended verification is to check the returned serial or SKU against the shipment and weigh the item against the expected weight.”

### 52–60 seconds — Financial impact

**Action:** Open **Financial impact**.

**Say:** “Finally, the financial panel compares approve-all, rule-only, and model-based policies using the configured FP, FN, and review costs. On the generated portfolio, approve-all loss is **951,600**, rule-only loss is **399,440**, and the model-policy loss is **31,720**—showing why targeted review is financially preferable to automatic approval.”
