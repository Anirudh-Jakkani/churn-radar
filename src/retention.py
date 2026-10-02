"""Turn the model's churn drivers into concrete retention actions."""

# (feature that must be a churn driver, condition on the customer, title, detail)
PLAYBOOK = [
    ("Contract", lambda f: f["Contract"] == "Month-to-month",
     "📝 Lock-in offer", "Offer a discounted upgrade to a 1- or 2-year contract."),
    ("tenure", lambda f: f["tenure"] < 12,
     "🤝 Early-life care", "New customer: schedule an onboarding call and send a welcome perk."),
    ("tenure_bucket", lambda f: f["tenure"] < 12,
     "🤝 Early-life care", "New customer: schedule an onboarding call and send a welcome perk."),
    ("PaymentMethod", lambda f: f["PaymentMethod"] == "Electronic check",
     "💳 Auto-pay nudge", "Give a small bill credit for switching to automatic payment."),
    ("auto_pay", lambda f: f["auto_pay"] == "No",
     "💳 Auto-pay nudge", "Give a small bill credit for switching to automatic payment."),
    ("InternetService", lambda f: f["InternetService"] == "Fiber optic",
     "🛠️ Fiber health check", "Check connection quality proactively and review fiber pricing."),
    ("OnlineSecurity", lambda f: f["OnlineSecurity"] == "No",
     "🛡️ Protection bundle", "Offer a free 3-month trial of Online Security + Tech Support."),
    ("TechSupport", lambda f: f["TechSupport"] == "No",
     "🛡️ Protection bundle", "Offer a free 3-month trial of Online Security + Tech Support."),
    ("MonthlyCharges", lambda f: f["MonthlyCharges"] >= 70,
     "💸 Price review", "Right-size the plan or apply a loyalty discount."),
    ("PaperlessBilling", lambda f: f["PaperlessBilling"] == "Yes",
     "📬 Billing touchpoint", "Send a clear, friendly monthly summary of the value they receive."),
]

FALLBACK = {"title": "✅ Keep them happy", "detail": "Low churn signals: enroll in the loyalty rewards program."}


def recommend(features: dict, reasons: list[dict], max_actions: int = 3) -> list[dict]:
    """features: engineered feature row; reasons: [{'feature', 'impact'}] from the explainer."""
    ranked = [r["feature"] for r in sorted(reasons, key=lambda r: -r["impact"]) if r["impact"] > 0]
    actions, seen = [], set()
    for feat in ranked:
        for driver, cond, title, detail in PLAYBOOK:
            if driver == feat and title not in seen and cond(features):
                actions.append({"title": title, "detail": detail, "driver": feat})
                seen.add(title)
        if len(actions) >= max_actions:
            break
    return actions or [FALLBACK]
