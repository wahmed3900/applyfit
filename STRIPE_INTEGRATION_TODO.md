# Stripe Integration — TODO

This file is the single source of truth for the Stripe Checkout integration
in `main.py`. An existing Checkout Session call was found (Scenario A), so
only its parameters were updated — no new files, routes, or refactoring.

## Values to Replace

None. The three sample-style fields Checkout Studio flags (`success_url`,
`cancel_url`, `line_items[].price`) already have real, non-placeholder values
in this codebase, sourced from the request body and the `STRIPE_PRICE_ID`
environment variable, so they were left as-is.

**File:** [main.py](main.py) — `create_checkout_session()`

## Configured Parameters

These parameters were set from Checkout Studio's UI configuration.

**File:** [main.py](main.py) — `create_checkout_session()`

| Parameter | Value |
|-----------|-------|
| ui_mode | hosted (SDK < 21.0.0 — see note below) |
| billing_address_collection | auto |
| phone_number_collection.enabled | false |
| automatic_tax.enabled | false |
| allow_promotion_codes | false |
| payment_method_collection | always |
| submit_type | auto |
| integration_identifier | hosted_mobile_app_0002 |
| origin_context | mobile_app |

## Deliberately left unchanged

Two things Checkout Studio's generic instructions would otherwise have
touched were intentionally left alone, since removing them would break
working functionality specific to this app:

- **`customer_email` and `metadata.email`** — not part of Checkout Studio's
  field intents, but essential here: the `/webhook` handler matches incoming
  Stripe events back to a subscriber record in MongoDB by email. Removing
  these would break subscription activation entirely.
- **`payment_method_types=["card"]`** — not part of Checkout Studio's field
  intents either. Left as-is since removing it changes which payment methods
  customers see, which wasn't something this task asked for.

## Setup and next steps

- **Environment variables** (unchanged by this task, already documented
  elsewhere in this repo): `ANTHROPIC_API_KEY`, `STRIPE_SECRET_KEY`,
  `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `MONGODB_URI`.
- **`ui_mode` depends on your installed Stripe SDK version.**
  `requirements.txt` currently pins `stripe>=10.0` (unpinned upper bound);
  the version actually installed in the last deploy was `15.6.1`, which is
  below the 21.0.0 threshold, so `ui_mode="hosted"` is correct today. If you
  later upgrade the `stripe` package past 21.0.0, change this to
  `ui_mode="hosted_page"`.
- **Unfamiliar parameters flagged for testing:** `integration_identifier`
  and `origin_context` are not parameters covered in this assistant's
  training data as of its last update — they were added exactly as specified
  by Checkout Studio's field intents, but haven't been verified against a
  live Stripe API call. The existing `try/except stripe.error.StripeError`
  block will surface a clear 400 error with Stripe's own message if either
  parameter is rejected. **Test `/create-checkout-session` end-to-end after
  deploying this change** (e.g. with Stripe's test card `4242 4242 4242
  4242`) to confirm the session is created successfully before relying on it.
- **Testing:** use Stripe test mode with card `4242 4242 4242 4242`, any
  future expiry, any CVC.
- **Resources:** https://support.stripe.com and https://docs.stripe.com/mcp
