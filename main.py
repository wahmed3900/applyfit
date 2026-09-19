# ============================================================
# 1) IMPORTS
# ============================================================
import os
import json
import time
import uuid
import logging                      # ← #1: import logging ✅
import traceback
from typing import Optional

# ... (other imports) ...

# ============================================================
# 2) LOGGING  (must come BEFORE any log.… call)
# ============================================================
logging.basicConfig(              # ← #2: basicConfig ✅
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("applyfit") # ← #3: getLogger ✅

# ============================================================
# 3) CONFIG
# ============================================================
load_dotenv()                      # ← #4: load_dotenv ✅

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")  # ← #5 ✅
STRIPE_SECRET_KEY     = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
MONGODB_URI           = os.getenv("MONGODB_URI")

# ============================================================
# 4) REQUIRED ENV VAR GUARD
# ============================================================
if not ANTHROPIC_API_KEY:          # ← #6 ✅
    raise RuntimeError("ANTHROPIC_API_KEY is required")
