"""
FastAPI tutorial lab — Part 2: Handling errors + bigger apps (APIRouter).

Official docs:
  https://fastapi.tiangolo.com/tutorial/handling-errors/
  https://fastapi.tiangolo.com/tutorial/bigger-applications/

Run from backend/ (port 8002 — Part 1 stays on 8001):

    .venv/bin/uvicorn labs.fastapi_lab_part2:app --reload --port 8002
    # open http://127.0.0.1:8002/docs

Try in order:
  1. GET  /shop/products/widget     → 200
  2. GET  /shop/products/missing    → 404 HTTPException
  3. GET  /shop/boom                → custom exception → 418 handler
  4. POST /shop/orders  {}          → 422 with body echoed (custom validation)
  5. GET  /admin/secret             → 401 without X-Lab-Token
  6. GET  /admin/secret + header    → 200 (router-level Depends)

How this maps to your REAL backend:
  routers/chat.py, routers/rag.py, routers/auth.py  ≈  APIRouter modules here
  main.py include_router(...)                       ≈  app.include_router(...)
  raise HTTPException(...) everywhere               ≈  section 1 below
  _http_error_for_llm in main.py                    ≈  "turn library errors into HTTP"

[JS] analogies in comments.
"""

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="FastAPI lab Part 2 — errors & routers")


# =============================================================================
# 1. HANDLING ERRORS — HTTPException
# =============================================================================
# [JS] throw new HttpException(404, "Item not found") in NestJS / return
#      res.status(404).json({ detail: "..." }) in Express.
# In FastAPI you RAISE — do not return — so nested helpers can abort the request.
#
# Docs: https://fastapi.tiangolo.com/tutorial/handling-errors/


class ProductOut(BaseModel):
    id: str
    name: str
    price: float


PRODUCTS = {
    "widget": ProductOut(id="widget", name="Widget", price=9.99),
    "gadget": ProductOut(id="gadget", name="Gadget", price=19.5),
}


# --- Custom domain error (not HTTP yet) ---
# [JS] class UnicornError extends Error { constructor(name) { ... } }
class StockError(Exception):
    """Business rule failed — we convert it to HTTP in an exception handler."""

    def __init__(self, product_id: str):
        self.product_id = product_id


@app.exception_handler(StockError)
async def stock_error_handler(request: Request, exc: StockError):
    # [JS] app.use((err, req, res, next) => { if (err instanceof StockError) ... })
    return JSONResponse(
        status_code=status.HTTP_418_IM_A_TEAPOT,  # silly code = easy to spot in lab
        content={
            "error": "stock_error",
            "message": f"No stock for '{exc.product_id}'",
            "path": request.url.path,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override the default 422 shape so learners can see the bad body.

    Real apps often keep the default; this is for debugging / teaching.
    Your main.py mostly relies on FastAPI's default 422 + HTTPException.
    """
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {
                "error": "validation_failed",
                "detail": exc.errors(),
                "body": exc.body,  # what the client actually sent
            }
        ),
    )


# =============================================================================
# 2. BIGGER APPLICATIONS — APIRouter
# =============================================================================
# [JS] Express: const shop = express.Router(); app.use("/shop", shop);
#      NestJS:  @Controller('shop') class ShopController { ... }
#
# Split by feature into modules, then mount them on the app.
# Your real backend already does this:
#   create_chat_router() / create_rag_router() / create_auth_router()
#
# Docs: https://fastapi.tiangolo.com/tutorial/bigger-applications/

shop_router = APIRouter(
    prefix="/shop",  # all routes below start with /shop
    tags=["shop"],  # groups them in /docs
)


@shop_router.get("/products/{product_id}", response_model=ProductOut)
def get_product(product_id: str) -> ProductOut:
    product = PRODUCTS.get(product_id)
    if product is None:
        # raise = throw. FastAPI turns this into JSON { "detail": "..." }
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found",
            headers={"X-Error": "product-missing"},  # optional custom header
        )
    return product


@shop_router.get("/boom")
def trigger_stock_error():
    """Raises StockError → handled by stock_error_handler (418)."""
    raise StockError("widget")


class OrderIn(BaseModel):
    product_id: str = Field(..., min_length=1)
    qty: int = Field(..., ge=1, le=100)


@shop_router.post("/orders")
def create_order(body: OrderIn):
    """Invalid JSON body → our custom 422 handler (includes `body`)."""
    if body.product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Unknown product for order")
    return {"ok": True, "ordered": body.product_id, "qty": body.qty}


# --- Router-level dependency (runs for EVERY route on this router) ---
# [JS] shop.use(requireLabToken) before route handlers.


def require_lab_token(x_lab_token: str | None = Header(default=None)) -> str:
    if x_lab_token != "lab-secret":
        raise HTTPException(
            status_code=401,
            detail="Missing or wrong X-Lab-Token header (use: lab-secret)",
        )
    return x_lab_token


admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_lab_token)],  # applies to all admin routes
)


@admin_router.get("/secret")
def admin_secret():
    return {"message": "welcome, admin"}


@admin_router.get("/ping")
def admin_ping():
    return {"pong": True}


# Mount routers on the app — same idea as main.py include_router(...)
app.include_router(shop_router)
app.include_router(admin_router)


@app.get("/")
def root():
    return {
        "lab": "part2-errors-routers",
        "try": [
            "GET /shop/products/widget",
            "GET /shop/products/nope",
            "GET /shop/boom",
            "POST /shop/orders with {}",
            "GET /admin/secret with header X-Lab-Token: lab-secret",
        ],
        "real_app_map": {
            "APIRouter": "backend/routers/*.py",
            "include_router": "main.py bottom",
            "HTTPException": "chat/rag/auth routes + helpers",
        },
    }
