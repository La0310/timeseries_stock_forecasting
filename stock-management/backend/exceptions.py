"""
exceptions.py
Domain exceptions for the PO lifecycle service.

These are plain Python exceptions — no HTTP or FastAPI knowledge here.
The router layer is responsible for translating these into HTTP status codes.

Why custom exceptions instead of HTTPException directly in the service?
  - Services can be called from CLI tools, background jobs, tests, etc.
    without a running FastAPI app.
  - Keeps the boundary between "what went wrong" and "how to tell the client"
    at the router layer, where it belongs.
"""


class PONotFound(Exception):
    """Raised when a PO ID does not exist in the database."""
    def __init__(self, po_id: str):
        self.po_id = po_id
        super().__init__(f"PO '{po_id}' not found")


class POInvalidTransition(Exception):
    """
    Raised when an operation is not valid for the PO's current status.
    Example: receiving stock on a CANCELLED PO, or cancelling a RECEIVED PO.
    """
    def __init__(self, po_id: str, current_status: str, operation: str):
        self.po_id = po_id
        self.current_status = current_status
        self.operation = operation
        super().__init__(
            f"PO '{po_id}' has status '{current_status}' — cannot {operation}"
        )


class POInvalidQuantity(Exception):
    """
    Raised when quantity_received is outside the valid range [1, remaining_qty].
    """
    def __init__(self, po_id: str, quantity_received: int, remaining: int):
        self.po_id = po_id
        self.quantity_received = quantity_received
        self.remaining = remaining
        super().__init__(
            f"quantity_received {quantity_received} is out of range "
            f"[1, {remaining}] for PO '{po_id}'"
        )
