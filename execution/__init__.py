"""
execution — IBKR execution layer for the SRV Quant System.

Modules:
    ibkr_gateway   : IBKRGateway (connection + order management) and SignalExecutor
    order_manager  : OrderManager (persistent order tracking and daily summaries)
"""

from execution.ibkr_gateway import IBKRGateway, SignalExecutor
from execution.order_manager import OrderManager

__all__ = ["IBKRGateway", "SignalExecutor", "OrderManager"]
