from pydantic import BaseModel


class TransactionData(BaseModel):
    No_Transactions: int
    No_Orders: int
    No_Payments: int
    Total_transaction_amt: float
    No_transactionsFail: int
    PaymentRegFail: int
    PaypalPayments: int
    ApplePayments: int
    CardPayments: int
    BitcoinPayments: int
    OrdersFulfilled: int
    OrdersPending: int
    OrdersFailed: int
    Trns_fail_order_fulfilled: int
    Duplicate_IP: int
    Duplicate_Address: int
    JCB_16: int
    AmericanExp: int
    VISA_16: int
    Discover: int
    Voyager: int
    VISA_13: int
    Maestro: int
    Mastercard: int
    DC_CB: int
    JCB_15: int