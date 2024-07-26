from pydantic import BaseModel
from typing import Literal

class AntiMoneyLaunderingInput(BaseModel):
    Time: str
    Date: str
    Sender_account: int
    Receiver_account: int
    Amount: float
    Payment_currency: str
    Received_currency: str
    Sender_bank_location: str
    Receiver_bank_location: str
    Payment_type: str
    Laundering_type: Literal[
        'Single_large', 'Smurfing', 'Stacked_Bipartite', 'Structuring'
    ]
