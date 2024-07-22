from pydantic import BaseModel

class AntiMoneyLaunderingInput(BaseModel):
        Sender_account :int
        Receiver_account:int
        Amount:float
        Year:int
        Month:int 
        Laundering_type_Single_large: bool
        Laundering_type_Smurfing:bool
        Laundering_type_Stacked_Bipartite :bool
        Laundering_type_Structuring:bool
        Is_laundering:int
        Length: 106
        dtype: object