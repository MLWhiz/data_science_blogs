from fastapi import FastAPI
from pydantic import BaseModel

class Input(BaseModel):
	age : int
	sex : str

app = FastAPI()

@app.put("/predict")
def predict_complex_model(d:Input):
	print("not processed")
	# Assume a big and complex model here. For this test I am using a simple rule based model
	if d.age<10 or d.sex=='F':
		return {'survived':1}
	else:
		return {'survived':0}