from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def predict_complex_model(age: int,sex:str):
    # Assume a big and complex model here. For this test I am using a simple rule based model
    if age<10 or sex=='F':
        return {'survived':1}
    else:
        return {'survived':0}