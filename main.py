from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import requests
import pandas as pd
from io import StringIO

app = FastAPI()

# Token de autenticação (substitua por algo único e secreto)
API_TOKEN = "meu-token-secreto-12345"
token_header = APIKeyHeader(name="Authorization")

def verify_token(token: str = Depends(token_header)):
    if token != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Token inválido")
    return token

# Diretório onde os modelos .joblib estão salvos (ajuste conforme necessário)
MODEL_DIR = "./models"



def GetDataBase(file_name):
    url = f"https://samarcodatalake.blob.core.windows.net/dbw-manutencao/{file_name}.csv?sp=rle&st=2025-03-25T17:20:01Z&se=2032-01-01T01:20:01Z&spr=https&sv=2024-11-04&sr=c&sig=m34YG3Dox614y9SiSNJWVnGsgejevMIp7EfUeX0riyM%3D"

    response = requests.get(url)
    if response.status_code == 200:
        print("Deu certo")
        csv_string = StringIO(response.text)
        df = pd.read_csv(csv_string)
        return df
    else:
        print(f"Erro ao baixar o arquivo: {response.status_code}")
        return None

def GetAllModelfromPath(path, input, output, y):
    url = (
        f"https://samarcodatalake.blob.core.windows.net/dbw-manutencao/"
        f"{path}/{input}_{output}_{y}.joblib?"
        "sp=rle&st=2025-03-25T17:20:01Z&se=2032-01-01T01:20:01Z&spr=https&"
        "sv=2024-11-04&sr=c&sig=m34YG3Dox614y9SiSNJWVnGsgejevMIp7EfUeX0riyM%3D"
    )

    # Criar a pasta 'models' se ela não existir
    os.makedirs("models", exist_ok=True)

    # Tentar fazer o download
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lança erro se status != 200
        file_path = os.path.join("models", f"{input}_{output}_{y}.joblib")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Arquivo '{file_path}' baixado com sucesso!")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o arquivo: {e}")



def DownloadModels(path,input_name,output_name,input_table_name):
    """
    args:
        path: Caminho onde se encontra os modelos
        input_name: nome do ponto de amostra de entrada
        output_name: nome do ponto de amostra de saida
        input_table_name: nome da tabela com as variaveis input do kolo
    
    """
    df=GetDataBase(input_table_name)

    columns = [column for column in df.columns if column != "DATA"]
    for column in columns:
        #GetAllModelfromPath("models/3_1","ciclone1_primario","alimentacao_flotacao_finos",column)
        GetAllModelfromPath(path,input_name,output_name,column)



DownloadModels("models/3_1","ciclone1_primario","alimentacao_flotacao_finos","alimentacaoflotacaofines_diario")


# Carregar todos os modelos disponíveis ao iniciar a API
MODELS = {}
for file_name in os.listdir(MODEL_DIR):
    if file_name.endswith(".joblib"):
        model_path = os.path.join(MODEL_DIR, file_name)
        model_name = file_name.replace(".joblib", "")  # Ex.: "ciclone1_primario_alimentacao_flotacao_finos_colunaX"
        try:
            MODELS[model_name] = joblib.load(model_path)
            print(f"Modelo carregado: {model_name}")
        except Exception as e:
            print(f"Erro ao carregar {model_name}: {e}")

# Definir o modelo de dados esperado do usuário
class InputData(BaseModel):
    features: dict  # Dados de entrada como dicionário (ex.: {"col1": 10.5, "col2": 20.3})
    target_model: str  # Nome do modelo a ser usado (ex.: "ciclone1_primario_alimentacao_flotacao_finos_colunaX")

@app.post("/predict", dependencies=[Depends(verify_token)])
def predict(data: InputData):
    # Extrair os dados de entrada e o modelo solicitado
    input_dict = data.features
    target_model = data.target_model

    # Verificar se o modelo existe
    if target_model not in MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo '{target_model}' não encontrado. Modelos disponíveis: {list(MODELS.keys())}")

    # Converter os dados de entrada para um DataFrame Pandas
    try:
        input_df = pd.DataFrame([input_dict])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar os dados de entrada: {e}")

    # Fazer a previsão com o modelo selecionado
    model = MODELS[target_model]
    try:
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}  # Retorna a previsão como float
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer previsão: {e}")

# Rodar a API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)