from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import requests
import pandas as pd
from io import StringIO
import numpy as np

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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def getColumnsfromDatabase(file_name):
    """
    Retorna as colunas de uma tabela a partir do nome, refere-se especificamente a tabelas com dados diários
    args:
        file_name: Nome do arquivo que refere-se a base dados de dados
    """
    url = (
        f"https://samarcodatalake.blob.core.windows.net/dbw-manutencao/"
        f"{file_name}.csv?"
        "sp=rle&st=2025-03-25T17:20:01Z&se=2032-01-01T01:20:01Z&spr=https&"
        "sv=2024-11-04&sr=c&sig=m34YG3Dox614y9SiSNJWVnGsgejevMIp7EfUeX0riyM%3D"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()  # Lança erro se status != 200
        csv_string = StringIO(response.text)
        df = pd.read_csv(csv_string)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o arquivo: {e}")

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def ReturnInputFeaturesModel(database_name):
    """
    Adiciona algumas features a mais para pegar o nome de todas as colunas de dados diários.
    Retorna as Features de Entrada do Modelo Filtrado.
        args:
            database_name: Nome da base de dados

    """
    df=getColumnsfromDatabase(database_name)
    input_features = [column for column in df.columns if column != "DATA"]
    features_add = ['ano', 'DATA_mes_sin', 'DATA_mes_cos', 'DATA_dia_sin',
        'DATA_dia_cos']
    return input_features + features_add



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------



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

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def GetAllModelfromPath(path, input, output, y):
    url = (
        f"https://samarcodatalake.blob.core.windows.net/dbw-manutencao/"
        f"{path}/{input}_{output}_{y}.joblib?"
        "sp=rle&st=2025-03-25T17:20:01Z&se=2032-01-01T01:20:01Z&spr=https&"
        "sv=2024-11-04&sr=c&sig=m34YG3Dox614y9SiSNJWVnGsgejevMIp7EfUeX0riyM%3D"
    )

    # Criar a pasta 'models' se ela não existir
    os.makedirs(path, exist_ok=True)

    # Tentar fazer o download
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lança erro se status != 200
        file_path = os.path.join(path, f"{input}_{output}_{y}.joblib")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Arquivo '{file_path}' baixado com sucesso!")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o arquivo: {e}")


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
#-----------------------------------------------------------------------------------------------------------
def FilteredModels(models):
    """
    Recebe como entrada um dicionário de modelos e retorna os nomes dos modelos que atendem às condições.
    args:
        models: Dicionário com os modelos treinados
    return:
        Lista com os nomes dos modelos filtrados
    """
    filtered_model_names = []
    for model_name in models.keys():
        if models[model_name].intercept_ > -10:
            print(f"{model_name}: {models[model_name].intercept_}")
            filtered_model_names.append(model_name)
    return filtered_model_names

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

DownloadModels(r"models\3_1","cicl1_prim","alim_flot_finos","alimentacaoflotacaofines_diario")
DownloadModels(r"models\3_2","cicl2_prim","alim_flot_finos","alimentacaoflotacaofines_diario")

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

def LoadModels(path_name):
    """
    Função para carregar todos os modelos a partir de um pathname. Será utilizada posteriormente em predict.
    args:
        path_name: Caminho onde está os modelos.
    """
    MODEL ={}
    for file_name in os.listdir(path_name):
        if file_name.endswith(".joblib"):
            model_path = os.path.join(path_name, file_name)
            model_name = file_name.replace(".joblib", "")  # Ex.: "ciclone1_primario_alimentacao_flotacao_finos_colunaX"
            try:
                MODEL[model_name] = joblib.load(model_path)
                print(f"Modelo carregado: {model_name}")
            except Exception as e:
                print(f"Erro ao carregar {model_name}: {e}")

    return MODEL

#Carregando todos os  Modelos----------------------------------------------------------------------
model_3_1=LoadModels("models/3_1")
model_3_2=LoadModels("models/3_2")




#Rota que retorna os coeficientes do modelo de regresse: [intercept,x1,x2,...,xn]
class InputGetParams(BaseModel):
    class_model:str
    target_model:str
@app.get('/params',dependencies=[Depends(verify_token)])
def params(data:InputGetParams):

    class_model = data.class_model
    target_model = data.target_model

     
    if class_model not in ["3_1","3_2"]:
        raise HTTPException(status_code=404, detail=f"Classe de modelo {class_model} não encontrado!")
    
    if class_model == "3_1":
            MODELS = model_3_1
    elif class_model == "3_2":
            MODELS = model_3_2
    else:
        pass

    if target_model not in MODELS.keys():
        raise HTTPException(status_code=404, detail=f"Modelo  {target_model} não encontrado!")
    
    try:
        params = np.concatenate([[MODELS[target_model].intercept_],MODELS[target_model].coef_])
        print("parametro",params)
        return {'params':params.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao retornar parâmetros dos modelos filtrados: {e}")
    



# Definir o modelo de dados esperado do usuário
class InputData(BaseModel):
    features: dict  # Dados de entrada como dicionário (ex.: {"col1": 10.5, "col2": 20.3})
    class_model: str
    target_model: str  # Nome do modelo a ser usado (ex.: "ciclone1_primario_alimentacao_flotacao_finos_colunaX")



@app.post("/predict", dependencies=[Depends(verify_token)])
def predict(data: InputData):
    # Extrair os dados de entrada e o modelo solicitado
    input_dict = data.features# Contém as feautres de entrada do modelo
    class_model = data.class_model# Refere=se a qual tipo de subpasta em models está o modelo ex: models\3_2,models\3_1
    target_model = data.target_model#Refere-se ao modelo específico dentro da pasta model
    
    #Carrega os modelos de acorod com class_model
    
    if class_model == "3_1":
        MODELS = model_3_1
    elif class_model == "3_2":
        MODELS = model_3_2
    else:
        pass



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
    

class InputClassFilteredModel(BaseModel):
    class_model: str   
@app.get("/filtered-models", dependencies=[Depends(verify_token)])
def get_filtered_models(data: InputClassFilteredModel):
    class_model = data.class_model
    # Verificar se a classe de modelo existe
    if class_model not in ["3_1","3_2"]:
        raise HTTPException(status_code=404, detail=f"Classe de modelo {class_model} não encontrado!")
    
    try:
        if class_model == "3_1":
            filtered_models = FilteredModels(model_3_1)
           
        elif class_model == "3_2":
            filtered_models = FilteredModels(model_3_2)

        return {"filtered_models":filtered_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao retornar modelos filtrados: {e}")
    




class InputDataModel(BaseModel):
    database:str #Refere-se a subpasta dentro de models
@app.get("/features", dependencies=[Depends(verify_token)])
def get_filtered_features(data: InputDataModel):
    database = data.database
    try:
        input_features = ReturnInputFeaturesModel(database)
        return {"features": input_features}  # Retorna as features calculadas
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nome de tabela informado errado ou não existe! Não foi possível retornar os dados corretamente: {e}")


# Rodar a API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)