import requests
from io import StringIO
import pandas as pd
def getColumnsfromDatabase(file_name):
    """
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


def ReturnInputFeaturesModel(database_name):
    """
        args:
            database_name: Nome da base de dados

    """
    df=getColumnsfromDatabase("ciclone1_diario")
    input_features = [column for column in df.columns if column != "DATA"]
    features_add = ['ano', 'DATA_mes_sin', 'DATA_mes_cos', 'DATA_dia_sin',
        'DATA_dia_cos']
    return input_features + features_add




input_features = ReturnInputFeaturesModel('ciclone1_primario')


input = {}
print()
for values in input_features:
    input[values] = 10

# Dados de entrada-------------------------------------------------------------------
input_data = {
    "features": input,  # Ajuste para suas colunas reais
    "target_model": "ciclone1_primario_alimentacao_flotacao_finos_AFF.U3_-_CaO"  # Ajuste para o nome real
}

# Enviar requisição-----------------------------------------------------------
headers = {"Authorization": "Bearer meu-token-secreto-12345"}
response = requests.post("http://127.0.0.1:8000/predict", json=input_data, headers=headers)
print(response.json())  # Exemplo: {"prediction": 42.5}