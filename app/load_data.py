import pandas as pd
import requests
import tempfile
import os
from app.database import SessionLocal, engine
from app.models import Base, Question

DATASET_URL = "https://huggingface.co/datasets/llm-semantic-router/category-classifier-supplement/resolve/main/data/train-00000-of-00001.parquet"


def download_parquet(url: str) -> str:
    print(f"Descargando {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def load_questions():
    Base.metadata.create_all(bind=engine)

    parquet_path = download_parquet(DATASET_URL)
    df = pd.read_parquet(parquet_path)
    os.unlink(parquet_path)

    print(f"Columnas disponibles: {list(df.columns)}")
    print(f"Filas: {len(df)}")
    print(df.head(3))

    session = SessionLocal()
    try:
        for _, row in df.iterrows():
            question = Question(
                question=row.get("text", ""),
                answer=row.get("answer", None) or row.get("answer_alias", None),
                category=row.get("label", None),
                source=row.get("source", None),
            )
            session.add(question)

        session.commit()
        print(f"Se insertaron {len(df)} preguntas correctamente.")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    load_questions()

