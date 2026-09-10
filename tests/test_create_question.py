import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app import models


# ---------------------------------------------------------------------
# Base de datos de prueba: SQLite en memoria
# ---------------------------------------------------------------------

SQLALCHEMY_TEST_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function", autouse=True)
def setup_database():
    """Crea las tablas antes de cada test y las tira abajo después.
    Así cada test arranca con la base limpia, sin depender de datos
    que hayas cargado con load_data.py."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    """TestClient con get_db sobreescrito para usar SQLite en memoria
    en lugar de conectarse a Postgres en localhost:5432."""

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------
# Casos felices
# ---------------------------------------------------------------------

def test_create_question_success(client):
    """Con todos los campos, debe crear la pregunta y devolver 201."""
    payload = {
        "question": "¿Qué es un JOIN en SQL?",
        "answer": "Una operación que combina filas de dos o más tablas.",
        "category": "SQL",
        "source": "Apunte clase 4",
    }

    response = client.post("/new_question", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["question"] == payload["question"]
    assert body["answer"] == payload["answer"]
    assert body["category"] == payload["category"]
    assert body["source"] == payload["source"]
    assert "id" in body


def test_create_question_without_optional_fields(client):
    """category y source son nullable=True en el modelo: deben poder faltar."""
    payload = {
        "question": "¿Qué es una clave foránea?",
        "answer": "Una columna que referencia la clave primaria de otra tabla.",
    }

    response = client.post("/new_question", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["category"] is None
    assert body["source"] is None

# ---------------------------------------------------------------------
# test de integración: verifica que la pregunta realmente quedó guardada en la base
# ---------------------------------------------------------------------

def test_create_question_persists_in_db(client):
    """Verifica que el registro realmente haya quedado guardado,
    consultándolo con una sesión nueva (no solo confiando en la respuesta HTTP)."""
    payload = {
        "question": "¿Qué es un índice en una base de datos?",
        "answer": "Una estructura que acelera las búsquedas.",
    }

    response = client.post("/new_question", json=payload)
    created_id = response.json()["id"]

    db = TestingSessionLocal()
    try:
        saved = db.query(models.Question).filter(models.Question.id == created_id).first()
        assert saved is not None
        assert saved.question == payload["question"]
    finally:
        db.close()


# ---------------------------------------------------------------------
# Casos de error / validación
# ---------------------------------------------------------------------

def test_create_question_missing_question_field(client):
    payload = {"answer": "Una respuesta cualquiera."}

    response = client.post("/new_question", json=payload)

    assert response.status_code == 400
    assert "obligatorios" in response.json()["detail"]


def test_create_question_missing_answer_field(client):
    payload = {"question": "¿Pregunta sin respuesta?"}

    response = client.post("/new_question", json=payload)

    assert response.status_code == 400


def test_create_question_missing_both_fields(client):
    payload = {"category": "General"}

    response = client.post("/new_question", json=payload)

    assert response.status_code == 400


def test_create_question_empty_strings_are_treated_as_missing(client):
    """Caso borde: '' es falsy en Python, 'not texto_pregunta' lo trata
    igual que None. Este test documenta ese comportamiento actual."""
    payload = {"question": "", "answer": ""}

    response = client.post("/new_question", json=payload)

    assert response.status_code == 400


def test_create_question_empty_body(client):
    response = client.post("/new_question", json={})

    assert response.status_code == 400
