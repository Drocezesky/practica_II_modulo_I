from app import models
from fastapi import FastAPI, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app.models import Base, Question
from sqlalchemy import func

app = FastAPI(title="Questions API", version="1.0.0")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"message": "Questions API funcionando", "endpoints": ["/questions", "/questions/{id}"]}


@app.get("/questions")
def list_questions(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    questions = db.query(Question).offset(skip).limit(limit).all()
    return questions


@app.get("/questions/{question_id}")
def get_question(question_id: int, db: Session = Depends(get_db)):
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        return {"error": "Pregunta no encontrada"}
    return question

@app.get("/questions/category/{category}")
def list_questions_by_category(category: str, skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    questions = db.query(Question).filter(Question.category == category).offset(skip).limit(limit).all()
    return questions

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total_questions = db.query(Question).count()

    category_counts = db.query(
        Question.category, 
        func.count(Question.id).label("count")
    ).group_by(Question.category).all()

    stats_by_category = {}
    for category, count in category_counts:
        nombre_categoria = category if category is not None else "Sin categoría"
        stats_by_category[nombre_categoria] = count

    return {
        "total_questions": total_questions,
        "category_stats": stats_by_category
    }

@app.post("/new_question", status_code=201)
def create_question(data: dict = Body(...), db: Session = Depends(get_db)):
    texto_pregunta = data.get("question")
    texto_respuesta = data.get("answer")
    categoria = data.get("category")
    fuente = data.get("source")

    if not texto_pregunta or not texto_respuesta:
        raise HTTPException(
            status_code=400, 
            detail="Los campos 'question' y 'answer' son obligatorios."
        )

    new_question = models.Question(
        question=texto_pregunta,
        answer=texto_respuesta,
        category=categoria,
        source=fuente
    )
    
    db.add(new_question)
    db.commit()
    db.refresh(new_question)
    
    return new_question

@app.put("/items/{item_id}")
def update_item(item_id: int, item: dict = Body(...), db: Session = Depends(get_db)):
    existing_item = db.query(Question).filter(Question.id == item_id).first()
    if not existing_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    existing_item.question = item.get("question", existing_item.question)
    existing_item.answer = item.get("answer", existing_item.answer)
    existing_item.category = item.get("category", existing_item.category)
    existing_item.source = item.get("source", existing_item.source)

    db.commit()
    db.refresh(existing_item)
    
    return existing_item

@app.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    existing_item = db.query(Question).filter(Question.id == item_id).first()
    if not existing_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    db.delete(existing_item)
    db.commit()
    
    return None

