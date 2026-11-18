"""Local single-user API for document ingestion and evidence-based exploration."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config.settings import Settings
from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.query import QueryPipeline
from src.pipeline.conversation import ConversationManager
from src.storage.neo4j_client import Neo4jClient
from src.retrieval.atomic_tools import AtomicTools
from main import create_llm

ROOT = Path(__file__).resolve().parents[2]
settings = Settings.load(str(ROOT / 'config.yaml'))
DATA = ROOT / settings.paths.get('extracted_dir', 'extracted') / 'web'
DB = DATA / 'jobs.sqlite3'
executor = ThreadPoolExecutor(max_workers=1)
sessions = {}
session_lock = threading.Lock()

@contextmanager
def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()

def init_db():
    with connect() as db:
        db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, text TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, done INTEGER DEFAULT 0, total INTEGER DEFAULT 0, error TEXT DEFAULT "", updated TEXT DEFAULT CURRENT_TIMESTAMP)')
        db.execute("CREATE TABLE IF NOT EXISTS chat_sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP, updated TEXT DEFAULT CURRENT_TIMESTAMP)")
        db.execute("CREATE TABLE IF NOT EXISTS chat_turns (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, question TEXT NOT NULL, result TEXT NOT NULL, entities TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(session_id) REFERENCES chat_sessions(id))")
        db.execute("UPDATE jobs SET status='interrupted', error='服务中断，请重试', updated=CURRENT_TIMESTAMP WHERE status IN ('queued','running')")

def change(job_id, **values):
    allowed = {'status', 'stage', 'done', 'total', 'error'}
    assert set(values) <= allowed
    with connect() as db:
        db.execute('UPDATE jobs SET ' + ', '.join(f'{key}=?' for key in values) + ', updated=CURRENT_TIMESTAMP WHERE id=?', [*values.values(), job_id])

def work(job_id):
    try:
        with connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        change(job_id, status='running', stage='starting', error='')
        def progress(stage, done, total):
            change(job_id, stage=stage, done=done, total=total)
        IngestionPipeline(settings, create_llm(settings)).run_from_texts([row['text']], source_doc=row['name'], document_id=job_id, progress=progress, skip_resolution=True)
        change(job_id, status='completed', stage='completed')
    except Exception as exc:
        # Do not expose provider response bodies or secret values in the browser.
        import logging
        logging.getLogger(__name__).exception('Ingestion task failed')
        change(job_id, status='failed', error=f'{type(exc).__name__}: 入库失败，请检查服务端日志和配置')

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield
    executor.shutdown(wait=False, cancel_futures=True)

app = FastAPI(title='领域GraphRAG Local API', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'], allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type'])

class DocumentInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2_000_000)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


@app.get('/api/sessions')
def list_sessions():
    with connect() as db:
        return [dict(row) for row in db.execute('SELECT id,title,created,updated FROM chat_sessions ORDER BY updated DESC, rowid DESC')]


@app.post('/api/sessions')
def create_session():
    session_id = uuid.uuid4().hex
    with connect() as db:
        db.execute('INSERT INTO chat_sessions(id,title) VALUES (?,?)', (session_id, '新会话'))
    return {'id': session_id, 'title': '新会话'}


@app.get('/api/sessions/{session_id}')
def session_detail(session_id: str):
    with connect() as db:
        session = db.execute('SELECT id,title,created,updated FROM chat_sessions WHERE id=?', (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, '会话不存在')
        turns = db.execute('SELECT question,result FROM chat_turns WHERE session_id=? ORDER BY id', (session_id,)).fetchall()
    return {**dict(session), 'turns': [{'question': t['question'], 'result': json.loads(t['result'])} for t in turns]}


@app.delete('/api/sessions/{session_id}')
def delete_session(session_id: str):
    with session_lock, connect() as db:
        if not db.execute('SELECT 1 FROM chat_sessions WHERE id=?', (session_id,)).fetchone():
            raise HTTPException(404, '会话不存在')
        db.execute('DELETE FROM chat_turns WHERE session_id=?', (session_id,))
        db.execute('DELETE FROM chat_sessions WHERE id=?', (session_id,))
        sessions.pop(session_id, None)
    return {'deleted': True}

def client():
    c = settings.neo4j
    return Neo4jClient(uri=c.get('uri', 'bolt://localhost:7687'), user=c.get('user', 'neo4j'), password=c.get('password', ''), database=c.get('database', 'neo4j'), graph_name=c.get('graph', 'default'))

@app.get('/api/documents')
def documents():
    with connect() as db:
        return [dict(row) for row in db.execute('SELECT id,name,status,stage,done,total,error,updated FROM jobs ORDER BY updated DESC')]

@app.post('/api/documents')
def upload(doc: DocumentInput):
    name = Path(doc.name.replace('\\', '/')).name
    if not name.lower().endswith(('.txt', '.md')) or not doc.text.strip():
        raise HTTPException(400, '仅支持非空 TXT 或 Markdown 文件')
    job_id = hashlib.sha256((name + '\0' + doc.text).encode()).hexdigest()[:24]
    with connect() as db:
        existing = db.execute('SELECT status FROM jobs WHERE id=?', (job_id,)).fetchone()
        if existing:
            return {'id': job_id, 'status': existing['status'], 'existing': True}
        db.execute('INSERT INTO jobs(id,name,text,status,stage) VALUES(?,?,?,?,?)', (job_id,name,doc.text,'queued','queued'))
    executor.submit(work, job_id)
    return {'id': job_id, 'status': 'queued', 'existing': False}

@app.post('/api/documents/{job_id}/retry')
def retry(job_id: str):
    with connect() as db:
        result = db.execute("UPDATE jobs SET status='queued', stage='queued', error='', updated=CURRENT_TIMESTAMP WHERE id=? AND status IN ('failed','interrupted')", (job_id,))
        if result.rowcount == 0:
            raise HTTPException(409, '任务不存在或当前不能重试')
    executor.submit(work, job_id)
    return {'status': 'queued'}

@app.post('/api/ask')
def ask(body: Question):
    session_id = body.session_id or create_session()['id']
    with session_lock:
        with connect() as db:
            exists = db.execute('SELECT title FROM chat_sessions WHERE id=?', (session_id,)).fetchone()
            if not exists:
                raise HTTPException(404, '会话不存在，请新建会话')
            history = db.execute('SELECT question,result,entities FROM chat_turns WHERE session_id=? ORDER BY id DESC LIMIT 10', (session_id,)).fetchall()
        conv = ConversationManager()
        for turn in reversed(history):
            conv.add_turn(turn['question'], json.loads(turn['result']).get('answer', ''), json.loads(turn['entities']))
        resolved = conv.resolve_pronouns(body.question)
        try:
            pipeline = QueryPipeline(settings, create_llm(settings))
            try:
                result = pipeline.ask(resolved)
            finally:
                pipeline.close()
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Query failed')
            raise HTTPException(503, '问答服务不可用，请检查 Neo4j、模型和 LLM 配置')
        entities = [p['entry_entity'] for p in result.get('paths', []) if p.get('entry_entity')]
        conv.add_turn(body.question, result['answer'], entities)
        with connect() as db:
            db.execute('INSERT INTO chat_turns(session_id,question,result,entities) VALUES (?,?,?,?)', (session_id, body.question, json.dumps(result, ensure_ascii=False), json.dumps(entities, ensure_ascii=False)))
            db.execute("UPDATE chat_sessions SET title=CASE WHEN title='新会话' THEN ? ELSE title END, updated=CURRENT_TIMESTAMP WHERE id=?", (body.question[:30], session_id))
    result['session_id'] = session_id
    result['resolved_query'] = resolved
    return result


@app.get('/api/graph/overview')
def graph_overview():
    c = None
    try:
        c = client()
        nodes = c.run('MATCH (e:Entity {graph: $graph}) RETURN e.name AS name, e.type AS type, e.description AS description ORDER BY e.name LIMIT 80', {'graph': c.graph_name})
        names = [n['name'] for n in nodes]
        edges = c.run('MATCH (a:Entity {graph: $graph})-[r:RELATES|IS_A]->(b:Entity {graph: $graph}) WHERE a.name IN $names AND b.name IN $names RETURN a.name AS source, b.name AS target, coalesce(r.type, type(r)) AS label LIMIT 180', {'graph': c.graph_name, 'names': names}) if names else []
        return {'nodes': nodes, 'edges': edges, 'limited': len(nodes) == 80}
    except Exception:
        raise HTTPException(503, '图谱不可用，请检查 Neo4j 连接')
    finally:
        if c: c.close()


@app.get('/api/graph/node')
def graph_node(name: str):
    c = None
    try:
        c = client()
        rows = c.run("MATCH (e:Entity {graph:$graph, name:$name}) OPTIONAL MATCH (e)-[:APPEARS_IN]->(chunk:Chunk {graph:$graph}) RETURN e.name AS name, e.type AS type, e.description AS description, coalesce(e.aliases, []) AS aliases, collect(DISTINCT {chunk_id:chunk.chunk_id, source_doc:chunk.source_doc, text:chunk.text, position:chunk.position})[0..5] AS sources", {'graph': c.graph_name, 'name': name})
        if not rows:
            raise HTTPException(404, '实体不存在')
        rows[0]['sources'] = [s for s in rows[0]['sources'] if s.get('chunk_id')]
        return rows[0]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, '实体详情不可用，请检查 Neo4j 连接')
    finally:
        if c: c.close()

@app.get('/api/graph')
def graph(entity: str = ''):
    c = None
    try:
        c = client()
        tools = AtomicTools(c)
        matches = tools.search_entity_by_name(entity, limit=30) if entity else []
        paths = tools.get_n_hop_neighbors(entity, depth=1, limit=50) if entity and any(m['name'] == entity for m in matches) else []
        ids = list({i for p in paths for edge in p.get('evidence_ids', []) for i in edge})
        sources = c.run('MATCH (c:Chunk {graph: $graph}) WHERE c.chunk_id IN $ids RETURN c.chunk_id AS chunk_id, c.source_doc AS source_doc, c.text AS text, c.position AS position', {'graph': c.graph_name, 'ids': ids}) if ids else []
        return {'entities': matches, 'paths': paths, 'sources': sources}
    except Exception:
        raise HTTPException(503, '图谱不可用，请检查 Neo4j 连接')
    finally:
        if c: c.close()

@app.get('/api/status')
def status():
    c = None
    try:
        c = client()
        neo = {'status': 'available'}
    except Exception:
        neo = {'status': 'unavailable'}
    finally:
        if c: c.close()
    return {'neo4j': neo, 'llm': {'configured': bool(settings.llm.get('api_key')), 'model': settings.llm.get('default_model')}, 'embedding': {'model': settings.embedding.get('model_name'), 'local_files': Path(settings.embedding.get('model_name', '')).exists()}, 'reranker': {'model': settings.reranker.get('model_name'), 'local_files': Path(settings.reranker.get('model_name', '')).exists(), 'runtime': '查询时检测'}, 'domain': settings.domain}

static = ROOT / 'web' / 'dist'
if static.exists():
    app.mount('/', StaticFiles(directory=static, html=True), name='web')
