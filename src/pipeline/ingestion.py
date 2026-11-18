from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import threading

import yaml
from src.extraction.chunker import SmartChunker
from src.extraction.boundary_defender import BoundaryDefender
from src.extraction.llm_extractor import SchemaConstrainedExtractor
from src.storage.neo4j_client import Neo4jClient
from src.storage.graph_builder import GraphBuilder
from src.storage.vector_index import VectorIndexManager

logger = logging.getLogger(__name__)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class IngestionPipeline:
    """Document-scoped extraction checkpoints; legacy extracted files are untouched."""
    def __init__(self, settings, llm):
        self.settings, self.llm = settings, llm
        path = Path(settings.paths.get('domain_config_dir', 'domain_config')) / settings.domain
        with path.open(encoding='utf-8') as f:
            self.domain_config = yaml.safe_load(f) or {}

    def run_from_texts(self, texts, source_doc='', skip_resolution=False, progress=None, document_id=None):
        results = []
        for index, text in enumerate(texts):
            source = source_doc or f'文档-{index + 1}'
            doc_id = document_id if document_id and len(texts) == 1 else hashlib.sha256((source + '\0' + text).encode()).hexdigest()[:24]
            results.append(self._run_document(text, source, doc_id, progress, skip_resolution))
        return results

    def _run_document(self, text, source, doc_id, progress, skip_resolution):
        if not text.strip():
            raise ValueError('文档内容为空')
        def update(stage, done=0, total=0):
            if progress:
                progress(stage, done, total)
        directory = Path(self.settings.paths.get('extracted_dir', 'extracted')) / 'documents' / doc_id
        state_path = directory / 'state.json'
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if state.get('text_hash') != text_hash or state.get('source_doc') != source:
                raise ValueError('文档标识与已有数据不匹配')
        else:
            chunker = SmartChunker(target_tokens=self.settings.extraction.get('chunk_target_tokens', 1500), min_tokens=self.settings.extraction.get('chunk_min_tokens', 512))
            chunks = []
            for i, chunk in enumerate(chunker.chunk(text, source_doc=source)):
                item = chunk.model_dump()
                item.update(chunk_id=hashlib.sha256(f'{doc_id}:{i}:{chunk.text}'.encode()).hexdigest()[:24], document_id=doc_id, position=i + 1)
                chunks.append(item)
            state = {'document_id': doc_id, 'source_doc': source, 'text_hash': text_hash, 'chunks': chunks, 'results': {}}
            atomic_json(state_path, state)
        chunks = state['chunks']
        update('extracting', len(state['results']), len(chunks))
        rels = self.domain_config.get('relation_types') or {}
        extractor = SchemaConstrainedExtractor(self.llm, BoundaryDefender(self.domain_config.get('terminology') or []), self.domain_config.get('entity_types') or [], rels.get('mandatory') or [], rels.get('allow_free', True))
        for chunk in chunks:
            cid = chunk['chunk_id']
            if cid in state['results']:
                continue
            result = extractor.extract(chunk['text'], cid)
            state['results'][cid] = {'entities': [e.model_dump() for e in result.entities], 'relations': [r.model_dump() for r in result.relations]}
            atomic_json(state_path, state)
            update('extracting', len(state['results']), len(chunks))
        entities = [x for result in state['results'].values() for x in result['entities']]
        relations = [x for result in state['results'].values() for x in result['relations']]
        update('building', len(chunks), len(chunks))
        config = self.settings.neo4j
        client = Neo4jClient(uri=config.get('uri', 'bolt://localhost:7687'), user=config.get('user', 'neo4j'), password=config.get('password', ''), database=config.get('database', 'neo4j'), graph_name=config.get('graph', 'default'))
        try:
            client.init_schema()
            builder = GraphBuilder(client)
            builder._write_chunks(chunks)
            builder._write_entities(entities)
            builder._write_relations(relations)
            update('embedding', len(chunks), len(chunks))
            vector = VectorIndexManager(client, model_name=self.settings.embedding.get('model_name', 'paraphrase-multilingual-MiniLM-L12-v2'), device=self.settings.embedding.get('device', 'cpu'))
            vector.embed_and_update_entities()
            vector.create_index()
            # Decisions are persisted for inspection, never applied to the graph.
            if not skip_resolution:
                update('resolution_review', len(chunks), len(chunks))
                from src.resolution.coarse_clustering import CoarseClusterer
                from src.resolution.frequency import FrequencyAnalyzer
                from src.resolution.llm_disambiguator import LLMDisambiguator
                unique = {e['name']: e for e in entities if e.get('name')}
                clusterer = CoarseClusterer(model_name=vector.model_name, device=vector.device,
                    threshold=self.domain_config.get('disambiguation', {}).get('coarse_similarity_threshold', 0.5))
                frequencies = FrequencyAnalyzer(chunks)
                judge = LLMDisambiguator(self.llm)
                suggestions = []
                for cluster in clusterer.cluster(list(unique.values())):
                    if len(cluster) > 1:
                        for offset in range(0, len(cluster), 20):
                            group = cluster[offset:offset + 20]
                            if len(group) > 1:
                                suggestions.extend(judge.decide(group, frequencies.get_frequencies(group)))
                atomic_json(directory / 'resolution_suggestions.json', suggestions)
        finally:
            client.close()
        update('completed', len(chunks), len(chunks))
        return {'document_id': doc_id, 'chunks': len(chunks), 'entities': len(entities), 'relations': len(relations)}
