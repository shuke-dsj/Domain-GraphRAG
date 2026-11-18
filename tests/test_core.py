import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import Reranker
from src.pipeline.query import QueryPipeline
from src.pipeline.ingestion import IngestionPipeline
from src.extraction.models import ExtractionResult, Entity, Relation

class CoreTests(unittest.TestCase):
    def test_path_direction_and_label(self):
        self.assertEqual(HybridSearchEngine._path_to_text(['砂岩','储层'], ['属于'], [False]), '砂岩 <-- 属于 -- 储层')

    def test_graph_isolation_and_reverse_path_deduplication(self):
        vector = SimpleNamespace(search_similar_entities=lambda *a, **kw: [{'name':'砂岩','score':.8}],
            search_similar_entities_fallback=lambda *a, **kw: [{'name':'砂岩','score':.8},{'name':'储层','score':.7}])
        engine = HybridSearchEngine(SimpleNamespace(graph_name='demo'), vector, top_k_entities=2)
        engine.tools = SimpleNamespace(get_n_hop_neighbors=lambda name, **kw:
            [{'node_chain':['砂岩','储层'] if name=='砂岩' else ['储层','砂岩'], 'edge_types':['属于'], 'directions':[name=='砂岩'], 'descriptions':[], 'evidence_ids':[['chunk']]}])
        paths = engine.search('问题')
        self.assertEqual(engine.last_mode, 'memory_vector')
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]['path_text'], '砂岩 -- 属于 --> 储层')

    def test_reranker_fallback_and_threshold(self):
        reranker = Reranker(); reranker._model = False
        result = reranker.rerank('问题', [{'path_text':'a','vector_score':.2},{'path_text':'b','vector_score':.8}], min_score=.9)
        self.assertEqual(result[0]['path_text'], 'b')
        self.assertEqual(reranker.last_mode, 'vector_fallback')
        reranker._model = SimpleNamespace(predict=lambda *a, **kw: [-2., -1.])
        self.assertEqual(reranker.rerank('问题', [{'path_text':'a'},{'path_text':'b'}], min_score=0), [])
        self.assertEqual(reranker.last_mode, 'cross_encoder')

    def test_query_reranks_once_and_requires_sources(self):
        p = QueryPipeline.__new__(QueryPipeline)
        p.settings = SimpleNamespace(retrieval={'top_n_paths':10,'reranker_threshold':0})
        p.vector_index = SimpleNamespace(embed_and_update_entities=lambda: None)
        p.hybrid_search = SimpleNamespace(search=lambda q:[{'path_text':'A→B','evidence_ids':[['c']]}], last_mode='neo4j_vector')
        p.reranker = Reranker(); p.reranker._model = SimpleNamespace(predict=MagicMock(return_value=[1.]))
        p.client = SimpleNamespace(graph_name='test', run=lambda *a, **kw:[{'chunk_id':'c','source_doc':'a.txt','text':'原文','position':1}])
        p.llm = SimpleNamespace(answer=MagicMock(return_value='回答 [c]'))
        result = p.ask('问题')
        self.assertEqual(p.reranker._model.predict.call_count, 1)
        self.assertEqual(result['sources'][0]['chunk_id'], 'c')
        self.assertIn('对应原文 [c]', result['context'])
        self.assertNotIn('[1]', result['context'])
        p.client.run=lambda *a, **kw: []
        self.assertEqual(p.ask('问题')['retrieval_status'], 'no_evidence')
        self.assertEqual(p.llm.answer.call_count, 1)

    def test_document_checkpoint_isolated_and_retry(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root,'domain').mkdir(); Path(root,'domain','d.yaml').write_text('terminology: []\n',encoding='utf8')
            settings=SimpleNamespace(paths={'extracted_dir':str(Path(root,'out')),'domain_config_dir':str(Path(root,'domain'))},domain='d.yaml',extraction={'chunk_target_tokens':5,'chunk_min_tokens':1},neo4j={},embedding={})
            pipeline=IngestionPipeline(settings,object())
            calls=[]
            class FakeExtractor:
                def __init__(self,*args): pass
                def extract(self,text,cid):
                    calls.append(cid)
                    if len(calls)==2: raise RuntimeError('interrupted')
                    return ExtractionResult(entities=[Entity(name='A',type='x',chunk_id=cid)],relations=[Relation(source='A',target='B',relation_type='属于',chunk_id=cid)])
            fake_client=MagicMock(); fake_vector=MagicMock()
            with patch('src.pipeline.ingestion.SchemaConstrainedExtractor',FakeExtractor), patch('src.pipeline.ingestion.Neo4jClient',return_value=fake_client), patch('src.pipeline.ingestion.GraphBuilder'), patch('src.pipeline.ingestion.VectorIndexManager',return_value=fake_vector):
                text='第一段内容比较长。\n\n第二段内容比较长。'
                with self.assertRaises(RuntimeError): pipeline.run_from_texts([text],source_doc='a.txt',document_id='doc1',skip_resolution=True)
                state=json.loads(Path(root,'out','documents','doc1','state.json').read_text(encoding='utf8'))
                self.assertEqual(len(state['results']),1)
                pipeline.run_from_texts([text],source_doc='a.txt',document_id='doc1',skip_resolution=True)
                state=json.loads(Path(root,'out','documents','doc1','state.json').read_text(encoding='utf8'))
                self.assertEqual(len(state['results']),len(state['chunks']))
                pipeline.run_from_texts([text],source_doc='b.txt',document_id='doc2',skip_resolution=True)
                other=json.loads(Path(root,'out','documents','doc2','state.json').read_text(encoding='utf8'))
                self.assertNotEqual(state['chunks'][0]['chunk_id'],other['chunks'][0]['chunk_id'])
                with self.assertRaises(ValueError): pipeline.run_from_texts(['different'],source_doc='a.txt',document_id='doc1',skip_resolution=True)

if __name__ == '__main__': unittest.main()
