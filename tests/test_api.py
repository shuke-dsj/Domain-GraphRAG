import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import src.web.app as web


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = patch.object(web, 'DATA', Path(self.tmp.name))
        self.db = patch.object(web, 'DB', Path(self.tmp.name) / 'jobs.sqlite3')
        self.data.start(); self.db.start()
        self.submit = patch.object(web.executor, 'submit'); self.submit.start()
        self.client = TestClient(web.app); self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.submit.stop(); self.db.stop(); self.data.stop(); self.tmp.cleanup()

    def test_upload_idempotent_retry_and_validation(self):
        self.assertEqual(self.client.post('/api/documents', json={'name': 'a.pdf', 'text': 'x'}).status_code, 400)
        doc = {'name': 'a.txt', 'text': '测试文本'}
        first = self.client.post('/api/documents', json=doc).json()
        second = self.client.post('/api/documents', json=doc).json()
        self.assertFalse(first['existing']); self.assertTrue(second['existing'])
        self.assertEqual(web.executor.submit.call_count, 1)
        self.assertEqual(self.client.post('/api/documents/' + first['id'] + '/retry').status_code, 409)
        web.change(first['id'], status='failed', error='error')
        self.assertEqual(self.client.post('/api/documents/' + first['id'] + '/retry').status_code, 200)
        self.assertEqual(self.client.get('/api/documents').json()[0]['status'], 'queued')

    def test_query_failure_is_visible(self):
        with patch.object(web, 'create_llm', side_effect=RuntimeError('secret response')):
            response = self.client.post('/api/ask', json={'question': '问题'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret response', response.text)

    def test_sessions_persist_isolated_turns_and_delete(self):
        class FakePipeline:
            def __init__(self, *args): pass
            def ask(self, query):
                return {'answer': '回答 ' + query, 'paths': [{'entry_entity': '砂岩'}], 'sources': []}
            def close(self): pass
        a = self.client.post('/api/sessions').json()['id']
        b = self.client.post('/api/sessions').json()['id']
        with patch.object(web, 'create_llm', return_value=object()), patch.object(web, 'QueryPipeline', FakePipeline):
            self.assertEqual(self.client.post('/api/ask', json={'question':'砂岩是什么？','session_id':a}).status_code, 200)
            follow = self.client.post('/api/ask', json={'question':'它是什么？','session_id':a}).json()
            separate = self.client.post('/api/ask', json={'question':'它是什么？','session_id':b}).json()
        self.assertEqual(follow['resolved_query'], '砂岩是什么？')
        self.assertEqual(separate['resolved_query'], '它是什么？')
        self.assertEqual(len(self.client.get('/api/sessions/'+a).json()['turns']), 2)
        self.assertEqual(len(self.client.get('/api/sessions/'+b).json()['turns']), 1)
        self.assertEqual(self.client.delete('/api/sessions/'+a).status_code, 200)
        self.assertEqual(self.client.get('/api/sessions/'+a).status_code, 404)
        self.assertEqual(len(self.client.get('/api/sessions/'+b).json()['turns']), 1)


if __name__ == '__main__': unittest.main()
