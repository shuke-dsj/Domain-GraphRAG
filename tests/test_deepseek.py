import unittest
from unittest.mock import patch
from src.llm.deepseek import DeepSeekAdapter


class DeepSeekTests(unittest.TestCase):
    @patch('src.llm.deepseek.requests.post')
    def test_official_requests_disable_thinking(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {'choices': [{'message': {'content': '{"entities": [], "relations": []}'}}]}
        model = DeepSeekAdapter('test', 'https://api.deepseek.com', 'deepseek-v4-pro')
        model.chat([{'role': 'user', 'content': 'test'}])
        self.assertEqual(post.call_args.kwargs['json']['thinking'], {'type': 'disabled'})
        with patch.object(model, '_save_response_log'):
            model.extract_entities('test')
        self.assertEqual(post.call_args.kwargs['json']['thinking'], {'type': 'disabled'})

    @patch('src.llm.deepseek.requests.post')
    def test_never_use_reasoning_as_answer(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {'choices': [{'message': {'content': '', 'reasoning_content': 'private'}}]}
        model = DeepSeekAdapter('test', 'https://api.deepseek.com', 'deepseek-v4-pro')
        with self.assertRaises(RuntimeError):
            model.chat([{'role': 'user', 'content': 'test'}])

    @patch('src.llm.deepseek.requests.post')
    def test_other_compatible_endpoints_do_not_receive_deepseek_specific_field(self, post):
        post.return_value.status_code = 200
        post.return_value.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
        DeepSeekAdapter('test', 'https://api.siliconflow.cn/v1').chat([{'role': 'user', 'content': 'test'}])
        self.assertNotIn('thinking', post.call_args.kwargs['json'])


if __name__ == '__main__': unittest.main()
