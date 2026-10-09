import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile
import contextlib
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from protocol import ROOT, MODELS, CONDITIONS, make_request, render_fixture, project_response, parse_verdict, digest, load_reference

class PublicExtensionTests(unittest.TestCase):
    def test_same_payload_every_condition(self):
        row = {'source': 'The red vessel holds 4 units.', 'original': 'The vessel holds 4 units.', 'edited': 'The vessel holds 5 units.'}
        payloads = [make_request(row, c)[1]['content'].split('\n', 1)[1] for c in CONDITIONS]
        self.assertEqual(len(set(payloads)), 1)
        self.assertEqual(set(json.loads(payloads[0])), {'source_document', 'edited_summary', 'original_summary', 'token_differences'})

    def test_native_family_rendering_contract(self):
        messages = [{'role': 'system', 'content': 'System.'}, {'role': 'user', 'content': 'User.'}]
        qwen = render_fixture(messages, MODELS[0])
        llama = render_fixture(messages, MODELS[1])
        self.assertTrue(qwen.endswith('<think>\n\n</think>\n\n'))
        self.assertEqual(llama.count('<|begin_of_text|>'), 1)
        self.assertIn('Today Date: 09 Oct 2026', llama)
        self.assertNotIn('<think>', llama)
        self.assertTrue(llama.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))

    def test_malformed_projection_does_not_rescue_accuracy(self):
        row = {'raw': 'Unlicensed source quotation', 'verdict': None, 'parse_error': 'invalid_json', 'label': 1,
               'decoded_output': 'Unlicensed source quotation', 'output_token_ids': [10, 11], 'decode_matches_raw': True}
        public = project_response(row)
        self.assertEqual(public['raw'], '')
        self.assertTrue(public['raw_redacted'])
        self.assertEqual(public['raw_sha256'], digest(row['raw']))
        self.assertEqual(public['parse_error'], row['parse_error'])
        self.assertNotIn('decoded_output', public)
        self.assertNotIn('output_token_ids', public)
        self.assertEqual(load_reference('analyze').correct(public), 0)

    def test_valid_json_and_diagnostic_mismatch_preserved(self):
        row = {'raw': ' {"verdict":"yes"}\n', 'verdict': 'yes', 'parse_error': None, 'label': 1,
               'seconds_unrounded': 1.23456789, 'input_tokens': 120, 'output_tokens': 7, 'decode_matches_raw': False}
        public = project_response(row)
        for field, value in row.items():
            self.assertEqual(public[field], value)
        self.assertFalse(public['raw_redacted'])
        self.assertEqual(load_reference('analyze').correct(public), 1)

    def test_extra_and_duplicate_keys_not_publishable_raw(self):
        for raw in ('{"verdict":"yes","quote":"Text"}', '{"verdict":"yes","verdict":"no"}', '```json\n{"verdict":"yes"}\n```'):
            self.assertIsNone(parse_verdict(raw))
            self.assertEqual(project_response({'raw': raw})['raw'], '')

    def test_fresh_inference_requires_explicit_flag_before_creating_files(self):
        import replicate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'new-run'
            with patch.object(sys, 'argv', ['replicate.py', '--output', str(output)]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    replicate.main()
            self.assertFalse(output.exists())

    def test_fresh_inference_cannot_overwrite_existing_run(self):
        import replicate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / 'responses.jsonl'
            marker.write_text('existing scientific records\n')
            with patch.object(sys, 'argv', ['replicate.py', '--run-inference', '--output', str(output)]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    replicate.main()
            self.assertEqual(marker.read_text(), 'existing scientific records\n')

if __name__ == '__main__':
    unittest.main()
