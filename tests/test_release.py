"""CPU-only regression checks for preserved fixtures, scoring and prompt semantics."""
from pathlib import Path
import json, sys, unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from external_protocol import canonical,digest,make_request,parse_verdict,render_without_tokenizer,token_differences
from reproduce_tables import natural_summary,read,reproduce,rows
from validate_release import validate
import controlled_editing_data as c

class PreservationTests(unittest.TestCase):
    def test_exact_deterministic_construction(self):
        built=c.build_records();c.validate_records(*built)
        for name,records in zip(('worlds','tasks','gold','stress_spans'),built):self.assertEqual(records,rows(f'data/controlled_editing/{name}.jsonl'))
    def test_source_and_span_integrity_rejects_mutation(self):
        built=c.build_records();built[1][0]['candidate_text']+=' Corruption.'
        with self.assertRaises(ValueError):c.validate_records(*built)
    def test_exact_prompt_and_render_fixtures(self):
        for fixture in read('tests/prompt_fixtures.json'):
            messages=make_request(fixture['row'],fixture['condition'])
            self.assertEqual(digest(canonical(messages)),fixture['expected_messages_sha256'])
            self.assertEqual(digest(render_without_tokenizer(messages)),fixture['expected_rendered_sha256'])
    def test_parser_does_not_repair_malformed_responses(self):
        for raw in ['```json\n{"verdict":"yes"}\n```','{"verdict":"yes","verdict":"no"}','{"verdict":true}','{"verdict":"yes","reason":"x"}','yes','[]','{"verdict":null}']:
            self.assertIsNone(parse_verdict(raw))
        self.assertEqual(parse_verdict(' \n{"verdict":"no"} '),'no')
    def test_token_difference_semantics_and_limit(self):
        self.assertEqual(token_differences('A B C','A D C'),["replace at original words [1:2]: ['B'] -> ['D']"])
        old=' '.join(f'anchor{i} old{i}' for i in range(20));new=' '.join(f'anchor{i} new{i}' for i in range(20))
        diff=token_differences(old,new);self.assertEqual(len(diff),13);self.assertEqual(diff[-1],'... 8 further changes')
    def test_original_test_outcomes_and_development_scope(self):
        report=reproduce();self.assertEqual(len(report['tables']),10)
        test=report['natural_metrics']['test'];self.assertEqual([test[c]['correct'] for c in ['whole','matched','delta_prompt']],[74,75,82])
        self.assertEqual([test[c]['invalid_accept_count'] for c in ['whole','matched','delta_prompt']],[23,22,14])
        self.assertEqual([test[c]['valid_reject_count'] for c in ['whole','matched','delta_prompt']],[3,3,4])
        dev=report['natural_metrics']['dev'];self.assertEqual(dev['delta_claims_v1']['correct'],31);self.assertEqual(dev['delta_prompt']['correct'],31)
        self.assertEqual(dev['delta_claims_v1']['token_total'],74995);self.assertEqual(dev['delta_prompt']['token_total'],46494)
        self.assertNotIn('delta_claims_v1',test)
    def test_malformed_is_never_accuracy_credit(self):
        example=rows('results/external/test_predictions.metadata.jsonl')[:2]
        example[0]['label']=0;example[0]['verdict']='no';example[0]['parse_error']='malformed_pipeline'
        example[1]['label']=1;example[1]['verdict']='yes';example[1]['parse_error']=None
        self.assertEqual(natural_summary(example)['correct'],1)
    def test_complete_release_hashes_and_lattices(self):self.assertEqual(validate()['status'],'PASS')
if __name__=='__main__':unittest.main()
