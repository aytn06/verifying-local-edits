#!/usr/bin/env python3
"""Recompute every numerical manuscript table from frozen public records on CPU.

Python standard library only. Never loads weights, contacts a network, generates
predictions, changes labels, or uses label-review flags to rescore.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, random, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference_code'))
import analyze_controlled_verification as controlled

def read(path):return json.loads((ROOT/path).read_text())
def rows(path):return [json.loads(s) for s in (ROOT/path).read_text().splitlines() if s.strip()]
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,c-h),min(1.,c+h)]
def natural_summary(records):
    correct=sum(r['verdict']==('yes' if r['label'] else 'no') and not r['parse_error'] for r in records)
    invalid=[r for r in records if r['label']==0];valid=[r for r in records if r['label']==1]
    ia=sum(r['verdict']=='yes' for r in invalid);vr=sum(r['verdict']!='yes' for r in valid)
    calls=[c for r in records for c in r['intermediate']]
    inp=sum(c['input_tokens'] for c in calls);out=sum(c['output_tokens'] for c in calls)
    return dict(n=len(records),correct=correct,n_invalid=len(invalid),n_valid=len(valid),
                strict_accuracy=correct/len(records),invalid_accept_count=ia,valid_reject_count=vr,
                accuracy_wilson_95=wilson(correct,len(records)),invalid_accept_wilson_95=wilson(ia,len(invalid)),
                valid_reject_wilson_95=wilson(vr,len(valid)),token_total_input=inp,token_total_output=out,
                token_total=inp+out,seconds_total=sum(c['seconds'] for c in calls),
                model_calls=len(calls),malformed_count=sum(r['parse_error'] is not None for r in records))
def natural_pairs(records):
    ids=list(dict.fromkeys(r['case_id'] for r in records)); by={(r['case_id'],r['condition']):r for r in records};out=[]
    for first,second in [('matched','whole'),('delta_prompt','whole'),('delta_prompt','matched')]:
        for metric in ['accuracy','invalid_accept_rate','valid_reject_rate']:
            points=[];case_ids=[]
            for cid in ids:
                a,b=by[cid,first],by[cid,second];label=a['label'];assert label==b['label']
                if metric=='accuracy':value=lambda r:int(r['verdict']==('yes' if label else 'no') and not r['parse_error'])
                elif metric=='invalid_accept_rate':
                    if label!=0:continue
                    value=lambda r:int(r['verdict']=='yes')
                else:
                    if label!=1:continue
                    value=lambda r:int(r['verdict']!='yes')
                points.append((value(a),value(b)));case_ids.append(cid)
            rng=random.Random(20261008);draws=[];n=len(points)
            for _ in range(10000):
                picks=[rng.randrange(n) for _ in points]
                draws.append(sum(points[i][0]-points[i][1] for i in picks)/n)
            draws.sort()
            out.append(dict(first=first,second=second,metric=metric,n_pairs=n,
                first_count=sum(x for x,y in points),second_count=sum(y for x,y in points),
                first_only_count=sum(x==1 and y==0 for x,y in points),second_only_count=sum(x==0 and y==1 for x,y in points),
                both_count=sum(x==y==1 for x,y in points),neither_count=sum(x==y==0 for x,y in points),
                first_minus_second_rate=sum(x-y for x,y in points)/n,
                paired_source_bootstrap_95_interval=[draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]],
                discordant_case_ids=[cid for cid,(x,y) in zip(case_ids,points) if x!=y]))
    return out

def reproduce():
    cases=rows('data/controlled_editing/tasks.jsonl');gold=rows('data/controlled_editing/gold.jsonl')
    raw=[]
    for model in ('qwen3_8b','qwen3_17b'):
        for stage in ('smoke','development','evaluation'):
            raw+=rows(f'results/controlled_verification/{model}/{stage}_judgments.jsonl')
    assert len(raw)==864
    joined=controlled.join_predictions(raw,cases,gold)
    audit=read('results/controlled_verification/qwen3_8b_qwen3_17b_audit.json')
    summary=[]
    for expected in audit['summaries_primary']:
        got=controlled.summarize(joined,model=expected['model'],split=expected['split'],condition=expected['condition'],output_class=expected['construction_class'])
        assert got==expected,('controlled summary mismatch',got,expected)
        summary.append(got)
    contrasts=[]
    for expected in audit['paired_contrasts']:
        select=[r for r in joined if r['model_key']==expected['model'] and r['split']==expected['split']]
        got=controlled.paired_cluster_contrast(select,expected['before'],expected['after'],expected['metric'],cluster=expected['cluster'])
        got.update(model=expected['model'],split=expected['split']);assert got==expected
        contrasts.append(got)
    dev=rows('results/external/dev_predictions.metadata.jsonl')+rows('results/external/delta_claims_v1.metadata.jsonl')
    test=rows('results/external/test_predictions.metadata.jsonl')
    natural={}
    for split,records,expected_list in [('dev',dev,read('results/external/development_summary.json')['conditions']),('test',test,read('results/external/test_summary.json'))]:
        natural[split]={}
        for expected in expected_list:
            key=expected['condition'];got=natural_summary([r for r in records if r['condition']==key])
            for field,value in got.items():
                if field in expected:
                    if isinstance(value,float):assert abs(value-expected[field])<1e-8,(split,key,field)
                    else:assert value==expected[field],(split,key,field)
            natural[split][key]=got
    paired=natural_pairs(test)
    for got,expected in zip(paired,read('results/external/test_paired_comparisons.json')):
        for key,value in got.items():assert value==expected[key],(key,got,expected)
    tables=[]
    def add(name,headers,body):tables.append({'name':name,'headers':headers,'rows':body})
    kinds=['acceptable_role','acceptable_minor','wrong_actor','reversed_polarity','foregrounded_minor_1','foregrounded_minor_2']
    names=['Supported helper','Minor event later','Wrong actor','Reversed polarity','Minor A first','Minor B first']
    model_names={'qwen3_8b':'8B','qwen3_17b':'1.7B'}
    conds=controlled.CONDITIONS;cond_names=dict(zip(conds,['F: fragment fact','W: whole fact','J: whole joint']))
    nm={'whole':'Whole','matched':'Matched','delta_prompt':'Delta-prompt','delta_claims_v1':'Delta-claims'}
    gold_map={g['case_id']:g for g in gold}
    add('construction',['Construction','Factual label','Prominence label','Dev count','Evaluation count'],[[name,int(next(g for g in gold if g['construction_type']==kind)['factual_valid']),int(next(g for g in gold if g['construction_type']==kind)['prominence_valid']),8,16] for kind,name in zip(kinds,names)])
    primary=[]
    for s in summary:
        if s['split']!='evaluation' or s['construction_class']!='all':continue
        metrics=['assembled_factual_error_admission','pure_prominence_error_admission','acceptable_false_reject_operational']
        ratios=[f"{s[m+'_numerator']}/{s[m+'_denominator']}" for m in metrics]
        accuracy='--' if s['declared_target_error_denominator']==0 else f"{100*(1-s['declared_target_error_rate']):.1f}%"
        primary.append([model_names[s['model']],cond_names[s['condition']],*ratios,s['malformed_count'],accuracy])
    primary.extend([['Analytic','Always accept','32/32','32/32','0/32',0,'--'],['Analytic','Always reject','0/32','0/32','32/32',0,'--']])
    add('primary',['Model','Condition','Factual accepted','Order accepted','Acceptable rejected','Malformed','Target accuracy'],primary)
    add('contrasts',['Model','Before','After','Metric','Difference pp','95% interval pp'],[[model_names[r['model']],r['before'],r['after'],r['metric'],f"{100*r['estimate']:.1f}",'['+','.join(f'{100*x:.1f}' for x in r['ci95_descriptive'])+']'] for r in contrasts if r['cluster']=='family_id'])
    tokens=[]
    for c in conds:
        ss={s['model']:s for s in summary if s['split']=='evaluation' and s['construction_class']=='all' and s['condition']==c}
        assert ss['qwen3_8b']['input_tokens']==ss['qwen3_17b']['input_tokens']
        n=ss['qwen3_8b']['input_tokens'];tokens.append([cond_names[c][0],n,f'{n/96:.1f}',ss['qwen3_8b']['output_tokens'],ss['qwen3_17b']['output_tokens']])
    add('tokens',['Check','Input total','Input mean','8B output','1.7B output'],tokens)
    body=[]
    for c,s in natural['test'].items():
        ci=','.join(f'{100*x:.1f}' for x in s['accuracy_wilson_95'])
        body.append([nm[c],f"{100*s['strict_accuracy']:.1f} [{ci}]",f"{s['invalid_accept_count']}/50",f"{s['valid_reject_count']}/50",s['token_total_input'],s['token_total_output'],f"{s['seconds_total']:,.2f}"])
    add('naturaltest',['Method','Accuracy [95% CI]','IA','VR','Input tokens','Output tokens','Time (s)'],body)
    body=[]
    for first,second in [('matched','whole'),('delta_prompt','whole'),('delta_prompt','matched')]:
        pp=[r for r in paired if r['first']==first and r['second']==second];vals=[]
        for r in pp:
            n=100*r['first_minus_second_rate'];point=f'{n:+.0f}' if n else '0';ci=','.join(f'{100*x:.0f}' for x in r['paired_source_bootstrap_95_interval']);vals.append(f'{point} [{ci}]')
        body.append([f'{nm[first]} - {nm[second]}',*vals,f"{pp[0]['first_only_count']}:{pp[0]['second_only_count']}"])
    add('naturalpairs',['Contrast','Accuracy pp','Invalid acceptance pp','Valid rejection pp','Only correct A:B'],body)
    add('naturaldev',['Development','Correct','IA','VR','Tokens'],[[nm[c],f"{s['correct']}/40",f"{s['invalid_accept_count']}/20",f"{s['valid_reject_count']}/20",s['token_total']] for c,s in natural['dev'].items()])
    for split in ['development','evaluation']:
        body=[]
        for kind,name in zip(kinds,names):
            body.append([name,*[sum(r['verdict']=='yes' and r['format_valid'] for r in joined if r['split']==split and r['construction_type']==kind and r['model_key']==m and r['condition_id']==c) for m in model_names for c in conds]])
        add('construction_yes_'+split,['Construction','8B F','8B W','8B J','1.7B F','1.7B W','1.7B J'],body)
    add('family_errors',['Family','8B F','8B W','8B J','1.7B F','1.7B W','1.7B J'],[[fam,*[sum(r['joint_classification_error'] for r in joined if r['template_family']==fam and r['model_key']==m and r['condition_id']==c) for m in model_names for c in conds]] for fam in sorted({r['template_family'] for r in joined if r['split']=='evaluation'})])
    diag=rows('results/controlled_rule_ablation/qwen3_8b/judgments.jsonl')
    baseline={r['case_id']:r for r in joined if r['model_key']=='qwen3_8b' and r['split']=='development' and r['condition_id']=='whole_factual'}
    transitions=Counter(f"{baseline[r['case_id']]['verdict']}->{r['verdict']}" for r in diag)
    assert transitions=={'yes->yes':20,'no->no':16,'no->yes':12}
    assert len(diag)==48 and sum(r['input_tokens'] for r in diag)==19593 and sum(r['output_tokens'] for r in diag)==288
    return {'tables':tables,'controlled_primary_calls':len(raw),'controlled_diagnostic_calls':len(diag),'external_baseline_dev_calls':120,'external_two_stage_dev_calls':80,'external_test_calls':300,'total_model_calls':1412,'natural_metrics':natural,'test_paired':paired,'controlled_diagnostic':dict(transitions)}

def write_report(report,out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'tables.json').write_text(json.dumps(report,indent=2)+'\n')
    md=['# Reproduced numerical tables','', 'Computed from cached observations; no inference or network. F/W/J have different targets.','']
    tex=['% Generated table fragments; include in a document with booktabs.']
    def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    for table in report['tables']:
        md+=['## '+table['name'],'','| '+' | '.join(table['headers'])+' |','| '+' | '.join('---' for _ in table['headers'])+' |']
        md+=['| '+' | '.join(str(x) for x in row)+' |' for row in table['rows']];md+=['']
        tex += ['% '+table['name'],r'\begin{tabular}{'+'l'*len(table['headers'])+'}',r'\toprule',' & '.join(map(esc,table['headers']))+r'\\',r'\midrule']
        tex += [' & '.join(map(esc,row))+r'\\' for row in table['rows']];tex += [r'\bottomrule',r'\end{tabular}','']
    (out/'tables.md').write_text('\n'.join(md));(out/'tables.tex').write_text('\n'.join(tex))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'generated');args=p.parse_args()
    report=reproduce();write_report(report,args.output)
    print(f"PASS: {len(report['tables'])} numerical tables; 864 primary + 48 diagnostic + 120 development + 80 two-stage + 300 test calls. Output: {args.output}")
if __name__=='__main__':main()
