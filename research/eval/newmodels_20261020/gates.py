import json
C = json.load(open('analysis_cells.json'))
prim = {'MC522': 'human_PQ', 'PAM400': 'pam_OVL', 'ME': 'human_OVL', 'ME-test': 'human_OVL'}
METR = ['aes_PQ','aes_CE','aes_CU','aes_PC','meva','muqeval_A1_MI','muqeval_A1_TA','muqeval_A3a_MI','muqeval_A3a_TA','cmi_empty_quality','cmi_empty_alignment','cmi_caption_quality','cmi_caption_alignment','pam']
fam = lambda m: 'aes' if m.startswith('aes') else 'meva' if m=='meva' else 'muq' if m.startswith('muq') else 'cmi' if m.startswith('cmi') else 'pam'
IND = {'aes': {'MC522': 0, 'PAM400': 1, 'ME': 1, 'ME-test': 0}, 'meva': {'MC522': 1, 'PAM400': 1, 'ME': 0, 'ME-test': 0},
       'muq': {'MC522': 1, 'PAM400': 1, 'ME': 0, 'ME-test': 0}, 'cmi': {'MC522': 1, 'PAM400': 1, 'ME': 0, 'ME-test': 1},
       'pam': {'MC522': 1, 'PAM400': 0, 'ME': 1, 'ME-test': 0}}
get = lambda b, m, t=None: next((c for c in C if c['bench'] == b and c['metric'] == m and c['target'] == (t or prim[b])), None)
f = lambda v: f'{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]'
out = ['| 指標 | 基準 | 獨立? | n | Spearman | Pearson | Kendall | 系統內 ρ | 成對一致率(整體) | 對AES-PQ ΔSpearman | 閘門 |', '|---|---|---|---|---|---|---|---|---|---|---|']
verdict = {}
for m in METR:
    gens_up = []; ind_n = 0; gen_ind = 0; allpass = True; detail = []
    for b in ['MC522', 'PAM400', 'ME', 'ME-test']:
        c = get(b, m)
        if not c: continue
        ind = IND[fam(m)][b]
        g = []
        g1 = c['spearman'][0] >= .5 and c['spearman'][1] > .3; g.append(('G1', g1))
        if 'within' in c: g.append(('G2', c['within'][0] >= .5 and c['within'][1] > .3))
        if 'pair_all' in c and c['pair_all'][3] >= 500: g.append(('G3', c['pair_all'][1] > .55))
        if ind:
            ind_n += 1; gen_ind += b in ('PAM400', 'ME-test') or (b == 'ME' and fam(m) in ('aes','pam'))
            allpass &= all(x for _, x in g)
            if b in ('PAM400', 'ME', 'ME-test'):
                if c['spearman'][2] < .5 or ('pair_all' in c and c['pair_all'][3] >= 500 and c['pair_all'][2] < .55): gens_up.append(b)
        gs = ' '.join(f'{k}{"✓" if v else "✗"}' for k, v in g)
        d = c.get('d_aes_PQ_spearman'); 
        out.append(f"| {m} | {b} | {'獨立' if ind else '描述'} | {c['n']} | {f(c['spearman'])} | {f(c['pearson'])} | {f(c['kendall'])} | {f(c['within']) if 'within' in c else '—'} | {f(c['pair_all'][:3])+' (n='+str(c['pair_all'][3])+')' if 'pair_all' in c else '—'} | {f(d) if d else '—'} | {gs}{'' if ind else ' (不計)'} |")
    v = '排除' if gens_up else ('通過' if allpass and ind_n >= 2 and gen_ind >= 1 else '未定')
    verdict[m] = (v, gens_up, ind_n, gen_ind, allpass)
out.append('\n| 指標 | 判定 | 獨立基準數 | 獨立生成基準數 | 全閘門過 | 排除依據 |\n|---|---|---|---|---|---|')
for m, (v, gu, i, gi, ap) in verdict.items(): out.append(f'| {m} | **{v}** | {i} | {gi} | {ap} | {",".join(gu)} |')
open('gate_tables.md', 'w').write('\n'.join(out)); print('\n'.join(out[-17:]))
