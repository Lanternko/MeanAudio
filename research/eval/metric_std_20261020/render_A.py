import json
d=json.load(open('sensitivity.json')); ft=d['flip_table']; lad=d['ladders']
m=json.load(open('m_defs.json'))
S={'PQ': 0.9189201662443987, 'CE': 1.2004732543363863, 'CU': 0.8452304576106027, 'PC': 0.34474729506918766, 'meva': 0.17236630871175196, 'clap': 0.10382564958097837}
o=[]
o.append('| 指標 | S_現行 | m₀（D0） | m₁（D1 seed 80 分位） | m₂（D2 人類 JND 0.5） | m₃（D3 0.1·IQR 31 系統） |\n|---|---|---|---|---|---|')
for k,s in S.items():
    row=[f'{s:.4f}',f'{0.1*s:.4f}']
    for D in ('D1','D2','D3'):
        v=m[D].get(k); row.append('—' if v is None else f'{v:.4f}（{v/s:.2f} S）')
    o.append(f'| {k} | '+' | '.join(row)+' |')
print('\n'.join(o)); print()
# pass counts
for cls,lab in (('tie','該平手格：通過數／格數'),('lose','該判差格：通過數／格數')):
    print(f'**{lab}**（D0＝現行；格＝資料集內「介入×系統（或曲源）」；翻轉＝與 D0 通過狀態不同的格數）\n')
    print('| 資料集 | 指標 | n 格 | D0 | D1 | D2 | D3 |\n|---|---|---|---|---|---|---|')
    for ds in ('126a','126b','126c'):
        for mt in ('PQ','CE','CU','PC','meva','clap'):
            if cls=='lose' and mt=='PC': continue
            row=[]; n=None; p0=None
            for D in ('D1','D2','D3'):
                v=ft.get(f'{ds}|{mt}|{cls}|{D}')
                if v is None: row.append('—'); continue
                n=v['n']; p0=v['p0']
                row.append(f"{v['pD']}（+{v['to_pass']}／−{v['to_fail']}）")
            if n is None: continue
            print(f'| {ds} | {mt} | {n} | {p0} | '+' | '.join(row)+' |')
    print()
