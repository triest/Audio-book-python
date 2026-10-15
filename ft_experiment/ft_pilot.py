import json,re,gzip,collections,random,copy,sys
import numpy as np, torch
sys.path.insert(0,'J:/audiobook')
import fb2_reader as fr
acc=fr._load_silero_stress(); A=acc.accentor
V='аеёиоуыэюя'
n=lambda s:s.replace('ё','е')
tok=lambda w:A._tokenize(w)[1][0]
d=json.load(open('J:/audiobook/forced_stress_words.json',encoding='utf-8'))
items=[(k,v.lower()) for k,v in d.items() if not k.startswith('_') and ' ' not in k and sum(c in V for c in k)>=2 and '+' in v]
def label(form):
    plain=form.replace('+',''); i=form.index('+'); return sum(c in V for c in plain[:i])+1   # 1-based vowel ordinal
def logits(model,words): return model([tok(w) for w in words])[0]
model=A.model
# sanity: does argmax == ordinal of the model's own stressed output?
S=[w for w in ['капуста','молоко','трикупис','знаменитый','собака','корабль'] ]
with torch.no_grad(): am=logits(model,S).argmax(1).tolist()
own=[label(A(w).lower()) if '+' in A(w) else None for w in S]
print('argmax',am,'ordinal from output',own)

lab0=lambda form: label(form)-1
# data
raw_items=[(k,lab0(v)) for k,v in items if 'ё' not in v]
# split by 5-letter stem so case forms of one lemma never straddle train/test
stems=sorted({k[:5] for k,_ in raw_items}); random.Random(1).shuffle(stems)
test_stems=set(stems[:len(stems)//4])
train=[(k,l) for k,l in raw_items if k[:5] not in test_stems]; test=[(k,l) for k,l in raw_items if k[:5] in test_stems]
corp=collections.defaultdict(set)
for l in gzip.open('J:/audiobook/stress_corpus_ru.tsv.gz','rt',encoding='utf-8'):
    q=l.rstrip('\n').split('\t')
    if len(q)==2: corp[n(q[0].lower())].add(n(q[1].lower().replace('^','+')))
t=''.join(open(f'J:/audiobook/audiobook_output2/_test_xenia_ch1_2/text/ch{i}_plain.txt',encoding='utf-8').read() for i in (5,9,20,31,44))
vocab=sorted({w for w in re.findall('[а-я]+',t.lower()) if sum(c in V for c in w)>=2})
random.Random(2).shuffle(vocab)
with torch.no_grad(): old=model([tok(w) for w in vocab])[0].argmax(1)
replay=list(zip(vocab[:4000],old[:4000].tolist())); drift_eval=list(zip(vocab[4000:],old[4000:].tolist()))
def acc_on(m,pairs):
    with torch.no_grad(): p=m([tok(w) for w,_ in pairs])[0].argmax(1).tolist()
    return float(np.mean([a==b for a,(_,b) in zip(p,pairs)]))
def corp_agree(m):
    ws=[w for w in vocab if len(corp.get(n(w),()))==1]
    with torch.no_grad(): p=m([tok(w) for w in ws])[0].argmax(1).tolist()
    ok=0
    for w,i in zip(ws,p):
        vs=[j for j,c in enumerate(w) if c in V]
        if i<len(vs): ok+= n(w[:vs[i]]+'+'+w[vs[i]:]) in corp[n(w)]
    return ok/len(ws),len(ws)
print('train',len(train),'test',len(test),'replay',len(replay),'drift-eval',len(drift_eval))
print('BEFORE test acc',acc_on(model,test),'train acc',acc_on(model,train),'drift-eval same-as-old',acc_on(model,drift_eval),'corpus agree',corp_agree(model))
for lr,epochs,rw in ((3e-4,30,1.0),(1e-3,60,1.0)):
    m=copy.deepcopy(model)
    for p in m.parameters(): p.requires_grad_(True)
    torch.set_grad_enabled(True); m.train()
    opt=torch.optim.Adam(m.parameters(),lr=lr)
    for ep in range(epochs):
        random.shuffle(replay); batch=train+replay[:len(train)*8]
        random.shuffle(batch)
        for i in range(0,len(batch),256):
            b=batch[i:i+256]
            out=m([tok(w) for w,_ in b])[0]
            loss=torch.nn.functional.cross_entropy(out,torch.tensor([l for _,l in b]))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); torch.set_grad_enabled(False)
    print(f'lr={lr} ep={epochs}: test acc {acc_on(m,test):.3f} train acc {acc_on(m,train):.3f} drift-eval {acc_on(m,drift_eval):.4f} corpus agree {corp_agree(m)}')
