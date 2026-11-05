"""Checkpoint publication/recovery tests; no public model or dataset is loaded.

Synthetic cases extract only I/O functions, avoiding the trainer's Torch imports.
Two optional, isolated CPU cases exercise real tiny-model continuation and both
supported import contexts. Run with unittest discovery; Torch absence skips them.
"""
import ast,copy,hashlib,importlib.util,json,math,os,pickle,random,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
REPO=HERE.parent
SOURCE=REPO/'training/fastpitch_experiment.py'
sys.path.insert(0,str(REPO))


def ast_fingerprint(node):
    # Python 3.12 adds an empty type_params field; it has no recipe meaning here.
    def canonical(value):
        if isinstance(value,ast.AST):
            return [type(value).__name__,[(key,canonical(item)) for key,item in ast.iter_fields(value)
                    if not (key=='type_params' and not item)]]
        if isinstance(value,list):return [canonical(item) for item in value]
        if isinstance(value,bytes):return ['bytes',value.hex()]
        if value is Ellipsis:return ['Ellipsis']
        return value
    return hashlib.sha256(json.dumps(canonical(node),separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

ORIGINAL_FUNCTION_AST = {'digest': '107f09da87198aaa592bbcee39dac1c30c9bbf3ed771bc0574c3a77b5b1f8eab', 'atomic_json': '12bd024c17c8b08516ca30448e0546fd9309f788cc75d8e3909b31459c55392f', 'atomic_torch': 'f71be612d9972f53190913581e041b5c060b00ecde4be6ce15c9ef23258147a1', 'append_history': 'a1b731f909eb683503177984360d17377c4132d19507fc1b2cbc25ed9703dcb4', 'rows': 'c15d278f8b7ee53f677d7c5dacb1898720040a642df54dddb2ebdae3db75eb1e', 'normalize': 'd584f0da2f7e8f3427725bc3b2d982fec3c19137b2d4ce1bea8d366ff7cc25f0', 'tokens': 'f63c96f866073ccda4702ef48fde0c7163b0277dcea54831d51ec07fff860931', 'mel_spectrogram': 'b3a26e5575b9c415f78b97774703ffe405d5ae1b65685dae9f3cb5af33293b31', 'prepare_one': '0c4884afd621d5d16dbe8c90ea1ff922c203ca77b277c63fa978054796f53e7f', 'prepare': 'f43429c524bfa3c065f1eb539912a1c422720a6473f500a4d2beda3189ac9d77', 'imports': '88ef4ef06bf6da93d7572819d0549b56410d1a8e335398ef592a192e7a309f09', 'sample_batch': '3b766361d5af5f112b837e576749aaca9aec8a8fdd9e8e8f988cb3d6b85d01bc', 'load_feature': 'fa6db7c347d9109adcdd3e5657ea74e26e81a0ced88552d314c5f7e0629ce4ce', 'audit_features': 'e77f68fd313e17762bad875428418f6d8b86878d43a8470d0d877375ce325a94', 'rng_state': '78eb7e382c82b460c04d8a87e3384c2874d7f8d546d052e89b7cffdec44f2e14', 'restore_rng': '2c0f74f8b1634132b24ca309f784aedfb6ce57f6884eb9ad8d247c0611b052fd', 'train': '8dd399c50de04286a52897f9dbd0fbb05c2a6195c7701fefe4a8e4e4612bb21d', 'synthesize': 'cf84d74b54ff959fd72c7e28d45daffe7bd076879dcf29aa0ae181b8b0b18cf2', 'main': '6a87f154ee592a6cbeafc5a20b35087cc1e46152d977111fb7d48a17f8a35cb4'}
ORIGINAL_SAVE_PUBLICATION = "atomic_torch(checkpoint_path, state)\nif is_best:\n    tmp = root / 'best.pt.tmp'\n    if tmp.exists():\n        tmp.unlink()\n    os.link(checkpoint_path, tmp)\n    os.replace(tmp, root / 'best.pt')\natomic_json(root / 'checkpoint.json', dict(step=step, epoch=epoch, cursor=cursor, checkpoint=str(checkpoint_path), sha256=digest(checkpoint_path), best=best))"

class Tensor:
    def __init__(self,value,shape=(1,),dtype='float32',device='cpu',requires_grad=True):
        self.values=list(value) if isinstance(value,(list,tuple)) else [value]
        self.shape=shape;self.dtype=dtype;self.device=types.SimpleNamespace(type=device);self.requires_grad=requires_grad
    def numel(self):return len(self.values)
    def is_floating_point(self):return self.dtype=='float32'
    def is_complex(self):return False
    def detach(self):return self
    def cpu(self):v=copy.deepcopy(self);v.device.type='cpu';return v
    def __float__(self):return float(self.values[0])
class Array:
    def __init__(self,values):self.values=list(values);self.dtype='uint32'
class Bool:
    def __init__(self,value):self.value=value
    def all(self):return self.value
class FakeTorch:
    uint8='uint8'
    def __init__(self):self.loads=0;self.hook=None;self.corrupt=None
    def is_tensor(self,x):return isinstance(x,Tensor)
    def isfinite(self,x):return Bool(all(math.isfinite(v) for v in x.values))
    def equal(self,a,b):return a.values==b.values and a.shape==b.shape and a.dtype==b.dtype
    def save(self,state,stream):pickle.dump(self.corrupt(copy.deepcopy(state)) if self.corrupt else state,stream)
    def load(self,stream,**kwargs):
        self.loads+=1
        if self.hook:self.hook(stream)
        def cpu(x):
            if isinstance(x,Tensor):return x.cpu()
            if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
            if isinstance(x,list):return [cpu(v) for v in x]
            if isinstance(x,tuple):return tuple(cpu(v) for v in x)
            return x
        return cpu(pickle.load(stream))
    class Generator:
        def __init__(self,device):assert device=='cpu'
        def set_state(self,state):
            if not isinstance(state,Tensor) or state.dtype!='uint8' or not state.numel():raise ValueError('bad Torch RNG')
class FakeNP:
    ndarray=Array
    def isfinite(self,x):return Bool(all(math.isfinite(v) for v in x.values))
    def array_equal(self,a,b):return a.values==b.values
    class random:
        class RandomState:
            def set_state(self,state):
                if state[0]!='numpy' or not isinstance(state[1],Array):raise ValueError('bad NumPy RNG')
class Model:
    def __init__(self):
        self.named=[('weight',Tensor([1.,2.],(2,),device='cuda')),('bias',Tensor(0.,device='cuda')),
            ('attention.attn_proj.weight',Tensor(0.,device='cuda')),('attention.attn_proj.bias',Tensor(0.,device='cuda'))]
    def named_parameters(self):return iter(self.named)
    def parameters(self):return iter([p for _,p in self.named])
    def state_dict(self):return {k:p for k,p in self.named}

def namespace():
    tree=ast.parse(SOURCE.read_text())
    names={'digest','atomic_json'}|{n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_checkpoint_')}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns=dict(torch=FakeTorch(),np=FakeNP(),os=os,Path=Path,hashlib=hashlib,json=json,math=math,random=random)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'private_ast_only','exec'),ns)
    return ns
def state(model,step=20003):
    optimizer=dict(state={i:dict(step=Tensor(float(step)),exp_avg=Tensor([0.]*p.numel(),p.shape),exp_avg_sq=Tensor([1.]*p.numel(),p.shape)) for i,(_,p) in enumerate(model.named[:2])} if step else {},
        param_groups=[dict(params=list(range(4)),betas=(.9,.98),eps=1e-8,weight_decay=1e-6,amsgrad=False,lr=.0002*min(1.,(step+1)/1000))])
    lr=optimizer['param_groups'][0]['lr']
    return dict(model=model.state_dict(),optimizer=optimizer,scheduler=dict(last_epoch=step,base_lrs=[.0002],_last_lr=[lr],lr_lambdas=[None]),scaler={},
        rng=dict(python=random.getstate(),numpy=('numpy',Array([1,2,3])),torch=Tensor([1,2,3],(3,),'uint8'),cuda=[Tensor([4,5],(2,),'uint8')]),
        step=step,epoch=105,cursor=1163,best=3.5238519241643504,bad=6,
        config=dict(precision='bf16',lr=.0002,warmup=1000,selection=dict(max_lr_reductions=1,plateau_lr_factor=.3)),
        provenance=dict(code_sha256='source',source_manifest_sha256='manifest',upstream='upstream',torch='fake',cuda='11.8',argv=['test'],utc='now'),lr_state=dict(multiplier=1.,reductions=0))
class IOTests(unittest.TestCase):
    def setUp(self):self.n=namespace();self.model=Model();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def publish(self,s=None,best=False):self.n['_checkpoint_publish'](self.root,s or state(self.model),self.model,best)
    def resume(self):return self.n['_checkpoint_resume'](self.root,self.model)
    def test_good_publish_full_state_and_hardlink_best(self):
        before=random.getstate();self.publish(best=True);loaded,digest=self.resume()
        self.assertEqual(loaded['step'],20003);self.assertEqual((self.root/'last.pt').stat().st_ino,(self.root/'best.pt').stat().st_ino)
        self.assertEqual(digest,json.loads((self.root/'checkpoint.json').read_text())['sha256']);self.assertFalse((self.root/'checkpoint.pending.json').exists());self.assertEqual(before,random.getstate())
    def test_empty_optimizer_is_valid_only_at_step0(self):
        self.publish(state(self.model,0));self.assertEqual(self.resume()[0]['step'],0)
        s=state(self.model);s['optimizer']['state']={}
        with self.assertRaises(ValueError):self.n['_checkpoint_validate'](self.n['torch'].load(__import__('io').BytesIO(pickle.dumps(s))),self.model)
    def test_wrong_SHA_refuses_before_load(self):
        self.publish();t=self.n['torch'];count=t.loads
        with self.assertRaisesRegex(ValueError,'BEFORE'):self.n['_checkpoint_load'](self.root/'last.pt',self.model,'0'*64)
        self.assertEqual(t.loads,count)
    def test_path_swap_during_load_is_rejected(self):
        self.publish();path=self.root/'last.pt';replacement=self.root/'replacement';replacement.write_bytes(path.read_bytes())
        self.n['torch'].hook=lambda stream:os.replace(replacement,path)
        with self.assertRaisesRegex(ValueError,'DURING'):self.n['_checkpoint_load'](path,self.model)
    def test_inplace_change_during_load_is_rejected(self):
        self.publish();path=self.root/'last.pt'
        def mutate(stream):
            with path.open('ab') as f:f.write(b'changed')
        self.n['torch'].hook=mutate
        with self.assertRaisesRegex(ValueError,'DURING'):self.n['_checkpoint_load'](path,self.model)
    def test_corrupt_or_missing_state_never_replaces_previous(self):
        self.publish();old=(self.root/'last.pt').read_bytes();receipt=(self.root/'checkpoint.json').read_bytes()
        self.n['torch'].corrupt=lambda s:dict(s,optimizer={})
        with self.assertRaises(Exception):self.publish(state(self.model,20004))
        self.assertEqual(old,(self.root/'last.pt').read_bytes());self.assertEqual(receipt,(self.root/'checkpoint.json').read_bytes());self.assertFalse((self.root/'checkpoint.pending.json').exists())
        with self.assertRaisesRegex(ValueError,'Unpublished checkpoint temp'):self.resume()
    def test_nonfinite_model_or_optimizer_is_rejected(self):
        for field in ('model','optimizer'):
            s=state(self.model)
            (s['model']['weight'] if field=='model' else s['optimizer']['state'][0]['exp_avg']).values[0]=float('nan')
            with self.assertRaises(ValueError):self.n['_checkpoint_validate'](s,self.model)
    def test_finite_but_wrong_readback_is_rejected(self):
        self.publish();old=(self.root/'last.pt').read_bytes()
        def corrupt(s):s['model']['weight'].values[0]=9.;return s
        self.n['torch'].corrupt=corrupt
        with self.assertRaisesRegex(ValueError,'snapshot'):self.publish(state(self.model,20004))
        self.assertEqual(old,(self.root/'last.pt').read_bytes())
    def test_bad_rng_and_missing_active_moments_are_rejected(self):
        for kind in ('rng','moments','scheduler'):
            s=state(self.model)
            if kind=='rng':s['rng']['torch']=Tensor([], (0,),'uint8')
            elif kind=='moments':s['optimizer']['state'].pop(0)
            else:s['scheduler']['last_epoch']-=1
            with self.assertRaises(ValueError):self.n['_checkpoint_validate'](s,self.model)
    def test_plateau_reduction_round_trip_and_inconsistent_lr_rejection(self):
        reduced=state(self.model,23000)
        reduced['lr_state']=dict(multiplier=.3,reductions=1)
        reduced['optimizer']['param_groups'][0]['lr']=.00006
        reduced['scheduler']['_last_lr']=[.00006]
        self.publish(reduced)
        loaded,_=self.resume()
        self.assertEqual(loaded['lr_state'],reduced['lr_state'])
        self.assertEqual(loaded['scheduler']['_last_lr'],[.00006])
        for kind in ('optimizer','scheduler','multiplier','reductions'):
            broken=copy.deepcopy(reduced)
            if kind=='optimizer':broken['optimizer']['param_groups'][0]['lr']=.0002
            elif kind=='scheduler':broken['scheduler']['_last_lr']=[.0002]
            elif kind=='multiplier':broken['lr_state']['multiplier']=1.
            else:broken['lr_state']['reductions']=2
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                self.n['_checkpoint_validate'](broken,self.model)
    def crash_publish(self,point,best=True):
        replacing=os.replace;unlinking=Path.unlink
        def replace(a,b):
            if point=='before_last' and Path(b).name=='last.pt':raise RuntimeError('crash')
            replacing(a,b)
            if point=='after_last' and Path(b).name=='last.pt':raise RuntimeError('crash')
            if point=='after_best' and Path(b).name=='best.pt':raise RuntimeError('crash')
        def unlink(path,*a,**k):
            if point=='before_unlink' and path.name=='checkpoint.pending.json':raise RuntimeError('crash')
            return unlinking(path,*a,**k)
        with patch.object(os,'replace',side_effect=replace),patch.object(Path,'unlink',new=unlink):
            with self.assertRaisesRegex(RuntimeError,'crash'):self.publish(state(self.model,20004),best)
    def test_crash_before_last_preserves_old_and_requires_reconcile(self):
        self.publish();old=(self.root/'last.pt').read_bytes();self.crash_publish('before_last')
        self.assertEqual(old,(self.root/'last.pt').read_bytes());self.assertTrue((self.root/'checkpoint.pending.json').exists())
        with self.assertRaisesRegex(ValueError,'Unpublished pending'):self.resume()
    def test_crashes_after_commit_recover_receipt_and_best(self):
        for point in ('after_last','after_best','before_unlink'):
            with self.subTest(point=point),tempfile.TemporaryDirectory() as tmp:
                self.root=Path(tmp);self.publish(best=True);self.crash_publish(point)
                recovered,_=self.resume();self.assertEqual(recovered['step'],20004)
                self.assertEqual((self.root/'last.pt').stat().st_ino,(self.root/'best.pt').stat().st_ino)
                self.assertEqual(json.loads((self.root/'checkpoint.json').read_text())['step'],20004)
                self.assertFalse((self.root/'checkpoint.pending.json').exists());self.assertEqual(self.resume()[0]['step'],20004)
    def test_first_publication_crash_recovers_without_old_receipt(self):
        self.crash_publish('after_last');self.assertFalse((self.root/'checkpoint.json').exists())
        self.assertEqual(self.resume()[0]['step'],20004)
    def test_identical_bytes_before_rename_recover_and_clear_owned_temp(self):
        self.publish();raw=(self.root/'last.pt').read_bytes();(self.root/'last.pt.tmp').write_bytes(raw)
        receipt=json.loads((self.root/'checkpoint.json').read_text());self.n['atomic_json'](self.root/'checkpoint.pending.json',dict(schema='fastpitch.checkpoint-publication.v1',receipt=receipt,is_best=False))
        self.assertEqual(self.resume()[0]['step'],20003);self.assertFalse((self.root/'last.pt.tmp').exists())
    def test_validation_then_journal_dirsync_then_publish_order(self):
        events=[];valid=self.n['_checkpoint_validate'];sync=self.n['_checkpoint_dirsync'];replace=os.replace
        def validator(*args):events.append('validate');return valid(*args)
        def directory(root):events.append('dirsync');return sync(root)
        def rename(a,b):events.append('replace:'+Path(b).name);return replace(a,b)
        with patch.dict(self.n,{'_checkpoint_validate':validator,'_checkpoint_dirsync':directory}),patch.object(os,'replace',side_effect=rename):self.publish()
        self.assertLess(events.index('validate'),events.index('replace:checkpoint.pending.json'))
        pending=events.index('replace:checkpoint.pending.json');last=events.index('replace:last.pt')
        self.assertIn('dirsync',events[pending+1:last]);self.assertIn('dirsync',events[last+1:])
    def test_scientific_AST_and_CLI_unchanged(self):
        # Recipe fingerprints from original source 453c3806; locations/format ignored.
        # Only the documented read/write migration is reversed for this comparison.
        tree=ast.parse(SOURCE.read_text())
        class BeforeIOMigration(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.name=='train':
                    node.body=[n for n in node.body if not (
                        isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='resume_state' for t in n.targets))
                        and not (isinstance(n,ast.If) and len(n.body)==1 and isinstance(n.body[0],ast.Assign)
                            and any(isinstance(t,ast.Tuple) and any(isinstance(e,ast.Name) and e.id=='resume_state' for e in t.elts) for t in n.body[0].targets))]
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if ast.unparse(node)=="init = resume_state['config']['initialization']":
                    return ast.parse("init = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['config']['initialization']").body[0]
                if ast.unparse(node)=='initial, initial_sha = _checkpoint_load(args.init_checkpoint, model, full_state=False)':
                    return ast.parse("initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)").body[0]
                if ast.unparse(node)=='ckpt = resume_state':
                    return ast.parse("ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)").body[0]
                return self.generic_visit(node)
            def visit_keyword(self,node):
                if node.arg=='sha256' and isinstance(node.value,ast.Name) and node.value.id=='initial_sha':
                    node.value=ast.parse('digest(args.init_checkpoint)',mode='eval').body
                return self.generic_visit(node)
            def visit_Expr(self,node):
                if ast.unparse(node)=='_checkpoint_publish(root, state, model, is_best)':
                    return ast.parse(ORIGINAL_SAVE_PUBLICATION).body
                return self.generic_visit(node)
        normalized=BeforeIOMigration().visit(tree)
        actual={n.name:ast_fingerprint(n) for n in normalized.body if isinstance(n,ast.FunctionDef) and not n.name.startswith('_checkpoint_')}
        self.assertEqual(actual,ORIGINAL_FUNCTION_AST)
        self.assertIsInstance(self.n['torch'],FakeTorch)
        self.assertIsInstance(self.n['np'],FakeNP)
    def test_init_uses_loaded_SHA_and_native_migration_record_remains(self):
        text=SOURCE.read_text()
        self.assertIn('sha256=initial_sha',text);self.assertNotIn('sha256=digest(args.init_checkpoint)',text)
        self.assertIn('resume-code-change-{step}',text);self.assertIn('code_sha256=digest(__file__)',text)
def tiny_real_cpu(context):
    if context=='script':
        sys.path=[item for item in sys.path if item and Path(item).resolve()!=REPO]
        sys.path.insert(0,str(REPO/'training'))
        try:__import__('training')
        except ModuleNotFoundError as error:
            assert error.name=='training'
        else:raise AssertionError('script case accidentally sees package root')
    else:sys.path.insert(0,str(REPO))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    tree=ast.parse(SOURCE.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and
           (n.name in ('digest','atomic_json') or n.name.startswith('_checkpoint_'))]
    ns=dict(torch=torch,np=np,os=os,Path=Path,hashlib=hashlib,json=json,math=math,random=random)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'private_IO_AST','exec'),ns)
    random.seed(72);np.random.seed(72);torch.manual_seed(72)
    model=torch.nn.Linear(2,1)
    def optimizer_for(m):
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,betas=(.9,.98),eps=1e-8,weight_decay=1e-6)
        return opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/1000))
    optimizer,scheduler=optimizer_for(model)
    def update(m,opt,sched):
        opt.zero_grad();m(torch.randn(1,2)).square().sum().backward();opt.step();sched.step()
    for _ in range(3):update(model,optimizer,scheduler)
    snapshot=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler={},
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[]),
        step=3,epoch=0,cursor=3,best=1.,bad=0,
        config=dict(precision='bf16',lr=.0002,warmup=1000,selection=dict(max_lr_reductions=1,plateau_lr_factor=.3)),
        provenance=dict(code_sha256='syntheticCPUtest',source_manifest_sha256='synthetic',upstream='synthetic',
                        torch=str(torch.__version__),cuda=str(torch.version.cuda),argv=['CPUtest'],utc='test'),
        lr_state=dict(multiplier=1.,reductions=0))
    before=copy.deepcopy(snapshot['rng'])
    with tempfile.TemporaryDirectory(prefix='fp-io-tiny-') as tmp:
        root=Path(tmp);ns['_checkpoint_publish'](root,snapshot,model,True)
        assert random.getstate()==before['python'] and np.array_equal(np.random.get_state()[1],before['numpy'][1])
        assert torch.equal(torch.get_rng_state(),before['torch'])
        loaded,_=ns['_checkpoint_resume'](root,model)
        assert random.getstate()==before['python']
        assert ns['_checkpoint_equal'](np.random.get_state(),before['numpy'])
        assert torch.equal(torch.get_rng_state(),before['torch'])
        update(model,optimizer,scheduler)
        uninterrupted=copy.deepcopy(model.state_dict())
        other=torch.nn.Linear(2,1);opt2,sched2=optimizer_for(other)
        other.load_state_dict(loaded['model'],strict=True);opt2.load_state_dict(loaded['optimizer']);sched2.load_state_dict(loaded['scheduler'])
        random.setstate(loaded['rng']['python']);np.random.set_state(loaded['rng']['numpy']);torch.set_rng_state(loaded['rng']['torch'])
        update(other,opt2,sched2)
        assert all(torch.equal(uninterrupted[k],v) for k,v in other.state_dict().items())
        assert ns['_checkpoint_equal'](optimizer.state_dict(),opt2.state_dict())
        assert scheduler.state_dict()==sched2.state_dict()
        assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='tiny_real_CPU_IO_and_resume_equivalent',context=context,cuda_initialized=False,public_checkpoint_or_data_loaded=False)))

class RealCPUCheckpointIOTests(unittest.TestCase):
    def run_context(self,context):
        if importlib.util.find_spec('torch') is None or importlib.util.find_spec('numpy') is None:
            self.skipTest('optional CPU Torch and NumPy are not installed')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',
                 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='fp-io-cpu-child-') as cwd:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--tiny-real-cpu',context],
                cwd=cwd,env=env,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout[-4096:]+'\n'+result.stderr[-4096:])
        report=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertFalse(report['cuda_initialized'])
        self.assertFalse(report['public_checkpoint_or_data_loaded'])
        self.assertEqual(report['context'],context)
    def test_tiny_cpu_continuation_with_package_import(self):self.run_context('package')
    def test_tiny_cpu_continuation_with_script_import_fallback(self):self.run_context('script')

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--tiny-real-cpu':
        tiny_real_cpu(sys.argv[2])
    else:unittest.main()

