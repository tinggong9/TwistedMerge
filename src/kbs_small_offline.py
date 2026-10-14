"""Offline, from-scratch digits experiment. No pretrained model dependencies."""
from __future__ import annotations

import contextlib
import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from src.kbs_revision.merge_baselines import (
    LoRAUpdate, compact_qr_svd, dense_reference_svd, effective_delta,
    naive_factor_average, aligned_factor_average,
)
from src.lora_gauge_practical import orthogonal_transitions, cycle_defects
from src.lora_gauge_alignment import align_factor, synchronize_transitions

COUNTS = dict(train=950, pilot=100, D_K=80, D_align=120, D_cert=120, D_select=120, D_test=307)
DOMAINS = ('clean', 'translate', 'noise', 'occlusion')
CANDIDATES = ('compact', 'naive', 'pairwise', 'global')
POLICIES = ('P0', 'P1', 'P2', 'P4')
SOURCE_FILES = ('src/kbs_small_offline.py', 'experiments/kbs_small_offline.py',
                'tests/test_kbs_small_offline.py', 'src/kbs_revision/merge_baselines.py',
                'src/lora_gauge_practical.py', 'src/lora_gauge_alignment.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def make_splits(labels, seed=20261014):
    labels = np.asarray(labels)
    if len(labels) != sum(COUNTS.values()):
        raise ValueError('Expected exactly the bundled 1797 digits')
    remaining = np.arange(len(labels))
    result = {}
    for offset, (role, count) in enumerate(COUNTS.items()):
        if count == len(remaining):
            selected = remaining
        else:
            selected, remaining = train_test_split(remaining, train_size=count,
                stratify=labels[remaining], random_state=seed+offset)
        result[role] = np.sort(selected)
    return result


def transform(images, ids, domain, seed=41014):
    images = np.asarray(images).reshape(-1, 8, 8).copy()
    if len(images) != len(ids) or domain not in DOMAINS:
        raise ValueError('Bad domain or original IDs')
    if domain == 'translate':
        images[:, :, 1:] = images[:, :, :-1].copy()
        images[:, :, 0] = 0
    elif domain == 'noise':
        for row, original_id in enumerate(ids):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(original_id)]))
            images[row] += rng.normal(0, .12, (8, 8))
        images = np.clip(images, 0, 1)
    elif domain == 'occlusion':
        images[:, 3:5, 3:5] = 0
    return images.reshape(-1, 64)


class SealedDigits:
    def __init__(self, images, labels, splits):
        self.images, self.labels, self.splits = np.asarray(images), np.asarray(labels), splits
        self.test_open = False

    def get(self, role, domain):
        if role == 'D_test' and not self.test_open:
            raise PermissionError('D_test sealed until frozen receipt verification')
        ids = self.splits[role]
        return transform(self.images[ids], ids, domain), self.labels[ids].copy()

    def unseal(self, protocol_path, expected_sha256):
        if self.test_open:
            raise PermissionError('Final-test access already unsealed')
        if sha(protocol_path) != expected_sha256:
            raise PermissionError('Frozen protocol hash mismatch')
        value = json.loads(Path(protocol_path).read_text())
        if value.get('phase') != 'frozen' or value.get('test_access') is not True:
            raise PermissionError('No frozen final-test authorization')
        self.test_open = True


def simplex_summary():
    edges = list(itertools.combinations(range(4), 2))
    faces = list(itertools.combinations(range(4), 3))
    d1 = np.zeros((4, 6), dtype=int)
    for row, face in enumerate(faces):
        for removed in range(3):
            edge = tuple(v for j, v in enumerate(face) if j != removed)
            d1[row, edges.index(edge)] = (-1)**removed
    d2 = np.array([[(-1)**i for i in range(4)]])[:, ::-1]
    assert np.array_equal(d2@d1, np.zeros((1, 6), dtype=int))
    r1, r2 = int(np.linalg.matrix_rank(d1)), int(np.linalg.matrix_rank(d2))
    return {'vertices': 4, 'edges': 6, 'faces': 4, 'tetrahedra': 1,
            'd1_rank': r1, 'd2_rank': r2, 'h2': 4-r1-r2,
            'coefficients': 'formal constant real coefficients; not identified for learned transitions'}


def merge_factors(updates_by_layer, method):
    functions = {'compact': lambda u: compact_qr_svd(u, 4, [.25]*4),
                 'dense': lambda u: dense_reference_svd(u, 4, [.25]*4),
                 'naive': lambda u: naive_factor_average(u, [.25]*4),
                 'pairwise': lambda u: aligned_factor_average(u, [.25]*4, method='pairwise'),
                 'global': lambda u: aligned_factor_average(u, [.25]*4, method='global')}
    return [functions[method](updates) for updates in updates_by_layer]


def gauge_updates(updates_by_layer, kind='orthogonal', seed=0):
    rng = np.random.default_rng(seed)
    result = []
    for updates in updates_by_layer:
        layer = []
        for u in updates:
            if kind == 'orthogonal':
                q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
            elif kind == 'diagonal':
                q = np.diag(np.exp(rng.uniform(-.5, .5, 4)))
            else:
                raise ValueError('Only predeclared well-conditioned gauges')
            qt = torch.as_tensor(q, dtype=u.B.dtype)
            layer.append(LoRAUpdate(u.B@qt, torch.linalg.solve(qt, u.A), u.scale))
        result.append(layer)
    return result


def pairwise_admitted(value, threshold=.35):
    return math.isfinite(value) and 0 <= value <= threshold


def choose(losses, admitted):
    if not admitted or not set(admitted).issubset(losses):
        raise ValueError('No comparable admitted candidate')
    if any(not math.isfinite(losses[k]) or not 0 <= losses[k] <= 1 for k in admitted):
        raise ValueError('Classification losses must be bounded')
    return min(admitted, key=lambda k: (losses[k], k))


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
        self.head = nn.Linear(64, 10)

    def forward(self, x, factors=None):
        for index, layer in enumerate(self.layers):
            value = layer(x)
            if index and factors is not None:
                u = factors[index-1]
                value = value + u.scale*((x@u.A.T)@u.B.T)
            x = torch.relu(value)
        return self.head(x)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.ParameterList([nn.Parameter(torch.randn(4, 64)*.03) for _ in range(2)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(64, 4)) for _ in range(2)])

    def factors(self):
        return [LoRAUpdate(b, a) for b, a in zip(self.b, self.a)]


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


class ResourceStop(RuntimeError):
    pass


class Guard:
    """Conservative admission plus high-water checks every batch/stage."""
    def __init__(self, out, baseline):
        self.out, self.baseline = Path(out), baseline
        self.started = float(os.environ.get('KBS_OFFLINE_STARTED_MONOTONIC', time.monotonic()))
        self.pilot_started = self.started
        self.pilot = True
        self.policy_peak = psutil.Process().memory_info().rss
        self.abort = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def _monitor(self):
        process = psutil.Process()
        while not self.done.wait(.005):
            current = process.memory_info().rss
            self.policy_peak = max(self.policy_peak, current)
            if peak_rss()-self.baseline > 1024**3:
                self.abort = 'Incremental process high-water RSS exceeded 1 GiB'
            if time.monotonic()-self.started > 90*60:
                self.abort = 'Global wall-clock cap exceeded'
            if self.pilot and time.monotonic()-self.pilot_started > 20*60:
                self.abort = 'Pilot wall-clock cap exceeded'

    def check(self):
        if self.abort:
            raise ResourceStop(self.abort)
        if peak_rss()-self.baseline > 1024**3:
            raise ResourceStop('Incremental RSS cap')

    def checkpoint(self, stage):
        self.check()
        size = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if size > 150_000_000:
            raise ResourceStop('150 MB artifact cap')
        receipt = {'stage': stage, 'wall_seconds': time.monotonic()-self.started,
                   'peak_process_rss_bytes': peak_rss(), 'startup_peak_rss_bytes': self.baseline,
                   'incremental_peak_rss_bytes': max(0, peak_rss()-self.baseline),
                   'artifact_bytes': size, 'available_host_ram_bytes': psutil.virtual_memory().available}
        with (self.out/'resource_checkpoints.jsonl').open('a') as h:
            h.write(json.dumps(receipt, sort_keys=True)+'\n')
        print(f"{stage}: wall={receipt['wall_seconds']:.2f}s incremental_peak_RSS={receipt['incremental_peak_rss_bytes']} bytes",flush=True)
        return receipt


def block_network():
    def denied(*args, **kwargs):
        raise RuntimeError('Network disabled for the offline experiment')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied


def tensors(data, role, domain, dtype=torch.float32):
    x, y = data.get(role, domain)
    return torch.as_tensor(x, dtype=dtype), torch.as_tensor(y, dtype=torch.long)


def train(base, adapter, x, y, seed, epochs, lr, guard):
    generator = torch.Generator().manual_seed(seed)
    parameters = base.parameters() if adapter is None else adapter.parameters()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    base.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for ids in order.split(64):
            guard.check()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(base(x[ids], None if adapter is None else adapter.factors()), y[ids])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
    base.eval()


def state_hash(base):
    h = hashlib.sha256()
    for name, value in base.state_dict().items():
        h.update(name.encode()); h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def save_group(path, base, updates):
    arrays = {'base__'+k: v.detach().numpy() for k, v in base.state_dict().items()}
    for layer, factors in enumerate(updates):
        for specialist, u in enumerate(factors):
            arrays[f'B_{layer}_{specialist}'] = u.B.detach().numpy()
            arrays[f'A_{layer}_{specialist}'] = u.A.detach().numpy()
    np.savez_compressed(path, **arrays)


def load_group(path):
    with np.load(path, allow_pickle=False) as v:
        base = Base().double()
        base.load_state_dict({k: torch.from_numpy(v['base__'+k].copy()) for k in base.state_dict()})
        updates = [[LoRAUpdate(torch.from_numpy(v[f'B_{l}_{s}'].copy()).double(),
                              torch.from_numpy(v[f'A_{l}_{s}'].copy()).double()) for s in range(4)] for l in range(2)]
    base.eval().requires_grad_(False)
    return base, updates


@torch.no_grad()
def score(base, factors, data, role):
    domains = []
    for domain in DOMAINS:
        x, y = tensors(data, role, domain, torch.float64)
        logits = base(x, factors)
        error = float((logits.argmax(1) != y).double().mean())
        domains.append({'domain': domain, 'classification_loss': error, 'accuracy': 1-error,
                        'n': len(y), 'nll': float(nn.functional.cross_entropy(logits, y))})
    return float(np.mean([v['classification_loss'] for v in domains])), domains


def fit_edges(updates):
    return [orthogonal_transitions([(u.B.numpy(), (u.scale*u.A).numpy()) for u in layer]) for layer in updates]


def aligned_from_edges(fitted, method):
    result = []
    for white, transitions in fitted:
        maps = [transitions[(i, 0)] for i in range(4)] if method == 'pairwise' else synchronize_transitions(transitions, 4, 4, anchor=0)
        aligned = [align_factor(f, q) for f, q in zip(white, maps)]
        result.append(naive_factor_average([LoRAUpdate(torch.from_numpy(b), torch.from_numpy(a)) for b, a in aligned], [.25]*4))
    return result


def policy_run(policy, path, data, inference_role, group, phase, temperature, guard, cache=None):
    """Each policy constructs its own candidates; true static early exit for P4."""
    begin = time.perf_counter()
    guard.policy_peak = psutil.Process().memory_info().rss
    rows = []
    @contextlib.contextmanager
    def stage(name):
        guard.check(); start = time.perf_counter()
        yield
        guard.check()
        rows.append({'group': group, 'phase': phase, 'policy': policy, 'temperature': temperature,
                     'stage': name, 'seconds': time.perf_counter()-start, 'inference_role': inference_role})
    with stage('loading'):
        base, updates = load_group(path) if cache is None else cache
    with stage('factor_preparation'):
        updates = [[LoRAUpdate(u.B.clone(), u.A.clone(), u.scale) for u in layer] for layer in updates]
        if any(u.rank != 4 or not torch.isfinite(u.B).all() or not torch.isfinite(u.A).all() for layer in updates for u in layer):
            raise ValueError('Incompatible trained factors')
    candidates, losses = {}, {}
    gate_value = None
    cycle_value = None
    status = 'ordinary_baseline'
    with stage('applicability'):
        if policy == 'P4':
            status = 'inapplicable'
            admitted = ['compact']
        elif policy == 'P0':
            admitted = ['compact']
        else:
            admitted = list(CANDIDATES)
    fitted = None
    if policy == 'P2':
        with stage('edge_fitting'):
            fitted = fit_edges(updates)
        with stage('diagnostics'):
            cycle_value = max(cycle_defects(transitions, 4)[0] for _, transitions in fitted)
            disagreements = []
            for domain in DOMAINS:
                x, _ = tensors(data, 'D_align', domain, torch.float64)
                with torch.no_grad():
                    pred = [base(x, [updates[l][s] for l in range(2)]).argmax(1) for s in range(4)]
                disagreements.extend(float((pred[i] != pred[j]).double().mean()) for i,j in itertools.combinations(range(4),2))
            gate_value = float(np.mean(disagreements))
            if not pairwise_admitted(gate_value):
                admitted = ['compact']
            status = 'pairwise_gate_pass' if len(admitted)>1 else 'pairwise_gate_fail'
    for method in admitted:
        if method in ('pairwise', 'global') and fitted is None:
            with stage('edge_fitting'):
                fitted = fit_edges(updates)
        with stage('candidate_generation_'+method):
            candidates[method] = aligned_from_edges(fitted, method) if method in ('pairwise','global') else merge_factors(updates, method)
    if len(admitted)>1:
        with stage('validation'):
            losses = {m: score(base, c, data, 'D_select')[0] for m,c in candidates.items()}
        with stage('selection'):
            selected = choose(losses, admitted)
    else:
        with stage('selection'):
            selected = 'compact'
    with stage('inference'):
        loss, domain_rows = score(base, candidates[selected], data, inference_role)
    elapsed = time.perf_counter()-begin
    rows.append({'group':group,'phase':phase,'policy':policy,'temperature':temperature,'stage':'standalone_total',
                 'seconds':elapsed,'inference_role':inference_role,'peak_rss_bytes':guard.policy_peak,
                 'startup_peak_rss_bytes':guard.baseline,'global_peak_rss_bytes':peak_rss(),
                 'output_rank_per_layer':4,'base_parameters':13130,'adapter_parameters':1024,'full_parameters':14154,
                 'constructed_candidates':len(candidates),'validation_records':480*len(candidates) if len(admitted)>1 else 0})
    action = {'group':group,'phase':phase,'policy':policy,'temperature':temperature,'selected':selected,
              'loss':loss,'structural_status':status,'pairwise_gate':gate_value,'constructed_candidates':len(candidates),
              'orthogonal_cycle_frobenius_diagnostic':cycle_value,
              'candidate_universe':'|'.join(CANDIDATES),'validation_losses':json.dumps(losses,sort_keys=True),
              'certificate_state':'not_certified' if policy=='P4' else 'not_a_certificate',
              'standalone_seconds':elapsed,'inference_role':inference_role}
    return action, domain_rows, rows, (base, updates)


def train_group(seed, phase, data, out, guard):
    start = time.perf_counter()
    torch.manual_seed(seed)
    base = Base()
    assert sum(p.numel() for p in base.parameters()) == 13130
    x,y = tensors(data,'train','clean')
    train(base,None,x,y,seed+10,20,.003,guard)
    base.requires_grad_(False)
    base_hash = state_hash(base)
    updates = [[],[]]
    for index, domain in enumerate(DOMAINS):
        torch.manual_seed(seed*100+index)
        adapter = Adapter()
        assert sum(p.numel() for p in adapter.parameters()) == 1024
        x,y = tensors(data,'train',domain)
        train(base,adapter,x,y,seed*100+index,12,.003,guard)
        for layer,u in enumerate(adapter.factors()):
            updates[layer].append(LoRAUpdate(u.B.detach().double().clone(),u.A.detach().double().clone()))
        assert state_hash(base)==base_hash,'Frozen base/head changed'
        guard.checkpoint(f'{phase}-{seed}-{domain}-trained')
    path=out/'weights'/f'{seed}.npz'
    save_group(path,base,updates)
    reloaded, restored=load_group(path)
    compare=copy.deepcopy(base).double()
    pilot_x,_=tensors(data,'pilot','clean',torch.float64)
    with torch.no_grad():
        reload_error=float((compare(pilot_x,[updates[l][0] for l in range(2)])-reloaded(pilot_x,[restored[l][0] for l in range(2)])).abs().max())
    compact,dense=merge_factors(updates,'compact'),merge_factors(updates,'dense')
    parity=max(float(torch.linalg.norm(effective_delta(a)-effective_delta(b))/torch.linalg.norm(effective_delta(b)).clamp_min(1e-15)) for a,b in zip(compact,dense))
    if reload_error>1e-10 or parity>1e-9:
        raise RuntimeError('Reload/compact-dense numerical parity failed')
    quality=[]
    base_loss,base_domains=score(reloaded,None,data,'pilot')
    for s,domain in enumerate(DOMAINS):
        _,metrics=score(reloaded,[updates[l][s] for l in range(2)],data,'pilot')
        own=next(v for v in metrics if v['domain']==domain)
        base_own=next(v for v in base_domains if v['domain']==domain)
        quality.append({'domain':domain,'specialist_accuracy':own['accuracy'],'base_accuracy':base_own['accuracy'],
                        'gain':own['accuracy']-base_own['accuracy']})
    manifest={'group':seed,'phase':phase,'model_path':str(path.relative_to(out)),'model_sha256':sha(path),
              'base_initialization_seed':seed,'base_shuffle_seed':seed+10,
              'specialist_seeds':{d:seed*100+i for i,d in enumerate(DOMAINS)},
              'base_state_sha256':base_hash,'base_parameters':13130,'adapter_parameters_per_specialist':1024,
              'cohort_adapter_parameters':4096,'merged_full_parameters':14154,'reload_max_logit_error':reload_error,
              'compact_dense_relative_delta_error':parity,'training_seconds':time.perf_counter()-start,
              'specialist_quality_on_pilot':quality,'base_pilot_macro_loss':base_loss,
              'base_clean_pilot_accuracy':base_domains[0]['accuracy'],
              'adapter_factor_hashes':[[digest({'B':u.B.tolist(),'A':u.A.tolist()}) for u in layer] for layer in updates],
              'training_dtype':'float32','merge_and_inference_dtype':'float64','frozen_base_head_verified':True}
    write_json(out/'groups'/f'{seed}.json',manifest)
    return manifest


def gauge_controls(group, path, data, guard):
    base,updates=load_group(path)
    x,_=tensors(data,'D_align','clean',torch.float64)
    result=[]
    with torch.no_grad():
        original={m:merge_factors(updates,m) for m in CANDIDATES}
        for kind in ('orthogonal','diagonal'):
            scrambled=gauge_updates(updates,kind,seed=group+901)
            individual_delta=max(float(torch.linalg.norm(effective_delta(u)-effective_delta(v))/torch.linalg.norm(effective_delta(u)).clamp_min(1e-15)) for a,b in zip(updates,scrambled) for u,v in zip(a,b))
            individual_logits=max(float((base(x,[updates[l][s] for l in range(2)])-base(x,[scrambled[l][s] for l in range(2)])).abs().max()) for s in range(4))
            if individual_delta>1e-10 or individual_logits>1e-9:
                raise RuntimeError('Gauge control changed specialist functions')
            for method in CANDIDATES:
                merged=merge_factors(scrambled,method)
                left,right=base(x,original[method]),base(x,merged)
                result.append({'group':group,'gauge':kind,'method':method,'role':'D_align','independent_unit':group,
                    'dependent_control':True,'individual_delta_relative_error':individual_delta,'individual_logit_max_error':individual_logits,
                    'merged_delta_relative_error':max(float(torch.linalg.norm(effective_delta(u)-effective_delta(v))/torch.linalg.norm(effective_delta(u)).clamp_min(1e-15)) for u,v in zip(original[method],merged)),
                    'merged_logit_max_error':float((left-right).abs().max()),'merged_prediction_disagreement':float((left.argmax(1)!=right.argmax(1)).double().mean()),
                    'max_Q_condition_bound':1 if kind=='orthogonal' else math.e})
            guard.check()
    return result


def write_csv(path, rows, default_fields):
    fields=list(dict.fromkeys(default_fields+[k for r in rows for k in r]))
    with Path(path).open('x',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)


def base_protocol(splits, dataset_hash, root):
    return {'phase':'pilot_locked','test_access':False,'split_seed':20261014,
        'roles':{r:v.tolist() for r,v in splits.items()},'role_id_sha256':{r:digest(v.tolist()) for r,v in splits.items()},
        'dataset':'sklearn.datasets.load_digits (bundled UCI handwritten digits)','dataset_sha256':dataset_hash,
        'pilot_seeds':[1101,1102],'post_pilot_seed_pool':list(range(1201,1209)),
        'redirect_fixed_audit_seeds':[1201,1202,1203,1204],
        'domains':{'clean':{},'translate':{'pixels_right':1,'fill':0},'noise':{'sigma':.12,'seed':41014,'per_original_id':True,'clip':[0,1]},'occlusion':{'rows':[3,5],'columns':[3,5],'fill':0}},
        'model':{'widths':[64,64,64,64,10],'activation':'ReLU','bias':True,'lora_layers':[1,2],'rank':4,'alpha':4,'dropout':0,'base_parameters':13130,'adapter_parameters':1024,'frozen_head':True},
        'training':{'base_epochs':20,'specialist_epochs':12,'batch_size':64,'optimizer':'Adam','learning_rate':.003,'weight_decay':0,'threads':2,'dtype':'float32','initial_A_std':.03,'initial_B':0,'base_shuffle_seed':'base_seed+10','specialist_initialization_and_shuffle_seed':'base_seed*100+domain_index'},
        'merging':{'candidate_universe':list(CANDIDATES),'output_rank_per_layer':4,'task_weights':[.25]*4,'dtype':'float64','dense_svd':'parity only; identical predictor within measured tolerance'},
        'policies':{'P0':'always compact','P1':'D_select macro-error minimum over identical fixed universe; lexicographic tie break',
                    'P2':'mean pairwise specialist prediction disagreement across D_align domains <= .35 admits universe; else compact',
                    'P4':'conservative certification-required admission; unidentified central coefficient/uncertainty model => static early exit to compact, no alternative construction or validation',
                    'Hodge_control':'inapplicable; no identified learned real-vector cochain/closure/uncertainty model'},
        'thresholds':{'pairwise_disagreement':.35,'gauge_delta_tolerance':1e-10,'gauge_logit_tolerance':1e-9,'compact_dense_relative_tolerance':1e-9},
        'contexts':{'role':'D_K','all_four_domains_share_original_ids':True,'complex':simplex_summary(),'fixed_before_edges':True},
        'pilot_admission':{'base_clean_accuracy_min':.75,'mean_specialist_own_accuracy_min':.65,'no_admissible_central_model':'REDIRECT to four fixed independent post-pilot seeds if functioning'},
        'endpoints':{'loss':'1-accuracy, equally averaged over four domains','primary':['P4-P0','P4-P1'],'independent_unit':'new independently initialized/trained base seed with four fresh domain adapters; conditional on one common dataset split','descriptive_group_bootstrap':{'resamples':2000,'seed':77109,'percentiles':[2.5,97.5]},'noninferiority_margin':None,'significance_claim':False},
        'cost_protocol':{'cold':'fresh model/factor state loaded from local NPZ; libraries/process/OS cache may be warm; not an OS cold-start claim',
                         'warm':'same policy loaded base/factors; no candidate reuse; actual construction/validation/inference repeated',
                         'policy_order_rotation_by_group':True,'inference_role':'pilot for pilot; D_test after frozen actions for post-pilot',
                         'stage_costs_are_standalone':True,'process_startup_excluded':True},
        'gauges':{'families':['orthogonal','positive diagonal with scales exp(U[-.5,.5])'],'repetitions_per_family':1,'role':'D_align','dependent_within_group':True},
        'caps':{'incremental_peak_rss_bytes':1024**3,'new_artifact_bytes':150_000_000,'pilot_wall_seconds':1200,'global_wall_seconds':5400},
        'source_hashes':{p:sha(root/p) for p in SOURCE_FILES}}


def run(out, root):
    block_network()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    startup_baseline=peak_rss()
    available=psutil.virtual_memory().available
    out,root=Path(out),Path(root)
    out.mkdir(parents=True,exist_ok=False)
    (out/'weights').mkdir();(out/'groups').mkdir()
    guard=Guard(out,startup_baseline)
    rows,domains,actions,costs,failures,gauges,manifests=[],[],[],[],[],[],[]
    env={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'platform':platform.platform(),'processor':platform.processor(),
         'python':sys.version,'torch':torch.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
         'working_tree_at_start':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),
         'startup_peak_rss_bytes':startup_baseline,'available_host_ram_bytes':available,'estimated_additional_working_bytes':128*1024**2,
         'host_safety_reserve_bytes':256*1024**2,'cpu_threads':2,'device':'cpu','network':'blocked at socket connect/DNS entry points; no network-capable data/model loader used'}
    write_json(out/'environment.json',env)
    (out/'environment.txt').write_text('\n'.join(f'{k}: {v}' for k,v in env.items())+'\n')
    if available<384*1024**2:
        write_json(out/'BLOCKED.json',{'status':'BLOCKED','reason':'Host available RAM below conservative 128MiB incremental estimate +256MiB safety reserve','environment':env})
        guard.done.set()
        return out
    try:
        dataset=load_digits()
        images=np.asarray(dataset.data,dtype=np.float32)/16
        labels=np.asarray(dataset.target,dtype=np.int64)
        assert images.shape==(1797,64)
        splits=make_splits(labels)
        data=SealedDigits(images,labels,splits)
        dataset_hash=hashlib.sha256(dataset.data.tobytes()+labels.tobytes()).hexdigest()
        hashes=[hashlib.sha256(row.tobytes()).hexdigest() for row in images]
        owners={int(i):r for r,ids in splits.items() for i in ids}
        duplicates=[(i,j) for i in range(len(hashes)) for j in range(i) if hashes[i]==hashes[j]]
        cross=[(i,j) for i,j in duplicates if owners[i]!=owners[j]]
        if cross:raise RuntimeError('Identical original pixel records cross roles')
        protocol=base_protocol(splits,dataset_hash,root)
        write_json(out/'protocol_pilot_locked.json',protocol)
        write_json(out/'source_inventory.json',{'entry_commit':env['source_commit'],'reused':list(SOURCE_FILES[3:]),
            'reviewed_existing_scripts':['experiments/real_lora_gauge_stability.py','experiments/lora_holonomy_merging.py','experiments/matched_selector_budget_benchmark.py'],
            'existing_report_scope':'large pretrained campaign remains out of scope; prior evidence unchanged',
            'network_requests':0,'pretrained_artifacts_loaded':0,'dataset_fetch_calls':0,'duplicate_original_image_pairs':duplicates,
            'cross_role_original_id_overlap':0,'cross_role_exact_image_duplicates':cross,
            'source_hashes':protocol['source_hashes']})
        # Contexts/complex are locked here, before any model or edge fitting.
        guard.checkpoint('contexts-and-pilot-protocol-locked')
        for seed in protocol['pilot_seeds']:
            try:
                m=train_group(seed,'pilot',data,out,guard);manifests.append(m)
                group_result(seed,'pilot','pilot',out,data,guard,rows,domains,actions,costs,gauges)
            except Exception as e:
                failures.append({'group':seed,'phase':'pilot','error':repr(e)})
                raise
        functioning=all(m['base_clean_pilot_accuracy']>=.75 and np.mean([q['specialist_accuracy'] for q in m['specialist_quality_on_pilot']])>=.65 for m in manifests)
        decision='REDIRECT' if functioning else 'STOP'
        pilot_wall=time.monotonic()-guard.pilot_started
        (out/'PILOT_GO_REDIRECT_STOP.md').write_text(f'''# PILOT {decision}

Two independently trained pilot bases and their four specialists each completed.
Functioning-model criteria passed: {functioning}. Pilot elapsed: {pilot_wall:.3f}s.
P4 statically abstains because learned O(4) transitions do not identify a central
real-vector coefficient/closure/uncertainty model. The formal full simplex has
constant-real H2=0; it supplies no neural certificate. No D_test scoring occurred.
{('Proceed only with the predeclared four-seed fixed negative comparison, not a larger campaign.' if functioning else 'Stop; no new model/data fallback is authorized.')}
''')
        guard.pilot=False
        if not functioning:
            finish(out,root,env,protocol,manifests,rows,domains,actions,costs,failures,gauges,guard,'STOP')
            return out
        protocol=copy.deepcopy(protocol)
        protocol.update({'phase':'frozen','test_access':True,'pilot_decision':'REDIRECT',
            'post_pilot_seeds':protocol['redirect_fixed_audit_seeds'],
            'pilot_decision_sha256':sha(out/'PILOT_GO_REDIRECT_STOP.md'),
            'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        write_json(out/'protocol_frozen.json',protocol)
        frozen_hash=sha(out/'protocol_frozen.json')
        expected={}
        for seed in protocol['post_pilot_seeds']:
            try:
                m=train_group(seed,'fixed_evaluation',data,out,guard);manifests.append(m)
                group_actions={}
                for policy in POLICIES:
                    a,_,_,_=policy_run(policy,out/'weights'/f'{seed}.npz',data,'D_select',seed,'selection_locked','cold',guard)
                    group_actions[policy]=a['selected']
                expected[seed]=group_actions
                guard.checkpoint(f'group-{seed}-actions-locked-before-test')
            except Exception as e:
                failures.append({'group':seed,'phase':'fixed_evaluation','error':repr(e)})
                if isinstance(e,ResourceStop):break
        if not expected:
            raise ResourceStop('No post-pilot group completed before the cap')
        write_json(out/'actions_frozen_before_test.json',{'protocol_sha256':frozen_hash,'actions':expected,
            'models':{str(m['group']):m['model_sha256'] for m in manifests if m['group'] in expected},'test_accessed':False})
        for p,h in protocol['source_hashes'].items():
            if sha(root/p)!=h:raise RuntimeError('Source changed before test unseal')
        for m in manifests:
            if m['group'] in expected and sha(out/m['model_path'])!=m['model_sha256']:
                raise RuntimeError('Model bytes changed before test unseal')
        data.unseal(out/'protocol_frozen.json',frozen_hash)
        write_json(out/'test_access_receipt.json',{'unsealed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'protocol_sha256':frozen_hash,'actions_sha256':sha(out/'actions_frozen_before_test.json'),
            'single_unseal':True,'selection_role':'D_select','test_used_for_selection':False})
        for seed in expected:
            group_result(seed,'fixed_evaluation','D_test',out,data,guard,rows,domains,actions,costs,gauges,expected[seed])
            guard.checkpoint(f'group-{seed}-heldout-complete')
        finish(out,root,env,protocol,manifests,rows,domains,actions,costs,failures,gauges,guard,'REDIRECT')
    except Exception as e:
        failures.append({'group':'run','phase':'blocked' if isinstance(e,ResourceStop) else 'failed','error':repr(e)})
        if not (out/'failures.csv').exists():
            write_csv(out/'failures.csv',failures,['group','phase','error'])
        for name,values,fields in [('per_group.csv',rows,['group','phase','role']),
                                  ('per_domain.csv',domains,['group','phase','role','method','domain']),
                                  ('actions.csv',actions,['group','phase','policy','temperature']),
                                  ('stage_costs.csv',costs,['group','phase','policy','temperature','stage','seconds']),
                                  ('gauge_controls.csv',gauges,['group','gauge','method'])]:
            if not (out/name).exists():write_csv(out/name,values,fields)
        write_json(out/'partial_execution_receipt.json',{'completed_models':manifests,'failures':failures,
            'completed_group_rows':len(rows),'reason':repr(e),'status':'BLOCKED' if isinstance(e,ResourceStop) else 'FAILED'})
        write_json(out/'BLOCKED.json',{'status':'BLOCKED' if isinstance(e,ResourceStop) else 'FAILED','error':repr(e),'not_a_negative_scientific_result':True})
        guard.done.set()
        raise
    finally:
        guard.done.set()
    return out


def group_result(seed,phase,role,out,data,guard,rows,domains,actions,costs,gauges,expected=None):
    path=out/'weights'/f'{seed}.npz'
    base,updates=load_group(path)
    candidate_losses={}
    candidate_predictions={}
    for method in CANDIDATES:
        factors=merge_factors(updates,method)
        loss,metric=score(base,factors,data,role)
        candidate_losses[method]=loss
        with torch.no_grad():
            candidate_predictions[method]=torch.cat([base(tensors(data,role,d,torch.float64)[0],factors).argmax(1) for d in DOMAINS])
        domains.extend({'group':seed,'phase':phase,'role':role,'method':method,**r} for r in metric)
    _,metric=score(base,None,data,role)
    domains.extend({'group':seed,'phase':phase,'role':role,'method':'base',**r} for r in metric)
    for s,domain in enumerate(DOMAINS):
        _,metric=score(base,[updates[l][s] for l in range(2)],data,role)
        domains.extend({'group':seed,'phase':phase,'role':role,'method':'specialist_'+domain,**r} for r in metric)
    selected={}
    order=list(POLICIES);offset=seed%len(order);order=order[offset:]+order[:offset]
    for policy in order:
        cache=None
        for temperature in ('cold','warm'):
            a,metric,timing,cache=policy_run(policy,path,data,role,seed,phase,temperature,guard,cache)
            if expected is not None and a['selected']!=expected[policy]:raise RuntimeError('Frozen action changed after test access')
            assert abs(a['loss']-candidate_losses[a['selected']])<1e-14
            if policy in selected and selected[policy]['selected']!=a['selected']:raise RuntimeError('Warm policy changed action')
            selected[policy]=a
            actions.append(a);costs.extend(timing)
            if temperature=='cold':domains.extend({'group':seed,'phase':phase,'role':role,'method':policy,**r} for r in metric)
    p0,p1,p4=(selected[p] for p in ('P0','P1','P4'))
    rows.append({'group':seed,'phase':phase,'role':role,'P0_loss':p0['loss'],'P1_loss':p1['loss'],'P4_loss':p4['loss'],
                 'P4_minus_P0':p4['loss']-p0['loss'],'P4_minus_P1':p4['loss']-p1['loss'],
                 'P0_action':p0['selected'],'P1_action':p1['selected'],'P4_action':p4['selected'],
                 'P4_changed_vs_P0':p4['selected']!=p0['selected'],'P4_changed_vs_P1':p4['selected']!=p1['selected'],
                 'P4_prediction_disagreement_vs_P0':float((candidate_predictions[p4['selected']]!=candidate_predictions[p0['selected']]).double().mean()),
                 'P4_prediction_disagreement_vs_P1':float((candidate_predictions[p4['selected']]!=candidate_predictions[p1['selected']]).double().mean()),
                 'P4_useful_vs_P0':p4['selected']!=p0['selected'] and p4['loss']<p0['loss'],
                 'P4_useful_vs_P1':p4['selected']!=p1['selected'] and p4['loss']<p1['loss'],
                 'P4_status':'inapplicable','neural_exact_trivial':False,'hodge_control':'inapplicable'})
    gauges.extend(gauge_controls(seed,path,data,guard))


def finish(out,root,env,protocol,manifests,rows,domains,actions,costs,failures,gauges,guard,status):
    write_csv(out/'per_group.csv',rows,['group','phase','role'])
    write_csv(out/'per_domain.csv',domains,['group','phase','role','method','domain'])
    write_csv(out/'actions.csv',actions,['group','phase','policy','temperature'])
    write_csv(out/'stage_costs.csv',costs,['group','phase','policy','temperature','stage','seconds'])
    write_csv(out/'gauge_controls.csv',gauges,['group','gauge','method'])
    write_csv(out/'failures.csv',failures,['group','phase','error'])
    write_json(out/'models_and_splits_manifest.json',{'dataset_sha256':protocol['dataset_sha256'],'roles':protocol['roles'],
        'role_id_sha256':protocol['role_id_sha256'],'models':manifests,'base_initialization':'independent local seed, from scratch',
        'models_committed':False,'weight_storage':'small safe numeric NPZ, allow_pickle=False, under this run during execution; excluded from Git',
        'domain_replicates_independent':False,'gauge_repetitions_independent':False})
    selected=[r for r in rows if r['phase']=='fixed_evaluation']
    estimates={}
    for endpoint in ('P4_minus_P0','P4_minus_P1'):
        values=np.array([r[endpoint] for r in selected])
        if len(values):
            rng=np.random.default_rng(77109)
            means=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
            estimates[endpoint]={'mean':float(values.mean()),'group_values':values.tolist(),'n_independent_training_seeds':len(values),
                'descriptive_percentile_bootstrap_95':np.percentile(means,[2.5,97.5]).tolist(),'formal_significance':False}
    summary={'status':status,'pilot_groups':len([m for m in manifests if m['phase']=='pilot']),
             'post_pilot_heldout_groups':len(selected),'primary_estimates':estimates,
             'P4_changed_vs_P0':sum(r['P4_changed_vs_P0'] for r in selected),
             'P4_changed_vs_P1':sum(r['P4_changed_vs_P1'] for r in selected),
             'P4_changed_predictions_vs_P1':sum(r['P4_prediction_disagreement_vs_P1']>0 for r in selected),
             'P4_useful_vs_P0':sum(r['P4_useful_vs_P0'] for r in selected),
             'P4_useful_vs_P1':sum(r['P4_useful_vs_P1'] for r in selected),
             'abstention':{'eligible':0,'neural_exact_trivial':0,'repairable_within_tolerance':0,'nontrivial_on_K':0,'not_certified':len(selected),'inapplicable':len(selected)},
             'formal_complex_h2':0,'neural_central_certificate_identified':False,'network_requests':0,'pretrained_weights_used':False}
    write_json(out/'results.json',summary)
    def table():
        return '\n'.join('| '+str(r['group'])+' | '+ ' | '.join(f"{r[k]:.6f}" for k in ['P0_loss','P1_loss','P4_loss','P4_minus_P0','P4_minus_P1'])+' |' for r in selected)
    cost_lines=[]
    for policy in ('P0','P1','P4'):
        for temp in ('cold','warm'):
            values=[r['seconds'] for r in costs if r['phase']=='fixed_evaluation' and r['policy']==policy and r['temperature']==temp and r['stage']=='standalone_total']
            if values:cost_lines.append(f'| {policy} | {temp} | {np.mean(values):.6f} | {len(values)} |')
    description='\n'.join(f"{k}: mean {v['mean']:.8f}, descriptive group-bootstrap 95% interval {v['descriptive_percentile_bootstrap_95']}." for k,v in estimates.items())
    (out/'results_summary.md').write_text(f'''# Small offline digits: {status}

Actual completed pilot groups: {summary['pilot_groups']}; independently trained
post-pilot groups scored on sealed heldout IDs: {len(selected)}. The post-pilot
REDIRECT audit used the first four predeclared seeds, rather than all eight,
because structural applicability was absent. These are training-randomness
replicates conditional on one shared split, not independent datasets or writers.

| Seed | P0 loss | P1 loss | P4 loss | P4-P0 | P4-P1 |
| --- | --- | --- | --- | --- | --- |
{table()}

{description}

Loss is 1-accuracy averaged equally over four domains of the same original
heldout digit IDs. Lower is better. The four domains and two gauge controls
within each seed are dependent. Small-n bootstrap intervals are descriptive;
no powered significance or formal noninferiority claim is made. All action
choices were hash-frozen before one test unseal. Pilot groups are excluded
from heldout estimates. See per_group.csv and per_domain.csv for exact rows.

P4 changed {summary['P4_changed_vs_P0']}/{len(selected)} actions versus P0 and
{summary['P4_changed_vs_P1']}/{len(selected)} versus P1; changed and lower-loss
decisions: {summary['P4_useful_vs_P0']} and {summary['P4_useful_vs_P1']} respectively.
Different heldout classification predictions versus P1 occurred in
{summary['P4_changed_predictions_vs_P1']}/{len(selected)} groups; per_group.csv
separates action names from actual classification disagreement rates.
P4 was inapplicable in every group because no justified neural central
coefficient/closure/uncertainty model was identified. Formal full-simplex
constant-real H2=0 is not a learned neural certificate; no Hodge projection,
finite class, ROC or risk/coverage curve was invented.

Actual stand-alone sequential times (seconds, means across heldout groups):

| Policy | Invocation | Mean total seconds | Groups |
| --- | --- | --- | --- |
{chr(10).join(cost_lines)}

Cold denotes fresh local model/factor state, with process/library/OS caches
potentially warm; it is not a process-startup or disk-cache cold claim. Warm
retains only loaded base/factors, rebuilding all policy-specific candidates.
Times include actual loading/preparation/diagnostics/generation/validation/
selection and heldout inference. No candidate panel is shared across policies;
P4 exits on applicability before edges or alternative candidates. Microsecond
overheads and invocation ordering make tiny P0/P4 time differences noisy.
Full stage rows and per-policy sampled RSS are in stage_costs.csv.

Every base has 13,130 parameters. Each specialist adds 1,024 parameters across
two rank-4 hidden layers; each selected predictor has 14,154 factorized
parameters. Base/head stayed frozen during specialist training. No rank-16
concatenation or ensemble capacity advantage is included. Dense SVD is parity
only, not a distinct leaderboard method. Numerical reload and compact/dense
errors, actual quality and training times are in the model/group manifests.

Gauge controls are deliberately imposed equivalent representations, not natural
trained holonomy. Their individual delta/logit invariance and merge instability
are measured on D_align in gauge_controls.csv; they add no independent groups.
Noise is independently seeded by original ID then held fixed across roles,
groups and repeated invocations; domain views share originals within a role.
Original IDs never cross roles, and exact identical source images were audited.
Transforms are intended label-preserving; translation clipping/occlusion may
remove digit information, and no writer-disjoint generalization is established.

This is a small real-data/from-scratch neural experiment. It does not establish
modern pretrained-model efficacy, a natural Brauer class, or useful structural
certification. The practical-method KBS advantage remains unsupported; retain
the narrower theory/diagnostic framing and negative result.
''')
    (out/'decision_utility.md').write_text(f'''# What decision changed, and was it useful?

On original trained factors, conservative P4 returned compact SVD in every
post-pilot group: {summary['P4_changed_vs_P0']}/{len(selected)} changes versus
always-fallback P0. That equality follows the predeclared inapplicability rule;
the experiment measures its realized losses/cost and the alternative validation
choices, rather than discovering a positive obstruction. Versus P1,
{summary['P4_changed_vs_P1']}/{len(selected)} actions changed, of which
{summary['P4_useful_vs_P1']} had strictly lower heldout domain-macro loss.

{description}

actions.csv identifies every choice, allowed candidate universe, actual
constructed count, validation losses and abstention. All four candidate
options are from the same fixed rank-4 universe available before admission;
P4 requires certification for any departure from fallback. This conservative
operationalization is explicit, not a claim that existing uncertified ordinary
selection has been proved unsafe. P1 receives the same allowed D_select budget;
P4 uses none after its early exit. P2's simple empirical disagreement gate is a
comparison heuristic, not a central certificate. No valid Hodge control exists.

stage_costs.csv contains actual complete sequential P0/P1/P4 executions, with
first-load and warm semantics and inference. Results_summary.md gives means;
no shared-panel attribution is sold as an early-exit saving. Timing variability
does not support a reliable P4-vs-P0 micro-overhead conclusion. The controlled
gauge repetitions in gauge_controls.csv are secondary and dependent; they do
not change the primary original-factor conclusion.

Recommend REDIRECT to the scoped theory/negative-diagnostic paper. This run
supports no useful incremental structural admission action beyond fallback.
Any P4/P1 loss difference arises from the frozen fallback versus validation
choice in this small corpus, not a calibrated structural safety guarantee.
''')
    (out/'arxiv_v2_claim_map.md').write_text('''# Claim map; no manuscript changes

| Statement | Evidence class | Limit |
| --- | --- | --- |
| Earlier scoped no-go, torus and matched real-vector Hodge results | verified existing mathematics | Reused evidence; not rerun or enlarged by this experiment |
| From-scratch digit MLP and two rank-4 layers can be trained/merged offline | small locally trained real-data outcome | See model manifests, specialist quality and actual heldout rows |
| P4's conservative certificate-required rule degenerates to fallback when identification fails | small real-data negative diagnostic outcome | Explicit operational policy; observed losses/costs conditional on one split, not a universal impossibility theorem |
| Simplex constant-real H2 is zero | verified finite topology control | Does not identify a cochain or certify learned O(4) transitions |
| Equivalent imposed factor gauges preserve specialists and can affect merges | controlled secondary outcome | D_align measurements; dependent within-seed controls, not naturally trained holonomy |
| P4 improves pretrained-model utility or supplies calibrated neural certification/risk coverage | unsupported | No pretrained models or justified central uncertainty model |
| Powered significant performance or formal noninferiority | unsupported | Small fixed group count; descriptive bootstrap only; no margin |

No external AI model/service, remote machine, paid compute, dataset fetching,
manuscript rewriting or arXiv upload is part of this experiment. Prior negative
evidence remains intact. This does not turn topology or a gauge control into a
natural Brauer-class discovery.
''')
    (out/'README.md').write_text(f'''# Reproduce the offline small digits audit

Source commit: `{env['source_commit']}`; exact module/test/helper hashes in
protocol_frozen.json (or pilot protocol if STOP). CPU-only, two Torch threads.
Uses installed sklearn load_digits, all weights initialized/trained locally.
Zero-network attestation: no network/data fetch/model service used; socket
connect, connect_ex, create_connection and DNS are blocked during execution.
No externally pretrained model, AI inference service or extra agent is an
experimental dependency. Codex wrote/reviewed software only.

```
.venv/bin/python -m pytest -q tests/test_kbs_small_offline.py
.venv/bin/python experiments/kbs_small_offline.py --out reports/kbs_small_offline/<fresh-UTC-run>
```

Limits: incremental peak RSS <=1 GiB over explicitly measured post-import
startup high-water baseline;150 MB newly written run artifacts;20 min pilot;
90 min total. Conservative 128 MiB incremental admission plus256 MiB host
safety reserve. A5ms monitor tracks RSS/time, high-water checks run every
batch/stage, with disk/time/memory checkpoints after training stages. Native
kernels are not preempted; any cap event stops the run as BLOCKED, not a
scientific negative. Actual receipt is resource_checkpoints.jsonl.

Do not use the old unavailable data/checkpoints mounts. Weights are small
safe numeric NPZ files used locally with allow_pickle=False; only hashes are
committed, and weights are retained in project scratch. Data IDs/hash/split
and model hashes are in models_and_splits_manifest.json. Stratification uses
labels to assign original IDs; sealed test labels are not scored/selected on
until all source, protocol and policy actions pass frozen-hash checks.

D_K declares common comparison contexts before edges; D_align supplies
pairwise-agreement/gauge controls; D_cert is reserved and not inspected because
no neural certificate model is applicable. D_select drives ordinary selection;
pilot supplies quality/admission only. D_test is unsealed once after all
post-pilot actions are frozen. No final-test tuning. Review results_summary.md,
decision_utility.md and arxiv_v2_claim_map.md with the raw CSVs.
''')
    final=guard.checkpoint('complete')
    write_json(out/'completion.json',{'status':status,'resource_receipt':final,'network_requests':0,'paid_spend':0,
        'source_hashes_unchanged':all(sha(root/p)==h for p,h in protocol['source_hashes'].items())})
