import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import flagship54_guard as g


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {'lane':g.LANE,'repo':g.REPO,'max_steps':37000,'base_model':'expected/base',
                    'group_id':'group-1','group_version':2,
                    'command_sha256':hashlib.sha256(g.canonical(['train','--max-steps','37000'])).hexdigest(),
                    'state_dir':str(self.root/'state'),'archive_dir':str(self.root/'archive'),
                    'hf_token_file':str(self.root/'hf'),'salad_key_file':str(self.root/'salad')}
        self.group = {'id':'group-1','name':g.LANE,'version':2,'replicas':1,
                      'container':{'command':['train','--max-steps','37000']},
                      'current_state':{'status':'running','instance_status_counts':{
                          'running_count':1,'allocating_count':0,'creating_count':0,'stopping_count':0}}}

    def entry(self,name,data=b'payload',lfs=True):
        digest = hashlib.sha256(data).hexdigest() if lfs else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        return NS(rfilename=name,size=len(data),lfs=NS(sha256=digest) if lfs else None,blob_id=digest)

    def arrange(self,step=37000,final=True):
        ck=f'checkpoint-{step}/'
        names=[ck+'trainer_state.json',ck+'adapter_model.safetensors',ck+'adapter_config.json',ck+'tokenizer.json',ck+'tokenizer_config.json']
        if final:
            names+=['final/adapter_model.safetensors','final/adapter_config.json']
        info=NS(sha='a'*40,private=True,siblings=[self.entry(n) for n in names])
        api=Mock()
        api.model_info.return_value=info
        salad=Mock()
        salad.request.return_value=copy.deepcopy(self.group)
        state=self.root/'state-input.json'
        state.write_text(json.dumps({'global_step':step,'max_steps':37000}))
        return api,salad,info,state

    def test_config_valid(self): g.validate_config(self.cfg)
    def test_other_lane_denied(self):
        with self.assertRaises(g.Hold): g.validate_config({**self.cfg,'lane':'some-other-lane'})
    def test_boolean_target_denied(self):
        with self.assertRaises(g.Hold): g.validate_config({**self.cfg,'max_steps':True})
    def test_relative_path_denied(self):
        with self.assertRaises(g.Hold): g.validate_config({**self.cfg,'state_dir':'relative'})
    def test_group_identity_denied(self):
        with self.assertRaises(g.Hold): g.check_group({**self.group,'id':'foreign'},self.cfg)
    def test_group_version_denied(self):
        with self.assertRaises(g.Hold): g.check_group({**self.group,'version':3},self.cfg)
    def test_command_drift_denied(self):
        with self.assertRaises(g.Hold): g.check_group({**self.group,'container':{'command':['other']}},self.cfg)
    def test_scheduled_scaling_denied(self):
        with self.assertRaises(g.Hold): g.check_group({**self.group,'scheduled-scaling-enabled':True},self.cfg)
    def test_autoscaler_denied(self):
        with self.assertRaises(g.Hold): g.check_group({**self.group,'queue_autoscaler':{'min':1}},self.cfg)
    def test_zero_desired_does_not_mean_drained(self):
        self.group['replicas']=0
        self.assertFalse(g.counts_drained(self.group))
    def test_missing_counts_does_not_mean_drained(self):
        self.assertFalse(g.counts_drained({'replicas':0,'current_state':{}}))
    def test_exactly_zero_all_counts_drained(self):
        self.group['replicas']=0
        self.group['current_state']['instance_status_counts']['running_count']=0
        self.assertTrue(g.counts_drained(self.group))
    def test_private_repo_required(self):
        api,_,info,_=self.arrange()
        info.private=False
        with self.assertRaises(g.Hold): g.metadata(api)
    def test_unknown_private_denied(self):
        api,_,info,_=self.arrange()
        info.private=None
        with self.assertRaises(g.Hold): g.metadata(api)
    def test_latest_partial_checkpoint_denied(self):
        api,_,info,_=self.arrange()
        info.siblings.append(self.entry('checkpoint-37250/adapter_model.safetensors'))
        with self.assertRaises(g.Hold): g.metadata(api)
    def test_training_target_mismatch_denied(self):
        with self.assertRaises(g.Hold): g.validate_state({'global_step':10,'max_steps':20},10,self.cfg)
    def test_checkpoint_step_mismatch_denied(self):
        with self.assertRaises(g.Hold): g.validate_state({'global_step':9,'max_steps':37000},10,self.cfg)
    def test_filename_presence_before_completion_never_stops(self):
        api,salad,_,state=self.arrange(step=5750,final=True)
        with patch.object(g,'download_checked',return_value=(state,{})):
            result=g.cycle(self.cfg,api,salad,apply=True)
        self.assertEqual(result['status'],'TRAINING')
        salad.request.assert_called_once_with()
    def test_incomplete_compute_not_running_holds(self):
        api,salad,_,state=self.arrange(step=5750)
        salad.request.return_value['replicas']=0
        with patch.object(g,'download_checked',return_value=(state,{})):
            result=g.cycle(self.cfg,api,salad,apply=True)
        self.assertEqual(result['status'],'HOLD_INCOMPLETE_COMPUTE_NOT_RUNNING')
        salad.request.assert_called_once_with()
    def test_complete_without_final_never_stops(self):
        api,salad,_,state=self.arrange(final=False)
        with patch.object(g,'download_checked',return_value=(state,{})):
            with self.assertRaises(g.Hold): g.cycle(self.cfg,api,salad,apply=True)
        salad.request.assert_called_once_with()
    def test_pickle_final_denied(self):
        api,_,info,_=self.arrange()
        info.siblings.append(self.entry('final/optimizer.pt'))
        with self.assertRaises(g.Hold): g.required_exports({x.rfilename:x for x in info.siblings},37000)
    def test_tokenizer_missing_denied(self):
        api,_,info,_=self.arrange()
        with self.assertRaises(g.Hold): g.required_exports({x.rfilename:x for x in info.siblings if not x.rfilename.endswith('tokenizer.json')},37000)
    def test_lfs_hash_checked(self):
        file=self.root/'blob'; file.write_bytes(b'payload')
        self.assertEqual(g.validate_file(file,self.entry('x'))['sha256'],hashlib.sha256(b'payload').hexdigest())
    def test_git_blob_hash_checked(self):
        file=self.root/'blob'; file.write_bytes(b'payload')
        self.assertEqual(g.validate_file(file,self.entry('x',lfs=False))['hub_oid_algorithm'],'git-blob-sha1')
    def test_byte_corruption_denied(self):
        file=self.root/'blob'; file.write_bytes(b'payloae')
        with self.assertRaises(g.Hold): g.validate_file(file,self.entry('x'))
    def test_size_mismatch_denied(self):
        file=self.root/'blob'; file.write_bytes(b'payload_extra')
        with self.assertRaises(g.Hold): g.validate_file(file,self.entry('x'))
    def test_unsafe_paths_denied(self):
        for name in ('../x','/absolute','final/../x','final\\x','final//x'):
            with self.subTest(name=name),self.assertRaises(g.Hold):g.safe_relative(name)
    def test_different_final_weights_denied(self):
        manifest={'files':{'final/adapter_model.safetensors':{'sha256':'x'},'checkpoint-37000/adapter_model.safetensors':{'sha256':'y'}}}
        with self.assertRaises(g.Hold):g.validate_export(self.root,manifest,37000,self.cfg)
    def test_atomic_receipt_digest_and_permissions(self):
        p=g.receipt(self.cfg,'TEST',{'claim':'test only'})
        value=json.loads(p.read_text()); actual=value.pop('payload_sha256')
        self.assertEqual(actual,hashlib.sha256(g.canonical(value)).hexdigest())
        self.assertEqual(p.stat().st_mode & 0o777,0o600)
        self.assertFalse(list(p.parent.glob('*.tmp')))
    def test_secret_not_world_readable(self):
        p=self.root/'token';p.write_text('test-token');p.chmod(0o644)
        with self.assertRaises(g.Hold):g.secret(str(p))
    def test_secret_bom_handled(self):
        p=self.root/'token';p.write_text('\ufefftest-token');p.chmod(0o600)
        self.assertEqual(g.secret(str(p)),'test-token')
    def test_secret_symlink_denied(self):
        p=self.root/'token';p.write_text('test-token');p.chmod(0o600)
        link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(g.Hold):g.secret(str(link))
    def test_no_gpu_training_imports(self):
        source=Path(g.__file__).read_text()
        self.assertNotIn('import torch',source)
        self.assertNotIn('torch.load',source)
        self.assertNotIn('pickle.load',source)

    def test_dry_run_never_mutates(self):
        api,salad,info,state=self.arrange()
        with patch.object(g,'download_checked',return_value=(state,{})),patch.object(g,'archive',return_value=(self.root,{'files':{}})),patch.object(g,'validate_export'):
            result=g.cycle(self.cfg,api,salad,apply=False)
        self.assertEqual(result['status'],'READY_DRY_RUN')
        self.assertTrue(all(call.args == () for call in salad.request.call_args_list))
    def test_revision_change_blocks_mutation(self):
        api,salad,info,state=self.arrange()
        api.model_info.side_effect=[info,NS(private=True,sha='b'*40)]
        with patch.object(g,'download_checked',return_value=(state,{})),patch.object(g,'archive',return_value=(self.root,{'files':{}})),patch.object(g,'validate_export'):
            with self.assertRaises(g.Hold):g.cycle(self.cfg,api,salad,apply=True)
        salad.request.assert_called_once_with()
    def test_verified_completion_scales_only_zero(self):
        api,salad,info,state=self.arrange()
        drained=copy.deepcopy(self.group);drained['replicas']=0;drained['version']=3
        drained['current_state']['instance_status_counts']['running_count']=0
        salad.request.side_effect=[self.group,self.group,drained,drained]
        with patch.object(g,'download_checked',return_value=(state,{})),patch.object(g,'archive',return_value=(self.root,{'files':{}})),patch.object(g,'validate_export'):
            result=g.cycle(self.cfg,api,salad,apply=True)
        self.assertEqual(result['status'],'DONE')
        self.assertTrue(result['mutation_performed'])
        self.assertEqual(salad.request.call_args_list[2].args,('PATCH',{'replicas':0}))
    def test_patch_success_running_instance_not_done(self):
        api,salad,info,state=self.arrange()
        draining=copy.deepcopy(self.group);draining['replicas']=0;draining['version']=3
        salad.request.side_effect=[self.group,self.group,draining,draining]
        with patch.object(g,'download_checked',return_value=(state,{})),patch.object(g,'archive',return_value=(self.root,{'files':{}})),patch.object(g,'validate_export'):
            result=g.cycle(self.cfg,api,salad,apply=True)
        self.assertEqual(result['status'],'DRAINING')
        self.assertFalse(list((self.root/'state'/'receipts').glob('done-*')))

if __name__=='__main__': unittest.main(verbosity=2)
