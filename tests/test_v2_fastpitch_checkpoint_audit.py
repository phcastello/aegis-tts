"""CPU metadata policy checks for checkpoint recovery across resource changes."""
import unittest

from training.v2_fastpitch_checkpoint_audit import adam_layout,compare_recipe


class FastPitchAuditPolicyTests(unittest.TestCase):
    def test_recovery_microbatch_changes_do_not_change_scientific_recipe(self):
        saved=dict(batch_size=8,accumulation=2,precision='bf16',lr=.0002,model={'layers':6},features={'train':'fixed'})
        current=dict(saved,batch_size=4,accumulation=4,args={'resume':True},provenance={'utc':'later'})
        resource,other=compare_recipe(saved,current)
        self.assertEqual(resource,{'batch_size':{'before':8,'after':4},'accumulation':{'before':2,'after':4}})
        self.assertEqual(other,{})
        current.update(batch_size=2,accumulation=8)
        resource,other=compare_recipe(saved,current)
        self.assertEqual(resource['batch_size'],{'before':8,'after':2})
        self.assertEqual(other,{})

    def test_data_or_schedule_change_is_not_a_resource_recovery(self):
        saved=dict(batch_size=8,accumulation=2,precision='bf16',features={'train':'fixed'},alignment_schedule={'binary_start':1000})
        current=dict(saved,features={'train':'changed'},alignment_schedule={'binary_start':0})
        _,other=compare_recipe(saved,current)
        self.assertEqual(set(other),{'features','alignment_schedule'})

    def test_adam_missing_moments_wrong_shapes_and_partial_coverage_are_rejected(self):
        try:import torch
        except ImportError:self.skipTest('CPU torch available in the audited F5 environment')
        import copy
        parameters=[('first',torch.nn.Parameter(torch.zeros(2,3))),('second',torch.nn.Parameter(torch.zeros(3)))]
        model_state={name:value.detach() for name,value in parameters}
        optimizer={'param_groups':[dict(params=[0,1],betas=(.9,.98),eps=1e-8,weight_decay=1e-6,amsgrad=False)],
                   'state':{i:dict(step=torch.tensor(2000.),exp_avg=torch.zeros_like(param),exp_avg_sq=torch.zeros_like(param))
                            for i,(name,param) in enumerate(parameters)}}
        checks,_=adam_layout(optimizer,parameters,model_state,2000,torch)
        self.assertTrue(all(checks.values()))
        missing=copy.deepcopy(optimizer);del missing['state'][1]
        self.assertFalse(adam_layout(missing,parameters,model_state,2000,torch)[0]['optimizer_active_parameters_covered'])
        partial=copy.deepcopy(optimizer);del partial['state'][0]['exp_avg_sq']
        self.assertFalse(adam_layout(partial,parameters,model_state,2000,torch)[0]['optimizer_moment_shapes_valid'])
        wrong=copy.deepcopy(optimizer);wrong['state'][0]['exp_avg']=torch.zeros(3,2)
        self.assertFalse(adam_layout(wrong,parameters,model_state,2000,torch)[0]['optimizer_moment_shapes_valid'])
        omitted=copy.deepcopy(optimizer);omitted['param_groups'][0]['params']=[0]
        self.assertFalse(adam_layout(omitted,parameters,model_state,2000,torch)[0]['optimizer_parameter_map_complete'])


if __name__=='__main__':unittest.main()
