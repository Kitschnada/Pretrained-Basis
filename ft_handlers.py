"""Extract task updates and low-rank factors from LoRA checkpoints."""

from collections import defaultdict, OrderedDict


class LoRAHandler():
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def get_ft_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        if not hasattr(sd, 'items'):
            # peft > 0.3.0 fix
            sd = sd.state_dict()

        for key, val in sd.items():
            if '.lora_A.default' in key:
                base_name = key.replace('.lora_A.default', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_A' in key:
                base_name = key.replace('.lora_A', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_B.default' in key:
                base_name = key.replace('.lora_B.default', '')
                layer2lora_parameters[base_name]['B'] = val
            elif '.lora_B' in key:
                base_name = key.replace('.lora_B', '')
                layer2lora_parameters[base_name]['B'] = val

        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            # A: [r, I]. B: [O, r]. BxA: [O,r]x[r,I]:[O,I].
            task_parameters[name] = (key2val['B'] @ key2val['A'])
        return OrderedDict(sorted(task_parameters.items()))

    def get_ft_ab_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        for key, val in sd.items():
            if '.lora_A.default' in key:
                base_name = key.replace('.lora_A.default', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_A' in key:
                base_name = key.replace('.lora_A', '')
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_B.default' in key:
                base_name = key.replace('.lora_B.default', '')
                layer2lora_parameters[base_name]['B'] = val
            elif '.lora_B' in key:
                base_name = key.replace('.lora_B', '')
                layer2lora_parameters[base_name]['B'] = val

        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            task_parameters[name] = (key2val['A'], key2val['B'])
        return OrderedDict(sorted(task_parameters.items()))
